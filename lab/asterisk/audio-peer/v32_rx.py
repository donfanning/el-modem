"""V.32 receiver (written from scratch; Python, with an optional NumPy data-distance path).

Pipeline: real samples -> complex baseband at 1800 Hz -> low-pass -> resample to 2 samples
per symbol (4800 Hz) -> T/2 complex LMS equalizer, trained on the known TRN sequence and then
decision-directed with a small carrier PLL -> slicer -> differential decode -> descrambler ->
bits (async framing is the application's business, see ``deframe_async``).

This first version is "offline-friendly": ``V32Rx.run`` takes a whole capture and the
approximate time the far end's TRN begins; it finds the exact symbol alignment by correlation
against the expected scrambled sequence. A streaming wrapper is a later step.
"""

from __future__ import annotations

import cmath
import math
import os

from v32_caller import (GPA, NR_DIFF, POINTS, Scrambler, TCM_BITS, TFSM, TE_DIFF, V32_TCM, V32_TCM_9600,
                        ZTAB4)

BAUD = 2400
CARRIER = 1800.0
TRN_AC = 256


def expected_trn(n: int, tap: int = GPA) -> list[complex]:
    scr = Scrambler(tap)
    out = []
    for m in range(n):
        b1, b2 = scr.fwd(1), scr.fwd(1)
        if m < TRN_AC:
            b2 = b1
        out.append(POINTS[(b1 << 1) | b2])
    return out


def lowpass_taps(fs: float, cutoff: float = 1500.0, n: int = 41) -> list[float]:
    taps = []
    mid = (n - 1) / 2
    for i in range(n):
        x = i - mid
        h = 2 * cutoff / fs if x == 0 else math.sin(2 * math.pi * cutoff * x / fs) / (math.pi * x)
        taps.append(h * (0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1))))
    s = sum(taps)
    return [t / s for t in taps]


def to_baseband(samples, fs: float) -> list[complex]:
    """Mix down by the carrier and low-pass; output at the input rate, group delay removed."""
    taps = lowpass_taps(fs)
    n = len(taps)
    mixed = [v * cmath.exp(-2j * math.pi * CARRIER * i / fs) for i, v in enumerate(samples)]
    half = n // 2
    out = []
    for i in range(len(mixed)):
        acc = 0j
        for j in range(n):
            k = i + j - half
            if 0 <= k < len(mixed):
                acc += taps[j] * mixed[k]
        out.append(acc * 2)
    return out


def resample(bb: list[complex], fs: float, rate: float = 2.0 * BAUD) -> list[complex]:
    """Cubic (Catmull-Rom/Lagrange) interpolation to ``rate`` samples per second."""
    out = []
    step = fs / rate
    pos = 1.0
    while pos < len(bb) - 2:
        i = int(pos)
        f = pos - i
        p0, p1, p2, p3 = bb[i - 1], bb[i], bb[i + 1], bb[i + 2]
        a = -f * (f - 1) * (f - 2) / 6
        b = (f + 1) * (f - 1) * (f - 2) / 2
        c = -(f + 1) * f * (f - 2) / 2
        d = (f + 1) * f * (f - 1) / 6
        out.append(a * p0 + b * p1 + c * p2 + d * p3)
        pos += step
    return out


class Equalizer:
    """T/2 complex LMS equalizer producing one output per symbol."""

    def __init__(self, taps: int = 32):
        self.n = taps
        self.w = [0j] * taps

    def output(self, u: list[complex], k: int, c: int):
        """Equalizer output for symbol k; window u[c+2k-n+1 .. c+2k] (newest first)."""
        base = c + 2 * k
        x = u[base - self.n + 1: base + 1][::-1]
        y = 0j
        for wi, xi in zip(self.w, x):
            y += wi * xi
        return y, x

    def adapt(self, x, err: complex, mu: float, leak: float = 0.0) -> None:
        norm = sum(v.real * v.real + v.imag * v.imag for v in x) + 1e-9
        g = mu * err / norm
        keep = 1.0 - leak
        self.w = [keep * wi + g * xi.conjugate() for wi, xi in zip(self.w, x)]


def slice4(z: complex) -> int:
    """Index of the nearest of A, B, D, C."""
    b1 = int(2.0 * z.imag > -z.real)
    b2 = int(z.real > z.imag)
    return (b1 << 1) | b2


# Inverse of V.32bis Table 1: the dibit that took the differential coder from `prev` to `new`,
# indexed prev << 2 | new.
TD_DIFF = [next(d for d in range(4) if TE_DIFF[(i >> 2) << 2 | d] == i & 3) for i in range(16)]
# Trace-back depth and the 0.99 survivor-metric decay in ViterbiStream.push are borrowed design
# parameters from fisher-modem's decoder; they are tuning choices, not part of the Recommendation.
TLEN = 20


def slice16(z: complex) -> complex:
    return min(ZTAB4, key=lambda p: abs(z - p))


# 32-point cross used by the 9600 bps mode: odd-grid points up to +-5, corners removed
CROSS32 = [complex(a, b) for a in (-5, -3, -1, 1, 3, 5) for b in (-5, -3, -1, 1, 3, 5)
           if not (abs(a) == 5 and abs(b) == 5)]


def slice_to(z: complex, points) -> complex:
    return min(points, key=lambda p: abs(z - p))


class ViterbiStream:
    """Streaming 8-state Viterbi for the trellis modes: 7200 (3 bits/symbol) and 9600/12000/14400
    (4/5/6 bits/symbol). ``push(z)`` takes one equalised symbol and returns a decoded group of bits once
    the trace-back window (TLEN symbols) is full, else None. The encoder is assumed to start in state 0
    with previous differential value 0. Groups are the scrambled bits, first bit in the top position."""

    INF = 1e30
    _COSETS: dict = {}

    def __init__(self, rate: int, backend: str | None = None):
        if rate != 7200 and rate not in TCM_BITS:
            raise ValueError("trellis rate must be 7200, 9600, 12000 or 14400")
        self.rate = rate
        self.sv = [0.0] + [self.INF] * 7
        self.paths: list[list[int]] = [[] for _ in range(8)]
        self.prev = 0                   # previous differential value (7200: k; others: Y2Y1)
        if rate in TCM_BITS and rate not in self._COSETS:
            # (y0, k) -> [(point, uncoded value)] with k = Y1<<1|Y2
            cos: dict = {}
            nb = TCM_BITS[rate] - 2
            for key, z in V32_TCM[rate].items():
                y0, y1, y2 = key[:3]
                val = 0
                for bit in key[3:]:
                    val = (val << 1) | bit
                cos.setdefault((y0, (y1 << 1) | y2), []).append((z, val))
            self._COSETS[rate] = cos
        self.cosets = self._COSETS.get(rate)
        # Array setup dominates smaller constellations in local measurements.
        # The environment opt-in targets the measured 128-point bottleneck;
        # an explicit backend remains available for parity diagnostics.
        self.backend = backend or ("numpy" if rate == 14400 and
                                   os.environ.get("V32_NUMPY_METRICS") == "1" else "python")
        if self.backend not in ("python", "numpy"):
            raise ValueError("unknown V32 distance backend")
        self._decision_points = ZTAB4 if rate == 7200 else list(V32_TCM[rate].values())
        if self.backend == "numpy" and self.cosets is not None:
            import numpy as np
            self._np = np
            self._keys = list(self.cosets)
            self._points = np.asarray([[p for p, _ in self.cosets[key]] for key in self._keys])
            self._values = np.asarray([[v for _, v in self.cosets[key]] for key in self._keys])
            # Flattened order must preserve the original slicer's tie order.
            if list(self._points.ravel()) != self._decision_points:
                raise ValueError("coset order differs from decision-point order")

    def metrics_and_decision(self, z: complex):
        """One distance pass for DD slicing and trellis metrics (opt-in).

        Fall back to the reference slicer near ties: sqrt/hypot rounding can
        coalesce distinct squared distances, and the original first-point
        rule must survive that case. 7200 keeps its small reference search.
        """
        if (self.backend != "numpy" or self.cosets is None or
                not math.isfinite(z.real) or not math.isfinite(z.imag) or abs(z) > 1e100):
            return self._python_metrics(z), slice_to(z, self._decision_points)
        np = self._np
        distances = (np.square(z.real - self._points.real) +
                     np.square(z.imag - self._points.imag))
        chosen = distances.argmin(axis=1)
        metrics, low = {}, {}
        for row, key in enumerate(self._keys):
            column = chosen[row]
            metrics[key] = float(distances[row, column])
            low[key] = int(self._values[row, column])
        flat = distances.ravel()
        nearest = int(flat.argmin())
        minimum = float(flat[nearest])
        if np.count_nonzero(flat <= minimum + 1e-12 * max(1., minimum)) > 1:
            decision = slice_to(z, self._decision_points)
        else:
            decision = complex(self._points.ravel()[nearest])
        return (metrics, low), decision

    def _metrics(self, z: complex):
        if self.backend == "numpy":
            return self.metrics_and_decision(z)[0]
        return self._python_metrics(z)

    def _python_metrics(self, z: complex):
        metric, low = {}, {}
        if self.rate == 7200:
            for sub in range(8):
                p0 = abs(z - ZTAB4[2 * sub]) ** 2
                p1 = abs(z - ZTAB4[2 * sub + 1]) ** 2
                metric[sub], low[sub] = (p0, 0) if p0 < p1 else (p1, 1)
        else:
            zr, zi = z.real, z.imag
            for key, pts in self.cosets.items():
                best = None
                for pz, val in pts:
                    dr, di = zr - pz.real, zi - pz.imag
                    d = dr * dr + di * di
                    if best is None or d < best[0]:
                        best = (d, val)
                metric[key], low[key] = best
        return metric, low

    def push(self, z: complex, *, metrics=None):
        metric, low = self._metrics(z) if metrics is None else metrics
        nsv, npaths = [self.INF] * 8, [None] * 8
        for j in range(8):
            if self.sv[j] >= self.INF:
                continue
            for k in range(4):
                u = TFSM[j][k]
                key = (j & 1) * 4 + k if self.rate == 7200 else (j & 1, k)
                m, b = metric[key], low[key]
                plen = self.sv[j] * 0.99 + m
                if plen < nsv[u]:
                    nsv[u] = plen
                    npaths[u] = self.paths[j] + [(k << 8) | b]
        self.sv = nsv
        self.paths = [pp[-TLEN:] if pp else [] for pp in npaths]
        best_state = min(range(8), key=lambda j: self.sv[j])
        path = self.paths[best_state]
        if len(path) < TLEN:
            return None
        r = path[0]
        k, low_bits = r >> 8, r & 0xFF
        if self.rate == 7200:
            q = ((self.prev << 2) | k) & 0xF
            self.prev = k
            return (TD_DIFF[q] << 1) | (low_bits & 1)
        y = ((k & 1) << 1) | (k >> 1)           # Y2Y1 from k = Y1<<1|Y2
        dq = (y - self.prev) & 3                # Q2Q1
        self.prev = y
        n = TCM_BITS[self.rate]
        return ((dq & 1) << (n - 1)) | ((dq >> 1) << (n - 2)) | low_bits


def viterbi_7200(ys: list[complex], prev_k: int = 0) -> list[int]:
    v = ViterbiStream(7200)
    v.prev = prev_k & 3
    return [g for g in (v.push(z) for z in ys) if g is not None]


def viterbi_9600(ys: list[complex]) -> list[int]:
    v = ViterbiStream(9600)
    return [g for g in (v.push(z) for z in ys) if g is not None]


class V32Rx:
    def __init__(self, fs: float, tap: int = GPA, eq_taps: int = 32, points=None):
        self.points = points or ZTAB4
        self.fs = fs
        self.tap = tap
        self.eq = Equalizer(eq_taps)

    def run(self, samples, trn_start_s: float, search_s: float = 0.4, trn_len: int = 8044,
            mu_train: float = 0.05, mu_dd: float = 0.01, passes: int = 1):
        """Returns (descrambled bits after the TRN, diagnostics dict)."""
        bb = to_baseband(samples, self.fs)
        u = resample(bb, self.fs)
        # coarse alignment: correlate symbol-rate samples (both T/2 phases) with the expected TRN
        ref = expected_trn(400, self.tap)
        c0 = int((trn_start_s - search_s) * BAUD)
        span = int(2 * search_s * BAUD)
        best = None
        for ph in (0, 1):
            for lag in range(span):
                s0 = c0 + lag
                seg = [u[2 * (s0 + i) + ph] for i in range(0, 400, 1)] if 2 * (s0 + 400) + ph < len(u) else None
                if seg is None:
                    break
                num = sum(r.conjugate() * z for r, z in zip(ref, seg))
                den = math.sqrt(sum(abs(z) ** 2 for z in seg) * 400 * 10) + 1e-9
                score = abs(num) / den
                if best is None or score > best[0]:
                    best = (score, s0, ph)
        score, s0, ph = best
        eq = self.eq
        n = eq.n
        c = ph + n // 2 + 0           # window centred on the nominal symbol position
        trn = expected_trn(trn_len, self.tap)
        # train
        errs = []
        for k in range(trn_len):
            sk = s0 + k
            if 2 * sk + c >= len(u):
                break
            y, x = eq.output(u, sk, c)
            err = trn[k] - y
            eq.adapt(x, err, mu_train)
            if k % 500 == 0:
                errs.append(abs(err))
        # decision-directed from the TRN end; ``passes`` > 1 re-runs it from the same start with
        # the equalizer weights left by the previous pass, so the first symbols (decided while
        # the equalizer was still adapting to the data constellation) are decided cleanly too
        ph_acc, freq = 0.0, 0.0
        for _pass in range(passes):
            syms = []
            ys: list[complex] = []
            sk = s0 + trn_len
            while 2 * sk + c < len(u):
                y, x = eq.output(u, sk, c)
                yr = y * cmath.exp(-1j * ph_acc)
                d = slice_to(yr, self.points)
                e = d - yr
                perr = (yr * d.conjugate()).imag / (abs(d) ** 2)
                ys.append(yr)
                if mu_dd > 0:                       # mu_dd == 0 freezes equalizer and PLL
                    freq += 0.0005 * perr
                    ph_acc += 0.02 * perr + freq
                    eq.adapt(x, e * cmath.exp(1j * ph_acc), mu_dd)
                syms.append(slice4(yr))
                sk += 1
        # differential decode + descramble (the descrambler self-synchronises)
        scr = Scrambler(self.tap)
        for k in range(min(30, len(syms))):
            pass
        prev = slice4(trn[-1])
        bits = []
        for k in syms:
            d = NR_DIFF[(prev << 2) | k]
            prev = k
            bits += [scr.rev((d >> 1) & 1), scr.rev(d & 1)]
        diag = {"corr": score, "start_symbol": s0, "phase": ph,
                "train_err": errs[:2] + errs[-2:] if errs else [], "ys": ys}
        return bits, diag


def deframe_async(bits) -> bytes:
    """Start bit 0, 8 data bits LSB first, stop bit 1; ones while idle."""
    out = bytearray()
    i = 0
    while i + 10 <= len(bits):
        if bits[i] == 0 and bits[i + 9] == 1:
            out.append(sum(bits[i + 1 + j] << j for j in range(8)))
            i += 10
        else:
            i += 1
    return bytes(out)


def find_word(bits, word: int, from_bit: int = 0):
    """Bit index of the first occurrence of the 16-bit ``word`` (MSB first), or -1."""
    pat = [(word >> (15 - i)) & 1 for i in range(16)]
    for i in range(from_bit, len(bits) - 16):
        if bits[i:i + 16] == pat:
            return i
    return -1


def trellis_bytes(ys, bits, e_word: int, rate: int = 7200) -> bytes:
    """Find E in the 4800-coded rate stream, then Viterbi-decode the 7200 bps data after it,
    descramble and deframe. ``ys`` are the equalised symbols aligned with ``bits``."""
    i = find_word(bits, e_word)
    if i < 0:
        return b""
    start = (i + 16) // 2                     # first trellis symbol
    scr = Scrambler(GPA)
    dbits = []
    if rate in TCM_BITS:
        v = ViterbiStream(rate)
        n = TCM_BITS[rate]
        for z in ys[start:]:
            g = v.push(z)
            if g is not None:
                for sh in range(n - 1, -1, -1):
                    dbits.append(scr.rev((g >> sh) & 1))
    else:
        for g in viterbi_7200(ys[start:], prev_k=0):
            for sh in (2, 1, 0):
                dbits.append(scr.rev((g >> sh) & 1))
    return deframe_async(dbits)
