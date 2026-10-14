"""V.34 data-mode style QAM transmitter pieces and the Phase 3/4 training signals (ITU-T V.34 10.1.3, clauses 5 and 7).

Written from the V.34 text. Symbols are complex numbers on the odd-integer grid of the superconstellation (Figure 5:
point 0 of the quarter superconstellation is (1, 1); the other three points of the 4-point constellation are point 0
rotated clockwise by 90, 180, 270 degrees). ``qam_modulate`` shapes them with a root raised cosine pulse and modulates a
carrier at 8 kHz.
"""

from __future__ import annotations

import math

import numpy as np

FS = 8000
P0 = complex(1, 1)
ROT_CW = {0: 1, 1: -1j, 2: -1, 3: 1j}                      # clockwise rotation by n * 90 degrees
# 16-point constellation: labels 0..3 of the quarter superconstellation (Figure 5), each rotated by
# multiples of 90 degrees (9.1: "L/4 points ... labels 0 through L/4 - 1, and the 3L/4 points obtained by
# 90, 180 and 270 degree rotations").  Points are (4i+1, 4j+1) ordered by magnitude, ties by the greater
# imaginary component first (see v34_data.quarter_points): (1,1), (-3,1), (1,-3), (-3,-3).
C16 = [(1, 1), (-3, 1), (1, -3), (-3, -3)]
# exact symbol rates S = (a/c) * 2400 (Table 1); tables elsewhere are keyed by the rounded value
SYMBOL_RATES = {0: 2400.0, 1: 2400.0 * 8 / 7, 2: 2400.0 * 7 / 6, 3: 2400.0 * 5 / 4, 4: 2400.0 * 4 / 3, 5: 2400.0 * 10 / 7}
# (low carrier, high carrier) in Hz, Table 2
def _carriers():
    """Carrier = (d/e) * S, Table 2: (low carrier d, e, high carrier d, e) per nominal symbol rate."""
    spec = {2400: (2, 3, 3, 4), 2743: (3, 5, 2, 3), 2800: (3, 5, 2, 3), 3000: (3, 5, 2, 3), 3200: (4, 7, 3, 5),
            3429: (4, 7, 4, 7)}
    exact = {2400: 2400.0, 2743: 2400.0 * 8 / 7, 2800: 2400.0 * 7 / 6, 3000: 3000.0, 3200: 3200.0, 3429: 2400.0 * 10 / 7}
    return {n: (exact[n] * s[0] / s[1], exact[n] * s[2] / s[3]) for n, s in spec.items()}


CARRIERS = _carriers()                                      # exact values, keyed by the nominal symbol rate
# Phase 3 J announces the constellation phase 4 will use, so there are two of them even though phase 3
# always runs on the 4-point constellation.  2400 bps stays 4-point; 4800 bps and up use 16 points or
# more.  Getting this wrong is invisible on our own loopback (both ends share the constant) and fatal
# against a real modem: a Conexant answerer at 33600 sends J_16PT, and a J_4PT-only receiver never
# matches it.
J_4PT = [int(c) for c in "0000100110010001"]                # 0x0991
J_16PT = [int(c) for c in "0000110110010001"]               # 0x0D91
J_PRIME = [int(c) for c in "1111100110010001"]              # 0xF991
# Keyed by the constant's own name, which is the one convention everywhere else: v34_rx.J_PATTERNS, the
# j_pattern parameters and Phase3Rx's reported J pattern (what the answerer stores as call_j_pattern).
J = {"J_4PT": J_4PT, "J_16PT": J_16PT}


def j_pattern_name(rate: int) -> str:
    """Name of the J pattern to send at ``rate``: "J_16PT" above 2400 bps, "J_4PT" at 2400."""
    return "J_16PT" if rate > 2400 else "J_4PT"


def constellation_power(constellation: int) -> float:
    """Mean |symbol|^2 of a constellation.

    StreamQam normalises by sqrt(power/2), so ``power`` has to be the mean symbol energy of
    whatever constellation is being transmitted, not a constant. The 4-point set is the
    single ring |s|^2 = 2; the 16-point superconstellation has inner points at |s|^2 = 2 and
    outer at |s|^2 = 18, averaging 10. Using the 4-point value for 16-point therefore
    transmits 5x the intended power -- 7 dB.

    Measured on the live TuxedoCat call: our transmit is 0.0196 rms through phase 3 (4-point)
    and 0.0458 rms from phase 4 onward (16-point), a 2.34 amplitude ratio against 2.24
    predicted here.

    Computed from constellation_points() rather than hardcoded so it cannot drift from the
    point set.
    """
    return float(np.mean(np.abs(np.asarray(constellation_points(constellation))) ** 2))


def constellation_for_rate(rate: int) -> int:
    """Our preferred receive training constellation, requested by our J.

    The remote J independently selects our transmit training constellation.
    """
    return 16 if rate > 2400 else 4


def rrc(t: np.ndarray, T: float, beta: float) -> np.ndarray:
    """Root raised cosine pulse with unit energy per symbol period (integral of g^2 = T), vectorized."""
    x = np.asarray(t, dtype=float) / T
    out = np.empty_like(x)
    at0 = np.abs(x) < 1e-9
    atq = np.abs(np.abs(x) - 1 / (4 * beta)) < 1e-9
    reg = ~(at0 | atq)
    xr = x[reg]
    out[reg] = (np.sin(math.pi * xr * (1 - beta)) + 4 * beta * xr * np.cos(math.pi * xr * (1 + beta))) / (
        math.pi * xr * (1 - (4 * beta * xr) ** 2))
    out[at0] = 1 - beta + 4 * beta / math.pi
    out[atq] = beta / math.sqrt(2) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
                                      + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta)))
    return out


def qam_modulate(symbols, baud: float, fc: float, rms: float, beta: float = 0.15, span: int = 8, n0: int = 0,
                 delay: float = 0.0, power: float | None = None, n_total: int | None = None,
                 baseband: bool = False) -> np.ndarray:
    """Passband samples for ``symbols`` (symbol k centred at (k + delay) / baud seconds, sample n at absolute n0 + n).

    ``rms`` is the rms of the passband signal for a symbol stream of average power ``power`` (default: the average
    power of ``symbols`` itself)."""
    a = np.asarray(symbols, dtype=complex)
    T = 1.0 / baud
    p = np.mean(np.abs(a) ** 2) if power is None else power
    n = int(math.ceil((len(a) + 2 * span) * FS / baud)) if n_total is None else n_total
    t = (np.arange(n) + 0.0) / FS - span * T                # leave `span` symbols of lead-in for the pulse tails
    base = np.zeros(n, dtype=complex)
    k0 = np.floor((t * baud) - delay).astype(int)
    for off in range(-span, span + 1):
        k = k0 + off
        ok = (k >= 0) & (k < len(a))
        tau = t - (k + delay) * T
        g = rrc(tau[ok], T, beta)
        base[ok] += a[k[ok]] * g
    if baseband:
        return base * (rms / math.sqrt(p / 2))
    ph = 2 * math.pi * fc * (np.arange(n) + n0) / FS
    # rms of Re(base * e^{j ph}) = sqrt(p / 2) for unit-energy pulses
    return (base * np.exp(1j * ph)).real * (rms / math.sqrt(p / 2))


class Scrambler:
    """Self-synchronizing scrambler of clause 7: y[n] = x[n] ^ y[n-a] ^ y[n-23]  (a = 5 answer, 18 call)."""

    def __init__(self, role: str):
        self.tap = 5 if role == "answer" else 18
        self.reg = [0] * 23

    def step(self, bit: int) -> int:
        y = bit ^ self.reg[self.tap - 1] ^ self.reg[22]
        self.reg = [y] + self.reg[:-1]
        return y


def s_signal(n: int, bar: bool = False) -> np.ndarray:
    """S: point 0, point 0 rotated CCW 90, ... S-bar: point 0 rotated 180, CCW 270, ... (10.1.3.7)."""
    base = [P0, P0 * 1j]
    if bar:
        base = [-P0, -P0 * 1j]
    return np.array([base[i % 2] for i in range(n)])


def pp_signal() -> np.ndarray:
    out = []
    for k in range(72):
        for i in range(4):
            ang = math.pi * (k * i + 4) / 6 if k % 3 == 1 else math.pi * k * i / 6
            out.append(complex(math.cos(ang), math.sin(ang)))
    return np.array(out)


def constellation_points(constellation: int) -> list[complex]:
    """The ``constellation``-point set used for phase-4 training (10.1.3.8/10.1.3.9): the 4-point set is point 0
    rotated by multiples of 90 deg; the 16-point set is quarter-superconstellation labels 0..3 (Figure 5) each
    rotated by multiples of 90 deg (16 distinct points).  Only 4 and 16 are used."""
    if constellation == 4:
        return [complex(*C16[0]) * r for r in ROT_CW.values()]
    assert constellation == 16, constellation
    out = []
    for (x, y) in C16:
        p = complex(x, y)
        out.extend(p * r for r in ROT_CW.values())
    return out


def rotation_of(p: complex) -> int:
    """Rotation index n (0..3) such that p = C16[q] * ROT_CW[n] for some quarter index q, i.e. the clockwise
    90-degree turns applied to whichever quarter-superconstellation point was selected.  This is the value the
    differential coder for MP / J' / E must be seeded with (it is initialized from the last TRN symbol)."""
    best = None
    for q in range(4):
        for n in range(4):
            d = abs(p - complex(*C16[q]) * ROT_CW[n])
            if best is None or d < best[0]:
                best = (d, n)
    if best is None or best[0] >= 1e-6:
        raise ValueError(f"{p} is not a training constellation point")
    return best[1]


def trn_last_rotation(trn: np.ndarray, constellation: int) -> int:
    """Rotation index of the final TRN symbol, which seeds the MP / J' / E differential coder (10.1.3.9)."""
    last = trn[-1]
    if constellation == 4:
        return int(round(-np.angle(last / P0) / (math.pi / 2))) % 4
    return rotation_of(last)


def decide_point_idx(points: list[complex]) -> list[complex]:
    """Distinct constellation points in a fixed order for nearest-point slicing."""
    return points


def trn_symbols(n: int, role: str, constellation: int = 4) -> tuple[np.ndarray, Scrambler]:
    """TRN: scrambled ones mapped to the 4- or 16-point 2D constellation (10.1.3.8).  The scrambler is initialized
    to zero.  16-point: four scrambled bits I1,I2,Q1,Q2 per symbol, point 2*Q2+Q1 of the quarter superconstellation
    rotated clockwise by In*90 deg (absolute, no differential coder)."""
    sc = Scrambler(role)
    if constellation == 4:
        out = []
        for _ in range(n):
            i1 = sc.step(1)
            i2 = sc.step(1)
            out.append(P0 * ROT_CW[2 * i2 + i1])
        return np.array(out), sc
    out = []
    for _ in range(n):
        i1 = sc.step(1)
        i2 = sc.step(1)
        q1 = sc.step(1)
        q2 = sc.step(1)
        q = 2 * q2 + q1
        p = complex(*C16[q])
        out.append(p * ROT_CW[2 * i2 + i1])
    return np.array(out), sc


def j_symbols(pattern: list[int], reps: int, role: str, sc: Scrambler, z0: int = 0, constellation: int = 4) -> tuple[np.ndarray, int]:
    """J / J': pattern bits through the scrambler, differentially encoded on the given constellation."""
    return bit_symbols(pattern * reps, role, sc, z0, constellation)


def bit_symbols(bits: list[int], role: str, sc: Scrambler, z0: int = 0, constellation: int = 4) -> tuple[np.ndarray, int]:
    """Scramble ``bits``, differential-encode, map to the 4- or 16-point constellation (10.1.3.3 / 10.1.3.9).
    This is how J, J', MP and E are sent.  ``z0`` is the differential coder state (mod 4, initialized from the last
    TRN symbol).  Returns the symbols and the final differential coder state.

    4-point: two bits (I1, I2) per symbol, rotate point 0 by Zn*90 deg.  16-point: four bits (I1,I2,Q1,Q2) per
    symbol, select quarter-superconstellation point 2*Q2+Q1 and rotate by Zn*90 deg."""
    pairs = 4 if constellation == 16 else 2
    if len(bits) % pairs:
        bits = list(bits) + [0] * (pairs - len(bits) % pairs)
    z = z0
    out = []
    if constellation == 4:
        for i in range(0, len(bits), 2):
            i1 = sc.step(bits[i])
            i2 = sc.step(bits[i + 1])
            z = (z + 2 * i2 + i1) % 4
            out.append(P0 * ROT_CW[z])
    else:
        for i in range(0, len(bits), 4):
            i1 = sc.step(bits[i])
            i2 = sc.step(bits[i + 1])
            q1 = sc.step(bits[i + 2])
            q2 = sc.step(bits[i + 3])
            z = (z + 2 * i2 + i1) % 4
            out.append(complex(*C16[2 * q2 + q1]) * ROT_CW[z])
    return np.array(out), z


def chunk_symbols(baud: float) -> int:
    """Smallest symbol count that is a whole number of samples and a multiple of 8 (the J pattern period)."""
    for m in range(8, 2000, 8):
        if abs(m * FS / baud - round(m * FS / baud)) < 1e-6:
            return m
    return 168


class StreamQam:
    """Continuous symbol-rate transmitter: queue symbols, read 8 kHz samples. Timing stays continuous across calls.

    Symbols are rendered a chunk at a time with ``span`` symbols of context either side (the next chunk's symbols
    must already be queued; the not yet rendered tail may still be replaced with ``truncate``)."""

    def __init__(self, baud: float, fc: float, rms: float, beta: float = 0.15, span: int = 8, power: float = 2.0):
        self.baud, self.fc, self.rms, self.beta, self.span, self.power = baud, fc, rms, beta, span, power
        # Default is the 4-point mean symbol energy; phase 4 and data mode call set_constellation.
        self.constellation = 4
        self.power = constellation_power(4)
        self.chunk = chunk_symbols(baud)
        self.samples_per_chunk = int(round(self.chunk * FS / baud))
        self.queue: list[complex] = []                     # symbols not yet rendered
        self.history: list[complex] = [0j] * span          # last `span` rendered symbols
        self.rendered_chunks = 0
        self.buf = np.zeros(0)
        self._M = None
        self.preemphasis = None
        self.fill: callable | None = None                  # called to obtain more symbols when the queue runs low

    @property
    def rendered_symbols(self) -> int:
        return self.rendered_chunks * self.chunk

    def set_constellation(self, constellation: int) -> None:
        """Re-normalise for a different constellation, keeping the level continuous.

        The same StreamQam carries phase 3 (4-point) and then phase 4 / data (16-point), so the
        scale factor has to follow the constellation rather than being fixed at construction.
        Output level goes as sqrt(mean|s|^2) / sqrt(power/2), so switching power alone changes the
        transmitted level -- and it is the wrong way round for anything still queued.

        That matters because the switch is not clean: per 10.1.3.4 J-prime is generated by the
        4-point rule of 10.1.3.3 even when J announced 16 points, and the answerer's phase-4 S and
        S-bar are 4-point too. Those are still queued when we switch, so rescale them by
        sqrt(new/old) to leave their output level exactly where it was.
        """
        new_power = constellation_power(constellation)
        old_power = constellation_power(self.constellation)
        if self.queue and abs(new_power - old_power) > 1e-9:
            k = math.sqrt(new_power / old_power)
            self.queue = [z * k for z in self.queue]
        self.power = new_power
        self.constellation = constellation

    def push(self, symbols) -> None:
        self.queue.extend(np.asarray(symbols, dtype=complex).tolist())

    def truncate(self, keep: int) -> None:
        """Drop queued symbols beyond the first ``keep`` (those not yet rendered)."""
        del self.queue[keep:]

    def _matrix(self) -> np.ndarray:
        """Pulse matrix (samples_per_chunk x (chunk + 2 span)): baseband = M @ [history | chunk | lookahead]. Chunks
        are a whole number of samples, so the matrix is the same for every chunk."""
        if self._M is None:
            T = 1.0 / self.baud
            n = self.samples_per_chunk
            t = np.arange(n) / FS
            k0 = np.floor(t * self.baud + 1e-9).astype(int)
            M = np.zeros((n, self.chunk + 2 * self.span))
            for off in range(-self.span, self.span + 1):
                k = k0 + off
                ok = (k >= -self.span) & (k < self.chunk + self.span)
                M[np.nonzero(ok)[0], k[ok] + self.span] += rrc((t - k * T)[ok], T, self.beta)
            self._M = M
            self._carrier = np.exp(2j * math.pi * self.fc * np.arange(n) / FS)
            self._chunk_rot = np.exp(2j * math.pi * self.fc * n / FS)
        return self._M

    def _render_chunk(self) -> None:
        need = self.chunk + self.span
        while len(self.queue) < need and self.fill is not None:
            self.fill(self)
        cur = self.queue[:self.chunk]
        ahead = self.queue[self.chunk:self.chunk + self.span]
        ahead = ahead + [0j] * (self.span - len(ahead))
        if len(cur) < self.chunk:
            self.underrun_symbols = getattr(self, 'underrun_symbols', 0) + self.chunk - len(cur)   # zero-padded: nothing queued
        cur = cur + [0j] * (self.chunk - len(cur))
        ctx = np.array(self.history + cur + ahead)
        M = self._matrix()
        base = M @ ctx
        rot = self._chunk_rot ** self.rendered_chunks
        out = (base * self._carrier * rot).real * (self.rms / math.sqrt(self.power / 2))
        if self.preemphasis is not None:
            out = self.preemphasis.process(out)
        self.history = (self.history + cur)[-self.span:]
        del self.queue[:self.chunk]
        self.rendered_chunks += 1
        self.buf = np.concatenate([self.buf, out])

    def read(self, n: int) -> np.ndarray:
        while len(self.buf) < n:
            self._render_chunk()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out
