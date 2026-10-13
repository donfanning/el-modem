"""V.34 Phase 3 receiver pieces: locate S/S-bar/PP, train a T/2 equalizer on the known PP and TRN, find the end of TRN
and decode J (ITU-T V.34 10.1.3, 11.3.1.1).

Block oriented (works on a buffer of 8 kHz audio); the streaming caller calls it as audio arrives.
"""

from __future__ import annotations

import math
from collections import deque
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import v34_tx as tx
from v34_transport import FeedbackValidity

FS = 8000


def phase4_equalizer_span() -> int:
    """Opt-in longer answer-side training window; default remains qualified span.

    Span 24 reduces hardware TRN/B1 error but does not yet decode hardware B1.
    It needs sustained SIP qualification before replacing the default.
    """
    import os
    span = int(os.environ.get('V34_PHASE4_EQ_SPAN', '12'))
    if span not in (12, 24):
        raise ValueError('V34_PHASE4_EQ_SPAN must be 12 or 24')
    return span


def to_baseband(x: np.ndarray, fc: float, bw: float = 1700.0, taps: int = 81) -> np.ndarray:
    n = np.arange(len(x))
    z = x * np.exp(-2j * math.pi * fc * n / FS)
    k = np.arange(taps) - taps // 2
    h = np.sinc(2 * bw / FS * k) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(z, h, mode="same")


def interp(z: np.ndarray, pos: np.ndarray, half: int = 6) -> np.ndarray:
    """Windowed-sinc interpolation of complex baseband ``z`` at fractional indices ``pos`` (vectorized over taps)."""
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    m = np.arange(-half + 1, half + 1)
    idx = i0[:, None] + m[None, :]
    np.clip(idx, 0, len(z) - 1, out=idx)
    d = m[None, :] - frac[:, None]
    w = np.sinc(d) * (0.5 + 0.5 * np.cos(math.pi * d / half))
    return np.einsum("ij,ij->i", z[idx], w)


def interp_pair(z: np.ndarray, pos: np.ndarray, half: int = 6, second: bool = False):
    """``interp`` and its derivative(s) with respect to position, sharing one gather.

    The window weights need sin/cos of pi*(m - f) and pi*(m - f)/half for integer taps m and per-row fraction
    f, so they are formed from per-row sin/cos of pi*f and pi*f/half with the angle-difference identities
    (4 trigonometric evaluations per row instead of per weight). With ``second`` the second derivative is
    returned as a third value.
    """
    i0 = np.floor(pos).astype(np.int64)
    f = pos - i0
    m = np.arange(-half + 1, half + 1)
    idx = i0[:, None] + m[None, :]
    np.clip(idx, 0, len(z) - 1, out=idx)
    sf, cf = np.sin(math.pi * f)[:, None], np.cos(math.pi * f)[:, None]
    sa, ca = np.sin(math.pi * f / half)[:, None], np.cos(math.pi * f / half)[:, None]
    sign = np.where(m % 2 == 0, 1.0, -1.0)[None, :]
    spm, cpm = np.sin(math.pi * m / half)[None, :], np.cos(math.pi * m / half)[None, :]
    d = m[None, :] - f[:, None]
    zero = np.abs(d) < 1e-12
    safe = np.where(zero, 1.0, d)
    sinpd, cospd = -sign * sf, sign * cf                      # sin(pi*d), cos(pi*d)
    sinc = np.where(zero, 1.0, sinpd / (math.pi * safe))
    dsinc = np.where(zero, 0.0, (cospd - sinc) / safe)         # d/dd of sinc(d)
    cos_a = cpm * ca + spm * sa                                # cos(pi*d/half)
    sin_a = spm * ca - cpm * sa                                # sin(pi*d/half)
    window = 0.5 + 0.5 * cos_a
    dwindow = -0.5 * math.pi / half * sin_a                    # d/dd of the window
    zv = z[idx]
    value = np.einsum("ij,ij->i", zv, sinc * window)
    derivative = -np.einsum("ij,ij->i", zv, dsinc * window + sinc * dwindow)
    if not second:
        return value, derivative
    d2sinc = np.where(zero, -math.pi ** 2 / 3, (-math.pi * sinpd - 2 * dsinc) / safe)
    d2window = -0.5 * (math.pi / half) ** 2 * cos_a
    # d^2/dp^2 = d^2/dd^2 (two sign flips)
    curvature = np.einsum("ij,ij->i", zv, d2sinc * window + 2 * dsinc * dwindow + sinc * d2window)
    return value, derivative, curvature


class Phase3Rx:
    """Receiver for the answer modem's S, S-bar, PP, TRN, J at ``baud`` with carrier ``fc`` (sender role 'answer')."""

    LEAD = 8                                                # pulse lead-in symbols used by tx.qam_modulate

    def __init__(self, baud: float, fc: float, sender_role: str = "answer"):
        self.baud, self.fc, self.role = baud, fc, sender_role
        self.spl = FS / baud
        self.bw = max(1700.0, 0.62 * baud)                  # baseband low-pass cut-off
        self.lag = self.delay = None
        self.gain = None
        self.taps = None

    def update_baseband(self, x: np.ndarray, *, origin: int = 0) -> np.ndarray:
        """Mix/filter an append-only audio buffer, revising the centered FIR's trailing edge.

        Old samples need no further work once the filter has its future
        context. The storage grows geometrically; callers retain the same
        absolute sample coordinates used by acquisition and the equalizer.
        """
        previous = getattr(self, "_bb_input_len", 0)
        old_origin = getattr(self, '_bb_origin', 0)
        if origin < old_origin or origin > old_origin + previous:
            raise ValueError('baseband input origin must advance inside retained PCM')
        drop = origin - old_origin
        if drop:
            self._bb_storage = self._bb_storage[drop:previous].copy()
            previous -= drop
        self._bb_origin = origin
        if len(x) == 0:
            self._bb_input_len = 0
            return np.zeros(0, dtype=complex)
        if len(x) < previous:
            previous = 0
        if len(x) == previous and hasattr(self, "_bb_storage"):
            return self._bb_storage[:len(x)]
        taps = 81
        half = taps // 2
        start = max(0, previous - 2 * half)
        n = np.arange(start, len(x)) + origin
        z = x[start:] * np.exp(-2j * math.pi * self.fc * n / FS)
        k = np.arange(taps) - half
        h = np.sinc(2 * self.bw / FS * k) * np.hamming(taps)
        h /= h.sum()
        filtered = np.convolve(z, h, mode="full")[half:half + len(z)]
        output_start = max(0, previous - half)
        storage = getattr(self, "_bb_storage", None)
        if storage is None or len(storage) < len(x):
            expanded = np.empty(max(len(x), 2 * len(storage) if storage is not None else 1024), dtype=complex)
            if storage is not None and output_start:
                expanded[:output_start] = storage[:output_start]
            self._bb_storage = storage = expanded
        storage[output_start:len(x)] = filtered[output_start - start:]
        self._bb_input_len = len(x)
        return storage[:len(x)]

    # -- acquisition -----------------------------------------------------------------------------------------------
    def acquire(self, x: np.ndarray, search: int | None = None, delays=(0.0, 0.25, 0.5, 0.75), seq=None,
                min_metric: float = 0.6) -> dict | None:
        """Find the start of S (128T) + S-bar (16T) + PP (288T) in x by correlation. Returns lag/delay/gain or None."""
        z = to_baseband(x, self.fc, self.bw)
        if seq is None:
            seq = np.concatenate([tx.s_signal(128), tx.s_signal(16, True), tx.pp_signal()])
        best = None
        n_ref = int((len(seq) + 2 * self.LEAD) * self.spl)
        if len(z) < n_ref + 100:
            return None
        zz = z[:search] if search else z
        for d in delays:
            ref = tx.qam_modulate(seq, self.baud, 0, 1, delay=d, baseband=True)
            ref = ref[:n_ref]
            nfft = 1 << int(math.ceil(math.log2(len(zz) + len(ref))))
            c = np.fft.ifft(np.fft.fft(zz, nfft) * np.conj(np.fft.fft(ref, nfft)))[:len(zz) - len(ref) + 1]
            e = np.convolve(np.abs(zz) ** 2, np.ones(len(ref)), mode="valid")
            m = np.abs(c) / np.sqrt(e * np.sum(np.abs(ref) ** 2) + 1e-9)
            i = int(np.argmax(m))
            if best is None or m[i] > best[0]:
                best = (float(m[i]), i, d, c[i] / np.sum(np.abs(ref) ** 2))
        if best is None or best[0] < min_metric:
            # Keep the best correlation we saw even when we reject it, so a peer we fail to acquire
            # can be diagnosed (how far below the threshold, and at which delay) instead of the
            # acquire simply reporting nothing.
            self.last_best = None if best is None else {"metric": best[0], "lag": best[1], "delay": best[2]}
            return None
        self.lag, self.delay, self.gain = best[1], best[2], best[3]
        self.metric = best[0]
        self._z = z
        return {"metric": best[0], "lag": self.lag, "delay": self.delay, "gain": abs(self.gain)}

    # -- symbol sampling (two samples per symbol) ------------------------------------------------------------------
    def samples(self, k0: int, count: int) -> np.ndarray:
        """T/2-spaced samples for symbols k0..k0+count-1 (shape count x 2)."""
        k = np.arange(k0, k0 + count)
        t = self.lag + (self.LEAD + k + self.delay) * self.spl
        origin = getattr(self, '_bb_origin', 0)
        s0 = interp(self._z, t-origin)
        s1 = interp(self._z, t + self.spl / 2-origin)
        return np.stack([s0, s1], axis=1)

    # -- equalizer --------------------------------------------------------------------------------------------------
    def train(self, n_pp: int = 288, n_trn: int = 700, span: int = 12, reg: float = 1e-3) -> float:
        """Least-squares T/2 equalizer from PP and the first TRN symbols (known). Returns the normalized final MSE."""
        k_start = 128 + 16                                  # PP begins after S + S-bar
        n = n_pp + n_trn
        trn, _ = tx.trn_symbols(n_trn, self.role)
        known = np.concatenate([tx.pp_signal()[:n_pp], trn])
        L = 2 * span
        self._span, self._L = span, L
        A = self._design(k_start, n)
        G = A.conj().T @ A
        self.taps = np.linalg.solve(G + reg * np.eye(L) * np.trace(G).real / L, A.conj().T @ known)
        err = A @ self.taps - known
        return float(np.mean(np.abs(err) ** 2) / np.mean(np.abs(known) ** 2))

    def train_trn_auto(self, n_trn: int = 700, span: int = 12, reg: float = 1e-3,
                       bad_mse: float = 0.25) -> tuple[float, str]:
        """Train from PP + TRN, detecting which scrambler polynomial the peer actually used.

        Clause 7 gives each transmission direction its own polynomial: the answer modem uses
        GPA = 1 + x^-5 + x^-23, the call modem GPC = 1 + x^-18 + x^-23. As the caller we expect
        GPA on the answerer's phase-3 TRN, so we try that first and only fall back to GPC if the
        fit is poor.

        The fallback exists because several real peers in the Gough collection (conexant-hsfi,
        swann-speeddemon, conexant-rs56, intel-ham) put out a phase-3 TRN that correlates with
        GPC at mse ~0.01 and with GPA at mse ~0.93, while others (bcm, ageresv92pp, ess) match GPA
        at ~0.025. PP trains cleanly in both groups (mse 0.001-0.014), so this is specific to the
        scrambled TRN and not a timing or alignment problem.

        Why those peers do it is not established -- it may be a genuine deviation, or their
        recordings may have the stereo channels swapped relative to the rest of the collection.
        Either way the receiver can decide from the signal rather than assume, which is why this
        detects rather than hardcodes. Returns (mse, role_used).
        """
        k_start = 128 + 16
        pp = tx.pp_signal()[:288]
        order = [self.role] + [r for r in ("answer", "call") if r != self.role]
        best: tuple[float, str] | None = None
        for role in order:
            trn, _ = tx.trn_symbols(n_trn, role)
            mse = self.train_known(np.concatenate([pp, trn]), k_start, span=span, reg=reg)
            if best is None or mse < best[0]:
                best = (mse, role)
            if mse <= bad_mse:
                break                       # conformant fit: stop, do not second-guess the peer
        return best

    def train_known(self, known: np.ndarray, k_start: int, span: int = 12, reg: float = 1e-3) -> float:
        """Least-squares T/2 equalizer from an arbitrary known symbol sequence starting at symbol ``k_start``."""
        n = len(known)
        L = 2 * span
        self._span, self._L = span, L
        A = self._design(k_start, n)
        G = A.conj().T @ A
        self.taps = np.linalg.solve(G + reg * np.eye(L) * np.trace(G).real / L, A.conj().T @ known)
        err = A @ self.taps - known
        return float(np.mean(np.abs(err) ** 2) / np.mean(np.abs(known) ** 2))

    def _design(self, k0: int, count: int) -> np.ndarray:
        span, L = self._span, self._L
        flat = self.samples(k0 - span, count + 2 * span + 2).reshape(-1)
        A = np.empty((count, L), dtype=complex)
        for i in range(count):
            A[i] = flat[2 * i + span:2 * i + span + L]
        return A

    def equalize(self, k0: int, count: int) -> np.ndarray:
        return self._design(k0, count) @ self.taps


def slice4(y: np.ndarray) -> np.ndarray:
    """Rotation index (clockwise quarter turns of point 0) of each equalized symbol."""
    ang = np.angle(y / tx.P0)                               # y = P0 * (-j)^n  ->  angle = -n * 90 degrees
    n = np.round(-ang / (math.pi / 2)).astype(int) % 4
    return n


def descramble_tap(role: str) -> int:
    """Descrambler tap: the answerer's polynomial is GPa (tap 5), the caller's is GPC (tap 18)."""
    return 5 if role == "answer" else 18


def descramble(bits: list[int], role: str) -> list[int]:
    tap = descramble_tap(role)
    out = []
    for i, y in enumerate(bits):
        a = bits[i - tap] if i >= tap else 0
        b = bits[i - 23] if i >= 23 else 0
        out.append(y ^ a ^ b)
    return out


def _interleave(vals: np.ndarray) -> np.ndarray:
    """Turn one 2-bit value per symbol into the descrambler's bit stream (I1 then I2)."""
    b = np.empty(2 * vals.size, dtype=np.int64)
    b[0::2] = vals & 1
    b[1::2] = vals >> 1
    return b


def _interleave4(rot: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """Turn per-symbol rotation (I) and quarter (Q) values into the 16-point descrambler's bit stream,
    four bits per symbol, ordered I1, I2, Q1, Q2."""
    b = np.empty(4 * rot.size, dtype=np.int64)
    b[0::4] = rot & 1
    b[1::4] = rot >> 1
    b[2::4] = qv & 1
    b[3::4] = qv >> 1
    return b


def _descramble_np(bits: np.ndarray, tap: int) -> np.ndarray:
    """Vectorized ``descramble``: the recursion is memoryless (each output reads the *input* taps),
    so this is an exact, allocation-cheap equivalent of the loop above."""
    d = bits.copy()
    if tap:
        d[tap:] ^= bits[:-tap]
    d[23:] ^= bits[:-23]
    return d


def trn_end_from_mismatch(mismatch, min_run: int = 64, last: bool = False) -> int | None:
    """Symbol index just past the last long run of agreement with the known TRN sequence.

    ``mismatch[k]`` is 0 when received index k equals the expected TRN index. Those agree essentially
    perfectly for the whole TRN and only sporadically afterwards, so the end of the longest run of
    agreements is the transition, pinned to the exact symbol - which matters, because the descrambler
    needs the split right: a couple of symbols out corrupts the first dozen or so descrambled bits of J
    and loses the match outright.

    A sliding-window mismatch test cannot do this. After TRN the signal is 4-point differentially coded
    and matches a known index a quarter of the time, so the mismatch rate approaches a 75% ceiling rather
    than 100%, and the rate right at the boundary averages somewhere below that - no fixed threshold
    separates them. Agreement has no such ceiling: the odds of 64 consecutive agreements after TRN are
    25**-64.

    Limitation: a timing slip *inside* the TRN would cut the run short. The reference capture has none,
    and bridging them would mean walking every mismatch on every call, which is far too slow for a
    scan that runs every eight symbols.

    ``last`` selects the end of the *last* run of at least ``min_run`` agreements instead of the longest.
    It is the recovery for a burst of decision errors inside the TRN (a real peer under drift and delay):
    the burst splits the TRN's run, the first fragment is the longest, and the longest-run rule then pins
    the end to the burst. After the TRN, agreement of ``min_run`` symbols is essentially impossible, so the
    last long run is the TRN's tail and its end is the true end. See ``Phase4Rx._rearm_end_estimate``.
    """
    m = np.asarray(mismatch)
    n = m.size
    if n == 0:
        return None
    ok = (m == 0).astype(np.int64)
    d = np.diff(np.concatenate(([0], ok, [0])))             # +1 opens a run of agreements, -1 closes it
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lens = ends - starts
    if lens.size == 0 or int(lens.max()) < min_run:
        return None
    if last:
        return int(ends[np.flatnonzero(lens >= min_run)[-1]])
    return int(ends[int(np.argmax(lens))])                 # first of the longest: the TRN, not a later echo


def find_trn_end(idx: np.ndarray, expected: np.ndarray) -> int | None:
    """First symbol index where the received rotation indices stop following the expected TRN sequence."""
    idx = np.asarray(idx)
    expected = np.asarray(expected)
    n = min(idx.size, expected.size)
    if n == 0:
        return None
    return trn_end_from_mismatch((idx[:n] != expected[:n]).astype(np.int64))


J_PATTERNS = ("J_4PT", "J_16PT")                # phase 3 J, named for the constellation phase 4 will use
# The abs/differential split is latched from a 16-symbol mismatch test, so it is only good to a few
# symbols. Getting it a little early corrupts the first ~23 descrambled bits after it; a little late
# is worse, because the symbols it wrongly calls absolute poison everything from there on. So the
# search tries a short ladder of splits instead of trusting the estimate exactly.
_SPLIT_LADDER = (0, -8, 8, -16, 16)


def _pattern_starts(bits: np.ndarray, pat: np.ndarray) -> list[int]:
    """Symbol offsets into a descrambled bit run where ``pat`` matches exactly, in ascending order."""
    if bits.size < pat.size:
        return []
    win = sliding_window_view(bits, pat.size)[::2]            # candidates are symbol aligned, so every other offset
    return np.flatnonzero((win == pat).all(axis=1)).tolist()


def decode_j(idx: np.ndarray, trn_end_estimate: int, role: str = "answer", reps: int = 3, lead: int = 128,
             patterns: tuple[str, ...] = J_PATTERNS, mismatch=None):
    """Find the J sequence after TRN.

    Returns ``(j_start, symbols_of_j_seen, pattern_name, split)`` or None. ``j_start`` is where J
    begins - which on a real modem is *not* the TRN end: the Conexant answerer at 33600 switches to
    differential coding at its own TRN end and only reaches J about 46 symbols later. ``split`` is the
    abs/differential transition the descrambler should use, which is what the phase-4 receive chain
    needs and is not the same number. Passing ``mismatch`` (against the known TRN) refines the
    transition to the exact symbol instead of the caller's coarse estimate.

    Both J patterns are tried, because J announces the constellation phase 4 will use and that is the
    16-point one at every rate above 2400 bps. Searching one pass per split and sliding over all
    candidate starts within it keeps the cost flat in the number of candidates.
    """
    if mismatch is not None:
        refined = trn_end_from_mismatch(mismatch)
        if refined is not None:
            trn_end_estimate = refined
    pats = [(name, np.array(getattr(tx, name) * reps, dtype=np.int64)) for name in patterns]
    span = max(p.size for _, p in pats)
    idx = np.asarray(idx, dtype=np.int64)
    lo = max(trn_end_estimate - 12, 24)
    hi = min(trn_end_estimate + lead, idx.size - 1)
    if hi <= lo:
        return None
    # A 48-bit exact match cannot happen by accident, so any candidate is real and the earliest one
    # across the ladder is the true J start. It is always found under a split close enough to the truth
    # that its bits are clean there, which is also the split its run length is measured with.
    best = None
    for off in _SPLIT_LADDER:
        split = trn_end_estimate + off
        if split < 24 or split >= idx.size:
            continue
        b = _descrambled_run(idx, role, split, hi - split + 1)
        for name, pat in pats:
            for j in _pattern_starts(b, pat):
                n = split + j
                if lo <= n <= hi and (best is None or n < best[0]):
                    best = (n, name, split)
    if best is None:
        return None
    n, name, split = best
    return n, _count_j_symbols(idx, role, n, getattr(tx, name), split), name, split


def _count_j_symbols(idx: np.ndarray, role: str, n: int, pattern, split: int | None = None) -> int:
    """How many consecutive symbols after the J start still match the repeated ``pattern``."""
    d = _full_descramble(idx, role, n if split is None else split)
    start = 2 * n
    pat = np.array(list(pattern) * 200, dtype=np.int64)
    m = min((d.size - start) // 2, pat.size // 2)
    if m <= 0:
        return 0
    got = d[start:start + 2 * m].reshape(m, 2)
    at = (2 * np.arange(m)) % pat.size
    bad = np.flatnonzero(~np.all(got == np.stack([pat[at], pat[at + 1]], axis=1), axis=1))
    return int(bad[0]) if bad.size else m


# Normalised correlation an S must reach. 0.65 sits between the false-alarm edge and the weakest real S:
# on the corpus (12.7 min of answering-lane audio away from any S) hits are identical at 0.70, 0.65 and 0.60
# and only grow below 0.60; the weakest real S on a clean local path scores 0.76, but on a live PSTN call
# the answerer's own phase-3 S scored 0.712 and its phase-4 S 0.713 against the 40-symbol reference, so the
# old 0.70 left 0.012 of margin (see lab/V34_PHASE3_STALL_DIAGNOSIS.md).
S_FLIP_THRESHOLD = 0.65


@lru_cache(maxsize=32)
def _s_flip_reference(baud, ref_len):
    """Immutable pulse reference; independent of PCM, carrier phase and gain."""
    lead = 8
    n_ref = int((ref_len + 2 * lead) * (FS / baud))
    ref = tx.qam_modulate(tx.s_signal(ref_len), baud, 0, 1,
                          delay=0.0, baseband=True)[:n_ref]
    ref.flags.writeable = False
    return ref


def detect_s_flip(x: np.ndarray, fc: float, baud: float, ref_len: int = 40, threshold: float = S_FLIP_THRESHOLD):
    """Look for S (alternating point 0 / point 0 + 90 degrees) followed by the inversion that starts S-bar.

    Returns None when no S is found, ("S", sample_index_where_S_starts) while S is still running, or
    ("flip", sample_index_of_the_transition) once S-bar has begun. ``x`` is recent audio (a few hundred ms).

    ``threshold`` is a normalised correlation against ``s_signal(ref_len)``, so it is already
    amplitude-invariant. It is ``S_FLIP_THRESHOLD`` (0.65); it was 0.70, which matches the min_metric
    phase-3 S/PP acquisition uses, and before that 0.75, which is what hid the real Courier's
    phase-4 S on the live TuxedoCat call: the burst begins at 14.500 s and the best correlation was
    0.744 at 14.538 s -- and 128T at 3428.57 baud is 37.3 ms, so 14.538 is exactly the S-to-S-bar
    transition. The peer sent S then S-bar and we discarded it for being 0.006 short.

    res[1] is a sample index into ``x`` in both cases, so a caller that acts on either can use it as
    the anchor for the phase-4 receive chain. On a live dialup.world call (2026-10-07) 0.70 hid the answerer's
    phase-4 S as well: its onset scored 0.713 in an aligned window but 0.698 in the live caller's windows, and the
    S/S-bar pair matched the same call's phase-3 S/S-bar (0.712/0.574) to three digits. Replayed at 0.65 the
    production detector returns a flip 0.465 s after our J. That matters for the "S" case: 11.4.1.2.1 has the answer
    modem send S for 128T, then condition to detect J-prime, and only then send S-bar -- so the
    S-to-S-bar transition is separated by the *caller's* J-prime. A caller that waits for the flip
    before sending J-prime is waiting for a transition it is itself responsible for causing.
    """
    spl = FS / baud
    z = to_baseband(x, fc, max(1700.0, 0.62 * baud))
    lead = 8
    n_ref = int((ref_len + 2 * lead) * spl)
    if len(z) < n_ref + 50:
        return None
    best = None
    ref = _s_flip_reference(baud, ref_len)
    nfft = 1 << int(math.ceil(math.log2(len(z) + len(ref))))
    c = np.fft.ifft(np.fft.fft(z, nfft) * np.conj(np.fft.fft(ref, nfft)))[:len(z) - len(ref) + 1]
    e = np.convolve(np.abs(z) ** 2, np.ones(len(ref)), mode="valid")
    m = np.abs(c) / np.sqrt(e * np.sum(np.abs(ref) ** 2) + 1e-9)
    good = np.nonzero(m > threshold)[0]
    if len(good) == 0:
        return None
    L0 = int(good[0])
    g = c[L0] / np.sum(np.abs(ref) ** 2)
    # symbols after the reference: compare with the continued S pattern
    k = ref_len
    run = 0
    while True:
        t = L0 + (lead + k) * spl
        if t + 8 >= len(z):
            break
        s = interp(z, np.array([t]))[0]
        expect = g * tx.s_signal(k + 1)[k]
        r = (s * np.conj(expect)).real / (abs(expect) ** 2 + 1e-9)
        run = run + 1 if r < 0 else 0
        if run >= 3:
            return ("flip", int(L0 + (lead + k - 2) * spl))
        k += 1
    return ("S", int(L0 + lead * spl))


def _differential(idx: np.ndarray) -> np.ndarray:
    """Each symbol's value relative to its own predecessor (the first symbol relative to itself)."""
    d = np.empty(idx.size, dtype=np.int64)
    d[0] = idx[0]
    d[1:] = (idx[1:] - idx[:-1]) % 4
    return d


def _descrambled_run(idx: np.ndarray, role: str, split: int, n_sym: int, c16: bool = False,
                     qidx=None) -> np.ndarray:
    """Descrambled bits for symbols ``split`` .. ``split + n_sym - 1``, in one vectorized pass.

    Symbols before ``split`` carry absolute rotation indices and those from ``split`` on are
    differentially encoded (the quarter index is always absolute), so the descrambler's 23-bit lookback
    reaches back across the transition - which is why the window has to start before it.  Building the
    whole run in one call is what lets a J search slide over many candidate starts for the price of a
    single pass.

    16-point (``c16``): per symbol the four scrambled bits are I1,I2 (rotation) and Q1,Q2 (quarter,
    from ``qidx``)."""
    tap = descramble_tap(role)
    pad = 24                                   # leading zero bits so the 23-bit lookback is always in range
    s0 = max(split - 12, 0)
    s1 = min(split + n_sym, idx.size)
    if s1 <= s0:
        return np.zeros(0, dtype=np.int64)
    k = np.arange(s0, s1)
    dv = idx[k].copy()
    if dv.size > 1:
        dv[1:] = (idx[k[1:]] - idx[k[:-1]]) % 4
    cut = 4 * (split - s0) if c16 else 2 * (split - s0)   # bit offset where the diff region starts in the window
    if c16:
        # Candidate searches need only this bounded window. Converting the
        # whole growing quarter-index list for every candidate made a short
        # MP search scale with the entire TRN history.
        qv = np.asarray(qidx[s0:s1], dtype=np.int64)
        buf = np.concatenate((np.zeros(pad, dtype=np.int64),
                              _interleave4(idx[k], qv)[:cut], _interleave4(dv, qv)[cut:]))
    else:
        buf = np.concatenate((np.zeros(pad, dtype=np.int64),
                              _interleave(idx[k])[:cut], _interleave(dv)[cut:]))
    d = _descramble_np(buf, tap)
    return d[pad + cut:pad + cut + (4 if c16 else 2) * (s1 - split)]


def _descramble_at(idx: np.ndarray, role: str, n: int, span: int, abs_end: int | None = None,
                   c16: bool = False, qidx=None) -> np.ndarray:
    """Descrambled bits ``[bpn, bpn+span)`` (bpn = 4n if ``c16`` else 2n) of the stream where symbols before
    ``abs_end`` are absolute and symbols from ``abs_end`` on are differential. Only the window needed for those
    bits is built, so cost does not grow with the length of ``idx``; the result may be shorter than ``span``
    near the end.

    ``abs_end`` defaults to ``n``, which says the transition into differential coding happens exactly
    where the bits we want begin. That is only true when ``n`` really is the TRN end. When the bits of
    interest start later - J does, on a real modem - the transition has to be passed separately, or the
    23-bit descrambler lookback reaches back into symbols that are themselves differential and the
    first 23 bits come out wrong.
    """
    bpn = 4 if c16 else 2
    a_end = n if abs_end is None else max(0, min(int(abs_end), n))
    b = _descrambled_run(idx, role, a_end, n + span // bpn + 1 - a_end, c16=c16, qidx=qidx)
    at = bpn * (n - a_end)
    return b[at:at + span]


def _full_descramble(idx: np.ndarray, role: str, n: int, c16: bool = False, qidx=None) -> np.ndarray:
    """Whole descrambled stream for the abs/differential split at ``n`` (used once, on a match)."""
    if c16:
        qv = np.asarray(qidx, dtype=np.int64)
        vals_rot = np.concatenate((idx[:n], _differential(idx)[n:]))
        return _descramble_np(_interleave4(vals_rot, qv), descramble_tap(role))
    vals = np.concatenate((idx[:n], _differential(idx)[n:]))
    return _descramble_np(_interleave(vals), descramble_tap(role))


def decode_after_trn(idx: np.ndarray, end_estimate: int, role: str, sync_ones: int = 17,
                     c16: bool = False, qidx=None):
    """Find where the TRN really ends: the descrambled bits there begin an MP frame (17 sync ones, a 0 start bit,
    a type bit, and a valid CRC).  Returns (trn_length, descrambled_bits_from_there) or None.

    Only the ``sync_ones + 1`` descrambled bits at each candidate end are inspected first, and the
    descrambler is memoryless with a 23-bit lookbehind, so a short window of raw symbols around the
    candidate range is enough - the whole symbol history is never rebuilt. Candidates are a fixed
    window around ``end_estimate``, so the cost per call is independent of how long the call has run.

    The 4-point TRN's descramble noise can spuriously produce ``sync_ones`` ones followed by a 0, so a
    candidate is only accepted once it parses as a whole MP frame (CRC and all) - which a shifted
    estimate never does.  ``c16`` selects the 16-point constellation (4 scrambled bits per symbol:
    I1,I2,Q1,Q2), in which case ``qidx`` supplies the quarter-superconstellation index per symbol
    (rotation bits in ``idx`` are differential after the TRN)."""
    idx = np.asarray(idx, dtype=np.int64)
    n_sym = idx.size
    span = sync_ones + 1
    cands = [n for n in range(max(end_estimate - 24, 2), end_estimate + 24) if n < n_sym]
    if not cands:
        return None
    sync = np.array([1] * sync_ones + [0], dtype=np.int64)
    import v34_mp
    bpn = 4 if c16 else 2
    for n in cands:
        seg = _descramble_at(idx, role, n, span, c16=c16, qidx=qidx)
        if seg.size != span or not np.array_equal(seg, sync):
            continue
        # A sync hit is necessary but not sufficient: parse a full MP frame (up to 188 bits) at this
        # offset so a spurious run of ones in the TRN descramble cannot be mistaken for the boundary.
        frame_bits = _descramble_at(idx, role, n, 188 + 4, c16=c16, qidx=qidx)
        if frame_bits.size >= 88 and v34_mp.parse_mp(frame_bits) is not None:
            return n, _full_descramble(idx, role, n, c16=c16, qidx=qidx)[bpn * n:].tolist()
    return None


class InputPowerGuard:
    """Preserve feedback state across input fades; power return is not lock qualification."""
    ENTER_RATIO = 1e-4
    EXIT_RATIO = 4e-4
    RETURN_SYMBOLS = 8
    MIN_REFERENCE_SAMPLES = 32

    def __init__(self, mode='silence'):
        if mode not in ('silence', 'fade'):
            raise ValueError('unknown input power guard mode')
        self.mode = mode
        if mode == 'fade':
            self.ENTER_RATIO, self.EXIT_RATIO = .25, .5
        self.training = deque(maxlen=128)
        self.reference = None
        self.reference_samples = 0
        self.armed = False
        self.holding = False
        self.return_count = 0
        self.events = self.held_symbols = self.longest_hold = self.current_hold = 0
        self.power_returns = 0

    def observe_training(self, energy):
        if not self.armed and math.isfinite(energy) and energy > 0:
            self.training.append(float(energy))

    def arm(self):
        if self.armed:
            return
        self.armed = True
        self.reference_samples = len(self.training)
        if len(self.training) >= self.MIN_REFERENCE_SAMPLES:
            self.reference = float(np.median(self.training))
        self.training.clear()

    def hold(self, energy):
        if self.reference is None:
            return False
        if not self.holding and (not math.isfinite(energy) or energy < self.ENTER_RATIO*self.reference):
            self.holding = True
            self.events += 1
            self.current_hold = 0
        if self.holding:
            self.return_count = (self.return_count+1 if math.isfinite(energy)
                                 and energy >= self.EXIT_RATIO*self.reference else 0)
            if self.return_count >= self.RETURN_SYMBOLS:
                self.holding = False
                self.return_count = 0
                self.power_returns += 1
                self.current_hold = 0
            else:
                self.current_hold += 1
                self.held_symbols += 1
                self.longest_hold = max(self.longest_hold, self.current_hold)
        return self.holding

    def report(self):
        return dict(mode=self.mode, armed=self.armed, reference_window_energy=self.reference,
            reference_samples=self.reference_samples,
            reference_source='median positive pre-data adaptive window energy; sampled every 16 symbols',
            reference_available=self.reference is not None, feedback_held=self.holding,
            events=self.events, held_symbols=self.held_symbols, longest_hold_symbols=self.longest_hold,
            power_returns=self.power_returns, entry_ratio=self.ENTER_RATIO,
            exit_ratio=self.EXIT_RATIO, return_symbols=self.RETURN_SYMBOLS,
            recovery_scope='input power restored; carrier/timing/payload reacquisition not qualified')


class SymbolEq:
    """Streaming adaptive T/2 equalizer with carrier PLL and tap-centroid timing slips.

    Window for symbol k: samples of the complex baseband ``R._z`` at t_k + (j - span) * T/2, j = 0..2*span-1, with
    t_k = lag + (LEAD + k + delay) * spl + dt. The taps start from the least-squares solution (``R.train*``); every
    symbol they are adapted with NLMS against a known or decided symbol, the phase of the output is tracked by a
    second-order PLL, and when the energy centroid of the taps walks away from the centre by more than ~1 sample the
    window moves by half a symbol (taps shifted) - which follows a sample clock offset of any size.
    """

    def __init__(self, R: Phase3Rx, k_start: int, mu: float | None = None, kp: float | None = None, ki: float | None = None,
                 timing_mode: str | None = None):
        import os
        mu = float(os.environ.get('V34_MU', 0.2)) if mu is None else mu
        kp = float(os.environ.get('V34_KP', 0.01)) if kp is None else kp
        ki = float(os.environ.get('V34_KI', 5e-6)) if ki is None else ki
        # Gradient tracking passes controlled clock tests but has not passed
        # sustained 33600/LAPM SIP qualification. Keep it opt-in until the
        # live media/detection transition is understood.
        requested_mode = timing_mode or os.environ.get('V34_TIMING_MODE', 'centroid')
        if requested_mode not in ('centroid','gradient','data-gradient'):
            raise ValueError('unknown timing mode')
        self.data_timing_mode = requested_mode
        self.timing_mode = 'centroid' if requested_mode == 'data-gradient' else requested_mode
        if self.timing_mode == 'gradient':
            kp,ki = max(kp,.03),max(ki,5e-5)
        self.R = R
        self.span = R._span
        self.L = 2 * self.span
        self.h = np.array(R.taps, dtype=complex)
        self.k = k_start
        self.dt = 0.0
        self.phase = 0.0
        self.freq = 0.0
        self.mu, self.kp, self.ki = mu, kp, ki
        self.adapt = True
        from v34_equalizer_backend import selected
        arithmetic=selected()
        self._rust_interp=arithmetic.interpolate if arithmetic is not None else None
        self._rust_feedback=arithmetic.feedback if arithmetic is not None else None
        self._rust_centroid=arithmetic.centroid if arithmetic is not None else None
        self.arithmetic_backend='rust' if arithmetic is not None else 'legacy'
        # Clock correction can require a fresh sinc window every symbol.
        # Preserve the NumPy reference/default; warm this opt-in in the DCE.
        self._native_interp = False
        if os.environ.get('V34_SYMBOL_INTERP_NATIVE') == '1':
            from v34_interp_native import njit
            self._native_interp = njit is not None
        guard_mode = os.environ.get('V34_INPUT_POWER_GUARD')
        self._input_guard = (InputPowerGuard('fade' if guard_mode == 'fade' else 'silence')
                             if guard_mode in ('1', 'fade') else None)
        # NLMS tap update and carrier PLL can be frozen independently of the timing loop, so a
        # receiver that has finished training (e.g. tracked a real peer's seconds-long MP re-send)
        # can keep the sampling window centred without the tap/phase gains walking into divergence.
        self.adapt_nlms = True
        self.slips = 0
        self.kff = float(os.environ.get('V34_KFF', 0.0))
        self.cf = 0.0
        self.cdead = float(os.environ.get('V34_CDEAD', 0.15))
        self.calpha = float(os.environ.get('V34_CALPHA', 0.03))
        self.c0 = None
        self.vel = 0.0                                      # window velocity, samples per symbol
        self.ktp = float(os.environ.get('V34_KTP', 0.0))
        self.kti = float(os.environ.get('V34_KTI', 0.0))
        self._n = 0
        self._gradient_rows = []
        self._gradient_error = 0.0
        self._gradient_information = 0.0
        self._offs = (np.arange(self.L) - self.span) * R.spl / 2
        self._interp_spl = R.spl
        self._interp_offs = self._offs.copy()
        self._grid = None
        self._g_zlen = -1
        self._g_k0 = 0
        self._g_shift = 0
        self._g_dt = 0.0
        self._g_spl = R.spl
        # Opt-in: keep the batched interpolation grid valid while the timing NCO moves at a learned velocity.
        self._traj = os.environ.get('V34_TRAJECTORY_GRID') == '1'
        self._native_feedback = os.environ.get('V34_NATIVE_FEEDBACK') == '1'
        self._g_vel = 0.0
        self._dgrid = None
        self._g_traj = False
        self._g_built_k = -10**9
        self._res_key = None
        self._res = None
        self._d2grid = None
        self._tj_k = -1                                    # symbol whose window the trajectory grid served
        self._tj_i = 0
        self._tj_dev = 0.0
        self._g_origin = 0
        self._g_zlen = -1
        gap_mode=os.environ.get('V34_GAP_FEEDBACK','0')
        self._transport_guard=FeedbackValidity(gap_mode) if gap_mode!='0' else None

    GRID = 128                                            # symbols interpolated per batch
    MARGIN = 6                                            # spare T/2 samples either side for slips

    def _build_grid(self, trajectory: bool = False) -> None:
        """Interpolate the baseband once onto the T/2 grid for the next GRID symbols (windows overlap heavily).

        With ``trajectory`` the grid follows the predicted timing path dt + vel*(symbols ahead), and a derivative
        grid is kept so each window can add its exact first-order residual (see ``window``)."""
        R = self.R
        half = R.spl / 2
        t0 = R.lag + (R.LEAD + self.k + R.delay) * R.spl + self.dt
        n = np.arange(2 * self.GRID + self.L + 2 * self.MARGIN)
        x = n - self.MARGIN - self.span
        pos = t0 + x * half
        if trajectory:
            pos = pos + self.vel * x / 2
            # Buffered data ends soon after the current symbol (the buffer is rebuilt every chunk), so interpolate
            # only the entries the data can serve; keep at least one full window plus margins.
            last = len(R._z) - 1 - 6 + getattr(R, '_bb_origin', 0)
            keep = max(self.L + 2 * self.MARGIN + 2, int(np.searchsorted(pos, last, side='right')))
            pos = pos[:keep]
        origin = getattr(R, '_bb_origin', 0)
        self._d2grid = None
        if trajectory:
            if self.timing_mode == 'gradient' or self.data_timing_mode == 'data-gradient':
                self._grid, self._dgrid, self._d2grid = interp_pair(R._z, pos-origin, second=True)
            else:
                self._grid, self._dgrid = interp_pair(R._z, pos-origin)
        else:
            self._grid = (self._rust_interp(R._z,pos-origin) if self._rust_interp is not None
                          else interp(R._z, pos-origin))
            self._dgrid = None
        self._g_traj = trajectory
        self._g_vel = self.vel if trajectory else 0.0
        self._g_built_k = self.k
        self._g_origin = getattr(R, '_bb_origin', 0)
        self._g_k0 = self.k
        self._g_shift = 0
        self._g_dt = self.dt
        self._g_spl = R.spl
        if getattr(self, '_clock_capture', False):
            lo = int(np.floor(pos.min()))-8
            hi = int(np.ceil(pos.max()))+9
            indices = np.clip(np.arange(lo, hi)-self._g_origin, 0, len(R._z)-1)
            self._clock_grid_source = (lo, R._z[indices].copy(), pos)

    TRAJECTORY_TOLERANCE = .02                            # samples; beyond this the first-order residual is not used
    TRAJECTORY_REBUILD_SPACING = 16                       # symbols between rebuilds before falling back to direct windows

    def _trajectory_window(self, k: int):
        """Window for symbol ``k`` from the trajectory grid, or None to use the exact per-window path.

        The grid holds the baseband on the T/2 grid along dt0 + vel*(symbols ahead). Window ``k`` needs the same
        positions shifted by r_m = -vel*(shift + m - span)/2 plus the deviation of the live dt from the predicted
        one, both small, so ``grid + shift * dgrid`` is exact to second order (relative error below 1e-5 for the
        velocities a sampling-clock drift produces).
        """
        R = self.R
        half = R.spl / 2
        # Lean fast path: valid trajectory grid on unchanged buffered data and a deviation within tolerance.
        if self._g_traj and self._g_zlen == len(R._z) and self._g_origin == getattr(R, '_bb_origin', 0):
            j = k - self._g_k0
            if 0 <= j < self.GRID:
                deviation = self.dt - (self._g_dt + self._g_shift * half + self._g_vel * j)
                if -self.TRAJECTORY_TOLERANCE <= deviation <= self.TRAJECTORY_TOLERANCE:
                    i = 2 * j + self._g_shift + self.MARGIN
                    if i >= 0 and i + self.L <= len(self._grid):
                        if self._res_key != (self._g_shift, self._g_vel):
                            self._res_key = (self._g_shift, self._g_vel)
                            self._res = -self._g_vel * (self._g_shift + np.arange(self.L) - self.span) / 2
                        self._tj_k, self._tj_i, self._tj_dev = k, i, deviation
                        return self._grid[i:i + self.L] + (self._res + deviation) * self._dgrid[i:i + self.L]
        rebuild = (self._grid is None or not self._g_traj or self._g_zlen != len(self.R._z)
                   or getattr(self, '_g_origin', 0) != getattr(self.R, '_bb_origin', 0)
                   or not (0 <= k - self._g_k0 < self.GRID))
        deviation = None
        if not rebuild:
            predicted = self._g_dt + self._g_shift * half + self._g_vel * (k - self._g_k0)
            deviation = self.dt - predicted
            rebuild = abs(deviation) > self.TRAJECTORY_TOLERANCE
        if rebuild:
            if self.k - self._g_built_k < self.TRAJECTORY_REBUILD_SPACING and self._g_traj:
                return None                                # timing is moving faster than the grid can follow: stay exact
            self._g_zlen = len(self.R._z)
            self._build_grid(trajectory=True)
            if not self._g_traj or self._dgrid is None:
                return None                                # a diagnostic hook replaced the batch with an exact one
            self._g_k0, self._g_shift, self._g_dt, self._g_spl = self.k, 0, self.dt, self.R.spl
            deviation = 0.0
        i = 2 * (k - self._g_k0) + self._g_shift + self.MARGIN
        if i < 0 or i + self.L > len(self._grid):
            return None
        key = (self._g_shift, self._g_vel)
        if getattr(self, '_res_key', None) != key:
            self._res_key = key
            self._res = -self._g_vel * (self._g_shift + np.arange(self.L) - self.span) / 2
        grid = self._grid[i:i + self.L]
        derivative = self._dgrid[i:i + self.L]
        self._tj_k, self._tj_i, self._tj_dev = k, i, deviation
        return grid + (self._res + deviation) * derivative

    def _interpolate(self, positions):
        if self._rust_interp is not None:
            return self._rust_interp(self.R._z,positions)
        if getattr(self, '_native_interp', False):
            from v34_interp_native import interpolate
            return interpolate(self.R._z, positions)
        return interp(self.R._z, positions)

    def window(self, k: int) -> np.ndarray:
        # Integer half-symbol slips are represented by _g_shift. Any remaining
        # fractional correction must reach interpolation immediately, including
        # while taps are frozen. A stale batch otherwise silently delays the NCO.
        represented_dt = self._g_dt + self._g_shift * self.R.spl / 2
        self._tj_k = -1
        if (self._traj and not getattr(self, '_clock_capture', False)
                and (abs(self.vel) > 1e-12 or (self._g_traj and abs(self._g_vel) > 1e-12))
                and self.R.spl == self._g_spl):
            window = self._trajectory_window(k)
            if window is not None:
                return window
        if abs(self.dt - represented_dt) > 1e-12 or self.R.spl != self._g_spl:
            # Interpolate only this window while the fractional NCO moves.
            # Rebuilding all 128 symbols for each feedback update wastes most
            # of the batch and can delay real-time media processing.
            center = self.R.lag + (self.R.LEAD+k+self.R.delay)*self.R.spl+self.dt
            if self._interp_spl != self.R.spl:
                self._interp_spl = self.R.spl
                self._interp_offs = (np.arange(self.L)-self.span)*self.R.spl/2
            return self._interpolate(center+self._interp_offs
                          -getattr(self.R, '_bb_origin', 0))
        if (self._grid is None or not (0 <= k - self._g_k0 < self.GRID)
                or self._g_zlen != len(self.R._z)
                or getattr(self, '_g_origin', 0) != getattr(self.R, '_bb_origin', 0)):
            self._g_zlen = len(self.R._z)
            self._grid = None
        if self._grid is None:
            self._build_grid()
        i = 2 * (k - self._g_k0) + self._g_shift + self.MARGIN
        if i < 0 or i + self.L > len(self._grid):
            # Timing slips can exhaust the spare samples before GRID symbols
            # have been consumed. Rebase at the current timing position so
            # NumPy never returns a truncated (or negative-indexed) window.
            self._build_grid()
            i = self.MARGIN
        return self._grid[i:i + self.L]

    def timing_snapshot(self) -> dict:
        """Internal trace in recording samples; no expected symbols required."""
        energy = abs(self.h)**2
        centroid = float(np.dot(np.arange(self.L), energy)/max(energy.sum(), 1e-30)-self.span)
        held = self._input_guard is not None and self._input_guard.holding
        result = dict(symbol=self.k, sample_position=float(self.R.lag +
            (self.R.LEAD+self.k+self.R.delay)*self.R.spl+self.dt),
            samples_per_symbol=float(self.R.spl+self.vel), timing_offset_samples=float(self.dt),
            timing_velocity=float(self.vel), tap_centroid=centroid, slips=self.slips,
            carrier_phase=float(self.phase), carrier_radians_per_symbol=float(self.freq),
            tap_adaptation=bool(self.adapt and self.adapt_nlms and not held), timing_feedback=bool(self.adapt and not held),
            timing_mode=self.timing_mode, timing_gradient_error=self._gradient_error,
            timing_gradient_information=self._gradient_information)
        if self._input_guard is not None:
            result['input_power_guard'] = self._input_guard.report()
        return result

    def begin_data_tracking(self, precoded: bool) -> None:
        """Activate data timing after B1, preserving oscillator position/state.

        Precoder decision feedback amplifies erroneous phase targets, so retain
        the conservative carrier gains there. Clear observations from the old
        constellation and gain domain when B1 normalization changes the taps.
        """
        self._gradient_rows=[]
        if self._input_guard is not None:
            self._input_guard.arm()
        # The centered 81-tap input FIR revises its last 40 samples when
        # another packet arrives. Data decisions must use settled filtered
        # samples, so changing packet/poll cadence cannot change their input.
        self._filter_future=40
        # B1 refinement can reshape the taps. The training centroid is then
        # an obsolete timing reference: treating that fit change as clock drift
        # drives the sampling window away from an already qualified boundary.
        # Rebase the measurement without moving the NCO or either oscillator.
        energy=abs(self.h)**2
        self.c0=float(np.dot(np.arange(self.L),energy)/max(float(energy.sum()),1e-30)-self.span)
        self.cf=0.
        if self.data_timing_mode == 'data-gradient':
            self.timing_mode='gradient'
            if not precoded:self.kp,self.ki=max(self.kp,.03),max(self.ki,5e-5)

    def available(self) -> int:
        """Number of symbols for which the whole window lies inside the baseband buffer."""
        R = self.R
        last = (len(R._z) + getattr(R, '_bb_origin', 0) - getattr(self,'_filter_future',0)
                - 8 - R.lag - self._offs[-1] - self.dt) / R.spl - R.LEAD - R.delay
        return int(last)

    def step(self, desired: complex | None = None, decide=None, *, _window=None) -> complex:
        # The diagnostic observer may supply the window it just obtained for
        # this step, before any adaptation. Avoid interpolating it twice.
        w = self.window(self.k) if _window is None else _window
        power = None
        held = False
        if self._input_guard is not None:
            power = float(np.vdot(w, w).real)
            if self._input_guard.armed:
                held = self._input_guard.hold(power)
                if held:
                    # Do not mix derivative observations from either side of a fade.
                    self._gradient_rows = []
            elif self.adapt and self.k % 16 == 0:
                self._input_guard.observe_training(power)
        transport=getattr(self,'_transport_guard',None)
        tainted=transport is not None and transport.tainted(self)
        before=(self.h.copy(),self.phase,self.freq,self.vel) if tainted else None
        hold_carrier=bool(tainted and transport.hold_carrier)       # taps + carrier loop
        hold_timing=bool(tainted and transport.hold_timing)         # sampling-window recovery
        transport_held=hold_carrier or hold_timing
        if hold_timing:self._gradient_rows=[]
        tracking = self.adapt and not held and not hold_carrier
        tracking_timing = self.adapt and not held and not hold_timing
        y = complex(w @ self.h)
        yr = y * complex(math.cos(-self.phase), math.sin(-self.phase))
        d = desired if desired is not None else decide(yr)
        if tracking_timing and self.timing_mode == 'gradient' and self.k % 2 == 0:
            self._gradient_timing(yr,d)
        if tracking and self.adapt_nlms:
            if self._rust_feedback is not None:
                self.phase,self.freq=self._rust_feedback(self.h,w,yr,d,self.mu,self.kp,self.ki,self.phase,self.freq)
            elif self._native_feedback:
                from v34_feedback_native import feedback
                self.phase,self.freq=feedback(self.h,w,yr,d,self.mu,self.kp,self.ki,self.phase,self.freq)
            else:
                e = d - yr
                rot = complex(math.cos(self.phase), math.sin(self.phase))
                self.h += self.mu * np.conj(w) * (e * rot) / ((float(np.vdot(w, w).real) if power is None else power) + 1e-9)
                if abs(d) > 0:
                    perr = math.atan2((yr * d.conjugate()).imag, (yr * d.conjugate()).real)
                    self.freq += self.ki * perr
                    self.phase += self.kp * perr + self.freq
        elif not tracking:
            self.phase += self.freq                          # keep rotating at the learned carrier offset
        if tracking_timing:
            # Timing recovery must keep running even when the NLMS tap update is frozen, otherwise a
            # sampling-clock offset drifts the window off the symbol over a long 4-point run.
            self._n += 1
            if self._n % 4 == 0 and self.timing_mode == 'centroid':
                self._timing()
        # Feed-forward timing: the remote carrier and symbol clock come from one oscillator, so a carrier offset of
        # f_off Hz at carrier fc means the symbol clock is off by f_off / fc; follow it with the sampling window.
        R = self.R
        # The learned sampling-clock velocity continues during B1 qualification,
        # even when feedback and tap adaptation are frozen.
        self.dt += self.vel
        self.dt += self.kff * (self.freq * R.baud / (2 * math.pi)) / R.fc * R.spl
        if before is not None:transport.record(before,self,transport_held)
        self.k += 1
        return yr

    def _gradient_derivative(self) -> complex:
        """d(output)/d(sampling position) of the window just taken, before the carrier rotation.

        When the trajectory grid served this symbol's window, the derivative comes from the grid's derivative and
        curvature (exact to second order in the window's tiny residual shift), with no interpolation at all.
        Otherwise it is the central difference of two interpolations (+-0.05 sample).
        """
        if self._tj_k == self.k and self._d2grid is not None and self._g_traj:
            i = self._tj_i
            window = self._dgrid[i:i + self.L] + (self._res + self._tj_dev) * self._d2grid[i:i + self.L]
            return complex(window @ self.h)
        R = self.R
        center = R.lag + (R.LEAD + self.k + R.delay) * R.spl + self.dt
        offs = (np.arange(self.L) - self.span) * R.spl / 2
        origin = getattr(R, '_bb_origin', 0)
        return complex(((self._interpolate(center + offs + .05 - origin)
                         - self._interpolate(center + offs - .05 - origin)) / .1) @ self.h)

    def _gradient_timing(self, y: complex, desired: complex) -> None:
        """Timing error with constant complex gain/phase projected out.

        Accumulate 16 observations over 32 symbols. A complex gain nuisance
        parameter removes the carrier-phase direction from the timing score;
        clock velocity and the carrier PLL remain independent estimates.
        """
        R=self.R
        derivative=self._gradient_derivative()*np.exp(-1j*self.phase)
        self._gradient_rows.append((derivative,y,desired-y))
        if len(self._gradient_rows)<16:return
        deriv,value,error=np.array(self._gradient_rows).T
        energy=float(np.vdot(value,value).real)+1e-12
        cross=np.vdot(value,deriv)
        information=float(np.vdot(deriv,deriv).real-abs(cross)**2/energy)
        score=(np.vdot(deriv,error)-cross.conjugate()*np.vdot(value,error)/energy).real
        correction=float(np.clip(score/max(information,1e-12),-.1,.1))
        self._gradient_error=correction
        self._gradient_information=information/energy
        self._gradient_rows=[]
        if information>energy*.01:
            self.dt+=.3*correction
            self.vel=float(np.clip(self.vel+.0005*correction,-R.spl*.001,R.spl*.001))

    def _timing(self) -> None:
        """Timing loop from the energy centroid of the taps (in T/2 samples, 0 = centre): the sampling window follows
        the signal (proportional + integral action, so a constant clock offset becomes a constant velocity); large
        excursions are still handled by shifting the taps by whole half-symbols."""
        if self._rust_centroid is not None:
            c=self._rust_centroid(self.h,self.span)
        elif self._native_feedback:
            from v34_feedback_native import centroid
            c=centroid(self.h,self.span)
        else:
            e = np.abs(self.h) ** 2
            c = float(np.sum(np.arange(self.L) * e) / (np.sum(e) + 1e-12)) - self.span
        if self.c0 is None:
            self.c0 = c                                       # the trained taps define the reference position
        c -= self.c0
        self.cf += self.calpha * (c - self.cf)                # the instantaneous centroid is noisy: low-pass it
        c = self.cf
        dead = self.cdead                                     # dead band: stay quiet unless the taps really moved
        c = 0.0 if abs(c) <= dead else c - math.copysign(dead, c)
        half = self.R.spl / 2
        # c > 0 means the taps reach later than the window centre; the window must then move EARLIER (the equalizer
        # taps shift to follow the signal, and the loop pulls the window after them): negative feedback needs this sign
        self.vel -= self.kti * c * half
        self.dt -= self.ktp * c * half * 4                     # velocity advances every symbol
        if c > 1.5:
            self.h = np.concatenate([self.h[1:], [0j]])
            self.dt += half
            self._g_shift += 1
            self.slips += 1
        elif c < -1.5:
            self.h = np.concatenate([[0j], self.h[:-1]])
            self.dt -= half
            self._g_shift -= 1
            self.slips += 1


def decide4(y: complex) -> complex:
    """Nearest point of the 4-point training constellation (point 0 rotated by multiples of 90 degrees)."""
    n = int(round(-math.atan2((y / tx.P0).imag, (y / tx.P0).real) / (math.pi / 2))) % 4
    return tx.P0 * tx.ROT_CW[n]


_C16_Q = [complex(*q) for q in tx.C16]                     # the four quarter-superconstellation points, unrotated
# The 16 points in (quarter, rotation) order, q outer / n inner, so argmin's first-wins tie-break matches a
# plain double loop. Slicing is on the per-symbol path (decide16 is the equalizer's decision callback), so the
# nearest-point search is done with array arithmetic rather than 16 Python-level abs() calls.
_C16_PTS = np.array([_C16_Q[q] * tx.ROT_CW[n] for q in range(4) for n in range(4)], dtype=np.complex128)
_C16_QN = [(q, n) for q in range(4) for n in range(4)]
_C16_MEAN_SQ = float(np.mean(np.abs(_C16_PTS) ** 2))         # the scale decide16 assumes (10.0)


def slice16(y: complex) -> tuple[int, int]:
    """Nearest 16-point constellation point as its (quarter index q, rotation n): the point is C16[q] rotated CW n*90.

    The 16 points are C16[q] for q in 0..3, each rotated by 0/90/180/270 degrees. Every symbol is read as one of
    those (q, n) pairs, so the slicer returns the closest one."""
    k = int(np.argmin(np.abs(_C16_PTS - y)))
    return _C16_QN[k]


def decide16(y: complex) -> complex:
    """Nearest point of the 16-point training constellation (labels 0..3, each rotated by multiples of 90 degrees)."""
    return _C16_PTS[int(np.argmin(np.abs(_C16_PTS - y)))]


class Phase4Receiver:
    """Follow the answerer's phase-4 S-bar + TRN, find where TRN ends and read MP / MP' / E.

    The phase-4 TRN and MP/MP'/E are on the constellation announced by the phase-3 J: 4-point at 2400 bps,
    16-point above it (10.1.3.8 / 10.1.3.9).  Each 16-point symbol carries four scrambled bits
    (I1, I2, Q1, Q2); the rotation bits I1/I2 are absolute in TRN and differentially coded in MP/J'/E
    (the differential coder is initialized by the last TRN symbol), while the quarter-superconstellation
    bits Q1/Q2 are always absolute.

    ``R`` is a trained Phase3Rx, ``k_trn`` the symbol index of the first TRN symbol. ``feed()`` processes
    every symbol available; it stops (and adaptation is frozen) as soon as E has been read, leaving
    ``b1_symbol`` set."""

    def __init__(self, R: Phase3Rx, k_trn: int, role: str = "answer", eq: 'SymbolEq | None' = None,
                 constellation: int = 4):
        if eq is None:
            eq = SymbolEq(R, k_trn)
        else:
            eq.k = k_trn                                    # continue with an already running equalizer
        self.eq = eq
        self.k_trn = k_trn
        self.role = role
        self.c16 = constellation == 16
        self.idx: list[int] = []                            # rotation index n per symbol
        self._idx_np = np.empty(1024, dtype=np.int64)      # mirrored copy of idx, grown geometrically
        self.qidx: list[int] = []                           # quarter-superconstellation index q per symbol (0 for 4-point)
        self.trn_end: int | None = None
        self.sc = tx.Scrambler(role)
        self.mismatch: list[int] = []
        self._end_est: int | None = None
        self.bits: list[int] = []                           # descrambled bits after the TRN end
        self.hist: list[int] = []                           # scrambled bits (descrambler history)
        self.pos = 0
        self.frames: list = []
        self.b1_symbol: int | None = None
        self.e_span = None
        self.mp_spans = []
        # Slicer scale tracking, see _rescale. _last holds the most recent rescaled symbol so the
        # (q, n) extraction and the NLMS decision see the same value.
        self._sq: float | None = None
        self._last: complex = 0j
        self._dec = self._decide16 if self.c16 else decide4

    # Slicer gain tracking. decide16/slice16 are absolute nearest-neighbour decisions against a fixed
    # point set (three radii, |s|^2 = 2 / 10 / 18), so they only read correctly when the equalizer
    # output sits at the nominal scale. eq.step is decision-directed -- d = decide(yr), e = d - yr --
    # which makes a gain error self-fulfilling: under-read the radius and the slicer returns an inner
    # point (q=0), and NLMS then converges the gain onto the inner ring, after which no radius can be
    # read again. The equalizer arrives from phase 3 carrying the phase-3 gain, so any level change
    # at the phase-4 boundary triggers exactly this, and it is why the loopback only ever locked on
    # the transmitter's 7 dB phase-4 over-transmit.
    #
    # Tracking the mean square of y and rescaling to the constellation's own mean makes the slicer
    # scale-invariant, so the decision stops depending on the gain and NLMS is left free to correct
    # the gain to match. The average is slow and spans many symbols on purpose: the radius difference
    # between constellation points is what carries q, so it has to survive the averaging.
    AGC_ALPHA = 0.02

    def _rescale(self, y: complex) -> complex:
        p = float(y.real * y.real + y.imag * y.imag)
        self._sq = p if self._sq is None else (1.0 - self.AGC_ALPHA) * self._sq + self.AGC_ALPHA * p
        if self._sq <= 1e-9:
            self._last = y
            return y
        self._last = y * math.sqrt(_C16_MEAN_SQ / self._sq)
        return self._last

    def _decide16(self, y: complex) -> complex:
        return decide16(self._rescale(y))

    def _push(self, r: int, q: int = 0) -> None:
        """Append one symbol as its rotation index ``r`` and quarter index ``q``, keeping the numpy mirror
        in step (amortized O(1), so the TRN-end rescan does not rebuild the array from the Python list)."""
        k = len(self.idx)
        if k >= self._idx_np.size:
            grown = np.empty(max(1024, 2 * self._idx_np.size), dtype=np.int64)
            grown[:k] = self._idx_np[:k]
            self._idx_np = grown
        self._idx_np[k] = r
        self.idx.append(r)
        self.qidx.append(q)

    def _expected(self) -> tuple[int, int]:
        """The rotation (I) and quarter (Q) bit values the known TRN scraper produces next."""
        i1 = self.sc.step(1)
        i2 = self.sc.step(1)
        if not self.c16:
            return 2 * i2 + i1, 0
        q1 = self.sc.step(1)
        q2 = self.sc.step(1)
        return 2 * i2 + i1, 2 * q2 + q1

    def feed(self, parse_mp) -> None:
        self._feed(parse_mp)
        observer=getattr(self,'observer',None)
        if observer is not None:observer.after_feed(self)

    def _feed(self, parse_mp) -> None:
        next_scan = 0
        while self.b1_symbol is None and self.eq.k < self.eq.available():
            if self.trn_end is None and self._end_est is not None and len(self.idx) >= next_scan:
                next_scan = len(self.idx) + 8
                found = decode_after_trn(self._idx_np[:len(self.idx)], self._end_est, self.role,
                                         c16=self.c16, qidx=self.qidx)
                if found is not None:
                    self.trn_end = found[0]
                    self._start_bits(found[0])
                    continue
                self._rearm_end_estimate()
            observer=getattr(self,'observer',None)
            if observer is not None:observer.before_symbol(self)
            y = self.eq.step(decide=self._dec)
            if self.c16:
                q, r = slice16(self._last)
            else:
                q, r = 0, int(round(-math.atan2((y / tx.P0).imag, (y / tx.P0).real) / (math.pi / 2))) % 4
            self._push(r, q)
            if self.trn_end is None:
                exp_i, exp_q = self._expected()
                self.mismatch.append(0 if (r == exp_i and q == exp_q) else 1)
                if self._end_est is None and len(self.mismatch) >= 16 and sum(self.mismatch[-16:]) >= 8:
                    # Only a real run of agreements may set this. The old fallback took
                    # len(idx) - 16 whenever trn_end_from_mismatch found no run of min_run
                    # agreements -- which is exactly the state during the acquisition transient at
                    # the start of TRN, while the equalizer is still converging. That latched a
                    # garbage estimate which was never revised, so decode_after_trn went on
                    # searching +/-24 of the wrong symbol for the rest of the call and trn_end
                    # never appeared. TRN runs at least 512T (11.3.1.1.2), so a conforming peer
                    # always leaves a real run to find; the fallback is kept only for a peer that
                    # breaks that, and then only once that many symbols have been seen.
                    est = trn_end_from_mismatch(self.mismatch)
                    self._end_est = est if est is not None else (
                        len(self.idx) - 16 if len(self.idx) >= 512 else None)
                continue
            self._add_symbol(r, q)
            self._parse(parse_mp)

    REARM_AFTER = 300                                          # symbols past the estimate with no MP frame parsed

    def _rearm_end_estimate(self) -> None:
        """Recover from a TRN-end estimate that never produces an MP frame.

        A burst of decision errors inside the TRN splits the run of agreements with the known sequence, and the
        longest-run rule can then latch an estimate in the middle of the TRN; the search window around it never
        finds an MP frame. When ``REARM_AFTER`` symbols have passed without one, re-estimate from the last long
        run of agreements; it follows the TRN's tail and closes at the true end. Does nothing when the first
        estimate was right (an MP frame is found well before ``REARM_AFTER``).
        """
        if self._end_est is None or len(self.idx) - self._end_est <= self.REARM_AFTER:
            return
        est = trn_end_from_mismatch(self.mismatch, last=True)
        if est is not None and est != self._end_est:
            self._end_est = est

    def _start_bits(self, n: int) -> None:
        hist, bits = [], []
        for k in range(len(self.idx)):
            i_k = self.idx[k] if k < n else (self.idx[k] - self.idx[k - 1]) % 4
            hist += [i_k & 1, i_k >> 1]                     # rotation bits (I1, I2)
            if self.c16:
                q = self.qidx[k]
                hist += [q & 1, q >> 1]                     # quarter bits (Q1, Q2), absolute
        d = descramble(hist, self.role)
        self.hist = hist[-23:]
        self.bits = d[4 * n:] if self.c16 else d[2 * n:]
        self.pos = 0
        self._parse(None)

    def _add_symbol(self, r: int, q: int = 0) -> None:
        i_k = (r - self.idx[-2]) % 4
        symbits = [i_k & 1, i_k >> 1]
        if self.c16:
            symbits += [q & 1, q >> 1]
        tap = 5 if self.role == "answer" else 18
        for b in symbits:
            h = self.hist
            self.bits.append(b ^ (h[-tap] if len(h) >= tap else 0) ^ (h[-23] if len(h) >= 23 else 0))
            self.hist = (h + [b])[-23:]

    def _mp_acknowledged(self) -> bool:
        return any(isinstance(f, dict) and f.get("ack") for f in self.frames)

    def _parse(self, parse_mp) -> None:
        import v34_mp
        b = self.bits
        while True:
            avail = len(b) - self.pos
            if avail < 18:
                return
            seg = b[self.pos:self.pos + 256]                # bounded window: MP frames are <=188 bits
            if seg[:17] == [1] * 17 and seg[17] == 1:
                if avail < 20:
                    return
                if seg[:20] == [1] * 20 and self._mp_acknowledged():
                    # E: B1 starts with the next symbol. E only follows an acknowledged MP (MP', 11.4.1); without that guard
                    # any 20 ones in the descrambled stream is an E, and a steady tone A decodes to such a run
                    # (lab/V34_LCP1_REPLAY.md: a false E inside tone A started a bogus B1 on a real call).
                    self.frames.append("E")
                    start_bit = self.pos
                    self.pos += 20
                    bits_per_symbol = 4 if self.c16 else 2
                    # Parsing may be retrospective: eq.k is the processing cursor,
                    # not the symbol containing the final E bit.
                    self.b1_symbol = self.k_trn + self.trn_end + (self.pos + bits_per_symbol - 1)//bits_per_symbol
                    self.e_span = dict(start_bit=start_bit,end_bit=self.pos,
                                       b1_symbol=self.b1_symbol,processing_symbol=self.eq.k)
                    return
                self.pos += 1
                continue
            mp = v34_mp.parse_mp(seg)
            if mp is None:
                if avail < 188 + 4 and seg[:17] == [1] * 17 and seg[17] == 0:
                    return                                   # frame not complete yet
                self.pos += 1                                # resync
                continue
            self.frames.append(mp)
            start_bit = self.pos
            self.pos += 88 if mp["type"] == 0 else 188
            self.mp_spans.append(dict(start_bit=start_bit,end_bit=self.pos,fields=dict(mp)))


class Phase3Receiver:
    """Follow the answerer's TRN (known) after PP with the adaptive equalizer and read J."""

    def __init__(self, R: Phase3Rx, k_trn: int = 144 + 288, role: str = "answer"):
        self.eq = SymbolEq(R, k_trn)
        self.k_trn, self.role = k_trn, role
        self.idx: list[int] = []
        self.sc = tx.Scrambler(role)
        self.mismatch: list[int] = []
        self.j: tuple[int, int, str, int] | None = None
        self._end_est: int | None = None

    def feed(self) -> None:
        while self.j is None and self.eq.k < self.eq.available():
            y = self.eq.step(decide=decide4)
            r = int(round(-math.atan2((y / tx.P0).imag, (y / tx.P0).real) / (math.pi / 2))) % 4
            self.idx.append(r)
            i1, i2 = self.sc.step(1), self.sc.step(1)
            self.mismatch.append(0 if r == 2 * i2 + i1 else 1)
            n = len(self.idx)
            if self._end_est is None and n >= 16 and sum(self.mismatch[-16:]) >= 8 and (n - 16) > 300:
                # The TRN is a known sequence, so pin the transition to the exact symbol rather than
                # leaving it at the granularity of the 16-symbol window that first noticed it.
                self._end_est = trn_end_from_mismatch(self.mismatch) or (n - 16)
            if self._end_est is not None and n % 8 == 0:
                got = decode_j(np.array(self.idx), self._end_est, self.role, mismatch=self.mismatch)
                if got is not None and got[1] >= 16:
                    self.j = got


def find_jprime(idx, n_trn: int, role: str, j_pattern: str = "J_4PT"):
    """After TRN (length ``n_trn``) and a run of J, find J': returns the symbol index where J' starts, or None.
    J' is one 16-bit pattern at a J-aligned position of the descrambled bits (the J pattern has period 16 bits).

    ``j_pattern`` is the name of the J pattern that ran before it, which is the one phase 3 negotiated -
    above 2400 bps that is ``J_16PT``, and requiring ``J_4PT`` there would miss every J'.

    The descrambled bits are needed from ``2 * n_trn`` to the end of the stream, so this cannot be done
    with a bounded per-candidate window like ``_descramble_at``: the scan is inherently over the whole
    remaining stream. It is therefore built once with the vectorized descrambler rather than a per-bit
    Python loop. For a stream that grows while it is being searched - which is what a modem waiting for
    J' actually has - use ``JPrimeScanner`` below, which keeps that rescanning linear.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size <= n_trn + 8:
        return None
    d = _full_descramble(idx, role, n_trn)[2 * n_trn:]
    pat16 = np.array(tx.J_PRIME, dtype=np.int64)
    run16 = np.array(getattr(tx, j_pattern), dtype=np.int64)
    hit = np.flatnonzero(d[:d.size - 15] == pat16[0])
    for p in hit.tolist():
        if d[p:p + 16].tolist() == pat16.tolist() and p >= 16 and p % 16 == 0 \
                and d[p - 16:p].tolist() == run16.tolist():
            return n_trn + p // 2
    return None


def jprime_search(bits: np.ndarray, n_trn: int, start_bit: int = 16, j_pattern: str = "J_4PT"):
    """Find J' in the *already-descrambled* stream, scanning only from ``start_bit`` onwards.

    ``bits`` is what ``_full_descramble(idx, role, n_trn)`` returns (the whole descrambled stream, not
    sliced). Positions are counted *relative to the end of TRN* - the first bit after the TRN is
    position 0 - so J' can only start on a 16-bit boundary of that grid, exactly as ``find_jprime``
    requires. ``start_bit`` is where to resume, so a caller rescanning a growing stream can pass the
    cursor from last time and inspect only new bits, keeping a repeated rescan linear rather than
    quadratic. ``j_pattern`` is the J pattern that ran before J' (see ``find_jprime``).
    Returns ``(symbol_index_or_None, next_start_bit)``; the cursor is always advanced past
    every position examined, so a position is never examined twice.
    """
    b = np.asarray(bits, dtype=np.int64)
    pat16 = np.array(tx.J_PRIME, dtype=np.int64)
    run16 = np.array(getattr(tx, j_pattern), dtype=np.int64)
    base = 2 * n_trn                                  # first bit after TRN, in absolute coordinates
    tail = b[base:]
    # J' immediately follows a J unit (11.4.1.1.1: the caller stops sending J, sends one J', then
    # TRN), so a candidate is J_PRIME preceded by a J unit.
    #
    # Positions are stepped by 2 bits -- one symbol -- rather than by a whole 16-bit J unit. The
    # absolute bit origin here is 2*n_trn, and n_trn is only ever an *estimate* of where the caller's
    # TRN ended, so the caller's J/J' run can sit a few symbols off this grid. Measured across the
    # corpus the miss was 12 bits (3com-messagemodem) and 2 bits (agere1648c): both carry a textbook
    # J-PRIME straight after a J unit, and a 16-bit step skipped over both, which is why six clips
    # never reported a J'. A random 32-bit match has probability 2**-32 per position, so scanning
    # every symbol rather than every eighth costs a factor of 8 and costs no false positives.
    start = max(start_bit, 16)
    start -= start % 2
    found = None
    p = start
    while p + 16 <= tail.size:
        if np.array_equal(tail[p:p + 16], pat16) and np.array_equal(tail[p - 16:p], run16):
            found = n_trn + p // 2
            break
        p += 2
    return found, max(p, start)


class JPrimeScanner:
    """Incremental J' search over a growing 4-point phase-3 symbol stream.

    ``find_jprime`` descrambles and rescans the whole symbol history on every call. That is fine once,
    but the answerer polls for J' every 8 symbols while it waits, so a J' that has not arrived yet turns
    the wait into an O(n^2) CPU sink - tens of thousands of symbols rebuilt per poll, on a live call.
    This seeds itself once from the stream as it stands when phase 4 starts (the J run is normally
    already there, so almost all the work happens in that one pass) and thereafter extends only the new
    symbols. The descrambler is memoryless - ``_descramble_np`` reads the *input* taps, not its own
    output - so extending it costs just the new bits plus their 23-bit lookbehind. Combined with
    ``jprime_search``'s 16-bit cursor the total cost is linear in the number of symbols seen.

    ``update`` returns the J' symbol index once it appears and ``None`` until then; it keeps returning
    that same index afterwards so a caller need not remember it found it.

    ``idx`` may be a plain list - which is what the answerer's phase-3 receiver keeps - so only the newly
    appended tail is read, and the raw / descrambled buffers double rather than being reallocated per
    poll. Both of those matter as much as the descrambler itself: converting the list or concatenating
    the buffers once per poll would put the O(n) right back and make the wait quadratic again.
    """

    def __init__(self, idx, role: str, n_trn: int, j_pattern: str = "J_4PT"):
        self.role = role
        self.tap = descramble_tap(role)
        self.n_trn = int(n_trn)
        self.j_pattern = j_pattern
        self.found: int | None = None
        self._cursor = 16
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size < self.n_trn:
            raise ValueError(f"JPrimeScanner must start past the TRN end: {idx.size} symbols < n_trn={self.n_trn}")
        # Same split _full_descramble uses: absolute up to the TRN end, differentially coded after it.
        raw = np.concatenate((_interleave(idx[:self.n_trn]), _interleave(_differential(idx)[self.n_trn:])))
        dec = _descramble_np(raw, self.tap)
        self._nsym = int(idx.size)
        self._nraw = self._ndec = raw.size
        self._cap = max(1024, 2 * raw.size)
        self._raw = np.zeros(self._cap, dtype=np.int64)
        self._dec = np.zeros(self._cap, dtype=np.int64)
        self._raw[:self._nraw] = raw
        self._dec[:self._ndec] = dec
        self.found, self._cursor = self._search()

    def _search(self):
        return jprime_search(self._dec[:self._ndec], self.n_trn, self._cursor, self.j_pattern)

    def _reserve(self, extra: int) -> None:
        if self._nraw + extra <= self._cap:
            return
        cap = max(self._nraw + extra, 2 * self._cap)
        self._raw = np.concatenate((self._raw[:self._nraw], np.zeros(cap - self._nraw, dtype=np.int64)))
        self._dec = np.concatenate((self._dec[:self._ndec], np.zeros(cap - self._ndec, dtype=np.int64)))
        self._cap = cap

    def _raw_at(self, start: int, count: int) -> np.ndarray:
        """Raw (pre-descramble) bits at absolute positions, zero-filled outside the known range."""
        out = np.zeros(count, dtype=np.int64)
        lo, hi = max(start, 0), min(start + count, self._nraw)
        if hi > lo:
            out[lo - start:hi - start] = self._raw[lo:hi]
        return out

    def _extend(self, raw_new: np.ndarray) -> None:
        """Append raw bits and their descrambled values, matching ``_descramble_np`` exactly."""
        m = raw_new.size
        lo = self._nraw
        self._reserve(m)
        # Append before gathering the taps: for the last `tap` positions of the new run the tap reads
        # raw bits that are themselves new, so they have to be in place first.
        self._raw[lo:lo + m] = raw_new
        self._nraw = lo + m
        d = raw_new.copy()
        if self.tap:
            d ^= self._raw_at(lo - self.tap, m)
        d ^= self._raw_at(lo - 23, m)
        self._dec[lo:lo + m] = d
        self._ndec = lo + m

    def update(self, idx) -> int | None:
        """Consume any symbols appended to ``idx`` and return the J' symbol index, or None while it is
        still absent."""
        if self.found is not None:
            return self.found
        n = len(idx)
        if n <= self._nsym:
            return None
        k0 = self._nsym
        # Everything from the TRN end on is differentially coded, and the scanner starts after that, so
        # each new symbol's value is its difference from *its own* predecessor - the last symbol already
        # consumed for the first one, and the preceding element of the chunk for the rest.
        new = np.asarray(idx[k0:], dtype=np.int64)
        prev = np.empty_like(new)
        prev[0] = idx[k0 - 1]
        prev[1:] = new[:-1]
        self._extend(_interleave((new - prev) % 4))
        self._nsym = n
        self.found, self._cursor = self._search()
        return self.found
