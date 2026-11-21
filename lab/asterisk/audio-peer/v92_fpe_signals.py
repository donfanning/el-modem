"""Streaming detectors for V.92 fast-parameter-exchange R signals."""

from __future__ import annotations

import numpy as np

import v92_fast_parameter_exchange as fpe
import v92_pcm_upstream as pu


class RfSequenceDetector:
    """Find the downstream Rf/Rf-bar codeword sequence at any sample phase.

    The exact pattern is long (384T plus the 24T reversal), so retaining only
    one pattern-length tail is enough to match it across arbitrary feed
    boundaries. This detects the generated symbol sequence after codeword
    demodulation; it does not perform a noisy-line symbol decision.
    """

    def __init__(self, downstream_profile: dict, law: str):
        _, self._rf, self._rbar = fpe.rf_signals(downstream_profile, law)
        self._pattern = self._rf * fpe.R_FRAMES + self._rbar * fpe.RBAR_FRAMES
        self._samples = bytearray()
        self._base = 0
        self.detected = False

    def feed(self, codewords) -> list[tuple[str, int]]:
        if self.detected:
            return []
        x = bytes(codewords)
        self._samples.extend(x)
        at = self._samples.find(self._pattern)
        if at >= 0:
            start = self._base + at
            self.detected = True
            return [("rf", start), ("rf-bar", start + fpe.R_FRAMES * len(self._rf))]
        keep = len(self._pattern) - 1
        if len(self._samples) > keep:
            drop = len(self._samples) - keep
            del self._samples[:drop]
            self._base += drop
        return []


class RmSequenceDetector:
    """Detect the analogue modem's RM/RM′ pair at any input sample phase.

    The caller supplies linear PCM after G.711 decoding and the upstream
    profile from the currently active CPd. A candidate must decode as eight
    complete RM data frames followed by two RM′ frames with a fresh data-mode
    receiver state. This is intentionally a narrow symbol-sequence detector;
    the session still owns timeout, frame-clock, and state-transition policy.
    """

    RM_FRAMES = 32
    RMP_FRAMES = 2
    WINDOW = (RM_FRAMES + RMP_FRAMES) * pu.INTERVALS
    def __init__(self, profile: pu.PcmUpstreamProfile, *, sample_phase: int | None = 0):
        if (sample_phase is not None and
                (isinstance(sample_phase, bool) or not isinstance(sample_phase, int)
                 or not 0 <= sample_phase < pu.INTERVALS)):
            raise ValueError("sample_phase must be None or an integer in 0..11")
        self.profile = profile
        self.sample_phase = sample_phase
        self._step = 1 if sample_phase is None else pu.INTERVALS
        self._samples = np.empty(0, dtype=float)
        self._base = 0
        self._next = 0 if sample_phase is None else sample_phase
        self._rm = pu.rm_intervals(profile.moduli)
        self._rmp = pu.rm_intervals(profile.moduli, prime=True)
        self._signed = pu.PcmUpstreamReceiver(profile)._signed
        self.detected = False

    def _rm_frame_candidates(self, starts: np.ndarray) -> np.ndarray:
        """Vector-filter starts whose first decoded interval frame is exactly RM.

        A full receiver is still used to confirm every candidate. This cheap
        first-frame test avoids constructing and running a Python receiver for
        nearly every sample while searching an unknown frame phase.
        """
        if not len(starts):
            return starts
        windows = np.lib.stride_tricks.sliding_window_view(
            self._samples, pu.INTERVALS)[starts]
        possible = np.ones(len(starts), dtype=bool)
        for interval, (modulus, set_index) in enumerate(
                zip(self.profile.moduli, self.profile.interval_set)):
            signed = self._signed[set_index]
            values = windows[:, interval]
            right = np.searchsorted(signed, values, side="left")
            lo = np.maximum(right - 1, 0)
            hi = np.minimum(right, len(signed) - 1)
            choose_lo = ((right == len(signed))
                         | ((right > 0)
                            & (values - signed[lo] <= signed[hi] - values)))
            eta = np.where(choose_lo, lo, hi) - len(signed) // 2
            if interval % pu.TRELLIS_FRAME < 3:
                decoded = eta % modulus
            else:
                parity = eta & 1
                decoded = ((eta - parity) // 2) % modulus
            possible &= decoded == self._rm[interval]
        return starts[possible]

    def _matches(self, start: int) -> bool:
        rx = pu.PcmUpstreamReceiver(self.profile)
        for frame in range(self.RM_FRAMES + self.RMP_FRAMES):
            begin = start + frame * pu.INTERVALS
            decoded = rx.decode_intervals(self._samples[begin:begin + pu.INTERVALS])
            expected = self._rm if frame < self.RM_FRAMES else self._rmp
            if decoded != expected:
                return False
        return True

    def feed(self, samples) -> list[tuple[str, int]]:
        if self.detected:
            return []
        x = np.asarray(samples, dtype=float).reshape(-1)
        if len(x):
            self._samples = np.concatenate((self._samples, x))
        events = []
        last = len(self._samples) - self.WINDOW
        if self._step == 1 and self._next <= last:
            starts = np.arange(self._next, last + 1, dtype=np.intp)
            candidates = self._rm_frame_candidates(starts)
            for candidate in candidates:
                candidate = int(candidate)
                if self._matches(candidate):
                    start = self._base + candidate
                    events.extend((
                        ("rm", start),
                        ("rm-prime", start + self.RM_FRAMES * pu.INTERVALS),
                    ))
                    self.detected = True
                    self._next = candidate
                    break
                self._next = candidate + 1
            else:
                self._next = last + 1
        else:
            while self._next <= last:
                if self._matches(self._next):
                    start = self._base + self._next
                    events.extend((
                        ("rm", start),
                        ("rm-prime", start + self.RM_FRAMES * pu.INTERVALS),
                    ))
                    self.detected = True
                    break
                self._next += self._step
        if self.detected:
            return events
        if self._next > last:
            if self._next:
                self._samples = self._samples[self._next:].copy()
                self._base += self._next
                self._next = 0
        return events
