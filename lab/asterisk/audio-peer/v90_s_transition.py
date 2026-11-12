"""Incremental detector for analogue S-to-S waveform transitions in V.90."""

from __future__ import annotations

import numpy as np

import v34_rx


FS = 8000


class V90STransitionDetector:
    """Find successive phase inversions in the V.90 S/S-bar pattern.

    The detector wraps El-Modem's measured S-flip correlator, but owns the
    incremental PCM window, absolute sample coordinates, candidate retention,
    and duplicate suppression. Input is ordered 8 kHz linear PCM. RTP gaps
    must be handled by the caller as a reset/failure; this class never inserts
    samples or treats a discontinuous stream as contiguous.
    """

    def __init__(self, baud: float, carrier: float, *,
                 threshold: float = v34_rx.S_FLIP_THRESHOLD,
                 max_window_seconds: float = 0.25):
        if baud <= 0 or carrier <= 0 or not 0 < threshold < 1:
            raise ValueError("invalid V.90 S-transition detector parameters")
        if max_window_seconds < 0.1:
            raise ValueError("S-transition search window must be at least 100 ms")
        self.baud = float(baud)
        self.carrier = float(carrier)
        self.threshold = float(threshold)
        self.max_window = int(max_window_seconds * FS)
        self._pcm = np.empty(0, dtype=np.float64)
        self._origin = 0
        self._next_search = 0
        self._expected_next: int | None = None
        self._last_event: int | None = None
        self._pending_s: int | None = None
        self.events: list[tuple[str, int]] = []
        self.transitions: list[int] = []

    def feed(self, samples, *, sample_at: int | None = None) -> list[tuple[str, int]]:
        """Consume PCM; return newly detected (``S`` or ``flip``, sample) events."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("S-transition PCM must be finite")
        if self._expected_next is None:
            start = 0 if sample_at is None else int(sample_at)
            self._origin = self._next_search = start
        else:
            start = self._expected_next if sample_at is None else int(sample_at)
        if start != self._origin + len(self._pcm):
            raise ValueError("S-transition PCM discontinuity; reset detector after a gap")
        if not len(x):
            return []
        self._pcm = np.concatenate((self._pcm, x))
        self._expected_next = start + len(x)
        found: list[tuple[str, int]] = []

        while True:
            local = self._next_search - self._origin
            if local < 0:
                local = 0
                self._next_search = self._origin
            window = self._pcm[local:]
            result = v34_rx.detect_s_flip(
                window, self.carrier, self.baud, threshold=self.threshold)
            if result is not None and result[0] == "S":
                absolute = self._next_search + int(result[1])
                if self._pending_s is None:
                    self._pending_s = absolute
                    self.events.append(("S", absolute))
                    found.append(("S", absolute))
                break

            if result is not None and result[0] == "flip":
                absolute = self._next_search + int(result[1])
                # A whole S/S-bar pair can arrive in one RTP packet. Recover
                # its S onset from the prefix so the caller and answerer see
                # both protocol events in order, even without an intermediate
                # packet boundary.
                if self._pending_s is None:
                    prefix = window[:int(result[1])]
                    onset = v34_rx.detect_s_flip(
                        prefix, self.carrier, self.baud,
                        threshold=self.threshold)
                    if onset is not None and onset[0] == "S":
                        s_at = self._next_search + int(onset[1])
                        self._pending_s = s_at
                        self.events.append(("S", s_at))
                        found.append(("S", s_at))
                if self._last_event is None or absolute > self._last_event:
                    self._last_event = absolute
                    self.transitions.append(absolute)
                    self.events.append(("flip", absolute))
                    found.append(("flip", absolute))
                    # Resume at the transition so the next S can be found.
                    self._next_search = absolute
                    self._pending_s = None
                    continue
                self._next_search = absolute + 1
                continue

            # A found S without its inversion is an unfinished candidate; keep
            # its complete context until more media arrives. Otherwise discard
            # old samples while preserving enough history to find a preamble
            # that began at the trailing edge of this packet.
            if (result is None or result[0] == "S") and len(window) > self.max_window:
                keep = self.max_window
                drop = len(self._pcm) - keep
                self._pcm = self._pcm[drop:]
                self._origin += drop
                self._next_search = self._origin
                continue
            break

        return found

    def reset(self, *, sample_at: int = 0) -> None:
        """Clear detector state after a deliberate media restart or gap policy."""
        if sample_at < 0:
            raise ValueError("sample_at must be nonnegative")
        self._pcm = np.empty(0, dtype=np.float64)
        self._origin = int(sample_at)
        self._next_search = int(sample_at)
        self._expected_next = int(sample_at)
        self._last_event = None
        self._pending_s = None
        self.events.clear()
        self.transitions.clear()
