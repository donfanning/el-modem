"""Opt-in (``V34_FREEZE_GUARD=1``) guard against a frozen carrier stream in the V.34 data phase.

A gateway or jitter buffer that loses its input can keep sending the last frame, so the receiver is handed the same 20 ms of
audio over and over (Astound Queens, 2026-10-08: 359 identical packets = 7.2 s at 33600, 1161 = 23 s at 9600, both starting
about 3.3 s into the data phase; lab/V34_QUEENS_FROZEN_RTP_FINDING.md). Data-phase modem audio does not repeat a whole frame
and a noisy line never delivers 20 ms of exact zeros, so a block identical to its predecessor, or a block of exact zeros, is
treated as lost and replaced by zeros, and **adaptation is held** (the engine sets ``eq.adapt = False``) for the lost stretch
plus a margin of ``HOLD_MARGIN`` samples, so the equaliser, carrier loop and timing loop coast instead of learning from it.
Zeros alone are not enough: the NLMS step divides by the window energy, so a window that is nearly but not exactly empty gives
a huge step. On the Queens recording the equaliser taps went from 0.005 to ~800 at a 60 ms zero dropout (capture 17.97 s) and
never recovered.

Not a pure-tone detector: apply it only after the data phase has started (a steady tone legitimately repeats).
"""
from __future__ import annotations

import numpy as np

MIN_BLOCK = 80          # shorter blocks are too small to compare
MIN_RMS = 50.0          # a repeated block must be at least this loud to count as a frozen frame
HOLD_MARGIN = 640       # samples (80 ms): keep adaptation held while windows still straddle the end of a lost stretch


LINE_BINS = 4           # a block whose energy sits in this many FFT bins is a steady tone, not a frozen data frame
LINE_FRACTION = 0.95


def is_line_spectrum(x) -> bool:
    """True for a steady tone such as Tone A (2400 + 1800 Hz) or Tone B (1200 Hz).

    At 8 kHz a 20 ms block holds whole cycles of those tones, so consecutive blocks are bit-identical and the repeat rule alone
    would call them frozen. Data-phase audio is broadband; a frozen *data* frame never concentrates its energy in a few bins.
    """
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=float))) ** 2
    total = float(p.sum())
    return total > 0 and float(np.sort(p)[-LINE_BINS:].sum()) >= LINE_FRACTION * total


class FreezeGuard:
    def __init__(self):
        self._prev = None
        self._run = 0
        self.frozen_blocks = 0
        self.frozen_samples = 0
        self.runs = []                       # (receive sample index where a run started, blocks in the run)
        self.longest_run = 0
        self.pending_event = None            # ("start", t) / ("end", blocks), read and cleared by the engine
        self._hold_left = 0
        self.zero_blocks = 0
        self.tone_blocks = 0                 # repeated blocks passed through because they are a steady tone
        self.run_samples = 0                 # samples in the current lost stretch (0 when the stream is not lost)

    @property
    def holding(self) -> bool:
        """True while the engine should keep adaptation held (inside a lost stretch or in the margin after it)."""
        return self._run > 0 or self._hold_left > 0

    def filter(self, x, t: int):
        """``x``: the next received block (float array); ``t``: receive sample index of its first sample."""
        x = np.asarray(x)
        big = len(x) >= MIN_BLOCK
        zero = big and not x.any()
        repeat = (self._prev is not None and len(x) == len(self._prev) and big and not zero
                  and float(np.sqrt(np.mean(x * x))) > MIN_RMS and np.array_equal(x, self._prev))
        self._prev = x.copy()                # always compare against the audio as received, never the zero-filled copy
        if repeat and is_line_spectrum(x):   # Tone A / Tone B: a retrain in progress, not a frozen stream
            repeat = False
            self.tone_blocks += 1
        if repeat or zero:
            if self._run == 0:
                self.runs.append([int(t), 0])
                self.pending_event = ("start", int(t))
            self._run += 1
            self.runs[-1][1] = self._run
            self._hold_left = HOLD_MARGIN
            self.run_samples += len(x)
            if zero:
                self.zero_blocks += 1
            else:
                self.frozen_blocks += 1
            self.frozen_samples += len(x)
            self.longest_run = max(self.longest_run, self._run)
            return np.zeros(len(x))
        if self._run:
            self.pending_event = ("end", self._run)
            self._run = 0
            self.run_samples = 0
        elif self._hold_left:
            self._hold_left = max(0, self._hold_left - len(x))
        return x

    def report(self) -> dict:
        return dict(enabled=True, frozen_blocks=self.frozen_blocks, zero_blocks=self.zero_blocks, frozen_samples=self.frozen_samples,
                    longest_run_blocks=self.longest_run, runs=[tuple(r) for r in self.runs[-16:]],
                    rule='block identical to the previous one (rms > %g) or all zeros, data phase only: replaced by zeros, adaptation held' % MIN_RMS)
