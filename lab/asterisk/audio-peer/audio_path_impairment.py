"""Deterministic, bounded sample impairments for software audio-loop tests.

This models digital test conditions before PCMU quantization. It is not an
analog line model and its settings must not be reported as dBm0 or a PSTN
qualification result.
"""
from __future__ import annotations

import math
import random


class AudioPathImpairment:
    """Apply gain, DC offset, additive white noise and hard clipping."""

    def __init__(self, *, gain_db: float = 0.0, dc_offset: float = 0.0,
                 noise_rms: float = 0.0, clip_level: int = 32767,
                 seed: int = 0):
        values = (gain_db, dc_offset, noise_rms)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("impairment values must be finite")
        if abs(gain_db) > 18 or abs(dc_offset) > 4096 or not 0 <= noise_rms <= 8192:
            raise ValueError("impairment outside declared digital test bounds")
        if not isinstance(clip_level, int) or not 256 <= clip_level <= 32767:
            raise ValueError("clip_level must be an integer in 256..32767")
        self.gain = 10.0 ** (gain_db / 20.0)
        self.dc_offset = float(dc_offset)
        self.noise_rms = float(noise_rms)
        self.clip_level = clip_level
        self.rng = random.Random(seed)
        self.clipped_samples = 0
        self.samples = 0

    def process(self, samples):
        out = []
        for sample in samples:
            value = sample * self.gain + self.dc_offset
            if self.noise_rms:
                value += self.rng.gauss(0.0, self.noise_rms)
            if value > self.clip_level:
                value = self.clip_level
                self.clipped_samples += 1
            elif value < -self.clip_level:
                value = -self.clip_level
                self.clipped_samples += 1
            out.append(round(value))
        self.samples += len(out)
        return out

    def report(self):
        return {
            "gain": self.gain,
            "dc_offset_samples": self.dc_offset,
            "noise_rms_samples": self.noise_rms,
            "clip_level_samples": self.clip_level,
            "clipped_samples": self.clipped_samples,
            "samples": self.samples,
        }


__all__ = ["AudioPathImpairment"]
