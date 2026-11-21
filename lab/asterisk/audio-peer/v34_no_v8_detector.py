"""Bounded detector for an opt-in direct V.34 caller that skips V.8.

This is a compatibility extension, not the standard V.34 Phase 1 path. It
requires a CRC-valid caller INFO0c followed by sustained 1200 Hz Tone B.
"""

from __future__ import annotations

import math
import time

import numpy as np

from v34_info import BAUD, FRAME_LENGTH, FS, InfoDemodulator


class NoV8V34Detector:
    BUFFER_SAMPLES = int(1.5 * FS)
    SCAN_INTERVAL = FS // 50       # recheck at most every 20 ms
    TONE_OFFSET = 160               # skip the INFO0c modulator's two tail symbols
    TONE_SAMPLES = 320              # 40 ms of Tone B
    DETECTION_WINDOW_SAMPLES = (
        math.ceil(FRAME_LENGTH["info0"] * FS / BAUD)
        + TONE_OFFSET + TONE_SAMPLES)
    SCAN_OVERLAP_SAMPLES = (
        math.ceil(DETECTION_WINDOW_SAMPLES / SCAN_INTERVAL) * SCAN_INTERVAL)
    TONE_RATIO_MIN = 0.80
    TONE_RMS_MIN = 60.0

    def __init__(self):
        self.samples = np.zeros(0, dtype=float)
        self.base = 0
        self.scan_samples = np.zeros(0, dtype=float)
        self.scan_base = 0
        self.last_scan = 0
        self.frames_checked = 0
        self.best_tone_ratio = 0.0
        self.best_tone_rms = 0.0
        self.last_match: dict | None = None
        self.feed_durations_ns: list[int] = []
        self._scored_frames: set[int] = set()
        self._demod = InfoDemodulator("call", phases=8)

    @staticmethod
    def _tone_metrics(x: np.ndarray, absolute_at: int) -> tuple[float, float]:
        n = len(x)
        if not n:
            return 0.0, 0.0
        t = absolute_at + np.arange(n)
        coefficient = np.dot(x, np.exp(-2j * math.pi * 1200.0 * t / FS))
        tone_rms = math.sqrt(2.0) * abs(coefficient) / n
        total_rms = math.sqrt(float(np.mean(x * x)))
        return (0.0 if total_rms == 0 else tone_rms * tone_rms / (total_rms * total_rms),
                tone_rms)

    def feed(self, samples) -> bool:
        started_ns = time.perf_counter_ns()
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not len(x):
            return False
        self.samples = np.concatenate((self.samples, x))
        if len(self.samples) > self.BUFFER_SAMPLES:
            drop = len(self.samples) - self.BUFFER_SAMPLES
            self.samples = self.samples[drop:]
            self.base += drop
        end = self.base + len(self.samples)
        self.scan_samples = np.concatenate((self.scan_samples, x))
        if len(self.scan_samples) > self.SCAN_OVERLAP_SAMPLES:
            drop = len(self.scan_samples) - self.SCAN_OVERLAP_SAMPLES
            self.scan_samples = self.scan_samples[drop:]
            self.scan_base += drop
        if (end - self.last_scan < self.SCAN_INTERVAL
                or len(self.scan_samples) < self.SCAN_OVERLAP_SAMPLES):
            self.feed_durations_ns.append(time.perf_counter_ns() - started_ns)
            return False
        self.last_scan = end
        frames = self._demod.decode(self.scan_samples, "info0")
        for frame in sorted(frames, key=lambda f: f["_start_sample"]):
            if (frame.get("ack") != 0
                    or not any(frame.get(name, 0) for name in
                               ("rate2743", "rate2800", "rate3429", "lo3000",
                                "hi3000", "lo3200", "hi3200"))):
                continue
            frame_end = self.scan_base + float(frame["_end_sample"])
            tone_at = int(math.ceil(frame_end)) + self.TONE_OFFSET
            lo = tone_at - self.scan_base
            hi = lo + self.TONE_SAMPLES
            if lo < 0 or hi > len(self.scan_samples):
                continue
            frame_key = round(
                (self.scan_base + float(frame["_start_sample"])) * BAUD / FS)
            if frame_key in self._scored_frames:
                continue
            self._scored_frames.add(frame_key)
            self.frames_checked += 1
            ratio, rms = self._tone_metrics(self.scan_samples[lo:hi], tone_at)
            self.best_tone_ratio = max(self.best_tone_ratio, ratio)
            self.best_tone_rms = max(self.best_tone_rms, rms)
            if ratio >= self.TONE_RATIO_MIN and rms >= self.TONE_RMS_MIN:
                self.last_match = {
                    "info0c_start_sample": self.scan_base + float(frame["_start_sample"]),
                    "info0c_end_sample": frame_end,
                    "tone_ratio": ratio,
                    "tone_rms": rms,
                }
                self.feed_durations_ns.append(
                    time.perf_counter_ns() - started_ns)
                return True
        self.feed_durations_ns.append(time.perf_counter_ns() - started_ns)
        return False

    def retained_pcm(self) -> bytes:
        return np.clip(np.rint(self.samples), -32768, 32767).astype("<i2").tobytes()

    def report(self) -> dict:
        durations = sorted(self.feed_durations_ns)
        def percentile(p):
            if not durations:
                return 0.0
            return durations[min(len(durations) - 1,
                                 math.ceil(p * len(durations)) - 1)] / 1e6
        return {
            "crc_valid_info0c_candidates": self.frames_checked,
            "best_following_tone_ratio": self.best_tone_ratio,
            "tone_ratio_threshold": self.TONE_RATIO_MIN,
            "tone_ratio_shortfall": self.TONE_RATIO_MIN - self.best_tone_ratio,
            "best_following_tone_rms": self.best_tone_rms,
            "tone_rms_threshold": self.TONE_RMS_MIN,
            "feed_calls": len(durations),
            "feed_mean_ms": (sum(durations) / len(durations) / 1e6
                             if durations else 0.0),
            "feed_p95_ms": percentile(.95),
            "feed_p99_ms": percentile(.99),
            "feed_max_ms": (durations[-1] / 1e6 if durations else 0.0),
            "scan_overlap_samples": self.SCAN_OVERLAP_SAMPLES,
            "match": self.last_match,
        }
