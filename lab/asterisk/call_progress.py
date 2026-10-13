#!/usr/bin/env python3
"""Modem-layer call-progress classifiers for received call audio."""

import math
import struct
from collections.abc import Callable

SAMPLE_RATE = 8000
FRAME_SAMPLES = 160


def _goertzel_power(samples: list[int], frequency: float) -> float:
    coefficient = 2.0 * math.cos(2.0 * math.pi * frequency / SAMPLE_RATE)
    previous = 0.0
    previous2 = 0.0
    for sample in samples:
        current = sample + coefficient * previous - previous2
        previous2 = previous
        previous = current
    power = previous2 * previous2 + previous * previous - coefficient * previous * previous2
    return max(0.0, power) / (len(samples) * len(samples))


class BusyToneDetector:
    """Detect repeated North American 480+620 Hz busy cadence in PCM."""

    def __init__(self):
        self._pending = bytearray()
        self._current_state: bool | None = None
        self._state_samples = 0
        self._last_on_samples: int | None = None
        self._valid_cycles = 0
        self.detected = False

    def _frame_has_busy_tone(self, frame: list[int]) -> bool:
        mean_square = sum(sample * sample for sample in frame) / len(frame)
        if mean_square < 300 * 300:
            return False
        tone_480 = _goertzel_power(frame, 480.0)
        tone_620 = _goertzel_power(frame, 620.0)
        return (
            tone_480 > 20_000
            and tone_620 > 20_000
            and tone_480 / mean_square > 0.12
            and tone_620 / mean_square > 0.12
        )

    def feed_pcm(self, pcm: bytes) -> bool:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        self._pending.extend(pcm)
        frame_bytes = FRAME_SAMPLES * 2
        while len(self._pending) >= frame_bytes:
            frame_data = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            frame = list(struct.unpack("<160h", frame_data))
            state = self._frame_has_busy_tone(frame)
            if self._current_state is None:
                self._current_state = state
                self._state_samples = FRAME_SAMPLES
                continue
            if state == self._current_state:
                self._state_samples += FRAME_SAMPLES
                continue

            previous_state = self._current_state
            duration = self._state_samples
            self._current_state = state
            self._state_samples = FRAME_SAMPLES

            if previous_state:
                if 2_800 <= duration <= 5_600:
                    self._last_on_samples = duration
                else:
                    self._last_on_samples = None
                    self._valid_cycles = 0
            elif self._last_on_samples is not None:
                if 2_800 <= duration <= 6_400:
                    self._valid_cycles += 1
                    if self._valid_cycles >= 2:
                        self.detected = True
                else:
                    self._valid_cycles = 0
                self._last_on_samples = None
        return self.detected


class RingbackToneDetector:
    """Detect North American 440+480 Hz ringback cadence from received audio."""

    def __init__(self):
        self._pending = bytearray()
        self._current_state: bool | None = None
        self._state_samples = 0
        self._valid_tone_burst = False
        self._ring_count = 0
        self.detected = False

    @property
    def ring_count(self) -> int:
        return self._ring_count

    def _frame_has_ringback(self, frame: list[int]) -> bool:
        mean_square = sum(sample * sample for sample in frame) / len(frame)
        if mean_square < 300 * 300:
            return False
        power_440 = _goertzel_power(frame, 440.0)
        power_480 = _goertzel_power(frame, 480.0)
        return (
            power_440 > 15_000
            and power_480 > 15_000
            and power_440 / mean_square > 0.08
            and power_480 / mean_square > 0.08
        )

    def feed_pcm(self, pcm: bytes) -> bool:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        self._pending.extend(pcm)
        frame_bytes = FRAME_SAMPLES * 2
        while len(self._pending) >= frame_bytes:
            frame_data = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            frame = list(struct.unpack("<160h", frame_data))
            state = self._frame_has_ringback(frame)
            if self._current_state is None:
                self._current_state = state
                self._state_samples = FRAME_SAMPLES
                continue
            if state == self._current_state:
                self._state_samples += FRAME_SAMPLES
                continue

            previous_state = self._current_state
            duration = self._state_samples
            self._current_state = state
            self._state_samples = FRAME_SAMPLES
            if previous_state:
                self._valid_tone_burst = 12_000 <= duration <= 20_000
            elif self._valid_tone_burst:
                if 24_000 <= duration <= 40_000:
                    self._ring_count += 1
                    self.detected = True
                self._valid_tone_burst = False
        return self.detected


class VoiceActivityDetector:
    """Heuristically detect speech-like broadband audio, excluding pure tones."""

    def __init__(self):
        self._pending = bytearray()
        self._active_frames = 0
        self._inactive_frames = 0
        self.detected = False

    def _frame_has_speech_like_energy(self, frame: list[int]) -> bool:
        mean_square = sum(sample * sample for sample in frame) / len(frame)
        if mean_square < 450 * 450:
            return False
        band_powers = [
            _goertzel_power(frame, frequency)
            for frequency in range(300, 3401, 100)
        ]
        total = sum(band_powers)
        if total == 0:
            return False
        strongest = max(band_powers)
        occupied_bins = sum(power >= strongest * 0.035 for power in band_powers)
        return occupied_bins >= 7 and strongest / total < 0.65

    def feed_pcm(self, pcm: bytes) -> bool:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        self._pending.extend(pcm)
        frame_bytes = FRAME_SAMPLES * 2
        while len(self._pending) >= frame_bytes:
            frame_data = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            frame = list(struct.unpack("<160h", frame_data))
            if self._frame_has_speech_like_energy(frame):
                self._active_frames += 1
                self._inactive_frames = 0
            else:
                self._inactive_frames += 1
                if self._inactive_frames >= 3:
                    self._active_frames = 0
            if self._active_frames >= 3:
                self.detected = True
        return self.detected


class OutgoingCallProgressMonitor:
    """Classify locally received call-progress audio and notify the DCE."""

    def __init__(self, notify: Callable[[str], None]):
        self._notify = notify
        self._busy = BusyToneDetector()
        self._ringback = RingbackToneDetector()
        self._voice = VoiceActivityDetector()
        self._finished = False

    def feed_pcm(self, pcm: bytes) -> str | None:
        if self._finished:
            return None
        if self._busy.feed_pcm(pcm):
            self._finished = True
            self._notify("BUSY")
            return "BUSY"
        previous_rings = self._ringback.ring_count
        self._ringback.feed_pcm(pcm)
        new_rings = self._ringback.ring_count - previous_rings
        if self._voice.feed_pcm(pcm):
            self._finished = True
            self._notify("VOICE")
            return "VOICE"
        if new_rings > 0:
            for _ in range(new_rings):
                self._notify("RING")
            return "RING"
        return None
