"""Small V.25 automatic-caller answer-tone/end-silence detector."""

from __future__ import annotations

import struct
import time

from autodetect import tone_fraction

SAMPLE_RATE = 8000
FRAME_SAMPLES = 160


class V25CallerProcedure:
    """Gate caller data-ready on V.25 ANS detection and its trailing silence.

    The caller recognizes at least 100 ms of 2100-Hz answer tone, then waits
    for 75 +/- 20 ms without it before data-ready. RTP is processed in 20-ms
    frames, so four absent frames (80 ms) are required. The answer-tone
    detector uses the V.25 selective-tone recommendation: 2100 Hz must be the
    dominant component, avoiding a false match on voice or service tones.
    """

    def __init__(self, *, started_at: float | None = None,
                 answer_timeout: float = 30.0):
        self.state = "waiting_answer"  # waiting_answer | answer_tone | complete | failed
        self.started_at = time.monotonic() if started_at is None else started_at
        self.answer_timeout = float(answer_timeout)
        self.answer_detected = False
        self.answer_tone_started_at: float | None = None
        self._pending = bytearray()
        self._answer_run = 0
        self._answer_elapsed_samples = 0
        self._silence_samples = 0

    @staticmethod
    def _is_answer_tone(frame: list[int]) -> bool:
        fraction, rms = tone_fraction(frame, 2100.0)
        return rms >= 150.0 and fraction >= 0.60

    def feed_pcm(self, pcm: bytes) -> str:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        if self.state in ("complete", "failed"):
            return self.state
        self._pending.extend(pcm)
        frame_bytes = FRAME_SAMPLES * 2
        while len(self._pending) >= frame_bytes:
            frame_data = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            frame = list(struct.unpack("<160h", frame_data))
            is_tone = self._is_answer_tone(frame)
            if self.state == "waiting_answer":
                self._answer_run = self._answer_run + FRAME_SAMPLES if is_tone else 0
                if self._answer_run >= int(0.100 * SAMPLE_RATE):
                    self.state = "answer_tone"
                    self.answer_detected = True
                    self.answer_tone_started_at = time.monotonic()
                    self._answer_elapsed_samples = self._answer_run
            elif self.state == "answer_tone":
                self._answer_elapsed_samples += FRAME_SAMPLES
                if is_tone:
                    self._silence_samples = 0
                else:
                    self._silence_samples += FRAME_SAMPLES
                    if self._silence_samples >= int(0.080 * SAMPLE_RATE):
                        self.state = "complete"
                        break
                # V.25 nominally limits ANS to 4 s. One frame of edge
                # uncertainty is inherent in the 20-ms RTP detector.
                if self._answer_elapsed_samples > int(4.02 * SAMPLE_RATE):
                    self.state = "failed"
                    break
        return self.state

    def poll(self, now: float | None = None) -> str:
        current = time.monotonic() if now is None else now
        if (self.state == "waiting_answer"
                and current - self.started_at >= self.answer_timeout):
            self.state = "failed"
        elif (self.state == "answer_tone" and self.answer_tone_started_at is not None
              and current - self.answer_tone_started_at >= 4.5):
            # Also bound a stalled RTP source after ANS begins; a sample-count
            # limit alone cannot fire when the stream itself stops arriving.
            self.state = "failed"
        return self.state
