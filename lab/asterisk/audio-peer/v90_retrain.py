"""Tone handshake for the V.90 retrain sequence (ITU-T V.90 §9.5).

A retrain starts the tone exchange before a full Phase 2. This class models one modem's side of
that exchange on an 8 kHz sample clock and tells its owner when to build fresh Phase 2 objects.

* The initiating modem sends silence for 70 ms, then its tone (Tone B, 1200 Hz, for the digital
  modem; Tone A, 2400 Hz, for the analogue modem).
* A responding modem is already running; once it has heard the peer's tone for more than 50 ms it
  sends silence for 70 ms and then its own tone.
* A modem is ``ready`` when it has started its own tone and has heard the peer's tone for more than
  50 ms. It keeps its tone on for ``HOLD_SAMPLES`` more (the responder becomes ready before the
  initiator, and stopping at once would cut off the peer's own 50 ms detection), then sends the
  75 ms silent period that ends Phase 1 and runs the full Phase 2 start-up procedure from
  ``phase2_start_at``. Without that silence, fresh Phase 2 detectors read the tail of the
  handshake tone as a Tone reversal and fail.

Interpretation: the text asks the initiating modem to "detect a Tone A phase reversal" before the
full Phase 2, and the initiating analogue modem to send one. I read those as the tone
synchronisation that the ordinary Phase 2 start-up already performs (INFO0 first, then the
reversal exchange), not as a second reversal in this preamble. That reading is unverified.
"""

from __future__ import annotations

import numpy as np
from v90_digital_phase2_rx import V90DigitalPhase2SignalRx

FS = 8000
TONE_HZ = {"digital": 1200.0, "analogue": 2400.0}
SILENCE_SAMPLES = 560                   # 70 ms
DETECT_SAMPLES = 400                    # 50 ms
_BLOCK = 80                             # 10 ms hop
_WINDOW = 160                           # 20 ms analysis window
_MIN_RMS = 15.0
_MIN_RATIO = 0.6
HOLD_SAMPLES = 800                      # 100 ms: covers the peer's 50 ms detection plus the delay
PHASE1_SILENCE_SAMPLES = 600            # the 75 ms silent period that ends Phase 1


class RetrainPreamble:
    def __init__(self, role: str, *, initiating: bool, peak: float = 1000.0):
        if role not in TONE_HZ:
            raise ValueError("role must be digital or analogue")
        if peak <= 0:
            raise ValueError("tone peak must be positive")
        self.role = role
        self.peer = "analogue" if role == "digital" else "digital"
        self.initiating = bool(initiating)
        self.peak = float(peak)
        self.tx_clock = 0
        self.rx_clock = 0
        self._tone_from: int | None = SILENCE_SAMPLES if initiating else None
        self._silent_until: int | None = SILENCE_SAMPLES if initiating else None
        self._peer_run = 0              # consecutive 10 ms blocks with the peer tone present
        self._pending = np.empty(0)
        self._history = np.empty(0)
        self.peer_heard_at: int | None = None       # rx sample where >50 ms of peer tone was reached
        self.peer_ended_at: int | None = None       # rx sample where the peer's tone stopped
        self._absent_run = 0
        self.ready_at: int | None = None            # tx sample at which both conditions held
        self.tone_ends_at: int | None = None
        self.peer_ended_tx: int | None = None       # tx sample when the peer's tone end was seen
        self.phase2_start_at: int | None = None     # tx sample at which fresh Phase 2 begins
        self.events: list[tuple[int, str]] = []
        self._tone_a_reversal_rx = V90DigitalPhase2SignalRx()

    @property
    def ready(self) -> bool:
        return self.ready_at is not None

    @property
    def transmitting_tone(self) -> bool:
        return (self._tone_from is not None and self.tx_clock >= self._tone_from
                and (self.tone_ends_at is None or self.tx_clock < self.tone_ends_at))

    @property
    def receive_ready(self) -> bool:
        """Fresh Phase 2 receive detectors may start: this side's handshake is complete and the
        peer's tone is gone (a detector started earlier reads the tone's end as a reversal)."""
        if self.role == "digital":
            # For either digital-initiated (§9.5.1.1) or digital-response
            # (§9.5.1.2) retrain, the peer's Tone A phase reversal is the
            # continuation boundary (§9.2.1.1.3). Waiting for the generic
            # post-detection hold delays the required Tone B reversal past
            # its 40 ± 1 ms slot.
            return (self._tone_from is not None
                    and self.tx_clock >= self._tone_from
                    and self.peer_ended_at is not None)
        return self.ready and self.peer_ended_at is not None

    @property
    def phase2_due(self) -> bool:
        return self.phase2_start_at is not None and self.tx_clock >= self.phase2_start_at

    def render(self, count: int) -> np.ndarray:
        """Next ``count`` transmit samples: silence until this modem's tone starts."""
        out = np.zeros(count)
        if self._tone_from is not None:
            start = max(0, self._tone_from - self.tx_clock)
            stop = count if self.tone_ends_at is None else min(
                count, max(0, self.tone_ends_at - self.tx_clock))
            if start < stop:
                t = np.arange(self.tx_clock + start, self.tx_clock + stop)
                phase = np.zeros(len(t))
                if self.role == "analogue":
                    # The analogue modem's Tone A reversal is part of the
                    # continuation into full Phase 2, after the 50 ms minimum.
                    reverse_at = self._tone_from + DETECT_SAMPLES
                    phase[t >= reverse_at] = np.pi
                out[start:stop] = self.peak * np.cos(
                    2 * np.pi * TONE_HZ[self.role] * t / FS + phase)
        self.tx_clock += count
        self._update_ready()
        return out

    def feed(self, samples) -> None:
        """Consume received samples (contiguous) and update detection."""
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("receive audio must contain finite samples")
        self._pending = np.concatenate((self._pending, x))
        if self.role == "digital":
            for event in self._tone_a_reversal_rx.feed(x, expect="tone_a"):
                if (event.kind == "tone_a_reversal"
                        and self.peer_ended_at is None):
                    self.peer_ended_at = int(event.at)
                    self.peer_ended_tx = self.tx_clock
                    self.events.append((self.rx_clock + len(x),
                                        "peer Tone A phase reversal"))
                    self._schedule_phase2()
        while len(self._pending) >= _BLOCK:
            block, self._pending = self._pending[:_BLOCK], self._pending[_BLOCK:]
            self._history = np.concatenate((self._history, block))[-_WINDOW:]
            self.rx_clock += _BLOCK
            self._detect()
        self._update_ready()

    def _detect(self) -> None:
        if len(self._history) < _WINDOW:
            return
        window = self._history
        n = np.arange(_WINDOW)
        k = 2 * np.pi * TONE_HZ[self.peer] * n / FS
        tone_power = 2.0 * abs(np.sum(window * np.exp(-1j * k))) ** 2 / _WINDOW ** 2
        total = float(np.mean(window ** 2)) + 1e-12
        rms = np.sqrt(total)
        present = rms >= _MIN_RMS and (tone_power / total) >= _MIN_RATIO
        self._peer_run = self._peer_run + 1 if present else 0
        self._absent_run = 0 if present else self._absent_run + 1
        if (self.role != "digital"
                and self.peer_heard_at is not None and self.peer_ended_at is None
                and self._absent_run * _BLOCK >= _WINDOW):
            # The window is 20 ms, so two empty hops mean the tone is fully gone.
            self.peer_ended_at = self.rx_clock
            self.peer_ended_tx = self.tx_clock
            self.events.append((self.rx_clock, "peer tone ended"))
            self._schedule_phase2()
        if self.peer_heard_at is None and self._peer_run * _BLOCK >= DETECT_SAMPLES:
            self.peer_heard_at = self.rx_clock
            self.events.append((self.rx_clock, "peer tone"))
            if self._tone_from is None:
                # Responder: silence for 70 ms from now, then our own tone.
                self._silent_until = self.tx_clock + SILENCE_SAMPLES
                self._tone_from = self._silent_until
                self.events.append((self.tx_clock, "responding"))

    def _update_ready(self) -> None:
        if (self.ready_at is None and self.peer_heard_at is not None
                and self._tone_from is not None and self.tx_clock >= self._tone_from):
            self.ready_at = self.tx_clock
            self.tone_ends_at = self.tx_clock + HOLD_SAMPLES
            self.events.append((self.tx_clock, "ready"))
            self._schedule_phase2()

    def _schedule_phase2(self) -> None:
        """Set the continuation point after the retrain tone exchange."""
        if self.peer_ended_tx is None:
            return
        if self.role == "digital":
            # V.90 §9.5.1.1 proceeds directly at §9.2.1.1.3: the Tone B
            # reversal is due 40 ± 1 ms after the detected Tone A reversal.
            # Waiting for tone cessation plus a new 75 ms Phase 1 silence
            # misses that deadline and incorrectly restarts at INFO0.
            self.phase2_start_at = self.peer_ended_tx
        else:
            if self.tone_ends_at is None:
                return
            self.phase2_start_at = (max(self.tone_ends_at, self.peer_ended_tx)
                                    + PHASE1_SILENCE_SAMPLES)
