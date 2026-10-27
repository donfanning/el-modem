"""Caller-side line-mode autodetection ("listen, then respond in kind").

A calling modem does not know what the answerer is. Real caller modems wait
for the answer tone, ignore it, and then respond to whatever the answerer
sends next. This module does the same for the lab's two real-modem-capable
line modes:

* Project V.22/V.22bis (answerer sends its 2400 Hz high-band carrier)
* Bell 103 at 300 bps (answerer sends a 2225 Hz mark carrier)
* ITU-T V.21 (answerer sends its 1650-Hz channel-2 mark with the 1800-Hz guard)

Behaviour, in order:

1. **Answer-tone gate.** A 2100 Hz answer tone (ANS, V.25) lies inside the
   V.22 answerer's receive band, and the V.22 caller treats any carrier
   as the start of the handshake -- it "trains" on the tone and starts
   transmitting too early (seen live against NIST: a false training success
   during the tone, then carrier loss when the tone ended). So audio is fed to
   the V.22 engine as silence while the tone is present.
2. **Listen.** The V.22 engine runs (silent: a V.22 caller sends nothing until
   it hears the answerer). If it reports carrier after the tone, the answerer
   is a V.22 modem -> V.22 mode.
3. **Respond in kind.** If instead a steady 2225 Hz tone appears and the V.22
   engine heard nothing, switch to Bell 103 originate.
4. **Timeout.** If nothing identifies the answerer within ``fallback_seconds``,
   try Bell 103 (a Bell 103 answerer may be waiting for our carrier).

This is a heuristic over the signals seen so far, not a full ITU handshake
(no V.8 / V.23 / V.32); unknown answerers simply fall through to the
Bell 103 attempt.
"""

from __future__ import annotations

import math
import struct
import threading

SAMPLE_RATE = 8000
ANS_HZ = 2100.0
BELL103_ANSWER_MARK_HZ = 2225.0
V21_ANSWER_MARK_HZ = 1650.0
V21_GUARD_HZ = 1800.0


def tone_fraction(samples, freq_hz: float) -> tuple[float, float]:
    """Returns (fraction_of_energy_at_freq, rms) for the chunk.

    The fraction is ~1.0 for a clean tone at ``freq_hz`` and small for
    broadband/modulated signals or other frequencies."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    mean_sq = sum(x * x for x in samples) / n
    if mean_sq <= 0:
        return 0.0, 0.0
    k = 2.0 * math.cos(2.0 * math.pi * freq_hz / SAMPLE_RATE)
    a = b = 0.0
    for x in samples:
        a, b = x + k * a - b, a
    power = max(0.0, a * a + b * b - k * a * b)
    amplitude = math.sqrt(power) * 2.0 / n
    return (amplitude * amplitude / 2.0) / mean_sq, math.sqrt(mean_sq)


class AnsGate:
    """Replaces audio with silence while an answer tone (2100 Hz) is present,
    plus a short hold-over so ANS phase reversals (every 450 ms) don't leak
    single frames through."""

    def __init__(self, hold_seconds: float = 0.12, min_rms: float = 150.0,
                 min_fraction: float = 0.6):
        self._hold_total = int(hold_seconds * SAMPLE_RATE)
        self._hold = 0
        self._min_rms = min_rms
        self._min_fraction = min_fraction
        self.active = False
        self.seen = False

    def process(self, samples: list[int]) -> list[int]:
        fraction, rms = tone_fraction(samples, ANS_HZ)
        if rms >= self._min_rms and fraction >= self._min_fraction:
            self._hold = self._hold_total
            self.active = True
            self.seen = True
        elif self._hold > 0:
            self._hold -= len(samples)
        if self._hold <= 0:
            self.active = False
        return [0] * len(samples) if self.active else samples


class AutoCallerLine:
    """Caller-side line that picks V.22, V.21 or Bell 103 from answer audio.

    Presents the same transmitter/receiver surface as the individual engines:
    ``feed_pcm``, ``tx_samples``, ``feed_bytes``, ``pending_bits``,
    ``connected``, ``bit_rate``.
    """

    def __init__(
        self,
        v22,
        bell103_tx_factory,
        bell103_rx_factory,
        fallback_seconds: float = 15.0,
        mark_confirm_seconds: float = 0.1,
        v22_commit_seconds: float = 0.15,
        v21_tx_factory=None,
        v21_rx_factory=None,
    ):
        self._v22 = v22
        # Share the modem's lock: receive (decide/switch engines) runs on one
        # thread and transmit on another, and switching closes the V.22 engine.
        self._lock = getattr(v22, "_lock", None) or threading.RLock()
        self._b103_tx_factory = bell103_tx_factory
        self._b103_rx_factory = bell103_rx_factory
        self._b103_tx = None
        self._b103_rx = None
        if (v21_tx_factory is None) != (v21_rx_factory is None):
            raise ValueError("V.21 autodetection requires both transmitter and receiver factories")
        self._v21_tx = v21_tx_factory() if v21_tx_factory is not None else None
        self._v21_rx = v21_rx_factory() if v21_rx_factory is not None else None
        self._v25 = None
        if self._v21_tx is not None:
            # The same V.25 answer-tone procedure is required before the caller
            # can select V.21; until ANS completes, its repeated 1300-Hz call
            # sequence is transmitted while the V.22 receiver listens silently.
            from v25 import V25CallerProcedure
            self._v25 = V25CallerProcedure()
        self._gate = AnsGate()
        self._fallback = int(fallback_seconds * SAMPLE_RATE)
        self._mark_needed = int(mark_confirm_seconds * SAMPLE_RATE)
        self._v22_commit = int(v22_commit_seconds * SAMPLE_RATE)
        self._carrier_up_at = None
        self._events_seen = 0
        self._mark_run = 0
        self._v21_mark_run = 0
        self._clock = 0  # received samples
        self._base = 0  # clock value at the last answer-tone end / start
        self._queued = bytearray()  # DTE bytes fed before a mode is chosen
        self.state = "listen"  # listen | v21 | v22 | bell103
        self.reason = ""

    # -- engine surface ----------------------------------------------------
    @property
    def connected(self) -> bool:
        if self.state == "v22":
            return self._v22.is_settled()
        return self.state in ("v21", "bell103")

    @property
    def bit_rate(self) -> int:
        if self.state in ("v21", "bell103"):
            return 300
        return self._v22.bit_rate

    def feed_bytes(self, data: bytes) -> None:
        with self._lock:
            self._feed_bytes(data)

    def _feed_bytes(self, data: bytes) -> None:
        if self.state == "bell103":
            self._b103_tx.feed_bytes(data)
        elif self.state == "v21":
            self._v21_tx.feed_bytes(data)
        elif self.state == "v22":
            self._v22.feed_bytes(data)
        else:
            self._queued.extend(data)

    def feed_bits(self, bits) -> None:
        """Raw framed line bits (V.42 ODP etc.); only meaningful in V.22 mode."""
        with self._lock:
            if self.state == "v22":
                self._v22.feed_bits(bits)

    def pending_bits(self) -> int:
        with self._lock:
            if self.state == "bell103":
                return self._b103_tx.pending()
            if self.state == "v21":
                return self._v21_tx.pending()
            return self._v22.pending_bits() + 10 * len(self._queued)

    def tx_samples(self, count: int) -> list[int]:
        with self._lock:
            if self.state == "bell103":
                return self._b103_tx.samples(count)
            if self.state == "v21":
                return self._v21_tx.samples(count)
            if self._v25 is not None and self._v25.state != "complete":
                return self._v21_tx.samples(count)
            return self._v22.tx_samples(count)

    def feed_pcm(self, pcm: bytes) -> bytes:
        with self._lock:
            return self._feed_pcm(pcm)

    def _feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        count = len(pcm) // 2
        samples = list(struct.unpack(f"<{count}h", pcm))
        self._clock += count
        if self.state == "bell103":
            return self._b103_rx.feed_pcm(pcm)
        if self.state == "v21":
            return self._v21_rx.feed_pcm(pcm)
        if self.state == "v22":
            if self._v22.is_settled():
                return self._v22.feed_pcm(pcm)
            # Not settled yet: keep silencing any answer tone, since the
            # V.22 engine would otherwise "train" on it.
            gated = self._gate.process(samples)
            return self._v22.feed_pcm(struct.pack(f"<{count}h", *gated))

        was_active = self._gate.active
        gated = self._gate.process(samples)
        if self._v25 is not None and self._v25.state not in ("complete", "failed"):
            previous = self._v25.state
            state = self._v25.feed_pcm(pcm)
            if previous == "waiting_answer" and state == "answer_tone":
                self._v21_tx.stop_calling_tone()
        if self._gate.active or was_active:
            self._base = self._clock  # timeout counts from the tone's end
        if self._gate.active:
            self._mark_run = 0
            self._v21_mark_run = 0
        else:
            fraction, rms = tone_fraction(samples, BELL103_ANSWER_MARK_HZ)
            if rms >= 150.0 and fraction >= 0.6:
                self._mark_run += count
            else:
                self._mark_run = 0
            if self._v21_tx is not None and self._v25.state == "complete":
                mark_fraction, mark_rms = tone_fraction(samples, V21_ANSWER_MARK_HZ)
                guard_fraction, guard_rms = tone_fraction(samples, V21_GUARD_HZ)
                if (mark_rms >= 150.0 and guard_rms >= 150.0
                        and mark_fraction >= 0.20 and guard_fraction >= 0.05):
                    self._v21_mark_run += count
                else:
                    self._v21_mark_run = 0

        if self._v21_mark_run >= self._mark_needed:
            self._enter_v21()
            return self._v21_rx.feed_pcm(pcm)

        data = self._v22.feed_pcm(struct.pack(f"<{count}h", *gated))
        # Carrier detect alone proves nothing: SpanDSP's V.22 caller also
        # reports carrier on a Bell 103 answerer's 2225 Hz mark tone, which
        # lies inside the same receive band. The tell is the signal's shape:
        # Bell 103 is a pure narrowband tone, V.22 is broadband DPSK. So
        # prefer the narrowband check, and commit to V.22 only once carrier
        # has persisted without that signature.
        # Track carrier from the engine's events. A carrier-up while an answer
        # tone is (or just was) being gated is the tone's leading edge leaking
        # through a partially-filled frame, not an answerer; a carrier-down
        # cancels whatever was pending.
        events = self._v22.events
        for name, _ in events[self._events_seen:]:
            if name == "CARRIER_UP" and not self._gate.active:
                self._carrier_up_at = self._clock
            elif name == "CARRIER_DOWN":
                self._carrier_up_at = None
        self._events_seen = len(events)
        if self._mark_run >= self._mark_needed:
            self._enter_bell103("steady 2225 Hz mark carrier (Bell 103 answerer)")
        elif (
            self._carrier_up_at is not None
            and not self._gate.active
            and self._mark_run == 0
            and self._clock - self._carrier_up_at >= self._v22_commit
        ):
            self._enter_v22()
            return b""
        elif self._clock - self._base >= self._fallback:
            self._enter_bell103("no answerer signal identified; trying Bell 103")
        return data if self.state == "v22" else b""

    # -- mode switches -----------------------------------------------------
    def _enter_v22(self) -> None:
        self.state = "v22"
        self.reason = "V.22 carrier heard after answer tone"
        if self._queued:
            self._v22.feed_bytes(bytes(self._queued))
            self._queued.clear()

    def _enter_v21(self) -> None:
        self.state = "v21"
        self.reason = "V.21 answer channel and 1800 Hz guard detected after V.25 ANS"
        self._v21_tx.release_call_sequence()
        self._v22.close()
        if self._queued:
            self._v21_tx.feed_bytes(bytes(self._queued))
            self._queued.clear()

    def _enter_bell103(self, reason: str) -> None:
        self.state = "bell103"
        self.reason = reason
        self._b103_tx = self._b103_tx_factory()
        self._b103_rx = self._b103_rx_factory()
        if self._queued:
            self._b103_tx.feed_bytes(bytes(self._queued))
            self._queued.clear()
        self._v22.close()
