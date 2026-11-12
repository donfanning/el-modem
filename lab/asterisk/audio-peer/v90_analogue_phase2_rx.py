"""Incremental CRC-checked INFO receiver for the V.90 analogue-caller role."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import v34_info
import v90_info_reference as v90
from v90_analogue_caller_phase2 import AnaloguePhase2Action, V90AnalogueCallerPhase2
from v90_digital_phase2_rx import V90DigitalPhase2SignalRx


FS = 8000
_WINDOW = int(1.5 * FS)
_SCAN_STRIDE = FS // 5
_DUPLICATE_SAMPLES = 50


@dataclass(frozen=True)
class AnaloguePhase2RxEvent:
    kind: str
    at: int
    complete_at: int
    fields: dict


class V90AnaloguePhase2FrameRx:
    """Scan contiguous 8 kHz PCM for digital INFO0d and INFO1d frames.

    The digital modem sends INFO at 1200 Hz. INFO0d uses the V.90 Table 7
    layout; INFO1d uses the V.34 INFO1c/Table 9 layout. Only CRC-checked,
    correctly framed messages are emitted. Results use absolute receive sample
    positions, and the caller must map that clock to its media/transmit clock.
    """

    def __init__(self, *, phases: int = 16):
        if isinstance(phases, bool) or not isinstance(phases, int) or phases <= 0:
            raise ValueError("demodulator phases must be a positive integer")
        self.phases = phases
        self._pending = np.zeros(0, dtype=float)
        self._base = 0
        self._total = 0
        self._until_scan = 0
        self._rx_next: int | None = None
        self._last_at: dict[str, int] = {}

    def feed(self, samples, *, rx_at: int,
             kinds: tuple[str, ...] | None = None
             ) -> list[AnaloguePhase2RxEvent]:
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("receive audio must contain finite samples")
        if isinstance(rx_at, bool) or not isinstance(rx_at, int) or rx_at < 0:
            raise ValueError("rx_at must be a nonnegative integer sample position")
        selected = (("info0d", "info1d") if kinds is None else tuple(kinds))
        if any(kind not in ("info0d", "info1d") for kind in selected):
            raise ValueError("frame kinds must contain only info0d or info1d")
        if not len(x):
            return []
        if self._rx_next is not None and rx_at != self._rx_next:
            raise ValueError(f"receive sample discontinuity; apply an explicit gap policy (rx_at {rx_at}, expected {self._rx_next})")
        if self._rx_next is None:
            self._base = rx_at
        self._rx_next = rx_at + len(x)
        self._total += len(x)
        self._until_scan += len(x)
        self._pending = np.concatenate((self._pending, x))
        if len(self._pending) > _WINDOW:
            drop = len(self._pending) - _WINDOW
            self._pending = self._pending[drop:]
            self._base += drop
        if self._until_scan < _SCAN_STRIDE:
            return []
        self._until_scan %= _SCAN_STRIDE

        found = []
        for kind, bits, decoder in (
            ("info0d", v90.INFO0D_BITS,
             lambda x: v90.decode_info0d(x, "call", self.phases)),
            ("info1d", 109,
             lambda x: v90.decode_info1d(x, "call", self.phases)),
        ):
            if kind not in selected:
                continue
            frames = decoder(self._pending)
            for frame in frames:
                at = self._base + int(round(frame["_start_sample"]))
                previous = self._last_at.get(kind)
                if previous is not None and abs(at - previous) <= _DUPLICATE_SAMPLES:
                    continue
                self._last_at[kind] = at
                complete = at + int(round(bits * FS / v34_info.BAUD))
                fields = {key: value for key, value in frame.items()
                          if not key.startswith("_")}
                found.append(AnaloguePhase2RxEvent(kind, at, complete, fields))
        return sorted(found, key=lambda event: (event.complete_at, event.at, event.kind))

    @property
    def samples_seen(self) -> int:
        return self._total


class V90AnaloguePhase2SignalRx:
    """Detect caller-side Tone B/reversals and L1/L2 receive intervals.

    ``expect`` is selected by the Phase 2 controller: ``tone_b`` enables the
    1200 Hz carrier detector, ``probe`` enables the shared state-gated probing
    comb detector, ``tone_b_probe`` enables both across the phase boundary,
    and ``none`` disables both semantic detections. This gating is essential
    because INFO0d/INFO1d also use the 1200 Hz carrier.
    """

    _WIN = 20  # 2.5 ms, three cycles at the 1200 Hz Tone B carrier.

    def __init__(self):
        self._total = 0
        self._origin: int | None = None
        self._rx_next: int | None = None
        self._tone_pending = np.zeros(0, dtype=float)
        self._tone_base = 0
        self._tone_history = np.zeros(0, dtype=float)
        self._phasors: list[complex] = []
        self._stable: list[bool] = []
        self._active = False
        self._last_reversal: int | None = None
        self._tone_expected = False
        self._probe = V90DigitalPhase2SignalRx()
        self._probe_origin = 0
        self._last_probe_expect: str | None = None

    def feed(self, samples, *, rx_at: int,
             expect: str) -> list[AnaloguePhase2RxEvent]:
        if expect not in ("tone_b", "probe", "tone_b_probe", "none"):
            raise ValueError("expect must be tone_b, probe, tone_b_probe, or none")
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("receive audio must contain finite samples")
        if isinstance(rx_at, bool) or not isinstance(rx_at, int) or rx_at < 0:
            raise ValueError("rx_at must be a nonnegative integer sample position")
        if not len(x):
            return []
        if self._rx_next is not None and rx_at != self._rx_next:
            raise ValueError("receive sample discontinuity; apply an explicit gap policy")
        if self._rx_next is None:
            self._origin = rx_at
        self._rx_next = rx_at + len(x)
        self._total += len(x)

        probe_expect = "probe" if expect in ("probe", "tone_b_probe") else "none"
        # The local hybrid echo and the remote L1/L2 are distinct probe
        # windows even when both are present in adjacent media ticks. Start a
        # fresh detector at the protocol transition so the local echo cannot
        # consume the remote-probe one-shot events.
        if self._last_probe_expect == "tone_b_probe" and expect == "probe":
            self._probe = V90DigitalPhase2SignalRx()
            self._probe_origin = self._total - len(x)
        self._last_probe_expect = expect
        events = []
        for event in self._probe.feed(x, expect=probe_expect):
            if event.kind in ("l1_complete", "l2_complete"):
                events.append(AnaloguePhase2RxEvent(
                    event.kind, int(self._origin) + self._probe_origin + event.at,
                    int(self._origin) + self._probe_origin + event.complete_at,
                    event.fields))

        enabled = expect in ("tone_b", "tone_b_probe")
        if enabled != self._tone_expected:
            self._tone_pending = np.zeros(0, dtype=float)
            self._tone_base = self._total - len(x)
            self._tone_history = np.zeros(0, dtype=float)
            self._phasors.clear()
            self._stable.clear()
            self._active = False
            self._last_reversal = None
            self._tone_expected = enabled
        if enabled:
            self._tone_pending = np.concatenate((self._tone_pending, x))
            while len(self._tone_pending) >= self._WIN:
                block = self._tone_pending[:self._WIN]
                self._tone_pending = self._tone_pending[self._WIN:]
                at = self._tone_base
                self._tone_base += self._WIN
                events.extend(self._tone_window(block, at))
        return sorted(events, key=lambda event: (event.complete_at, event.at, event.kind))

    def reset_tone_detector(self) -> None:
        """Re-arm steady Tone B detection after a validated DPSK INFO frame."""
        self._tone_pending = np.zeros(0, dtype=float)
        self._tone_base = self._total
        self._tone_history = np.zeros(0, dtype=float)
        self._phasors.clear()
        self._stable.clear()
        self._active = False
        self._last_reversal = None
        self._tone_expected = True

    def _tone_window(self, block: np.ndarray, at: int) -> list[AnaloguePhase2RxEvent]:
        events = []
        self._tone_history = np.concatenate((self._tone_history, block))[-3 * self._WIN:]
        t = at + np.arange(self._WIN)
        phasor = np.sum(block * np.exp(-2j * np.pi * 1200.0 * t / FS))
        coherence = abs(phasor) / (float(np.abs(block).sum()) + 1e-12)
        level = float(np.abs(block).mean())
        stable = coherence > .45 and level > 60
        self._phasors.append(phasor)
        self._stable.append(stable)
        if len(self._phasors) > 16:
            self._phasors.pop(0)
            self._stable.pop(0)
        j = len(self._phasors) - 1
        if not self._active and j >= 5 and all(self._stable[j - k] for k in range(6)):
            self._active = True
            events.append(AnaloguePhase2RxEvent(
                "tone_b", int(self._origin) + at - 5 * self._WIN,
                int(self._origin) + at + self._WIN, {}))
        elif self._active and j >= 1 and not stable and not self._stable[j - 1]:
            self._active = False

        # Require a stable carrier on both sides of the candidate edge, then
        # fit the reversal location instead of reporting a whole-window edge.
        if (self._active and j >= 12
                and all(self._stable[j - k] for k in range(2, 12))
                and self._stable[j] and self._stable[j - 2]
                and (self._phasors[j] * np.conj(self._phasors[j - 2])).real < 0):
            start = at - 2 * self._WIN
            if self._last_reversal is None or start - self._last_reversal >= 400:
                fitted = self._fit_reversal(start)
                if fitted is not None:
                    reversal_at, fit_error = fitted
                    self._last_reversal = reversal_at
                    events.append(AnaloguePhase2RxEvent(
                        "tone_b_reversal", int(self._origin) + reversal_at,
                        int(self._origin) + at + self._WIN,
                        {"normalized_fit_error": fit_error}))
        return events

    def _fit_reversal(self, start: int) -> tuple[int, float] | None:
        if len(self._tone_history) < 3 * self._WIN:
            return None
        x = self._tone_history
        n = len(x)
        t = start + np.arange(n)
        c = np.cos(2 * np.pi * 1200.0 * t / FS)
        s = np.sin(2 * np.pi * 1200.0 * t / FS)
        best_error, best_energy, best_at = None, 0.0, None
        for split in range(self._WIN - 4, 2 * self._WIN + 5):
            sign = np.where(np.arange(n) < split, 1.0, -1.0)
            matrix = np.column_stack((sign * c, sign * s))
            coeff, *_ = np.linalg.lstsq(matrix, x, rcond=None)
            residual = x - matrix @ coeff
            error = float(residual @ residual)
            energy = float(np.sum((matrix @ coeff) ** 2))
            if best_error is None or error < best_error:
                best_error, best_energy, best_at = error, energy, split
        if best_error is None or best_at is None:
            return None
        normalized = best_error / (best_energy + 1e-12)
        if normalized > .2:
            return None
        return start + best_at, float(normalized)

    @property
    def samples_seen(self) -> int:
        return self._total


@dataclass(frozen=True)
class AnaloguePhase2DeadlineMiss:
    event: str
    detected_at: int
    deadline: int
    tx_now: int


class V90AnaloguePhase2Adapter:
    """Route caller-side receive events to the Phase 2 controller.

    Receive and transmit clocks must be related by the caller-provided
    ``rx_to_media`` mapping. This adapter does not conceal gaps or infer a
    clock relationship. INFO frames are delivered at CRC-validated frame
    completion; tone reversals retain their fitted line-sample positions.
    """

    def __init__(self, controller: V90AnalogueCallerPhase2, *, rx_to_media):
        if not callable(rx_to_media):
            raise TypeError("rx_to_media must map receive samples to media samples")
        self.controller = controller
        self.rx_to_media = rx_to_media
        self.frames = V90AnaloguePhase2FrameRx()
        self.signals = V90AnaloguePhase2SignalRx()
        self.started = False
        self.deadline_misses: list[AnaloguePhase2DeadlineMiss] = []
        self._pending_early_tones: list[AnaloguePhase2RxEvent] = []
        self._last_routed_tone_b_rx_at: int | None = None
        # Keep a bounded diagnostic trace: RX, mapped-media and TX clocks
        # must be visible together to diagnose turnaround failures.
        self.event_trace: list[dict] = []

    def start(self, media_at: int) -> list[AnaloguePhase2Action]:
        if self.started:
            raise RuntimeError("analogue Phase 2 adapter already started")
        self.started = True
        return self.controller.start(media_at)

    def start_v90_retrain_response(
            self, media_at: int, *, info0d: dict
            ) -> list[AnaloguePhase2Action]:
        """Start the post-CONNECT V.90 continuation on a fresh adapter."""
        if self.started:
            raise RuntimeError("analogue Phase 2 adapter already started")
        self.started = True
        try:
            return self.controller.start_v90_retrain_response(
                media_at, info0d=info0d)
        except Exception:
            self.started = False
            raise

    def start_v90_retrain_initiation(
            self, media_at: int, *, info0d: dict
            ) -> list[AnaloguePhase2Action]:
        """Start an analogue-initiated retrain on a fresh Phase-2 adapter."""
        if self.started:
            raise RuntimeError("analogue Phase 2 adapter already started")
        self.started = True
        try:
            return self.controller.start_v90_retrain_initiation(
                media_at, info0d=info0d)
        except Exception:
            self.started = False
            raise

    def feed_tx_complete(self, event: str, *, media_at: int,
                         tx_now: int) -> list[AnaloguePhase2Action]:
        if not self.started:
            raise RuntimeError("start the analogue Phase 2 adapter first")
        if event != "info0a_complete":
            raise ValueError("unsupported caller-side TX completion event")
        return self.controller.feed(event, media_at, now=tx_now)

    def feed(self, samples, *, rx_at: int, tx_now: int) -> list[AnaloguePhase2Action]:
        if not self.started:
            raise RuntimeError("start the analogue Phase 2 adapter first")
        if rx_at < 0 or tx_now < 0:
            raise ValueError("sample positions must be nonnegative")
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not len(x):
            return []
        state = self.controller.state
        if state in (self.controller.WAIT_INFO0D,
                     self.controller.WAIT_TONE_B,
                     self.controller.WAIT_TONE_B_REVERSAL):
            frame_kinds = ("info0d",)
        elif state == self.controller.WAIT_INFO1D:
            frame_kinds = ("info1d",)
        else:
            frame_kinds = ()
        detected = self.frames.feed(x, rx_at=rx_at, kinds=frame_kinds)
        expected_signal = self._expected_signal()
        detected.extend(self.signals.feed(x, rx_at=rx_at,
                                          expect=expected_signal))
        detected.sort(key=lambda event: (event.complete_at, event.at, event.kind))
        # INFO0d/INFO1d and Tone B all occupy 1200 Hz. A stable run in the
        # DPSK symbols can satisfy the simple tone detector before frame CRC
        # validation. Hold candidates until a valid frame resolves whether a
        # tone began before the message or inside its carrier.
        info_frames = [event for event in detected
                       if event.kind in ("info0d", "info1d")]
        early_tone_events = []
        late_tone_events = []
        filtered = []
        for event in detected:
            if (event.kind != "tone_b" or self.controller.state not in
                    (self.controller.WAIT_INFO0D, self.controller.WAIT_INFO1D)):
                filtered.append(event)
            else:
                self._pending_early_tones.append(event)
        reset_tone_after_info = False
        pending = []
        rx_end = rx_at + len(x)
        for tone_event in self._pending_early_tones:
            matching = next((frame for frame in info_frames
                             if tone_event.at < frame.complete_at
                             and tone_event.complete_at >= frame.at), None)
            if matching is not None:
                if tone_event.at < matching.at:
                    early_tone_events.append(tone_event)
                elif tone_event.at < matching.complete_at:
                    # The candidate starts inside the validated DPSK frame.
                    reset_tone_after_info = True
                else:
                    late_tone_events.append(tone_event)
            elif any(0 <= tone_event.at - frame.complete_at
                     <= 400 * FS // 1000 for frame in info_frames):
                # A validated INFO0d/INFO1d immediately followed by Tone B
                # resolves the 1200-Hz ambiguity in the other direction: this
                # is the post-frame tone, not carrier inside the DPSK frame.
                # Route it with the frame in this batch instead of retaining
                # it for the full 400 ms pre-frame search window. The analog
                # caller's response depends on this distinction and deadline.
                late_tone_events.append(tone_event)
            elif rx_end - tone_event.complete_at >= 400 * FS // 1000:
                # The frame scanner searches every 200 ms and a complete
                # INFO0d is about 120 ms; this hold lets it reject a carrier
                # run before treating it as a pre-INFO0d Tone B.
                early_tone_events.append(tone_event)
            else:
                pending.append(tone_event)
        self._pending_early_tones = pending
        # A tone that predates INFO0d must reach the controller first so the
        # subsequent valid frame produces the bit-28 acknowledgement.
        early_tone_events.sort(
            key=lambda event: (event.at, event.complete_at, event.kind))
        early_tone_ids = {id(event) for event in early_tone_events}
        remainder = filtered + late_tone_events
        remainder.sort(key=lambda event: (event.complete_at, event.at, event.kind))
        detected = early_tone_events + remainder
        actions = []
        for event in detected:
            # A pre-INFO tone candidate can be held for 400 ms while the INFO
            # scanner decides whether the same 1200 Hz carrier was actually
            # DPSK. During that hold the controller may already have accepted
            # INFO0d, detected the ensuing Tone B, and advanced past the
            # frame-wait state. Do not reinterpret that old candidate using
            # the *new* state (e.g. as tone_b_after_probes).
            stale_early_tone = (
                id(event) in early_tone_ids
                and self._last_routed_tone_b_rx_at is not None
                and event.at <= self._last_routed_tone_b_rx_at)
            routed = None if stale_early_tone else self._route(event)
            name = None if routed is None else routed[0]
            event_sample = None if routed is None else routed[1]
            media_at = (None if event_sample is None else
                        int(self.rx_to_media(event_sample)))
            trace = {
                "event": event.kind,
                "rx_at": int(event.at),
                "complete_at": int(event.complete_at),
                "state": self.controller.state,
                "routed": name,
                "rx_event_sample": event_sample,
                "media_event_sample": media_at,
                "tx_now": int(tx_now),
                "fields": dict(event.fields),
            }
            if stale_early_tone:
                trace["discarded_reason"] = "stale pre-INFO tone candidate"
            if name is not None:
                try:
                    payload = dict(event.fields)
                    if name in ("l1_complete", "l2_complete"):
                        # Keep protocol duration checks in the actual receive
                        # sample domain.  `media_at` below remains the common
                        # scheduling clock for generated TX actions.
                        payload["receive_sample_at"] = int(event_sample)
                    produced = self.controller.feed(name, media_at, now=tx_now,
                                                     payload=payload)
                except (TypeError, ValueError, RuntimeError, KeyError, IndexError) as exc:
                    trace["error"] = str(exc)
                    self.event_trace.append(trace)
                    if len(self.event_trace) > 128:
                        del self.event_trace[:-128]
                    raise
                self._record_deadline(name, media_at, tx_now, produced)
                if name == "tone_b":
                    self._last_routed_tone_b_rx_at = int(event.at)
                trace["state_after"] = self.controller.state
                trace["actions"] = [action.signal for action in produced]
                actions.extend(produced)
            self.event_trace.append(trace)
            if len(self.event_trace) > 128:
                del self.event_trace[:-128]
        if reset_tone_after_info:
            self.signals.reset_tone_detector()
        return actions

    def feed_l2_echo_complete(self, *, rx_at: int,
                              tx_now: int) -> list[AnaloguePhase2Action]:
        if not self.started:
            raise RuntimeError("start the analogue Phase 2 adapter first")
        media_at = int(self.rx_to_media(rx_at))
        actions = self.controller.feed("l2_echo_complete", media_at, now=tx_now)
        self._record_deadline("l2_echo_complete", media_at, tx_now, actions)
        return actions

    def feed_l2_window_end(self, *, media_at: int,
                           tx_now: int) -> list[AnaloguePhase2Action]:
        if not self.started:
            raise RuntimeError("start the analogue Phase 2 adapter first")
        return self.controller.feed("l2_window_end", media_at, now=tx_now)

    def tick(self, *, media_now: int) -> list[AnaloguePhase2Action]:
        """Pump controller deadlines from the caller's transmit media clock."""
        if not self.started:
            raise RuntimeError("start the analogue Phase 2 adapter first")
        actions = self.controller.tick(media_now)
        self._record_deadline("timer", media_now, media_now, actions)
        return actions

    def _expected_signal(self) -> str:
        state = self.controller.state
        if state in (self.controller.WAIT_INFO0D,
                     self.controller.WAIT_TONE_B,
                     self.controller.WAIT_TONE_B_REVERSAL,
                     self.controller.WAIT_RECOVERY_TONE_B,
                     self.controller.WAIT_INFO1D,
                     self.controller.WAIT_TONE_B_REVERSAL_AFTER_PROBES):
            if state == self.controller.WAIT_TONE_B_REVERSAL_AFTER_PROBES:
                return "tone_b_probe"
            return "tone_b"
        if state in (self.controller.TX_LOCAL_PROBES,
                     self.controller.WAIT_TONE_B_AFTER_PROBES):
            # The caller transmits L1/L2 while listening for both Tone B and
            # the returned local hybrid echo of its own probing waveform.
            return "tone_b_probe"
        if state == self.controller.RX_REMOTE_PROBES:
            return "probe"
        return "none"

    def _route(self, event: AnaloguePhase2RxEvent):
        state = self.controller.state
        if event.kind == "info0d" and state in (
                self.controller.WAIT_INFO0D, self.controller.WAIT_TONE_B,
                self.controller.WAIT_TONE_B_REVERSAL):
            return "info0d", event.complete_at
        if event.kind == "info1d" and state == self.controller.WAIT_INFO1D:
            return "info1d", event.complete_at
        if event.kind == "tone_b":
            if state in (self.controller.WAIT_INFO0D,
                         self.controller.WAIT_TONE_B,
                         self.controller.WAIT_RECOVERY_TONE_B,
                         self.controller.WAIT_INFO1D):
                return "tone_b", event.at
            if state in (self.controller.TX_LOCAL_PROBES,
                         self.controller.WAIT_TONE_B_AFTER_PROBES):
                return "tone_b_after_probes", event.at
            if (state == self.controller.WAIT_TONE_B_REVERSAL_AFTER_PROBES
                    and self.controller.accept_early_peer_recovery):
                # A bounded paired recovery can have no observable continuous
                # Tone B immediately before the peer's reversal: the caller's
                # received carrier went silent during L1/L2. The detector then
                # reports the valid returning phase as a fresh stable Tone B.
                # In this timed-out state, accept that carrier as the expected
                # recovery reversal; ordinary policy remains unchanged.
                return "tone_b_reversal_after_probes", event.at
        if event.kind == "tone_b_reversal":
            if state == self.controller.WAIT_TONE_B_REVERSAL:
                return "tone_b_reversal", event.at
            if (state == self.controller.WAIT_TONE_B_AFTER_PROBES
                    and self.controller.accept_early_peer_recovery):
                # V.90 §9.2.1.2.4 has the digital peer send a Tone-B
                # reversal, L1 and L2 after it misses the caller's Tone-A
                # reversal. Route the reversal into RX_REMOTE_PROBES; the
                # controller then validates the 160 ms L1 and <=500 ms L2.
                # Saved captured peer traces showed this sequence was previously
                # ignored until its L2 was mislabeled as local echo.
                return "early_peer_recovery_tone_b_reversal", event.at
            if state == self.controller.WAIT_TONE_B_REVERSAL_AFTER_PROBES:
                return "tone_b_reversal_after_probes", event.at
        if event.kind in ("l1_complete", "l2_complete"):
            if state == self.controller.RX_REMOTE_PROBES:
                return event.kind, event.at
            if (event.kind == "l2_complete" and state in (
                    self.controller.TX_LOCAL_PROBES,
                    self.controller.WAIT_TONE_B_AFTER_PROBES)):
                return "l2_echo_complete", event.at
        return None

    def _record_deadline(self, event: str, detected_at: int, tx_now: int,
                         actions: list[AnaloguePhase2Action]) -> None:
        for action in actions:
            if action.latest is not None and tx_now > action.latest:
                self.deadline_misses.append(AnaloguePhase2DeadlineMiss(
                    event, detected_at, action.latest, tx_now))
