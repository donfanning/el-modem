"""Bridge Phase 2 PCM receive events into the V.90 digital-side controller.

This adapter deliberately does not render output or infer a receive/transmit
clock relationship. The caller supplies both: ``rx_to_media`` maps absolute
receive sample positions to the controller's media timeline, and ``tx_now`` is
the current position on that same timeline. A discontinuous receive stream is
rejected so packet loss cannot be hidden as ordinary contiguous audio.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from v90_digital_phase2_control import Phase2Action, V90DigitalPhase2Controller
from v90_digital_phase2_rx import (
    Phase2RxEvent,
    V90DigitalPhase2FrameRx,
    V90DigitalPhase2SignalRx,
)


@dataclass(frozen=True)
class Phase2DeadlineMiss:
    event: str
    detected_at: int
    response_deadline: int
    tx_now: int


class V90DigitalPhase2Adapter:
    """Translate detected PCM events to controller events and actions.

    ``rx_origin`` is the absolute receive sample index of the first chunk.
    ``rx_to_media`` converts that receive clock into the controller's common
    media clock; a caller must not assume that the two clocks share an origin.
    The adapter can be started only once and accepts contiguous 8 kHz chunks.
    """

    def __init__(self, controller: V90DigitalPhase2Controller, *,
                 rx_to_media: Callable[[int], int], trace_events: bool = False):
        if not callable(rx_to_media):
            raise TypeError("rx_to_media must map receive sample to media sample")
        self.controller = controller
        self.rx_to_media = rx_to_media
        self.frames = V90DigitalPhase2FrameRx()
        self.signals = V90DigitalPhase2SignalRx()
        self.rx_origin: int | None = None
        self.rx_next: int | None = None
        self.started = False
        self.deadline_misses: list[Phase2DeadlineMiss] = []
        self.trace_events = bool(trace_events)
        self.event_trace: list[dict] = []
        self.signal_trace: list[dict] = []
        self._pending_tone_a_reversals: list[Phase2RxEvent] = []
        self.info1a_complete_rx_sample: int | None = None

    def start(self, media_at: int) -> list[Phase2Action]:
        if self.started:
            raise RuntimeError("Phase 2 adapter already started")
        self.started = True
        return self.controller.start(media_at)

    def start_retrain_continuation(self, media_at: int, *, reversal_at: int,
                                   info0a: dict) -> list[Phase2Action]:
        """Resume at the retained V.90 Phase 2 Tone A reversal boundary."""
        if self.started:
            raise RuntimeError("Phase 2 adapter already started")
        self.started = True
        return self.controller.start_retrain_continuation(
            reversal_at, now=media_at, info0a=info0a)

    def feed(self, samples, *, rx_at: int, tx_now: int) -> list[Phase2Action]:
        """Feed one contiguous receive chunk and return newly scheduled actions."""
        if not self.started:
            raise RuntimeError("start the Phase 2 adapter before feeding audio")
        if rx_at < 0 or tx_now < 0:
            raise ValueError("sample positions must be nonnegative")
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("receive audio must contain finite samples")
        if not len(x):
            return []
        if self.rx_origin is None:
            self.rx_origin = rx_at
        if self.rx_next is not None and rx_at != self.rx_next:
            raise ValueError("receive sample discontinuity; restart with an explicit gap policy")
        self.rx_next = rx_at + len(x)

        expected = self._expected_signal()
        trace_started = time.thread_time_ns() if self.trace_events else 0
        signal_events = self.signals.feed(x, expect=expected)
        signal_cpu_ns = time.thread_time_ns() - trace_started if self.trace_events else 0
        trace_row = None
        if self.trace_events:
            phasors = self.signals._tone_phasor
            phase_pair = None
            if len(phasors) >= 3:
                phase_pair = float((phasors[-1] * np.conj(phasors[-3])).real)
            trace_row = {
                "rx_at": int(rx_at),
                "samples": int(len(x)),
                "expected": expected,
                "tone_a_active": bool(self.signals._tone_a_active),
                "stable_tail": [bool(value)
                                for value in self.signals._tone_stable[-8:]],
                "phase_pair_real": phase_pair,
                "last_reversal": self.signals._last_a_reversal,
                "events": [event.kind for event in signal_events],
            }
        detected = signal_events
        frame_started = time.thread_time_ns() if self.trace_events else 0
        detected.extend(self.frames.feed(x, kinds=self._expected_frame_kinds()))
        frame_cpu_ns = time.thread_time_ns() - frame_started if self.trace_events else 0
        # Detection completion order, rather than list append order, preserves
        # state transitions when a frame and a waveform event finish together.
        detected.sort(key=lambda event: (event.complete_at, event.at, event.kind))

        actions: list[Phase2Action] = []
        route_started = time.thread_time_ns() if self.trace_events else 0
        for event in detected:
            if (event.kind == "tone_a_reversal"
                    and self.controller.info0a is None):
                # The phase detector can confirm a real reversal before the
                # CRC frame scanner completes INFO0a. Hold it until the frame
                # establishes whether this was a reversal inside the DPSK
                # message or the subsequent Tone A reversal.
                self._pending_tone_a_reversals.append(event)
                if len(self._pending_tone_a_reversals) > 16:
                    del self._pending_tone_a_reversals[:-16]
                continue

            actions.extend(self._deliver(event, tx_now))
            if event.kind == "info0a" and self._pending_tone_a_reversals:
                held, self._pending_tone_a_reversals = (
                    self._pending_tone_a_reversals, [])
                for reversal in held:
                    # INFO0a itself is DPSK on the Tone-A carrier. Only a
                    # transition timestamped at or after the CRC-valid frame
                    # end can be the subsequent Tone A phase reversal.
                    if reversal.at >= event.complete_at:
                        actions.extend(self._deliver(reversal, tx_now))
        if trace_row is not None:
            trace_row.update(
                signal_cpu_ms=signal_cpu_ns / 1_000_000.0,
                frame_cpu_ms=frame_cpu_ns / 1_000_000.0,
                route_cpu_ms=(time.thread_time_ns() - route_started) / 1_000_000.0,
            )
            self.signal_trace.append(trace_row)
            if len(self.signal_trace) > 320:
                del self.signal_trace[:-320]
        return actions

    def _expected_frame_kinds(self) -> tuple[str, ...]:
        """Search only frame types that can advance the current controller state."""
        if self.controller.state in (
                self.controller.WAIT_INFO0A, self.controller.WAIT_A_REV2,
                self.controller.RX_PROBE):
            return ("info0",)
        if self.controller.state in (self.controller.WAIT_INFO1A,
                                      self.controller.WAIT_INFO1A_RETRAIN):
            return ("info1a",)
        return ()

    def _deliver(self, event: Phase2RxEvent, tx_now: int) -> list[Phase2Action]:
        state_before = self.controller.state
        if event.kind == "info1a" and self.rx_origin is not None:
            self.info1a_complete_rx_sample = (
                int(self.rx_origin) + int(event.complete_at))
        media_at = int(self.rx_to_media(self.rx_origin + event.at))
        # TX_PROBE may overlap the tail of the caller's previous L2 during a
        # retrain. A comb interval completed before our scheduled local L2 is
        # not its echo; discard it and reopen the detector for the actual
        # post-transmit return instead of failing Phase 2 on a stale burst.
        if (event.kind == "l2_complete"
                and self.controller.state == self.controller.TX_PROBE
                and self.controller.l2_started_at is not None
                and media_at < self.controller.l2_started_at):
            self.signals.reset_probe_detection()
            routed = None
        else:
            routed = self._route(event)
        produced: list[Phase2Action] = []
        event_name = None
        if routed is not None:
            event_name, payload = routed
            produced = self.controller.feed(event_name, media_at, now=tx_now,
                                            payload=payload)
            if (event_name == "info0a"
                    and state_before == self.controller.RX_PROBE
                    and self.controller.state == self.controller.WAIT_A_REV2):
                # Repeated INFO0a means the peer has not synchronized to the
                # ranging attempt. Discard any partial L1/L2 detector window
                # before restarting Tone-A/Tone-B ranging.
                self.signals.reset_probe_detection()
            if (event_name == "retrain_tone_a_reversal"
                    or (event_name == "tone_a_reversal_after_l2"
                        and self.controller.retrain_response_started)):
                # The caller may begin its local L1 after hearing our first
                # reversed Tone B, while we still await its second A reversal.
                # Reset before that probe can arrive. The post-probe reversal
                # also opens a new L1/L2 echo observation interval.
                self.signals.reset_probe_detection()
            self._record_deadline(event_name, media_at, tx_now, produced)
        if self.trace_events:
            self.event_trace.append({
                "event": event.kind,
                "rx_at": int(event.at),
                "complete_at": int(event.complete_at),
                "media_at": media_at,
                "routed": None if routed is None else routed[0],
                "state": state_before,
                "state_after": self.controller.state,
                "tx_now": int(tx_now),
                "actions": [action.signal for action in produced],
            })
            if len(self.event_trace) > 128:
                del self.event_trace[:-128]
        return produced

    def feed_external_event(self, event: str, *, rx_at: int, tx_now: int,
                            payload: dict | None = None) -> list[Phase2Action]:
        """Feed a separately measured media event, such as returned L2 echo."""
        if not self.started:
            raise RuntimeError("start the Phase 2 adapter before feeding events")
        if event not in ("l2_echo_complete",):
            raise ValueError("only externally measured L2 echo events are supported")
        media_at = int(self.rx_to_media(rx_at))
        actions = self.controller.feed(event, media_at, now=tx_now, payload=payload)
        self._record_deadline(event, media_at, tx_now, actions)
        return actions

    def _expected_signal(self) -> str:
        state = self.controller.state
        if state in (self.controller.WAIT_INFO0A, self.controller.WAIT_A_REV2,
                     self.controller.WAIT_A_AFTER_PROBE, self.controller.WAIT_A_REV3):
            return "tone_a"
        if state == self.controller.WAIT_RETRAIN_A_REV2:
            return "all"
        if state in (self.controller.WAIT_INFO1A_RETRAIN,
                     self.controller.WAIT_RETRAIN_TONE_B,
                     self.controller.WAIT_RETRAIN_A_REVERSAL):
            return "tone_a"
        if state == self.controller.RX_PROBE:
            return "probe"
        if state == self.controller.TX_PROBE:
            # The digital answerer is sending L1/L2 while simultaneously
            # listening for their returned copy and the caller's Tone A.
            return "all"
        return "none"

    def _route(self, event: Phase2RxEvent) -> tuple[str, dict] | None:
        state = self.controller.state
        if event.kind == "info0a":
            if state in (self.controller.WAIT_INFO0A, self.controller.WAIT_A_REV2,
                         self.controller.RX_PROBE,
                         self.controller.WAIT_RETRAIN_A_REV2):
                return event.kind, event.fields
            return None
        if event.kind == "info1a":
            if state in (self.controller.WAIT_INFO1A,
                         self.controller.WAIT_INFO1A_RETRAIN):
                return event.kind, event.fields
            return None
        if event.kind == "tone_a_reversal":
            # INFO0a is phase-modulated on the same carrier; phase changes
            # inside that frame must not be counted as Tone A reversals.
            if self.controller.info0a is None:
                return None
            if state in (self.controller.WAIT_INFO0A, self.controller.WAIT_A_REV2,
                         self.controller.WAIT_RETRAIN_A_REV2):
                return event.kind, event.fields
            if state in (self.controller.WAIT_RETRAIN_TONE_B,
                         self.controller.WAIT_RETRAIN_A_REVERSAL):
                return "retrain_tone_a_reversal", event.fields
            if state == self.controller.WAIT_A_REV3:
                return "tone_a_reversal_after_l2", event.fields
            return None
        if event.kind == "tone_a":
            if state in (self.controller.WAIT_A_AFTER_PROBE, self.controller.TX_PROBE):
                return event.kind, event.fields
            if state == self.controller.WAIT_INFO1A_RETRAIN:
                return event.kind, event.fields
            return None
        if event.kind in ("l1_complete", "l2_complete"):
            if state == self.controller.RX_PROBE:
                return event.kind, event.fields
            if event.kind == "l2_complete" and state == self.controller.TX_PROBE:
                return "l2_echo_complete", event.fields
        return None

    def _record_deadline(self, event: str, detected_at: int, tx_now: int,
                         actions: list[Phase2Action]) -> None:
        for action in actions:
            if action.latest is not None and tx_now > action.latest:
                self.deadline_misses.append(Phase2DeadlineMiss(
                    event=event,
                    detected_at=detected_at,
                    response_deadline=action.latest,
                    tx_now=tx_now,
                ))
