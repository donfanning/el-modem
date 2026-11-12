"""Sample-timeline renderer for the V.90 digital-side Phase 2 actions.

The renderer emits mono 8 kHz linear PCM. It does not packetize RTP or convert
samples to G.711; the existing media path owns those operations. Actions that
depend on finite-frame completion are resolved from the actual generated frame
length, while late actions are surfaced in ``late_actions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from v34_info import FS, CARRIER
from v90_digital_phase2 import V90DigitalPhase2Signals
from v90_digital_phase2_control import Phase2Action


@dataclass(frozen=True)
class LatePhase2Action:
    signal: str
    requested_at: int
    rendered_at: int
    latest: int | None


@dataclass(frozen=True)
class RenderedHandoff:
    at: int
    signal: str
    mode: int | None


class V90DigitalPhase2Transmitter:
    """Render controller actions continuously across arbitrary PCM chunk sizes.

    ``probe_builder`` receives ``(duration_seconds, rms)`` and returns linear
    PCM, typically the project's spec-derived ``v34_call.probing_signal``.
    Both its waveform source and level are explicit so this renderer does not
    silently choose a profile. The output must still pass through the selected
    PCMU/PCMA RTP path for V.90 digital downstream.
    """

    def __init__(self, *, signals: V90DigitalPhase2Signals,
                 probe_builder: Callable[[float, float], np.ndarray],
                 tone_b_peak: float, probe_rms: float,
                 start_at: int = 0):
        if not callable(probe_builder):
            raise TypeError("probe_builder must generate the L1/L2 waveform")
        if tone_b_peak <= 0 or probe_rms <= 0 or start_at < 0:
            raise ValueError("signal levels must be positive and start_at nonnegative")
        self.signals = signals
        self.probe_builder = probe_builder
        self.tone_b_peak = float(tone_b_peak)
        self.probe_rms = float(probe_rms)
        self.cursor = int(start_at)
        self._events: list[tuple[int, int, str, object]] = []
        self._sequence = 0
        self._mode = "silence"
        self._tone_b_offset = 0.0
        self._tone_b_reversed = False
        self._array: np.ndarray | None = None
        self._array_pos = 0
        self._array_name: str | None = None
        self._completion: dict[str, int] = {}
        self._last_info0d_complete: int | None = None
        self._last_finite_end = self.cursor
        self.late_actions: list[LatePhase2Action] = []
        self.handoffs: list[RenderedHandoff] = []

    def submit(self, actions: list[Phase2Action]) -> None:
        """Schedule a controller action batch on the absolute media timeline."""
        for action in actions:
            requested = self._resolve_time(action)
            at = requested
            if at < self.cursor:
                self.late_actions.append(LatePhase2Action(
                    action.signal, requested, self.cursor, action.latest))
                at = self.cursor
            elif action.latest is not None and at > action.latest:
                self.late_actions.append(LatePhase2Action(
                    action.signal, requested, at, action.latest))

            if action.signal in ("info0d", "info1d", "l1", "l2"):
                if at < self._last_finite_end:
                    raise ValueError("finite Phase 2 waveforms may not overlap")
                samples = self._make_array(action, at)
                if samples.ndim != 1 or not np.all(np.isfinite(samples)):
                    raise ValueError(f"{action.signal} generator returned invalid PCM")
                samples = np.asarray(samples, dtype=float)
                if not len(samples):
                    raise ValueError(f"{action.signal} generator returned no PCM")
                self._schedule(at, "array", (samples, action.signal))
                complete = at + len(samples)
                self._last_finite_end = complete
                self._schedule(complete, "silence", None)
                self._completion[f"{action.signal}-complete"] = complete
                if action.signal == "info0d":
                    self._last_info0d_complete = complete
            elif action.signal in ("tone_b", "reverse-tone-b", "silence",
                                   "retrain-timeout"):
                self._schedule(at, action.signal, None)
            elif action.signal.startswith("handoff-"):
                self.handoffs.append(RenderedHandoff(at, action.signal, action.value))
            else:
                raise ValueError(f"unsupported Phase 2 signal action {action.signal!r}")

    def render(self, sample_count: int) -> np.ndarray:
        """Render the next ``sample_count`` samples, retaining phase and state."""
        if not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        output = np.zeros(sample_count, dtype=float)
        out_pos = 0
        while out_pos < sample_count:
            self._apply_due_events()
            next_event = self._events[0][0] if self._events else self.cursor + sample_count - out_pos
            until_event = max(0, next_event - self.cursor)
            count = min(sample_count - out_pos, until_event)
            if count == 0:
                # An event may be scheduled exactly at the cursor.
                continue
            if self._mode == "tone_b":
                t = self.cursor + np.arange(count)
                phase = 2 * np.pi * CARRIER["call"] * t / FS + self._tone_b_offset
                if self._tone_b_reversed:
                    phase = phase + np.pi
                output[out_pos:out_pos + count] = self.tone_b_peak * np.cos(phase)
            elif self._mode == "array":
                if self._array is None:
                    raise RuntimeError("array mode has no active waveform")
                take = min(count, len(self._array) - self._array_pos)
                output[out_pos:out_pos + take] = self._array[self._array_pos:self._array_pos + take]
                self._array_pos += take
                if take < count:
                    # The generated finite signal ended before another queued
                    # action; silence is the defined inter-signal output.
                    self._mode = "silence"
                    self._array = None
                    self._array_name = None
            self.cursor += count
            out_pos += count
        self._apply_due_events()
        return output

    def _resolve_time(self, action: Phase2Action) -> int:
        if action.at is not None:
            return int(action.at)
        if action.after is None:
            raise ValueError(f"action {action.signal!r} has no time or completion dependency")
        if action.after == "current-info0d-complete":
            complete = self._last_info0d_complete
        else:
            complete = self._completion.get(action.after)
        if complete is None:
            raise ValueError(f"completion dependency {action.after!r} is unavailable")
        return complete

    def _make_array(self, action: Phase2Action, at: int) -> np.ndarray:
        if action.signal == "info0d":
            self.signals.set_phase_at_sample(at)
            return self.signals.modulate_info0d(ack_info0a=action.value)
        if action.signal == "info1d":
            self.signals.set_phase_at_sample(at)
            return self.signals.modulate_info1d()
        if action.signal == "l1":
            # V.90 §8.2.4 refers line probes to V.34 §10.1.2.4: L1 is
            # transmitted 6 dB above nominal, while L2 is at nominal power.
            # probe_rms denotes the nominal/L2 RMS level, as it does in the
            # analogue-caller renderer. A factor of two is 6.02 dB.
            result = self.probe_builder(.160, self.probe_rms * 2.0)
            if len(result) != int(.160 * FS):
                raise ValueError("L1 probe generator must return exactly 160 ms of PCM")
            return result
        if action.signal == "l2":
            duration = int(action.value or 0)
            if duration <= 0:
                raise ValueError("L2 action requires a positive sample duration")
            result = self.probe_builder(duration / FS, self.probe_rms)
            if len(result) != duration:
                raise ValueError("L2 probe generator returned the wrong duration")
            return result
        raise AssertionError("unreachable waveform action")

    def _schedule(self, at: int, kind: str, value: object) -> None:
        self._sequence += 1
        self._events.append((at, self._sequence, kind, value))
        self._events.sort(key=lambda event: (event[0], event[1]))

    def _apply_due_events(self) -> None:
        while self._events and self._events[0][0] <= self.cursor:
            at, _, kind, value = self._events.pop(0)
            if kind == "array":
                samples, name = value
                self._mode = "array"
                self._array = samples
                self._array_pos = 0
                self._array_name = name
            elif kind == "tone_b":
                self._mode = "tone_b"
            elif kind == "reverse-tone-b":
                self._tone_b_reversed = not self._tone_b_reversed
            else:
                self._mode = "silence"
                self._array = None
                self._array_pos = 0
                self._array_name = None
