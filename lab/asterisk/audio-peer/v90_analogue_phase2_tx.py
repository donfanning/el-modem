"""Sample-timeline transmitter for V.90 analogue-caller Phase 2 actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

import v34_info
import v90_info_reference as v90
from v90_analogue_caller_phase2 import AnaloguePhase2Action


FS = 8000


@dataclass(frozen=True)
class LateAnalogueAction:
    signal: str
    requested_at: int
    rendered_at: int
    latest: int | None


@dataclass(frozen=True)
class AnalogueTxHandoff:
    at: int
    signal: str
    mode: int | None


class V90AnaloguePhase2Transmitter:
    """Render caller INFO0a/Tone A/probes/INFO1a into 8 kHz linear PCM.

    The caller supplies the V.90 or V.34 INFO1a bit builder and the V.34 probe
    generator. Output timing is absolute-sample based; RTP packetization and
    G.711 conversion are deliberately outside this renderer.
    """

    def __init__(self, *, info0a_fields: dict,
                 info1a_builder: Callable[[int], list[int]],
                 probe_builder: Callable[[float, float], np.ndarray],
                 nominal_peak: float, probe_rms: float,
                 start_at: int = 0):
        if not callable(info1a_builder) or not callable(probe_builder):
            raise TypeError("INFO1a and probe waveform builders are required")
        if (not math.isfinite(nominal_peak) or not math.isfinite(probe_rms)
                or nominal_peak <= 0 or probe_rms <= 0
                or isinstance(start_at, bool) or not isinstance(start_at, int)
                or start_at < 0):
            raise ValueError("signal levels must be positive and start_at nonnegative")
        self.info0a_fields = dict(info0a_fields)
        self.info1a_builder = info1a_builder
        self.probe_builder = probe_builder
        self.nominal_peak = float(nominal_peak)
        self.probe_rms = float(probe_rms)
        self.cursor = int(start_at)
        self._events: list[tuple[int, int, str, object, str | None]] = []
        self._sequence = 0
        self._mode = "silence"
        self._tone_a_reversed = False
        self._tone_a_phase_offset = 0.0
        self._array: np.ndarray | None = None
        self._array_pos = 0
        self._completion: dict[str, int] = {}
        self._completion_tags: dict[str, str | None] = {}
        self._finite_intervals: list[dict] = []
        self._last_finite_end = self.cursor
        self.late_actions: list[LateAnalogueAction] = []
        self.handoffs: list[AnalogueTxHandoff] = []

    def submit(self, actions: list[AnaloguePhase2Action]) -> None:
        """Schedule a controller action batch on the absolute media timeline."""
        for action in actions:
            requested = self._resolve_time(action)
            at = requested
            if at < self.cursor:
                self.late_actions.append(LateAnalogueAction(
                    action.signal, requested, self.cursor, action.latest))
                at = self.cursor
            elif action.latest is not None and at > action.latest:
                self.late_actions.append(LateAnalogueAction(
                    action.signal, requested, at, action.latest))
            if action.cancel_tag is not None:
                # A valid Tone-B reversal supersedes pending INFO0a retries
                # at the reversal boundary.  Resume the (now reversed) Tone A
                # immediately; waiting for L1 to cancel the retry would leave
                # stale INFO0a on the line across this protocol transition.
                resume_signal = ("tone_a" if action.signal == "reverse-tone-a"
                                 else "silence")
                self._cancel_tagged_actions(
                    action.cancel_tag, at, resume_signal=resume_signal)

            if action.signal in ("info0a", "info1a", "l1", "l2"):
                if at < self._last_finite_end:
                    raise ValueError("finite caller Phase 2 signals may not overlap")
                samples = np.asarray(self._make_array(action, at), dtype=float)
                if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)):
                    raise ValueError(f"{action.signal} builder returned invalid PCM")
                self._schedule(at, "array", samples, tag=action.tag)
                complete = at + len(samples)
                self._last_finite_end = complete
                self._completion[f"{action.signal}-complete"] = complete
                self._completion_tags[f"{action.signal}-complete"] = action.tag
                self._finite_intervals.append({
                    "start": at, "end": complete, "tag": action.tag,
                })
                self._schedule(complete, "silence", None, tag=action.tag)
            elif action.signal in ("tone_a", "reverse-tone-a", "silence"):
                self._schedule(at, action.signal, None, tag=action.tag)
                self._completion[action.signal] = at
                self._completion_tags[action.signal] = action.tag
            elif action.signal.startswith("listen-") or action.signal == "recovery-required":
                self._completion[action.signal] = at
                self._completion_tags[action.signal] = action.tag
            elif action.signal.startswith("handoff-"):
                self.handoffs.append(AnalogueTxHandoff(at, action.signal, action.value))
            else:
                raise ValueError(f"unsupported analogue Phase 2 action {action.signal!r}")

    def render(self, sample_count: int) -> np.ndarray:
        """Render the next samples, retaining oscillator and signal state."""
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        output = np.zeros(sample_count, dtype=float)
        out_pos = 0
        while out_pos < sample_count:
            self._apply_due_events()
            next_event = self._events[0][0] if self._events else self.cursor + sample_count - out_pos
            count = min(sample_count - out_pos, max(0, next_event - self.cursor))
            if count == 0:
                continue
            if self._mode == "tone_a":
                t = self.cursor + np.arange(count)
                phase = 2 * np.pi * 2400.0 * t / FS
                if self._tone_a_reversed:
                    phase += np.pi
                output[out_pos:out_pos + count] = (
                    self.nominal_peak * 10 ** (-1 / 20) * np.cos(phase)
                    + self.nominal_peak * 10 ** (-7 / 20)
                    * np.cos(2 * np.pi * 1800.0 * t / FS))
            elif self._mode == "array":
                if self._array is None:
                    raise RuntimeError("array mode has no active waveform")
                take = min(count, len(self._array) - self._array_pos)
                output[out_pos:out_pos + take] = self._array[self._array_pos:self._array_pos + take]
                self._array_pos += take
                if take < count:
                    self._mode = "silence"
                    self._array = None
                    self._array_pos = 0
            self.cursor += count
            out_pos += count
        self._apply_due_events()
        return output

    def _make_array(self, action: AnaloguePhase2Action, at: int) -> np.ndarray:
        if action.signal == "info0a":
            fields = dict(self.info0a_fields)
            if action.value is not None:
                fields["ack_info0d"] = int(action.value)
            fields.pop("v92", None)
            bits = v90.build_info0a(**fields)
            modulator = v34_info.InfoModulator("answer", guard=True)
            start_phase = 2 * np.pi * 2400.0 * at / FS
            guard_phase = 2 * np.pi * 1800.0 * at / FS
            self._tone_a_phase_offset = np.pi * (sum(bits) & 1)
            return modulator.modulate(bits, start_phase=start_phase,
                                      guard_start_phase=guard_phase)
        if action.signal == "info1a":
            if (action.value is None or isinstance(action.value, bool)
                    or not 0 <= int(action.value) <= 6):
                raise ValueError("INFO1a requires a negotiated mode in 0..6")
            bits = list(self.info1a_builder(int(action.value)))
            if len(bits) != 70 or any(bit not in (0, 1) for bit in bits):
                raise ValueError("INFO1a builder must return 70 binary frame bits")
            modulator = v34_info.InfoModulator("answer", guard=True)
            start_phase = (2 * np.pi * 2400.0 * at / FS
                           + self._tone_a_phase_offset
                           + (np.pi if self._tone_a_reversed else 0.0))
            guard_phase = 2 * np.pi * 1800.0 * at / FS
            return modulator.modulate(bits, start_phase=start_phase,
                                      guard_start_phase=guard_phase)
        if action.signal == "l1":
            result = np.asarray(self.probe_builder(.160, self.probe_rms * 2.0), dtype=float)
            if len(result) != int(.160 * FS):
                raise ValueError("L1 generator must return exactly 160 ms")
            return result
        if action.signal == "l2":
            duration = int(action.value or 0)
            if duration <= 0:
                raise ValueError("L2 action requires a positive sample duration")
            result = np.asarray(self.probe_builder(duration / FS, self.probe_rms), dtype=float)
            if len(result) != duration:
                raise ValueError("L2 generator returned the wrong duration")
            return result
        raise AssertionError("unreachable finite signal")

    def _resolve_time(self, action: AnaloguePhase2Action) -> int:
        if action.at is not None:
            return int(action.at)
        if action.after is None:
            raise ValueError(f"action {action.signal!r} has no time or completion dependency")
        complete = self._completion.get(action.after)
        if complete is None:
            raise ValueError(f"completion dependency {action.after!r} is unavailable")
        return complete

    def _schedule(self, at: int, kind: str, value: object,
                  *, tag: str | None = None) -> None:
        self._sequence += 1
        self._events.append((at, self._sequence, kind, value, tag))
        self._events.sort(key=lambda event: (event[0], event[1]))

    def _cancel_tagged_actions(self, tag: str, at: int, *,
                               resume_signal: str = "silence") -> None:
        """Retire obsolete tagged output at a protocol progression boundary."""
        if resume_signal not in ("silence", "tone_a"):
            raise ValueError("tag cancellation may resume only silence or Tone A")
        truncate_active = any(
            interval["tag"] == tag
            and interval["start"] < at < interval["end"]
            for interval in self._finite_intervals
        )
        self._events = [event for event in self._events
                        if event[4] != tag or event[0] < at]
        kept = []
        for interval in self._finite_intervals:
            if interval["tag"] != tag:
                kept.append(interval)
                continue
            if interval["start"] >= at:
                continue
            interval = dict(interval)
            interval["end"] = min(interval["end"], at)
            kept.append(interval)
        self._finite_intervals = kept
        self._last_finite_end = max(
            (interval["end"] for interval in kept), default=self.cursor)
        for name, owner in list(self._completion_tags.items()):
            # Do not rewrite an already completed retry's timestamp.  Only a
            # still-pending finite action has its effective completion moved
            # by this cancellation boundary.
            if owner == tag:
                if self._completion.get(name, at + 1) > at:
                    self._completion[name] = at
                self._completion_tags[name] = None
        if truncate_active or resume_signal == "tone_a":
            # Resume the protocol's continuous signal at the exact boundary,
            # including when a retry ended just before this boundary; the
            # queued reverse-tone-a action follows it at the same sample.
            self._schedule(at, resume_signal, None)

    def _apply_due_events(self) -> None:
        while self._events and self._events[0][0] <= self.cursor:
            _, _, kind, value, _ = self._events.pop(0)
            if kind == "array":
                self._mode = "array"
                self._array = value
                self._array_pos = 0
            elif kind == "tone_a":
                self._mode = "tone_a"
            elif kind == "reverse-tone-a":
                self._tone_a_reversed = not self._tone_a_reversed
            else:
                self._mode = "silence"
                self._array = None
                self._array_pos = 0
