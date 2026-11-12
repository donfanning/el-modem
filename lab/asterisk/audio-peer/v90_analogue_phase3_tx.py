"""Sample-clocked V.90 analogue-caller Phase 3 waveform transmitter.

The protocol controller owns event order; this module turns its initial
S/S-bar/PP/TRN/Ja actions into the continuous V.34-shaped caller waveform.
Receive-side S-transition qualification and the later S/S-bar responses are
owned by the caller's Phase 3 session and are not inferred here.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

import v34_tx
from v90_analogue_caller_phase3 import V90AnalogueCallerPhase3


class V90AnaloguePhase3Transmitter:
    """Render the caller's initial Phase 3 training and repeat Ja until stopped.

    ``start(at)`` honors the controller's post-INFO1a silence. ``read_pcm``
    advances an absolute 8 kHz media clock and can be called in arbitrary
    packet sizes. ``notify_sd_transition`` stops Ja at the receive event's
    sample position, records any already-transmitted lateness, and lets the
    controller enter its TRN1d receive state.
    """

    FS = 8000

    def __init__(self, controller: V90AnalogueCallerPhase3, *,
                 carrier: float, rms: float, preemphasis=None, span: int = 8):
        if not isinstance(controller, V90AnalogueCallerPhase3):
            raise TypeError("Phase 3 transmitter requires its caller controller")
        if (not math.isfinite(float(carrier)) or carrier <= 0
                or not math.isfinite(float(rms)) or rms <= 0):
            raise ValueError("carrier and RMS must be positive and finite")
        self.controller = controller
        self.carrier = float(carrier)
        self.rms = float(rms)
        self.preemphasis = preemphasis
        self.span = int(span)
        if self.span < 1:
            raise ValueError("pulse-shaper span must be positive")
        self.qam: v34_tx.StreamQam | None = None
        self.cursor: int | None = None
        self.qam_start: int | None = None
        self._started = False
        self._ja_stopped = False
        self._stop_at: int | None = None
        self.late_stop_samples = 0
        self._segments: list[dict] = []
        self._completed_actions: deque[dict] = deque()

    def start(self, at: int) -> list:
        if self._started:
            raise RuntimeError("V.90 caller Phase 3 transmitter already started")
        actions = self.controller.start(at)
        if ([action.signal for action in actions[:6]]
                != ["silence", "S", "Sbar", "PP", "TRN", "Ja-repeat"]):
            raise RuntimeError("caller Phase 3 controller emitted an unexpected initial sequence")
        if actions[0].duration_samples != 70 * self.FS // 1000:
            raise RuntimeError("caller Phase 3 silence does not match the 70 ms requirement")

        self.cursor = int(at)
        self.qam_start = int(actions[1].at)
        self.qam = v34_tx.StreamQam(
            self.controller.baud, self.carrier, self.rms, span=self.span)
        self.qam.preemphasis = self.preemphasis

        s = v34_tx.s_signal(128)
        s_bar = v34_tx.s_signal(16, True)
        pp = v34_tx.pp_signal() * math.sqrt(2.0)
        trn, self._scrambler = v34_tx.trn_symbols(512, "answer")
        self._rotation = v34_tx.trn_last_rotation(trn, 4)
        self.qam.push(np.concatenate((s, s_bar, pp, trn)))
        self.qam.fill = self._fill_ja
        self._started = True
        return actions

    def _fill_ja(self, qam: v34_tx.StreamQam) -> None:
        bits = list(self.controller.ja_bits)
        symbols, self._rotation = v34_tx.bit_symbols(
            bits, "answer", self._scrambler, self._rotation, constellation=4)
        if not len(symbols):
            raise RuntimeError("caller Ja descriptor produced no symbols")
        qam.push(symbols)

    def read_pcm(self, sample_count: int) -> np.ndarray:
        if not self._started or self.cursor is None or self.qam_start is None:
            raise RuntimeError("caller Phase 3 transmitter has not started")
        if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
                or sample_count < 0):
            raise ValueError("sample count must be a nonnegative integer")
        output = np.zeros(sample_count, dtype=float)
        stop = self.cursor + sample_count
        active_start = max(self.cursor, self.qam_start)
        if active_start < stop and not self._ja_stopped:
            offset = active_start - self.cursor
            qam_end = stop
            if self._stop_at is not None:
                qam_end = min(qam_end, max(active_start, self._stop_at))
            count = qam_end - active_start
            if count:
                output[offset:offset + count] = self.qam.read(count)
            if self._stop_at is not None and stop >= self._stop_at:
                self._apply_ja_stop()
        for segment in self._segments:
            left = max(self.cursor, segment["start"])
            right = min(stop, segment["end"])
            if left < right:
                rendered = segment["qam"].read(right - left)
                offset = left - self.cursor
                output[offset:offset + len(rendered)] += rendered
            if not segment["completed"] and stop >= segment["end"]:
                segment["completed"] = True
                for signal in segment["signals"]:
                    self._completed_actions.append({
                        "signal": signal, "at": segment["end"]})
        self.cursor = stop
        self._segments = [segment for segment in self._segments
                          if segment["end"] > self.cursor]
        return output

    def schedule_actions(self, actions) -> None:
        """Queue controller S/S-bar actions as sample-timed QAM segments."""
        if not self._started or self.qam_start is None or self.cursor is None:
            raise RuntimeError("start caller Phase 3 before scheduling responses")
        selected = [action for action in actions
                    if action.signal in ("S", "Sbar")]
        index = 0
        while index < len(selected):
            first = selected[index]
            group = [first]
            end = first.at + first.duration_samples
            index += 1
            while (index < len(selected) and selected[index].at == end):
                group.append(selected[index])
                end += selected[index].duration_samples
                index += 1
            start = max(int(first.at), self.cursor)
            symbols = np.concatenate([
                v34_tx.s_signal(action.symbols, bar=action.signal == "Sbar")
                for action in group])
            qam = v34_tx.StreamQam(
                self.controller.baud, self.carrier, self.rms, span=self.span)
            qam.preemphasis = self.preemphasis
            qam.push(symbols)
            qam._matrix()
            phase_samples = start - self.qam_start
            qam._carrier *= np.exp(
                2j * math.pi * self.carrier * phase_samples / self.FS)
            self._segments.append({
                "start": start,
                "end": start + sum(action.duration_samples for action in group),
                "qam": qam,
                "signals": [action.signal for action in group],
                "completed": False,
            })
        self._segments.sort(key=lambda item: item["start"])

    def drain_completed_actions(self) -> list[dict]:
        completed = list(self._completed_actions)
        self._completed_actions.clear()
        return completed

    def notify_sd_transition(self, at: int) -> list:
        """Stop repeated Ja when the caller detects answerer Sd -> S-bar-d."""
        if not self._started or self.cursor is None or self.qam_start is None:
            raise RuntimeError("caller Phase 3 transmitter has not started")
        if self._ja_stopped:
            raise RuntimeError("caller Phase 3 Ja was already stopped")
        at = int(at)
        if at < self.qam_start:
            raise ValueError("Sd transition arrived before caller Ja began")
        actions = self.controller.feed("sd_transition", at)
        self.late_stop_samples = max(0, self.cursor - int(at))
        self._stop_at = max(at, self.cursor)
        if self._stop_at <= self.cursor:
            self._apply_ja_stop()
        return actions

    def _apply_ja_stop(self) -> None:
        if self._ja_stopped:
            return
        # Samples already returned are immutable. The remaining QAM buffer is
        # not yet transmitted, so erase it from this protocol boundary onward.
        self.qam.buf[:] = 0.0
        self.qam.queue.clear()
        self.qam.fill = None
        self.qam.history = [0j] * self.qam.span
        self._ja_stopped = True
