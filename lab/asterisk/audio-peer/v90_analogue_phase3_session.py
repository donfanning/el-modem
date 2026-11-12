"""Join the analogue caller Phase 3 controller to digital G.711 events."""

from __future__ import annotations

from v90_analogue_downstream_rx import V90AnalogueDownstreamPhase3Receiver
from v90_analogue_caller_phase3 import V90AnalogueCallerPhase3
from v90_analogue_phase3_tx import V90AnaloguePhase3Transmitter


class V90AnaloguePhase3Session:
    """Drive caller Phase 3 from qualified downstream G.711 codewords.

    The input here is exact negotiated-law RTP payload, not signed-linear PCM.
    A caller-side media path must preserve/recover those octets before using
    this adapter. Returned actions after Jd (the caller S/S-bar/DIL response)
    still need a live waveform/media renderer.
    """

    def __init__(self, controller: V90AnalogueCallerPhase3,
                 receiver: V90AnalogueDownstreamPhase3Receiver,
                 transmitter: V90AnaloguePhase3Transmitter | None = None):
        if not isinstance(controller, V90AnalogueCallerPhase3):
            raise TypeError("session requires the analogue-caller Phase 3 controller")
        if not isinstance(receiver, V90AnalogueDownstreamPhase3Receiver):
            raise TypeError("session requires the digital-downstream Phase 3 receiver")
        if transmitter is not None and transmitter.controller is not controller:
            raise ValueError("transmitter and session must share one controller")
        self.controller = controller
        self.receiver = receiver
        self.transmitter = transmitter
        self.receiver.set_expected_dil(controller.dil_symbols)
        self.started = False

    def start(self, at: int):
        if self.started:
            raise RuntimeError("analogue Phase 3 session already started")
        self.started = True
        if self.transmitter is not None:
            return self.transmitter.start(at)
        return self.controller.start(at)

    def feed_downstream_g711(self, codewords, *, at: int,
                             sequence_at: int | None = None) -> list[dict]:
        """Consume contiguous G.711 chunks and react to qualified protocol events."""
        if not self.started:
            raise RuntimeError("start caller Phase 3 before feeding downstream audio")
        actions_by_id = {}

        def react(event):
            name = event["event"]
            if name == "sd_transition":
                if self.transmitter is not None:
                    actions = self.transmitter.notify_sd_transition(event["at"])
                else:
                    actions = self.controller.feed("sd_transition", event["at"])
            elif name == "trn1d_min_complete":
                actions = self.controller.feed("trn1d_2040_complete", event["at"])
            elif name == "jd":
                actions = self.controller.feed("jd", event["at"])
            elif name == "jd_prime":
                actions = self.controller.feed("jd_prime", event["at"])
            elif name == "dil_codewords":
                actions = self.controller.feed_dil_codewords(
                    event["codewords"], at=event["at"])
            else:
                actions = []
            if self.transmitter is not None and actions:
                self.transmitter.schedule_actions(actions)
                self._arm_s_response(actions)
            actions_by_id[id(event)] = actions

        events = self.receiver.feed(codewords, at=at, sequence_at=sequence_at,
                                     on_event=react)
        return [{"event": event["event"], "at": event["at"],
                 "details": event, "actions": actions_by_id.get(id(event), [])}
                for event in events]

    def notify_caller_s_sent(self, at: int) -> None:
        """Arm J′d parsing at the sample where caller S starts transmitting."""
        if self.controller.state != self.controller.WAIT_JD_PRIME:
            raise RuntimeError("caller S notification is out of Phase 3 order")
        self.receiver.notify_caller_s_sent(at)

    def _arm_s_response(self, actions) -> None:
        """Arm the parser when S is scheduled, not after its 128T duration."""
        if self.transmitter is None or self.transmitter.cursor is None:
            return
        for action in actions:
            if (action.signal == "S"
                    and self.controller.state == self.controller.WAIT_JD_PRIME):
                # schedule_actions clips a response to the current TX cursor
                # when RX processing runs after the packet's audio was read.
                # Keep the parser's expected Jd remainder anchored to that
                # actual first transmit sample.
                self.receiver.notify_caller_s_sent(
                    max(int(action.at), int(self.transmitter.cursor)))
