"""Message-level model of the V.92 final-training exchange (ITU-T V.92 clause 9.6, Figures 12-14).

Both modems run the same rules, differing only in how long they send TRN2 first:

* send TRN2, then SUV sequences once ready (digital: TRN2d for at least 2040T; analogue:
  TRN2u for at least 12000T *or* as soon as an SUV is received);
* on the first SUV from the peer, send a single CP, then more SUV;
* the acknowledge bit of a CP or SUV says "I have received your CP", so it is set from the
  first CP received onward;
* if no CP or SUV with the acknowledge bit set arrives within ``ack_wait`` after the end of the
  modem's own CP, repeat the CP (Figure 14);
* once the modem has sent something with its acknowledge bit set and has received a CP or SUV
  with the acknowledge bit set, or an E, it finishes the current sequence and sends E, then B1;
* it is connected when it has sent E and received the peer's E.

Time here is whole sequence slots, one message per slot, not symbols, and detection is perfect:
this checks the ordering and acknowledgement rules, not the signal processing. The SUV/CP
payloads are in ``v92_cp``; the recovery rule that 20 s plus 6 round-trip delays without B1 means
retrain (9.6.1.2.1) appears here as a slot budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    kind: str                 # "TRN", "SUV", "CP", "E", "B1"
    ack: bool = False

    def __str__(self) -> str:
        return self.kind + ("'" if self.ack else "")


class Party:
    def __init__(self, role: str, *, trn_slots: int, ack_wait: int):
        if role not in ("digital", "analogue"):
            raise ValueError("role must be digital or analogue")
        self.role = role
        self.trn_slots = trn_slots
        self.ack_wait = ack_wait
        self.slot = 0
        self.phase = "TRN"            # TRN -> SUV -> B1 (after the single E)
        self.sent: list[Message] = []
        self.saw_suv = False
        self.saw_cp = False           # peer's CP received: our acknowledge bit
        self.saw_ack = False          # a CP/SUV with its ack bit set received
        self.saw_e = False
        self.cp_sent_at: int | None = None
        self.sent_ack = False         # we have transmitted a CP/SUV with ack set
        self.sent_e = False

    @property
    def connected(self) -> bool:
        return self.sent_e and self.saw_e

    def tick(self, received: Message | None) -> Message:
        if received is not None:
            if received.kind in ("SUV", "CP"):
                self.saw_suv = True
                if received.kind == "CP":
                    self.saw_cp = True
                if received.ack:
                    self.saw_ack = True
            elif received.kind == "E":
                self.saw_e = True
            elif received.kind == "B1":
                self.saw_e = True       # B1 only follows E
        out = self._decide()
        self.sent.append(out)
        self.slot += 1
        return out

    def _decide(self) -> Message:
        if self.phase == "TRN":
            early = self.role == "analogue" and self.saw_suv
            if self.slot < self.trn_slots and not early:
                return Message("TRN")
            self.phase = "SUV"
        if self.phase == "B1":
            return Message("B1")
        # SUV phase
        ack = self.saw_cp
        if ack:
            self.sent_ack = True
        if (self.sent_ack and (self.saw_ack or self.saw_e)
                and self.cp_sent_at is not None):
            self.phase = "B1"           # E is sent once; every later slot is B1
            self.sent_e = True
            return Message("E", ack)
        if self.saw_suv and self.cp_sent_at is None:
            self.cp_sent_at = self.slot
            return Message("CP", ack)
        if (self.cp_sent_at is not None and not self.saw_ack
                and self.slot - self.cp_sent_at >= self.ack_wait):
            self.cp_sent_at = self.slot
            return Message("CP", ack)
        return Message("SUV", ack)


def run(latency: int = 1, *, digital_trn: int = 4, analogue_trn: int = 12,
        ack_wait: int | None = None, drop=None, max_slots: int = 200) -> dict:
    """Run both parties. ``latency`` is the one-way delay in slots; ``drop(sender, slot, message)``
    returning True discards a message in flight. ``ack_wait`` defaults to the round trip
    (``2 * (latency + 1)`` slots) plus a two-slot allowance, the analogue of the standard's
    "100 ms plus a round-trip delay" before a CP is repeated. Returns the traces and outcome."""
    if ack_wait is None:
        ack_wait = 2 * (latency + 1) + 2
    parties = {"digital": Party("digital", trn_slots=digital_trn, ack_wait=ack_wait),
               "analogue": Party("analogue", trn_slots=analogue_trn, ack_wait=ack_wait)}
    inbox: dict[str, dict[int, Message]] = {"digital": {}, "analogue": {}}
    traces = {"digital": [], "analogue": []}
    for slot in range(max_slots):
        for name, party in parties.items():
            peer = "analogue" if name == "digital" else "digital"
            message = party.tick(inbox[name].pop(slot, None))
            traces[name].append(message)
            if drop is None or not drop(name, slot, message):
                inbox[peer][slot + 1 + latency] = message
        if all(p.connected for p in parties.values()):
            return {"connected": True, "slots": slot + 1, "traces": traces, "parties": parties}
    return {"connected": False, "slots": max_slots, "traces": traces, "parties": parties}
