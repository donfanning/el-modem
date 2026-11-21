"""V.92 modem-on-hold transactions (clauses 9.10.1-9.10.2, Table 34, Figures 20-24), offline.

A two-party simulation at 1 ms resolution. What crosses is a label (Tone RT, an MH sequence, ANSam, QC, data,
silence) with the real 40-bit MH frame behind each MH sequence, built and parsed with ``v92_mh``; a delay of
``latency_ms`` each way. Detection is by duration: an MH sequence is recognised after one whole frame
(40 bits at the 600 baud of INFO, 67 ms) and only if the frame parses; Tone RT after 50 ms (100 ms is needed to
enter the on-hold state); silence after 2 s where the text asks for that; ANSam for 1 s before Phase 1.

Covered: request acknowledged then hold with T1 from the MHack timeout code, resume by QC, and the T1 expiry
(Figure 20); request denied then cleardown (21) or fast reconnect (22); cleardown request (23); fast
reconnect request (24); no response in 2 s plus a round trip. Not modelled: the retrain alternative of 9.10.1.1
(a Tone A reversal), the 30 s limit on continuing MHreq, circuit 107, and anything below the frame level (the
MH frames are not modulated here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import v92_mh

FRAME_MS = 67                  # 40 bits at 600 baud
RT_DETECT_MS = 50
RT_HOLD_MS = 100               # Tone RT seen this long (or 2 s of silence) puts the responder on hold
SILENCE_HOLD_MS = 2000
ANSAM_MS = 1000
SILENCE_BEFORE_ANSAM_MS = 40   # "within 80 ms"
RESPONSE_WAIT_MS = 2000
MH_NAMES = tuple(v92_mh.SIGNALS)


@dataclass
class Modem:
    name: str
    grant: bool = True                  # answer an MHreq with MHack (else MHnack)
    timeout_code: int = 3               # Table 33: 30 s
    after_ack: str = "rt"               # initiator after MHack: "rt" or "silence"
    after_nack: str = "cleardown"       # initiator after MHnack: "cleardown" or "fast_reconnect"
    reason: str = "incoming call"
    deaf: bool = False                  # ignores everything (a peer that does not answer)
    latency: int = 0
    tx: str = "silence"
    tx_bits: list = field(default_factory=list)
    events: list = field(default_factory=list)
    state: str = "data"
    disconnected: bool = False
    phase1: str | None = None           # "answer" or "caller" once Phase 1 is reached
    hold_started: int | None = None
    hold_limit_ms: int | None = None
    _seq_start: int = 0
    _responded: str | None = None
    _pending: list = field(default_factory=list)    # (at_ms, label, bits)
    _frr_at: int = 0
    _enter_hold_at: int = 0

    # -- transmit --------------------------------------------------------------------------------
    def send(self, t: int, label: str, bits=None, *, rt_first: bool = False) -> None:
        """Start sending ``label`` at t, optionally after 60 ms of Tone RT as in Figures 20-24."""
        if rt_first:
            self.tx, self.tx_bits = "RT", []
            self._pending = [(t + 60, label, bits or [])]
        else:
            self.tx, self.tx_bits = label, bits or []
            self._pending = []
        self.events.append((t, "tx", label))

    def tick(self, t: int) -> None:
        if self._pending and t >= self._pending[0][0]:
            _, label, bits = self._pending.pop(0)
            self.tx, self.tx_bits = label, bits

    def initiate(self, t: int, what: str) -> None:
        self._seq_start = t
        frame = {"MHreq": v92_mh.build_mh("MHreq"),
                 "MHclrd": v92_mh.build_mh("MHclrd", cleardown_reason=self.reason),
                 "MHfrr": v92_mh.build_mh("MHfrr")}[what]
        self.send(t, what, frame, rt_first=True)
        self.state = f"requested-{what}"

    # -- receive ---------------------------------------------------------------------------------
    def hear(self, t: int, label: str, bits: list, since: int) -> None:
        """What the peer's line carries at t, and since when it has carried exactly that."""
        if self.deaf:
            return
        run = t - since
        sig = None
        if label in MH_NAMES and run >= FRAME_MS:
            parsed = v92_mh.parse_mh(bits)
            if parsed is not None and parsed["signal"] == label:
                sig = parsed
        st = self.state
        if st == "data":
            if sig and self._responded is None:
                self._respond_to_initiating(t, sig)
        elif st == "requested-MHreq" and sig:
            if sig["signal"] == "MHack":
                limit = v92_mh.TIMEOUT_SECONDS.get(sig["timeout_code"])
                self.hold_limit_ms = None if limit is None else limit * 1000
                self.events.append((t, "mhack", sig["timeout_code"]))
                self.state = "acknowledged"
                self.send(t, "RT" if self.after_ack == "rt" else "silence")
            elif sig["signal"] == "MHnack":
                self.events.append((t, "mhnack", None))
                if self.after_nack == "cleardown":
                    self.send(t, "MHcda", v92_mh.build_mh("MHcda"))
                    self.state = "sent-cleardown-ack"
                else:
                    self.send(t, "MHfrr", v92_mh.build_mh("MHfrr"))
                    self.state = "requested-MHfrr"
                    self._seq_start = t
        elif st == "requested-MHclrd" and sig and sig["signal"] == "MHcda":
            self._disconnect(t, "cleardown acknowledged")
        elif st == "requested-MHfrr" and label == "ANSam" and run >= ANSAM_MS:
            self.phase1, self.state = "caller", "phase1"
            self.events.append((t, "phase1", "caller"))
        elif st == "granted":
            if (label == "RT" and run >= RT_HOLD_MS) or (label == "silence" and run >= SILENCE_HOLD_MS):
                self.send(t, "silence")
                self.state, self._enter_hold_at = "entering-hold", t + SILENCE_BEFORE_ANSAM_MS
        elif st == "entering-hold":
            if t >= self._enter_hold_at:
                self.send(t, "ANSam")
                self.state, self.hold_started = "on-hold", t
                self.events.append((t, "on-hold", self.hold_limit_ms))
        elif st == "on-hold":
            if label == "QC":
                self.phase1, self.state = "answer", "phase1"
                self.events.append((t, "phase1", "answer (resumed)"))
            elif self.hold_limit_ms is not None and t - self.hold_started >= self.hold_limit_ms:
                self._disconnect(t, "T1 expired")
        elif st == "denied":
            if sig and sig["signal"] == "MHcda":
                self._disconnect(t, "cleardown after MHnack")
            elif sig and sig["signal"] == "MHfrr":
                self.state, self._frr_at = "fast-reconnect", t
                self.send(t, "silence")
        elif st == "fast-reconnect":
            if t >= self._frr_at + SILENCE_BEFORE_ANSAM_MS:
                self.send(t, "ANSam")
                self.phase1, self.state = "answer", "phase1"
                self.events.append((t, "phase1", "answer (fast reconnect)"))
        elif st == "cleardown-answered":
            if (label != "MHclrd" and run >= 200) or (label in ("RT", "silence") and run >= RT_DETECT_MS):
                self._disconnect(t, "cleardown")
        elif st == "sent-cleardown-ack":
            pass
        if (self.state.startswith("requested-") and self.state != "requested-MHfrr"
                and t - self._seq_start >= RESPONSE_WAIT_MS + 2 * self.latency):
            self.events.append((t, "no-response", self.state))
            self._disconnect(t, "no response")

    def _respond_to_initiating(self, t: int, sig: dict) -> None:
        name = sig["signal"]
        self._responded = name
        self.events.append((t, "heard", name))
        if name == "MHreq":
            if self.grant:
                limit = v92_mh.TIMEOUT_SECONDS[self.timeout_code]
                self.hold_limit_ms = None if limit is None else limit * 1000
                self.send(t, "MHack", v92_mh.build_mh("MHack", timeout_code=self.timeout_code),
                          rt_first=True)
                self.state = "granted"
            else:
                self.send(t, "MHnack", v92_mh.build_mh("MHnack"), rt_first=True)
                self.state = "denied"
        elif name == "MHclrd":
            self.send(t, "MHcda", v92_mh.build_mh("MHcda"), rt_first=True)
            self.state = "cleardown-answered"
            self.events.append((t, "cleardown-reason", sig.get("cleardown_reason")))
        elif name == "MHfrr":
            self.send(t, "silence")
            self.state, self._frr_at = "fast-reconnect", t

    def _disconnect(self, t: int, why: str) -> None:
        if not self.disconnected:
            self.disconnected, self.state = True, "disconnected"
            self.send(t, "silence")
            self.events.append((t, "disconnect", why))


def run(initiator: Modem, responder: Modem, what: str, *, latency_ms: int = 10, max_ms: int = 70000,
        resume_at_ms: int | None = None, start_ms: int = 0) -> dict:
    """Run one transaction: ``initiator`` starts ``what`` (MHreq, MHclrd or MHfrr) at ``start_ms``.
    ``resume_at_ms`` makes the initiator send QC (a resumed call) that long after the hold began."""
    modems = (initiator, responder)
    for m in modems:
        m.latency = latency_ms
    hist = {m.name: [] for m in modems}
    run_start = {m.name: [] for m in modems}
    for t in range(max_ms):
        if t == start_ms:
            initiator.initiate(t, what)
        if (resume_at_ms is not None and initiator.state == "acknowledged" and responder.state == "on-hold"
                and t - responder.hold_started >= resume_at_ms and initiator.tx != "QC"):
            initiator.send(t, "QC")
        for m in modems:
            m.tick(t)
            cur = (m.tx, tuple(m.tx_bits))
            h = hist[m.name]
            h.append(cur)
            run_start[m.name].append(run_start[m.name][-1] if h[-2:-1] == [cur] and t > 0 else t)
        for m, peer in ((initiator, responder), (responder, initiator)):
            if m.disconnected:
                continue
            at = t - latency_ms
            if at < 0:
                m.hear(t, "silence", [], 0)
            else:
                label, bits = hist[peer.name][at]
                m.hear(t, label, list(bits), run_start[peer.name][at] + latency_ms)
        if all(m.disconnected or m.phase1 for m in modems):
            break
    return {"initiator": initiator, "responder": responder}
