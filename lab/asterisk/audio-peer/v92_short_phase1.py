"""V.92 short Phase 1 (Quick Connect), the V.8 path of clause 9.2 and Figures 3 and 4, as a two-party simulation.

Roles: the digital modem answers (or calls) with QC1d / QCA1d, the analogue modem with QC1a / QCA1a; whoever answers
sends ANSam. The digital modem always sends QTS, QTS-bar and ANSpcm and the analogue modem always sends TONEq:

    caller:    ANSam heard for 1 s -> QC1x (60 bits at 300 baud, 200 ms) then CM
    answerer:  silent 200 ms, ANSam; QC1x heard -> QCA1y (70 bits, 233 ms), then, if digital, silence 75 ms,
               QTS (768T) QTS-bar (48T) and ANSpcm; if analogue, silence and wait for QTS
    caller:    QCA1y heard -> stop CM, silence 75 ms and, if digital, QTS / QTS-bar / ANSpcm
    analogue:  ANSpcm heard for 1 s -> TONEq (at least 50 ms) until ANSpcm stops, then silence 75 ms -> Phase 2
    digital:   TONEq heard -> silence 75 ms -> Phase 2; no TONEq within 2 s of the QCA -> ANSam and V.8

The QC / QCA sequences are the real bit sequences of ``v92_quick_connect`` (built, carried and parsed); the UQTS
code in QC1a / QCA1a picks the Ucode V of QTS and the LM bits of QC1d / QCA1d pick the ANSpcm level (``v92_anspcm``).
QTS and ANSpcm are generated as G.711 codewords; the analogue side checks the QTS sign pattern and measures the
tone. Everything else is a label with the durations above. Not covered: the V.8 bis variants (CRe, QC2x, QCA2x,
Figures 5, 6 and 8), both modems analogue (Figure 7), the on-hold QTS cleardown, and what Phase 2 does next
(``v92_short_phase2``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import v34_g711
import v90_dil
import v92_anspcm
import v92_quick_connect as qc

BIT_MS = 1000 / 300
QC_MS = 200                                  # 60 bits
QCA_MS = 234                                 # 70 bits
EDGE_MS = 5                                  # a timed signal counts as heard just before it ends
SILENCE_MS = 75
ANSAM_SILENT_MS = 200
ANSAM_DETECT_MS = 1000
ANSPCM_DETECT_MS = 1000
TONEQ_MIN_MS = 50
TONEQ_DETECT_MS = 50
FALLBACK_MS = 2000


def qts_codewords(uqts_ucode: int, law: str) -> bytes:
    """QTS: 128 x {+V, +0, +V, -V, -0, -V} then QTS-bar: 8 x {-V, -0, -V, +V, +0, +V}, V the codeword of Ucode
    ``uqts_ucode``, 0 that of Ucode 0 (clause 8.3.6). 768T + 48T."""
    table = v90_dil.g711_ucode_codewords(law)
    pos_v, pos_0 = int(table[uqts_ucode]), int(table[0])
    neg = lambda c: c ^ 0x80                                 # the sign bit is the MSB in both laws
    qts = bytes([pos_v, pos_0, pos_v, neg(pos_v), neg(pos_0), neg(pos_v)]) * 128
    qts_bar = bytes([neg(pos_v), neg(pos_0), neg(pos_v), pos_v, pos_0, pos_v]) * 8
    return qts + qts_bar


def check_qts(codewords: bytes, law: str) -> int | None:
    """Recover V's Ucode from a received QTS / QTS-bar (768 + 48 symbols) or None if the pattern is wrong."""
    if len(codewords) != 816:
        return None
    table = v90_dil.g711_ucode_codewords(law)
    for ucode in range(1, 128):
        if codewords == qts_codewords(ucode, law):
            return ucode
    return None


def tone_hz(codewords: bytes, law: str) -> float:
    x = v34_g711.decode(np.frombuffer(codewords, dtype=np.uint8), law).astype(float)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spectrum) * 8000 / len(x))


@dataclass
class Party:
    role: str                                # "analogue" or "digital"
    calling: bool
    law: str = "pcmu"
    lapm: bool = True
    uqts_code: int = 0b0100                  # analogue: Ucode 67
    level_code: int = 0b01                   # digital: ANSpcm at -12 dBm0
    deaf_to_toneq: bool = False              # analogue peer that never answers ANSpcm (test)
    latency: int = 0
    tx: str = "silence"
    tx_data: object = None
    tx_until: int | None = None              # when a timed signal ends
    events: list = field(default_factory=list)
    state: str = "idle"
    t_state: int = 0
    phase2_at: int | None = None
    fallback: str | None = None
    peer_qc: dict | None = None
    qts_ucode: int | None = None
    ansp_level: float | None = None
    _queue: list = field(default_factory=list)       # [(start_ms, label, data, duration)]
    _fallback_from: int | None = None
    _stopped_cm: bool = False

    def send(self, t: int, label: str, data=None, duration: int | None = None) -> None:
        self.tx, self.tx_data = label, (tuple(data) if isinstance(data, list) else data)
        self.tx_until = None if duration is None else t + duration
        self.events.append((t, "tx", label))

    def plan(self, t: int, label: str, data=None, duration: int | None = None) -> None:
        self._queue.append((t, label, data, duration))
        self._queue.sort(key=lambda q: q[0])

    def tick(self, t: int) -> None:
        if self.tx_until is not None and t >= self.tx_until:
            self.tx, self.tx_data, self.tx_until = "silence", None, None
        while self._queue and self._queue[0][0] <= t:
            _, label, data, duration = self._queue.pop(0)
            self.send(t, label, data, duration)

    def start(self, t: int) -> None:
        if not self.calling:
            self.state = "answer-silent"
            self.plan(t + ANSAM_SILENT_MS, "ANSam")
        else:
            self.state = "wait-ansam"

    # -- what the peer's line carries now: (label, data, how long it has carried this label) ----------
    def hear(self, t: int, label: str, data, run: int) -> None:
        if self.phase2_at is not None or self.fallback:
            return
        digital = self.role == "digital"
        if self.calling:
            if self.state == "wait-ansam" and label == "ANSam" and run >= ANSAM_DETECT_MS:
                kind = "QC1d" if digital else "QC1a"
                bits = qc.build(kind, lapm=self.lapm, code=self.level_code if digital else self.uqts_code)
                self.send(t, kind, bits, QC_MS)
                self.plan(t + QC_MS, "CM")
                self.state = "sent-qc"
                self._ansam_heard_1s = True
            elif self.state == "sent-qc" and label in ("QCA1d", "QCA1a") and run >= QCA_MS - EDGE_MS:
                self._on_qca(t, data)
        else:
            if self.state in ("answer-silent", "answering") and label in ("QC1a", "QC1d") and run >= QC_MS - EDGE_MS:
                parsed = qc.parse(list(data))
                if parsed is not None and (parsed["kind"] == "QC1d") != digital:
                    self.peer_qc = parsed
                    kind = "QCA1d" if digital else "QCA1a"
                    code = self.level_code if digital else self.uqts_code
                    self.send(t, kind, qc.build(kind, lapm=self.lapm, code=code), QCA_MS)
                    self._after_qca(t, parsed)
                    self.state = "sent-qca"
                    self._fallback_from = t + QCA_MS
            elif self.state == "answer-silent" and self.tx == "ANSam":
                self.state = "answering"
        self._phase2_logic(t, label, data, run)

    def _on_qca(self, t: int, data) -> None:
        parsed = qc.parse(list(data))
        if parsed is None:
            return
        self.peer_qc = parsed
        self.send(t, "silence")                        # CM stops without completing its octet
        self.events.append((t, "stop-cm", None))
        self._after_qca(t, parsed)
        self.state = "after-qca"
        self._fallback_from = t

    def _after_qca(self, t: int, peer: dict) -> None:
        """The digital modem sends QTS / QTS-bar / ANSpcm 75 ms after the QCA exchange."""
        if self.role != "digital":
            return
        # The peer is analogue: its QC1a / QCA1a carries UQTS; mine carried the ANSpcm level.
        self.qts_ucode = peer["ucode"]
        start = t + (QCA_MS if not self.calling else 0) + SILENCE_MS
        self.plan(start, "QTS", qts_codewords(self.qts_ucode, self.law), 102)
        level = qc.LEVELS_DBM0[self.level_code]
        self.ansp_level = level
        self.plan(start + 102, "ANSpcm", level)

    def _phase2_logic(self, t: int, label: str, data, run: int) -> None:
        if self.role == "analogue":
            if label == "QTS" and self.state != "qts-ok":
                ucode = check_qts(bytes(data), self.law)
                if ucode is not None:
                    self.qts_ucode = ucode
                    self.events.append((t, "qts", ucode))
                    self.state = "qts-ok"
            if label == "ANSpcm" and run >= ANSPCM_DETECT_MS and self.tx != "TONEq" and not self.deaf_to_toneq \
                    and self.state in ("qts-ok", "after-qca", "sent-qca"):
                self.send(t, "TONEq")
                self.state = "toneq"
                self.t_state = t
            elif self.state == "toneq" and label != "ANSpcm" and t - self.t_state >= TONEQ_MIN_MS:
                self.send(t, "silence")
                self.phase2_at = t + SILENCE_MS
                self.events.append((t + SILENCE_MS, "phase2", "analogue"))
        else:
            if label == "TONEq" and run >= TONEQ_DETECT_MS and self.tx == "ANSpcm":
                self.send(t, "silence")
                self.phase2_at = t + SILENCE_MS
                self.events.append((t + SILENCE_MS, "phase2", "digital"))
        # fallback: no TONEq / ANSpcm in time
        if self._fallback_from is not None and self.phase2_at is None and t - self._fallback_from >= FALLBACK_MS \
                and self.role == "digital" and not self.calling:
            self.send(t, "ANSam")
            self.fallback = "v8"
            self.events.append((t, "fallback", "ANSam and V.8 (no TONEq within 2 s)"))


def run(caller: Party, answerer: Party, *, latency_ms: int = 10, max_ms: int = 8000) -> dict:
    parties = (caller, answerer)
    for p in parties:
        p.latency = latency_ms
    caller.start(0)
    answerer.start(0)
    hist = {id(p): [] for p in parties}
    starts = {id(p): [] for p in parties}
    for t in range(max_ms):
        for p in parties:
            p.tick(t)
            cur = (p.tx, p.tx_data)
            h = hist[id(p)]
            h.append(cur)
            starts[id(p)].append(starts[id(p)][-1] if t > 0 and h[-2] == cur else t)
        for p, peer in ((caller, answerer), (answerer, caller)):
            at = t - latency_ms
            if at < 0:
                p.hear(t, "silence", None, 0)
            else:
                label, data = hist[id(peer)][at]
                p.hear(t, label, data, t - (starts[id(peer)][at] + latency_ms))
        if all(p.phase2_at is not None or p.fallback for p in parties) and \
                all(p.phase2_at is None or t >= p.phase2_at for p in parties):
            break
    return {"caller": caller, "answerer": answerer}
