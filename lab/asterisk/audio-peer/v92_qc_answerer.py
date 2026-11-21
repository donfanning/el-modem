"""Live V.92 Quick Connect Phase 1 response for a digital answer modem.

The V.8 negotiator identifies QC1a and hands its decoded fields to this
sequencer. This class sends QCA1d, QTS/QTS-bar and ANSpcm as exact G.711
codewords, then watches for TONEq. The V.8 chain owns the timeout/fallback and
the handoff into the ordinary V.90/V.92 digital Phase 2 engine.
"""

from __future__ import annotations

import math

import numpy as np

import v8
import v92_anspcm
import v92_quick_connect as qc
import v92_short_phase1 as short_phase1
from g711 import alaw_to_linear, linear_to_alaw, linear_to_ulaw, ulaw_to_linear


FS = 8000
SILENCE_SAMPLES = 600                    # 75 ms
QCA_SAMPLES = math.ceil(70 * FS / v8.BAUD)
TONEQ_BLOCK = 80                         # 10 ms
TONEQ_MIN_BLOCKS = 5                     # 50 ms
TONEQ_MIN_RMS = 200.0                     # approximately -38 dBm0 in linear PCM
TIMEOUT_AFTER_QCA = 2 * FS
_SILENCE = {"pcmu": 0xFF, "pcma": 0xD5}


class V92QuickConnectAnswerer:
    """Sequence one V.92 QC1a -> QCA1d -> QTS/ANSpcm exchange."""

    def __init__(self, *, peer: dict, law: str, lapm_offer: bool,
                 level_code: int = 0b01, amplitude: float = 2500.0):
        if law not in _SILENCE:
            raise ValueError("law must be pcmu or pcma")
        if peer.get("kind") != "QC1a" or peer.get("cleardown"):
            raise ValueError("digital-answerer requires a non-cleardown QC1a")
        if peer.get("ucode") is None:
            raise ValueError("QC1a UQTS code does not map to a QTS Ucode")
        if type(level_code) is not int or not 0 <= level_code <= 3:
            raise ValueError("ANSpcm level code must be an integer in 0..3")

        self.peer = dict(peer)
        self.law = law
        self.lapm_offer = bool(lapm_offer)
        self.lapm = bool(self.lapm_offer and peer.get("lapm", False))
        self.level_code = level_code
        self.level_dbm0 = qc.LEVELS_DBM0[level_code]
        self._decode = ulaw_to_linear if law == "pcmu" else alaw_to_linear
        self._encode = linear_to_ulaw if law == "pcmu" else linear_to_alaw

        self.stage = "qca1d"
        self.tx_samples = 0
        self.rx_samples = 0
        self._qca_left = QCA_SAMPLES
        self._silence_left = 0
        self._qts = short_phase1.qts_codewords(peer["ucode"], law)
        self._qts_at = 0
        self._anspcm_at = 0
        self._fsk = v8._Fsk(v8.V21_H, amplitude)
        self._fsk.bits = qc.build(
            "QCA1d", lapm=self.lapm_offer, code=level_code)
        self._rx_pending: list[int] = []
        self._toneq_blocks = 0
        self._toneq_seen = False
        self._post_toneq_left = 0
        # Start the TONEq response timer when QCA1d has actually finished
        # transmitting. The required inter-signal silence is not part of the
        # two-second allowance.
        self._toneq_deadline: int | None = None
        self.events: list[tuple[str, object]] = [
            ("qca1d", {"lapm": self.lapm, "level_code": level_code,
                        "level_dbm0": self.level_dbm0}),
        ]
        self.complete = False

    @property
    def expired(self) -> bool:
        return (self._toneq_deadline is not None
                and self.rx_samples >= self._toneq_deadline
                and not self._toneq_seen and not self.complete)

    def tx_g711(self, count: int, codec: str) -> bytes:
        if codec != self.law:
            raise ValueError("QC transmit law does not match the negotiated RTP codec")
        if count < 0:
            raise ValueError("count must be non-negative")
        out = bytearray()
        while len(out) < count:
            need = count - len(out)
            if self.stage == "qca1d":
                n = min(need, self._qca_left)
                samples = self._fsk.read(n)
                out.extend(self._encode(int(round(x))) for x in samples)
                self._qca_left -= n
                self.tx_samples += n
                if self._qca_left == 0:
                    self.stage = "post-qca-silence"
                    self._silence_left = SILENCE_SAMPLES
                    self._toneq_deadline = self.rx_samples + TIMEOUT_AFTER_QCA
                    self.events.append(("qca1d-complete", self.tx_samples))
            elif self.stage == "post-qca-silence":
                n = min(need, self._silence_left)
                out.extend(bytes([_SILENCE[self.law]]) * n)
                self._silence_left -= n
                self.tx_samples += n
                if self._silence_left == 0:
                    self.stage = "qts"
                    self.events.append(("qts-start", self.tx_samples))
            elif self.stage == "qts":
                n = min(need, len(self._qts) - self._qts_at)
                out.extend(self._qts[self._qts_at:self._qts_at + n])
                self._qts_at += n
                self.tx_samples += n
                if self._qts_at == len(self._qts):
                    self.stage = "anspcm"
                    self.events.append(("anspcm-start", self.tx_samples))
            elif self.stage == "anspcm":
                n = need
                out.extend(v92_anspcm.stream(
                    n, self.level_dbm0, self.law,
                    start_symbol=self._anspcm_at))
                self._anspcm_at += n
                self.tx_samples += n
            elif self.stage == "post-toneq-silence":
                n = min(need, self._post_toneq_left)
                out.extend(bytes([_SILENCE[self.law]]) * n)
                self._post_toneq_left -= n
                self.tx_samples += n
                if self._post_toneq_left == 0:
                    self.stage = "complete"
                    self.complete = True
                    self.events.append(("post-toneq-silence-complete", self.tx_samples))
            else:
                out.extend(bytes([_SILENCE[self.law]]) * need)
                self.tx_samples += need
        return bytes(out)

    def tx_samples_pcm(self, count: int) -> list[int]:
        wire = self.tx_g711(count, self.law)
        return [int(self._decode(value)) for value in wire]

    @staticmethod
    def _toneq_present(block: list[int]) -> bool:
        x = np.asarray(block, dtype=float)
        x -= float(np.mean(x))
        power = float(x @ x)
        if power < len(x) * TONEQ_MIN_RMS ** 2:
            return False
        n = np.arange(len(x), dtype=float)
        projection = abs(np.sum(x * np.exp(-2j * math.pi * 980.0 * n / FS))) ** 2
        return float(2.0 * projection / (len(x) * power + 1e-30)) >= 0.70

    def feed_pcm(self, samples) -> None:
        if self.complete:
            return
        values = np.asarray(samples, dtype=np.int16).astype(int).tolist()
        self.rx_samples += len(values)
        if self.stage == "anspcm":
            self._rx_pending.extend(values)
            while len(self._rx_pending) >= TONEQ_BLOCK:
                block = self._rx_pending[:TONEQ_BLOCK]
                del self._rx_pending[:TONEQ_BLOCK]
                if self._toneq_present(block):
                    self._toneq_blocks += 1
                    if self._toneq_blocks >= TONEQ_MIN_BLOCKS:
                        self._toneq_seen = True
                        self.stage = "post-toneq-silence"
                        self._post_toneq_left = SILENCE_SAMPLES
                        self.events.append(("toneq", self.rx_samples))
                        return
                else:
                    self._toneq_blocks = 0
        else:
            self._rx_pending.clear()

    def feed_g711(self, payload: bytes, codec: str) -> None:
        if codec != self.law:
            raise ValueError("QC receive law does not match the negotiated RTP codec")
        self.feed_pcm([self._decode(value) for value in payload])

    def feed_gap(self, samples: int) -> None:
        if samples > 0:
            self.feed_pcm([0] * samples)
