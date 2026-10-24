"""Answer-side V.34 engine for the lab DCE: V.8 followed by V34Answerer."""

from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

from v34_ans import V34Answerer
from v42_layer import V42Layer, _Framer
from v34_retrain_budget import RetrainBudget, observe as observe_retrain_budget

FS = 8000
SETUP_TIMEOUT = 90.0


class V34AnswerModem(V42Layer):
    is_v32 = False
    is_v34 = True
    capture_detect_bits = False

    def __init__(self, fs: int = FS, data_rate: int | None = None, min_data_rate: int = 2400, debug: bool = False, rms: float = 900.0, with_v8: bool = True):
        if fs != FS:
            raise ValueError("the V.34 engine runs at 8 kHz")
        from v8 import V8_MOD_V34, V8Negotiator

        rate = int(data_rate if data_rate is not None else os.environ.get("V34_RATE", 9600))
        syms = os.environ.get("V34_SYMS")
        self.fs = fs
        self.debug = debug
        self.events: list[tuple[float, str]] = []
        self._ev_seen = 0
        self._answerer_args = dict(rms=rms, data_rate=rate, min_data_rate=min_data_rate,
            symbol_rates=[int(x) for x in syms.split(",")] if syms else None)
        self._retrains = 0
        self._retrain_budget = RetrainBudget(self.RETRAIN_MAX)
        self._retraining = False
        self._retrain_t0 = 0
        self._tone_b_run = 0
        self.answerer = self._new_answerer()
        self._v8 = V8Negotiator(False, modulations=V8_MOD_V34)
        self._phase = "v8"
        if not with_v8:                                     # V.8 already done by the caller of this engine (autochain.py)
            self._v8, self._phase = None, "p2"
        self.state = "listen"
        self.bit_rate = 0
        self.rx_bit_rate = 0
        self._t = 0
        self._rtp_media_timing = os.environ.get("V34_RTP_MEDIA_TIMING", "0") == "1"
        self._rx_media_deadline: float | None = None
        self._tx_media_deadline: float | None = None
        self._rx_power = 0.0
        self._plain_bits: deque[int] = deque()
        self._deframe = _Framer()
        self._rx_plain = bytearray()
        self._v42_setup()

    RETRAIN_MAX = 2
    RETRAIN_TIMEOUT = 45.0

    def _new_answerer(self):
        self._rate_epoch_seen = 0
        answerer = V34Answerer(**self._answerer_args)
        # Lab-only control for reproducing asymmetric MP negotiation through
        # the real DCE/RTP path. AT+MS has one overall ceiling; this test hook
        # lowers only the answerer's answer-to-call transmit offer while
        # retaining its call-to-answer receive capability. It is intentionally
        # opt-in and does not change user profiles or normal operation.
        test_tx_rate = os.environ.get("V34_TEST_ANSWER_TX_RATE")
        if test_tx_rate is not None:
            rate = int(test_tx_rate)
            if rate % 2400 or not 2400 <= rate <= self._answerer_args["data_rate"]:
                raise ValueError("V34_TEST_ANSWER_TX_RATE must be a supported rate no higher than the answerer ceiling")
            answerer._local_mp["max_a2c"] = rate // 2400
        rates=getattr(self,'_rate_fallback_rates',None)
        if rates is not None:
            from v34_negotiation import directional_offer
            answerer._retrain_local_mp = directional_offer(answerer._local_mp,'answer',rates)
            answerer._rate_local_mp = answerer._retrain_local_mp
        answerer.bit_source = self._bit_source
        answerer._batch_bit_source = (answerer.bit_source, self.v42_tx_bits)
        answerer.bit_sink = self._bit_sink
        from v34_receive_dsp import configured
        self.receive_dsp = configured(answerer)
        self._ev_seen = 0
        return answerer

    @staticmethod
    def _is_tone_b(x):
        # Tone B is 1200 Hz. Reject weak input and broadband/data energy.
        n = len(x)
        if n < 80 or float(np.mean(x*x)) < 100.0**2:
            return False
        line = abs(np.dot(x, np.exp(-2j*np.pi*1200*np.arange(n)/FS)))**2*2/n
        return line >= .9*float(np.dot(x, x))

    def _watch_for_caller_retrain(self, x):
        if ((os.environ.get('V34_CALLER_RETRAIN') != '1'
                and not getattr(self.answerer, '_connected_rate_enabled', False))
                or not self.connected or self._retraining):
            return
        self._tone_b_run = self._tone_b_run+len(x) if self._is_tone_b(x) else 0
        if self._tone_b_run < .080*FS:
            return
        if self._retrain_budget.used >= self.RETRAIN_MAX:
            self._fail(f'caller retrained again (after {self._retrains} retrains)')
            return
        # Keep the original ceiling and exchanged INFO0 capabilities. Replace
        # only the DSP core: LAPM queues, sequence numbers and DCD remain alive.
        info0c = dict(self.answerer.info0c) if self.answerer.info0c is not None else None
        if info0c is None:
            self._fail('caller retrain has no retained INFO0c capabilities')
            return
        self._retrain_budget.take()
        self._retrains += 1
        self._retraining, self._retrain_t0 = True, self._t
        self._tone_b_run = 0
        self._ev(f'caller initiated a retrain: answering it, retrain {self._retrain_budget.used}/{self.RETRAIN_MAX} (total {self._retrains}) (the V.42 link stays up)')
        self.answerer = self._new_answerer()
        self.answerer.respond_to_retrain(info0c)

    def request_retrain(self, reason='host requested full retrain'):
        """Bounded 11.5.2.1 initiation retaining LAPM and queued bytes."""
        if not self.connected or self._retraining:
            raise RuntimeError('not in connected data mode')
        if self._retrain_budget.used >= self.RETRAIN_MAX:
            self._fail(f'{reason} (after {self._retrains} retrains)')
            return
        info0c = self.answerer.info0c
        if info0c is None:
            self._fail('retrain has no retained INFO0c capabilities')
            return
        self._retrain_budget.take()
        self._retrains += 1
        self._retraining, self._retrain_t0 = True, self._t
        self._tone_b_run = 0
        self._ev(f'{reason}: retrain {self._retrain_budget.used}/{self.RETRAIN_MAX} (total {self._retrains}) (the V.42 link stays up)')
        self.answerer = self._new_answerer()
        self.answerer.begin_retrain(info0c)

    @property
    def connected(self) -> bool:
        return self.state == "data"

    @property
    def failed_reason(self) -> str:
        return getattr(self, "_failed_reason", "")

    def is_settled(self) -> bool:
        return self.connected

    def feed_bits(self, bits) -> None:
        pass

    @property
    def rx_signal_power_dbm0(self) -> float:
        return 10 * math.log10(self._rx_power / 8.0e6 + 1e-12) - 3.0 if self._rx_power > 0 else -99.0

    def feed_bytes(self, data: bytes) -> None:
        self.v42_feed_bytes(data)

    def request_rate_change(self, cap=None, *, resync=False, rates=None):
        from v34_rate_live import request
        request(self.answerer, 'answer', cap, resync=resync, rates=rates)

    def pending_bits(self) -> int:
        return len(self._plain_bits) + self.v42_pending_bits()

    def _plain_tx_bit(self) -> int:
        return self._plain_bits.popleft() if self._plain_bits else 1

    def _plain_feed_bytes(self, data: bytes) -> None:
        for value in data:
            self._plain_bits.extend([0] + [(value >> i) & 1 for i in range(8)] + [1])

    def _plain_feed_marks(self, n: int) -> None:
        self._plain_bits.extend([1] * n)

    def _plain_rx_bits(self, bits) -> bytes:
        return self._deframe.push(list(bits))

    def _bit_source(self) -> int:
        return self.v42_tx_bit()

    def _bit_sink(self, bits) -> None:
        self._rx_plain += self.v42_rx_bits(bits)

    def _ev(self, text: str) -> None:
        self.events.append((round(self._t / self.fs, 3), text))

    def _sync_events(self) -> None:
        while self._ev_seen < len(self.answerer.events):
            t, text = self.answerer.events[self._ev_seen]
            self.events.append((round(t / self.fs, 3), text))
            self._ev_seen += 1

    def tx_samples(self, count: int) -> list[int]:
        self._t += count
        if self._phase == "v8":
            self._tx_media_deadline = None
            return self._v8.tx(count)
        if self._phase != "p2":
            return [0] * count
        if self._tx_media_deadline is not None:
            self.answerer.set_tx_media_deadline(self._tx_media_deadline)
            self._tx_media_deadline = None
        return np.clip(np.round(self.answerer.tx(count)), -32768, 32767).astype(int).tolist()

    def set_tx_media_deadline(self, deadline: float) -> None:
        """Set the scheduled media time of the next transmit block."""
        self._tx_media_deadline = float(deadline) if self._rtp_media_timing else None

    def set_rx_media_deadline(self, deadline: float) -> None:
        """Set the first-sample media time of the next received block."""
        self._rx_media_deadline = float(deadline) if self._rtp_media_timing else None

    def note_concealed_rx(self, n: int) -> None:
        """Mark the next PCM chunk as transport concealment, like the caller."""
        self._conceal_pending = int(n)

    def note_concealed(self, n: int) -> None:
        # Preserve existing direct callers of the older answerer-only name.
        self.note_concealed_rx(n)

    def feed_pcm(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype="<i2")
        concealed = getattr(self, "_conceal_pending", 0)
        self._conceal_pending = 0
        if len(x) == 0:
            return b""
        if self.connected and not self._retraining:
            self._watch_for_caller_retrain(x.astype(float))
        if self._phase == "failed":return b""
        if self.receive_dsp is not None:x=self.receive_dsp.process(x)
        self._rx_power = float(np.mean(x.astype(float) ** 2))
        if self._phase == "v8":
            self._v8.rx(x.tolist())
            if self._v8.status == 2:
                self._ev("V.8 complete: " + str(self._v8.result))
                self._phase = "p2"
                self.answerer.rx_t = 0
                self.answerer.t = 0
                self._rx_media_deadline = None
            elif self._v8.status in (3, 4) or self._t / self.fs > SETUP_TIMEOUT:
                self._fail("V.8 failed")
            self._rx_media_deadline = None
            return b""
        if self._phase == "p2":
            if concealed and concealed == len(x):
                self.answerer.note_concealed(self.answerer.rx_t,len(x))
            if self._rx_media_deadline is not None:
                self.answerer.set_rx_media_deadline(self._rx_media_deadline)
                self._rx_media_deadline = None
            self.answerer.rx(x.astype(float))
            self._sync_events()
            epoch = getattr(self.answerer, '_rate_epoch', 0)
            if epoch != getattr(self, '_rate_epoch_seen', 0):
                self._rate_epoch_seen = epoch
                self.bit_rate, self.rx_bit_rate = self.answerer.data_rate, self.answerer.rx_rate
                self.set_v42_rate(self.bit_rate)
                self._ev(f'connected rate changed: tx={self.bit_rate} rx={self.rx_bit_rate}')
            reason = getattr(self.answerer, '_rate_retrain_reason', None)
            if reason and not self._retraining:
                self.request_retrain('connected rate failed: ' + reason)
                return b''
            if getattr(self.answerer, "failed_reason", None):
                self._fail(self.answerer.failed_reason)
                return b""
            if self._retraining:
                if self.answerer.stage == "data" and self.answerer._rx_state is not None:
                    self._retraining = False
                    self.bit_rate = self.answerer.data_rate
                    self.rx_bit_rate = self.answerer.rx_rate
                    self.set_v42_rate(self.bit_rate)
                    self._ev(f"retrain complete: data mode again at {self.bit_rate} bit/s")
                elif (self._t-self._retrain_t0)/self.fs > self.RETRAIN_TIMEOUT:
                    self._fail(f"retrain did not reach data mode within {self.RETRAIN_TIMEOUT:g} s")
                    return b""
            if self.answerer.stage == "data" and self.answerer._rx_state is not None and self.state != "data":
                self.state = "data"
                self.bit_rate = self.answerer.data_rate
                self.rx_bit_rate = self.answerer.rx_rate
                self._ev(f"connected at {self.bit_rate} bit/s")
            elif self.state != "data" and self._t / self.fs > SETUP_TIMEOUT:
                self._fail("start-up timed out")
        out = bytes(self._rx_plain) + self.v42_read()
        self._rx_plain.clear()
        observe_retrain_budget(self, self.answerer, len(x), out, concealed=bool(concealed))
        return out

    def _fail(self, why: str) -> None:
        self.state = "failed"
        self._phase = "failed"
        self._failed_reason = why
        self._ev("FAILED: " + why)
