"""V.34 call-side modem engine for the lab DCE: SpanDSP V.8 (offering V.34) followed by v34_call.V34Caller, behind the
same surface the DCE uses for the other engines (feed_pcm / tx_samples / feed_bytes / connected / bit_rate ...),
with V.42 / MNP available through the V42Layer mixin.

Environment: V34_RATE (data rate to offer, default 9600), V34_SYMS (comma list of symbol rates to offer, default all).
"""

from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

from v34_call import V34Caller
from v42_layer import V42Layer, _Framer
from v34_retrain_budget import RetrainBudget, observe as observe_retrain_budget

FS = 8000
SETUP_TIMEOUT = 90.0                                        # seconds from the start to data mode


class V34Modem(V42Layer):
    is_v32 = False
    is_v34 = True
    capture_detect_bits = False

    def __init__(self, fs: int = FS, data_rate: int | None = None, min_data_rate: int = 2400, debug: bool = False, rms: float = 900.0, with_v8: bool = True):
        if fs != FS:
            raise ValueError("the V.34 engine runs at 8 kHz")
        from v8 import V8_MOD_V34, V8Negotiator

        # AT+MS/automode supplies a per-call ceiling. Environment is only the
        # standalone default, not permission to override the terminal profile.
        rate = int(data_rate if data_rate is not None else os.environ.get("V34_RATE", 9600))
        syms = os.environ.get("V34_SYMS")
        self.fs = fs
        self.debug = debug
        self.events: list[tuple[float, str]] = []
        self._ev_seen = 0
        self._caller_args = dict(rms=rms, data_rate=rate, min_data_rate=min_data_rate,
                                 symbol_rates=[int(x) for x in syms.split(",")] if syms else None,
                                 info0=dict(rate2743=1, rate2800=1, rate3429=1, lo3000=1, hi3000=1, lo3200=1,
                                            hi3200=1, allow3429=1, power_reduction=1, pts1664=1))
        self._retrains = 0
        self._retrain_budget = RetrainBudget(self.RETRAIN_MAX)
        self._retraining = False
        self._retrain_t0 = 0
        self.caller = self._new_caller()
        self._v8 = V8Negotiator(True, modulations=V8_MOD_V34)
        self._phase = "v8"                                  # v8 -> p2 (V34Caller) -> failed
        if not with_v8:                                     # V.8 already done by the caller of this engine (autochain.py)
            self._v8, self._phase = None, "p2"
        self.state = "listen"
        self.bit_rate = 0
        self.rx_bit_rate = 0
        self._t = 0
        self._rx_power = 0.0
        self._rtp_media_timing = os.environ.get("V34_RTP_MEDIA_TIMING", "0") == "1"
        self._rx_media_deadline: float | None = None
        self._tx_media_deadline: float | None = None
        self._plain_bits: deque[int] = deque()
        self._deframe = _Framer()
        self._rx_plain = bytearray()
        self._v42_setup()

    V42_RX_LAG_MARGIN_S = 1.0                                # receive queue behind real time (up to 0.65 s measured) plus slack

    @property
    def v42_detect_extra_seconds(self) -> float:
        """The ADP comes back one round trip after our ODP, through a receive path that runs behind real time.

        V42_DETECT_EXTRA_S overrides it (0 restores the plain 0.75 s T400)."""
        override = os.environ.get("V42_DETECT_EXTRA_S")
        if override is not None:
            return max(0.0, float(override))
        rtd = getattr(self.caller, "rtd_samples", None) or 0
        return rtd / self.fs + self.V42_RX_LAG_MARGIN_S

    def _new_caller(self) -> V34Caller:
        self._rate_epoch_seen = 0
        caller = V34Caller(**self._caller_args)
        rates=getattr(self,'_rate_fallback_rates',None)
        if rates is not None:
            from v34_negotiation import directional_offer
            caller._retrain_local_mp = directional_offer(caller._local_mp,'call',rates)
            caller._rate_local_mp = caller._retrain_local_mp
        caller.bit_source = self._bit_source
        caller._batch_bit_source = (caller.bit_source, self.v42_tx_bits)
        from v34_receive_dsp import configured
        self.receive_dsp = configured(caller)
        caller.bit_sink = self._bit_sink
        self._ev_seen = 0
        return caller

    RETRAIN_MAX = 2                                          # retrains before sustained qualified LAPM recovery
    RETRAIN_TIMEOUT = 45.0                                   # seconds for a retrain to reach data mode again

    def _carrier_stream_lost(self) -> None:
        """The freeze guard saw the receive stream lost for longer than V34_FREEZE_LIMIT_S (default 1 s): initiate a retrain
        (V.34 11.5.1.1) with a fresh caller under the unchanged V.42 layer, or end the call if retraining is off or used up.
        ``connected`` stays True through the retrain, as a real modem's DCD does, so the host does not see a carrier loss; V.42
        retransmits what the gap cost. V34_FREEZE_RETRAIN=0 turns the retrain off."""
        if os.environ.get("V34_FREEZE_RETRAIN", "1") == "0" or self._retrain_budget.used >= self.RETRAIN_MAX:
            self._fail("carrier stream lost for longer than the limit" +
                       ("" if self._retrains == 0 else f" (after {self._retrains} retrains)"))
            return
        self._retrain_budget.take()
        self._retrains += 1
        self._ev(f"carrier stream lost: retrain {self._retrain_budget.used}/{self.RETRAIN_MAX} (total {self._retrains}) (the V.42 link stays up)")
        self._retraining, self._retrain_t0 = True, self._t
        self.caller = self._new_caller()
        self.caller.begin_retrain()

    def request_retrain(self, reason='host requested full retrain'):
        """Bounded 11.5.1.1 initiation retaining the application/link state."""
        if not self.connected or self._retraining:
            raise RuntimeError('not in connected data mode')
        if self._retrain_budget.used >= self.RETRAIN_MAX:
            self._fail(f'{reason} (after {self._retrains} retrains)')
            return
        self._retrain_budget.take()
        self._retrains += 1
        self._ev(f'{reason}: retrain {self._retrain_budget.used}/{self.RETRAIN_MAX} (total {self._retrains}) (the V.42 link stays up)')
        self._retraining, self._retrain_t0 = True, self._t
        self.caller = self._new_caller()
        self.caller.begin_retrain()

    def _answerer_retrain(self) -> None:
        """The answerer started a retrain with tone A (11.5.2.1): answer it with silence then tone B (11.5.1.2) from a fresh caller,
        under the unchanged V.42 layer. Shares the RETRAIN_MAX budget with the lost-stream retrain."""
        if self._retrain_budget.used >= self.RETRAIN_MAX:
            self._fail(f"answerer retrained again (after {self._retrains} retrains)")
            return
        self._retrain_budget.take()
        self._retrains += 1
        self._ev(f"answerer initiated a retrain: answering it, retrain {self._retrain_budget.used}/{self.RETRAIN_MAX} (total {self._retrains}) (the V.42 link stays up)")
        self._retraining, self._retrain_t0 = True, self._t
        self.caller = self._new_caller()
        self.caller.begin_retrain()

    # -- the surface the DCE expects from an engine ----------------------------------------------
    @property
    def connected(self) -> bool:
        return self.state == "data"                        # also True during a retrain (see _carrier_stream_lost)

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
        request(self.caller, 'call', cap, resync=resync, rates=rates)

    def pending_bits(self) -> int:
        return len(self._plain_bits) + self.v42_pending_bits()

    # -- V42Layer hooks ---------------------------------------------------------------------------
    def _plain_tx_bit(self) -> int:
        return self._plain_bits.popleft() if self._plain_bits else 1

    def _plain_feed_bytes(self, data: bytes) -> None:
        for v in data:
            self._plain_bits.extend([0] + [(v >> i) & 1 for i in range(8)] + [1])

    def _plain_feed_marks(self, n: int) -> None:
        self._plain_bits.extend([1] * n)

    def _plain_rx_bits(self, bits) -> bytes:
        return self._deframe.push(list(bits))

    def _bit_source(self) -> int:
        return self.v42_tx_bit()

    def _bit_sink(self, bits) -> None:
        self._rx_plain += self.v42_rx_bits(bits)

    # -- audio ---------------------------------------------------------------------------------------
    def _ev(self, text: str) -> None:
        self.events.append((round(self._t / self.fs, 3), text))

    def _sync_events(self) -> None:
        ev = self.caller.events
        while self._ev_seen < len(ev):
            t, text = ev[self._ev_seen]
            self.events.append((round(t / self.fs, 3), text))
            self._ev_seen += 1

    def tx_samples(self, count: int) -> list[int]:
        self._t += count
        if self._phase == "v8":
            # V.8 has its own sample origin. Never carry a V.8 packet's RTP
            # anchor into the reset V.34 caller counters at phase 2.
            self._tx_media_deadline = None
            return self._v8.tx(count)
        if self._phase != "p2":
            return [0] * count
        if self._tx_media_deadline is not None:
            self.caller.set_tx_media_deadline(self._tx_media_deadline)
            self._tx_media_deadline = None
        out = np.clip(np.round(self.caller.tx(count)), -32768, 32767).astype(int)
        return out.tolist()

    def set_tx_media_deadline(self, deadline: float) -> None:
        """Set the scheduled media time of the next transmit block."""
        self._tx_media_deadline = float(deadline) if self._rtp_media_timing else None

    def set_rx_media_deadline(self, deadline: float) -> None:
        """Set the first-sample media time of the next received block."""
        self._rx_media_deadline = float(deadline) if self._rtp_media_timing else None

    def note_concealed_rx(self, n: int) -> None:
        """The next ``feed_pcm`` chunk (``n`` samples) is the DCE's concealment of a lost RTP stretch, not received audio."""
        self._conceal_pending = int(n)

    def feed_pcm(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype="<i2")
        concealed = getattr(self, "_conceal_pending", 0)
        self._conceal_pending = 0
        if len(x) == 0:
            return b""
        if self.receive_dsp is not None:x=self.receive_dsp.process(x)
        self._rx_power = float(np.mean(x.astype(float) ** 2)) * 8.0e6 / 8.0e6
        if self._phase == "v8":
            self._v8.rx(x.tolist())
            if self._v8.status == 2:
                self._ev("V.8 complete: " + str(self._v8.result))
                self._phase = "p2"
                self.caller.rx_t = 0
                self.caller.t = 0
            elif self._v8.status in (3, 4) or self._t / self.fs > SETUP_TIMEOUT:
                self._fail("V.8 failed")
            # The V.8 block cannot anchor the fresh phase-2 sample counters.
            self._rx_media_deadline = None
            return b""
        if self._phase == "p2":
            if concealed and concealed == len(x):
                self.caller.note_concealed(self.caller.rx_t, len(x))
            if self._rx_media_deadline is not None:
                self.caller.set_rx_media_deadline(self._rx_media_deadline)
                self._rx_media_deadline = None
            self.caller.rx(x.astype(float))
            self._sync_events()
            epoch = getattr(self.caller, '_rate_epoch', 0)
            if epoch != getattr(self, '_rate_epoch_seen', 0):
                self._rate_epoch_seen = epoch
                self.bit_rate, self.rx_bit_rate = self.caller.data_rate, self.caller.rx_rate
                self.set_v42_rate(self.bit_rate)
                self._ev(f'connected rate changed: tx={self.bit_rate} rx={self.rx_bit_rate}')
            reason = getattr(self.caller, '_rate_retrain_reason', None)
            if reason and not self._retraining:
                self.request_retrain('connected rate failed: ' + reason)
                return b''
            if self.caller.failed_reason:
                self._fail(self.caller.failed_reason)
                return b""
            if self.caller.answerer_retrain and not self._retraining:
                self._answerer_retrain()
                if self._phase == "failed":
                    return b""
            if self.caller.lost_too_long and not self._retraining:
                self._carrier_stream_lost()
                if self._phase == "failed":
                    return b""
            if self._retraining:
                if self.caller.stage == "data" and getattr(self.caller, "_rx_state", None) is not None:
                    self._retraining = False
                    self.bit_rate = self.caller.data_rate
                    self.rx_bit_rate = self.caller.rx_rate
                    self.set_v42_rate(self.bit_rate)
                    self._ev(f"retrain complete: data mode again at {self.bit_rate} bit/s")
                elif (self._t - self._retrain_t0) / self.fs > self.RETRAIN_TIMEOUT:
                    self._fail(f"retrain did not reach data mode within {self.RETRAIN_TIMEOUT:g} s")
                    return b""
            if self.caller.stage == "data" and getattr(self.caller, "_rx_state", None) is not None and self.state != "data":
                self.state = "data"
                self.bit_rate = self.caller.data_rate
                self.rx_bit_rate = self.caller.rx_rate
                self._ev(f"connected at {self.bit_rate} bit/s")
            elif self.state != "data" and self._t / self.fs > SETUP_TIMEOUT:
                self._fail("start-up timed out")
        out = bytes(self._rx_plain) + self.v42_read()
        self._rx_plain.clear()
        observe_retrain_budget(self, self.caller, len(x), out, concealed=bool(concealed))
        return out

    def _fail(self, why: str) -> None:
        self.state = "failed"
        self._phase = "failed"
        self._failed_reason = why
        self._ev("FAILED: " + why)
