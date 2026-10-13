"""V.32 / V.32bis answering modem (ITU-T V.32bis 6.2), on the caller engine's machinery.

Procedure, with the timings the recommendation gives:

    V.8 (ANSam, CM -> JM -> CJ) in parallel with a plain V.25 ANS fallback, then silence
    AC       alternate A/C states (600 + 3000 Hz) until the caller's AA (1800 Hz) has been heard
    CA       reverse to C/A and start the counter MT; the caller's AA->CC phase reversal stops it
             and 64 symbols later we revert to AC; the caller then goes silent
    TRN1     after 16 silent symbols: S (256), S-bar (16), TRN (GPA), then R1 (what we offer)
    wait     the caller starts its S when it sees R1: we stop transmitting, wait for its TRN
             (S lasts NT + 256 symbols, so its TRN begins about MT + 272 symbols after its S begins)
    train    locate/train on its TRN (GPC), decision-directed decode of its R2
    TRN2     on R2: second S/S-bar/TRN, then R3 = the best rate that R2 and we share
    E        on the caller's E: send E, scrambled ones, then data (trellis at 7200+)

Same surface as ``V32Modem`` (feed_pcm / tx_samples / feed_bytes / connected / bit_rate), V.42 included.
"""

from __future__ import annotations

import cmath
import math
import os

from v32_caller import GPC, RATE_BITS, RATE_WORD_BASE, V32CallerTx, A
from v32_stream import BAUD, CARRIER, IMPLEMENTED_RATES, V32Modem

ANS_LEGACY_SECONDS = 3.3
ANS_LEAD_SILENCE = 2.0                 # V.25: the answerer is silent for ~2 s before ANS
V8_GIVE_UP_SECONDS = 7.0
SILENCE_AFTER_ANS = 0.075
ANSWER_TIMEOUT_SECONDS = 90.0


class V32Answerer(V32Modem):
    def __init__(self, fs: int = 8000, my_rates=(14400, 12000, 9600, 7200, 4800), amplitude: float = 900.0,
                 trn_length: int = 2976, use_v8=None, **kw):
        super().__init__(fs=fs, my_rates=my_rates, amplitude=amplitude, use_v8=False, **kw)
        self.tx = V32CallerTx(my_rates=self.my_rates, sample_rate=fs, amplitude=float(os.environ.get("V32_AMP", amplitude)),
                              rolloff=0.3, span=12, role="answerer", s_length=256, trn_length=trn_length)
        self.tx.bit_source = self.v42_tx_bit
        self._tx_train_symbols = 256 + 16 + trn_length
        self._rx_tap = GPC
        from v32_retrain import RequestTone
        self._request_tone = RequestTone("answer", fs)
        self._c1800 = [cmath.exp(-2j * math.pi * 1800 * i / fs) for i in range(self.blk)]
        self.state = "ans"
        self._v8a = None
        if use_v8 is None:
            use_v8 = fs == 8000 and os.environ.get("V32_V8", "1") != "0"
        if use_v8:
            try:
                from v8 import V8_MOD_V32, V8Negotiator, available
                if available():
                    self._v8a = V8Negotiator(False, modulations=V8_MOD_V32)
            except Exception as exc:                  # noqa: BLE001
                self.events.append((0.0, f"V.8 unavailable: {exc}"))
        self._ans_samples = 0
        self._lead_left = int(float(os.environ.get("V32_ANS_DELAY", ANS_LEAD_SILENCE)) * fs)
        self._answered_at = 0
        self._gap_left = int(SILENCE_AFTER_ANS * fs)
        self._ca_tx_at = None
        self._mt = 0.0                                # symbols
        self._revert_in = None                        # samples until we revert CA -> AC
        self._aa_blocks = 0
        self._gone_blocks = 0
        self._s_seen_at = None
        self._quiet_for_s = 0
        self._e_expected = None
        self._started_at = 0

    # -- transmit ---------------------------------------------------------------------
    def _tx_samples(self, count: int) -> list[int]:
        st = self.state
        if st == "ans":
            out = self._ans_audio(count)
        elif st == "gap":
            out = [0] * count
            self._gap_left -= count
            if self._gap_left <= 0:
                self.state = "ac"
                self.tx.start_ac()
                self._ev("sending AC")
        elif st in ("ac", "ca", "ac2", "tx1", "tx2", "data", "rx_wait", "train_c"):
            if st == "ca" and self._ca_tx_at is None:
                self._ca_tx_at = self._txn
            if self._revert_in is not None:
                self._revert_in -= count
                if self._revert_in <= 0:
                    self._revert_in = None
                    self.tx.reverse_ac()                     # CA -> AC, 64 symbols after the reversal
                    self.state = "ac2"
                    self._ev("reverted CA->AC; waiting for the caller to go quiet")
            out = self.tx.tx_samples(count)
        else:
            out = [0] * count
        return self._finish_tx(out, count)

    def _ans_audio(self, count: int) -> list[int]:
        if self._lead_left > 0:
            self._lead_left -= count
            return [0] * count
        if self._v8a is not None:
            return self._v8a.tx(count)
        out = []
        for i in range(count):
            n = self._ans_samples + i
            flip = int(n / (0.45 * self.fs)) % 2                  # V.25 ANS: phase reversal every 450 ms
            out.append(round(3000 * math.cos(2 * math.pi * 2100 * n / self.fs + math.pi * flip)))
        self._ans_samples += count
        return out

    # -- receive ------------------------------------------------------------------------
    def _rx_block(self, x) -> None:
        self._t += len(x)
        if self.state == "ans":
            self._ans_progress(x)
        elif self.state in ("ac", "ca", "ac2"):
            self._answer_tones(x)
        for pos, v in self.bb.feed(x):
            self._on_baseband(pos, v)
        self._step()

    def _ans_progress(self, x) -> None:
        if self._v8a is not None:
            self._v8a.rx(x)
            name = self._v8a.status_name
            if name != self._v8_status:
                self._v8_status = name
                self._ev(f"V.8: {name}")
            done = self._v8a.status in (2, 3, 4)
            if self._t / self.fs > V8_GIVE_UP_SECONDS + ANS_LEAD_SILENCE:
                done = True
            if done:
                if self._v8a.status == 2 and self._v8a.result:
                    from v8 import V8_MOD_V32
                    if not self._v8a.result["modulations"] & V8_MOD_V32:
                        return self._fail("caller's V.8 choice is not V.32")
                self._v8a.close()
                self._v8a = None
                self.state = "gap"
        elif self._lead_left <= 0 and self._ans_samples / self.fs > ANS_LEGACY_SECONDS:
            self.state = "gap"

    def _answer_tones(self, x) -> None:
        a18 = sum(v * c for v, c in zip(x, self._c1800)) / len(x)
        present = abs(a18) * 2 > 300
        if self._retraining:
            power = sum(v*v for v in x)/len(x)
            present = present and power > 0 and 2*abs(a18)**2/power >= .85
        ph = cmath.phase(a18)
        if self.state == "ac":
            self._aa_blocks = self._aa_blocks + 1 if present else 0
            if self._aa_blocks >= 3 and (not self._retraining or self._txn >= math.ceil(128*self.fs/BAUD)):                   # 1800 Hz for well over 64 symbols
                self.tx.reverse_ac()
                self.state = "ca"
                self._ca_tx_at = None
                self._ph_hist, self._prev_age = [ph], 0
                self._ev("caller's AA heard: AC->CA")
            return
        if self.state == "ca":
            if present:
                hist = self._ph_hist if self._prev_age <= 3 else []
                flip = any(abs((ph - h + math.pi) % (2 * math.pi) - math.pi) > 2.2 for h in hist)
                self._ph_hist = ([] if flip else self._ph_hist)[-2:] + [ph]
                self._prev_age = 0
                if flip:
                    rev = self._t - self.blk // 2
                    start = self._ca_tx_at if self._ca_tx_at is not None else rev
                    self._mt = max(0.0, (rev - start) / self.fs * BAUD)
                    self._revert_in = int(64 * self.fs / BAUD)
                    self._ev(f"caller's AA->CC reversal: MT = {self._mt:.0f} symbols")
            else:
                self._prev_age += 1
            return
        if self.state == "ac2":
            self._gone_blocks = self._gone_blocks + 1 if abs(a18) * 2 < 100 else 0
            if self._gone_blocks >= 2:                 # amplitude drop: the caller has stopped
                self._begin_training1()

    def _begin_training1(self) -> None:
        self.tx.s_length = 256
        self.tx.start_training_now(gap_symbols=16)
        self._tx_started = True
        self._tx_tm0 = self._txn
        self.state = "tx1"
        self._rx_mode = "wait_caller"
        self._quiet_for_s = 0
        self._ev("caller quiet: sending S, S-bar, TRN, R1")

    # -- state machine --------------------------------------------------------------------
    def _step(self) -> None:
        if self.state == "failed":
            return
        if self._t / self.fs > ANSWER_TIMEOUT_SECONDS and not self.connected and self.state not in ("data", "down"):
            return self._fail("handshake timed out")
        if self.connected and not self._retraining:
            if self._rx_power < 0.01 * self._ref_power:
                self._quiet_blocks += 1
                if self._quiet_blocks > 150:
                    self.connected = False
                    self.state = "down"
                    self._ev("carrier lost")
            else:
                self._quiet_blocks = 0
        self._echo_step()
        mode = self._rx_mode
        if mode == "wait_caller":
            self._wait_caller()
        elif mode == "locate_c":
            self._locate_caller()
        elif mode == "train_c":
            self._train_known(self._rx_tap, 7000, next_mode="dd2")
        elif mode == "dd":
            self._dd()
        self._trim()

    def _wait_caller(self) -> None:
        """The caller keeps silent until it has seen our R1, then starts S: energy rising from quiet."""
        if self._rx_power < self.SILENT_FRAC * self._ref_power:
            self._quiet_for_s += 1
            return
        if self._rx_power > 0.1 * self._ref_power and self._quiet_for_s >= 25 \
                and (self._txn - self._tx_tm0) / self.fs > (256 + 16 + 1280) / BAUD:
            self._s_seen_at = self._t / self.fs - 0.03
            self.tx.stop()                              # cease transmitting on detecting S
            self.state = "rx_wait"
            self._rx_mode = "locate_c"
            self._ev("caller's S detected: we stop and wait for its TRN")
        elif self._rx_power > 0.1 * self._ref_power:
            self._quiet_for_s = 0

    def _locate_caller(self) -> None:
        predicted = self._s_seen_at + (self._mt + 256 + 16 + 40) / BAUD
        loc = self._locate(predicted, 0.5, self._rx_tap)
        if loc == "wait":
            return
        if loc is None or loc[0] < 0.5:
            return self._fail("caller's TRN not found")
        score, self._s0, self._ph, f_hz = loc
        self._k = self._s0
        self._trained = 0
        self._centroid0 = None
        self._tmse, self._tmse_min = 10.0, 1e9
        self._retune(f_hz)
        self._ev(f"caller's TRN located (corr {score:.2f})")
        self._rx_mode = "train_c"
        self.state = "train_c"

    # -- rate signals ------------------------------------------------------------------------
    def _scan_words(self) -> None:
        if self._phase == "r2":
            w = self._repeated_rate_word()
            if w is None:
                return
            self._r2 = w
            agreed = w & self.my_rate_word
            rate = next((r for r in IMPLEMENTED_RATES if agreed & RATE_BITS[r]), 0)
            self._ev(f"R2 = 0x{w:04X}")
            if not rate:
                return self._fail("no common rate")
            self._rate = rate
            self._e_expected = 0xF000 | RATE_WORD_BASE | RATE_BITS[rate]
            self.tx.rate_word = RATE_WORD_BASE | RATE_BITS[rate]              # R3: the chosen rate only
            self.tx.s_length = 256
            self.tx.start_training_now(gap_symbols=0)                         # second S, S-bar, TRN, then R3
            self.state = "tx2"
            self._phase = "e"
            self._ev(f"sending second training and R3 for {rate} bps")
            return
        if self._phase == "e" and self._word_at_end() == self._e_expected and self._nbits % 2 == 0:
            self.tx.request_e(self._rate)
            self.bit_rate = self._rate
            self.connected = True
            self._ev(f"E received: connected at {self._rate} bps")
            if self._rate == 7200 or self._rate in self._trellis_rates():
                from v32_rx import ViterbiStream
                self._trellis = ViterbiStream(self._rate)
                self._dd_points = self._points_for(self._rate)
            self._deframe = type(self._deframe)()
            self.state = "data"
            self._phase = "data"

    @staticmethod
    def _trellis_rates():
        from v32_caller import TCM_BITS
        return TCM_BITS

    @staticmethod
    def _points_for(rate):
        from v32_caller import V32_TCM, ZTAB4
        return ZTAB4 if rate == 7200 else list(V32_TCM[rate].values())
