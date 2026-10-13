"""V.34 answer-modem start-up and data mode (ITU-T V.34 11.2.1.2, 11.3.1.2, 11.4.1.2), the mirror of v34_call.V34Caller.

Streaming at 8 kHz like the caller: ``tx(n)`` / ``rx(x)``, events logged as (sample, text). The clock starts when V.8
has finished (CJ received); ``V34Answerer`` itself starts with the 75 ms silence that ends phase 1.

  silence75 -> INFO0a (repeat only for recovery) -> tone A -> A reversal -> (caller's B reversal) -> A reversal
  -> L1 L2 -> tone B heard -> A, A reversal, silence -> (caller's L1 L2) -> tone A -> INFO1c in -> INFO1a -> phase 3
  (S S-bar PP TRN J; caller's S->S-bar ends our J) -> read the caller's PP TRN J -> S S-bar TRN MP ... E B1 data.
"""

from __future__ import annotations

import copy
import math
import os
import threading

import numpy as np

import v34_data
import v34_receive
import v34_datarx
import v34_info as vi
import v34_negotiation
import v34_mp
import v34_rx
import v34_tx
import v34_viterbi
from v34_call import (FS, WIN, P4_MP_BURNIN_SYMBOLS, ArraySrc, Silence, ToneSrc, _QamSrc,
                     analyze_probe, probing_signal)

# How long to hunt for the caller's L1 before giving up and proceeding anyway. The call modem sends
# L1 10 ms after its tone B phase reversal (11.2.1.1.6), so this is generous by two orders of
# magnitude; it exists only so a probing signal we cannot recognise cannot stall phase 2 forever.
PROBE_WAIT = 0.45

# Bands occupied by the L1/L2 comb (10.1.2.4 Table 17) and by the caller's tone B carrier. The comb
# is detected as energy in its own bands relative to the tone B band, which is what separates the
# two signals: tone B is one tone, L1/L2 is twenty-one.
_PROBE_COMB_HZ = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950, 2100, 2250,
                  2550, 2700, 2850, 3000, 3150, 3300, 3450, 3600, 3750)
_PROBE_B_HZ = 1200.0


def _comb_onset(seg: np.ndarray, block: int = 40, thresh_db: float = 10.0) -> int | None:
    """Offset of the caller's L1 inside ``seg``, by comb-versus-tone-B energy ratio.

    Returns None when nothing comb-like is present. Scanning block by block and requiring the ratio
    to hold for two consecutive blocks keeps one noisy window from triggering it. A modest power
    step is tried first, at a threshold real peers actually meet.
    """
    n = len(seg)
    if n < block * 4:
        return None
    x = seg.astype(float)
    # 1) a level rise, as the original detector looked for, but with a threshold real peers meet
    pw = np.convolve(x ** 2, np.ones(block) / block, mode="valid")
    base = float(np.median(pw[:block])) if len(pw) > block else 0.0
    if base > 0:
        for k in range(block, len(pw) - block, block):
            if pw[k:k + block].mean() > base * 1.30:
                return int(k)
    # 2) the comb by spectrum: energy outside tone B, relative to energy inside it
    w = np.hanning(block)
    f = np.fft.rfftfreq(block, 1 / FS)
    in_b = np.abs(f - _PROBE_B_HZ) <= 60
    comb = np.zeros(block // 2 + 1, dtype=bool)
    for fc in _PROBE_COMB_HZ:
        comb |= np.abs(f - fc) <= 30
    comb &= ~in_b
    prev = False
    for k in range(0, n - block, block):
        S = np.abs(np.fft.rfft(x[k:k + block] * w)) ** 2
        at_b, in_comb = float(S[in_b].sum()), float(S[comb].sum())
        now = in_comb > 0 and 10 * math.log10(in_comb / max(at_b, 1e-30)) > thresh_db
        if now and prev:
            return k - block
        prev = now
    return None


class ToneASrc:
    """Tone A (2400 Hz, 1 dB below nominal) plus the 1800 Hz guard tone at nominal power; reversal flips only A."""

    def __init__(self, amp: float):
        self.a = ToneSrc(2400.0, amp * 10 ** (-1 / 20))
        self.g = ToneSrc(1800.0, amp)
        self.guard = True

    def read(self, n: int) -> np.ndarray:
        return self.a.read(n) + (self.g.read(n) if self.guard else 0.0)

    def reverse(self) -> None:
        self.a.reverse()


class ToneTracker:
    """Coherent windows of the received audio at ``freq`` Hz: steady tone and phase reversals (to the sample)."""

    def __init__(self, freq: float):
        self.f = freq
        self.w: list[complex] = []
        self.coh: list[float] = []
        self.lvl: list[float] = []
        self.reversals: list[float] = []
        self.cessations: list[float] = []
        self.done = 0

    def steady(self, j: int) -> bool:
        return j >= 0 and self.coh[j] > 0.45 and self.lvl[j] > 60

    def feed(self, rx: np.ndarray) -> None:
        total = len(rx) // WIN
        k = np.arange(WIN)
        while self.done < total:
            i = self.done * WIN
            z = rx[i:i + WIN] * np.exp(-2j * math.pi * self.f * (i + k) / FS)
            self.w.append(z.sum())
            self.coh.append(abs(z.sum()) / (np.abs(z).sum() + 1e-9))
            self.lvl.append(float(np.abs(z).sum() / WIN))
            self.done += 1
            self._detect(rx)

    def steady_for(self, windows: int) -> bool:
        j = self.done - 1
        return j >= windows and all(self.steady(m) for m in range(j - windows + 1, j + 1))

    def _cessation_sample(self, rx: np.ndarray, j: int) -> int | None:
        """Sample at which a previously steady tone stopped, on a 2 ms magnitude grid."""
        sub = 16                                              # 2 ms at 8 kHz
        steady = max(self.lvl[m] for m in range(j - 12, j - 2))
        if steady <= 0:
            return None
        k = np.arange(sub)
        lo = max(0, (j - 2) * WIN)
        hi = min(len(rx), (j + 1) * WIN)
        last_hi = first_lo = None
        for i in range(lo, hi - sub, sub):
            seg = rx[i:i + sub]
            mag = abs((seg * np.exp(-2j * math.pi * self.f * (i + k) / FS)).sum()) / sub
            if mag > 0.5 * steady:
                last_hi = i
            elif last_hi is not None:
                first_lo = i
                break
        if last_hi is None or first_lo is None:
            return None
        return (last_hi + first_lo) // 2

    def _detect(self, rx: np.ndarray) -> None:
        j = self.done - 1
        if j < 12:
            return
        if self.reversals and (j - 1) * WIN - self.reversals[-1] < 400:
            return
        if all(self.steady(m) for m in range(j - 11, j - 1)):
            if self.steady(j) and self.steady(j - 2) and (self.w[j] * np.conj(self.w[j - 2])).real < 0:
                i0 = (j - 2) * WIN
                n = 3 * WIN
                tt = i0 + np.arange(n)
                x = rx[i0:i0 + n]
                c = np.cos(2 * math.pi * self.f * tt / FS)
                sn = np.sin(2 * math.pi * self.f * tt / FS)
                best, at = None, WIN
                for s in range(WIN - 4, 2 * WIN + 5):
                    sg = np.where(np.arange(n) < s, 1.0, -1.0)
                    m = np.stack([sg * c, sg * sn], axis=1)
                    coef = np.linalg.lstsq(m, x, rcond=None)[0]
                    err = float(np.sum((x - m @ coef) ** 2))
                    if best is None or err < best:
                        best, at = err, s
                self.reversals.append(float(i0 + at))
                return
        # Keep loss of carrier separate: cessation alone does not prove a phase reversal.
        if all(self.steady(m) for m in range(j - 12, j - 2)) and not self.steady(j - 1) and not self.steady(j):
            at = self._cessation_sample(rx, j)
            if at is not None:
                self.cessations.append(float(at))


class V34Answerer:
    def __init__(self, rms: float = 900.0, info0: dict | None = None, log=None, symbol_rates=None, data_rate: int = 4800):
        self.rms = rms
        self.amp = rms * math.sqrt(2)
        self.data_rate = data_rate
        self._local_mp = v34_negotiation.offer(data_rate)
        self.negotiated = None
        self.ans_rate = data_rate                            # initial transmit rate ceiling
        self.rx_rate = 0                                     # negotiated call->answer rate; 0 until MP lands
        # Phase-4 training constellation, latched when phase 4 starts (see _start_phase4).  A snapshot rather
        # than a property of data_rate: the MP exchange may lower data_rate, but the training signals already
        # on the wire stay on the constellation J announced.
        self._p4_c = v34_tx.constellation_for_rate(data_rate)
        self._rx_p4_c = self._p4_c
        self.symbol_rates = symbol_rates
        self.t = 0
        self.rx_t = 0
        self._retrain_first_b_retries = 0                   # 11.2.2.2.2: a missed first B reversal reconditions twice before the call fails
        self.events: list[tuple[int, str]] = []
        self._log = log
        self._info0 = info0 or dict(rate2743=1, rate2800=1, rate3429=1, lo3000=1, hi3000=1, lo3200=1, hi3200=1,
                                    allow3429=1, power_reduction=1, pts1664=1)
        self.stage = "silence75"
        self._queue: list = []
        self._src = Silence(int(0.075 * FS))
        self._actions: list = []
        self._rx = np.zeros(0)
        self.lost_too_long = False
        self._freeze = None
        if os.environ.get('V34_FREEZE_GUARD') == '1':
            from v34_freeze_guard import FreezeGuard
            self._freeze = FreezeGuard()
        self.tone_b = ToneTracker(1200.0)
        self.bit_source = None
        self.bit_sink = None
        self.info0c: dict | None = None
        self.info1c: dict | None = None
        self._info_checked = 0
        self._info0c_at = None                        # rx position of the INFO0c frame already answered
        self._info0c_rep_checked = 0
        self.a_rev1_t: int | None = None
        self.rtd_samples: int | None = None
        self.l1_start: int | None = None
        self.probe_result: dict | None = None
        self.mp_call: dict | None = None
        self._mp_ack = 0
        self._mp_ack_sent = False
        self.e_sent = False
        self.rx_bytes = bytearray()
        self._app = bytearray()
        self._app_bits: list[int] = []
        self._rx_state = None
        self._rx_b1 = None
        self._p3 = None
        self._p4rcv = None
        self._jp_scan = None
        # What the caller's phase 3 told us, learned in _p3_poll just before _start_phase4: the abs/diff
        # split (TRN end) and the J pattern, which together are what the J' search needs.
        self.call_trn_len = 0
        self.call_j_pattern = v34_tx.j_pattern_name(data_rate)
        self._p3_last = 0
        self._t_a = None
        self._tracking_packet_poll = os.environ.get('V34_TRACKING_PACKET_POLL') == '1'
        self._connected_rate_enabled = os.environ.get('V34_CONNECTED_RATE') == '1'
        self._tx_lock = threading.RLock()
        if self._tracking_packet_poll:
            self._ev('packet polling enabled for acquired answerer phase-four tracking')

    def respond_to_retrain(self, info0c):
        """Fresh-core V.34 11.5.2.2 response, retaining INFO0 capabilities.

        Do not send V.8 or INFO0a again: 70 ms silence then tone A, followed
        by the ordinary 11.2.1.2.3 reversal/ranging path.
        """
        self.info0c = dict(info0c)
        # Retained capabilities came from an earlier core, not an INFO0 frame
        # in this core's PCM coordinates. Enable detection of new repetitions.
        self._info0c_at = -FS
        self._retrain_first_b_retries = 0
        self.stage = "retrain_silence"
        self._play(Silence())
        def tone_a():
            self._play(ToneASrc(self.amp))
            self._t_a = self.t
            self.stage = "toneA"
            self._ev("retrain: 70 ms silence sent, tone A")
        self._at(self.t+int(.070*FS), tone_a)

    def begin_retrain(self, info0c):
        """Initiate 11.5.2.1 on a fresh core with retained capabilities."""
        self.respond_to_retrain(info0c)

    # -- scheduling (same as the caller) ------------------------------------------------------------------
    def _ev(self, text: str) -> None:
        self.events.append((self.rx_t, text))
        if self._log:
            self._log(f"[{self.rx_t / FS:7.3f}s] {text}")

    def _play(self, src, then=None) -> None:
        self._src = src
        self._queue = list(then or [])

    def _at(self, tx_sample: int, fn) -> None:
        self._actions.append((tx_sample, fn))
        self._actions.sort(key=lambda a: a[0])

    def tx(self, n: int) -> np.ndarray:
        if self._connected_rate_enabled:
            with self._tx_lock:
                return self._tx_locked(n)
        return self._tx_locked(n)

    def _tx_locked(self, n: int) -> np.ndarray:
        out, need = [], n
        while need > 0:
            if self._actions and self._actions[0][0] <= self.t:
                _, fn = self._actions.pop(0)
                fn()
                continue
            limit = need if not self._actions else min(need, self._actions[0][0] - self.t)
            chunk = self._src.read(limit)
            if len(chunk) < limit:
                if self._queue:
                    self._src = self._queue.pop(0)
                    if len(chunk):
                        out.append(chunk)
                        self.t += len(chunk)
                        need -= len(chunk)
                    continue
                chunk = np.concatenate([chunk, np.zeros(limit - len(chunk))])
            out.append(chunk)
            self.t += limit
            need -= limit
        return np.concatenate(out) if out else np.zeros(0)

    def _freeze_hold(self) -> None:
        """Hold or release equaliser adaptation as the freeze guard asks (the equaliser's own freeze: taps, carrier and timing coast)."""
        p4 = getattr(self, '_p4rcv', None)
        eq = getattr(p4, 'eq', None)
        if eq is None:
            return
        if self._freeze.holding:
            if not getattr(self, '_freeze_held', False):
                self._freeze_saved_adapt, self._freeze_held = eq.adapt, True
            eq.adapt = False
        elif getattr(self, '_freeze_held', False):
            eq.adapt, self._freeze_held = self._freeze_saved_adapt, False

    def rx(self, x) -> None:
        x = np.asarray(x, dtype=float)
        if self._freeze is not None and getattr(self, '_rx_state', None) is not None:
            x = self._freeze.filter(x, self.rx_t)
            event, self._freeze.pending_event = self._freeze.pending_event, None
            if event is not None:
                self._ev('carrier stream lost: repeated or all-zero audio replaced by silence, adaptation held' if event[0] == 'start'
                         else f'carrier stream resumed after {event[1]} lost blocks')
            self._freeze_hold()
            if (not self.lost_too_long and self._freeze.run_samples
                    >= float(os.environ.get('V34_FREEZE_LIMIT_S', '1.0')) * FS):
                self.lost_too_long = True
                self._ev(f'carrier stream lost for more than {os.environ.get("V34_FREEZE_LIMIT_S", "1.0")} s: this data link is gone')
        self._rx = np.concatenate([self._rx, x])
        self.rx_t += len(x)
        if getattr(self,'_rx_state',None) is None and getattr(self, '_connected_rate', None) is None:
            self.tone_b.feed(self._rx)
        self._step()

    # -- phase 2 ----------------------------------------------------------------------------------------------
    def _send_info0a(self) -> None:
        ack = 1 if self.info0c is not None else 0
        bits = vi.build_info0(**{**self._info0, "ack": ack})
        sig = vi.InfoModulator("answer", amplitude=self.amp, guard=True).modulate(bits, tail_bits=0)
        self._play(ArraySrc(sig))

    def _try_info0c(self) -> None:
        if self.rx_t < int(0.14 * FS) or self.rx_t - self._info_checked < FS // 50:
            return
        self._info_checked = self.rx_t
        lo = max(0, self.rx_t - int(1.5 * FS))
        got = vi.InfoDemodulator("call", phases=8).decode(self._rx[lo:], "info0")
        if got:
            self.info0c = got[-1]
            self._info0c_at = lo + (self.info0c["_bit_index"] + self.info0c["_phase"] / 8) * FS / vi.BAUD
            self._ev(f"INFO0c received (ack {self.info0c['ack']})")

    def _info0c_repeated(self) -> bool:
        """V.34 11.2.2.2.1: a repeated INFO0c means the caller did not get our INFO0a, so repeat INFO0a.

        Only before the first Tone B reversal (the rule applies in 11.2.1.2.2-.4), once per distinct repeat, and not
        for the INFO0c already answered. V34_INFO0_RECOVERY=0 disables it.
        """
        if os.environ.get("V34_INFO0_RECOVERY", "1") == "0" or self._info0c_at is None:
            return False
        if self.rx_t < int(0.14 * FS) or self.rx_t - self._info0c_rep_checked < FS // 50:
            return False
        self._info0c_rep_checked = self.rx_t
        lo = max(0, self.rx_t - int(1.0 * FS))
        for frame in sorted(vi.InfoDemodulator("call", phases=8).decode(self._rx[lo:], "info0"), key=lambda f: f["_bit_index"]):
            at = lo + (frame["_bit_index"] + frame["_phase"] / 8) * FS / vi.BAUD
            if at > self._info0c_at + 0.2 * FS:
                self._info0c_at = at
                self.info0c = frame
                self._ev("repeated INFO0c received: repeating INFO0a (11.2.2.2.1)")
                return True
        return False

    def _step(self) -> None:
        st = self.stage
        if st == "silence75" and self._src.done:
            self.stage = "info0a"
            self._send_info0a()
        elif st == "info0a":
            self._try_info0c()
            if self._src.done:
                self._play(ToneASrc(self.amp))
                self._t_a = self.t
                self.tone_b.reversals.clear()
                self.stage = "toneA"
                self._ev("INFO0a sent: tone A")
        elif st == "toneA":
            # Duplex 11.2.1.2.3: initiate A reversal after >=50 ms of A.
            if self.info0c is None:
                self._try_info0c()
                if self.info0c is None and self.tone_b.steady_for(12):
                    # 11.2.2.2.1: recover a lost INFO0c by repeating INFO0a.
                    # The caller then returns INFO0c with its acknowledgement set.
                    self.stage = "info0a"
                    self._send_info0a()
                    return
            if self.info0c is not None and self._info0c_repeated():
                self.stage = "info0a"
                self._send_info0a()
                return
            if self.info0c is not None and self.tone_b.steady_for(12) and self.t >= self._t_a + int(0.050 * FS):
                self.tone_b.reversals.clear()
                self._src.reverse()
                self.a_rev1_t = self.t
                self._a_rev1_rx = self.rx_t
                self.stage = "wait_first_b"
                self._ev("first tone A phase reversal sent; waiting for B reversal")
        elif st == "wait_first_b":
            if not self.tone_b.reversals and self._info0c_repeated():
                self.stage = "info0a"
                self._send_info0a()
                return
            if self.tone_b.reversals:
                t_brev = int(self.tone_b.reversals[0])
                self.rtd_samples = max(0, t_brev - self._a_rev1_rx - int(0.040 * FS))
                delay = max(0, t_brev + int(0.040 * FS) - self.rx_t)
                self._at(self.t + delay, self._second_a_reversal)
                self.stage = "a_rev1"
                self._ev(f"round trip delay estimate {self.rtd_samples / FS * 1000:.1f} ms")
            elif self.rx_t - self._a_rev1_rx >= 2 * FS:
                if (hasattr(self, '_retrain_first_b_retries')
                        and self._retrain_first_b_retries < 2):
                    # 11.2.2.2.2 requires detecting B again and returning to
                    # 11.2.1.2.3, not abandoning the call after a lost reversal.
                    self._retrain_first_b_retries += 1
                    self.tone_b.reversals.clear()
                    self._t_a = self.t
                    self.stage = "toneA"
                    self._ev("first B reversal timeout: reconditioning for tone B (11.2.2.2.2)")
                else:
                    self.failed_reason = "first B reversal timeout"
                    self.stage = "failed"
        elif st == "l_tx":
            self._l_tx_poll()
        elif st == "wait_b_rev":
            self._wait_b_rev()
        elif st == "probe_rx":
            self._probe_rx()
        elif st == "wait_info1c":
            self._try_info1c()
        elif st == "info1a_tx" and self._src.done and not self._queue:
            self._start_phase3()
        elif st == "p3_tx":
            self._p3_poll()
        elif st in ("p4", "data"):
            self._p4_poll()

    def _second_a_reversal(self) -> None:
        self._src.reverse()
        self._at(self.t + int(0.010 * FS), self._start_l)
        self._ev("second tone A phase reversal sent")

    def _start_l(self) -> None:
        l1 = probing_signal(0.160, self.rms * 2.0)
        l2 = probing_signal(0.550 + (self.rtd_samples or 0) / FS, self.rms)
        self._play(ArraySrc(l1), [ArraySrc(l2), Silence()])
        self.stage = "l_tx"
        self._l_t0 = self.t
        self.tone_b.reversals.clear()
        self._ev("L1 then L2 started")

    def _l_tx_poll(self) -> None:
        l2_start = self._l_t0 + int(0.160 * FS)
        if self.t < l2_start + int(0.1 * FS):
            return
        if self.tone_b.steady_for(8):
            self._ev("tone B heard: ending L2, A, A reversal")
            tone = ToneASrc(self.amp)
            self._play(tone)
            self._at(self.t + int(0.050 * FS), self._a_rev3)
            self.stage = "wait_b_rev"
            self.tone_b.reversals.clear()

    def _a_rev3(self) -> None:
        self._src.reverse()
        self._at(self.t + int(0.010 * FS), lambda: self._play(Silence()))
        self._ev("third tone A phase reversal sent, then silence")

    def _wait_b_rev(self) -> None:
        if self.tone_b.reversals and self.rx_t > self.tone_b.reversals[-1] + int(0.04 * FS):
            self.probe_start = int(self.tone_b.reversals[-1])
            self.stage = "probe_rx"
            self._ev("caller's tone B reversal: expecting L1/L2")

    def _probe_rx(self) -> None:
        """Find the caller's L1, but never let a missing L1 deadlock the call.

        L1 is the 21-tone comb of 10.1.2.4 Table 17 on a 150 Hz grid, while the caller is on a single
        1200 Hz tone B, so the comb separates from tone B by *spectrum* even when its level barely
        moves. That matters because the original test -- short-term power exceeding 2x the tone B
        background -- never fires against real peers: across the corpus the measured step is only
        1.1-1.8x, well under the 2x asked for, let alone the 6 dB (4x) 10.1.2.4 specifies. A real
        PSTN compresses a 21-tone comb hard against a single tone, so the power ratio is not a
        reliable cue and the comb/B-tone ratio is used instead.

        If nothing looks like an L1 within PROBE_WAIT we give up on measuring the channel and proceed
        anyway. 11.2.1.2.8 proceeds to tone A and waits for INFO1c, so a probing signal we
        failed to recognise must cost us a channel measurement (and therefore a lower offered rate,
        via the snr fallback in _send_info1a) rather than the whole call.
        """
        t0 = self.probe_start
        if self.l1_start is None:
            hi = min(len(self._rx), t0 + int(0.15 * FS))
            seg = self._rx[t0:hi]
            if len(seg) >= int(0.15 * FS):
                hit = _comb_onset(seg)
                if hit is not None:
                    self.l1_start = t0 + int(hit)
                    self._ev(f"caller's L1 detected at sample {self.l1_start} "
                             f"(+{hit / FS * 1000:.0f} ms)")
            if self.l1_start is None:
                if self.rx_t < t0 + int(PROBE_WAIT * FS):
                    return
                self._ev(f"no caller L1 within {PROBE_WAIT * 1000:.0f} ms; proceeding with a "
                         f"conservative rate")
                self._goto_wait_info1c()
                return
        l2_start = self.l1_start + int(0.160 * FS)
        if self.rx_t >= l2_start + int(0.30 * FS):
            self.l2 = self._rx[l2_start + int(0.03 * FS):l2_start + int(0.30 * FS)]
            self.probe_result = analyze_probe(self.l2)
            self._ev(f"probe analysis: {self.probe_result}")
            self._goto_wait_info1c()

    def _goto_wait_info1c(self) -> None:
        self._play(ToneASrc(self.amp))
        self.stage = "wait_info1c"
        self._info_checked = self.rx_t

    def _try_info1c(self) -> None:
        if self.rx_t - self._info_checked < FS // 5:
            return
        self._info_checked = self.rx_t
        lo = max(0, self.rx_t - int(1.2 * FS))
        got = vi.InfoDemodulator("call", phases=8).decode(self._rx[lo:], "info1c")
        if not got:
            return
        self.info1c = got[-1]
        self._ev(f"INFO1c received: {[b for b in self.info1c['blocks']]}")
        # INFO1c bits 12:14 are "Minimum power reduction to be implemented by the answer modem
        # transmitter" -- they command *our* transmit level, the mirror of the INFO1a field the
        # caller applies. We parsed power_min and then ignored it, so we transmitted at nominal while
        # the caller had asked us to come down. 10.1.2: phase-3 signals use the negotiated power
        # level.
        #
        # power_min only: power_extra is what the caller's *receiver* can tolerate on top, so using it
        # as a further instruction would be second-guessing the peer with its own number.
        red_db = max(0, min(7, int(self.info1c.get("power_min", 0) or 0)))
        if red_db:
            self.rms /= 10 ** (red_db / 20.0)
            self.amp = self.rms * math.sqrt(2)
            self._ev(f"INFO1c requests {red_db} dB transmit power reduction; applied")
        self._send_info1a()

    @property
    def freq_offset_units(self) -> int:
        l2 = getattr(self, "l2", None)
        if l2 is None or len(l2) < 400:
            return 0
        n = len(l2) // 2
        t = np.arange(len(l2)) / FS
        ref = np.exp(-2j * math.pi * 1050.0 * t)
        off = np.angle((l2[n:2 * n] * ref[n:2 * n]).sum() * np.conj((l2[:n] * ref[:n]).sum())) / (2 * math.pi * n / FS)
        return int(max(-511, min(511, round(off / 0.02))))

    def _send_info1a(self) -> None:
        blocks = self.info1c["blocks"]
        nominal = (2400, 2743, 2800, 3000, 3200, 3429)
        best = 0
        for i, b in enumerate(blocks):
            if b["max_rate"] > 0 and (self.symbol_rates is None or nominal[i] in self.symbol_rates):
                best = i
        blk = blocks[best]
        self.sym_ans = self.sym_call = best
        self.call_baud = self.ans_baud = v34_tx.SYMBOL_RATES[best]
        self.ans_fc = v34_tx.CARRIERS[nominal[best]][blk["hi_carrier"]]            # the caller asked for this carrier
        hi_call = 1 if nominal[best] == 3429 else 0
        self.call_fc = v34_tx.CARRIERS[nominal[best]][0 if nominal[best] == 3429 else hi_call]
        snr = self.probe_result["snr_db"] if self.probe_result else 30.0
        snr_rate = math.floor(math.log2(1 + 10 ** ((snr - 7.8) / 10)))
        max_rate = max(1, min(14, int(snr_rate * nominal[best] / 2400), self.data_rate // 2400))
        self.ans_rate = 2400 * max_rate                 # bit/s we will send at; picks the phase-3 J pattern
        info = vi.build_info1a(preemph=0, max_rate=max_rate, sym_ans=best, sym_call=best, hi_carrier=hi_call,
                               freq_offset=self.freq_offset_units)
        sig = vi.InfoModulator("answer", amplitude=self.amp, guard=True).modulate(info, tail_bits=2)
        self._play(ArraySrc(sig), [Silence(int(0.070 * FS))])
        self.stage = "info1a_tx"
        self._ev(f"INFO1a sent: {nominal[best]} baud, ans carrier {self.ans_fc:.1f}, call carrier {self.call_fc:.1f}")

    # -- phase 3 ----------------------------------------------------------------------------------------------
    def _start_phase3(self) -> None:
        self._qam = v34_tx.StreamQam(self.ans_baud, self.ans_fc, self.rms)
        from v34_preemphasis import PreEmphasis
        index = self.info1c["blocks"][self.sym_ans].get("preemph", 0) if self.info1c else 0
        self._qam.preemphasis = PreEmphasis(self.ans_baud, self.ans_fc, index)
        trn, sc = v34_tx.trn_symbols(3000, "answer")
        last_i = v34_tx.trn_last_rotation(trn, 4)
        self._j_sc, self._j_z = sc, last_i
        self._unit_states: list = []
        self._qam.push(np.concatenate([v34_tx.s_signal(128), v34_tx.s_signal(16, True),
                                       v34_tx.pp_signal() * math.sqrt(2), trn]))
        self._qam.fill = self._fill_j
        self._play(_QamSrc(self._qam))
        self.stage = "p3_tx"
        self.p3_start = self.rx_t
        self._p3 = None
        self._flip_checked = 0
        self._ev("phase 3: sending S, S-bar, PP, TRN, J")

    def _fill_j(self, qam) -> None:
        pre = (copy.deepcopy(self._j_sc), self._j_z)
        pat = v34_tx.J["J_16PT" if self._rx_p4_c == 16 else "J_4PT"]
        sy, z = v34_tx.j_symbols(pat, 1, "answer", self._j_sc, self._j_z)
        self._j_z = z
        self._unit_states.append((qam.rendered_symbols + len(qam.queue), pre))
        if self.stage=='data' and self._rx_state is not None:
            self._unit_states=self._unit_states[-2:]
        qam.push(sy)

    def _p3_poll(self) -> None:
        """Wait for the caller's S -> S-bar transition (it ends our J), then read its PP, TRN and J."""
        if self._p3 is None:
            if self.rx_t - self.p3_start < int(0.5 * FS) or self.rx_t - self._flip_checked < FS // 100:
                return
            self._flip_checked = self.rx_t
            win = self._rx[-int(0.25 * FS):]
            res = v34_rx.detect_s_flip(win, self.call_fc, self.call_baud)
            if res is None or res[0] != "flip":
                return
            flip_abs = len(self._rx) - len(win) + res[1]
            self._ev("caller's S -> S-bar transition: silence, reading its PP/TRN/J")
            self._play(Silence())
            self._p3 = {"flip": flip_abs, "R": None}
            return
        p3 = self._p3
        spl = FS / self.call_baud
        if p3["R"] is None:
            x0 = max(0, p3["flip"] - int(0.075 * FS))
            x = self._rx[x0:]
            if len(x) < int((144 + 288 + 780) * spl):
                return
            R = v34_rx.Phase3Rx(self.call_baud, self.call_fc, "call")
            got = R.acquire(x, search=int(0.25 * FS))
            if got is None:
                return
            R._z = R.update_baseband(x)
            # We are the answerer receiving the *caller's* TRN, and clause 7 selects the polynomial by
            # modem role: the caller transmits GPC = 1+x^-18+x^-23, the answerer GPA = 1+x^-5+x^-23.
            # Phase3Rx was constructed with sender_role="call", so train() would already build the
            # caller's reference -- but detect it rather than trust it, for the same reason the caller
            # side does: peers in the Gough collection emit the other polynomial, and a blind
            # assumption costs us the whole phase-4 chain when it is wrong.
            mse, trn_role = R.train_trn_auto()
            if trn_role != R.role:
                self._ev(f"caller TRN does not match the expected {R.role} scrambler; "
                         f"using {trn_role} instead")
            p3.update(R=R, x0=x0, rcv=v34_rx.Phase3Receiver(R, 144 + 288, trn_role))
            self._ev(f"caller's S/PP found (corr {got['metric']:.2f})")
            return
        R, rcv = p3["R"], p3["rcv"]
        R._z = R.update_baseband(self._rx[p3["x0"]:])
        rcv.feed()
        if rcv.j is None:
            return
        # rcv.j[3] is the abs/differential split (the TRN end), which is what the phase-4 receive chain
        # needs; rcv.j[0] is where J starts, and on a real modem that is well past it.
        self.call_trn_len = rcv.j[3]
        self.call_j_pattern = rcv.j[2]
        self._ev(f"caller's TRN length {self.call_trn_len}, {rcv.j[2]} from symbol {rcv.j[0]}; "
                 f"sending S, S-bar, TRN, MP")
        self._start_phase4()

    # -- phase 4 ----------------------------------------------------------------------------------------------
    def _start_phase4(self) -> None:
        self._qam = v34_tx.StreamQam(self.ans_baud, self.ans_fc, self.rms)
        from v34_preemphasis import PreEmphasis
        index = self.info1c["blocks"][self.sym_ans].get("preemph", 0) if self.info1c else 0
        self._qam.preemphasis = PreEmphasis(self.ans_baud, self.ans_fc, index)
        # The caller's J requests our transmit constellation (10.1.3.3).
        c = self._p4_c = 16 if self.call_j_pattern == "J_16PT" else 4
        trn, sc = v34_tx.trn_symbols(1500, "answer", constellation=c)
        last_i = v34_tx.trn_last_rotation(trn, c)
        self._mp_sc, self._mp_z = sc, last_i
        # 11.4.1.2.1 is sequential: transmit S for 128T, condition the receiver to detect J' followed by
        # TRN and turn on 107, and THEN transmit S-bar for 16T followed by TRN. S-bar is not sent now
        # because it is not ours to send until the caller's J' proves it is listening -- and a caller
        # that waits for our S->S-bar transition before sending J' would otherwise be waiting for a
        # transition we are waiting on it to cause. S alone goes out; S-bar + TRN follow once J' lands,
        # in _p4_poll.
        self._p4_trn = trn
        # S is 4-point (10.1.3.7), like the phase-3 sequence before it, so the transmitter keeps
        # the 4-point normaliser for it.
        self._qam.push(v34_tx.s_signal(128))
        # 11.4.1.2.1 is sequential and unconditional: S for 128T, condition the receiver to detect J'
        # followed by TRN and turn on 107, and *then* transmit S-bar for 16T followed by TRN.
        #
        # S-bar used to be held back until the caller's J' arrived, on the reasoning that a caller
        # waiting for the S-to-S-bar transition would otherwise be deadlocked against us. But
        # 11.4.1.1.1 is precisely that caller -- it sends J' *after* detecting S followed by S-bar --
        # so withholding S-bar deadlocked against every conformant peer: six of the clips that stall
        # in phase 4 never produce a J' because they are still waiting for the S-bar we were waiting
        # on them for.
        self._qam.push(v34_tx.s_signal(16, True))
        self._qam.set_constellation(self._p4_c)
        self._qam.push(self._p4_trn)
        self._qam.fill = None            # nothing after TRN until our receiver is trained
        self._play(_QamSrc(self._qam))
        self.stage = "p4"
        self._p4_state = "jprime"
        self._p4_last = self.rx_t
        # Seed the incremental J' search on the J run the caller is already transmitting.
        self._jp_scan = v34_rx.JPrimeScanner(self._p3["rcv"].idx, "call", self.call_trn_len, self.call_j_pattern)

    def _fill_mp(self, qam) -> None:
        if self.e_sent and self._mp_ack_sent:
            sy, self._mp_z = v34_tx.bit_symbols([1] * 20, "answer", self._mp_sc, self._mp_z, constellation=self._p4_c)
            qam.push(sy)
            self._start_data_mode(qam)
            return
        # Capability fields remain stable through MP/MP'; only ACK changes.
        bits = v34_mp.build_mp(1, ack=self._mp_ack, **getattr(self,'_retrain_local_mp',self._local_mp))
        sy, self._mp_z = v34_tx.bit_symbols(bits, "answer", self._mp_sc, self._mp_z, constellation=self._p4_c)
        qam.push(sy)
        self._mp_ack_sent = bool(self._mp_ack)

    def _p4_poll(self) -> None:
        if self._rx_state is not None or getattr(self, '_connected_rate', None) is not None:
            self._data_rx()
            return
        # Acquisition/J-prime retains its original probe cadence. Once a
        # phase-four receiver exists, avoid accumulating multiple RTP frames
        # of symbol tracking/MP work in one already-connected retrain tick.
        if (not (self._tracking_packet_poll and self._p4rcv is not None)
                and self.rx_t - self._p4_last < FS // 20):
            return
        self._p4_last = self.rx_t
        p3 = self._p3
        R, rcv3 = p3["R"], p3["rcv"]
        R._z = R.update_baseband(self._rx[p3["x0"]:])
        if self._p4_state == "jprime":
            eq = rcv3.eq
            while eq.k < eq.available():
                y = eq.step(decide=v34_rx.decide4)
                r = int(round(-math.atan2((y / v34_tx.P0).imag, (y / v34_tx.P0).real) / (math.pi / 2))) % 4
                rcv3.idx.append(r)
                # 10.1.3.4: J' is generated by the 4-point rule of 10.1.3.3 whatever J announced, which
                # is why it can be hunted on the phase-3 slicer before the phase-4 constellation is
                # known.  Checking every 8 symbols bounds the overrun: Phase4Receiver rewinds eq.k to the
                # TRN start, so running on far past J' would ask the delay line for symbols it no longer
                # holds.  The scanner keeps those repeated checks cheap - only the new symbols get
                # descrambled and only new bit positions examined - so this is no longer a full rescan.
                if len(rcv3.idx) % 8 == 0 and len(rcv3.idx) > rcv3.j[0] + 40:
                    if self._jp_scan.update(rcv3.idx) is not None:
                        break
            jp = self._jp_scan.found
            if jp is not None:
                k_trn_rel = jp + 8
                self._ev(f"caller's J' at symbol {jp}; S-bar and TRN already sent, reading its MP")
                # S-bar and TRN already went out with S in _start_phase4, per 11.4.1.2.1; there is
                # nothing left to owe the caller here, only a receiver to point at its phase-4 signal.
                # fill stays None: 11.4.1.2.2 -- "The modem shall transmit TRN for at least 512T but
                # no longer than 2000 ms plus a round trip delay. It shall then send MP sequences."
                # MP waits on our receiver being trained on the caller's TRN, so _p4_poll installs
                # the fill once the equalizer has burned in. Filling on our own queue draining sends
                # MP before the caller is trained, and its first MP frames are lost.
                self._qam.fill = None
                # Our J requested the caller's transmit training constellation,
                # independently of the caller's request for our transmitter.
                self._p4rcv = v34_rx.Phase4Receiver(R, rcv3.k_trn + k_trn_rel, "call", eq=eq,
                                                   constellation=self._rx_p4_c)
                self._p4_state = "mp"
            return
        rcv = self._p4rcv
        rcv.observer=getattr(self,'receive_dsp',None)
        rcv.feed(None)
        if self._qam.fill is None and len(rcv.idx) >= P4_MP_BURNIN_SYMBOLS:
            # Equalizer burned in on the caller's TRN, so 11.4.1.2.2 lets us stop TRN and send MP.
            # Not trn_end: that is latched from mismatches against the expected TRN index and only
            # appears once the caller sends MP, so gating on it deadlocks both sides.
            self._qam.fill = self._fill_mp
            self._ev(f"trained on caller's TRN ({len(rcv.idx)} symbols); sending MP")
        if rcv.trn_end is None:
            return
        frames = rcv.frames
        selected_mp = v34_negotiation.final_mp(frames)
        if selected_mp is not None and (self.mp_call is None or
                any(self.mp_call.get(k) != v for k, v in selected_mp.items() if k != "ack")):
            if self.e_sent:
                self.failed_reason = "peer changed MP parameters after E"
                self.stage = "failed"
                return
            self.mp_call = dict(selected_mp)
            self._mp_ack = 1
            try:
                self._prepare_data_mode()
            except ValueError as exc:
                self.failed_reason = str(exc)
                self.stage = "failed"
                return
            self._ev(f"caller's MP received: {frames[0]}")
        if selected_mp is not None:
            self.mp_call = {**(self.mp_call or {}), **selected_mp}
        if any((isinstance(f, dict) and f["ack"]) or f == "E" for f in frames) and not self.e_sent:
            self._ev("caller's MP' / E received: sending E then data")
            self.e_sent = True
        if rcv.b1_symbol is not None and self._rx_b1 is None:
            self._rx_b1 = rcv.b1_symbol
            self._ev(f"caller's E seen; B1 at symbol {self._rx_b1}")
        if self._rx_b1 is not None and self.stage == "data":
            self._data_rx()

    # -- data mode ----------------------------------------------------------------------------------------------
    def _prepare_data_mode(self) -> None:
        peer = self.mp_call
        if peer is None:
            raise ValueError('cannot enter data mode before receiving MP')
        # Minimal dictionaries used by startup tests default to enabled local features.
        peer = {**self._local_mp, **(peer or {})}
        self.negotiated = v34_negotiation.resolve(
            getattr(self,'_retrain_local_mp',self._local_mp), peer, "answer", self.ans_baud, self.call_baud,
            getattr(self, "ans_fc", 0.0), getattr(self, "call_fc", 0.0))
        self.data_rate = self.negotiated.tx.rate
        self.rx_rate = self.negotiated.rx.rate
        self._ev(f"negotiated: tx={self.data_rate} rx={self.rx_rate} "
                 f"trellis tx={self.negotiated.tx.trellis} rx={self.negotiated.rx.trellis}")
        h = v34_data.precoder_from_mp(self.negotiated.tx.precoder)
        self._dtx = v34_data.DataTx(self.data_rate, self.ans_baud,
                                   expanded=self.negotiated.tx.shaping, precoder=h,
                                   nonlinear=self.negotiated.tx.nonlinear,
                                   trellis=self.negotiated.tx.trellis)
        self._dsc = v34_tx.Scrambler("answer")
        self._dscale = math.sqrt(v34_tx.constellation_power(self._p4_c) / self._dtx.avg_power)
        N = self._dtx.bits_per_data_frame
        b1 = [self._dsc.step(1) for _ in range(N)]
        self._b1_sy = np.array(self._dtx.data_frame(b1)) * self._dscale

    def _start_data_mode(self, qam) -> None:
        qam.push(self._b1_sy)
        qam.fill = self._fill_data
        self.stage = "data"
        self._ev(f"E sent, B1 sent; data mode at {self.data_rate} bit/s ({self.ans_baud:.1f} baud)")

    def feed_bytes(self, data: bytes) -> None:
        self._app += data

    def _fill_data(self, qam) -> None:
        N = self._dtx.bits_per_data_frame
        if self.bit_source is not None:
            batch=getattr(self,'_batch_bit_source',None)
            raw = (batch[1](N) if batch is not None and batch[0] is self.bit_source
                   else [self.bit_source() for _ in range(N)])
        else:
            while len(self._app_bits) < N:
                if self._app:
                    v = self._app.pop(0)
                    self._app_bits += [0] + [(v >> i) & 1 for i in range(8)] + [1]
                else:
                    self._app_bits += [1] * 10
            raw = self._app_bits[:N]
            del self._app_bits[:N]
        qam.push(np.array(self._dtx.data_frame([self._dsc.step(b) for b in raw])) * self._dscale)

    def _data_rx(self) -> None:
        rate_controller = None
        if getattr(self, '_connected_rate_enabled', False):
            from v34_rate_live import controller
            rate_controller = controller(self, 'answer')
            if rate_controller is not None and rate_controller.poll():
                return
        eq = self._p4rcv.eq
        R = self._p3["R"]
        from v34_streaming import update_data_baseband, trim_data_pcm
        update_data_baseband(self,R,self._p3['x0'])
        rate = self.rx_rate
        if self._rx_state is None:
            if not v34_receive.initialize(self, eq, "call", self.call_baud):
                return
        st = self._rx_state
        drx, vit = st["drx"], st["vit"]
        limit = rate_controller.limit(eq.available()) if rate_controller is not None else eq.available()
        while eq.k < limit:
            y = eq.step(decide=self._decide)
            st["ys"].append(y)
            if len(st["ys"]) == 2:
                vit.feed(st["ys"][0], st["ys"][1])
                st["ys"] = []
            while vit.decided:
                u0, u1 = vit.decided.pop(0)
                if vit.emit_states:
                    st.setdefault('pend_states', []).append(vit.decided_states.pop(0))
                st["pend"] += [u0, u1]
                if len(st["pend"]) == 8:
                    states = st.pop('pend_states', None)
                    bits, _ = (drx.decode_frame(st['pend'], trellis_states=states) if states is not None
                               else drx.decode_frame(st['pend']))
                    st["pend"] = []
                    raw = v34_datarx.descramble_stream(bits, st["hist"][-23:], "call")
                    st["hist"] = (st["hist"] + bits)[-23:]
                    if self.bit_sink is not None:
                        self.bit_sink(raw)
                    else:
                        st["bits"] += raw
        v34_receive.observe_quality(self, st, eq)
        self._async_bytes(st)
        trim_data_pcm(self,eq,self._p3['x0'])

    def _async_bytes(self, st) -> None:
        b = st["bits"]
        i = 0
        while i + 10 <= len(b):
            if b[i] == 0 and b[i + 9] == 1:
                self.rx_bytes.append(sum(b[i + 1 + j] << j for j in range(8)))
                i += 10
            else:
                i += 1
        del b[:i]
