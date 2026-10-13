"""V.34 call-modem start-up, Phase 2 (ITU-T V.34 11.2): INFO0 exchange, tone A/B reversals, round trip delay, probing.

Streaming at 8 kHz: ``tx(n)`` returns the next n transmit samples, ``rx(x)`` takes the next block of received audio.
All timing is counted in samples. Events are logged as (sample, text). This file is built stage by stage against a real
V.34 answerer; the stages implemented so far are listed in ``stage``:

  silence75 -> info0 -> toneB -> (A reversal) -> b_reversal -> silent -> (second A reversal) -> probing -> ...
"""

from __future__ import annotations

import copy
import math
import os
import threading

import numpy as np

import v34_info as vi
import v34_data
import v34_receive
import v34_datarx
import v34_negotiation
import v34_mp
import v34_rx
import v34_viterbi
import v34_tx

FS = 8000
WIN = 20                                                   # analysis window: 2.5 ms = 6 cycles of 2400 Hz
L_TONES = {150: 0, 300: 180, 450: 0, 600: 0, 750: 0, 1050: 0, 1350: 0, 1500: 0, 1650: 180, 1950: 0, 2100: 0,
           2250: 180, 2550: 0, 2700: 180, 2850: 0, 3000: 180, 3150: 180, 3300: 180, 3450: 180, 3600: 0, 3750: 0}


def probing_signal(duration_s: float, rms: float) -> np.ndarray:
    """L1/L2 (10.1.2.4, Table 17): 21 cosines 150 Hz apart; ``rms`` is the rms of the whole signal."""
    t = np.arange(int(round(duration_s * FS))) / FS
    sig = sum(np.cos(2 * math.pi * f * t + math.radians(p)) for f, p in L_TONES.items())
    return sig * (rms / math.sqrt(np.mean(sig ** 2)))


class ToneSrc:
    def __init__(self, freq: float, amp: float, phase: float = 0.0):
        self.w = 2 * math.pi * freq / FS
        self.amp, self.phase = amp, phase

    def read(self, n: int) -> np.ndarray:
        ph = self.phase + self.w * np.arange(n)
        self.phase = (self.phase + self.w * n) % (2 * math.pi)
        return self.amp * np.cos(ph)

    def reverse(self) -> None:
        self.phase = (self.phase + math.pi) % (2 * math.pi)


class ArraySrc:
    def __init__(self, data: np.ndarray):
        self.data, self.pos = data, 0

    def read(self, n: int) -> np.ndarray:
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out                                          # shorter than n when exhausted

    @property
    def done(self) -> bool:
        return self.pos >= len(self.data)


class Silence(ArraySrc):
    def __init__(self, n: int | None = None):
        super().__init__(np.zeros(n if n is not None else 1))
        self.endless = n is None

    def read(self, n: int) -> np.ndarray:
        if self.endless:
            return np.zeros(n)
        return super().read(n)

    @property
    def done(self) -> bool:
        return not self.endless and self.pos >= len(self.data)


# How long after INFO1c we keep looking for the answerer's S / S-bar / PP burst. This was 2.0 s,
# which was too tight for part of the chipset set: peers that reach phase 3 briskly put S/PP about
# 0.45 s after our INFO1c, but conexant-hsfi, swann-speeddemon and conexant-rs56 put it 3.3-3.8 s
# after it, so the old window never contained their burst at all. Widening it does locate the burst
# (peak lag 26611-29479 instead of being pinned at ~11000 on our own echo) but does NOT yet complete
# acquisition for them -- the correlation there is only ~0.14-0.19 against a 0.6 threshold, versus
# ~0.68 when the same burst is scanned straight out of the capture. That gap points at a residual
# sample offset between the tx clock we derive x0 from and the rx buffer, which is still open.
# Searching longer is safe: the S reference is 432 symbols long and min_metric still gates the lock.
P3_SEARCH_SECS = 8.0

# How long to wait after confirming J before starting our own phase-3 S. 11.3.1.1.3 allows "up to
# 500 ms"; this is deliberately far below that, because the answerer's deadline runs on its own clock
# and we are the slow half of the path.
#
# Budget, all measured on the live TuxedoCat calls (see pcm_capture's tx_minus_rx_origin_s, without
# which these numbers cannot be separated):
#     answerer's J start                          13.719 s (rx clock)
#     our rx sees it                              +280 ms  -- Asterisk's jitter buffer is on the
#                                                                    receive side only, so our rx is
#                                                                    ~280 ms behind our tx
#     confirm J, 31 symbols                       +9 ms
#     the 500 ms wait, then 50 ms guard silence   +550 ms
#     our tx to the peer                          +6 ms
#     ------------------------------------------------------------------
#     our S -> S-bar reaches the peer              ~845 ms
#     its deadline, 600 ms + RTD (286 ms)          886 ms
#
# So the old setting landed at ~845 ms against an 886 ms deadline: inside, by 37 ms. That is the
# deadlock. It is not a large lag -- it is a 4% margin, which is why it looks intermittent and why
# every earlier attempt to reason about it from a single capture went wrong.
#
# Cutting the wait to 50 ms puts S on the line at ~395 ms after the answerer's J started and turns a
# 37 ms margin into roughly 490 ms. Per 10.1.3.3 the answerer repeats J "until interrupted", so
# overlapping its J is the intended behaviour, not a collision to be avoided by waiting.
P3_POST_J_WAIT_SECS = 0.050

# How many symbols of the answerer's phase-4 TRN to run through the equalizer before we start MP.
# 11.4.1.1.2 permits continuing TRN for up to 2000 ms before "ceasing to transmit TRN and send
# sequence MP", i.e. the wait is on our receiver converging, and 2000 ms at 3200 baud is ~6400
# symbols -- so this is a conservative fraction of the allowance. It must be a burn-in on the
# equalizer and not trn_end: trn_end is detected from mismatches against the expected TRN index,
# which only start once the *peer* sends MP, so gating on it deadlocks (both sides wait for MP
# that neither will send).
P4_MP_BURNIN_SYMBOLS = 256

P4_TRN_SYMBOLS = 1500                                    # the caller's phase-4 TRN (11.4.1.1.2: at least 512T, at most 2000 ms)


def withhold_mp(env=None) -> bool:
    """TEST HOOK ``V34_TEST_WITHHOLD_MP=1``: never send MP, so the answerer cannot enter data mode (used to ask whether a path
    fault appears without a data phase, lab/V34_QUEENS_LIVE_GUARD_CHECK.md). After the queued TRN the caller then sends silence."""
    return (os.environ if env is None else env).get("V34_TEST_WITHHOLD_MP") == "1"


def p4_trn_symbol_count(baud: float, env=None) -> int:
    """Symbols of TRN the caller sends in phase 4. ``V34_P4_TRN_SYMBOLS`` lengthens it for experiments (lab/V34_QUEENS_LIVE_GUARD_CHECK.md:
    does the Queens freeze move with the start of data?); the result is held between 512 symbols and 2000 ms of symbols.
    Without the variable it is the long-standing 1500, whatever the baud rate."""
    env = os.environ if env is None else env
    raw = env.get("V34_P4_TRN_SYMBOLS")
    if raw is None:
        return P4_TRN_SYMBOLS
    try:
        want = int(raw)
    except ValueError:
        return P4_TRN_SYMBOLS
    return max(512, min(want, int(2.0 * baud)))




def mp_acknowledged(frames) -> bool:
    """True once the answerer has acknowledged our MP: an MP' (ack 1), or an E that follows one.

    A bare E does not count: 20 ones in the descrambled stream are an E to the receiver, and a steady tone decodes to such a run
    (lab/V34_LCP1_REPLAY.md)."""
    seen = False
    for f in frames:
        if isinstance(f, dict) and f["ack"]:
            seen = True
        elif f == "E" and seen:
            return True
    return seen


class V34Caller:
    def __init__(self, rms: float = 900.0, info0: dict | None = None, log=None, symbol_rates=None, data_rate: int = 4800):
        self.data_rate = data_rate
        self._data_rate_initial = data_rate
        self._local_mp = v34_negotiation.offer(data_rate)
        self.negotiated = None
        self.rx_rate = 0
        self.call_rate = data_rate                           # initial transmit rate ceiling
        # Phase-4 training constellation, latched when phase 4 starts.  It is deliberately a snapshot and not a
        # property of data_rate: the MP exchange can lower data_rate (see _fill_mp) but the training signals
        # already on the wire stay on the constellation J announced, so it must not move mid-phase-4.
        self._p4_c = v34_tx.constellation_for_rate(data_rate)
        self._rx_p4_c = self._p4_c
        self.bit_source = None                              # optional: callable -> next line bit (V.42 layer)
        self.bit_sink = None                                # optional: callable(list of descrambled bits)
        # Low rates fit the mandatory 2400-baud channel. Prefer its narrower
        # occupied band; an explicit symbol-rate request remains available for diagnostics.
        self.symbol_rates = symbol_rates if symbol_rates is not None else ([2400] if data_rate <= 4800 else None)
        self.rms = rms
        self.amp = rms * math.sqrt(2)                      # peak amplitude of a nominal-power sinusoid
        self.t = 0                                         # samples transmitted so far
        self.rx_t = 0
        self.events: list[tuple[int, str]] = []
        self.stage = "silence75"
        self._log = log
        self._tracking_packet_poll = os.environ.get('V34_TRACKING_PACKET_POLL') == '1'
        self._connected_rate_enabled = os.environ.get('V34_CONNECTED_RATE') == '1'
        if self._tracking_packet_poll:
            self._ev('packet polling enabled for acquired phase-three/four tracking')
        self._info0 = info0 or {}
        self._queue: list = []                             # sources to play after the current one
        self._src = Silence(int(0.075 * FS))
        self._actions: list[tuple[int, object]] = []       # (tx sample, callable)
        self._rx = np.zeros(0)                             # all received audio (float)
        self.lost_too_long = False                         # the freeze guard saw the stream lost for longer than V34_FREEZE_LIMIT_S
        self._freeze = None
        self.answerer_retrain = False                      # data mode: the answerer sent tone A for > ANSWERER_RETRAIN_S (it initiated a retrain)
        self._tone_a_run = 0
        if os.environ.get('V34_FREEZE_GUARD') == '1':
            from v34_freeze_guard import FreezeGuard
            self._freeze = FreezeGuard()
        self._wins = 0                                     # analysis windows processed
        self._w2400: list[complex] = []                    # per-window coherent sum at 2400 Hz
        self._coh: list[float] = []
        self._lvl: list[float] = []
        self.info0a: dict | None = None
        self._info_checked = 0
        self._info0a_last = None
        self._info0_ack_sent = False
        self._info0_repeats = 0                          # INFO0c repeats sent while INFO0a was missing (11.2.2.1.1)
        self._info0_repeat_t = None
        self.b_reversal_t: int | None = None
        self._b_reversal_rx: int | None = None
        self.rtd_samples: int | None = None
        self._a_rev_seen = 0
        self.a_reversals: list[float] = []
        self.probe_start: int | None = None
        self._toneb2_t0 = None
        self.b2_reversal_t = None
        self.l_started = False
        self.l2_start_t = 1 << 60
        self.l2_planned_end_t = 1 << 60                           # tx sample at which the planned L2 ends
        self._b2_win = None                               # receive analysis window of the second tone B reversal
        self._p3_j_at: int | None = None             # rx_t when the peer's J was first decoded
        self.info1a = None
        self._concealed = []                              # (start sample, length) of concealed rx audio, in rx_t units
        self.l1_start = None
        self._probe_retries = 0
        self._info1_retries = 0
        self._phase_retrains = 0
        self.e_sent = False
        self._tx_underrun_seen = 0
        self._tx_lock = threading.RLock()                  # the DCE's sender thread calls tx() while rx() splices the TX queue
        # Set when the handshake cannot continue (e.g. the peer picked a symbol rate we do not
        # implement). V34Modem reads this and fails the call; nothing else should advance.
        self.failed_reason: str | None = None

    # -- helpers ----------------------------------------------------------------------------------------------
    def _ev(self, text: str) -> None:
        self.events.append((self.rx_t, text))
        if self._log:
            self._log(f"[{self.rx_t / FS:7.3f}s] {text}")

    def _play(self, src, then=None) -> None:
        self._src = src
        self._queue = list(then or [])

    # -- transmit ---------------------------------------------------------------------------------------------
    def tx(self, n: int) -> np.ndarray:
        with self._tx_lock:
            out = self._tx_locked(n)
            u = getattr(self._qam, 'underrun_symbols', 0) if getattr(self, '_qam', None) is not None else 0
            if u > self._tx_underrun_seen:
                self._ev(f"TX underrun: {u - self._tx_underrun_seen} zero symbols rendered at stage {self.stage}")
                self._tx_underrun_seen = u
            return out

    def _tx_locked(self, n: int) -> np.ndarray:
        out = []
        need = n
        while need > 0:
            if self._actions and self._actions[0][0] <= self.t:
                _, fn = self._actions.pop(0)
                fn()
                continue
            limit = need if not self._actions else min(need, self._actions[0][0] - self.t)
            chunk = self._src.read(limit)
            if len(chunk) < limit:                         # source exhausted
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

    def _at(self, tx_sample: int, fn) -> None:
        self._actions.append((tx_sample, fn))
        self._actions.sort(key=lambda a: a[0])

    # -- receive ----------------------------------------------------------------------------------------------
    def note_concealed(self, start: int, n: int) -> None:
        """The receiver replaced ``n`` lost samples starting at ``start`` (this caller's rx_t clock) with concealment audio."""
        self._concealed.append((int(start), int(n)))
        if len(self._concealed) > 256:
            del self._concealed[:-256]

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

    ANSWERER_RETRAIN_S = 0.080                              # 11.5.1.2: respond after more than 50 ms of tone A; 80 ms rides out one block
    TONE_A_MIN_RMS = 100.0
    TONE_A_LINE_FRACTION = 0.9

    @classmethod
    def _is_tone_a(cls, x) -> bool:
        """Tone A at 2400 Hz, with optional guard for peer interoperability.

        V.34 10.1.2.1 specifies an 1800 Hz guard. SmartLink's observed
        data-mode retrain omits it. Require a strong 2400 Hz component and
        concentrated line energy; 1800 Hz alone is never a request.
        The opt-in watcher separately requires 80 ms of consecutive blocks.
        """
        n = len(x)
        if n < 80 or float(np.sqrt(np.mean(x * x))) < cls.TONE_A_MIN_RMS:
            return False
        k = np.arange(n)
        e = [abs(np.dot(x, np.exp(-2j * math.pi * f * k / FS))) ** 2 * 2.0 / n for f in (2400.0, 1800.0)]
        total = float(np.dot(x, x))
        return e[0] > 0.1 * max(e) and sum(e) >= cls.TONE_A_LINE_FRACTION * total

    def _watch_for_answerer_retrain(self, x) -> None:
        """Data mode only (V34_ANSWERER_RETRAIN=1): tone A is how the answerer starts a retrain (11.5.2.1); data audio is broadband."""
        if self.answerer_retrain or (os.environ.get('V34_ANSWERER_RETRAIN', '0') != '1'
                and not getattr(self, '_connected_rate_enabled', False)):
            return
        self._tone_a_run = self._tone_a_run + len(x) if self._is_tone_a(x) else 0
        if self._tone_a_run >= self.ANSWERER_RETRAIN_S * FS:
            self.answerer_retrain = True
            self._ev(f'answerer sent tone A for {self._tone_a_run / FS * 1000:.0f} ms in data mode: it initiated a retrain (11.5.2.1)')

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
        rate_active = getattr(self, '_connected_rate', None) is not None
        if getattr(self, '_rx_state', None) is not None or rate_active:
            self._watch_for_answerer_retrain(x)
        self._rx = np.concatenate([self._rx, x])
        self.rx_t += len(x)
        if getattr(self,'_rx_state',None) is None and not rate_active:
            self._analyze()
        self._step()

    def _analyze(self) -> None:
        total = len(self._rx) // WIN
        k = np.arange(WIN)
        while self._wins < total:
            i = self._wins * WIN
            seg = self._rx[i:i + WIN]
            z = seg * np.exp(-2j * math.pi * 2400.0 * (i + k) / FS)
            w = z.sum()
            self._w2400.append(w)
            self._coh.append(abs(w) / (np.abs(z).sum() + 1e-9))
            self._lvl.append(float(np.abs(z).sum() / WIN))
            self._wins += 1
            self._detect_reversal()

    def _steady(self, j: int) -> bool:
        return j >= 0 and self._coh[j] > 0.45 and self._lvl[j] > 60

    def _detect_reversal(self) -> None:
        j = self._wins - 1
        if j < 12:
            return
        # windows j-2 and j steady and opposite in phase, window j-1 (the transition) not required to be
        if all(self._steady(m) for m in range(j - 11, j - 1)) is False:
            return
        if not (self._steady(j) and self._steady(j - 2)):
            return
        if (self._w2400[j] * np.conj(self._w2400[j - 2])).real >= 0:
            return
        if self.a_reversals and (j - 1) * WIN - self.a_reversals[-1] < 400:
            return
        # locate the transition inside windows j-2..j: fit +-A cos(wt+phi) with a sign change at s, least squares
        i0 = (j - 2) * WIN
        n = 3 * WIN
        tt = i0 + np.arange(n)
        x = self._rx[i0:i0 + n]
        c, sn = np.cos(2 * math.pi * 2400.0 * tt / FS), np.sin(2 * math.pi * 2400.0 * tt / FS)
        gc, gs = np.cos(2 * math.pi * 1800.0 * tt / FS), np.sin(2 * math.pi * 1800.0 * tt / FS)
        best, at, tone_energy = None, WIN, 0.0
        for s in range(WIN - 4, 2 * WIN + 5):
            sg = np.where(np.arange(n) < s, 1.0, -1.0)
            # Optional guard tone does not reverse with tone A.
            m = np.stack([sg * c, sg * sn, gc, gs], axis=1)
            coef, res, *_ = np.linalg.lstsq(m, x, rcond=None)
            err = float(np.sum((x - m @ coef) ** 2))
            if best is None or err < best:
                best, at = err, s
                tone_energy = float(np.sum((m[:,:2] @ coef[:2])**2))
        t_rev = i0 + at
        # A sign change in one demodulation window can also come from INFO0
        # DPSK during peer recovery. Require the existing three-window fit to
        # explain the waveform; do not wait for additional post-reversal audio.
        # Normalize by fitted A power: a loud optional guard must not hide a
        # bad reversal fit by inflating the denominator.
        nmse = float(best / (tone_energy + 1e-9))
        if not hasattr(self, '_reversal_candidates'):
            self._reversal_candidates = []
        if len(self._reversal_candidates) < 128:
            self._reversal_candidates.append(dict(sample=t_rev, nmse=nmse, accepted=nmse<=0.2))
        if nmse > 0.2:
            return
        self.a_reversals.append(float(t_rev))
        self._ev(f"tone A phase reversal #{len(self.a_reversals)} at sample {t_rev}")

    # -- the state machine ----------------------------------------------------------------------------------
    def _step(self) -> None:
        if self.failed_reason:
            return                                   # terminal: V34Modem turns this into NO CARRIER
        st = self.stage
        if st == "silence75" and self._src.done:
            self.stage = "info0"
            self._send_info0(ack=0)
        elif st in ("info0", "info0_ack"):
            self._try_info0a()
        if self.stage == "info0" and self.info0a is None and self.a_reversals:
            self._info0_recovery()
        if self.stage == "info0" and self.info0a is not None:
            # Normal 11.2.1.1.2 keeps Tone B continuous. Repeating INFO0c
            # belongs to recovery after repeated INFO0a (11.2.2.1.1).
            self.stage = "info0_ack"
            self.a_reversals.clear()
        if (self.stage == "info0_ack" and self.b_reversal_t is None
                and isinstance(self._src, ToneSrc) and self.a_reversals):
            # Duplex 11.2.1.1.3: respond to the FIRST A reversal after 40 ms.
            first = int(self.a_reversals[0])
            delay = max(0, first + int(0.040 * FS) - self.rx_t)
            self.b_reversal_t = self.t + delay
            self._b_reversal_rx = self.rx_t + delay
            self._at(self.b_reversal_t, self._do_b_reversal)
            self._ev(f"scheduled B reversal at tx sample {self.b_reversal_t}")
            self.stage = "b_reversal_wait"
        if self.stage == "b_done" and len(self.a_reversals) >= 2:
            t_ar = int(self.a_reversals[1])
            self.rtd_samples = max(0, t_ar - self._b_reversal_rx - int(0.040 * FS))
            self._ev(f"round trip delay estimate {self.rtd_samples / FS * 1000:.1f} ms")
            self.stage = "probing"
            self.probe_start = t_ar
        elif self.stage == "b_done" and self.rx_t - self._b_reversal_rx >= 2 * FS:
            self._ev("second A reversal timeout: restarting INFO0c exchange")
            self.a_reversals.clear()
            self.b_reversal_t = self._b_reversal_rx = None
            self._info0_ack_sent = True
            self._send_info0(ack=1)
            self.stage = "info0_ack"
        if self.stage == "probing":
            self._probing()
        if self.stage == "toneB2":
            self._toneB2()
        if (self.stage in self.PHASE_RETRAIN_STAGES or self._awaiting_b1_after_e()) and self._phase_retrain_due():
            return
        if self.stage == "info1c_sent":
            self._try_info1a()
        elif self.stage == "phase3":
            self._phase3_rx()
        elif self.stage == "p3_tx":
            self._phase3_wait_s()
        elif self.stage in ("p4_trn", "p4_mp"):
            self._phase4_rx()
        elif self.stage == "data":
            if self._rx_b1 is None:
                self._phase4_rx()
            else:
                self._data_rx()

    INFO0_RECOVERY_GRACE_S = 0.15      # let a late INFO0a decode win before repeating INFO0c
    INFO0_RECOVERY_SPACING_S = 0.30    # INFO0c (82 ms) plus tone B between repeats
    INFO0_RECOVERY_MAX = 10

    def _info0_recovery(self) -> None:
        """V.34 11.2.2.1.1: Tone A detected before INFO0a was received, so repeatedly send INFO0c.

        A single-shot INFO0a that a network loss damaged leaves the caller in stage ``info0`` forever: the answerer has
        gone on to Tone A and its phase reversals, but the caller schedules its Tone B reversal only after INFO0a.
        The standard's remedy is to send INFO0c again (bit 28 stays 0, since INFO0a was not received); the answerer
        answers repeated INFO0c with a repeated INFO0a (11.2.2.2.1, Figure 18/V.34). Bounded, and only after a Tone A
        phase reversal proves the answerer is past its INFO0 and a short grace so a late decode is not pre-empted.
        Set V34_INFO0_RECOVERY=0 to disable.
        """
        if os.environ.get("V34_INFO0_RECOVERY", "1") == "0" or self._info0_repeats >= self.INFO0_RECOVERY_MAX:
            return
        if self.rx_t - self.a_reversals[0] < int(self.INFO0_RECOVERY_GRACE_S * FS):
            return
        if self._info0_repeat_t is not None and self.rx_t - self._info0_repeat_t < int(self.INFO0_RECOVERY_SPACING_S * FS):
            return
        self._info0_repeats += 1
        self._info0_repeat_t = self.rx_t
        self._ev(f"Tone A detected but no INFO0a received: repeating INFO0c (11.2.2.1.1, repeat {self._info0_repeats})")
        self._send_info0(ack=0)

    def _send_info0(self, ack: int) -> None:
        bits = vi.build_info0(**{**self._info0, "ack": ack})
        sig = vi.InfoModulator("call", amplitude=self.amp, guard=False).modulate(bits, tail_bits=2)
        self._play(ArraySrc(sig), [ToneSrc(1200.0, self.amp)])
        self._tone_b = None
        self._ev("sending INFO0c, then tone B")

    def _try_info0a(self) -> None:
        if self.rx_t < int(0.14 * FS) or self.rx_t - self._info_checked < FS // 50:
            return
        self._info_checked = self.rx_t
        lo = max(0, self.rx_t - int(1.5 * FS))
        got = vi.InfoDemodulator("answer", phases=8).decode(self._rx[lo:], "info0")
        if got:
            for frame in sorted(got, key=lambda f: f["_bit_index"]):
                at = lo + (frame["_bit_index"] + frame["_phase"] / 8) * FS / vi.BAUD
                if self._info0a_last is not None and at - self._info0a_last < 4 * FS / vi.BAUD:
                    continue  # phase searches and later polls can find the same frame
                repeated = self.info0a is not None
                self._info0a_last = at
                self.info0a = frame
                self._ev(f"INFO0a received: { {k: v for k, v in frame.items() if v and not k.startswith('_')} }")
                if (repeated and self.stage == 'probing' and self.l1_start is None
                        and at > self.probe_start):
                    # A DPSK transition can pass the short reversal fit. If no
                    # probe follows and a new valid INFO0a arrives, the peer is
                    # explicitly recovering its INFO0 exchange (11.2.2.1.1).
                    # Resume that exchange instead of waiting forever for L1.
                    self.a_reversals.clear()
                    self.b_reversal_t = self._b_reversal_rx = None
                    self.probe_start = None
                    self._probe_retries = 0
                    self._info0_ack_sent = True
                    self._send_info0(ack=1)
                    self.stage = 'info0_ack'
                    self._ev('repeated INFO0a without L1: resuming INFO0 recovery')
                    continue
                if (repeated and not frame.get('ack') and not self.a_reversals
                        and self.stage in ('info0', 'info0_ack')):
                    # 11.2.2.1.1: repeated unacknowledged INFO0a requires
                    # repeated INFO0c, including when our first reply was lost.
                    # Never splice a new reply into an in-flight INFO0c. Only
                    # replace its following Tone B, under the sender lock.
                    # One response per distinct received frame bounds the queue;
                    # acknowledged INFO0a / normal probing stops this recovery.
                    with self._tx_lock:
                        if isinstance(self._src, ToneSrc):
                            self._send_info0(ack=1)
                            self._info0_ack_sent = True

    def _do_b_reversal(self) -> None:
        if isinstance(self._src, ToneSrc):
            self._src.reverse()
        else:                                              # the tone source may still be queued behind INFO0c
            for s in self._queue:
                if isinstance(s, ToneSrc):
                    s.reverse()
        self._ev("tone B phase reversal sent")
        self._at(self.t + int(0.010 * FS), self._b_silent)

    def _b_silent(self) -> None:
        self._play(Silence())
        self.stage = "b_done"
        self._ev("B finished, silent; waiting for the second A reversal")

    def _try_info1a(self) -> None:
        if self.rx_t < self.info1c_sent_t or self.rx_t - self._info_checked < FS // 5:
            return
        self._info_checked = self.rx_t
        lo = max(0, self.info1c_sent_t - FS // 2)
        got = vi.InfoDemodulator("answer", phases=8).decode(self._rx[lo:], "info1a")
        if not got and self._info1_retrain_due():
            return
        if not got and os.environ.get("V34_INFO_HOLE_TOLERANT") == "1":
            # Opt-in: a lost stretch of the single-shot INFO1a's tail (replaced by the receiver's repeat-the-last-frame
            # concealment) leaves its payload intact but fails the CRC (lab/V34_PSTN_EARLY_FAILURES.md). Accept it
            # only with the evidence recover_info1a_concealed / recover_info1a require, and say so.
            spans = [(a - lo, n) for a, n in self._concealed if a + n > lo]
            rec = vi.recover_info1a_concealed(self._rx[lo:], spans) if spans else None
            how = None
            if rec:
                how = (f"{rec['_concealed_samples'] / FS * 1000:.0f} ms of loss concealment over its CRC/fill bits")
            else:
                rec = vi.recover_info1a(self._rx[lo:])
                if rec:
                    how = f"a {rec['_dropout_ms']:.0f} ms media dropout over its CRC/fill bits"
            if rec:
                self._ev(f"INFO1a CRC failed but the frame ends in {how}; payload identical on "
                         f"{rec['_support_phases']} of 8 timing phases and plausible; accepting it UNVERIFIED "
                         f"(V34_INFO_HOLE_TOLERANT=1)")
                got = [rec]
        if got:
            self.info1a = got[0]
            # Log what the peer chose before acting on it: if the rate turns out to be one we do
            # not implement, the reason is only visible in this line.
            self._ev(f"INFO1a received: { {k: v for k, v in got[0].items() if not k.startswith('_')} }")
            if self._setup_phase3():
                self.stage = "phase3"

    INFO1_RETRY_WAIT_S = 0.700
    INFO1_RETRY_MAX = 3

    def _info1_retrain_due(self) -> bool:
        """V.34 11.2.2.1.6 / 11.5.1.2: no INFO1a 700 ms + RTD after INFO1c, answerer still sending Tone A.

        Tone A here means the answerer never decoded our INFO1c and is waiting for a retrain: respond as to a
        retrain, i.e. 70 ms of silence, then Tone B, and restart the ranging (the A phase reversal, RTD, L1/L2,
        INFO1c). Bounded; V34_INFO1_RETRY=0 disables. Returns True if it started the retrain.
        """
        if os.environ.get("V34_INFO1_RETRY", "1") == "0" or self._info1_retries >= self.INFO1_RETRY_MAX:
            return False
        wait = int(self.INFO1_RETRY_WAIT_S * FS) + (self.rtd_samples or 0)
        if self.rx_t < self.info1c_sent_t + wait or not self._tone_a_on():
            return False
        self._info1_retries += 1
        self._ev(f"no INFO1a {(self.rx_t - self.info1c_sent_t) / FS * 1000:.0f} ms after INFO1c and tone A still "
                 f"present: answering the retrain (11.5.1.2), attempt {self._info1_retries}/{self.INFO1_RETRY_MAX}")
        self._respond_to_retrain()
        return True

    def begin_retrain(self) -> None:
        """Initiate a retrain from a fresh caller (V.34 11.5.1.1): 70 ms of silence, then Tone B, then wait for the answerer's
        Tone A and its phase reversals and run the start-up again. Used by V34Modem to replace a data link that was lost."""
        self._ev("initiating a retrain (11.5.1.1): silence 70 ms, then tone B")
        self._respond_to_retrain()

    def _respond_to_retrain(self) -> None:
        """11.5.1.2: silence 70 ms, then tone B, and restart the ranging (A reversal, RTD, L1/L2, INFO1c)."""
        self.a_reversals.clear()
        self.b_reversal_t = self._b_reversal_rx = None
        self.probe_start = self.l1_start = None
        self._probe_retries = 0
        self._actions.clear()
        self._play(ArraySrc(np.zeros(int(0.070 * FS))), [ToneSrc(1200.0, self.amp)])
        self.stage = "info0_ack"

    PHASE4_TONE_SKIP_S = 0.2                                   # steady tone A this long: stop tracking phase 4 (see _phase4_rx)
    PHASE_RETRAIN_MAX = 3
    PHASE_RETRAIN_TONE_S = 0.6                                 # steady tone A this long, after phase-3/4 signals, is a retrain
    PHASE_RETRAIN_STAGES = ("phase3", "p3_tx", "p4_trn", "p4_mp")

    def _tone_a_steady(self, seconds: float) -> bool:
        """The 2400 Hz coherence windows have been steady (tone A) for the last ``seconds``."""
        n = -(-int(seconds * FS) // WIN)
        j = self._wins - 1
        return j >= n + 12 and all(self._steady(m) for m in range(j - n + 1, j + 1))

    def _awaiting_b1_after_e(self) -> bool:
        """We sent E and are in data mode, but the answerer's own E/B1 has not been found (opt-in V34_RETRAIN_AFTER_E=1).

        lab/V34_GU_BATCH.md (gu4): the answerer sent MP' for 2.2 s, got our acknowledgement late, and returned to tone A for the
        rest of the call; our E went out after that, so ``e_sent`` disabled the phase retrain and the data-mode watcher needs B1.
        """
        return (os.environ.get("V34_RETRAIN_AFTER_E") == "1" and self.stage == "data"
                and self.e_sent and self._rx_b1 is None)

    def _phase_retrain_due(self) -> bool:
        """The answerer abandoned phase 3 or 4 and went back to tone A (V.34 11.5.2: it initiates a retrain).

        Tone A is narrow-band and coherent at 2400 Hz; the broadband TRN and MP of phases 3-4 are not (coherence about 0.06
        against more than 0.45). lab/V34_LCP1_REPLAY.md: after 154 unacknowledged MP frames the answerer returned to tone A for
        the last 15 s of the call and the caller never reacted. Respond as to any retrain (11.5.1.2) and restart the ranging with
        the original capabilities. Bounded; V34_PHASE_RETRAIN=0 disables. Returns True if it started the retrain.
        """
        if os.environ.get("V34_PHASE_RETRAIN", "1") == "0" or self._phase_retrains >= self.PHASE_RETRAIN_MAX:
            return False
        if self.e_sent and not self._awaiting_b1_after_e():
            return False
        if not self._tone_a_steady(self.PHASE_RETRAIN_TONE_S):
            return False
        self._phase_retrains += 1
        self._ev(f"answerer returned to tone A during {self.stage} (retrain): answering it (11.5.1.2), "
                 f"attempt {self._phase_retrains}/{self.PHASE_RETRAIN_MAX}")
        # forget phase 3/4 and the MP outcome; the ranging, INFO1c/INFO1a and phase 3 run again
        self.data_rate = self._data_rate_initial
        self._local_mp = v34_negotiation.offer(self.data_rate)
        self.negotiated = None
        self.rx_rate = 0
        self.info1a = None
        self.mp_ans = None
        self.e_sent = False
        self._rx_b1 = None
        self._rx_state = None
        self._p4rcv = None
        self._respond_to_retrain()
        return True

    # -- phase 3 (11.3.1.1) -------------------------------------------------------------------------------------
    def _setup_phase3(self) -> bool:
        """Adopt the peer's symbol rates. False if it chose one we do not implement.

        ``sym_call``/``sym_ans`` are 3-bit indices into the rate set, so a peer may legitimately
        answer 0-6. Index 6 is the V.90 rate (4000 baud) which V.34 does not define and we do not
        implement, and index 7 is reserved. A real dial-up bank does exactly this when it is a
        V.90 bank that ignored our lower offer, so this must end the call cleanly rather than
        raise -- an unhandled KeyError here takes the whole modem process down.
        """
        a = self.info1a
        for key, label in (("sym_call", "call"), ("sym_ans", "answer")):
            idx = a[key]
            if idx not in v34_tx.SYMBOL_RATES:
                self.failed_reason = (
                    f"peer chose symbol rate index {idx} for the {label} direction "
                    f"({'V.90 4000 baud' if idx == 6 else 'reserved'}); V.34 supports 0-5 "
                    f"(2400-3429 baud, up to 33600 bit/s)")
                self._ev("FAILED: " + self.failed_reason)
                return False
        self.call_baud = v34_tx.SYMBOL_RATES[a["sym_call"]]
        self.ans_baud = v34_tx.SYMBOL_RATES[a["sym_ans"]]
        # INFO1a bits 12:14 are "Minimum power reduction to be implemented by the call modem
        # transmitter" and bits 15:17 are the extra reduction the caller's receiver can tolerate --
        # i.e. these command *our* transmit level (10.1.2: "All signals in Phase 3 are transmitted
        # using the selected symbol rate, carrier frequency, pre-emphasis filter, and power level").
        # We parsed them and then ignored them, so we transmitted at nominal while the peer had asked
        # us to come down. On the live TuxedoCat call INFO1a asked for power_min 3 and we measured
        # ourselves 8.4 dB below the Courier's phase-3 level.
        #
        # Only power_min is applied. power_extra is what the *answerer's receiver* can tolerate on top
        # of that, and reading it as an instruction to reduce further would be second-guessing the
        # peer with its own number; the minimum is the explicit instruction.
        red_db = max(0, min(7, int(a.get("power_min", 0) or 0)))
        if os.environ.get("V34_NO_PWR_RED"):
            # Diagnostic only, and deliberately non-conformant: ignore the answerer's requested
            # reduction. It exists to answer one question -- whether the answerer can hear our S at
            # all while it is still transmitting its own J. At the 3 dB it asked for we sit ~14 dB
            # below it; at nominal, ~11 dB. If it has an echo canceller good enough to hear us
            # through that, it should enter phase 4; if not, no level we are permitted to use will
            # reach it and the wait has to move instead. Set V34_NO_PWR_RED=1 in the environment.
            red_db = 0
        if red_db:
            self.rms /= 10 ** (red_db / 20.0)
            self.amp = self.rms * math.sqrt(2)
            self._ev(f"INFO1a requests {red_db} dB transmit power reduction; applied")
        self.call_rate = 2400 * a["max_rate"]          # phase-2 rate ceiling, resolved by MP
        self.call_fc = v34_tx.CARRIERS[round(self.call_baud)][a["hi_carrier"]]
        hi_ans = 0                                           # what we asked for in INFO1c (blocks use hi_carrier 0 except 3429)
        self.ans_fc = v34_tx.CARRIERS[round(self.ans_baud)][1 if round(self.ans_baud) == 3429 else hi_ans]
        self._p3 = v34_rx.Phase3Rx(self.ans_baud, self.ans_fc, "answer")
        self._p3_locked = False
        self._p3_trained = False
        self._p3_last = 0
        self.p4_trn_symbols = p4_trn_symbol_count(self.call_baud)
        self.mp_ans = None
        self.e_sent = False
        self._rx_b1 = None
        self._p4rcv = None
        self._p4_precoded = False
        self.rx_bytes = bytearray()
        self._ev(f"phase 3: answerer {self.ans_baud} baud @ {self.ans_fc} Hz, caller {self.call_baud} baud @ {self.call_fc} Hz")
        return True

    def _phase3_rx(self) -> None:
        if (not (getattr(self, '_tracking_packet_poll', False) and self._p3_trained)
                and self.rx_t - self._p3_last < FS // 10):
            return
        self._p3_last = self.rx_t
        x0 = self.info1c_sent_t
        x = self._rx[x0:]
        R = self._p3
        if not self._p3_locked:
            if len(x) < int(1.2 * FS):
                return
            got = R.acquire(x, search=int(P3_SEARCH_SECS * FS))
            if got is None:
                return
            self._p3_locked = True
            self._ev(f"S/PP found: lag {x0 + got['lag']} corr {got['metric']:.3f}")
            return
        R._z = R.update_baseband(x)
        spl = R.spl
        avail = int((len(x) - R.lag) / spl) - R.LEAD - 20
        if not self._p3_trained:
            if avail < 144 + 288 + 760:
                return
            mse, trn_role = R.train_trn_auto()
            if trn_role != R.role:
                self._ev(f"peer TRN does not match the expected {R.role} scrambler; "
                         f"using {trn_role} instead")
            self._p3_trained = True
            self._p3_rcv = v34_rx.Phase3Receiver(R, role=trn_role)
            self._ev(f"equalizer trained, mse {mse:.3f}")
        rcv = self._p3_rcv
        rcv.feed()
        if rcv.j is not None:
            # rcv.j[3] is the abs/differential split (the TRN end), which is what the phase-4 receive chain
            # needs; rcv.j[0] is where J starts, and on a real modem that is well past it.
            self.ans_trn_len = rcv.j[3]
            self.ans_j_seen = rcv.j[1]
            self.ans_j_pattern = rcv.j[2]
            if self._p3_j_at is None:
                self._p3_j_at = self.rx_t
            self._ev(f"TRN length {self.ans_trn_len}, {rcv.j[2]} from symbol {rcv.j[0]} "
                     f"seen for {rcv.j[1]} symbols (timing slips {rcv.eq.slips})")
        # Once J has been confirmed the deadline stands on its own. rcv.j goes back to None whenever
        # the answerer stops transmitting -- it repeats J "until interrupted" (10.1.3.3) but pauses
        # between repeats, and on the live TuxedoCat call it paused for 400 ms and then for good.
        # Returning early in that state (as this did) meant the 11.3.1.1.3 500 ms bound was only ever
        # evaluated on polls where J happened to be decodable right now, so the wait stretched to
        # whenever J next appeared: measured 2.9 s from the answerer's J start to our S on the wire,
        # against a 600 ms + RTD deadline on its side. Latch the confirmation instead.
        if self._p3_j_at is None:
            return
        # Figure 19/V.34: phase 3 is half-duplex, answerer first. The answerer keeps repeating J
        # "until interrupted" (10.1.3.3), so decoding J does NOT mean it has stopped talking -- its J
        # ran a further ~5000 symbols on the live TuxedoCat call. Starting our S on J decode put us
        # on the wire at 13.36 s while the Courier was still at full level, for 1.8 s, and it never
        # recovered (it ran its 11.3.2.2.1 phase-2 timeout instead).
        #
        # 11.3.1.1.3 bounds the wait at 500 ms after J. That bound is the whole gate.
        #
        # There used to be a second condition here: transmit only once the answerer's carrier had
        # actually dropped (_peer_silent). It was added because starting S on J decode "put us on the
        # wire at 13.36 s while the Courier was still at full level, for 1.8 s, and it never
        # recovered". The duplex timeline says that reasoning was backwards. Per 10.1.3.3 the
        # answerer repeats J *until interrupted*, and the whole point of its 11.3.2.2.1 deadline --
        # the S-to-S-bar transition within 600 ms + RTD of the start of J -- is that the caller is
        # supposed to interrupt it, not wait for it to stop. An answerer only falls silent when that
        # deadline expires, so gating on silence gates on the very timeout we are trying to beat.
        #
        # Measured on the live TuxedoCat capture, off the wire: its J started at 13.719 s and stayed
        # up to 14.78 s, paused 400 ms, resumed to 15.80 s, then went silent permanently. Our S
        # reached the line 845 ms after its J began, against its 886 ms deadline -- a 37 ms margin.
        #
        # An earlier version of this comment read "2.98 s after its J began". That was wrong: it took
        # the index of S in tx.raw and compared it to the index of J in rx.raw, and those two files
        # do not share a clock. The send loop starts before media arrives -- tx_minus_rx_origin_s
        # measures -2.132 s on this path, stable to 20 ms across calls -- so most of that 2.98 s was
        # the gap between the two files' origins. pcm_capture now records the skew and p4_duplex.py
        # applies it; see P3_POST_J_WAIT_SECS above for the budget built on the corrected figure.
        if self.rx_t - self._p3_j_at < int(P3_POST_J_WAIT_SECS * FS):
            return
        self._start_call_training()

    # _peer_silent used to live here and gated the phase-3 transmit on the answerer's carrier having
    # dropped. Removed with the gate: the premise in its docstring was wrong. It said self._rx "includes
    # the local echo of our own transmit", but fitting our transmit back out of the line signal gives
    # an echo gain of 0.007 -- Asterisk's echo canceller is active on the channel, so rx.raw is very
    # nearly the peer alone (debug/p4_duplex.py measures this). And the gate itself waited for the
    # silence that the answerer only reaches by expiring its 11.3.2.2.1 deadline. See the note at the
    # 500 ms bound above.

    def _start_call_training(self) -> None:
        self._qam = v34_tx.StreamQam(self.call_baud, self.call_fc, self.rms)
        from v34_preemphasis import PreEmphasis
        self._qam.preemphasis = PreEmphasis(self.call_baud, self.call_fc, self.info1a.get("preemph", 0))
        S = v34_tx.s_signal(128)
        Sb = v34_tx.s_signal(16, True)
        PP = v34_tx.pp_signal() * math.sqrt(2)
        trn, sc = v34_tx.trn_symbols(1024, "call")
        last_i = v34_tx.trn_last_rotation(trn, 4)
        self._j_sc, self._j_z = sc, last_i
        self._unit_states: list = []
        self._qam.push(np.concatenate([S, Sb, PP, trn]))
        self._qam.fill = self._fill_j
        self._p3_tx_t0 = self.t + int(0.050 * FS)
        self._play(Silence(int(0.050 * FS)), [_QamSrc(self._qam)])
        self.stage = "p3_tx"
        self.p3_tx_started = self.rx_t
        # Earliest rx sample worth searching for the answerer's phase-4 S->S-bar: just past the
        # phase-3 S/PP we locked in phase3_rx, so the phase-3 transition itself cannot match.
        self._p3_search_lo = self.info1c_sent_t + self._p3.lag + int(0.5 * FS)
        self._ev("transmitting S, S-bar, PP, TRN, J")

    def _fill_j(self, qam) -> None:
        """Queue one more 8-symbol J unit; remember the coder state so a later cut can roll back."""
        pre = (copy.deepcopy(self._j_sc), self._j_z)
        pat = v34_tx.J["J_16PT" if self._rx_p4_c == 16 else "J_4PT"]
        sy, z = v34_tx.j_symbols(pat, 1, "call", self._j_sc, self._j_z)
        self._j_z = z
        self._unit_states.append((qam.rendered_symbols + len(qam.queue), pre))
        if self.stage=='data' and getattr(self,'_rx_state',None) is not None:
            self._unit_states=self._unit_states[-2:]
        qam.push(sy)

    def _phase3_wait_s(self) -> None:
        # Test-only (V34_JPRIME_DELAY_MS): J-prime was held back after the answerer's S; the S search stays off until it goes.
        if getattr(self, '_jprime_due', None) is not None:
            if self.rx_t >= self._jprime_due:
                self._jprime_due = None
                self._send_j_prime()
            return
        if self.rx_t - self.p3_tx_started < int((144 / self.call_baud + 0.08) * FS) or self.rx_t - self._p3_last < FS // 100:
            return
        self._p3_last = self.rx_t
        # The answerer's phase-4 S/S-bar may already have gone by while we were still decoding its
        # phase-3 TRN and J: J decode completes only once the TRN end is pinned, which for a long TRN
        # is a second or more after it started. conexant-hsfi decodes J at 17.34 s but the answerer's
        # S->S-bar was at ~17.30 s, so a "look at the last 250 ms" window started after our own
        # transmit can never see it. Scan the whole span since phase 3 instead, bounded below by the
        # S/PP lock so the phase-3 S->S-bar we have already consumed cannot re-trigger this.
        lo = getattr(self, "_p3_search_lo", None)
        hi = len(self._rx)
        if lo is None:
            return
        step = int(0.25 * FS)
        # Scan newest -> oldest so the most recent transition wins. Scanning forward would latch an
        # older match: that regressed bcm and ess, which previously took the latest flip because the
        # old code only ever looked at the final 250 ms.
        for start in range(hi - step, max(lo, hi - int(3.0 * FS)) - 1, -step):
            win = self._rx[start:start + step]
            res = v34_rx.detect_s_flip(win, self.ans_fc, self.ans_baud)
            # Act on S itself, not only on the S -> S-bar transition.
            #
            # 11.4.1.2.1 is sequential and that is the whole point: "The answer modem shall transmit
            # signal S for 128T, condition its receiver to detect sequence J-prime followed by signal
            # TRN, and turn on circuit 107. The modem shall THEN transmit signal S-bar for 16T
            # followed by signal TRN." The answerer's S and S-bar are separated by *our* J-prime, and
            # 11.4.2.2.1 gives it only 100 ms + RTD to receive that J-prime before abandoning the
            # attempt. Waiting for the transition before sending J-prime is waiting for a transition
            # we are ourselves responsible for causing. 11.3.1.1.7 agrees: "condition its receiver to
            # detect signal S. After detecting signal S, the modem shall proceed to Phase 4."
            #
            # This and v34_ans._start_phase4 sending S-bar only after the caller's J' are one change.
            # The caller half alone reaches data mode but fails the loopback, because the answerer was
            # pushing S + S-bar + TRN as a single block and was not listening for J' at that point.
            if res is not None and res[0] in ("S", "flip"):
                what = "S -> S-bar transition" if res[0] == "flip" else "S"
                self._ev(f"answerer {what} detected at rx sample {start + res[1]}; "
                         f"sending J-prime then TRN")
                # The match can belong to an older search window. Preserve
                # its absolute sample coordinate rather than rebasing on hi.
                self._p4_flip_abs = start + res[1]
                # Test-only A/B knob (default 0 = unchanged): hold J-prime for this many milliseconds.
                delay = int(float(os.environ.get('V34_JPRIME_DELAY_MS', '0') or 0) * FS / 1000)
                if delay > 0:
                    self._jprime_due = self.rx_t + delay
                    self._ev(f"J' held {delay * 1000 // FS} ms (test-only V34_JPRIME_DELAY_MS)")
                    return
                self._send_j_prime()
                return

    def _send_j_prime(self) -> None:
        q = self._qam
        # The answerer's J requests our transmit constellation (10.1.3.3).
        c = 16 if self.ans_j_pattern == "J_16PT" else 4
        # The heavy part (modulating the whole TRN) is done BEFORE the transmit queue is touched. The DCE's sender thread keeps
        # calling tx() while this runs on the receive thread; truncating first and computing afterwards left the queue
        # empty for the whole computation and put 60 ms of exact zeros on the wire right after J-prime on the bbs-8 call
        # (lab/V34_PHASE4_TX_HOLE.md). The splice below is short and atomic with tx().
        trn, sc2 = v34_tx.trn_symbols(self.p4_trn_symbols, "call", constellation=c)
        # The differential coder for MP runs from the last TRN symbol. On the 16-point constellation the
        # rotation still takes two bits and the quarter index is separate; the last-TRN rotation index is
        # all the coder needs, as with 4-point.
        last_i = v34_tx.trn_last_rotation(trn, c)
        with self._tx_lock:
            # The chunk rendered last already contains the pulse tails of the next `span` queued symbols, so those must
            # stay as they are: keep one J unit (8 symbols = span), cut after it and roll the scrambler / differential
            # coder back to the state before the first dropped unit.
            keep = q.span
            for pos, pre in self._unit_states:
                if pos >= q.rendered_symbols + keep:
                    self._j_sc, self._j_z = pre
                    break
            q.truncate(keep)
            # 10.1.4: J' is "generated as described in 10.1.3.3" -- and 10.1.3.3 (J) is the *4-point* rule
            # (two bits I1,I2, differential Zn, point 0 rotated Zn*90 deg).  So J' is 4-point even when J
            # announced a 16-point constellation: J' is a phase-3 sequence that terminates J, which is why the
            # answerer can hunt for it on its phase-3 4-point slicer before it knows the phase-4 constellation
            # is anything but 4 or 16.  Only TRN / MP / MP' / E follow J (10.1.3.2, .8, .9).
            jp, z = v34_tx.j_symbols(v34_tx.J_PRIME, 1, "call", self._j_sc, self._j_z, constellation=4)
            self._mp_sc, self._mp_z = sc2, last_i
            self._p4_c = c
            self._mp_ack = 0
            self._mp_ack_sent = False
            q.fill = None
            # MP is not queued yet. 11.4.1.1.2: "After transmitting signal TRN for a minimum of 512T, the
            # modem shall condition its receiver to receive sequence MP and may continue sending TRN for
            # up to 2000 ms. After training adequately, the call modem shall then cease transmitting TRN
            # and send sequence MP." So MP waits on *our receiver* being trained on the answerer's TRN, not
            # on our own TRN running out. Filling with MP the moment our queue drains sends it before the
            # answerer is trained, so its first MP frames are lost and it never answers with MP'.
            #
            # "Trained adequately" must NOT be read as trn_end. trn_end is latched from mismatches against
            # the expected TRN index, and those only appear once the peer starts sending MP -- so gating
            # MP on trn_end is circular and neither modem ever sends MP. It is gated on an equalizer
            # burn-in instead; see P4_MP_BURNIN_SYMBOLS.
            q.push(jp)                       # J' is 4-point even when J announced 16 points (10.1.3.4)
            q.set_constellation(c)           # rescales the queued J' so its level does not step
            q.push(trn)
            self.stage = "p4_trn"
            self._p4_last = self.rx_t
        self._ev("J' queued, then TRN, then MP")


    def _fill_mp(self, qam) -> None:
        """Queue one MP (MP' once the answerer's MP has been received) frame; after the answerer's MP' / E, send E and
        switch to data mode (B1, then data frames)."""
        if self.e_sent and self._mp_ack_sent:
            sy, self._mp_z = v34_tx.bit_symbols([1] * 20, "call", self._mp_sc, self._mp_z, constellation=self._p4_c)
            qam.push(sy)
            self._start_data_mode(qam)
            return
        # Capability fields remain stable through MP/MP'; only ACK changes.
        bits = v34_mp.build_mp(1, ack=self._mp_ack, **getattr(self,'_retrain_local_mp',self._local_mp))
        sy, self._mp_z = v34_tx.bit_symbols(bits, "call", self._mp_sc, self._mp_z, constellation=self._p4_c)
        qam.push(sy)
        self._mp_ack_sent = bool(self._mp_ack)

    # -- data mode transmit -------------------------------------------------------------------------------------
    def _prepare_data_mode(self) -> None:
        peer = self.mp_ans
        if peer is None:
            raise ValueError('cannot enter data mode before receiving MP')
        # Minimal dictionaries used by startup tests default to enabled local features.
        peer = {**self._local_mp, **(peer or {})}
        self.negotiated = v34_negotiation.resolve(
            getattr(self,'_retrain_local_mp',self._local_mp), peer, "call", self.call_baud, self.ans_baud,
            getattr(self, "call_fc", 0.0), getattr(self, "ans_fc", 0.0))
        self.data_rate = self.negotiated.tx.rate
        self.rx_rate = self.negotiated.rx.rate
        self._ev(f"negotiated: tx={self.data_rate} rx={self.rx_rate} "
                 f"trellis tx={self.negotiated.tx.trellis} rx={self.negotiated.rx.trellis}")
        h = v34_data.precoder_from_mp(self.negotiated.tx.precoder)
        self._dtx = v34_data.DataTx(self.data_rate, self.call_baud,
                                   expanded=self.negotiated.tx.shaping, precoder=h,
                                   nonlinear=self.negotiated.tx.nonlinear,
                                   trellis=self.negotiated.tx.trellis)
        self._dsc = v34_tx.Scrambler("call")
        self._dscale = math.sqrt(v34_tx.constellation_power(self._p4_c) / self._dtx.avg_power)
        self._app = bytearray()
        self._app_bits: list[int] = []
        N = self._dtx.bits_per_data_frame
        b1 = [self._dsc.step(1) for _ in range(N)]          # B1: one data frame of scrambled ones
        self._b1_sy = np.array(self._dtx.data_frame(b1)) * self._dscale

    def _start_data_mode(self, qam) -> None:
        qam.push(self._b1_sy)
        qam.fill = self._fill_data
        self.stage = "data"
        self.data_start_t = self.rx_t
        self._ev(f"E sent, B1 sent; data mode at {self.data_rate} bit/s ({self.call_baud} baud, M={self._dtx.M}, b={self._dtx.b})")

    def feed_bytes(self, data: bytes) -> None:
        self._app += data

    def _fill_data(self, qam) -> None:
        N = self._dtx.bits_per_data_frame
        while self.bit_source is None and len(self._app_bits) < N:
            if self._app:
                v = self._app.pop(0)
                self._app_bits += [0] + [(v >> i) & 1 for i in range(8)] + [1]
            else:
                self._app_bits += [1] * 10                  # idle: marks
        if self.bit_source is not None:
            batch=getattr(self,'_batch_bit_source',None)
            raw = (batch[1](N) if batch is not None and batch[0] is self.bit_source
                   else [self.bit_source() for _ in range(N)])
        else:
            raw = self._app_bits[:N]
            del self._app_bits[:N]
        bits = [self._dsc.step(b) for b in raw]
        qam.push(np.array(self._dtx.data_frame(bits)) * self._dscale)

    def _phase4_rx(self) -> None:
        """Follow the answerer's S-bar + TRN with the adaptive equalizer and read its MP / MP' / E."""
        if (not (getattr(self, '_tracking_packet_poll', False) and self._p4rcv is not None)
                and self.rx_t - self._p4_last < FS // 10):
            return
        self._p4_last = self.rx_t
        spl = FS / self.ans_baud
        if self._p4rcv is None:
            # Search for S-bar + TRN only, not S + S-bar + TRN. Per 11.4.1.2.1 the answerer sends S
            # for 128T, waits for *our* J-prime (100 ms + RTD, 11.4.2.2.1), and only then sends
            # S-bar for 16T followed by TRN -- so its S and S-bar are now about a J-prime exchange
            # apart, not adjacent. Anchoring on the S and expecting S-bar next would look for a
            # transition that arrives a second later than the reference allows.
            s_at = self._p4_flip_abs
            # A flip timestamp is a symbol center after the detector has
            # confirmed the inversion. Retain the S-bar pulse/filter context
            # needed by correlation, even when a whole block arrives at once.
            x0 = max(0, s_at - math.ceil(24 * spl) - 40)
            x = self._rx[x0:]
            if len(x) < int((16 + 700 + 120) * spl):
                return
            R = v34_rx.Phase3Rx(self.ans_baud, self.ans_fc, "answer")
            # Our J requests the answerer's phase-4 constellation. Its J
            # independently requests ours; the two preferences may differ.
            c = self._rx_p4_c
            trn, _ = v34_tx.trn_symbols(800, "answer", constellation=c)
            seq = np.concatenate([v34_tx.s_signal(16, True), trn[:200]])
            got = R.acquire(x, search=int(3.0 * FS), seq=seq, min_metric=0.7)
            if got is None:
                return
            R._z = R.update_baseband(x)
            # k_start is the symbol index, within the matched sequence, at which ref begins. The
            # acquire reference *starts* at S-bar (that is the whole point of the reordering), so
            # S-bar is symbol 0 -- passing 16 here trains 16 symbols late and yields mse ~0.98.
            mse = R.train_known(np.concatenate([v34_tx.s_signal(16, True), trn[:700]]), 0,
                                span=v34_rx.phase4_equalizer_span())
            self._p4R, self._p4_x0 = R, x0
            # TRN is 16 symbols after that, since S-bar is 16T.
            self._p4rcv = v34_rx.Phase4Receiver(R, 16, constellation=c)
            if self.data_rate >= 24000 and not __import__("os").environ.get("V34_NOTL"):   # dense constellations need the timing loop (see README)
                self._p4rcv.eq.ktp, self._p4rcv.eq.kti, self._p4rcv.eq.mu = 0.003, 5e-8, 0.1
            self._ev(f"phase 4: answerer S-bar/TRN found (corr {got['metric']:.2f}), equalizer mse {mse:.3f}")
        rcv = self._p4rcv
        R = self._p4R
        x = self._rx[self._p4_x0:]
        R._z = R.update_baseband(x)
        rcv.observer=getattr(self,'receive_dsp',None)
        if self._tone_a_steady(self.PHASE4_TONE_SKIP_S):
            # The answerer has left phase 4 for tone A (a retrain, answered by _phase_retrain_due once the tone has lasted
            # PHASE_RETRAIN_TONE_S). There is nothing to track: the decision-directed equalizer only diverges on a tone and the
            # TRN-end rescan (decode_after_trn) then dominates the CPU, which on the lcp-1/bbs-8 calls put the receive path
            # up to 3.5 s behind real time and delayed the retrain response by about 4 s (lab/V34_TUXEDOCAT_BBS8_CALL.md).
            return
        rcv.feed(None)
        q = getattr(self, "_qam", None)      # absent in the phase-4 watchdog tests
        if q is not None and q.fill is None and len(rcv.idx) >= P4_MP_BURNIN_SYMBOLS:
            if withhold_mp():
                if not getattr(self, "_mp_withheld_logged", False):
                    self._mp_withheld_logged = True
                    self._ev(f"trained on answerer TRN ({len(rcv.idx)} symbols); WITHHOLDING MP (V34_TEST_WITHHOLD_MP)")
            else:
                # Equalizer has burned in on the answerer's TRN, so 11.4.1.1.2 lets us stop TRN and send MP.
                q.fill = self._fill_mp
                self._ev(f"trained on answerer TRN ({len(rcv.idx)} symbols); sending MP")
        if rcv.trn_end is None:
            if self.stage == "p4_trn":
                self._ev(f"answerer TRN tracked ({len(rcv.idx)} symbols, slips {rcv.eq.slips}), waiting for its MP")
            # Watchdog: once the answerer's TRN has clearly ended (mismatch run found) but no MP
            # frame has decoded past a generous bound, the peer is not sending a decodable V.34 MP
            # (the real Astound V.90 bank sends ~41 clean 16-point symbols then switches; those are
            # not a valid MP under any bit ordering/split).  Failing here ends the call cleanly with a
            # reason instead of tracking TRN forever.  The bound is far beyond any MP/MP'/E exchange
            # (a Type-1 MP is 47 symbols, E is 5), so it cannot fire on a slow-but-valid peer.
            if rcv._end_est is not None and len(rcv.idx) > rcv._end_est + 3000 and self.failed_reason is None:
                self.failed_reason = (
                    f"answerer ended TRN at symbol {rcv._end_est} but sent no decodable MP frame "
                    f"({len(rcv.idx) - rcv._end_est} symbols past it; slips {rcv.eq.slips})")
                self._ev("FAILED: " + self.failed_reason)
            return
        frames = rcv.frames
        selected_mp = v34_negotiation.final_mp(frames)
        if selected_mp is not None and (self.mp_ans is None or
                any(self.mp_ans.get(k) != v for k, v in selected_mp.items() if k != "ack")):
            if self.e_sent:
                self.failed_reason = "peer changed MP parameters after E"
                self.stage = "failed"
                return
            self.mp_ans = dict(selected_mp)
            self._mp_ack = 1
            try:
                self._prepare_data_mode()
            except ValueError as exc:
                self.failed_reason = str(exc)
                self.stage = "failed"
                return
            self._ev(f"answerer TRN length {rcv.trn_end}; MP received: {frames[0]}")
            self.stage = "p4_mp"
        if selected_mp is not None:
            self.mp_ans = {**(self.mp_ans or {}), **selected_mp}
        got_ack = mp_acknowledged(frames)
        if got_ack and not self.e_sent:
            self._ev(f"answerer MP' / E received ({len(frames)} frames): sending E then data")
            self.e_sent = True
        if rcv.b1_symbol is not None and self._rx_b1 is None:
            self._rx_b1 = rcv.b1_symbol
            self._rx_state = None
            self._ev(f"answerer E seen; B1 at symbol {self._rx_b1}")

    def _data_rx(self) -> None:
        """Decode the answerer's data frames (B1 first) with the adaptive equalizer."""
        rate_controller = None
        if getattr(self, '_connected_rate_enabled', False):
            from v34_rate_live import controller
            rate_controller = controller(self, 'call')
            if rate_controller is not None and rate_controller.poll():
                return
        if self._rx_b1 is None or (self._rx_state is None and self.rx_t - self._p3_last < FS // 20):
            return
        self._p3_last = self.rx_t
        eq = self._p4rcv.eq
        R = self._p4R
        from v34_streaming import update_data_baseband, trim_data_pcm
        update_data_baseband(self,R,self._p4_x0)
        rate = self.rx_rate
        if self._rx_state is None:
            if not v34_receive.initialize(self, eq, "answer", self.ans_baud):
                return
        st = self._rx_state
        drx = st["drx"]
        vit = st.get("vit")
        limit = rate_controller.limit(eq.available()) if rate_controller is not None else eq.available()
        while eq.k < limit:
            y = eq.step(decide=self._decide)
            if vit is not None:
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
                        self._data_frame(st, st["pend"], drx)
                        st["pend"] = []
            else:
                st["ys"].append(y)
                if len(st["ys"]) == 8:
                    self._data_frame(st, st["ys"], drx)
                    st["ys"] = []
        v34_receive.observe_quality(self, st, eq)
        self._async_bytes(st)
        trim_data_pcm(self,eq,self._p4_x0)

    def _data_frame(self, st, syms, drx) -> None:
        states = st.pop('pend_states', None)
        bits, worst = (drx.decode_frame(syms, trellis_states=states) if states is not None
                       else drx.decode_frame(syms))
        st["frames"] += 1
        raw = v34_datarx.descramble_stream(bits, st["hist"][-23:], "answer")
        st["hist"] = (st["hist"] + bits)[-23:]
        if self.bit_sink is not None:
            self.bit_sink(raw)
        else:
            st["bits"] += raw
        st["ones"][0] += sum(raw)
        st["ones"][1] += len(raw)
        if st["frames"] % 200 == 0:
            self._ev(f"data rx: {st['frames']} mapping frames, ones {st['ones'][0] / max(1, st['ones'][1]):.3f}, "
                     f"worst {worst:.2f}, slips {self._p4rcv.eq.slips}, freq {self._p4rcv.eq.freq:.2e}")

    def _async_bytes(self, st) -> None:
        b = st["bits"]
        i = 0
        # skip the scrambled-ones lead-in (the first 23 descrambled bits are not yet valid) by framing on start bits
        while i + 10 <= len(b):
            if b[i] == 0 and b[i + 9] == 1:
                self.rx_bytes.append(sum(b[i + 1 + j] << j for j in range(8)))
                i += 10
            else:
                i += 1
        del b[:i]

    # -- probing (11.2.1.1.4 / 11.2.1.1.5) ----------------------------------------------------------------
    def _probing(self) -> None:
        """Find where L1 starts (a 6 dB power step ~10 ms after the second A reversal), then keep L2 for a while."""
        t0 = int(self.probe_start)
        if self.l1_start is None:
            lo, hi = t0, min(len(self._rx), t0 + int(0.12 * FS))
            seg = self._rx[lo:hi]
            if len(seg) < int(0.12 * FS):
                return
            pw = np.convolve(seg ** 2, np.ones(40) / 40, mode="valid")
            base = np.median(pw[:40]) if len(pw) > 40 else 0
            idx = np.nonzero(pw > max(base * 2.0, 1e4))[0]
            if len(idx) == 0:
                # Decode only after the complete short probe-onset window is
                # available, and only while no L1 has been found. A validated
                # repeated INFO0a is positive recovery evidence, not a timeout
                # or a relaxation of probe/qualification thresholds.
                self._try_info0a()
                if self.stage != 'probing':
                    return
                # Recovery heuristic: if no probe follows the duplex second A
                # reversal, retry a later detected reversal within a bounded budget.
                nxt = [r for r in self.a_reversals if r > t0 + int(0.12 * FS)]
                if nxt and self._probe_retries < 3 and self.rx_t > t0 + int(0.30 * FS):
                    self._probe_retries += 1
                    self.probe_start = int(nxt[0])
                    self._ev(f"no L1 after the reversal at {t0}; retrying from {self.probe_start}")
                return
            self.l1_start = lo + int(idx[0])
            self._ev(f"L1 detected at sample {self.l1_start}")
        l2_start = self.l1_start + int(0.160 * FS)
        if self.rx_t >= l2_start + int(0.30 * FS):
            self.l2 = self._rx[l2_start + int(0.03 * FS):l2_start + int(0.30 * FS)]
            self.l1 = self._rx[self.l1_start + int(0.02 * FS):l2_start]
            self.probe_result = analyze_probe(self.l2)
            self._ev(f"probe analysis: {self.probe_result}")
            self._play(ToneSrc(1200.0, self.amp))
            self.stage = "toneB2"
            self._ev("sending tone B after L2")

    def _toneB2(self) -> None:
        """B is on the line: wait for the answerer's tone A reversal, answer it 40 ms later, then send L1 and L2."""
        if self._toneb2_t0 is None:
            self._toneb2_t0 = self.rx_t
            self._n_rev0 = len(self.a_reversals)
        if len(self.a_reversals) > self._n_rev0 and self.b2_reversal_t is None:
            due = self.t + int(self.a_reversals[-1]) - self.rx_t + int(0.040 * FS)
            self.b2_reversal_t = due
            self._at(due, self._do_b2_reversal)
            self._ev(f"scheduled second B reversal at tx sample {due}")
        if self.stage == "toneB2" and self.l_started and self._a_tone_seen():
            self._finish_l2()

    def _do_b2_reversal(self) -> None:
        self._b2_win = self._wins
        self._src.reverse()
        self._ev("tone B phase reversal sent (second)")
        self._at(self.t + int(0.010 * FS), self._start_l)

    def _start_l(self) -> None:
        l1 = probing_signal(0.160, self.rms * 2.0)          # +6 dB
        l2 = probing_signal(0.550 + (self.rtd_samples or 0) / FS, self.rms)
        self._play(ArraySrc(l1), [ArraySrc(l2), Silence()])
        self.l_started = True
        self.l2_start_t = self.t + len(l1)
        self.l2_planned_end_t = self.l2_start_t + len(l2)
        self._ev("L1 then L2 started")

    def _a_tone_seen(self) -> bool:
        j = self._wins - 1
        l2_start_rx = self.l2_start_t
        if j * WIN < l2_start_rx or j < 6:
            return False
        if not all(self._steady(m) for m in range(j - 5, j + 1)):
            return False
        if os.environ.get('V34_L2_GUARD') == '1' and self._b2_win is not None and self.t < self.l2_planned_end_t:
            # Opt-in (lab/V34_NO_INFO1A_GT1_GT6.md): the answerer's tone A pauses while it receives our L1/L2 (312-328 non-steady
            # windows between our second tone B reversal and its return in every normal call). Tone A that never paused is stale:
            # on gt6 it ended L2 after 0.04 s and the answerer never got a usable L2. Keep sending L2 until it ends.
            gap = sum(1 for m in range(self._b2_win, j - 5) if not self._steady(m))
            if gap < self.L2_GUARD_MIN_GAP_WINDOWS:
                return False
        return True

    L2_GUARD_MIN_GAP_WINDOWS = 40                           # 100 ms of 2.5 ms windows without steady tone A

    def _tone_a_on(self) -> bool:
        """Tone A solidly present, which is all 11.2.1.1.3 asks before the first tone B reversal."""
        j = self._wins - 1
        if j < 12:
            return False
        return all(self._steady(m) for m in range(j - 5, j + 1))

    def _finish_l2(self) -> None:
        self._ev("tone A heard; ending L2, sending INFO1c")
        blocks = []
        snr = self.probe_result["snr_db"]
        for sym, hi in ((2400, 0), (2743, 0), (2800, 0), (3000, 0), (3200, 0), (3429, 1)):  # nominal rates
            bits = math.floor(math.log2(1 + 10 ** ((snr - 7.8) / 10)))
            rate = max(0, min(14, int(bits * sym / 2400)))
            if self.symbol_rates is not None and sym not in self.symbol_rates:
                rate = 0                                    # 0 = this symbol rate cannot be used
            blocks.append({"hi_carrier": hi, "preemph": 0, "max_rate": rate})
        info = vi.build_info1c(blocks=blocks, freq_offset=self.freq_offset_units)
        sig = vi.InfoModulator("call", amplitude=self.amp, guard=False).modulate(info, tail_bits=2)
        self._play(ArraySrc(sig), [Silence()])
        self.stage = "info1c_sent"
        self.info1c_sent_t = self.t + len(sig)
        self._info_checked = 0

    @property
    def freq_offset_units(self) -> int:
        l2 = getattr(self, "l2", None)
        if l2 is None or len(l2) < 400:
            return 0
        # Use middle portion and average over several stable tones for robustness
        start = int(0.05 * FS)
        end = int(len(l2) - 0.05 * FS)
        if end <= start + 400:
            start = 0
            end = len(l2)
        seg = l2[start:end]
        n = len(seg) // 2
        if n < 200:
            n = len(seg) // 2
        t = np.arange(len(seg)) / FS
        offs = []
        for f in (600.0, 1050.0, 1350.0, 1500.0, 1650.0, 1950.0):
            ref = np.exp(-2j * math.pi * f * t)
            z1 = (seg[:n] * ref[:n]).sum()
            z2 = (seg[n:2 * n] * ref[n:2 * n]).sum()
            dphi = np.angle(z2 * np.conj(z1))
            if abs(dphi) > math.pi - 1e-6:
                continue
            off_hz = dphi / (2 * math.pi * n / FS)
            if abs(off_hz) < 50:  # reasonable carrier offset
                offs.append(off_hz)
        if not offs:
            return 0
        off = float(np.median(offs))
        return int(max(-511, min(511, round(off / 0.02))))


def analyze_probe(l2: np.ndarray) -> dict:
    """Per-tone level (dB re full scale amplitude) of the 21 probing tones plus the noise at the omitted ones."""
    n = len(l2)
    t = np.arange(n) / FS
    out = {}
    for f in list(L_TONES) + [900, 1200, 1800, 2400]:
        z = (l2 * np.exp(-2j * math.pi * f * t)).sum() * 2 / n
        out[f] = abs(z)
    sig = np.array([out[f] for f in L_TONES])
    noise = np.array([out[f] for f in (900, 1200, 1800, 2400)])
    return {"tone_rms": float(np.sqrt(np.mean(sig ** 2) / 2)), "omitted_rms": float(np.sqrt(np.mean(noise ** 2) / 2)),
            "snr_db": float(20 * math.log10((np.mean(sig ** 2) + 1e-9) / (np.mean(noise ** 2) + 1e-9)) / 2)}

class _QamSrc:
    def __init__(self, qam):
        self.qam = qam
        self.done = False

    def read(self, n: int) -> np.ndarray:
        return self.qam.read(n)
