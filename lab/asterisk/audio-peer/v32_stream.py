"""Streaming V.32 caller modem in Python.

Reference-informed design choices are credited in ``docs/SOURCE_PROVENANCE.md``;
this module does not assert clean-room authorship.

``V32Modem`` has the same surface as the lab's other engines (``feed_pcm`` / ``tx_samples`` /
``feed_bytes`` / ``connected`` / ``bit_rate`` / ``pending_bits``) and runs the whole V.32 caller
sequence on a live 8 kHz (or, for the lab rig, 9.6 kHz) sample stream:

    LISTEN   ANS / AC from the answerer  ->  we send AA
    AA / CC  AC->CA reversal -> we send CC; second reversal ends the round-trip estimate
    TRAIN1   the answerer trains (S, S-bar, TRN 8044, R1) while we are silent; we locate its TRN,
             train the T/2 equaliser on it, and schedule our own training
    TX       we send S (752) S-bar (16) TRN (7424) and R2; the answerer is silent, which is when the
             echo canceller learns (delay search, then NLMS)
    TRAIN2   the answerer answers with a second S/S-bar/TRN and R3; we retrain on it, decode R3,
             pick the best common rate and send E
    DATA     wait for its E, then decode trellis/4800 data (Viterbi at 7200/9600) into bytes

Timeline constants come from measurements against the Smart Link answerer (lab/slmodem-peer).
Signal chain: real samples -> mix by the carrier -> windowed-sinc low-pass evaluated only at the
2-samples-per-symbol output instants (fractional position, so the symbol clock can be nudged) ->
echo cancel (complex NLMS on the baseband) -> T/2 LMS equaliser -> slicer/PLL -> bits.
"""

from __future__ import annotations

import cmath
import math
import os
import struct
from collections import deque

from v32_caller import (
    A, GPA, GPC, NR_DIFF, POINTS, RATE_BITS, RATE_WORD_BASE, TCM_BITS, V32CallerTx, V32_TCM, ZTAB4,
    Scrambler,
)
from v42_layer import V42Layer
from v32_rx import (
    Equalizer, ViterbiStream, expected_trn, slice4, slice_to,
)

BAUD = 2400
CARRIER = 1800.0
IMPLEMENTED_RATES = (14400, 12000, 9600, 7200, 4800)

# measured against the Smart Link answerer (seconds, from the detected second AC reversal)
S_AFTER_REVERSAL = 0.078             # answerer's S starts this long after the back-reversal
TRN_AFTER_S = (256 + 16) / BAUD      # its TRN follows S and S-bar
ECHO_TRAINING_MAX_S = 8192 / BAUD     # V.32 5.4 Note 3: longest echo-canceller sequence before S
# How far past the prediction to keep looking: that sequence plus a long S. A Patton dial-in pool sent
# 3.6 s of echo-canceller training and then S for more than 0.56 s (its end unseen, 2026-11-21).
LOCATE1_MAX_S = 8.0
# V.32 5.5 / Figure 5b: an answerer that restarts with AC wants the start-up again. While we look for its
# first TRN, AC that follows a break (silence or echo-canceller training) and lasts this long is that.
ANSWERER_RESTART_AC_S = 1.0
ANSWERER_RESTARTS_MAX = 2
# An automode answerer (a Patton dial-in pool) whose short AC window closes before our CC can reach it over a
# 1.25 s round trip moves on (V.21, then AC again ~4 s later). Its AC/CA gone this long means that attempt is
# over: back to AA, so its next AC starts a fresh handshake rather than reading as the second reversal.
AC_GONE_S = 0.5
LOCATE1_STEP_S = 0.6                  # slide per empty search: overlaps the +-0.5 s window by more than _trim can cut
ANS_TRN_LEN = 8044
TX_AFTER_TRN_END = 0.15              # fallback: we begin our S this long after its TRN ends
TX_AFTER_R1 = 0.137                  # we begin our S this long after the answerer's R1 starts
SECOND_TRN_AFTER_RESUME = 0.123      # answerer's second TRN starts this long after it resumes
SECOND_RESUME_SILENCE = 1.0
TRN_SETTLE = 500                     # symbols before the equaliser is trusted to spot a TRN ending
TRN_FLOOR_SPAN = 400                 # trailing window the training-error floor is measured over
R3_WORD_BUDGET = 2.5                 # seconds of rate-word-only reading before giving the leg up


class Baseband:
    """Real samples at ``fs`` -> complex baseband at 2 samples per symbol (4800 Hz), streaming.

    ``positions[m]`` is the absolute input-sample position (float) of output m."""

    def __init__(self, fs: float, half: int = 12, phases: int = 64):
        self.fs, self.half, self.nph = fs, half, phases
        self.step = fs / (2.0 * BAUD)
        fc = 1500.0
        self.table = []
        for p in range(phases):
            frac = p / phases
            w = []
            for j in range(-half + 1, half + 1):
                d = j - frac
                x = 2 * fc * d / fs
                h = 2 * fc / fs if abs(x) < 1e-12 else (2 * fc / fs) * math.sin(math.pi * x) / (math.pi * x)
                ham = 0.54 + 0.46 * math.cos(math.pi * d / half)
                w.append(h * ham)
            s = sum(w)
            self.table.append([2.0 * v / s for v in w])      # DC gain 2: mixing halves the amplitude
        self.rot = 1 + 0j
        self.w = cmath.exp(-2j * math.pi * CARRIER / fs)
        self.buf: list[complex] = []
        self.base = 0                    # absolute index of buf[0]
        self.n = 0                       # samples fed
        self.pos = float(half)
        self.out: list[complex] = []
        self._batch_fir = os.environ.get('V32_ECHO_SAFE') == '1'
        if self._batch_fir:
            import numpy as np
            self._np = np
            self._table_array = np.asarray(self.table)

    def feed(self, samples) -> list[complex]:
        rot, w, buf = self.rot, self.w, self.buf
        for v in samples:
            buf.append(v * rot)
            rot *= w
        self.rot = rot / abs(rot)
        self.n += len(samples)
        outs: list[tuple[float, complex]] = []
        half, nph, base = self.half, self.nph, self.base
        if self._batch_fir:
            positions, origins, phases = [], [], []
            # Preserve repeated-addition sample positions and phase quantisation.
            while self.pos + half < self.n:
                i0 = int(self.pos)
                positions.append(self.pos)
                origins.append(i0-half+1-base)
                phases.append(int((self.pos-i0)*nph))
                self.pos += self.step
            if positions:
                np = self._np
                first, last = origins[0], origins[-1]+2*half
                segment = np.asarray(buf[first:last],dtype=np.complex128)
                starts = np.asarray(origins)-first
                coefficients = self._table_array[phases]
                acc = np.zeros(len(positions),dtype=np.complex128)
                # Same multiply/add order as the scalar FIR, across output rows.
                for j in range(2*half):
                    acc += coefficients[:,j]*segment[starts+j]
                outs = list(zip(positions,acc.tolist()))
            drop = int(self.pos)-half-8-base
            if drop > 2048:
                del buf[:drop]
                self.base += drop
            return outs
        while self.pos + half < self.n:
            i0 = int(self.pos)
            tbl = self.table[int((self.pos - i0) * nph)]
            seg = buf[i0 - half + 1 - base: i0 + half + 1 - base]
            acc = 0j
            for a, b in zip(tbl, seg):
                acc += a * b
            outs.append((self.pos, acc))
            self.pos += self.step
        drop = int(self.pos) - half - 8 - base
        if drop > 2048:
            del buf[:drop]
            self.base += drop
        return outs

    def shift(self, delta: float) -> None:
        """Move the sampling instants by ``delta`` input samples (symbol-clock correction)."""
        self.pos += delta


class EchoCanceller:
    """Complex NLMS canceller on the baseband: removes ``taps`` taps of our own transmit signal
    starting ``delay`` T/2-samples back, using absolute sample indices so the alignment does not
    depend on how the audio is chopped into blocks. The transmit history is always recorded so a
    delay found later can be trained on what has already gone out."""

    def __init__(self, taps: int = 48, history: int = 12000):
        self.L = taps
        self.c = [0j] * taps
        self.delay = 0
        self.enabled = False
        self.tx: list[complex] = []
        self.base = 0                      # absolute index of tx[0]
        self.keep = history
        self.p_in = 0.0
        self.p_out = 0.0
        self._ref = None
        self._peak_ref = 0.0
        self._cen = 0.0
        self._native_frozen = os.environ.get('V32_ECHO_SAFE') == '1'
        self._weights = None
        if self._native_frozen:
            import numpy as np
            self._np = np
            self._ring_capacity = 2*history
            self._ring = np.zeros(self._ring_capacity+taps-1,dtype=np.complex128)

    def configure(self, delay: int) -> None:
        self.delay = max(0, delay)
        self.c = [0j] * self.L
        self._ref = None
        self._cen = 0.0
        self._peak_ref = 0.0
        self.enabled = True
        self._weights = None

    def recenter(self) -> None:
        """Follow whole-sample drift of the echo path (e.g. a far-end clock offset): the centroid of
        the significant taps is smoothed over time and, when it has moved two samples from where
        it settled, the bulk delay and the taps shift by one sample (two samples of hysteresis so
        noise in the taps cannot cause slips)."""
        e = [abs(c) ** 2 for c in self.c]
        peak = max(e)
        if peak <= 0:
            return
        keep = [(i, v) for i, v in enumerate(e) if v >= 0.1 * peak]
        if len(keep) > 12 or peak < 0.5 * self._peak_ref:        # taps are mostly noise: do not trust
            return
        tot = sum(v for _, v in keep)
        cen = sum(i * v for i, v in keep) / tot
        if self._ref is None:
            self._ref = cen
            self._cen = cen
            self._peak_ref = peak
            return
        self._cen += 0.1 * (cen - self._cen)
        if self._cen - self._ref >= 2.0:
            self.delay += 1
            self.c = self.c[1:] + [0j]
            self._weights = None
            self._cen -= 1.0
        elif self._cen - self._ref <= -2.0 and self.delay > 0:
            self.delay -= 1
            self.c = [0j] + self.c[:-1]
            self._weights = None
            self._cen += 1.0

    def push_tx(self, s: complex) -> None:
        if self._native_frozen:
            position = (self.base+len(self.tx)) % self._ring_capacity
            self._ring[position] = s
            if position < self.L-1:
                self._ring[self._ring_capacity+position] = s
        self.tx.append(s)
        if len(self.tx) > 2 * self.keep:
            del self.tx[:self.keep]
            self.base += self.keep

    def process(self, u: complex, m: int, mu: float) -> complex:
        """Clean rx baseband sample ``m`` (absolute output index); adapt with step ``mu`` if > 0."""
        if not self.enabled:
            return u
        j0 = m - self.delay - self.L + 1 - self.base          # tx index of the oldest tap input
        tx, L = self.tx, self.L
        if j0 < 0 or j0 + L > len(tx):
            return u
        if self._native_frozen and mu == 0:
            if self._weights is None:
                self._weights = self._np.asarray(self.c[::-1],dtype=self._np.complex128)
            position = (j0+self.base) % self._ring_capacity
            y = complex(self._weights @ self._ring[position:position+L])
            e = u-y
            self.p_in = .999*self.p_in+.001*(u.real*u.real+u.imag*u.imag)
            self.p_out = .999*self.p_out+.001*(e.real*e.real+e.imag*e.imag)
            return e
        x = tx[j0:j0 + L][::-1]                                # newest first: x[i] = tx[m-delay-i]
        y = 0j
        energy = 1e-6
        for ci, xi in zip(self.c, x):
            y += ci * xi
            energy += xi.real * xi.real + xi.imag * xi.imag
        e = u - y
        self.p_in = 0.999 * self.p_in + 0.001 * (u.real ** 2 + u.imag ** 2)
        self.p_out = 0.999 * self.p_out + 0.001 * (e.real ** 2 + e.imag ** 2)
        if mu > 0:
            self._weights = None
            g = mu * e / energy
            self.c = [ci + g * xi.conjugate() for ci, xi in zip(self.c, x)]
        return e


def find_echo_delay(tx: list[complex], rx: list[complex], max_lag: int, threshold: float = 6.0):
    """Cross-correlate transmitted and received baseband; return (lag, peak/median ratio) or None."""
    n = min(len(tx), len(rx)) - max_lag
    if n < 400:
        return None
    ref = tx[:n]
    scores = []
    for lag in range(max_lag):
        acc = 0j
        for a, b in zip(ref, rx[lag:lag + n]):
            acc += a.conjugate() * b
        scores.append(abs(acc))
    best = max(range(max_lag), key=lambda i: scores[i])
    med = sorted(scores)[len(scores) // 2] + 1e-9
    return (best, scores[best] / med) if scores[best] / med >= threshold else None


class V32Modem(V42Layer):
    """V.32 caller. ``my_rates`` limits what is offered (R2); ``debug`` keeps ``events``."""

    ACTIVE_FRAC = 0.3             # of the answerer's training power: far end is transmitting
    SILENT_FRAC = 0.04            # below this: silent (echo residue counts as silence)

    def __init__(self, fs: int = 8000, my_rates=(14400, 12000, 9600, 7200, 4800), amplitude: float = 900.0,
                 echo: bool = True, debug: bool = False, tx_rolloff: float = 0.3, use_v8=None,
                 s_length: int = None, trn_length: int = None, capture: bool = True):
        from threading import RLock
        from v32_retrain import RequestTone
        self._line_lock = RLock()
        self._request_tone = RequestTone("call", fs)
        self._retraining = False
        self._rrp = None
        self._rate_changes_enabled = False
        self._rate_change_history = []
        self._retrain_attempts = 0
        from v34_retrain_budget import RetrainBudget
        self._retrain_budget = RetrainBudget()
        self._retrain_untrusted_samples = 0
        # A single contiguous concealed-media gap is a concrete indication of
        # an unsatisfactory receive signal.  Keep the policy configurable so
        # deployments can disable it (0) or tune it to their media path.
        self._auto_loss_retrain_ms = max(0, int(os.environ.get("V32_AUTO_LOSS_RETRAIN_MS", "160")))
        self._retrain_history = deque(maxlen=8)
        self._retrain_counts = dict(local_started=0, peer_started=0, automatic_started=0,
                                    completed=0, failed=0)
        self._retrain_rx_hold = []
        self._event_offset = 0.0
        self.fs = fs
        self.blk = fs // 50
        self.my_rates = tuple(r for r in my_rates if r in IMPLEMENTED_RATES)
        self.my_rate_word = RATE_WORD_BASE
        for r in self.my_rates:
            self.my_rate_word |= RATE_BITS[r]
        amplitude = float(os.environ.get("V32_AMP", amplitude))
        # Opt-in transmit gain for the handshake tones and the data signal. We send AA/CC at -18.3 dBm0, the
        # Patton's AC arrives at -12.8; real PSTN modems sit nearer -10..-14 (Gough Lui's V.92 set: CM median
        # -14.5). A network echo canceller's NLP clipping our quieter tone would look like the amplitude drop the
        # Patton answers with silence (2026-11-21, hypothesis).
        gain = 10 ** (float(os.environ.get("EL_MODEM_V32_TX_GAIN_DB", "0")) / 20)
        amplitude *= gain
        self._tone_amp = 2700 * gain
        if os.environ.get("V32_RATES"):
            my_rates = tuple(int(r) for r in os.environ["V32_RATES"].split(","))
            self.my_rates = tuple(r for r in my_rates if r in IMPLEMENTED_RATES)
            self.my_rate_word = RATE_WORD_BASE
            for r in self.my_rates:
                self.my_rate_word |= RATE_BITS[r]
        self._tx_after_r1 = float(os.environ.get("V32_AFTER_R1", TX_AFTER_R1))
        s_length = s_length or int(os.environ.get("V32_SLEN", "752"))
        trn_length = trn_length or int(os.environ.get("V32_TRNLEN", "7424"))   # S + S-bar + TRN = 8192 symbols
        self._tx_train_symbols = s_length + 16 + trn_length
        self.tx = V32CallerTx(my_rates=self.my_rates, sample_rate=fs, amplitude=amplitude,
                              rolloff=tx_rolloff, span=12, s_length=s_length, trn_length=trn_length)
        self._configured_rates = self.my_rates
        self._retrain_kwargs = dict(fs=fs, echo=echo, debug=debug,
            amplitude=amplitude, tx_rolloff=tx_rolloff, s_length=s_length)
        self.bb = Baseband(fs)
        self.bbtx = Baseband(fs)
        self.ec = EchoCanceller()
        self.use_echo = echo
        self._echo_safe = os.environ.get('V32_ECHO_SAFE') == '1'
        self._echo_detected_delay = None
        self.debug = debug
        self._trn_clock = None
        if os.environ.get('V32_TRN_CLOCK') == '1':
            from v32_trn_clock import TrainingClock
            self._trn_clock = TrainingClock()
        self.events: list[tuple[float, str]] = []
        cap = os.environ.get("V32_CAPTURE") if capture else None          # prefix: writes <prefix>.rx.raw / .tx.raw (int16 LE)
        self._cap_rx = open(cap + ".rx.raw", "wb") if cap else None
        self._cap_tx = open(cap + ".tx.raw", "wb") if cap else None
        self._no_track = bool(os.environ.get("V32_NOTRACK"))
        self._no_pll = bool(os.environ.get("V32_NOPLL"))
        self._pll_kp = float(os.environ.get("V32_PLLKP", "0.002"))
        self._pll_ki = float(os.environ.get("V32_PLLKI", "0"))
        self._leak = float(os.environ.get("V32_LEAK", "0.0001"))
        self._mu_tr = float(os.environ.get("V32_MUTR", "0.05"))
        self._mu_dd = float(os.environ.get("V32_MUDD", "0.06"))
        self._track_every = int(os.environ.get("V32_TRACKN", "128"))
        self._track_gain = float(os.environ.get("V32_TRACKG", "-0.01"))
        self._ff_sign = float(os.environ.get("V32_FFSIGN", "1"))
        self._track_dead = float(os.environ.get("V32_TRACKD", "0.0"))
        self.state = "listen"
        self.connected = False
        self.bit_rate = 0
        # V.8 (call menu / joint menu) in parallel with the V.32 tone detector: modern answerers open
        # with ANSam and wait for our CM; older ones send ANS then AC and never need it
        self._v8 = None
        self._v8_status = ""
        if use_v8 is None:
            use_v8 = fs == 8000 and bool(os.environ.get("V32_V8", "1") != "0")
        if use_v8:
            try:
                from v8 import V8_MOD_V32, V8Negotiator, available
                if available():
                    self._v8 = V8Negotiator(True, modulations=V8_MOD_V32)
            except Exception as exc:        # noqa: BLE001 - V.8 is optional
                self._v8 = None
                self.events.append((0.0, f"V.8 unavailable: {exc}"))
        self.failed_reason = ""
        self._t = 0
        self._txn = 0
        self._blockbuf: list[int] = []
        self._c600 = [cmath.exp(-2j * math.pi * 600 * i / fs) for i in range(self.blk)]
        self._c3000 = [cmath.exp(-2j * math.pi * 3000 * i / fs) for i in range(self.blk)]
        # Opt-in: V.32 5.4.1 has the caller send AA once it has heard the answer tone for at least 1 s,
        # before AC. A Banksia MyModem 144 does (Gough Lui's stereo V.32bis capture: AA 0.16 s before AC);
        # waiting to hear AC first missed a Patton's 0.8 s AC window over a 1.1 s round trip.
        # "during" (slmodemd's way): AA once the tone has lasted 1 s, without V.8 -- for answerers with V.8 off
        # (a Patton dial-in line, whose automode then skips its V.22bis/V.21 probing). "1": AA when the tone ends,
        # unless the answerer is sending JM (a V.8 answerer such as dialup.world: we then finish V.8 instead).
        mode = os.environ.get("EL_MODEM_V32_AA_AFTER_ANS", "")
        self._aa_after_ans = mode in ("1", "during")
        self._aa_during_ans = mode == "during"
        self._c2100 = [cmath.exp(-2j * math.pi * 2100 * i / fs) for i in range(self.blk)]
        self._c1650 = [cmath.exp(-2j * math.pi * 1650 * i / fs) for i in range(self.blk)]
        self._c1850 = [cmath.exp(-2j * math.pi * 1850 * i / fs) for i in range(self.blk)]
        self._ans_blocks = 0
        self._ans_gone = 0
        self._ac_gone = 0               # blocks without pure AC/CA while we send AA/CC
        # Opt-in: make the AA->CC 180 degree reversal a constant-amplitude phase ramp over a few ms rather than a
        # one-sample jump. On most rounds the Patton answered our CC with 90 ms of silence (V.32 5.4.2's
        # response to an amplitude drop) instead of its CA->AC reversal, then echo-canceller training and a restart;
        # the jump passes through an envelope null (min 0.00 behind a 200 Hz filter; a 10 ms ramp keeps 0.85).
        # Experiment only: a real modem's A->C transition also crosses zero through its pulse filter.
        self._cc_ramp = int(float(os.environ.get("EL_MODEM_V32_CC_RAMP_MS", "10")) * fs / 1000) \
            if os.environ.get("EL_MODEM_V32_CC_RAMP") == "1" else 0
        self._ramp_left = 0
        self._ac_run = 0
        self._ph_hist: list[float] = []
        self._prev_age = 0
        self._tone_phase = cmath.phase(A)
        self._tone_n = 0
        self._cc_tx_at = None
        self._cc_rx_at = None           # receive-clock time of the AC->CA reversal that started our CC
        self._rx_tap = GPA              # the far end's scrambler tap (GPA from an answerer, GPC from a caller)
        self._tx_start_sample = None
        self._tx_fallback = None
        self._phase = "r1"
        self._reset_training_error()
        self._tx_started = False
        # rx symbol stream
        self.u: list[complex] = []
        self.upos: list[float] = []
        self.ubase = 0
        self.eq = Equalizer(32)
        self._rev2_at = None
        self._locate1_shift = 0.0       # how far the answerer-TRN search window has slid past the prediction
        self._restart_gap = 0           # non-tone blocks seen since the back-reversal (answerer-restart watch)
        self._restart_ac = 0            # consecutive AC blocks after such a break
        self._answerer_restarts = 0
        self._rx_mode = "idle"          # idle | locate1 | train1 | wait_resume | locate2 | train2 | dd
        self._s0 = 0
        self._ph = 0
        self._centroid0 = None
        self._cofs = 0                  # equaliser window offset (T/2 samples) from timing slips
        self._cpts: list[tuple[int, float]] = []
        self._k = 0                     # next symbol index
        self._trained = 0
        self._ph_acc = 0.0
        self._freq = 0.0
        self._rx_power = 0.0
        self._fcorr_step = 0.0
        self._phi = 0.0
        self._fcorr_total = 0.0
        self._silent_blocks = 0
        self._resume_at = None
        self._tx_tm0 = None
        self._echo_search = None
        self._quiet_blocks = 0
        self._tx_out = 0
        self._ref_from_train = True
        self._ref_power = 8.0e6
        self._echo_state = "idle"       # idle | collect | search | done
        self._centroid0 = None
        self._sym_count = 0
        self._tmse = 10.0
        self._mse = 0.0
        self._dbg: list[tuple[float, float]] = []
        # decode
        self._bits: deque[int] = deque(maxlen=512)
        self._nbits = 0
        self._desc = Scrambler(GPA)
        self._prev_quad = 0
        self._r2_word = None
        self._r3 = None
        self._r3_deadline = None
        self._rate = 0
        self._e3_word = None
        self._trellis = None
        self._dd_points = POINTS
        self._deframe = _AsyncDeframer()
        self._out = bytearray()
        self._v42_setup()
        self.tx.bit_source = self.v42_tx_bit

    # -- the surface the DCE session expects from an engine ----------------------------
    is_v32 = True
    capture_detect_bits = False

    @property
    def distance_backend(self):
        """Actual data-detector backend; None until trellis data starts."""
        return self._trellis.backend if self._trellis is not None else None

    @property
    def echo_stats(self):
        return dict(requested=bool(self.use_echo), enabled=self.ec.enabled,
                    policy='training-only' if self._echo_safe else 'continuous',
                    detection_backend='numpy' if self._echo_safe else 'python',
                    frozen_backend='numpy' if self.ec._native_frozen else 'python',
                    detected_delay=self._echo_detected_delay,filter_delay=self.ec.delay,
                    delay_units='T/2 samples',taps=self.ec.L,
                    peak_tap=max(range(self.ec.L),key=lambda i:abs(self.ec.c[i])),
                    input_power=self.ec.p_in,output_power=self.ec.p_out,
                    power_scope='total received baseband, including remote signal')

    def is_settled(self) -> bool:
        return self.connected

    def feed_bits(self, bits) -> None:             # raw line bits (V.42 detection): not used
        pass

    @property
    def rx_signal_power_dbm0(self) -> float:
        return 10 * math.log10(self._rx_power / 8.0e6 + 1e-12) - 3.0 if self._rx_power > 0 else -99.0

    # -- helpers -----------------------------------------------------------------
    def _ev(self, msg: str) -> None:
        self.events.append((round(self._event_offset + self._t / self.fs, 3), msg))

    def feed_bytes(self, data: bytes) -> None:
        with self._line_lock:
            self.v42_feed_bytes(data)

    def pending_bits(self) -> int:
        return 10 * len(self.tx._tx_bytes) + self.v42_pending_bits()

    # hooks for V42Layer: the transmitter's async framing and our own descrambled-bit deframer
    def _plain_tx_bit(self) -> int:
        return self.tx._frame_bit()

    def _plain_feed_bytes(self, data: bytes) -> None:
        self.tx.feed_bytes(data)

    def _plain_feed_marks(self, n: int) -> None:
        self.tx.feed_marks(n)

    def _plain_rx_bits(self, bits) -> bytes:
        return self._deframe.push(list(bits))

    @property
    def retrain_stats(self):
        return dict(active=self._retraining, attempts=self._retrain_attempts,
                    **self._retrain_counts,
                    limit=self._retrain_budget.limit, timeout_seconds=45,
                    budget=self._retrain_budget.report(), events=list(self._retrain_history))

    def request_retrain(self, cap=None):
        """Section-7 full retrain; optional cap limits this exchange, not the profile."""
        with self._line_lock:
            if not self.connected or self.state != 'data' or self._retraining or (self._rrp and self._rrp.active):
                return False
            return self._restart_line('local request', cap)

    def enable_rate_changes(self):
        """Explicit experimental LAPM controller; no DCE default enables it."""
        with self._line_lock:
            if self.state != 'data' or not self.connected or self.v42_mode != 'lapm':
                return False
            if self._rrp is None:
                from v32_rate_session import RateSession
                self._rrp = RateSession(self)
            self._rate_changes_enabled = True
            return True

    def request_rate_change(self, rate):
        with self._line_lock:
            return bool(self._rrp and self._rrp.request(rate))

    def _restart_line(self, reason, cap=None):
        rates = tuple(r for r in self._configured_rates if cap is None or r <= cap)
        if not rates:
            return False
        if not self._retrain_budget.take():
            self._fail('V32 full-retrain budget exhausted')
            self.connected = False
            return False
        category = ('local_started' if reason == 'local request' else
                    'peer_started' if reason == 'peer request' else 'automatic_started')
        self._retrain_counts[category] += 1
        # Construct fresh line DSP, keeping the enclosing link, locks, files and
        # application queues. No V.8/ANS, capture-file reopening or link restart.
        kwargs = dict(self._retrain_kwargs, my_rates=rates, use_v8=False,
                      trn_length=self.tx.trn_length, capture=False)
        fresh = type(self)(**kwargs)
        fresh.my_rates = rates  # A startup V32_RATES env override cannot enlarge the cap.
        fresh.my_rate_word = RATE_WORD_BASE
        for rate in rates:
            fresh.my_rate_word |= RATE_BITS[rate]
        fresh.tx.rate_word = fresh.my_rate_word
        if self._rrp is not None and self._rrp.active:
            self._rrp.events[-1].update(status='interrupted',reason='full retrain: '+reason)
            self._rrp.active = False
        keep = {k:v for k,v in self.__dict__.items() if k.startswith('_v42') or k.startswith('v42_')}
        keep.update({k:self.__dict__[k] for k in ('_line_lock','_cap_rx','_cap_tx',
            'events','_out','_blockbuf','_configured_rates','_retrain_kwargs',
            '_retrain_history','_retrain_budget','_retrain_untrusted_samples',
            '_retrain_counts',
            '_rate_changes_enabled','_rate_change_history')})
        keep['v42_mode'] = self.v42_mode
        keep['_event_offset'] = self._event_offset+self._t/self.fs
        keep['_retrain_attempts'] = self._retrain_attempts+1
        keep['_retraining'] = True
        keep['connected'], keep['bit_rate'] = True, self.bit_rate
        queued = deque(self.tx._tx_bytes)
        if self.tx._current_byte is not None:
            queued.appendleft(self.tx._current_byte)
        fresh.tx._tx_bytes = queued
        state = dict(fresh.__dict__)
        state.update(keep)
        self.__dict__ = state
        self.tx.bit_source = self.v42_tx_bit
        self._retrain_history.append(dict(reason=reason, start_seconds=self._event_offset,
                                         offered_rates=list(rates), previous_rate=self.bit_rate,
                                         attempt=self._retrain_attempts,
                                         budget_at_start=self._retrain_budget.report()))
        if self._rx_tap == GPA:
            self.state = 'aa'
        else:
            self.state = 'ac'
            self.tx.start_ac()
        self._ev('V32 full retrain: '+reason+' (carrier and application link retained)')
        return True

    def _retrain_block(self, block):
        if self.state == 'data' and not self._retraining:
            if self._request_tone.feed(block):
                self._restart_line('peer request')
            elif not self._request_tone.run and self._retrain_rx_hold:
                bits, self._retrain_rx_hold = self._retrain_rx_hold, []
                self._out += self.v42_rx_bits(bits)
        if self._retraining and self._t >= 45*self.fs:
            self._fail('V32 full retrain timed out')
            self.connected = False

    def _retrain_completed(self):
        if self._retraining and self.state == 'data':
            self._retraining = False
            self.set_v42_rate(self.bit_rate)
            self._retrain_history[-1].update(completed_seconds=self._event_offset+self._t/self.fs,
                                             negotiated_rate=self.bit_rate)
            self._retrain_counts['completed'] += 1
            self._ev('V32 full retrain complete at '+str(self.bit_rate))
            if self._rate_changes_enabled:
                from v32_rate_session import RateSession
                self._rrp = RateSession(self)

    V42_RX_LAG_MARGIN_S = 1.0                                # as V34Modem: receive path behind real time, plus slack

    def _round_trip_samples(self):
        """Our AA->CC transition to the answerer's back-reversal, in received samples, or None.

        Both ends on the receive clock: the transmit counter can run ahead of it (in --line-mode v32 the
        sender pulls samples before the first RTP packet arrives; 0.5 s on a live call made NT 670 ms
        instead of ~1.2 s, our S too short, and the answerer never trained, 2026-11-21). The interval
        still includes our transmit latency, since the reversal comes back only after CC has gone out."""
        start = self._cc_rx_at if self._cc_rx_at is not None else self._cc_tx_at
        if start is None or self._rev2_at is None:
            return None
        return max(0, self._rev2_at - self.blk // 2 - start)

    @property
    def v42_rtt_hint_seconds(self):
        rtd = self._round_trip_samples()
        return rtd / self.fs if rtd else None

    @property
    def v42_detect_extra_seconds(self) -> float:
        """The ADP comes back one round trip after our ODP (a Patton dial-in pool over the relay: 1.1 s, so the
        plain 0.75 s T400 expired first and the call fell to raw mode under LAPM flags, 2026-11-21).

        V42_DETECT_EXTRA_S overrides it (0 restores the plain 0.75 s T400)."""
        override = os.environ.get("V42_DETECT_EXTRA_S")
        if override is not None:
            return max(0.0, float(override))
        rtd = self._round_trip_samples()
        if rtd is None:
            return 0.0
        return rtd / self.fs + self.V42_RX_LAG_MARGIN_S

    def _set_s_length(self) -> None:
        """V.32bis 6.1: after R1 the caller sends S for NT, the round-trip estimate from its AA->CC
        transition to the second reversal, and then S for 256 more symbols. A fixed length only works when
        the path's round trip is short, so measure it (unless V32_SLEN pins the length)."""
        rtd = self._round_trip_samples()                         # the reversal fell inside its block
        if os.environ.get("V32_SLEN") or rtd is None:
            return
        nt = rtd / self.fs * BAUD
        self.tx.s_length = int(min(max(nt + 256 + 16, 256 + 64), 6000))      # 6000T = 2.5 s round trip
        self._tx_train_symbols = self.tx.s_length + 16 + self.tx.trn_length
        self._ev(f"round-trip estimate NT = {nt:.0f} symbols ({nt / BAUD * 1000:.0f} ms) -> S of {self.tx.s_length} symbols")

    def _fail(self, why: str) -> None:
        self.state = "failed"
        self._rx_mode = "idle"
        self.failed_reason = why
        self.connected = False
        if self._rrp is not None and self._rrp.active:
            self._rrp.events[-1].update(status='failed',reason=why)
            self._rrp.active = False
        if self._retraining:
            self._retrain_history[-1]['failure'] = why
            self._retrain_counts['failed'] += 1
            self._retraining = False
        self._ev("FAILED: " + why)

    # -- transmit ------------------------------------------------------------------
    def tx_samples(self, count: int) -> list[int]:
        with self._line_lock:
            if self.state == 'failed':
                return self._finish_tx([0] * count, count)
            try:
                return self._tx_samples(count)
            except TimeoutError as exc:
                exchange = self._rrp.tx if self._rrp is not None else None
                if not (self._rrp and self._rrp.active and exchange is not None
                        and exchange.state == 'failed'):
                    raise
                self._fail('V32 rate-change transmit timeout: ' + str(exc))
                return self._finish_tx([0] * count, count)

    def _tx_samples(self, count: int) -> list[int]:
        if self._v8 is not None and self.state == "listen":
            out = self._v8.tx(count)
            self._txn += 0
        elif self.state in ("aa", "cc"):
            if self.state == "cc" and self._cc_tx_at is None:
                self._cc_tx_at = self._txn                       # our AA->CC transition
            if self._ramp_left:
                # the last _ramp_left samples of the reversal still to come: phase lags the new value by
                # pi * left / ramp, so the change is spread evenly at constant amplitude
                out = []
                for i in range(count):
                    lag = math.pi * self._ramp_left / self._cc_ramp if self._ramp_left else 0.0
                    self._ramp_left = max(0, self._ramp_left - 1)
                    out.append(round(self._tone_amp * math.cos(2 * math.pi * CARRIER * (self._tone_n + i) / self.fs
                                                     + self._tone_phase - lag)))
            else:
                out = [round(self._tone_amp * math.cos(2 * math.pi * CARRIER * (self._tone_n + i) / self.fs + self._tone_phase))
                       for i in range(count)]
            self._tone_n += count
        elif self._tx_start_sample is not None and not self._tx_started \
                and self._txn + count > self._tx_start_sample:
            self._tx_started = True
            self._tx_tm0 = self._txn
            self._set_s_length()
            self.tx.start_training()
            self.state = "tx"
            self._ev("tx: starting our training (S, S-bar, TRN, R2)")
            lead = max(0, self._tx_start_sample - self._txn)
            out = [0] * lead + self.tx.tx_samples(count - lead)
        elif self._tx_started:
            out = self.tx.tx_samples(count)
        else:
            out = [0] * count
        return self._finish_tx(out, count)

    def _finish_tx(self, out: list[int], count: int) -> list[int]:
        self._txn += count
        if self._cap_tx:
            self._cap_tx.write(struct.pack(f"<{len(out)}h", *out))
        if self.use_echo:
            for pos, v in self.bbtx.feed(out):
                self.ec.push_tx(v)
                self._tx_out += 1
                if self._echo_state == "collect" and self._echo_search is not None:
                    self._echo_search["tx"].append(v)
        return out

    def note_concealed_rx(self, n):
        n = max(0, int(n))
        self.note_transport_discontinuity_rx(n)
        threshold = self._auto_loss_retrain_ms * self.fs // 1000
        if (threshold and n >= threshold and self.connected and self.state == 'data'
                and self.v42_mode == 'lapm' and not self._retraining
                and not (self._rrp and self._rrp.active)):
            duration_ms = 1000 * n / self.fs
            with self._line_lock:
                self._restart_line(f'concealed media gap {duration_ms:.0f} ms')

    def note_transport_discontinuity_rx(self, n=0):
        """Invalidate delivery proof on concealed, duplicate, late or reset RTP."""
        with self._line_lock:
            self._retrain_untrusted_samples = max(self._retrain_untrusted_samples, int(n)+2*self.blk)
            self._observe_retrain_budget(0, 0, untrusted=True)

    def _observe_retrain_budget(self, samples, delivered, *, untrusted=False):
        link = self._v42
        bad = (getattr(link, 'stats', None) or {}).get('hdlc_bad')
        healthy = (self.connected and self.state == 'data' and not self._retraining
                   and not (self._rrp and self._rrp.active)
                   and not untrusted and not self._retrain_untrusted_samples
                   and not self._request_tone.run and not self._quiet_blocks
                   and self._rx_power >= .01*self._ref_power
                   and self.v42_mode == 'lapm' and getattr(link, 'state', None) == 'lapm'
                   and bad is not None)
        if self._retrain_budget.observe(core=self.eq, samples=samples, fs=self.fs,
                healthy=healthy, delivered=delivered, bad_frames=bad):
            self._ev('V32 retrain budget renewed after sustained LAPM delivery')
        self._retrain_untrusted_samples = max(0, self._retrain_untrusted_samples-samples)

    # -- receive -------------------------------------------------------------------
    def feed_pcm(self, pcm: bytes) -> bytes:
        with self._line_lock:
            return self._feed_pcm(pcm)

    def _feed_pcm(self, pcm: bytes) -> bytes:
        count = len(pcm) // 2
        if self._cap_rx:
            self._cap_rx.write(pcm[:2 * count])
            self._cap_rx.flush()
            self._cap_tx.flush()
        self._blockbuf += struct.unpack(f"<{count}h", pcm[:2 * count])
        while len(self._blockbuf) >= self.blk:
            block, self._blockbuf = self._blockbuf[:self.blk], self._blockbuf[self.blk:]
            before = len(self._out)
            self._retrain_block(block)
            self._rx_block(block)
            self._retrain_completed()
            self._out += self.v42_read()
            self._observe_retrain_budget(len(block), len(self._out)-before)
        self._out += self.v42_read()
        out, self._out = bytes(self._out), bytearray()
        return out

    def _rx_block(self, x) -> None:
        self._t += len(x)
        if self._v8 is not None:
            self._v8_rx(x)
        if self.state in ("listen", "aa", "cc"):
            self._handshake_tones(x)
        elif self.state == "train1" and self._rx_mode == "locate1":
            self._watch_answerer_restart(x)
        for pos, v in self.bb.feed(x):
            self._on_baseband(pos, v)
        self._step()

    def _v8_rx(self, x) -> None:
        v8 = self._v8
        v8.rx(x)
        name = v8.status_name
        if name != self._v8_status:
            self._v8_status = name
            self._ev(f"V.8: {name}")
            if v8.status == 2:                                  # negotiated
                from v8 import V8_MOD_V32
                mods = v8.result["modulations"] if v8.result else 0
                self._ev(f"V.8 joint menu: modulations 0x{mods:X}, call function {v8.result['call_function']}")
                if not mods & V8_MOD_V32:
                    self._fail("answerer's V.8 menu does not offer V.32/V.32bis")
            elif v8.status in (3, 4):
                self._v8_stop()

    def _v8_stop(self) -> None:
        if self._v8 is not None:
            self._v8.close()
            self._v8 = None

    def _watch_answerer_restart(self, x) -> None:
        """A Patton dial-in pool, after a silence and 3.6 s of echo-canceller training, sent AC for 5 s
        instead of S/TRN while we waited silently for its TRN (2026-11-21): answer it with AA again."""
        a6 = sum(v * c for v, c in zip(x, self._c600)) / len(x)
        a30 = sum(v * c for v, c in zip(x, self._c3000)) / len(x)
        power = sum(v * v for v in x) / len(x)
        tone = (abs(a6) * 2 > 400 and abs(a30) * 2 > 400 and power > 0
                and 2 * (abs(a6) ** 2 + abs(a30) ** 2) / power >= .85)
        if not tone:
            self._restart_gap += 1
            self._restart_ac = 0
            return
        if self._restart_gap < 3:              # still the AC it holds until it hears us stop
            return
        self._restart_ac += 1
        if self._restart_ac * len(x) < ANSWERER_RESTART_AC_S * self.fs:
            return
        if self._answerer_restarts >= ANSWERER_RESTARTS_MAX:
            return self._fail("answerer keeps restarting with AC")
        self._answerer_restarts += 1
        self._ev("answerer restarted with AC instead of training (V.32 Figure 5b): sending AA again")
        self.state = "aa"
        self._rx_mode = "idle"
        self._rev2_at = self._cc_rx_at = self._cc_tx_at = None
        self._locate1_shift = 0.0
        self._restart_gap = self._restart_ac = 0
        self._ph_hist, self._prev_age = [], 0
        self._tone_n = 0

    def _answer_tone_ended(self, x) -> bool:
        """True on the third block without 2100 Hz after at least 1 s of it (a phase reversal can blank
        one or two blocks, so shorter misses neither end the tone nor reset the count). In "during" mode, True
        as soon as the tone has lasted 1 s. A JM after the tone turns the option off for the call."""
        a21 = sum(v * c for v, c in zip(x, self._c2100)) / len(x)
        power = sum(v * v for v in x) / len(x)
        ans = power > 0 and abs(a21) * 2 > 400 and 2 * abs(a21) ** 2 / power >= 0.6
        if ans:
            self._ans_blocks += 1
            self._ans_gone = 0
            return self._aa_during_ans and self._ans_blocks * len(x) >= self.fs
        if power > 0 and self._ans_blocks * len(x) >= self.fs:
            a16 = sum(v * c for v, c in zip(x, self._c1650)) / len(x)
            a18 = sum(v * c for v, c in zip(x, self._c1850)) / len(x)
            if 2 * (abs(a16) ** 2 + abs(a18) ** 2) / power >= 0.4:
                self._aa_after_ans = False                       # JM: a V.8 answerer, let V.8 finish
                self._ev("JM after the answer tone: completing V.8 instead of sending AA")
                return False
        self._ans_gone += 1
        if self._ans_gone < 3:
            return False
        if self._ans_blocks * len(x) < self.fs:
            self._ans_blocks = 0                  # under 1 s of tone: not an answer tone
            return False
        return True

    def _handshake_tones(self, x) -> None:
        a6 = sum(v * c for v, c in zip(x, self._c600)) / len(x)
        a30 = sum(v * c for v, c in zip(x, self._c3000)) / len(x)
        present = abs(a6) * 2 > 400 and abs(a30) * 2 > 400
        if self._retraining:
            power = sum(v*v for v in x)/len(x)
            present = present and power > 0 and 2*(abs(a6)**2+abs(a30)**2)/power >= .85
        ph = cmath.phase(a6)
        if self.state == "listen":
            if self._aa_after_ans and self._answer_tone_ended(x):
                self.state = "aa"
                self._tone_n = 0
                self._v8_stop()
                self._ev("answer tone heard for 1 s -> sending AA (V.32 5.4.1)" if self._aa_during_ans
                         else "answer tone ended after at least 1 s -> sending AA (V.32 5.4.1)")
                return
            self._ac_run = self._ac_run + 1 if present else 0
            if self._ac_run >= 3:
                self.state = "aa"
                self._tone_n = 0
                self._v8_stop()
                self._ev("heard AC -> sending AA")
                return
        else:
            # AC/CA judged by purity here: the answerer's other automode signals let stray blocks through the
            # amplitude test and keep a stale phase history alive (2026-11-21: a 4.8 s "round trip").
            power = sum(v * v for v in x) / len(x)
            present = present and power > 0 and 2 * (abs(a6) ** 2 + abs(a30) ** 2) / power >= .85
            self._ac_gone = 0 if present else self._ac_gone + 1
            if self._ac_gone * len(x) >= AC_GONE_S * self.fs and (self._ph_hist or self.state == "cc"):
                if self.state == "cc":
                    self._ev("answerer's AC/CA stopped before our CC was answered: back to AA for its next attempt")
                self.state = "aa"
                self._cc_rx_at = self._cc_tx_at = None
                self._ph_hist, self._prev_age = [], 99
        if self.state != "listen" and present:
            # compare with the last good block (a reversal inside a block leaves it weak or split
            # across two blocks, so a block-to-block jump alone can be missed)
            hist = self._ph_hist if self._prev_age <= 3 else []
            flip = any(abs((ph - h + math.pi) % (2 * math.pi) - math.pi) > 2.2 for h in hist)
            if flip and self.state == "aa":
                self.state = "cc"
                self._cc_rx_at = self._t
                self._tone_phase += math.pi
                self._ramp_left = self._cc_ramp
                self._ev("AC->CA reversal -> sending CC")
            elif flip:
                self.state = "train1"
                self._rev2_at = self._t
                self._rx_mode = "locate1"
                self._ev("CA->AC reversal: round trip done")
            self._ph_hist = ([] if flip else self._ph_hist)[-2:] + [ph]
            self._prev_age = 0
            return
        self._prev_age += 1

    def _on_baseband(self, pos: float, v: complex) -> None:
        mu = 0.0
        if self._echo_state == "done" and self._tx_started:
            # fast adaptation only while the answerer is silent (our training), then a trickle
            training = (self._txn - self._tx_tm0) / self.fs < self._tx_train_symbols / BAUD
            mu = 0.1 if training else (0.0 if self._echo_safe else 0.005)
        if self._fcorr_step:
            v *= cmath.exp(-1j * self._phi)
            self._phi += self._fcorr_step
        clean = self.ec.process(v, self.ubase + len(self.u), mu) if self.ec.enabled else v
        if self.ec.enabled and (self.ubase + len(self.u)) % 256 == 0:
            self.ec.recenter()
        self.u.append(clean)
        self.upos.append(pos)
        self._rx_power = 0.995 * self._rx_power + 0.005 * (clean.real ** 2 + clean.imag ** 2)
        if self._echo_state == "collect" and self._echo_search is not None:
            self._echo_search["rx"].append(v)

    def _retune(self, df_hz: float) -> None:
        """Add ``df_hz`` to the carrier-offset correction: derotate what is already buffered from a
        little before the training start, and every later baseband sample."""
        if abs(df_hz) < 0.05:
            return
        step = 2 * math.pi * df_hz / (2 * BAUD)                       # rad per T/2 sample
        i0 = max(0, self._uidx(2 * self._s0 + self._ph) - 80)
        for j in range(i0, len(self.u)):
            self.u[j] *= cmath.exp(-1j * step * (j - i0))
        n_after = len(self.u) - i0
        # continue the same rotation for samples that arrive later
        self._fcorr_step += step
        self._phi += step * n_after
        self._fcorr_total += df_hz
        self._ev(f"carrier offset {df_hz:+.2f} Hz (total correction {self._fcorr_total:+.2f} Hz)")

    def _trim(self) -> None:
        if len(self.u) > 8192:
            drop = len(self.u) - 4096
            del self.u[:drop]
            del self.upos[:drop]
            self.ubase += drop

    def _uidx(self, m: int) -> int:
        return m - self.ubase

    def _have_m(self) -> int:
        return self.ubase + len(self.u)

    def _time_of_m(self, m: int) -> float:
        i = self._uidx(m)
        if 0 <= i < len(self.upos):
            return self.upos[i]
        return self.upos[-1] + (m - (self.ubase + len(self.upos) - 1)) * self.bb.step

    # -- state machine ---------------------------------------------------------------
    def _step(self) -> None:
        if self.state == "failed":
            return
        if self.connected and not self._retraining:
            # carrier watchdog: the far end gone quiet for a while means the line dropped
            if self._rx_power < 0.01 * self._ref_power:
                self._quiet_blocks += 1
                if self._quiet_blocks > 150:                    # 3 s
                    self.connected = False
                    self.state = "down"
                    self._ev("carrier lost")
            else:
                self._quiet_blocks = 0
        self._echo_step()
        mode = self._rx_mode
        if mode == "locate1":
            self._locate1()
        elif mode == "train1":
            self._train_known(self._rx_tap, 7000, next_mode="dd1")
        elif mode == "wait_resume":
            self._wait_resume()
        elif mode == "locate2":
            self._locate2()
        elif mode == "train2":
            self._train_known(self._rx_tap, 7000, next_mode="dd")
        elif mode == "dd":
            self._dd()
        self._trim()

    def _locate(self, predicted_s: float, window_s: float, tap: int):
        """Correlate against the expected TRN around ``predicted_s`` (seconds, input time)."""
        pred_m = int(predicted_s * 2 * BAUD)
        need_m = int((predicted_s + window_s) * 2 * BAUD) + 2 * 320
        if self._have_m() < need_m:
            return "wait"
        ref = expected_trn(300, tap)
        lo = max((self.ubase + 4) // 2, pred_m // 2 - int(window_s * BAUD))
        hi = pred_m // 2 + int(window_s * BAUD)
        nb, bl = 6, 50                       # six 50-symbol blocks, combined non-coherently
        # Same candidates and six non-coherent block correlations as the scalar
        # search. NumPy correlation avoids holding the sender lock for ~250 ms
        # while Python scans a one-second TRN window during full retrain.
        import numpy as np
        samples = np.asarray(self.u, dtype=np.complex128)
        reference = np.asarray(ref, dtype=np.complex128)
        best = None
        for ph in (0, 1):
            first = max(lo, math.ceil((self.ubase-ph)/2))
            stop = min(hi, (self.ubase+len(self.u)-600-ph)//2+1)
            if first >= stop:
                continue
            i = self._uidx(2*first+ph)
            sequence = samples[i:i+2*(stop-first+299):2]
            count = stop-first
            correlations = np.empty((nb, count), dtype=np.complex128)
            for b in range(nb):
                correlations[b] = np.correlate(
                    sequence[b*bl:b*bl+count+bl-1],
                    reference[b*bl:(b+1)*bl], mode='valid')
            energy = np.correlate(sequence.real**2+sequence.imag**2,
                                  np.ones(300), mode='valid')
            scores = np.sum(np.abs(correlations), axis=0)/(np.sqrt(energy*3000.)+1e-9)
            winner = int(np.argmax(scores))
            score = float(scores[winner])
            if best is None or score > best[0]:
                best = (score, first+winner, ph, correlations[:, winner].tolist())
        if best is None:
            return None
        score, k0, ph, cs = best
        # carrier offset from the block-to-block phase progression (rad per 50-symbol block)
        acc = sum(cs[j + 1] * cs[j].conjugate() for j in range(nb - 1))
        f_hz = cmath.phase(acc) * BAUD / (2 * math.pi * bl) if abs(acc) > 0 else 0.0
        return (score, k0, ph, f_hz)

    def _locate1(self) -> None:
        # V.32 5.4.2: the answerer holds AC until it hears our tone stop, so its S is a round trip after
        # the back-reversal. That round trip is the measured AA->CC to second-reversal time less the answerer's
        # 64T turnaround; S_AFTER_REVERSAL alone only fits short paths (a 1.3 s VoIP round trip missed it).
        rtd = self._round_trip_samples()
        extra = max(0.0, rtd / self.fs - 64 / BAUD) if rtd is not None else 0.0
        predicted = self._rev2_at / self.fs + S_AFTER_REVERSAL + extra + TRN_AFTER_S
        loc = self._locate(predicted + self._locate1_shift, 0.5, GPA)
        if loc == "wait":
            return
        if loc is None or loc[0] < 0.5:
            # V.32 5.4 Note 3: the answerer may precede S with up to 8192T of echo-canceller training of its own
            # (a Patton dial-in pool did, 2026-11-21), so slide the window on, overlapping, before giving up.
            if self._locate1_shift + LOCATE1_STEP_S <= LOCATE1_MAX_S:
                self._locate1_shift += LOCATE1_STEP_S
                return
            return self._fail("answerer TRN not found")
        if self._locate1_shift:
            self._ev(f"answerer TRN found {self._locate1_shift:.1f} s past the prediction (echo-canceller training first)")
        score, self._s0, self._ph, f_hz = loc
        self._k = self._s0
        self._trained = 0
        self._centroid0 = None
        self._reset_training_error()
        self._retune(f_hz)
        self._ev(f"answerer TRN located (corr {score:.2f}) at {self._time_of_m(2 * self._s0 + self._ph) / self.fs:.3f}s")
        # fallback only: the blob's TRN length; the real trigger is the answerer's R1 (see _scan_words)
        m_end = 2 * (self._s0 + ANS_TRN_LEN) + self._ph
        self._tx_fallback = int(self._time_of_m(m_end) + 1.5 + TX_AFTER_TRN_END * self.fs)
        self._rx_mode = "train1"

    def _reset_training_error(self) -> None:
        """Clear the training-error average and the trailing floor it is judged against."""
        self._tmse = 10.0
        self._tmse_min = 1e9
        self._tmse_hist = deque(maxlen=TRN_FLOOR_SPAN)
        self._tmse_hot = 0

    def _train_known(self, tap: int, upto: int, next_mode: str) -> None:
        trn = expected_trn(upto, tap)
        n = self.eq.n
        c = self._ph + n // 2 + self._cofs
        while self._k - self._s0 < upto:
            c = self._ph + n // 2 + self._cofs
            end = self._uidx(c + 2 * self._k)
            if end >= len(self.u):
                return
            x = self.u[end - n + 1: end + 1][::-1]
            y = sum(w * v for w, v in zip(self.eq.w, x))
            err = trn[self._k - self._s0] - y
            self.eq.adapt(x, err, self._mu_tr)
            if self._trn_clock is not None:
                fit = self._trn_clock.feed(self,self._k-self._s0+1)
                if fit is not None and fit['applied']:
                    self._ev(f"TRN clock correction {fit['residual_clock_ppm']:+.2f} ppm")
            self._k += 1
            done = self._k - self._s0
            self._tmse = 0.98 * self._tmse + 0.02 * abs(err) ** 2
            # A peer's training sequence need not be as long as we assume, so measure the error
            # against a *trailing* floor. A fixed window opening later (it used to start at symbol
            # 2500) is too late for a short TRN: by the time it opens the error is already huge,
            # the floor is pinned at the corrupted value and the jump test can never fire.
            self._tmse_hist.append(self._tmse)
            self._tmse_min = min(self._tmse_hist)
            self._tmse_hot = self._tmse_hot + 1 if self._tmse > 5.0 * self._tmse_min + 1.0 else 0
            if self._tmse_hot >= 2 and done > TRN_SETTLE:
                self._ev(f"   TRN ended after {done} symbols (training error jumped)")
                self._k -= 40                      # the jump lags the end by about the averaging span
                break
            if self.debug and done % 1000 == 0:
                self._ev(f"   training {done}: mse {self._tmse:.3f} centroid {self._centroid():.2f}")
            if done == 3000 and self._ref_from_train:
                self._ref_power = self._rx_power
                self._ref_from_train = False
            if done == 1500:
                self._centroid0 = self._centroid()          # reference once the taps have formed
            elif done > 1500 and done % self._track_every == 0:
                self._track_timing()
        self._ev("equaliser trained")
        if self._trn_clock is not None:
            self._trn_clock.release()
        if next_mode in ("dd", "dd1", "dd2"):
            self._begin_dd()
            self._phase = {"dd1": "r1", "dd2": "r2"}.get(next_mode, "r3")
            self._rx_mode = "dd"
        else:
            self._rx_mode = next_mode

    def _centroid(self) -> float:
        e = [abs(w) ** 2 for w in self.eq.w]
        tot = sum(e) + 1e-12
        return sum(i * v for i, v in enumerate(e)) / tot

    def _wait_resume(self) -> None:
        # the answerer is silent while we train; it transmits again once our TRN has ended
        if not self._tx_started:
            return
        if self.debug and (self._t // self.blk) % 25 == 0:
            self._ev(f"   wait_resume: rx power {self._rx_power:.3g}, silent blocks {self._silent_blocks}, ec in {self.ec.p_in:.3g} out {self.ec.p_out:.3g}")
        if self._rx_power < self.SILENT_FRAC * self._ref_power:
            self._silent_blocks += 1
            return
        if self._rx_power > self.ACTIVE_FRAC * self._ref_power:
            tx_len = self._tx_train_symbols / BAUD
            ready = (self._txn - self._tx_tm0) / self.fs >= tx_len - 0.3
            if ready and self._silent_blocks >= 10:
                self._resume_at = self._t / self.fs
                self._ev("answerer's second transmission started")
                self._rx_mode = "locate2"
            self._silent_blocks = 0

    def _locate2(self) -> None:
        predicted = self._resume_at - 0.05 + SECOND_TRN_AFTER_RESUME
        loc = self._locate(predicted, 0.4, GPA)
        if loc == "wait":
            return
        if loc is None or loc[0] < 0.5:
            # Not every answerer trains a second time. NIST ACTS resumes with a
            # *repeated rate word*: measured 0.976 correlation at a one-bit lag and a
            # training-sequence score no better than the noise after a TRN (7.6 against 14.3).
            # That word is R3, which is the only thing this leg has to deliver, so read it the
            # same way R1 is read instead of giving up on a sequence that was never sent.
            #
            # There is no TRN to correlate against, but the repeated word is self-delimiting: nine
            # consecutive identical 16-bit words can only start on a symbol boundary inside the
            # region, and the descrambler is self-synchronising too, so bit phase does not matter.
            # What does matter is *where* we start reading. _k is still pointing at the first
            # leg's TRN, five seconds and one silent gap back, so re-anchor it on the prediction
            # _locate just searched around -- the rate word starts within a few tens of ms of the
            # line coming back up.
            pred_m = int(predicted * 2 * BAUD)
            self._k = pred_m // 2
            self._ph = pred_m % 2
            self._centroid0 = None
            self._reset_training_error()
            self._begin_dd()
            self._phase = "r3"
            self._rx_mode = "dd"
            self._r3_deadline = self._t + R3_WORD_BUDGET * self.fs
            self._ev(f"no second TRN: reading R3 from the rate word from {self._time_of_m(pred_m) / self.fs:.3f}s")
            return
        score, self._s0, self._ph, f_hz = loc
        self._k = self._s0
        self._centroid0 = None
        self._reset_training_error()
        self._retune(f_hz)
        self._ev(f"second TRN located (corr {score:.2f})")
        self._rx_mode = "train2"

    # -- decision-directed decode -------------------------------------------------------
    def _begin_dd(self) -> None:
        self._ph_acc, self._freq = 0.0, 0.0
        self._prev_quad = slice4(POINTS[0])
        self._desc = Scrambler(self._rx_tap)
        self._bits.clear()
        self._nbits = 0
        self._dd_points = POINTS
        self._trellis = None
        self._rate = 0
        self._sym_count = 0

    def _dd(self) -> None:
        if self._phase == "r1" and self._tx_start_sample is None and self._tx_fallback is not None \
                and self._t >= self._tx_fallback:
            self._ev("R1 not recognised in time: starting our training from the prediction")
            self._start_tx(self._t)
            return
        n = self.eq.n
        c = self._ph + n // 2 + self._cofs
        while True:
            c = self._ph + n // 2 + self._cofs
            end = self._uidx(c + 2 * self._k)
            if end >= len(self.u):
                return
            x = self.u[end - n + 1: end + 1][::-1]
            y = sum(w * v for w, v in zip(self.eq.w, x))
            yr = y * cmath.exp(-1j * self._ph_acc)
            gain = self._rrp.gain if self._rrp else 1
            if self._rrp:
                yr /= gain
            trellis_metrics = None
            if self._rrp and self._rrp.mode in ('control', 'cleardown'):
                d = slice_to(yr, POINTS)
            elif self._trellis is not None and self._trellis.backend == "numpy":
                trellis_metrics, d = self._trellis.metrics_and_decision(yr)
            else:
                d = slice_to(yr, self._dd_points)
            perr = (yr * d.conjugate()).imag / (abs(d) ** 2 + 1e-9)
            if not self._no_pll:
                self._freq += self._pll_ki * perr
                self._ph_acc += self._pll_kp * perr + self._freq
            self._mse = 0.99 * self._mse + 0.01 * abs(d - yr) ** 2
            if self.debug:
                a_err = cmath.phase(yr * d.conjugate()) if abs(d) > 0 else 0.0
                self._dbg.append((a_err, abs(yr) / (abs(d) + 1e-9)))
            self.eq.adapt(x, (d - yr) * gain * cmath.exp(1j * self._ph_acc), self._mu_dd, self._leak)
            self._k += 1
            self._sym_count += 1
            if self._rrp:
                self._rrp.symbol(yr, self._k-1, trellis_metrics)
            else:
                self._symbol(yr, trellis_metrics)
            if self._rx_mode != "dd":                # a state change inside _symbol (e.g. R1 seen)
                return
            if self._sym_count % self._track_every == 0:
                self._track_timing()
            if self.debug and self._sym_count % 600 == 0:
                if self._dbg:
                    am = sum(a for a, _ in self._dbg) / len(self._dbg)
                    asd = math.sqrt(sum((a - am) ** 2 for a, _ in self._dbg) / len(self._dbg))
                    rm = sum(r for _, r in self._dbg) / len(self._dbg)
                    self._ev(f"   phase err mean {am:+.3f} sd {asd:.3f} rad, radial {rm:.3f}")
                    self._dbg = []
                self._ev(f"dd: {self._sym_count} symbols, mse {self._mse:.3f}, centroid drift "
                         f"{(self._centroid() - (self._centroid0 or 0)):+.2f}, freq {self._freq:+.5f}, ec in {self.ec.p_in:.2g} out {self.ec.p_out:.2g} D {self.ec.delay}")
            if self.state == "failed":
                return

    def _slip(self, d: int) -> None:
        """Re-centre the equaliser by whole T/2 samples (fisher's every-2-seconds timing slip)."""
        self._cofs -= d         # a larger centroid = pulse older = move the window end earlier
        w = self.eq.w
        self.eq.w = (w[1:] + [0j]) if d > 0 else ([0j] + w[:-1])

    def _track_timing(self) -> None:
        """The equaliser follows fractional drift by itself; here only whole-sample drift of its tap
        energy (centroid vs where it was when training settled) is removed, by sliding the window."""
        if self._centroid0 is None or self._no_track:
            return
        drift = self._centroid() - self._centroid0
        if drift >= 1.0:
            self._slip(+1)
        elif drift <= -1.0:
            self._slip(-1)

    def _symbol(self, yr: complex, trellis_metrics=None) -> None:
        if self._trellis is not None:
            g = self._trellis.push(yr, metrics=trellis_metrics)
            if g is not None:
                nb = TCM_BITS.get(self._rate, 3)
                bits = [self._desc.rev((g >> sh) & 1) for sh in range(nb - 1, -1, -1)]
                self._emit(bits)
            return
        # 4800-style stream: rate words, E, and (at 4800) data
        quad = slice4(yr)
        d = NR_DIFF[(self._prev_quad << 2) | quad]
        self._prev_quad = quad
        bits = [self._desc.rev((d >> 1) & 1), self._desc.rev(d & 1)]
        for b in bits:
            self._bits.append(b)
            self._nbits += 1
        if self._rate == 4800 and self.state == 'data':
            self._emit(bits)
            return
        self._scan_words()                       # the word can end on any symbol boundary

    def _word_at_end(self, back: int = 0):
        b = list(self._bits)
        if len(b) < 16 + back:
            return None
        seg = b[len(b) - 16 - back: len(b) - back]
        return int("".join(map(str, seg)), 2)

    def _repeated_rate_word(self):
        """A rate word (matches 0x0111 pattern) repeated 8 times back-to-back at the end of the bits."""
        if len(self._bits) < 9 * 16:
            return None
        w = self._word_at_end()
        if w is None or (w & 0xF111) != 0x0111:
            return None
        return w if all(self._word_at_end(16 * i) == w for i in range(1, 8)) else None

    def _start_tx(self, at_sample: int) -> None:
        # at_sample is on the receive clock; the transmit counter can run ahead of it (see
        # _round_trip_samples), so move it onto the transmit clock or our training starts early.
        ahead = self._txn - self._t
        if ahead <= self.blk:
            ahead = 0                                                # aligned within a block (chain mode)
        self._tx_start_sample = max(at_sample + ahead, self._txn)
        self._phase = "wait"
        self._rx_mode = "wait_resume"
        self._resume_at = None
        self._silent_blocks = 0

    def _scan_words(self) -> None:
        if self._phase == "r1":
            w = self._repeated_rate_word()
            if w is None:
                return
            self._r1 = w
            self._ev(f"R1 = 0x{w:04X}")
            # eight words (64 symbols) have gone by since the first one began
            r1_start = self._t - int(64 * self.fs / BAUD)
            self._start_tx(r1_start + int(self._tx_after_r1 * self.fs))
            return
        if self._r3 is None:
            w = self._repeated_rate_word()
            if w is None:
                if self._r3_deadline is not None and self._t > self._r3_deadline:
                    self._r3_deadline = None
                    return self._fail("answerer sent no second training sequence and its "
                                      "repeated rate word did not decode")
                return
            self._r3 = w
            agreed = w & self.my_rate_word
            rate = next((r for r in IMPLEMENTED_RATES if agreed & RATE_BITS[r]), 0)
            self._ev(f"R3 = 0x{w:04X}")
            if not rate:
                return self._fail("no common rate")
            self._rate = rate
            self._e3_word = 0xF000 | RATE_WORD_BASE | RATE_BITS[rate]
            self.tx.request_e(rate)
            self._ev(f"sending E for {rate} bps")
            return
        if self._e3_word is not None and self._word_at_end() == self._e3_word and self._nbits % 2 == 0:
            self.bit_rate = self._rate
            self.connected = True
            self._ev(f"E received: connected at {self._rate} bps")
            if self._rate == 7200 or self._rate in TCM_BITS:
                self._trellis = ViterbiStream(self._rate)
                self._ev(f"V32 data distance backend: {self._trellis.backend}")
                self._dd_points = ZTAB4 if self._rate == 7200 else list(V32_TCM[self._rate].values())
            self._deframe = _AsyncDeframer()
            self.state = "data"

    def _emit(self, bits) -> None:
        if self.state != 'data':
            return
        if self._request_tone.run and not self._retraining:
            self._retrain_rx_hold.extend(bits)
            return
        if self._rrp:
            self._rrp.emit(bits)
            return
        data = self.v42_rx_bits(bits)
        if data:
            self._out += data

    # -- echo canceller training ---------------------------------------------------------
    def _echo_step(self) -> None:
        if not self.use_echo or not self._tx_started or self._echo_state == "done":
            return
        elapsed = (self._txn - self._tx_tm0) / self.fs
        if self._echo_state == "idle" and elapsed > (self.tx.s_length + 16) / BAUD + 0.1:
            self._echo_search = {"tx": [], "rx": [], "tx0": self._tx_out, "rx0": self.ubase + len(self.u)}
            self._echo_state = "collect"
        elif self._echo_state == "collect":
            if len(self._echo_search["tx"]) >= 1400 and len(self._echo_search["rx"]) >= 1400 + 1500:
                self._echo_state = "search"
                self._echo_gen = _find_delay_gen(self._echo_search["tx"][:1400], self._echo_search["rx"], 1500)
        elif self._echo_state == "search":
            for _ in range(1):
                try:
                    next(self._echo_gen)
                except StopIteration as fin:
                    res = fin.value
                    tx, rx = self._echo_search["tx"], self._echo_search["rx"]
                    if res is None:
                        self._ev("echo: none detected")
                        self.ec.enabled = False
                    else:
                        lag, ratio = res
                        # the two arrays start at different absolute output indices
                        lag += self._echo_search["rx0"] - self._echo_search["tx0"]
                        self._echo_detected_delay = lag
                        self._ev(f"echo: delay {lag / (2 * BAUD) * 1000:.0f} ms (peak/median {ratio:.0f})")
                        self.ec.configure(max(0,lag-self.ec.L//2) if self._echo_safe else lag)
                        # pre-train on what we recorded (the canceller already holds the tx history)
                        r0 = self._echo_search["rx0"] - (self._echo_search["tx0"] - 0)
                        for idx, b in enumerate(rx):
                            self.ec.process(b, self._echo_search["rx0"] + idx, 0.2)
                    self._echo_state = "done"
                    self._echo_search = None
                    return


def _find_delay_gen(tx, rx, max_lag, threshold: float = 6.0):
    """Generator version of find_echo_delay: yields every 20 lags so it can be time-sliced."""
    if os.environ.get('V32_ECHO_SAFE') == '1':
        import numpy as np
        scores = np.abs(np.correlate(np.asarray(rx),np.asarray(tx),'valid')[:max_lag])
        if len(scores) != max_lag:
            raise ValueError('echo delay search requires full reference support')
        best = int(np.argmax(scores))
        med = float(np.partition(scores,len(scores)//2)[len(scores)//2])+1e-9
        ratio = float(scores[best]/med)
        return (best,ratio) if ratio >= threshold else None
    n = len(tx)
    scores = []
    for lag in range(max_lag):
        acc = 0j
        for a, b in zip(tx, rx[lag:lag + n]):
            acc += a.conjugate() * b
        scores.append(abs(acc))
        if lag % 20 == 19:
            yield
    best = max(range(max_lag), key=lambda i: scores[i])
    med = sorted(scores)[len(scores) // 2] + 1e-9
    return (best, scores[best] / med) if scores[best] / med >= threshold else None


class _AsyncDeframer:
    """Incremental async framing: start 0, 8 data bits LSB first, stop 1; idle ones."""

    def __init__(self):
        self.buf: list[int] = []

    def push(self, bits) -> bytes:
        b = self.buf
        b += bits
        out = bytearray()
        i = 0
        while i + 10 <= len(b):
            if b[i] == 0 and b[i + 9] == 1:
                out.append(sum(b[i + 1 + j] << j for j in range(8)))
                i += 10
            else:
                i += 1
        del b[:i]
        return bytes(out)
