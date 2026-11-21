"""V.92 Phase 3: Ru..TRN1u..Ja up, Sd..TRN1d..Jd down (clause 9.5), including nonzero DIL.

Two reactive halves for an offline loopback; no MD:

``DigitalSide`` (9.5.1.1.1-9.5.1.1.4): silent; detects Ru and the Ru-to-Ru-bar flip; takes TRN1u as
starting 24 symbols after the flip; scans for a CRC-valid, variable-length Ja descriptor after the first
2040 TRN1u symbols (the 24 ones and descriptor); then sends Sd 384T, Sd' 48T, TRN1d 2040T and Jd repeatedly.

``AnalogueSide`` (9.5.2.1.1-9.5.2.1.5): sends the plan of ``v92_analogue_phase3_tx`` (silence, Ru,
Ru-bar, TRN1u) and keeps repeating Ja; detects Sd and the Sd-to-Sd' flip; stops Ja at the next 12-bit
boundary and goes silent; takes TRN1d as starting 48 symbols after the flip; finds Jd after the first
2040 TRN1d symbols and parses it.

Then the Su exchange (Figure 10, read from the rendered page; 9.5.1.1.6-9.5.1.1.10, 9.5.2.1.6-9.5.2.1.9):
the analogue modem sends Su 144T, Su-bar 24T, then Su until it has decoded Jp, then Su-bar 24T and a second
TRN1u; the digital modem detects Su, the Su-to-Su-bar flip and the flip back (the phase measurement point),
sends Jp (repeated) and, when it sees the last Su-to-Su-bar flip, Jp'. The paired digital anchors its integer
sample grid on the initial Su, then measures Su's fractional arrival phase again after the first Su-bar and
encodes the complementary Jp fraction. For N > 0 it then sends DIL until CPt;
for N = 0 it sends SCR. The required 24.5T first Su-bar and Jp phase fraction need sub-sample timing.
The experimental ``fractional_tx_timing`` option applies them with a causal fractional-delay interpolator;
it is off by default.

The end of Phase 3 for N = 0 (9.5.1.1.13, 9.5.2.1.10; Figure 11): after SCR the digital modem sends Ri; the analogue
modem, on detecting Ri, sends 24 ones and CPt repeatedly (TRN1u modulation: scrambled, differentially encoded
from the last TRN1u symbol; each CPt padded with zero bits to a 12-symbol boundary, my reading); the digital modem
finds CPt, parses it and sends R-bar-i and goes to Phase 4; the analogue modem, on R-bar-i, finishes the current
CPt, sends E1u (12 zero bits, scrambled and differentially encoded) and goes to Phase 4.

For N > 0, the analogue sends CPt after the required TRN1u and continues it until the digital modem detects it.
The digital modem requests DIL stop, completes the active DIL segment, then sends Ri/R-bar-i and proceeds to
Phase 4. Paired tests cover N=1, 2 and 7; real-peer DIL behavior is unqualified.

Not here: retrain timers, and anything but one
sample per symbol with a unit-gain, in-phase line (the sign of a sample is the decision). A real line needs an
equalizer before the sign decisions.

Symbol timing is the sample clock: the analogue modem sends one symbol per PCM sample, so there is no
timing recovery, only the knowledge of where each sequence starts.
"""

from __future__ import annotations

from concurrent.futures import Future
import os
from queue import Queue
import threading
import time

import numpy as np

import v34_g711
import v34_tx
import v90_digital_tx
import v92_analogue_phase3_tx as a3
import v92_ja
import v90_digital_phase4 as dp4
import v92_cp
import v92_phase3
from v92_phase3_rx import (PATTERNS, PatternTracker, TX_DELAY_BANK,
                           TX_DELAY_BULK, TX_DELAY_PHASES, TX_DELAY_TAPS)

FS = 8000
SCR_MIN = 1200                  # symbols of SCR before the digital modem calls itself trained
TRN_MIN = 2040
TRN1D_MIN_MATCH = 0.80       # reject pattern-tracker false flips before scanning Jd
TRN1U2_FIRST_PHASE_MIN_MATCH = 0.85  # Su-bar2 anchors the scan without fractional reacquisition
E1U_MIN_ZERO_BITS = 10
E1U_WAIT_TIMEOUT_SYMBOLS = 3 * FS
JA_PREAMBLE = 24
JA_SEARCH_TIMEOUT_SYMBOLS = 36000  # 4.5 s at 8 ksymbols/s, measured from TRN1u
_ALIGN_SPAN = (-24, 25, 6)          # start-of-TRN alignment search around flip + 24
PHASE3_EQ_MIN_MATCH = 0.60         # above the random 600-symbol sign-match floor, below the measured moderate-ISI cases
SU_PHASE_SETTLE = 12               # skip the first two periods after the Su-bar transition
SU_PHASE_WINDOW = 120              # 20 periods for a gain/DC-invariant phase fit
SU_PHASE_MIN_FIT = 0.20            # minimum received energy explained by Su's 1333-Hz fundamental
RX_PHASE_TAPS = 33
RX_PHASE_KAISER_BETA = 8.6
RX_PHASE_TRAINING_SYMBOLS = 1000
RX_PHASE_MIN_FIT = 0.20
RX_PHASE_MEASUREMENT_STRIDE = 8
RX_PHASE_COARSE_STEP = 0.05
RX_PHASE_WINDOW = np.kaiser(RX_PHASE_TAPS, RX_PHASE_KAISER_BETA)

_DIL_PREPARE_TASKS: Queue = Queue()


def _dil_prepare_worker() -> None:
    """Pre-render Ja DILs off the receive and transmit media callbacks."""
    while True:
        future, transmitter, descriptor = _DIL_PREPARE_TASKS.get()
        if future.set_running_or_notify_cancel():
            started = time.perf_counter_ns()
            try:
                stream = transmitter.prepare_dil_stream(descriptor)
                future.set_result((stream, time.perf_counter_ns() - started))
            except BaseException as exc:
                future.set_exception(exc)
        _DIL_PREPARE_TASKS.task_done()


# Start once at module load, well before a call's media deadlines.
_DIL_PREPARE_THREAD = threading.Thread(
    target=_dil_prepare_worker, name="v92-dil-prepare", daemon=True)
_DIL_PREPARE_THREAD.start()


def _estimate_su_fractional_phase(samples, *, sample_start: int,
                                  su_reference_start: int) -> tuple[float, float]:
    """Estimate Su's fractional phase from its 1333-Hz fundamental component.

    Su also contains a Nyquist component, which has no unambiguous fractional
    phase in real 8-kHz samples. Use the known fundamental's coherent phase
    against the anchored integer grid and report the fraction of signal energy
    explained by that component as a confidence score.
    """
    y = np.asarray(samples, dtype=float).reshape(-1)
    if len(y) < 6 or not np.all(np.isfinite(y)):
        raise ValueError("Su phase measurement needs at least six finite samples")
    pattern = PATTERNS["su"]
    centered_y = y - float(np.mean(y))
    if float(centered_y @ centered_y) <= 1e-12:
        raise ValueError("Su phase measurement has no pattern energy")
    omega = 2.0 * np.pi / 6.0
    n = sample_start + np.arange(len(y), dtype=np.int64) - int(su_reference_start)
    reference = np.sum(pattern * np.exp(-1j * omega * np.arange(6)))
    observed = np.sum(centered_y * np.exp(-1j * omega * n))
    if abs(reference) <= 1e-12 or abs(observed) <= 1e-12:
        raise ValueError("Su phase measurement has no coherent fundamental")
    phase = (-float(np.angle(observed / reference)) / omega) % 1.0
    if phase > 1.0 - 1e-8:
        phase = 0.0
    fundamental_energy = 2.0 * float(abs(observed) ** 2) / len(y)
    fit = min(1.0, fundamental_energy / float(centered_y @ centered_y))
    return phase, fit


def _signs_to_bits(x) -> list[int]:
    return [1 if v < 0 else 0 for v in x]


def _decode_diff_scrambled(symbol_bits, previous: int, history, tap: int, count: int):
    """Undo differential coding and the self-synchronising scrambler (y = x ^ y[n-tap] ^ y[n-23])."""
    reg = list(reversed([int(b) for b in history[-23:]]))
    out = []
    for transmitted in symbol_bits[:count]:
        scrambled = transmitted ^ previous
        previous = transmitted
        out.append(scrambled ^ reg[tap - 1] ^ reg[22])
        reg = [scrambled] + reg[:-1]
    return out


def _decode_stateful(symbol_bits, previous: int, reg, tap: int):
    """Like ``_decode_diff_scrambled`` but returns the state to carry on with: (bits, previous, reg),
    ``reg`` being the last 23 scrambled bits, newest first."""
    out = []
    reg = list(reg)
    for transmitted in symbol_bits:
        scrambled = transmitted ^ previous
        previous = transmitted
        out.append(scrambled ^ reg[tap - 1] ^ reg[22])
        reg = [scrambled] + reg[:-1]
    return out, previous, reg


def _trn1u_bits(n: int) -> list[int]:
    s = v34_tx.Scrambler("answer")
    return [s.step(1) for _ in range(n)]


def _trn1d_bits(n: int) -> list[int]:
    s = v90_digital_tx._GPC()
    return [s.step(1) for _ in range(n)]


def _windowed_sinc_sample(samples, positions, *, taps: int = RX_PHASE_TAPS,
                          beta: float = RX_PHASE_KAISER_BETA) -> np.ndarray:
    """Sample real PCM at fractional positions with a normalized Kaiser-windowed sinc."""
    x = np.asarray(samples, dtype=float).reshape(-1)
    p = np.asarray(positions, dtype=float).reshape(-1)
    if taps < 3 or taps % 2 != 1:
        raise ValueError("windowed-sinc tap count must be odd and at least three")
    if not len(x) or not np.all(np.isfinite(p)):
        raise ValueError("windowed-sinc input must be nonempty with finite positions")
    radius = taps // 2
    centers = np.floor(p).astype(np.intp)
    indices = centers[:, None] + np.arange(-radius, radius + 1, dtype=np.intp)[None, :]
    distance = p[:, None] - indices
    window = RX_PHASE_WINDOW if taps == RX_PHASE_TAPS and beta == RX_PHASE_KAISER_BETA else np.kaiser(taps, beta)
    weights = np.sinc(distance) * window[None, :]
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.sum(x[np.clip(indices, 0, len(x) - 1)] * weights, axis=1)


def _estimate_trn1u_fractional_phase(samples, start: int, *,
                                     training_symbols: int = RX_PHASE_TRAINING_SYMBOLS
                                     ) -> tuple[float, float]:
    """Fit a constant sub-sample receive phase using known TRN1u, never Ja or later data."""
    x = np.asarray(samples, dtype=float).reshape(-1)
    radius = RX_PHASE_TAPS // 2
    if (start < radius or training_symbols < 64
            or len(x) < start + training_symbols + radius):
        raise ValueError("TRN1u fractional-phase window is incomplete")
    expected_bits = np.asarray(_trn1u_bits(training_symbols), dtype=float)
    expected = 1.0 - 2.0 * expected_bits
    sample_indices = np.arange(0, training_symbols, RX_PHASE_MEASUREMENT_STRIDE,
                               dtype=np.intp)
    expected = expected[sample_indices]
    positions = start + sample_indices.astype(float)

    def score(phase: float) -> float:
        observed = _windowed_sinc_sample(x, positions + phase)
        observed -= float(np.mean(observed))
        norm = float(np.linalg.norm(observed))
        if norm <= 1e-12:
            return -1.0
        return float(expected @ observed) / (float(np.linalg.norm(expected)) * norm)

    coarse = np.arange(-0.5, 0.500001, RX_PHASE_COARSE_STEP)
    coarse_scores = [score(float(phase)) for phase in coarse]
    best = max(range(len(coarse)), key=lambda i: (coarse_scores[i], -abs(float(coarse[i]))))
    phase = float(coarse[best])
    fit = float(coarse_scores[best])
    if 0 < best < len(coarse) - 1:
        left, right = float(coarse_scores[best - 1]), float(coarse_scores[best + 1])
        curvature = left - 2.0 * fit + right
        if curvature < -1e-12:
            delta = 0.5 * (left - right) / curvature * RX_PHASE_COARSE_STEP
            phase += max(-RX_PHASE_COARSE_STEP, min(RX_PHASE_COARSE_STEP, delta))
    return phase, fit


def _match(received, expected_bits, *, positive_is_one: bool = False) -> float:
    got = np.array([1 if v > 0 else 0 for v in received] if positive_is_one
                   else _signs_to_bits(received))
    return float(np.mean(got == np.array(expected_bits[:len(got)])))


class DigitalSide:
    def __init__(self, law: str, *, uinfo: int = 67, rate_mask: int = 0x3FFFFF,
                 equalise: bool = False, fractional_rx_timing: bool = False,
                 fractional_rx_reacquisition: bool | None = None):
        self.law = law
        self.uinfo = uinfo
        self.rate_mask = rate_mask
        # Phase 3 decides on signs. With ``equalise`` a marginal first TRN1u (sign match 0.75-0.95) trains an FIR on the known
        # sequence; every decision after that reads the filtered stream, whose symbols lag the raw ones by ``eq_delay``.
        self._equalise = equalise
        if type(fractional_rx_timing) is not bool:
            raise TypeError("fractional_rx_timing must be bool")
        self.fractional_rx_timing = fractional_rx_timing
        if fractional_rx_reacquisition is None:
            fractional_rx_reacquisition = fractional_rx_timing
        if type(fractional_rx_reacquisition) is not bool:
            raise TypeError("fractional_rx_reacquisition must be bool")
        if fractional_rx_reacquisition and not fractional_rx_timing:
            raise ValueError("fractional RX reacquisition requires fractional RX timing")
        self.fractional_rx_reacquisition = fractional_rx_reacquisition
        self.rx_fractional_phase: float | None = None
        self._fractional_rx = np.empty(0)
        self._fractional_rx_feed_block = 0
        self._fractional_rx_update_ns: list[int] = []
        self._fractional_rx_update_max_ns = 0
        self._fractional_rx_update_max_location: dict | None = None
        self._filter = None
        self.eq_delay = 0
        self._raw = np.empty(0)
        self._tx = v90_digital_tx.V90DigitalPhase3Transmitter(uinfo, law)
        self._queue = bytearray()
        self.tx_symbols_emitted = 0
        self.sd_start_sample: int | None = None
        self.jd_start_sample: int | None = None
        self.dil_start_sample: int | None = None
        self.ri_start_sample: int | None = None
        self.rbar_i_start_sample: int | None = None
        # This prefix is determined by UINFO/law and the V.92 transmit
        # scrambler seed, not by the caller's Ja. Render it at Phase 3 object
        # creation so accepting Ja only queues ready bytes on the media path.
        self._initial_downstream = bytearray(v90_digital_tx.sd_codewords(uinfo, law))
        self._initial_downstream += v90_digital_tx.sd_prime_codewords(uinfo, law)
        self._initial_downstream += self._tx.trn1d(TRN_MIN)
        self._silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
        self.tracker = PatternTracker(families=("ru",))
        self._rx = np.empty(0)
        self.state = "wait-ru"
        self.events: list[tuple[str, object]] = []
        self.ru_flip_at: int | None = None
        self.trn1u_start: int | None = None
        self.ja: dict | None = None
        self.ja_start: int | None = None
        self._next_ja_try = TRN_MIN
        # (Ja onset, descriptor offset within Ja, descriptor length, candidate
        # repeated-frame lengths, largest repetition count already tried)
        self._ja_pending: list[dict] = []
        self._ja_trn_history: list[int] | None = None
        self._ja_min_len = len(v92_ja.build_ja_descriptor(N=0))
        self._prepared_dil = None
        self._dil_prepare_future: Future | None = None
        self._dil_prepare_worker_ns = 0
        self._dil_prepare_fallbacks = 0
        self._dil = None
        self.dil_symbols_sent = 0
        self._dil_stop_reported = False
        self.jd_sent = 0
        self.jp_sent = 0
        self.su_tracker: PatternTracker | None = None
        self._su_seen = self._flip1 = self._send_jp = self._flip3 = False
        self._su_event_start: int | None = None
        self._su_reference_start: int | None = None
        self._su_back_at: int | None = None
        self._su_initial_phase: float | None = None
        self.su_phase: float | None = None
        self.jp_fraction: int | None = None
        self._jprime_sent = False
        self.trn1u2_start: int | None = None
        # Fractional TRN1u#2 search is advanced one phase per media block. The
        # old all-at-once scan held the receive path for ~80 ms on real calls.
        self._trn1u2_search: dict | None = None
        self.scr_symbols = 0
        self.ri_units = 0
        self.cpt: dict | None = None
        self.rbar_sent = False
        self._next_cpt_try = 0
        self._cpt_pending: list[tuple[int, int, int]] = []
        self.cpt_at: int | None = None
        self.trn2u_start: int | None = None          # first sample of the analogue modem's Phase 4 (after E1u)
        self._cpt_dec: tuple | None = None           # (absolute index, previous bit, scrambler register)
        self._cpt_last_frame_dec: tuple | None = None
        self.e1u_last_sign: int | None = None
        self._cpt_trn1u_scrambler = v34_tx.Scrambler("answer")
        self._cpt_trn1u_history: list[int] = []
        self._e1u_wait_start: int | None = None
        self._e1u_windows_seen = 0
        self._stage_timing_enabled = os.environ.get("V92_DSP_STAGE_TIMING") == "1"
        self._stage_timings: dict[str, dict] = {}

    def _timed_stage(self, name: str, callback, *args, **kwargs):
        """Run one Phase 3 step with opt-in wall-time accounting."""
        if not self._stage_timing_enabled:
            return callback(*args, **kwargs)
        started = time.perf_counter_ns()
        try:
            return callback(*args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - started
            record = self._stage_timings.setdefault(name, {
                "count": 0, "total_ns": 0, "max_ns": 0,
                "max_context": None,
            })
            record["count"] += 1
            record["total_ns"] += elapsed
            if elapsed > record["max_ns"]:
                record["max_ns"] = elapsed
                record["max_context"] = {
                    "feed_block": self._fractional_rx_feed_block,
                    "rx_samples": len(self._rx),
                    "tx_queue_octets": len(self._queue),
                    "state": self.state,
                }

    def stage_timing_report(self) -> dict:
        """Return bounded aggregate timings; empty unless explicitly enabled."""
        return {
            "enabled": self._stage_timing_enabled,
            "dil_prepare_fallbacks": self._dil_prepare_fallbacks,
            "dil_prepare_worker_ms": self._dil_prepare_worker_ns / 1_000_000.0,
            "stages": {
                name: {
                    "count": row["count"],
                    "total_ms": row["total_ns"] / 1_000_000.0,
                    "max_ms": row["max_ns"] / 1_000_000.0,
                    "max_context": row["max_context"],
                }
                for name, row in self._stage_timings.items()
            },
        }

    def feed(self, samples) -> None:
        return self._timed_stage("rx.feed_block", self._feed_block, samples)

    def _feed_block(self, samples) -> None:
        x = np.asarray(samples, dtype=float).reshape(-1)
        self._fractional_rx_feed_block += 1
        old_rx_len = len(self._rx)
        self._raw = np.concatenate((self._raw, x))
        rx_chunk = x if self._filter is None else self._filter.apply(x)
        self._rx = np.concatenate((self._rx, rx_chunk))
        if self.rx_fractional_phase is not None:
            self._timed_stage("rx.fractional_resample", self._update_fractional_rx,
                              old_rx_len)
        for name, at in self.tracker.feed(x):
            if name == "ru-bar" and self.ru_flip_at is None and self.state == "wait-ru":
                self.ru_flip_at = at
                self.state = "wait-trn1u"
                self.events.append(("ru-flip", at))
        if self.state == "wait-trn1u":
            self._timed_stage("rx.find_trn1u", self._find_trn1u)
        if self.state == "wait-ja":
            if self.fractional_rx_timing and self.rx_fractional_phase is None:
                self._timed_stage("rx.fractional_resample", self._update_fractional_rx,
                                  old_rx_len)
            self._timed_stage("rx.find_ja", self._find_ja)
        elif self.su_tracker is not None:
            self._timed_stage("rx.su_tracking", self._su_events, rx_chunk)
        if self._flip3 and self.trn1u2_start is None:
            self._timed_stage("rx.find_trn1u2", self._find_trn1u2)
        if self.trn1u2_start is not None and self.cpt is None and self.state == "trn1u-2":
            self._timed_stage("rx.find_cpt", self._find_cpt)
        if self.cpt is not None and self.trn2u_start is None:
            self._timed_stage("rx.find_e1u", self._find_e1u)

    @property
    def _decision_rx(self) -> np.ndarray:
        return self._fractional_rx if self.rx_fractional_phase is not None else self._rx

    def _update_fractional_rx(self, old_len: int) -> None:
        if not self.fractional_rx_timing or self.trn1u_start is None:
            return
        start_ns = time.perf_counter_ns()
        stage = ("fit-and-backfill" if self.rx_fractional_phase is None
                 else "incremental-resample")
        try:
            self._update_fractional_rx_impl(old_len)
        finally:
            elapsed_ns = time.perf_counter_ns() - start_ns
            self._fractional_rx_update_ns.append(elapsed_ns)
            if elapsed_ns > self._fractional_rx_update_max_ns:
                self._fractional_rx_update_max_ns = elapsed_ns
                self._fractional_rx_update_max_location = {
                    "stage": stage,
                    "feed_block": self._fractional_rx_feed_block,
                    "sample_position": len(self._rx),
                }

    def _update_fractional_rx_impl(self, old_len: int) -> None:
        """Perform fractional sampling; wrapper records opt-in per-call cost."""
        radius = RX_PHASE_TAPS // 2
        if self.rx_fractional_phase is None:
            if len(self._rx) < self.trn1u_start + RX_PHASE_TRAINING_SYMBOLS + radius:
                return
            phase, fit = _estimate_trn1u_fractional_phase(
                self._rx, self.trn1u_start, training_symbols=RX_PHASE_TRAINING_SYMBOLS)
            if fit < RX_PHASE_MIN_FIT:
                self.events.append(("trn1u-fractional-phase-low-confidence", {
                    "sample_phase": phase, "fit": fit,
                    "fallback_sample_phase": 0.0,
                }))
                phase = 0.0
            self.rx_fractional_phase = phase
            self.events.append(("trn1u-fractional-phase", {
                "sample_phase": phase, "fit": fit,
                "training_symbols": RX_PHASE_TRAINING_SYMBOLS,
            }))
            self._fractional_rx = self._rx.copy()
            start = max(0, self.trn1u_start - radius)
            positions = np.arange(start, len(self._rx), dtype=float) + phase
            self._fractional_rx[start:] = _windowed_sinc_sample(self._rx, positions)
            return
        start = max(0, old_len - radius)
        positions = np.arange(start, len(self._rx), dtype=float) + self.rx_fractional_phase
        updated = _windowed_sinc_sample(self._rx, positions)
        if len(self._fractional_rx) < len(self._rx):
            self._fractional_rx = np.pad(
                self._fractional_rx, (0, len(self._rx) - len(self._fractional_rx)))
        self._fractional_rx[start:] = updated

    def fractional_rx_timing_diagnostics(self) -> dict:
        """Summarize opt-in sampler cost and where its slowest update occurred."""
        durations = np.asarray(self._fractional_rx_update_ns, dtype=np.float64)
        return {
            "count": int(len(durations)),
            "p99_ms": (float(np.percentile(durations, 99) / 1_000_000.0)
                       if len(durations) else 0.0),
            "max_ms": float(self._fractional_rx_update_max_ns / 1_000_000.0),
            "max_location": self._fractional_rx_update_max_location,
            "sample_phase": self.rx_fractional_phase,
        }

    def _su_events(self, x) -> None:
        for name, at in self.su_tracker.feed(x):
            at += self._su_origin
            bar = name.endswith("bar")
            if not self._su_seen:
                self._su_seen = True
                self._su_event_start = at
                self.events.append(("su", at))
            elif bar and not self._flip1:
                expected = (None if self._su_event_start is None
                            else self._su_event_start + 144)
                if expected is not None and abs(at - expected) > 12:
                    self.events.append(("su-bar-outside-first-window", {
                        "sample": at, "expected": expected,
                    }))
                    continue
                self._flip1 = True
                self.events.append(("su-flip-1", at))
            elif not bar and self._flip1 and not self._send_jp:
                # Keep receiving Su long enough to estimate its phase against the
                # integer grid established by the first Su segment.
                self._su_back_at = at
                self.events.append(("su-back", at))
            elif bar and self._send_jp and not self._flip3:
                self._flip3 = True
                self._flip3_at = at
                self.events.append(("su-flip-3", at))
        self._measure_su_phase()

    def _measure_su_phase(self) -> None:
        if self._su_event_start is None:
            return
        if self._su_initial_phase is None:
            start = self._su_event_start + SU_PHASE_SETTLE
            end = start + SU_PHASE_WINDOW
            if len(self._rx) < end:
                return
            # PatternTracker is a coarse event detector. Refine its integer
            # edge locally; the fractional fit resolves the arrival phase,
            # while the repeated six-sample pattern makes offsets by 3T
            # equivalent and leaves the fractional phase unchanged.
            candidates = []
            # The pattern repeats every six samples, so six consecutive
            # candidate edges cover every distinct integer phase. Avoid
            # fitting equivalent copies across the full detector tolerance.
            for reference in range(self._su_event_start - 3,
                                   self._su_event_start + 3):
                phase, fit = _estimate_su_fractional_phase(
                    self._rx[start:end], sample_start=start,
                    su_reference_start=reference)
                candidates.append((fit, reference, phase))
            fit, reference, initial_phase = max(
                candidates, key=lambda v: (v[0], -abs(v[1] - self._su_event_start)))
            if fit < SU_PHASE_MIN_FIT:
                self.events.append(("su-phase-low-confidence", {
                    "initial_phase": initial_phase, "fit": fit}))
                initial_phase = 0.0
            self._su_reference_start = reference
            self._su_initial_phase = initial_phase
            self.events.append(("su-phase-anchor", {
                "sample_phase": initial_phase,
                "reference_sample": reference,
                "fit": fit,
            }))
        if self._send_jp or self._su_back_at is None:
            return

        # Measure the actual Su phase after the first Su-bar against the integer
        # grid anchored on the initial Su. The 24.5T bar should contribute a
        # half-sample phase change, but measure the received waveform instead
        # of assuming the transmitter and channel realized that exact shift.
        start = self._su_back_at + SU_PHASE_SETTLE
        end = start + SU_PHASE_WINDOW
        if len(self._rx) < end:
            return
        phase, fit = _estimate_su_fractional_phase(
            self._rx[start:end], sample_start=start,
            su_reference_start=self._su_reference_start)
        if fit < SU_PHASE_MIN_FIT:
            self.events.append(("su-phase-low-confidence", {
                "phase": phase, "fit": fit, "measurement_point": "post-su-bar"}))
            # Keep a bounded fallback for the experimental paired path, but
            # make its provenance explicit in the event stream.
            phase = (self._su_initial_phase + 0.5) % 1.0
        self.su_phase = phase
        self.jp_fraction = int(round(((1.0 - phase) % 1.0) * 65536.0)) & 0xFFFF
        self._send_jp = True
        self.events.append(("su-phase", {
            "initial_delay_samples": self._su_initial_phase,
            "delay_samples": phase,
            "fit": fit,
            "measurement_point": "post-su-bar",
            "jp_fraction": self.jp_fraction,
        }))

    def raw_from(self, start: int) -> np.ndarray:
        """Unfiltered samples for a position found in the (possibly filtered, delayed) decision stream."""
        return self._raw[max(0, int(start) - self.eq_delay):]

    def _best_alignment(self, base: int):
        lo, hi, step = _ALIGN_SPAN
        if self._filter is not None:
            lo, hi, step = lo + self.eq_delay, hi + self.eq_delay, 1
        expected = _trn1u_bits(600)
        return max(((_match(self._decision_rx[base + d:base + d + 600], expected), base + d)
                    for d in range(lo, hi, step)), key=lambda t: (t[0], -abs(t[1] - base)))

    def _best_near_alignment(self, base: int, radius: int = 1):
        """Refine the existing coarse TRN window at every sample, with guard samples."""
        shift = self.eq_delay if self._filter is not None else 0
        lo, hi, _ = _ALIGN_SPAN
        offsets = np.arange(lo - radius + shift, hi + radius + shift, dtype=np.intp)
        first = base + int(offsets[0])
        source = self._decision_rx[first:base + int(offsets[-1]) + 600]
        windows = np.lib.stride_tricks.sliding_window_view(source, 600)[:len(offsets)]
        expected = np.asarray(_trn1u_bits(600), dtype=bool)
        scores = np.mean((windows < 0.0) == expected[None, :], axis=1)
        best = max(range(len(offsets)), key=lambda i: (float(scores[i]), -abs(int(offsets[i]))))
        return float(scores[best]), base + int(offsets[best])

    def _train_phase3_filter(self, start: int) -> bool:
        import v92_equaliser as equaliser
        from v92_equaliser import StreamFilter
        lead = 8
        bits = _trn1u_bits(600)
        sent = np.concatenate((np.zeros(lead), np.where(np.array(bits) == 1, -1.0, 1.0)))
        received = self._raw[start - lead:start + 600]
        if len(received) != len(sent) or start < lead:
            return False
        sent *= float(np.sqrt(np.mean(received[lead:] ** 2)))      # filtered stream keeps the line's level (trackers use absolute thresholds)
        w, delay, _ = equaliser.train(received, sent)
        self._filter, self.eq_delay = StreamFilter(w), int(delay)
        self._rx = self._filter.apply(self._raw)
        self.events.append(("phase3-equaliser", {"delay": self.eq_delay, "taps": len(w)}))
        return True

    def _find_trn1u2(self) -> None:
        base = self._flip3_at + 24                  # Su-bar lasts 24T; Jp supplies the fraction
        _, hi, _ = _ALIGN_SPAN
        shift = self.eq_delay if self._filter is not None else 0
        if len(self._rx) < base + hi + 1 + shift + 600:
            return
        if self.fractional_rx_reacquisition:
            # Jp changes the analogue transmitter's phase before this second
            # TRN1u.  Reusing the fractional phase fitted to TRN1u #1 can put
            # the known training sequence between decision samples.  Acquire
            # this burst on the un-resampled (but optionally equalized) stream,
            # then carry the new phase into CPt and the following receive path.
            lo, hi, _ = _ALIGN_SPAN
            if self._trn1u2_search is None:
                offsets = np.arange(lo - 1 + shift, hi + 1 + shift, dtype=np.intp)
                first = base + int(offsets[0])
                last = base + int(offsets[-1])
                self._trn1u2_search = {
                    "first": first,
                    "positions": np.arange(first, last + 600, dtype=float),
                    "expected": np.asarray(_trn1u_bits(600), dtype=bool),
                    "phases": np.arange(0.0, 1.0, 0.05),
                    "phase_index": 0,
                    "old_phase": self.rx_fractional_phase,
                    "best": (-1.0, None, None),
                }
            search = self._trn1u2_search
            phase = float(search["phases"][search["phase_index"]])
            sampled = _windowed_sinc_sample(self._rx, search["positions"] + phase)
            windows = np.lib.stride_tricks.sliding_window_view(sampled, 600)
            scores = np.mean((windows < 0.0) == search["expected"][None, :], axis=1)
            index = int(np.argmax(scores))
            score = float(scores[index])
            at = search["first"] + index
            # Prefer the closest phase to the previous estimate only when
            # sequence scores tie; the training correlation remains the
            # acquisition criterion.
            old_phase = search["old_phase"]
            best = search["best"]
            old_distance = (0.0 if old_phase is None else
                            abs((phase - old_phase + 0.5) % 1.0 - 0.5))
            best_distance = (0.0 if best[2] is None or old_phase is None else
                             abs((best[2] - old_phase + 0.5) % 1.0 - 0.5))
            if (score > best[0] or
                    (score == best[0] and old_distance < best_distance)):
                search["best"] = (score, at, phase)
            search["phase_index"] += 1
            if search["phase_index"] < len(search["phases"]):
                return
            best = search["best"]
            self._trn1u2_search = None
            if best[0] >= 0.95:
                self.trn1u2_start = best[1]
                self.rx_fractional_phase = best[2]
                radius = RX_PHASE_TAPS // 2
                # The earlier TRN1u, Ja and Su samples have already been
                # consumed using the first phase. Only downstream decisions
                # from this reacquired training burst use the new phase. Keep
                # the old prefix and resample the new-phase suffix; rebuilding
                # all call history here caused a 78 ms media-thread stall.
                start = max(0, self.trn1u2_start - radius)
                if len(self._fractional_rx) < len(self._rx):
                    self._fractional_rx = np.pad(
                        self._fractional_rx, (0, len(self._rx) - len(self._fractional_rx)))
                resample_positions = np.arange(start, len(self._rx), dtype=float) + best[2]
                self._fractional_rx[start:] = _windowed_sinc_sample(
                    self._rx, resample_positions)
                self.events.append(("trn1u2-fractional-phase", {
                    "sample_phase": best[2], "previous_phase": old_phase,
                    "match": best[0], "start": best[1],
                }))
                self.events.append(("trn1u-2", best[1]))
                self.state = "trn1u-2"
            return
        best = self._best_near_alignment(base, radius=1)
        # Su-bar2 gives us a bounded receive window even when fractional
        # reacquisition is disabled. The old 0.95 gate silently left CPt RX
        # disarmed when Jp moved the second TRN1u by a fraction of a sample.
        min_match = (TRN1U2_FIRST_PHASE_MIN_MATCH if self.fractional_rx_timing
                     else 0.95)
        if best[0] >= min_match:
            self.trn1u2_start = best[1]
            self.events.append(("trn1u-2", best[1]))
            self.events.append(("trn1u-2-first-phase", {
                "start": best[1], "match": best[0],
                "sample_phase": self.rx_fractional_phase,
            }))
            self.state = "trn1u-2"

    def _find_cpt(self) -> None:
        """The analogue modem's CPt: 24 ones and the frame, after an unknown length of the second TRN1u."""
        need = JA_PREAMBLE + v92_cp.MAX_CPU_FRAME_BITS
        sync_bits = [1] * 17 + [0]
        prefix_bits = JA_PREAMBLE + len(sync_bits)
        # The 24 differentially encoded ones are a useful timing reference,
        # but their hard decisions can be imperfect at the TRN1u/CPt boundary.
        # Require the following CPt sync and the full Table 23 CRC instead of
        # making an exact 24/24 preamble match a condition of acquiring CPt.
        # The sync gate keeps the long TRN1u scan cheap; a CRC-valid parsed
        # Table 23 frame is the acceptance criterion.
        for k, a, preamble_ones in list(self._cpt_pending):
            available = min(len(self._rx) - a, need)
            if available <= prefix_bits:
                continue
            history_bits = self._cpt_trn1u_history
            bits = _decode_diff_scrambled(
                _signs_to_bits(self._decision_rx[a:a + available]),
                history_bits[k - 1], history_bits[k - 23:k], 5, available)
            parsed = v92_cp.parse_cpu(bits[JA_PREAMBLE:])
            if parsed is not None and parsed["kind"] == "cpt":
                self._cpt_pending.remove((k, a, preamble_ones))
                self.events.append(("cpt-preamble", {"ones": preamble_ones, "expected": JA_PREAMBLE}))
                self._accept_cpt(k, a, parsed, history_bits)
                return
            if available == need:
                self._cpt_pending.remove((k, a, preamble_ones))

        while self.trn1u2_start + self._next_cpt_try + prefix_bits <= len(self._rx):
            k = self._next_cpt_try
            self._next_cpt_try += 12
            if k < 24:
                continue
            while len(self._cpt_trn1u_history) < k + 12:
                self._cpt_trn1u_history.append(self._cpt_trn1u_scrambler.step(1))
            history_bits = self._cpt_trn1u_history
            a = self.trn1u2_start + k
            prefix = _decode_diff_scrambled(
                _signs_to_bits(self._decision_rx[a:a + prefix_bits]),
                history_bits[k - 1], history_bits[k - 23:k], 5, prefix_bits)
            preamble_ones = sum(prefix[:JA_PREAMBLE])
            if prefix[JA_PREAMBLE:] != sync_bits:
                continue
            available = min(len(self._rx) - a, need)
            bits = _decode_diff_scrambled(
                _signs_to_bits(self._decision_rx[a:a + available]),
                history_bits[k - 1], history_bits[k - 23:k], 5, available)
            parsed = v92_cp.parse_cpu(bits[JA_PREAMBLE:])
            if parsed is not None and parsed["kind"] == "cpt":
                self.events.append(("cpt-preamble", {"ones": preamble_ones, "expected": JA_PREAMBLE}))
                self._accept_cpt(k, a, parsed, history_bits)
                return
            if available < need:
                self._cpt_pending.append((k, a, preamble_ones))

    def _accept_cpt(self, k: int, a: int, parsed: dict, history_bits: list[int]) -> None:
        advertised_lookahead = v92_phase3.MAX_ADVERTISED_LOOKAHEAD
        if int(parsed.get("lookahead", 0)) > advertised_lookahead:
            self.state = "failed"
            self.events.append(("unsupported-cpt-lookahead", {
                "requested": int(parsed["lookahead"]),
                "advertised": advertised_lookahead,
            }))
            return
        self.cpt = parsed
        self.cpt_at = a                      # where the 24 ones begin: the second TRN1u ends here
        self.state = "cpt-received"
        self.events.append(("cpt", a))
        if self._dil is not None:
            self._dil.request_stop()
            self.events.append(("dil-stop-request", self.dil_symbols_sent))
            self._record_dil_stopped()
        # Carry the decoder on from the end of the 24 ones: CPt repeats, then E1u (12 zero bits).
        _, prev, reg = _decode_stateful(_signs_to_bits(self._decision_rx[a:a + JA_PREAMBLE]), history_bits[k - 1],
                                        list(reversed(history_bits[k - 23:k])), 5)
        self._cpt_dec = (a + JA_PREAMBLE, prev, reg)
        self._cpt_last_frame_dec = None
        self._cpt_len = -(-int(parsed["frame_bits"]) // 12) * 12
        self._e1u_wait_start = None
        self._e1u_windows_seen = 0

    def _decode_e1u_slip(self, at: int, prev: int, reg: list[int], offset: int):
        """Decode a 12-bit E1u candidate at a small frame-boundary slip."""
        candidate_at = at + offset
        if candidate_at < 0 or candidate_at + 12 > len(self._decision_rx):
            return None
        if offset >= 0:
            if offset:
                _, candidate_prev, candidate_reg = _decode_stateful(
                    _signs_to_bits(self._decision_rx[at:candidate_at]), prev, reg, 5)
            else:
                candidate_prev, candidate_reg = prev, reg
        else:
            prior = self._cpt_last_frame_dec
            if prior is None:
                return None
            prior_at, prior_prev, prior_reg = prior
            prefix_len = candidate_at - prior_at
            if prefix_len < 0 or prefix_len > self._cpt_len:
                return None
            _, candidate_prev, candidate_reg = _decode_stateful(
                _signs_to_bits(self._decision_rx[prior_at:candidate_at]),
                prior_prev, prior_reg, 5)
        bits, _, _ = _decode_stateful(
            _signs_to_bits(self._decision_rx[candidate_at:candidate_at + 12]),
            candidate_prev, candidate_reg, 5)
        return candidate_at, bits

    def _find_e1u(self) -> None:
        """After the CPt frames comes E1u, 12 zero bits where another CPt would start; the analogue modem's
        Phase 4 begins right after it. Noisy E1u decisions are tolerated, and
        Ri continues while ambiguous frames are scanned up to a bounded timeout."""
        # E1u is a response to our Ri. Do not interpret an early zero run (or
        # start its timeout) before at least one complete Ri has gone on wire.
        if self.ri_units < 24:
            return
        at, prev, reg = self._cpt_dec
        while at + self._cpt_len <= len(self._rx):
            bits, prev2, reg2 = _decode_stateful(_signs_to_bits(self._decision_rx[at:at + self._cpt_len]), prev, reg, 5)
            cp_sync_ones = sum(bits[:17])
            if cp_sync_ones >= 15 and bits[17] == 0:                # another CPt
                self.events.append(("e1u-window", {
                    "at": at, "classification": "cpt-repeat",
                    "cp_sync_ones": cp_sync_ones, "bits": "".join(map(str, bits)),
                }))
                self._e1u_windows_seen += 1
                self._cpt_last_frame_dec = (at, prev, reg)
                at, prev, reg = at + self._cpt_len, prev2, reg2
                self._cpt_dec = (at, prev, reg)
                continue
            slips = []
            e1u = None
            for offset in (0, -1, 1, -12, 12):
                decoded = self._decode_e1u_slip(at, prev, reg, offset)
                if decoded is None:
                    continue
                candidate_at, candidate_bits = decoded
                zero_count = sum(bit == 0 for bit in candidate_bits)
                slips.append({"offset": offset, "at": candidate_at,
                              "zero_bits": zero_count,
                              "bits": "".join(map(str, candidate_bits))})
                if zero_count >= E1U_MIN_ZERO_BITS:
                    e1u = (candidate_at, candidate_bits, zero_count, offset)
                    break
            self.events.append(("e1u-window", {
                "at": at, "classification": "e1u" if e1u else "ambiguous",
                "cp_sync_ones": cp_sync_ones,
                "bits": "".join(map(str, bits)), "slips": slips,
            }))
            self._e1u_windows_seen += 1
            if e1u is not None:
                e1u_at, e1u_bits, zero_count, slip = e1u
                self.trn2u_start = e1u_at + 12
                # E1u occupies only the first 12 symbols in this aligned
                # window.  The remaining symbols belong to TRN2u, so the
                # decoder's final state (`prev2`) is already inside training.
                # Carry the transmitted sign of E1u's final symbol instead.
                self.e1u_last_sign = int(self._decision_rx[e1u_at + 11] < 0)
                self.events.append(("e1u", e1u_at))
                self.events.append(("e1u-qualified", {
                    "at": e1u_at, "zero_bits": zero_count,
                    "threshold": E1U_MIN_ZERO_BITS, "slip": slip,
                }))
                return
            if self._e1u_wait_start is None:
                self._e1u_wait_start = at
            self._cpt_last_frame_dec = (at, prev, reg)
            at, prev, reg = at + self._cpt_len, prev2, reg2
            self._cpt_dec = (at, prev, reg)
            if len(self._rx) - self._e1u_wait_start >= E1U_WAIT_TIMEOUT_SYMBOLS:
                self.state = "failed"
                self.events.append(("e1u-timeout", {
                    "at": at, "windows": self._e1u_windows_seen,
                    "waited_symbols": len(self._rx) - self._e1u_wait_start,
                }))
                return

    def _find_trn1u(self) -> None:
        base = self.ru_flip_at + a3.RU_BAR_SYMBOLS
        if len(self._rx) < base + _ALIGN_SPAN[1] + 600:
            return
        best = self._best_alignment(base)
        if best[0] < 0.95 and self._equalise and self._filter is None and best[0] > PHASE3_EQ_MIN_MATCH:
            if self._train_phase3_filter(best[1]):
                before = best[0]
                if len(self._rx) < base + _ALIGN_SPAN[1] + self.eq_delay + 600:
                    return                              # wait for the delayed stream to cover the search
                best = self._best_alignment(base)
                self.events[-1] = (self.events[-1][0], {**self.events[-1][1],
                                                        "match_before": before, "match_after": best[0]})
        if best[0] < 0.95:
            self.state = "failed"
            self.events.append(("no-trn1u", best[0]))
            return
        self.trn1u_start = best[1]
        self.state = "wait-ja"
        self.events.append(("trn1u", self.trn1u_start))

    def _find_ja(self) -> None:
        if self.fractional_rx_timing and self.rx_fractional_phase is None:
            return
        # The old 1.25 s guard was shorter than the V.92 Ja acquisition window
        # and rejected this peer's Ja before it appeared.  Generate the known
        # TRN1u scrambler history once; scanning thousands of 12-symbol phases
        # must not regenerate the entire history for every candidate.
        if self._ja_trn_history is None:
            self._ja_trn_history = self._timed_stage(
                "rx.ja_history_init", _trn1u_bits, JA_SEARCH_TIMEOUT_SYMBOLS + 128)
        history_bits = self._ja_trn_history

        # First retry all candidates that already had a valid sync and length
        # header.  Standard padded frames can be accepted individually.  For
        # error-corrupted repeats, require an odd majority of at least five
        # copies and then use the unchanged strict descriptor/CRC parser.
        for candidate in list(self._ja_pending):
            ja_offset = candidate["ja_offset"]
            desc_offset = candidate["descriptor_offset"]
            descriptor_len = candidate["descriptor_len"]
            ja_start = self.trn1u_start + ja_offset
            for period in candidate["periods"]:
                available = (len(self._rx) - (ja_start + desc_offset)) // period
                if available < 1:
                    continue
                # One clean, canonical descriptor remains the fast path.
                if (period == descriptor_len and available >= 1
                        and not candidate.get("single_tried")):
                    total = desc_offset + descriptor_len
                    samples = self._decision_rx[ja_start:ja_start + total]
                    decoded = self._timed_stage(
                        "rx.ja_decode", _decode_diff_scrambled,
                        _signs_to_bits(samples), history_bits[ja_offset - 1],
                        history_bits[ja_offset - 23:ja_offset], 5, total)
                    parsed = self._timed_stage(
                        "rx.ja_parse", v92_ja.parse_ja_descriptor,
                        decoded[desc_offset:total])
                    candidate["single_tried"] = True
                    if parsed is not None:
                        self._accept_ja(parsed, ja_start)
                        return
                repetitions = min(7, available)
                if repetitions % 2 == 0:
                    repetitions -= 1
                if repetitions < 5 or repetitions <= candidate["tried"].get(period, 0):
                    continue
                total = desc_offset + period * repetitions
                samples = self._decision_rx[ja_start:ja_start + total]
                decoded = self._timed_stage(
                    "rx.ja_decode", _decode_diff_scrambled,
                    _signs_to_bits(samples), history_bits[ja_offset - 1],
                    history_bits[ja_offset - 23:ja_offset], 5, total)
                parsed = self._timed_stage(
                    "rx.ja_parse_repeated", v92_ja.parse_repeated_ja_descriptor,
                    decoded[desc_offset:total], repeat_bits=period,
                    repetitions=repetitions)
                candidate["tried"][period] = repetitions
                if parsed is not None:
                    self._accept_ja(parsed, ja_start)
                    return

        # Search two legal-looking onset geometries.  The standard form has a
        # separate 24-one preamble followed by the descriptor sync.  The
        # interoperability form observed here has the 17-bit descriptor sync
        # as the final 17 symbols of that 24-one run (descriptor offset +7).
        # In both cases the candidate descriptor sync positions are scanned on
        # the 12-symbol phase and all accepted descriptors still pass CRC.
        while (self._next_ja_try <= JA_SEARCH_TIMEOUT_SYMBOLS
               and self.trn1u_start + self._next_ja_try + JA_PREAMBLE + 50 <= len(self._rx)):
            k = self._next_ja_try
            self._next_ja_try += 12

            # Canonical: k is the start of 24 ones; descriptor begins at k+24.
            ja_start_offset = k
            a = self.trn1u_start + ja_start_offset
            total = JA_PREAMBLE + 50
            decoded = _decode_diff_scrambled(_signs_to_bits(self._decision_rx[a:a + total]),
                                              history_bits[ja_start_offset - 1],
                                              history_bits[ja_start_offset - 23:ja_start_offset], 5, total)
            if decoded[:JA_PREAMBLE] == [1] * JA_PREAMBLE:
                prefix = decoded[JA_PREAMBLE:]
                descriptor_len = v92_ja.descriptor_bits_from_prefix(prefix)
                periods = v92_ja.descriptor_repeat_lengths_from_prefix(prefix)
                if descriptor_len is not None and periods is not None:
                    self._ja_pending.append({"ja_offset": ja_start_offset,
                                             "descriptor_offset": JA_PREAMBLE,
                                             "descriptor_len": descriptor_len,
                                             "periods": periods, "tried": {}})

            # Compact peer framing: sync starts at k, while the first seven
            # Ja ones precede it. Decode from k-7 so the scrambler and
            # differential state include those seven symbols.
            ja_start_offset = k - 7
            if ja_start_offset <= 0:
                continue
            a = self.trn1u_start + ja_start_offset
            if a + total > len(self._rx):
                continue
            decoded = _decode_diff_scrambled(_signs_to_bits(self._decision_rx[a:a + total]),
                                              history_bits[ja_start_offset - 1],
                                              history_bits[ja_start_offset - 23:ja_start_offset], 5, total)
            if decoded[:JA_PREAMBLE] != [1] * JA_PREAMBLE:
                continue
            prefix = decoded[7:57]
            descriptor_len = v92_ja.descriptor_bits_from_prefix(prefix)
            periods = v92_ja.descriptor_repeat_lengths_from_prefix(prefix)
            if descriptor_len is not None and periods is not None:
                self._ja_pending.append({"ja_offset": ja_start_offset,
                                         "descriptor_offset": 7,
                                         "descriptor_len": descriptor_len,
                                         "periods": periods, "tried": {}})

        if self._next_ja_try > JA_SEARCH_TIMEOUT_SYMBOLS:
            # A candidate that began before the acquisition deadline may need
            # several repeated frames to reach a CRC-valid majority. Give each
            # plausible, length-checked candidate enough time for up to seven
            # copies, then fail boundedly if none validates.
            self._ja_pending = [candidate for candidate in self._ja_pending
                                if len(self._rx) < (self.trn1u_start + candidate["ja_offset"]
                                                    + candidate["descriptor_offset"]
                                                    + max(candidate["periods"]) * 7)]
            if not self._ja_pending:
                self.state = "failed"
                self.events.append(("no-ja", self._next_ja_try))

    def _begin_downstream(self) -> None:
        self.state = "sending"
        # The first Su-bar is 24.5T after Su and can land between PCM samples.
        # A lower correlation floor handles the resampled SmartLink capture;
        # the fixed 144T transition window below prevents early false flips.
        self.su_tracker = PatternTracker(
            families=("su",), fractional_timing=True, min_correlation=0.76)
        self._su_origin = len(self._rx)         # the tracker counts from here
        self.sd_start_sample = self.tx_symbols_emitted + len(self._queue)
        self.events.append(("sd-tx-start", self._tx_frame_marker(self.sd_start_sample)))
        self._queue += self._initial_downstream
        self._initial_downstream.clear()

    def _tx_frame_marker(self, sample: int) -> dict:
        """Report a TX position relative to Sd, the V.90 data-frame origin."""
        phase = None if self.sd_start_sample is None else (int(sample) - self.sd_start_sample) % 6
        return {"sample": int(sample), "frame_phase_from_sd": phase}

    def _accept_ja(self, parsed: dict, ja_start: int) -> None:
        """Latch Ja and prepare its DIL before J'_d can arrive on the TX path."""
        self.ja, self.ja_start = parsed, ja_start
        self.events.append(("ja", ja_start))
        if int(parsed["N"]):
            self._dil_prepare_future = Future()
            self._timed_stage("rx.prepare_dil_dispatch", _DIL_PREPARE_TASKS.put,
                              (self._dil_prepare_future, self._tx, dict(parsed)))
        self._timed_stage("rx.begin_downstream", self._begin_downstream)

    def _prepared_dil_for_tx(self):
        """Take the off-thread Ja-derived DIL, falling back without waiting."""
        ready = (self._dil_prepare_future is not None
                 and self._dil_prepare_future.done())
        if ready:
            self._prepared_dil, self._dil_prepare_worker_ns = (
                self._dil_prepare_future.result())
            return self._prepared_dil
        self._dil_prepare_fallbacks += 1
        self.events.append(("dil-preparation-late", self._dil_prepare_fallbacks))
        # Do not wait for a worker that missed the protocol boundary. The
        # synchronous fallback preserves the signal sequence while the event
        # makes any deadline miss explicit in diagnostics.
        return self._tx.prepare_dil_stream(self.ja)

    def _jp_frame(self) -> bytes:
        t = self._tx
        signs = []
        for bit in v92_phase3.build_jp(self.jp_fraction or 0):
            t.differential_sign ^= t.scrambler.step(bit)
            signs.append(t.differential_sign)
        self.jp_sent += 1
        return bytes(t._codeword(bool(sign)) for sign in signs)

    def _scr_chunk(self) -> bytes:
        t = self._tx
        self.scr_symbols += 6
        return bytes(t._codeword(bool(t.scrambler.step(1))) for _ in range(6))

    def tx(self, count: int, *, stop_at_phase4: bool = False) -> bytes:
        """Render up to ``count`` symbols.

        The live session may request a short return at the exact R-bar-i end
        so its Phase 4 transmitter can continue in the same media block.
        Other callers keep the historical fixed-size, silence-padded result.
        """
        return self._timed_stage("tx.render_block", self._render_tx_block,
                                 count, stop_at_phase4)

    def _render_tx_block(self, count: int, stop_at_phase4: bool = False) -> bytes:
        while self.state in ("sending", "trn1u-2", "cpt-received") and len(self._queue) < count:
            if self._jprime_sent and self._dil is None and int(self.ja["N"]):
                self.dil_start_sample = self.tx_symbols_emitted + len(self._queue)
                self.events.append(("dil-tx-start", self._tx_frame_marker(self.dil_start_sample)))
                prepared = self._prepared_dil_for_tx()
                self._dil = self._timed_stage(
                    "tx.create_dil", self._tx.dil_stream, self.ja,
                    prepared=prepared)
                self.events.append(("dil-start", self.ja["N"]))
            if self._jprime_sent and self._dil is not None and not self._dil.stopped:
                chunk = self._timed_stage(
                    "tx.read_dil", self._dil.read, count - len(self._queue))
                self.dil_symbols_sent += len(chunk)
                self._queue += chunk
                self._record_dil_stopped()
            elif self._jprime_sent and self._dil is not None and self.cpt is not None:
                if self.trn2u_start is not None:
                    self.rbar_i_start_sample = self.tx_symbols_emitted + len(self._queue)
                    self.events.append(("r-bar-i-tx-start",
                                        self._tx_frame_marker(self.rbar_i_start_sample)))
                    self._queue += self._timed_stage(
                        "tx.render_rbar_i", dp4.rbar_i_codewords, self.uinfo,
                        self.law, duration_symbols=24)
                    self.rbar_sent = True
                    self.state = "phase4-ready"
                    self.events.append(("r-bar-i-sent", self.ri_units))
                else:
                    if self.ri_units == 0:
                        self.ri_start_sample = self.tx_symbols_emitted + len(self._queue)
                        self.events.append(("ri-tx-start",
                                            self._tx_frame_marker(self.ri_start_sample)))
                    self._queue += self._timed_stage(
                        "tx.render_ri", dp4.ri_codewords, self.uinfo,
                        self.law, duration_symbols=24)
                    self.ri_units += 24
                    if self.ri_units == 24:
                        self.events.append(("ri-sent", "cpt"))
            elif self._jprime_sent and self.cpt is not None:
                # CPt seen: R-bar-i for 24T, then Phase 4.
                self.rbar_i_start_sample = self.tx_symbols_emitted + len(self._queue)
                self.events.append(("r-bar-i-tx-start",
                                    self._tx_frame_marker(self.rbar_i_start_sample)))
                self._queue += self._timed_stage(
                    "tx.render_rbar_i", dp4.rbar_i_codewords, self.uinfo,
                    self.law, duration_symbols=24)
                self.rbar_sent = True
                self.state = "phase4-ready"
                self.events.append(("r-bar-i-sent", self.ri_units))
            elif self._jprime_sent and self.scr_symbols >= SCR_MIN:
                if self.ri_units == 0:
                    self.ri_start_sample = self.tx_symbols_emitted + len(self._queue)
                    self.events.append(("ri-tx-start",
                                        self._tx_frame_marker(self.ri_start_sample)))
                self._queue += self._timed_stage(
                    "tx.render_ri", dp4.ri_codewords, self.uinfo,
                    self.law, duration_symbols=24)
                self.ri_units += 24
                if self.ri_units == 24:
                    self.events.append(("ri-sent", self.scr_symbols))
            elif self._jprime_sent:
                self._queue += self._timed_stage("tx.render_scr", self._scr_chunk)
            elif self._flip3 and self.jp_sent:
                # The last Su-to-Su-bar flip: finish the current Jp (it is whole), then Jp'.
                self._queue += self._timed_stage("tx.render_jprime_d", self._tx.jprime_d)
                self._jprime_sent = True
                self.events.append(("jp-prime-sent", self.scr_symbols))
            elif self._send_jp:
                self._queue += self._timed_stage("tx.render_jp", self._jp_frame)
            else:
                if self.jd_sent == 0 and self.jd_start_sample is None:
                    self.jd_start_sample = self.tx_symbols_emitted + len(self._queue)
                    self.events.append(("jd-tx-start",
                                        self._tx_frame_marker(self.jd_start_sample)))
                self._queue += self._timed_stage(
                    "tx.render_jd", self._tx.jd_frame, self.rate_mask)
                self.jd_sent += 1
        out = bytes(self._queue[:count])
        del self._queue[:count]
        if stop_at_phase4 and self.state == "phase4-ready":
            self.tx_symbols_emitted += len(out)
            return out
        # Silence padding is transmitted media too and advances the frame
        # phase. Keep this coordinate identical to the DCE's output timeline.
        self.tx_symbols_emitted += count
        return out + bytes([self._silence]) * (count - len(out))

    def _record_dil_stopped(self) -> None:
        if (self._dil is not None and self._dil.stopped
                and not self._dil_stop_reported):
            self._dil_stop_reported = True
            self.events.append(("dil-stopped", self.dil_symbols_sent))


class AnalogueSide:
    def __init__(self, descriptor, *, lu: float = 1000.0, law: str = "pcmu",
                 silence_samples: int = a3.SILENCE_SAMPLES, cpt_bits=None,
                 fractional_tx_timing: bool = False):
        self.law = law
        parsed_descriptor = v92_ja.parse_ja_descriptor(descriptor)
        if parsed_descriptor is None:
            raise ValueError("analogue Phase 3 needs a valid V.92 Ja descriptor")
        ja_bits = max(2232, 24 + 2 * len(descriptor))
        ja_bits += -ja_bits % 12
        self.plan = a3.build(descriptor, lu=lu, ja_bits=ja_bits, silence_samples=silence_samples)
        self._pos = 0
        self._ja_end: int | None = None
        self.tracker = PatternTracker(families=("su",))     # Sd has the Su shape
        self._rx = np.empty(0)
        self.state = "sending"
        self.events: list[tuple[str, object]] = []
        self.sd_flip_at: int | None = None
        self.trn1d_start: int | None = None
        self.jd: dict | None = None
        self._next_jd_try = TRN_MIN
        # Jd can arrive at any point after a long TRN1d. Retain the GPC
        # history and extend it only for newly examined positions. Rebuilding
        # the entire sequence from zero for every candidate made each RTP
        # packet repeat O(k) scrambler work, and the total Phase-3 scan grew
        # quadratically until Jd was found.
        self._trn1d_scrambler = v90_digital_tx._GPC()
        self._trn1d_history: list[int] = []
        self.uinfo_guess = None
        self.lu = lu
        self._su_state: str | None = None
        self._su_start = 0
        self._su_n = 0
        self.trn1u2_start: int | None = None
        self._trn2 = _trn1u_bits(a3.DEFAULT_TRN1U)
        self.jp: dict | None = None
        self.jp_prime_seen = False
        self._dec_at = 0
        self._dec_prev = 0
        self._dec_reg: list[int] = []
        self._jp_seen_flag = False
        if type(fractional_tx_timing) is not bool:
            raise TypeError("fractional_tx_timing must be bool")
        self.fractional_tx_timing = fractional_tx_timing
        # The source is still generated at one sample per T.  This cumulative
        # causal fractional delay moves the two Su-bar boundaries off the
        # sample grid while keeping the externally clocked PCM block size fixed.
        # V.92 §9.5.2.1.7 adds 0.5T after the first bar; §9.5.2.1.8 adds Jp's
        # 0..1T fraction after the second bar.
        self._tx_fractional_delay = 0.0
        self._last_tx_delay_trace = np.empty(0, dtype=float)
        self.ri_tracker: PatternTracker | None = None
        self._ri_origin: int | None = 0
        self._ri_seen = self._rbar_seen = False
        self._t2_scr = v34_tx.Scrambler("answer")
        self._t2_last = 0
        self._cpt_bits = (list(cpt_bits) if cpt_bits is not None else
                          v92_cp.build_cpu(kind=v92_cp.CPT, drn=9, constellations=[(10, 20, 30)]))
        self._nonzero_dil = bool(parsed_descriptor["N"])
        self._cpt_queue: list[int] = []
        self._cpt_started = False
        self.cpt_frames_sent = 0
        self.e1u_sent = False
        # Sample position immediately after the final E1u symbol.  The live
        # caller uses this to hand the remainder of the current RTP block to
        # Phase 4 without inserting packet-tail silence.
        self.e1u_end_sample: int | None = None
        self._cpt_prev = 0

    def _su_sample(self) -> float:
        """Next Su sample; fractional boundary timing is applied by the TX delay stage."""
        a = (1.5 ** 0.5) * self.lu
        pattern = (a, 0.0, a, -a, 0.0, -a)
        st = self._su_state
        n = self._su_n
        if st == "trn2":
            y = self._t2_scr.step(1)
            self._t2_last = y
            self._su_n += 1
            # With nonzero DIL, send CPt once the required TRN1u has been sent;
            # the digital modem then responds with Ri. For N=0, retain the
            # existing SCR/Ri-before-CPt sequence.
            if self._nonzero_dil and self._su_n >= TRN_MIN and self._su_n % 12 == 0:
                self._su_state, self._cpt_prev = "cpt", self._t2_last
            elif self._su_n >= a3.DEFAULT_TRN1U and self._su_n % 12 == 0 and self._ri_seen:
                self._su_state, self._cpt_prev = "cpt", self._t2_last
            return -self.lu if y else self.lu
        if st == "cpt":
            return self._cpt_sample()
        if st in ("done", None):
            return 0.0
        sign = -1.0 if st in ("bar1", "bar2") else 1.0
        value = sign * pattern[n % 6]
        self._su_n += 1
        if st == "su1" and self._su_n >= 144:
            self._su_state, self._su_n = "bar1", 0
        elif st == "bar1" and self._su_n >= 24:
            self._su_state, self._su_n = "hold", 0
            if self.fractional_tx_timing:
                self._tx_fractional_delay += 0.5
                self.events.append(("su-bar1-fractional-end", {
                    "added_samples": 0.5,
                    "cumulative_delay_samples": self._tx_fractional_delay,
                }))
        elif st == "hold" and self._jp_seen_flag and self._su_n % 6 == 0:
            self._su_state, self._su_n = "bar2", 0
        elif st == "bar2" and self._su_n >= 24:
            self._su_state, self._su_n = "trn2", 0
            if self.fractional_tx_timing:
                fraction = (0.0 if self.jp is None else
                            int(self.jp["phase_fraction"]) / 65536.0)
                self._tx_fractional_delay += fraction
                self.events.append(("su-bar2-fractional-end", {
                    "jp_fraction_symbols": fraction,
                    "cumulative_delay_samples": self._tx_fractional_delay,
                }))
        return value

    def _cpt_sample(self) -> float:
        """24 ones then CPt, repeated; finish with E1u at the negotiated response edge."""
        if not self._cpt_queue:
            if not self._cpt_started:
                self._cpt_queue = [1] * 24 + self._padded_cpt()
                self._cpt_started = True
            elif self.e1u_sent:
                self._su_state = "done"
                return 0.0
            elif (self._ri_seen if self._nonzero_dil else self._rbar_seen):
                self._cpt_queue = [0] * 12                    # E1u: scrambled, differentially encoded zeros
                self.e1u_sent = True
            else:
                self._cpt_queue = self._padded_cpt()
        bit = self._cpt_queue.pop(0)
        if not self._cpt_queue and not self.e1u_sent:
            self.cpt_frames_sent += 1
        y = self._t2_scr.step(bit)
        self._cpt_prev ^= y
        return -self.lu if self._cpt_prev else self.lu

    def _padded_cpt(self) -> list[int]:
        bits = [int(b) & 1 for b in self._cpt_bits]
        return bits + [0] * (-len(bits) % 12)

    def tx(self, count: int) -> np.ndarray:
        out = np.zeros(count)
        delay_trace = (np.zeros(count, dtype=float)
                       if self.fractional_tx_timing else None)
        n = len(self.plan.samples)
        for i in range(count):
            at = self._pos + i
            if self._su_state is not None and at >= self._su_start:
                before = self._su_state
                if delay_trace is not None:
                    delay_trace[i] = self._tx_fractional_delay
                out[i] = self._su_sample()
                if before == "bar2" and self._su_state == "trn2" and self.trn1u2_start is None:
                    self.trn1u2_start = at + 1
                # E1u is the final 12-symbol sequence emitted by Phase 3.  Its
                # bit queue empties on the sample carrying its final symbol;
                # the next sample is the start of Phase 4.
                if (self.e1u_sent and not self._cpt_queue
                        and self.e1u_end_sample is None):
                    self.e1u_end_sample = at + 1
            elif at < n and (self._ja_end is None or at < self._ja_end):
                out[i] = self.plan.samples[at]
        if delay_trace is not None:
            self._last_tx_delay_trace = delay_trace
        self._pos += count
        return out

    def feed(self, samples) -> None:
        x = np.asarray(samples, dtype=float).reshape(-1)
        self._rx = np.concatenate((self._rx, x))
        for name, at in self.tracker.feed(x):
            if name.endswith("bar") and self.sd_flip_at is None and self.state == "sending":
                self.sd_flip_at = at
                self.events.append(("sd-flip", at))
                # Ja ends at the next 12-bit boundary of its own bit count.
                ja0 = self.plan.boundaries["ja"][0]
                done = max(0, self._pos - ja0)
                self._ja_end = ja0 + -(-done // 12) * 12
                self.state = "wait-trn1d"
        if self.state == "wait-trn1d":
            self._find_jd()
        if self.state in ("jd-received", "jp-received"):
            self._read_frames()
        if self.ri_tracker is not None:
            if self._ri_origin is None:
                self._ri_origin = len(self._rx) - len(x)     # the tracker starts with this chunk
            for name, at in self.ri_tracker.feed(x):
                if not self._ri_seen:
                    self._ri_seen = True
                    self.events.append(("ri", at + self._ri_origin))
                elif name.endswith("bar") and not self._rbar_seen:
                    self._rbar_seen = True
                    self.events.append(("r-bar-i", at + self._ri_origin))

    def _find_jd(self) -> None:
        base = self.sd_flip_at + 48
        need = v92_phase3.JD_BITS
        while base + self._next_jd_try + need <= len(self._rx):
            k = self._next_jd_try
            self._next_jd_try += 6
            self._ensure_trn1d_history(k + 6)
            history_bits = self._trn1d_history
            a, b = base + k, base + k + need
            if k == TRN_MIN:
                trn_match = _match(self._rx[base:base + 600], history_bits,
                                   positive_is_one=True)
                if trn_match < TRN1D_MIN_MATCH:
                    # The six-sample pattern tracker can see an Sd-shaped
                    # interval in earlier call audio (for example a phase-2
                    # signal). A bad first TRN1d window rejects this candidate;
                    # it must not terminate Phase 3 before the real Sd/Sd' pair.
                    # Keep the shared tracker running and re-arm on its next
                    # Ru/Su-to-bar transition.
                    self.events.append(("sd-candidate-rejected", {
                        "flip_at": self.sd_flip_at,
                        "trn1d_match": trn_match,
                    }))
                    self.sd_flip_at = None
                    self._next_jd_try = TRN_MIN
                    self._trn1d_scrambler = v90_digital_tx._GPC()
                    self._trn1d_history = []
                    self.state = "sending"
                    return
            # Sign convention for the digital modem: positive codeword = sign bit 1.
            symbol_bits = [1 if v > 0 else 0 for v in self._rx[a:b]]
            bits = _decode_diff_scrambled(symbol_bits, history_bits[k - 1],
                                          history_bits[k - 23:k], 18, need)
            parsed = v92_phase3.parse_jd_jp(bits)
            if parsed is not None and parsed["kind"] == "jd":
                self.jd = parsed
                self.state = "jd-received"
                self.events.append(("jd", a))
                # Carry the decoder on from the end of this Jd, and start the Su sequence now.
                _, prev, reg = _decode_stateful(symbol_bits, history_bits[k - 1],
                                                list(reversed(history_bits[k - 23:k])), 18)
                self._dec_at, self._dec_prev, self._dec_reg = b, prev, reg
                self._su_state, self._su_n, self._su_start = "su1", 0, self._pos
                return
            if k > TRN_MIN + 32000:
                self.state = "failed"
                self.events.append(("no-jd", k))
                return

    def _ensure_trn1d_history(self, count: int) -> None:
        """Extend the known TRN1d GPC sequence without regenerating its prefix."""
        if count < 0:
            raise ValueError("TRN1d history length cannot be negative")
        while len(self._trn1d_history) < count:
            self._trn1d_history.append(self._trn1d_scrambler.step(1))

    def _read_frames(self) -> None:
        """After the first Jd: read on frame by frame. Another Jd, then a Jp (which releases the
        second Su-bar), then more Jp until Jp' (twelve zero bits) is seen."""
        while self._dec_at + v92_phase3.JD_BITS <= len(self._rx) and not self.jp_prime_seen:
            a = self._dec_at
            symbol_bits = [1 if v > 0 else 0 for v in self._rx[a:a + v92_phase3.JD_BITS]]
            bits, prev, reg = _decode_stateful(symbol_bits, self._dec_prev, self._dec_reg, 18)
            parsed = v92_phase3.parse_jd_jp(bits)
            if parsed is not None:
                self._dec_at, self._dec_prev, self._dec_reg = a + v92_phase3.JD_BITS, prev, reg
                if parsed["kind"] == "jp" and self.jp is None:
                    self.jp = parsed
                    self._jp_seen_flag = True
                    self.state = "jp-received"
                    self.events.append(("jp", a))
                    self.events.append(("jp-decoded", {
                        "sample": a,
                        "phase_fraction": parsed["phase_fraction"],
                        "phase_symbols": parsed["phase_symbols"],
                        "training_8point": parsed["training_8point"],
                        "renegotiation_8point": parsed["renegotiation_8point"],
                    }))
                continue
            if self.jp is not None and bits[:12] == [0] * 12:
                self.jp_prime_seen = True
                self.state = "jp-prime-seen"
                self.ri_tracker = PatternTracker(families=("ru",))     # Ri has the Ru shape
                self._ri_origin = None
                self.events.append(("jp-prime", a))
                return
            return                                   # not a frame yet: wait for more samples
