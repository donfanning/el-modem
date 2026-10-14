"""Bounded per-call TRN-only clock/tap fit; opt-in and ordinary-detector handoff."""
from collections import deque
import math
import os
import time
import numpy as np
import v34_tx as tx
from v34_duplex_model import bootstrap_benefit

MAX_ROWS = 4096
# A fitted clock slope this large (samples per 1000 symbols) is a drift of about 20 ppm or more; undrifted lanes,
# including noise-limited ones, fitted at most 0.007 in the lane studies (lab/V34_CLOCK_ACCEPT_LANE.md).
MIN_SLOPE_PER_1000 = 0.05


def decide(improvement, bounds, slope_per_symbol):
    """TRN-only acceptance of the fitted clock; returns (accepted, reason).

    Primary rule: at least 10% frozen-continuation validation improvement and positive 1% block-bootstrap bounds.
    Supplementary rule: a clearly non-zero fitted slope with a positive improvement. At moderate drift the
    validation blocks span too little time for the improvement to stand out, but the slope extrapolated to the
    data phase matters (lab/V34_CLOCK_SLOPE_GATE.md). Neither rule reads B1 symbols, errors or payload.
    """
    if improvement >= .1 and all(v[0] > 0 for v in bounds.values()):
        return True, 'validation'
    if improvement > 0 and abs(slope_per_symbol)*1000 >= MIN_SLOPE_PER_1000:
        return True, 'slope'
    return False, None

def windows(source, positions, rotation, shift):
    source, positions = np.asarray(source), np.asarray(positions)
    if os.environ.get('V34_TRN_CLOCK_NATIVE') == '1':
        from v34_interp_native import windows as native_windows
        return native_windows(source, positions, np.asarray(rotation), np.asarray(shift))
    p = positions+np.asarray(shift)[:, None]
    whole = np.floor(p).astype(int)
    offsets = np.arange(-5, 7)
    ids = whole[:, :, None]+offsets
    if ids.min() < 0 or ids.max() >= source.shape[1]:
        raise ValueError('timing intervention exceeds retained interpolation support')
    distance = offsets-(p-whole)[:, :, None]
    weights = np.sinc(distance)*(.5+.5*np.cos(np.pi*distance/6))
    values = source[np.arange(len(source))[:, None, None], ids]
    return np.sum(values*weights, axis=2)*np.asarray(rotation)[:, None]


def fit_clock(source, positions, rotation, live_dt, indices, wanted, prior, fit):
    anchor = float(np.mean(indices[fit]))
    time = (indices-anchor)/1000
    beta = float(live_dt[np.flatnonzero(fit)[0]])
    rate = 0.; iterations = []
    # The normal equations use only fit rows. Validation rows are evaluated
    # once afterward with the finished model, not during these iterations.
    source, positions, rotation, live_dt, wanted, time = (
        value[fit] for value in (source, positions, rotation, live_dt, wanted, time))
    for _ in range(5):
        shift = beta+rate*time-live_dt
        value = windows(source, positions, rotation, shift)@prior
        derivative = ((windows(source, positions, rotation, shift+.05)
                       -windows(source, positions, rotation, shift-.05))@prior/.1)
        basis = np.column_stack((wanted, 1j*wanted, derivative, derivative*time))
        matrix = np.vstack((basis.real, basis.imag))
        if np.linalg.matrix_rank(matrix) != 4:
            raise ValueError('timing/gain fit is rank deficient')
        coefficients = np.linalg.lstsq(matrix, np.r_[value.real, value.imag], rcond=None)[0]
        beta -= float(np.clip(coefficients[2], -.2, .2))
        rate -= float(np.clip(coefficients[3], -.5, .5))
        iterations.append(dict(offset=beta, samples_per_1000_symbols=rate,
                               proposed_step=coefficients[2:].tolist()))
    return dict(anchor_trn_symbol=anchor, offset_samples=beta,
                samples_per_symbol_correction=rate/1000, iterations=iterations,
                fit_uses_b1=False, finite_difference_samples=.05)


def clock_shift(model, indices, live_dt):
    return (model['offset_samples']+model['samples_per_symbol_correction']
            *(np.asarray(indices)-model['anchor_trn_symbol'])-live_dt)


def ridge_taps(matrix, wanted):
    gram = matrix.conj().T@matrix
    ridge = .001*np.trace(gram).real/len(gram)
    return np.linalg.solve(gram+ridge*np.eye(len(gram)), matrix.conj().T@wanted)



class TrnClock:
    def __init__(self):
        self.rows = deque(maxlen=MAX_ROWS)
        self.model = None
        self.taps = None
        self.applied = False
        self.peak_rows = 0

    def before_symbol(self, receiver):
        if self.model is not None or receiver.trn_end is not None:
            return
        eq = receiver.eq
        eq._clock_capture = True
        window = eq.window(eq.k).copy()
        half = eq.R.spl/2
        center = eq.R.lag+(eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl+eq.dt
        positions = center+(np.arange(eq.L)-eq.span)*half
        origin, source = getattr(eq.R, '_bb_origin', 0), eq.R._z
        direct = abs(eq.dt-(eq._g_dt+eq._g_shift*half)) > 1e-12 or eq.R.spl != eq._g_spl
        if not direct:
            if not hasattr(eq, '_clock_grid_source'):
                return
            origin, source, grid_positions = eq._clock_grid_source
            first = 2*(eq.k-eq._g_k0)+eq._g_shift+eq.MARGIN
            positions = grid_positions[first:first+eq.L]
        lo = int(np.floor(positions.min()))-8
        width = int(np.ceil((eq.L-1)*half))+19
        indices = np.arange(lo, lo+width)-origin
        if indices.min() < 0 or indices.max() >= len(source):
            self.model = dict(accepted=False, reason='incomplete interpolation support')
            self.rows.clear()
            eq._clock_capture = False
            return
        rotation = np.exp(-1j*eq.phase)
        self.rows.append(dict(index=eq.k-receiver.k_trn, source=source[indices].copy(),
                              positions=positions-lo, origin=lo, rotation=rotation, dt=eq.dt,
                              taps=eq.h.copy(), window=window*rotation))
        self.peak_rows = max(self.peak_rows, len(self.rows))

    def after_feed(self, receiver):
        if self.model is not None or receiver.trn_end is None:
            return
        started = time.monotonic()
        try:
            end = receiver.trn_end
            if end < 2400:
                raise ValueError('insufficient guarded TRN')
            rows = [r for r in self.rows if end-1600 <= r['index'] < end-100]
            ids = np.array([r['index'] for r in rows])
            fit, hold = ids < end-1100, ids >= end-600
            if (not np.array_equal(ids[fit], np.arange(end-1600, end-1100))
                    or not np.array_equal(ids[hold], np.arange(end-600, end-100))):
                raise ValueError('missing guarded TRN symbols')
            source, positions, rotation, dt, original = [np.asarray([r[k] for r in rows])
                for k in ('source', 'positions', 'rotation', 'dt', 'window')]
            # Retained rows have absolute positions reconstructed from the actual
            # cached interpolation operator, plus complete 81-tap FIR support.
            centers = np.array([r['origin'] for r in rows])
            gap = float((centers[hold]-40).min()-(centers[fit]+source.shape[1]-1+40).max())
            if gap <= 0:
                raise ValueError('TRN PCM support overlaps')
            known, _ = tx.trn_symbols(end, receiver.role, 16 if receiver.c16 else 4)
            desired = known[ids]
            self._dump_inputs(receiver, end, ids, fit, hold, source, positions, rotation, dt, original, desired, rows[0]['taps'])
            zero = windows(source, positions, rotation, np.zeros(len(ids)))
            if np.max(abs(zero-original)) > 1e-10:
                raise ValueError('zero-shift interpolation parity failed')
            result = fit_clock(source, positions, rotation, dt, ids, desired, rows[0]['taps'], fit)
            corrected = windows(source, positions, rotation, clock_shift(result, ids, dt))
            baseline_h = ridge_taps(zero[fit], desired[fit])
            candidate_h = ridge_taps(corrected[fit], desired[fit])
            baseline = abs(zero[hold]@baseline_h-desired[hold])**2
            candidate = abs(corrected[hold]@candidate_h-desired[hold])**2
            improvement = 1-float(candidate.mean())/max(float(baseline.mean()), 1e-30)
            bounds = bootstrap_benefit(baseline-candidate)
            accepted, accepted_by = decide(improvement, bounds, result['samples_per_symbol_correction'])
            result.update(accepted=accepted, accepted_by=accepted_by,
                          native_windows=os.environ.get('V34_TRN_CLOCK_NATIVE') == '1',
                          validation_improvement=improvement, benefit_block_bootstrap_1_99=bounds,
                          fit_symbols=[end-1600,end-1100], validation_symbols=[end-600,end-100],
                          pcm_support_gap_samples=gap, uses_b1_symbols=False, uses_b1_errors=False)
            self.model, self.taps = result, candidate_h
        except (ValueError, np.linalg.LinAlgError) as error:
            self.model = dict(accepted=False, reason=str(error))
        finally:
            self.model['fit_seconds'] = time.monotonic()-started
            self.rows.clear()
            receiver.eq._clock_capture = False
            receiver.eq.__dict__.pop('_clock_grid_source', None)

    @staticmethod
    def _dump_inputs(receiver, end, ids, fit, hold, source, positions, rotation, dt, window, desired, taps0):
        """Opt-in (V34_CLOCK_FIT_DUMP=<path prefix>): save the fit inputs so a call's fit can be redone offline with
        fit_clock(source, positions, rotation, dt, ids, desired, taps0, fit) and other gates, whether or not the call connected."""
        prefix = os.environ.get('V34_CLOCK_FIT_DUMP')
        if not prefix:
            return
        try:
            np.savez_compressed(prefix+'.trnfit.npz', source=source, positions=positions, rotation=rotation, dt=dt, window=window,
                                ids=ids, fit=fit, hold=hold, desired=desired, taps0=taps0, trn_end=end,
                                role=str(receiver.role), c16=bool(receiver.c16))
        except Exception:
            pass     # diagnostics must never affect the call

    GATE_RELATIVE = float(os.environ.get("V34_CLOCK_FIT_GATE_REL", "0.15"))   # fitted velocity may differ from the carrier-derived value by this fraction ... (env: test-only tolerance)
    GATE_ABSOLUTE = 5e-5       # ... or by this many samples per symbol, whichever is larger

    @staticmethod
    def carrier_derived_velocity(eq):
        """Timing velocity (samples/symbol) implied by the carrier loop's frequency offset.

        A sampling-clock error moves the carrier and the symbol rate by the same fraction, so the carrier loop's offset
        (radians per symbol) fixes the timing drift: v = -freq * (baud / 2 pi) / fc * samples-per-symbol."""
        return -eq.freq*(eq.R.baud/(2*math.pi))/eq.R.fc*eq.R.spl

    FALLBACK_MIN_VELOCITY = 1e-4   # carrier-derived |velocity| below this (about 30 ppm) means no correction is needed

    def _gate_fallback(self, eq):
        """Opt-in (V34_CLOCK_FIT_GATE=1): a fit that validation rejected is still applied, with the carrier-derived velocity,
        when the carrier loop shows a real clock offset and the fit's own velocity agrees with it (lab/V34_DIALUP_CLOCK_FIT_SWEEP.md:
        dialup.world gs3 was rejected at -48% yet decoded cleanly once applied, and failed B1 with 25 errors without it).
        Every rejected fit on Queens and TuxedoCat has a carrier-derived velocity below 3e-6, so those calls are untouched."""
        m = self.model
        if os.environ.get('V34_CLOCK_FIT_GATE') != '1' or not m or m.get('accepted'):
            return None
        if 'offset_samples' not in m or 'samples_per_symbol_correction' not in m or 'anchor_trn_symbol' not in m:
            return None
        expected = self.carrier_derived_velocity(eq)
        fitted = m['samples_per_symbol_correction']
        if abs(expected) < self.FALLBACK_MIN_VELOCITY:
            return None
        if abs(fitted-expected) > max(self.GATE_RELATIVE*abs(expected), self.GATE_ABSOLUTE):
            return None
        return expected

    def apply(self, eq, k_trn):
        if self.applied or not self.model:
            return
        if not self.model.get('accepted'):
            velocity = self._gate_fallback(eq)
            if velocity is None:
                return
            eq.dt = self.model['offset_samples']+velocity*(eq.k-k_trn-self.model['anchor_trn_symbol'])
            eq.vel = velocity                       # the rejected fit's taps are not used
            eq._grid = None
            self.applied = True
            self.model.update(gate='fallback', gate_carrier_velocity=velocity, applied_at_symbol=eq.k,
                              applied_timing_offset=eq.dt, applied_velocity=eq.vel)
            return
        velocity = self.model['samples_per_symbol_correction']
        if os.environ.get('V34_CLOCK_FIT_GATE') == '1':
            # Opt-in plausibility gate (lab/V34_DIALUP_CLOCK_FIT_SWEEP.md): on dialup.world try1 the accepted fit came out
            # 27% below the carrier-derived value and the data phase never locked, while 15 other fits agreed within 10%.
            expected = self.carrier_derived_velocity(eq)
            if abs(velocity-expected) > max(self.GATE_RELATIVE*abs(expected), self.GATE_ABSOLUTE):
                self.model.update(gate='replaced', gate_fitted_velocity=velocity, gate_carrier_velocity=expected)
                velocity = expected
            else:
                self.model.update(gate='passed', gate_carrier_velocity=expected)
        override = os.environ.get('V34_CLOCK_VELOCITY_OVERRIDE')
        if override:
            # Test-only intervention (lab/V34_NEG_DRIFT_MINUS250_2026_10_12.md): apply this slope instead of the fitted one.
            velocity = float(override)
            self.model.update(velocity_override=velocity)
        eq.dt = self.model['offset_samples']+velocity*(
            eq.k-k_trn-self.model['anchor_trn_symbol'])
        eq.vel = velocity
        eq.h = self.taps.copy()
        eq._grid = None
        self.applied = True
        self.model.update(applied_at_symbol=eq.k, applied_timing_offset=eq.dt, applied_velocity=eq.vel)

    def report(self):
        return dict(model=self.model, applied=self.applied, retained_rows=len(self.rows),
                    peak_rows=self.peak_rows, maximum_rows=MAX_ROWS)
