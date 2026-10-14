"""Opt-in trajectory interpolation grid: accuracy against direct sampling while the timing NCO moves."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from v34_rx import SymbolEq, interp


def signal():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=60000)+1j*rng.normal(size=60000)
    taps = np.hanning(33)*np.sinc((np.arange(33)-16)*0.45)         # band-limited like the demodulated baseband
    return np.convolve(raw, taps, mode='same')


def equalizer(enabled):
    r = SimpleNamespace(_span=12, spl=7/3, lag=50, LEAD=8, delay=0, taps=np.ones(24, complex), _z=signal(), baud=3428.57, fc=1959.18)
    with patch.dict(os.environ, {'V34_TRAJECTORY_GRID': '1'} if enabled else {}, clear=False):
        if not enabled:
            os.environ.pop('V34_TRAJECTORY_GRID', None)
        return SymbolEq(r, 40)


def exact(eq, k):
    return interp(eq.R._z, eq.R.lag+(eq.R.LEAD+k+eq.R.delay)*eq.R.spl+eq.dt+eq._offs)


def run(eq, symbols, events=None):
    worst, builds = 0., 0
    original = eq._build_grid
    def counted(*a, **kw):
        nonlocal builds;builds += 1;return original(*a, **kw)
    eq._build_grid = counted
    for j in range(symbols):
        if events and j in events:events[j](eq)
        w = eq.window(eq.k);e = exact(eq, eq.k)
        worst = max(worst, float(np.max(abs(w-e))/np.sqrt(np.mean(abs(e)**2))))
        eq.dt += eq.vel;eq.k += 1
    return worst, builds


class TrajectoryGridTests(unittest.TestCase):
    def test_default_off_is_exact_and_unchanged(self):
        eq = equalizer(False);eq.vel = 3e-4
        worst, builds = run(eq, 300)
        self.assertLess(worst, 1e-12)
        self.assertFalse(eq._traj)

    def test_drift_velocity_matches_direct_sampling_with_few_builds(self):
        for vel in (3.1e-4, -3.1e-4, 8e-4):
            eq = equalizer(True);eq.vel = vel
            worst, builds = run(eq, 600)
            self.assertLess(worst, 2e-4, vel)
            self.assertLessEqual(builds, 600//SymbolEq.GRID+3, (vel, builds))

    def test_small_dt_jump_and_slip_stay_accurate_and_large_jump_falls_back_exactly(self):
        eq = equalizer(True);eq.vel = 3e-4
        worst, _ = run(eq, 400, {150: lambda e: setattr(e, 'dt', e.dt+.01),
                                 250: lambda e: (setattr(e, 'dt', e.dt+e.R.spl/2), setattr(e, '_g_shift', e._g_shift+1))})
        self.assertLess(worst, 5e-4)
        eq = equalizer(True);eq.vel = 3e-4
        worst, _ = run(eq, 300, {100: lambda e: setattr(e, 'dt', e.dt+.2)})
        self.assertLess(worst, 2e-4)

    def test_clock_capture_keeps_exact_windows(self):
        eq = equalizer(True);eq.vel = 3e-4;eq._clock_capture = True
        worst, _ = run(eq, 200)
        self.assertLess(worst, 1e-12)

    def test_zero_velocity_uses_the_original_cache_exactly(self):
        eq = equalizer(True)
        worst, builds = run(eq, 300)
        self.assertLess(worst, 1e-12)
        self.assertFalse(eq._g_traj)


if __name__ == '__main__':
    unittest.main()


class InterpPairTests(unittest.TestCase):
    def test_pair_matches_interp_and_its_numerical_derivative(self):
        from v34_rx import interp_pair
        z = signal()
        rng = np.random.default_rng(3)
        pos = np.r_[rng.uniform(100, 50000, 4000), np.arange(200, 260, 1.0), np.arange(300, 330, .5)]   # includes exact integers
        value, derivative = interp_pair(z, pos)
        np.testing.assert_allclose(value, interp(z, pos), rtol=0, atol=1e-11)
        h = 1e-5
        numeric = (interp(z, pos+h)-interp(z, pos-h))/(2*h)
        scale = np.sqrt(np.mean(abs(numeric)**2))
        self.assertLess(np.max(abs(derivative-numeric))/scale, 1e-4)


class GradientDerivativeTests(unittest.TestCase):
    def equalizer(self, vel=3.1e-4):
        eq = equalizer(True)
        eq.vel = vel
        eq.timing_mode = 'gradient'
        eq.h = (np.random.default_rng(5).normal(size=24)+1j*np.random.default_rng(6).normal(size=24))/10
        return eq

    def central_difference(self, eq):
        R = eq.R
        center = R.lag+(R.LEAD+eq.k+R.delay)*R.spl+eq.dt
        offs = (np.arange(eq.L)-eq.span)*R.spl/2
        return complex(((interp(R._z, center+offs+.05)-interp(R._z, center+offs-.05))/.1)@eq.h)

    def test_grid_derivative_matches_central_difference_along_a_drifting_run(self):
        eq = self.equalizer()
        errors, references, used = [], [], 0
        for j in range(500):
            eq.window(eq.k)
            fast = eq._gradient_derivative()
            used += int(eq._tj_k == eq.k and eq._d2grid is not None)
            ref = self.central_difference(eq)
            errors.append(abs(fast-ref));references.append(abs(ref))
            if j == 200:eq.dt += .006            # a small timing correction, as the gradient loop makes
            eq.dt += eq.vel;eq.k += 1
        self.assertGreater(used, 450)
        # The reference is a +-0.05 sample secant, itself about 7e-4 off the true derivative at this bandwidth.
        self.assertLess(max(errors)/np.sqrt(np.mean(np.square(references))), 3e-3)

    def test_falls_back_to_the_exact_derivative_when_the_window_was_not_from_the_grid(self):
        eq = equalizer(False)
        eq.timing_mode = 'gradient'
        eq.window(eq.k)
        self.assertEqual(eq._gradient_derivative(), self.central_difference(eq))

    def test_curvature_matches_numerical_second_derivative(self):
        from v34_rx import interp_pair
        z = signal();rng = np.random.default_rng(9)
        pos = np.r_[rng.uniform(100, 50000, 2000), np.arange(200, 230, 1.0)]
        _, _, curvature = interp_pair(z, pos, second=True)
        h = 1e-3
        numeric = (interp(z, pos+h)-2*interp(z, pos)+interp(z, pos-h))/h**2
        self.assertLess(np.max(abs(curvature-numeric))/np.sqrt(np.mean(abs(numeric)**2)), 2e-3)
