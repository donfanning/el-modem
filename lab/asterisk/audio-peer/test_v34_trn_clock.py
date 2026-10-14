"""Bounded per-call opt-in lifecycle and timing handoff safety."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from v34_trn_clock import TrnClock, MAX_ROWS
from v34_receive_dsp import ReceiveDsp, configured


class TrnClockTests(unittest.TestCase):
    def test_default_is_off_and_clock_only_retains_no_tx_history(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured(SimpleNamespace()))
        dsp = ReceiveDsp(SimpleNamespace(), trn_clock=True)
        for _ in range(1000):
            dsp.commit_tx(np.ones(160))
        self.assertEqual(dsp.length, 0)
        self.assertEqual(len(dsp.source), 1024)

    def test_models_are_isolated_and_rows_bounded(self):
        left, right = TrnClock(), TrnClock()
        for i in range(MAX_ROWS+100):
            left.rows.append(dict(index=i))
        self.assertEqual(len(left.rows), MAX_ROWS)
        self.assertEqual(len(right.rows), 0)

    def test_rejected_handoff_preserves_state_and_accepted_handoff_is_once(self):
        clock = TrnClock()
        eq = SimpleNamespace(k=8500, dt=2., vel=.01, h=np.ones(24), _grid='old')
        clock.model = dict(accepted=False)
        clock.apply(eq, 16)
        self.assertEqual(eq.dt, 2.)
        self.assertEqual(eq._grid, 'old')
        clock.model = dict(accepted=True, offset_samples=.7,
                           samples_per_symbol_correction=.0003, anchor_trn_symbol=4500)
        clock.taps = np.full(24, 2j)
        clock.apply(eq, 16)
        self.assertAlmostEqual(eq.dt, .7+.0003*(8500-16-4500))
        self.assertEqual(eq.vel, .0003)
        self.assertIsNone(eq._grid)
        clock.taps[0] = 19
        self.assertEqual(eq.h[0], 2j)
        eq.dt = 7.
        clock.apply(eq, 16)
        self.assertEqual(eq.dt, 7.)

    def _gate_eq(self, freq):
        # baud 3428.57, fc 1959.18, 2.3333 samples per symbol: velocity = -freq * 0.6499
        return SimpleNamespace(k=8500, dt=2., vel=.01, h=np.ones(24), _grid='old', freq=freq,
                               R=SimpleNamespace(baud=3428.5714285714284, fc=1959.1836734693877, spl=8000/3428.5714285714284))

    def _accepted_clock(self, velocity):
        clock = TrnClock()
        clock.model = dict(accepted=True, offset_samples=.7, samples_per_symbol_correction=velocity, anchor_trn_symbol=4500)
        clock.taps = np.full(24, 2j)
        return clock

    def test_gate_is_off_by_default(self):
        import os
        from unittest import mock
        eq = self._gate_eq(-4.9e-4)
        clock = self._accepted_clock(2.33e-4)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('V34_CLOCK_FIT_GATE', None)
            clock.apply(eq, 16)
        self.assertEqual(eq.vel, 2.33e-4)
        self.assertNotIn('gate', clock.model)

    def test_gate_replaces_an_implausible_fit_velocity(self):
        import os
        from unittest import mock
        eq = self._gate_eq(-4.9e-4)                      # carrier-derived velocity 3.18e-4
        clock = self._accepted_clock(2.33e-4)            # the dialup.world try1 fit, 27% low
        with mock.patch.dict(os.environ, {'V34_CLOCK_FIT_GATE': '1'}):
            clock.apply(eq, 16)
        self.assertAlmostEqual(eq.vel, 3.18e-4, delta=.02e-4)
        self.assertEqual(clock.model['gate'], 'replaced')
        self.assertAlmostEqual(clock.model['gate_fitted_velocity'], 2.33e-4)
        self.assertAlmostEqual(eq.dt, .7+eq.vel*(8500-16-4500))

    def test_gate_keeps_a_plausible_fit_velocity(self):
        import os
        from unittest import mock
        for fitted in (3.0e-4, 3.3e-4, 2.9e-4):          # the 15 good dialup.world fits lie within about 10%
            eq = self._gate_eq(-4.9e-4)
            clock = self._accepted_clock(fitted)
            with mock.patch.dict(os.environ, {'V34_CLOCK_FIT_GATE': '1'}):
                clock.apply(eq, 16)
            self.assertEqual(eq.vel, fitted)
            self.assertEqual(clock.model['gate'], 'passed')

    def test_gate_uses_an_absolute_floor_when_the_carrier_offset_is_tiny(self):
        import os
        from unittest import mock
        eq = self._gate_eq(-1e-6)                        # Queens-like: ~1e-6 rad/symbol
        clock = self._accepted_clock(3e-5)               # differs by far more than 15% of 6.5e-7 but under the floor
        with mock.patch.dict(os.environ, {'V34_CLOCK_FIT_GATE': '1'}):
            clock.apply(eq, 16)
        self.assertEqual(eq.vel, 3e-5)

    def _rejected_clock(self, velocity=2.945e-4):
        clock = TrnClock()
        clock.model = dict(accepted=False, offset_samples=.533, samples_per_symbol_correction=velocity, anchor_trn_symbol=4703.5)
        clock.taps = np.full(24, 2j)
        return clock

    def test_rejected_fit_stays_bypassed_without_the_gate(self):
        import os
        from unittest import mock
        eq = self._gate_eq(-4.9e-4)
        clock = self._rejected_clock()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('V34_CLOCK_FIT_GATE', None)
            clock.apply(eq, 16)
        self.assertEqual(eq.dt, 2.)
        self.assertEqual(eq._grid, 'old')

    def test_gate_applies_a_rejected_fit_with_the_carrier_velocity_when_they_agree(self):
        import os
        from unittest import mock
        eq = self._gate_eq(-4.9e-4)                      # carrier-derived 3.18e-4; fit 2.945e-4 is 7% below
        clock = self._rejected_clock()
        with mock.patch.dict(os.environ, {'V34_CLOCK_FIT_GATE': '1'}):
            clock.apply(eq, 16)
        self.assertAlmostEqual(eq.vel, 3.18e-4, delta=.02e-4)
        self.assertAlmostEqual(eq.dt, .533+eq.vel*(8500-16-4703.5))
        self.assertEqual(clock.model['gate'], 'fallback')
        self.assertEqual(eq.h[0], 1)                     # taps of a rejected fit are not used
        self.assertIsNone(eq._grid)

    def test_gate_leaves_a_rejected_fit_alone_when_the_clock_error_is_small_or_disagrees(self):
        import os
        from unittest import mock
        for freq, fitted in ((-1e-6, 3e-7), (-4.9e-4, 1.0e-4)):    # Queens-like; fit far from the carrier value
            eq = self._gate_eq(freq)
            clock = self._rejected_clock(fitted)
            with mock.patch.dict(os.environ, {'V34_CLOCK_FIT_GATE': '1'}):
                clock.apply(eq, 16)
            self.assertEqual(eq.dt, 2.)
            self.assertNotIn('gate', clock.model)

    def test_missing_trn_rejects_and_releases_rows(self):
        clock = TrnClock(); clock.rows.append(dict(index=1))
        eq = SimpleNamespace(_clock_capture=True, _clock_grid_source=np.ones(100))
        clock.after_feed(SimpleNamespace(trn_end=6000, eq=eq, role='answer', c16=True))
        self.assertFalse(clock.model['accepted'])
        self.assertEqual(len(clock.rows), 0)
        self.assertFalse(eq._clock_capture)
        self.assertFalse(hasattr(eq, '_clock_grid_source'))

    def test_old_domain_predictor_cannot_be_combined_with_clock(self):
        with self.assertRaisesRegex(ValueError, 'ordinary detection'):
            ReceiveDsp(SimpleNamespace(), trn_clock=True, predictive=True)


if __name__ == '__main__':
    unittest.main()


class VelocityOverrideTests(unittest.TestCase):
    """The test-only override replaces only the slope; the fitted offset and anchor are kept."""

    def _apply(self, env):
        clock = TrnClock()
        clock.model = dict(accepted=True, offset_samples=0.25, samples_per_symbol_correction=5.348e-4,
                           anchor_trn_symbol=4661.5)
        clock.taps = np.zeros(4)
        eq = SimpleNamespace(k=6973, dt=0.0, vel=0.0, h=None, _grid=object())
        with patch.dict(os.environ, env, clear=False):
            if not env:
                os.environ.pop('V34_CLOCK_VELOCITY_OVERRIDE', None)
            clock.apply(eq, 0)
        return clock, eq

    def test_unset_keeps_the_fitted_slope(self):
        clock, eq = self._apply({})
        self.assertEqual(eq.vel, 5.348e-4)
        self.assertNotIn('velocity_override', clock.model)

    def test_override_replaces_only_the_slope(self):
        clock, eq = self._apply({'V34_CLOCK_VELOCITY_OVERRIDE': '5.833e-4'})
        self.assertEqual(eq.vel, 5.833e-4)
        self.assertAlmostEqual(eq.dt, 0.25 + 5.833e-4 * (6973 - 4661.5))
        self.assertEqual(clock.model['velocity_override'], 5.833e-4)
