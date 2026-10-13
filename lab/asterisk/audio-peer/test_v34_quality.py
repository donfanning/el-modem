"""Data quality: independent waveform cost oracle, backend parity and holds."""
import random
import unittest
from types import SimpleNamespace

import numpy as np
import v34_data as data
from v34_quality import DataQuality, observe
from v34_viterbi import TrellisDetector
from v34_trellis_native import njit


class QualityTests(unittest.TestCase):
    def signal(self, backend, sigma=.05, frames=16):
        tx = data.DataTx(24000, 3429, True)
        rng = random.Random(3519)
        symbols = []
        for _ in range(frames):
            symbols += tx.data_frame([rng.randrange(2) for _ in range(tx.bits_per_data_frame)])
        detector = TrellisDetector(4*tx.M*2**tx.q, tx.J, tx.P, backend=backend)
        profile = SimpleNamespace(rate=24000, nonlinear=False)
        monitor = DataQuality(detector, profile, 3429, min_symbols=64, history=2)
        noise_rng = np.random.default_rng(42)
        noise = noise_rng.normal(0, sigma, len(symbols)) + 1j*noise_rng.normal(0, sigma, len(symbols))
        ordinary = TrellisDetector(4*tx.M*2**tx.q, tx.J, tx.P, backend=backend)
        for i in range(0, len(symbols), 2):
            pair = np.array(symbols[i:i+2]) + noise[i:i+2]
            detector.feed(*pair)
            ordinary.feed(*pair)
            monitor.sample()
        detector.flush(); ordinary.flush()
        self.assertEqual(detector.decided, ordinary.decided)
        return detector, monitor, noise, symbols

    def test_cost_matches_true_noise_energy_when_decode_is_exact(self):
        detector, monitor, noise, symbols = self.signal('python')
        out = [point for pair in detector.decided for point in pair]
        self.assertEqual(out, symbols)
        # This independent known-transmit oracle applies only to this clean
        # experiment, not to the live decision-directed report.
        self.assertAlmostEqual(monitor.total_cost, float(np.vdot(noise, noise).real), places=9)
        self.assertEqual(monitor.symbols, len(symbols))
        self.assertEqual(len(monitor.report()['recent']), 2)
        self.assertEqual(monitor.window_count, len(symbols)//64)

    @unittest.skipIf(njit is None, 'optional Numba backend unavailable')
    def test_native_and_python_costs_agree_in_noisy_data(self):
        ref, a, _, _ = self.signal('python', .8)
        native, b, _, _ = self.signal('numba', .8)
        self.assertEqual(native.decided, ref.decided)
        self.assertAlmostEqual(a.total_cost, b.total_cost, places=8)
        self.assertEqual(a.window_count, b.window_count)

    def test_partial_hold_is_discarded_even_after_power_returns(self):
        detector = SimpleNamespace(m4=40, metric_offset=12., h=[], pts=range(16))
        monitor = DataQuality(detector, SimpleNamespace(rate=9600, nonlinear=False), 3200,
                              min_symbols=8, held_symbols=15)
        state = {'quality': monitor}
        guard = SimpleNamespace(holding=False, held_symbols=17)
        eq = SimpleNamespace(_input_guard=guard)
        detector.m4 += 2; detector.metric_offset += 8
        observe(None, state, eq)
        self.assertEqual(monitor.discarded_symbols, 4)
        detector.m4 += 4; detector.metric_offset += 16
        observe(None, state, eq)
        self.assertEqual(monitor.report()['path_cost_per_symbol'], 2.)
        self.assertEqual(monitor.symbols, 8)

    def test_nonfinite_cost_and_detector_reset_do_not_become_clean_quality(self):
        detector = SimpleNamespace(m4=40, metric_offset=12., h=[], pts=range(16))
        monitor = DataQuality(detector, SimpleNamespace(rate=9600, nonlinear=False), 3200)
        detector.metric_offset = float('nan'); detector.m4 += 1
        monitor.sample()
        detector.metric_offset = 0.; detector.m4 = 0
        monitor.sample()
        self.assertEqual(monitor.invalid_batches, 2)
        self.assertIsNone(monitor.report()['path_cost_per_symbol'])


if __name__ == '__main__':
    unittest.main()
