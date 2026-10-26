import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
from v34_realtime import RealtimeStats, _Samples
from v34_rx import _SymbolEqStepProfile


class RealtimeStatsTests(unittest.TestCase):
    def test_symbol_eq_profile_reports_bounded_stage_samples(self):
        profile = _SymbolEqStepProfile(capacity=2)
        profile.add((1000, 2000, 9000, 4000, 5000, 6000, 7000, 8000, 33000))
        profile.add((2000, 3000, 11000, 5000, 6000, 7000, 8000, 9000, 40000))
        profile.add((9000, 9000, 18000, 9000, 9000, 9000, 9000, 9000, 63000))
        report = profile.report()
        self.assertEqual(report['sample_count'], 3)
        self.assertEqual(report['sample_every_symbols'], 256)
        self.assertEqual(report['stages']['window']['retained_samples'], 2)
        self.assertEqual(report['stages']['window']['mean_us'], 5.5)
        self.assertEqual(report['stages']['projection_dot']['mean_us'], 7.0)
        self.assertEqual(report['stages']['derotation']['mean_us'], 7.5)
        self.assertEqual(report['stages']['projection']['mean_us'], 14.5)
        self.assertEqual(report['stages']['total']['max_us'], 63.0)

    def test_fixed_ring_keeps_recent_percentiles_and_lifetime_totals(self):
        samples = _Samples(3)
        for value in (1, 2, 3, 100):
            samples.add(value)
        result = samples.report()
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 26.5)
        self.assertEqual(result['maximum'], 100)
        # Percentiles cover the last three observations, matching the old deque.
        self.assertEqual(result['p99'], 100)
        self.assertEqual(result['p995'], 100)

    def test_report_preserves_realtime_gate_schema(self):
        stats = RealtimeStats(thread_timing=False)
        stats.add('tx_seconds', .002, data=True)
        stats.add('tx_seconds', .004, data=True)
        stats.add('rx_seconds', .005, data=True)
        report = stats.report()
        self.assertEqual(report['measurements']['data_tx_seconds']['count'], 2)
        self.assertEqual(report['measurements']['data_tx_seconds']['mean'], .003)
        self.assertEqual(report['combined_data_processing_seconds']['mean'], .008)
        self.assertIn('p99_upper_budget', report['combined_data_processing_seconds'])
        self.assertFalse(report['thread_cpu_timing_enabled'])

    def test_thread_cpu_timing_can_be_enabled_from_environment(self):
        with patch.dict(os.environ, {'EL_MODEM_THREAD_TIMING': '1'}):
            stats = RealtimeStats()
        self.assertTrue(stats.thread_timing)
        stats.add('rx_thread_cpu_seconds', .001, data=True)
        row = stats.report()['measurements']['data_rx_thread_cpu_seconds']
        self.assertEqual(row['count'], 1)
        self.assertEqual(row['mean'], .001)

    def test_thread_cpu_timing_rejects_unsupported_runtime(self):
        with patch('v34_realtime.time.thread_time', None):
            with self.assertRaises(RuntimeError):
                RealtimeStats(thread_timing=True)


if __name__ == '__main__':
    unittest.main()
