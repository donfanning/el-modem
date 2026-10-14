"""Regression for the timing-domain error that hid frame-generation stalls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('analyze_full_dce_stalls',
    Path(__file__).parent / 'tools' / 'analyze_full_dce_stalls.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

replay_spec = importlib.util.spec_from_file_location('full_dce_replay_stalls',
    Path(__file__).parent / 'tools' / 'full_dce_replay_stalls.py')
replay = importlib.util.module_from_spec(replay_spec)
replay_spec.loader.exec_module(replay)


class ReplayArmIsolationTests(unittest.TestCase):
    def test_baseline_overrides_inherited_production_warmup(self):
        parent = {'V34_PREIMPORT': '1', 'EL_MODEM_GC_PAUSE': '1',
                  'V34_TRN_CLOCK': '1'}
        env = replay.replay_environment(parent, preimport=False, gc_pause=False)
        self.assertEqual(env['V34_PREIMPORT'], '0')
        self.assertEqual(env['EL_MODEM_GC_PAUSE'], '0')
        self.assertEqual(env['V34_TRN_CLOCK'], '1')
        self.assertEqual(parent['V34_PREIMPORT'], '1')

    def test_each_arm_switch_is_independent_of_parent(self):
        for preimport in (False, True):
            for gc_pause in (False, True):
                with self.subTest(preimport=preimport, gc_pause=gc_pause):
                    env = replay.replay_environment(
                        {'V34_PREIMPORT': '0', 'EL_MODEM_GC_PAUSE': '0'},
                        preimport=preimport, gc_pause=gc_pause)
                    self.assertEqual(env['V34_PREIMPORT'], str(int(preimport)))
                    self.assertEqual(env['EL_MODEM_GC_PAUSE'], str(int(gc_pause)))


class CompletionDeadlineTests(unittest.TestCase):
    def test_on_time_start_does_not_hide_late_completion(self):
        row = dict(started=114.705, deadline=114.700, ended=114.748)
        # Sender arrived only 5 ms late, but spent 43 ms generating its frame;
        # the receiver observed a 68 ms gap including normal frame spacing.
        arrivals = [[114.680, 0], [114.748, 1]]
        r = audit.window_stats([row], arrivals, 100., 14.5, 15.5)
        self.assertAlmostEqual(r['maximum_start_lateness_ms'], 5.)
        self.assertAlmostEqual(r['maximum_generation_ms'], 43.)
        self.assertAlmostEqual(r['maximum_completion_lateness_ms'], 48.)
        self.assertAlmostEqual(r['maximum_arrival_gap_ms'], 68.)

    def test_other_stages_cannot_replace_data_start_maximum(self):
        rows = [dict(started=111., deadline=110.9, ended=111.02),
                dict(started=114.8, deadline=114.8, ended=114.801)]
        r = audit.window_stats(rows, [], 100., 14.5, 15.5)
        self.assertEqual(r['frames'], 1)
        self.assertAlmostEqual(r['maximum_completion_lateness_ms'], 1.)
        self.assertIsNone(audit.window_stats(rows, [], 100., 12., 13.))


if __name__ == '__main__':
    unittest.main()
