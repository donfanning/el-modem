"""TRN end estimate: recovery when a burst of decision errors splits the run of agreements."""
import unittest
import numpy as np
from v34_rx import trn_end_from_mismatch


def split_trn(first=5740, burst=24, tail=436, after=200, rng=None):
    rng = rng or np.random.default_rng(1)
    agree = np.zeros(first, int)
    errors = np.ones(burst, int)
    second = np.zeros(tail, int)
    post = (rng.random(after) < .75).astype(int)           # after the TRN about 3 in 4 symbols mismatch
    return np.r_[agree, errors, second, post]


class RearmTests(unittest.TestCase):
    def test_longest_run_rule_is_pinned_to_the_burst_but_last_run_follows_the_trn(self):
        m = split_trn()
        self.assertEqual(trn_end_from_mismatch(m), 5740)                    # the failure: end placed at the burst
        self.assertEqual(trn_end_from_mismatch(m, last=True), 5740+24+436)  # the true end of the TRN

    def test_clean_trn_gives_the_same_answer_either_way(self):
        rng = np.random.default_rng(2)
        m = np.r_[np.zeros(6000, int), (rng.random(300) < .75).astype(int)]
        self.assertEqual(trn_end_from_mismatch(m), trn_end_from_mismatch(m, last=True))
        self.assertEqual(trn_end_from_mismatch(m, last=True), 6000)

    def test_no_long_run_returns_none_and_short_tail_is_ignored(self):
        self.assertIsNone(trn_end_from_mismatch(np.ones(500, int), last=True))
        m = np.r_[np.zeros(3000, int), np.ones(10, int), np.zeros(30, int), np.ones(40, int)]
        self.assertEqual(trn_end_from_mismatch(m, last=True), 3000)         # a 30-symbol fragment is below min_run

    def test_rearm_only_acts_after_the_wait_and_only_when_the_estimate_changes(self):
        from v34_rx import Phase4Receiver as Phase4Rx
        stub = type('S', (), {})()
        stub.REARM_AFTER = Phase4Rx.REARM_AFTER
        stub.mismatch = list(split_trn(after=0))
        stub.idx = list(range(len(stub.mismatch)))
        stub._end_est = 5740
        stub.idx = list(range(5740+100))                                     # only 100 symbols past the estimate
        Phase4Rx._rearm_end_estimate(stub)
        self.assertEqual(stub._end_est, 5740)
        stub.idx = list(range(len(stub.mismatch)))                           # 460 symbols past it
        Phase4Rx._rearm_end_estimate(stub)
        self.assertEqual(stub._end_est, 5740+24+436)


if __name__ == '__main__':
    unittest.main()
