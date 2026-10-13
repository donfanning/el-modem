
import os
import types
import unittest
from unittest import mock

import numpy as np

from v34_call import V34Caller


def caller(steady, wins, b2_win, t, l2_end):
    c = V34Caller.__new__(V34Caller)
    c._wins = wins
    c._steady = lambda m: steady(m)
    c.l2_start_t = 0
    c.l2_planned_end_t = l2_end
    c._b2_win = b2_win
    c.t = t
    return c


class L2GuardTests(unittest.TestCase):
    def test_off_by_default_accepts_stale_tone_a(self):
        c = caller(lambda m: True, 100, 20, 1000, 10000)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('V34_L2_GUARD', None)
            self.assertTrue(c._a_tone_seen())

    def test_guard_rejects_tone_a_that_never_paused(self):
        c = caller(lambda m: True, 100, 20, 1000, 10000)
        with mock.patch.dict(os.environ, {'V34_L2_GUARD': '1'}):
            self.assertFalse(c._a_tone_seen())

    def test_guard_accepts_tone_a_that_returned_after_a_pause(self):
        c = caller(lambda m: not (30 <= m < 90), 200, 20, 1000, 10000)     # 60 non-steady windows, steady again
        with mock.patch.dict(os.environ, {'V34_L2_GUARD': '1'}):
            self.assertTrue(c._a_tone_seen())

    def test_guard_lets_stale_tone_a_end_l2_once_the_planned_l2_is_over(self):
        c = caller(lambda m: True, 100, 20, 10000, 10000)
        with mock.patch.dict(os.environ, {'V34_L2_GUARD': '1'}):
            self.assertTrue(c._a_tone_seen())

    def test_guard_needs_a_substantial_pause(self):
        c = caller(lambda m: not (30 <= m < 35), 200, 20, 1000, 10000)     # 5 non-steady windows only
        with mock.patch.dict(os.environ, {'V34_L2_GUARD': '1'}):
            self.assertFalse(c._a_tone_seen())


if __name__ == '__main__':
    unittest.main()
