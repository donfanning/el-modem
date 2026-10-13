#!/usr/bin/env python3
"""Our V32Modem caller against our V32Answerer over a simulated line (v32_loop.py)."""
import unittest

import v8
from v32_loop import run

CALLER, ANSWERER = b"caller says hello\r\n", b"answerer says hello\r\n"


def ok(result):
    c, a, got_c, got_a, _ = result
    return c.connected and a.connected and got_a.endswith(CALLER) and got_c.endswith(ANSWERER)


class LoopTests(unittest.TestCase):
    def test_every_rate_connects_and_carries_data_both_ways(self):
        for rate in (14400, 9600, 4800):
            r = run(rate=rate, seconds=60, use_v8=False)
            self.assertTrue(ok(r), rate)
            self.assertEqual((r[0].bit_rate, r[1].bit_rate), (rate, rate))

    def test_the_answerer_picks_the_highest_common_rate(self):
        r = run(rate=None, seconds=60, use_v8=False)
        self.assertTrue(ok(r))
        self.assertEqual(r[1].bit_rate, 14400)

    def test_short_answerer_trn_is_detected_not_trained_past(self):
        # ACTS (303) 494-4774 sends a training sequence that stops matching our expected TRN at
        # ~2440 symbols, then keeps transmitting something else. We must notice the training end
        # and fall back to decision-directed decoding, not equalise against the wrong sequence.
        # The old detector could not: it started watching the error floor at symbol 2500, so for a
        # shorter TRN the floor was pinned at the already-corrupted value and it never fired.
        for trn_len in (2400, 2000, 1200, 700):
            r = run(rate=9600, seconds=60, use_v8=False, ans_trn=trn_len)
            self.assertTrue(ok(r), f"trn_len={trn_len}: {r[0].state}/{r[1].state}")
            # the first transmission's TRN end must be reported, and close to where it really is
            ends = [int(e.split("after ")[1].split(" ")[0])
                    for _, e in r[0].events if "TRN ended after" in e]
            self.assertTrue(ends, f"trn_len={trn_len}: training end never detected")
            self.assertLessEqual(abs(ends[0] - trn_len), 120,
                                 f"trn_len={trn_len}: reported end at symbol {ends[0]}")

    def test_long_path_delay_and_echo(self):
        self.assertTrue(ok(run(rate=9600, delay_ms=150, seconds=80, use_v8=False)))
        self.assertTrue(ok(run(rate=14400, delay_ms=40, echo_db=-20, seconds=80, use_v8=False)))

    def test_v42_lapm_between_the_two_engines(self):
        r = run(rate=9600, seconds=80, v42=True, use_v8=False)
        self.assertTrue(ok(r))
        self.assertEqual((r[0].v42_mode, r[1].v42_mode), ("lapm", "lapm"))

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 negotiation cannot run")
    def test_v8_negotiation_precedes_the_v32_handshake(self):
        r = run(rate=9600, seconds=80, use_v8=True)
        self.assertTrue(ok(r))
        self.assertIn("V.8: v8-call", " | ".join(e for _, e in r[0].events))


if __name__ == "__main__":
    unittest.main()
