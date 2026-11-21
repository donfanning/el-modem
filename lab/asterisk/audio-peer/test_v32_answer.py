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
        # NIST ACTS sends a training sequence that stops matching our expected TRN at
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

    def test_voip_round_trip_of_1_3_s(self):
        # A Patton dial-in pool over the relay measured a 1.3 s round trip (2026-11-21): the answerer's S comes a
        # round trip after the back-reversal, and our S must last that long too.
        for delay in (350, 650):
            self.assertTrue(ok(run(rate=9600, delay_ms=delay, seconds=80, use_v8=False)), delay)

    def test_answerer_training_delayed_by_an_echo_canceller_sequence(self):
        # V.32 5.4 Note 3: the answerer may precede S/S-bar/TRN with up to 8192T of its own echo-canceller
        # training; a Patton dial-in pool did (2026-11-21) and the caller's +-0.5 s TRN window missed it.
        # Silence of the same length stands in for that sequence here.
        from unittest import mock
        import v32_answer
        original = v32_answer.V32Answerer._begin_training1

        def late(self):
            original(self)
            self.tx.start_training_now(gap_symbols=16 + 3600)        # 1.5 s later than usual

        with mock.patch.object(v32_answer.V32Answerer, "_begin_training1", late):
            self.assertTrue(ok(run(rate=9600, delay_ms=150, seconds=80, use_v8=False)))

        def longest(self):                                           # a full 8192T sequence, then a 1 s S
            original(self)
            self.tx.s_length = 2400
            self.tx.start_training_now(gap_symbols=16 + 8192)

        with mock.patch.object(v32_answer.V32Answerer, "_begin_training1", longest):
            self.assertTrue(ok(run(rate=9600, delay_ms=150, seconds=90, use_v8=False)))

    def test_caller_transmit_clock_ahead_of_its_receive_clock(self):
        # --line-mode v32 on a live call: the sender pulled samples for 0.5 s before the first RTP packet,
        # so NT (transmit counter minus receive counter) came out 670 ms for a ~1.2 s round trip and the
        # Patton never trained on our short S (2026-11-21). Times must stay on the receive clock.
        from unittest import mock
        import v32_stream
        original = v32_stream.V32Modem.__init__

        def ahead(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if type(self) is v32_stream.V32Modem:                     # the caller only
                self._txn += 4000

        with mock.patch.object(v32_stream.V32Modem, "__init__", ahead):
            c, a, got_c, got_a, _ = r = run(rate=9600, delay_ms=500, seconds=90, use_v8=True)
        self.assertTrue(ok(r), [e for _, e in a.events if "FAIL" in e])
        nt = [int(e.split("NT = ")[1].split(" ")[0]) for _, e in c.events if "round-trip estimate" in e][0]
        self.assertAlmostEqual(nt / 2400, 1.05, delta=0.1)

    def test_aa_after_the_answer_tone_option(self):
        # EL_MODEM_V32_AA_AFTER_ANS=1: AA as soon as >=1 s of answer tone ends (V.32 5.4.1), not after AC -- unless
        # the answerer follows the tone with JM (dialup.world 620, 2026-11-21): then V.8 completes first.
        # "during": AA once the tone has lasted 1 s, as slmodemd does, with or without a V.8 answerer.
        from unittest import mock
        import os
        cases = (("1", False, "answer tone ended", "heard AC"), ("1", True, "JM after the answer tone", None),
                 ("during", False, "answer tone heard for 1 s", "heard AC"),
                 ("during", True, "answer tone heard for 1 s", "heard AC"))
        for mode, use_v8, expected, absent in cases:
            with mock.patch.dict(os.environ, {"EL_MODEM_V32_AA_AFTER_ANS": mode}):
                r = run(rate=14400, delay_ms=500, seconds=90, use_v8=use_v8)
            self.assertTrue(ok(r), (mode, use_v8))
            events = " | ".join(e for _, e in r[0].events)
            self.assertIn(expected, events, (mode, use_v8))
            if absent:
                self.assertNotIn(absent, events, (mode, use_v8))

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
