import unittest

import v92_short_phase2 as sp2


class ShortPhase2Tests(unittest.TestCase):
    def test_the_round_trip_estimate_is_the_two_one_way_delays(self):
        for down, up in ((0, 0), (10, 10), (40, 25), (120, 80)):
            with self.subTest(down=down, up=up):
                r = sp2.run(down, up)
                self.assertTrue(r["negotiated"]["short_phase2"])
                self.assertEqual(r["rtd_ms"], down + up)

    def test_the_turnaround_tolerance_shows_up_one_for_one(self):
        self.assertEqual(sp2.run(30, 30, turnaround_error_ms=1)["rtd_ms"], 61)
        self.assertEqual(sp2.run(30, 30, turnaround_error_ms=-1)["rtd_ms"], 59)

    def test_the_order_of_events_follows_figure_9(self):
        r = sp2.run(20, 20)
        names = [e[2] for e in r["events"]]
        order = ["tone B reversal", "tone B reversal detected", "tone A reversal", "INFO1a"]
        self.assertEqual([n for n in names if n in order], order)
        self.assertLess(names.index("tone A reversal"), names.index("tone A reversal detected"))
        self.assertLess(max(names.index("INFO0d"), names.index("INFO0a")), names.index("tone B reversal"))
        t = {e[2]: e[0] for e in r["events"]}
        self.assertGreaterEqual(t["tone B reversal"] - t["tone B"], sp2.TONE_B_MIN_MS)    # B on the line >= 50 ms
        self.assertEqual(t["silence"] - t["tone B reversal"], 10)                          # B kept 10 ms
        self.assertEqual(t["tone A reversal"] - t["tone B reversal detected"], 40)         # the 40 ms turnaround
        self.assertEqual(t["INFO1a"] - t["tone A reversal"], 10)                           # A kept 10 ms
        self.assertGreater(t["INFO1a"], t["tone A reversal"])

    def test_the_digital_modem_waits_for_the_slower_of_tone_b_50_ms_and_tone_a(self):
        near = sp2.run(5, 5)
        far = sp2.run(5, 300)
        t_near = {e[2]: e[0] for e in near["events"]}
        t_far = {e[2]: e[0] for e in far["events"]}
        self.assertEqual(t_near["tone B reversal"] - t_near["tone B"], 50)
        self.assertGreater(t_far["tone B reversal"] - t_far["tone B"], 50)

    def test_short_phase_2_needs_both_modems_v92_and_both_asking(self):
        for kw in ({"short_digital": False}, {"short_analogue": False},
                   {"v92_digital": False}, {"v92_analogue": False}):
            with self.subTest(kw=kw):
                r = sp2.run(10, 10, **kw)
                self.assertFalse(r["negotiated"]["short_phase2"])
                self.assertNotIn("rtd_ms", r)
                self.assertTrue(r["reason"])


if __name__ == "__main__":
    unittest.main()
