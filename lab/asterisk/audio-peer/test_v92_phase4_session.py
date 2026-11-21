import unittest

import v92_phase4_session as s


class Phase4SessionTests(unittest.TestCase):
    def check(self, law, latency):
        r = s.run_session(law, latency=latency)
        self.assertTrue(r["connected"], r.get("model"))
        traces = r["model"]["traces"]
        self.assertEqual(r["up"]["frames"], s.expected_frames(traces["analogue"], "up"))
        self.assertEqual(r["down"]["frames"], s.expected_frames(traces["digital"], "down"))
        self.assertEqual(r["up"]["e_zero_bits"] >= r["up"]["e_expected"], True)
        self.assertEqual(r["down"]["e_zero_bits"] >= r["down"]["e_expected"], True)
        self.assertTrue(r["up"]["b1_all_ones"] and r["down"]["b1_all_ones"])
        return r

    def test_the_frames_acks_and_tail_survive_both_laws_and_latencies(self):
        for law in ("pcmu", "pcma"):
            for latency in (0, 1, 3, 6):
                with self.subTest(law=law, latency=latency):
                    self.check(law, latency)

    def test_the_digital_cp_carries_the_acknowledge_and_nothing_before_it_does(self):
        r = self.check("pcmu", 2)
        down = r["down"]["frames"]
        first_ack = next(i for i, (_, ack) in enumerate(down) if ack)
        self.assertEqual(down[first_ack][0], "cpd")
        self.assertTrue(all(not ack for _, ack in down[:first_ack]))
        self.assertTrue(all(ack for _, ack in down[first_ack:]))
        # The analogue modem acknowledges only in its E (it has seen the CP last).
        self.assertTrue(all(not ack for _, ack in r["up"]["frames"]))

    def test_a_lost_cp_is_repeated_and_both_streams_read_back_with_the_repeat(self):
        drop = lambda who, slot, m: who == "digital" and m.kind == "CP" and slot < 14
        r = s.run_session("pcma", latency=2, drop=drop)
        self.assertTrue(r["connected"])
        traces = r["model"]["traces"]
        self.assertGreaterEqual(sum(m.kind == "CP" for m in traces["digital"]), 2)
        self.assertEqual(r["down"]["frames"], s.expected_frames(traces["digital"], "down"))
        self.assertEqual(r["up"]["frames"], s.expected_frames(traces["analogue"], "up"))


if __name__ == "__main__":
    unittest.main()
