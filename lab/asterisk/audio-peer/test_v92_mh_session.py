import unittest

import v92_mh_session as mh
from v92_mh_session import Modem, run


def kinds(m):
    return [(e[1], e[2]) for e in m.events]


class ModemOnHoldTests(unittest.TestCase):
    def test_request_acknowledged_then_hold_then_resume(self):            # Figure 20
        a, b = Modem("A", after_ack="rt"), Modem("B", grant=True, timeout_code=3)
        r = run(a, b, "MHreq", resume_at_ms=2000)
        self.assertIn(("mhack", 3), kinds(a))                    # the timeout code crossed in the frame
        self.assertEqual(b.hold_limit_ms, 30000)
        self.assertEqual(b.phase1, "answer")                     # resumed by QC
        names = [e for e in b.events if e[1] == "on-hold"]
        self.assertEqual(len(names), 1)
        # RT (100 ms) took the responder on hold, ANSam followed within 80 ms of the RT detection.
        ack_t = next(t for t, k, _ in a.events if k == "mhack")
        hold_t = names[0][0]
        self.assertGreaterEqual(hold_t - ack_t, mh.RT_HOLD_MS)
        self.assertLessEqual(hold_t - ack_t, mh.RT_HOLD_MS + mh.FRAME_MS + 80 + 2 * 10)

    def test_silence_instead_of_tone_rt_needs_two_seconds(self):
        a, b = Modem("A", after_ack="silence"), Modem("B", timeout_code=1)
        run(a, b, "MHreq")
        t_ack = next(t for t, k, _ in a.events if k == "mhack")
        t_hold = next(t for t, k, _ in b.events if k == "on-hold")
        self.assertGreaterEqual(t_hold - t_ack, mh.SILENCE_HOLD_MS)

    def test_the_hold_ends_when_t1_expires(self):
        a, b = Modem("A"), Modem("B", timeout_code=1)             # 10 s
        run(a, b, "MHreq", max_ms=30000)
        self.assertTrue(b.disconnected)
        t_hold = next(t for t, k, _ in b.events if k == "on-hold")
        t_end = next(t for t, k, _ in b.events if k == "disconnect")
        self.assertEqual(t_end - t_hold, 10000)
        self.assertEqual(b.events[-1][2], "T1 expired")

    def test_request_denied_then_cleardown(self):                         # Figure 21
        a, b = Modem("A", after_nack="cleardown"), Modem("B", grant=False)
        run(a, b, "MHreq")
        self.assertIn(("mhnack", None), kinds(a))
        self.assertTrue(b.disconnected)
        self.assertIn(("disconnect", "cleardown after MHnack"), kinds(b))

    def test_request_denied_then_fast_reconnect(self):                    # Figure 22
        a, b = Modem("A", after_nack="fast_reconnect"), Modem("B", grant=False)
        run(a, b, "MHreq")
        self.assertEqual(b.phase1, "answer")
        self.assertEqual(a.phase1, "caller")
        # ANSam follows MHfrr after at most 80 ms of silence and A waits for 1 s of it.
        t_frr = next(t for t, k, v in b.events if k == "tx" and v == "silence" and t > 100)
        t_ans = next(t for t, k, v in b.events if k == "tx" and v == "ANSam")
        self.assertLessEqual(t_ans - t_frr, 80)

    def test_cleardown_request_carries_its_reason(self):                  # Figure 23
        a, b = Modem("A", reason="outgoing call"), Modem("B")
        run(a, b, "MHclrd")
        self.assertIn(("cleardown-reason", "outgoing call"), kinds(b))
        self.assertTrue(a.disconnected and b.disconnected)
        self.assertEqual(kinds(a)[-1], ("disconnect", "cleardown acknowledged"))

    def test_fast_reconnect_request(self):                                # Figure 24
        a, b = Modem("A"), Modem("B")
        run(a, b, "MHfrr")
        self.assertEqual((a.phase1, b.phase1), ("caller", "answer"))

    def test_no_response_ends_the_request_after_two_seconds_and_a_round_trip(self):
        a, b = Modem("A"), Modem("B", deaf=True)
        run(a, b, "MHreq", latency_ms=50)
        t = next(t for t, k, _ in a.events if k == "no-response")
        self.assertEqual(t, mh.RESPONSE_WAIT_MS + 100)
        self.assertTrue(a.disconnected)

    def test_a_garbled_frame_is_not_a_signal(self):
        a, b = Modem("A"), Modem("B")
        bits = mh.v92_mh.build_mh("MHreq")
        bits[20] ^= 1                                        # one CRC-covered bit flipped
        self.assertIsNone(mh.v92_mh.parse_mh(bits))
        b.hear(500, "MHreq", bits, 0)                        # heard for 500 ms, but it does not parse
        self.assertEqual(b.state, "data")


if __name__ == "__main__":
    unittest.main()
