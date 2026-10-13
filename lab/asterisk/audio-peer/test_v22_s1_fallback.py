import os
import unittest

import numpy as np

import v22


def duplex(caller, answerer, seconds, hook=None):
    for blk in range(int(seconds * 50)):
        c = caller.tx_samples(160)
        a = answerer.tx_samples(160)
        answerer.feed_pcm(np.asarray(c, dtype="<i2").tobytes())
        caller.feed_pcm(np.asarray(a, dtype="<i2").tobytes())
        if hook:
            hook(blk)


class S1FallbackTests(unittest.TestCase):
    def pair(self):
        caller, answerer = v22.V22Modem(True, 2400), v22.V22Modem(False, 2400)
        duplex(caller, answerer, 4)
        self.assertTrue(caller.connected and answerer.connected)
        self.assertEqual(caller.bit_rate, 2400)
        return caller, answerer

    def restart_handshake(self, answerer):
        """What NIST ACTS does: 100 ms of S1 (alternating 00/11), then scrambled ones again."""
        answerer._set_tx("s1")
        answerer._at(100, lambda: answerer._set_tx("scr1", 2400))

    def test_a_normal_2400_link_does_not_fall_back(self):
        caller, answerer = self.pair()
        duplex(caller, answerer, 6)
        self.assertEqual(caller.bit_rate, 2400)
        self.assertNotIn("RATE_FALLBACK_1200", [e[0] for e in caller.events])

    def test_repeated_s1_after_the_lock_drops_the_caller_to_1200(self):
        caller, answerer = self.pair()
        os.environ["V22_S1_FALLBACK"] = "inplace"
        self.addCleanup(os.environ.pop, "V22_S1_FALLBACK", None)
        self.restart_handshake(answerer)
        duplex(caller, answerer, 2)
        self.assertIn("RATE_FALLBACK_1200", [e[0] for e in caller.events])
        self.assertEqual(caller.bit_rate, 1200)
        self.assertIsNone(caller._qam_from)                  # receive decisions are 4-phase again
        self.assertEqual((caller._tx_mode, caller._tx_rate_now), ("data", 1200))   # 765 ms of scrambled ones, then data

    def test_default_is_a_cold_restart_that_completes_at_1200_with_a_fresh_answerer(self):
        caller, answerer = self.pair()
        self.restart_handshake(answerer)
        duplex(caller, answerer, 1)                          # S1 seen, the caller goes quiet
        self.assertIn("RATE_RESTART_1200", [e[0] for e in caller.events])
        self.assertEqual((caller._state, caller._tx_mode, caller.bit_rate, caller.target_rate), ("wait", "silence", 1200, 1200))
        self.assertTrue(caller.connected)                     # the host sees no carrier loss
        fresh = v22.V22Modem(False, 1200)                     # the answerer starts over (USB1 again)
        got_a, got_c = bytearray(), bytearray()
        for blk in range(int(12 * 50)):
            c, a = caller.tx_samples(160), fresh.tx_samples(160)
            got_a += fresh.feed_pcm(np.asarray(c, dtype="<i2").tobytes())
            got_c += caller.feed_pcm(np.asarray(a, dtype="<i2").tobytes())
            if blk == 400:
                caller.feed_bytes(b"hello from the caller\r\n")
                fresh.feed_bytes(b"hello from the answerer\r\n")
        self.assertEqual(caller.bit_rate, 1200)
        self.assertGreaterEqual([e[0] for e in caller.events].count("TRAINING_SUCCEEDED"), 2)
        self.assertEqual(bytes(got_a)[-23:], b"hello from the caller\r\n")
        self.assertEqual(bytes(got_c)[-25:], b"hello from the answerer\r\n")

    def test_the_fallback_can_be_disabled(self):
        caller, answerer = self.pair()
        old = os.environ.get("V22_S1_FALLBACK")
        os.environ["V22_S1_FALLBACK"] = "0"
        try:
            self.restart_handshake(answerer)
            duplex(caller, answerer, 2)
        finally:
            os.environ.pop("V22_S1_FALLBACK", None) if old is None else os.environ.__setitem__("V22_S1_FALLBACK", old)
        self.assertEqual(caller.bit_rate, 2400)
        self.assertNotIn("RATE_FALLBACK_1200", [e[0] for e in caller.events])

    def test_the_answerer_never_falls_back(self):
        caller, answerer = self.pair()
        caller._set_tx("s1")
        caller._at(100, lambda: caller._set_tx("scr1", 2400))
        duplex(caller, answerer, 2)
        self.assertNotIn("RATE_FALLBACK_1200", [e[0] for e in answerer.events])


class BisSettleHoldTests(unittest.TestCase):
    """Opt-in V22_BIS_SETTLE_S: CONNECT is not allowed before the link has been stable for that long; a handshake restart by
    the answerer inside the hold ends the attempt instead of leaving a stale CONNECT 2400."""

    def setUp(self):
        os.environ["V22_BIS_SETTLE_S"] = "3.5"
        self.addCleanup(os.environ.pop, "V22_BIS_SETTLE_S", None)

    def locked_pair(self):
        caller, answerer = v22.V22Modem(True, 2400), v22.V22Modem(False, 2400)
        duplex(caller, answerer, 2.5)
        self.assertTrue(caller.connected)
        self.assertEqual(caller.bit_rate, 2400)
        return caller, answerer

    def test_settles_only_after_the_hold(self):
        caller, answerer = self.locked_pair()
        self.assertFalse(caller.is_settled())
        duplex(caller, answerer, 3.0)
        self.assertTrue(caller.is_settled())

    def test_an_s1_restart_inside_the_hold_ends_the_attempt_before_connect(self):
        caller, answerer = self.locked_pair()
        self.assertFalse(caller.is_settled())
        answerer._set_tx("s1")
        answerer._at(100, lambda: answerer._set_tx("scr1", 2400))
        duplex(caller, answerer, 1.5)
        self.assertEqual(caller.state, "failed")
        self.assertIn("BIS_REJECTED", [e[0] for e in caller.events])
        self.assertNotIn("RATE_RESTART_1200", [e[0] for e in caller.events])
        self.assertFalse(caller.is_settled())
        answerer._set_tx("s1")                                      # a second restart must not start a fallback either
        answerer._at(100, lambda: answerer._set_tx("scr1", 2400))
        duplex(caller, answerer, 1.0)
        self.assertNotIn("RATE_RESTART_1200", [e[0] for e in caller.events])

    def test_an_s1_restart_after_the_hold_still_restarts_at_1200(self):
        caller, answerer = self.locked_pair()
        duplex(caller, answerer, 3.0)
        self.assertTrue(caller.is_settled())
        answerer._set_tx("s1")
        answerer._at(100, lambda: answerer._set_tx("scr1", 2400))
        duplex(caller, answerer, 1.5)
        self.assertEqual(caller.state, "")
        self.assertIn("RATE_RESTART_1200", [e[0] for e in caller.events])

    def test_the_chain_forwards_the_failed_state_to_the_dce(self):
        from autochain import V8Chain
        chain = V8Chain(True)
        self.assertEqual(getattr(chain, "state", ""), "")            # no engine yet
        chain._choose("v22")
        self.assertEqual(chain.state, "")
        chain.engine._hold_failed = True
        self.assertEqual(chain.state, "failed")                       # what OnlineAudioSession.failed reads

    def test_without_the_option_nothing_changes(self):
        os.environ.pop("V22_BIS_SETTLE_S")
        caller, answerer = self.locked_pair()
        self.assertTrue(caller.is_settled())                    # settles ~0.1 s after the lock, as before
        self.assertEqual(caller.state, "")


if __name__ == "__main__":
    unittest.main()
