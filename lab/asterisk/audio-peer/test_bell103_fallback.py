import os
import unittest

import numpy as np

import v22
from autochain import V8Chain
from bell103_engine import Bell103Engine


def pcm(samples):
    return np.asarray(samples, dtype="<i2").tobytes()


def duplex(caller, answerer, seconds, hook=None):
    got_c, got_a = bytearray(), bytearray()
    for blk in range(int(seconds * 50)):
        c, a = caller.tx_samples(160), answerer.tx_samples(160)
        got_a += answerer.feed_pcm(pcm(c)) or b""
        got_c += caller.feed_pcm(pcm(a)) or b""
        if hook:
            hook(blk)
    return bytes(got_c), bytes(got_a)


class Bell103EngineTests(unittest.TestCase):
    def test_two_engines_connect_and_exchange_bytes(self):
        caller, answerer = Bell103Engine(True), Bell103Engine(False)
        sent = {"done": False}

        def hook(blk):
            if blk == 100 and not sent["done"]:
                caller.feed_bytes(b"hello 300 from the caller\r\n")
                answerer.feed_bytes(b"hello 300 from the answerer\r\n")
                sent["done"] = True
        got_c, got_a = duplex(caller, answerer, 8, hook)
        self.assertTrue(caller.connected and answerer.connected)
        self.assertEqual(got_a, b"hello 300 from the caller\r\n")
        self.assertEqual(got_c, b"hello 300 from the answerer\r\n")
        self.assertEqual(caller.bit_rate, 300)

    def test_the_v25_answer_tone_is_not_a_carrier(self):
        caller = Bell103Engine(True)
        t = np.arange(8000 * 2)
        caller.feed_pcm(pcm(np.round(3000 * np.sin(2 * np.pi * 2100 * t / 8000))))
        self.assertFalse(caller.connected)

    def test_silence_is_not_a_carrier(self):
        caller = Bell103Engine(True)
        caller.feed_pcm(pcm(np.zeros(8000)))
        self.assertFalse(caller.connected)


class ChainFallbackTests(unittest.TestCase):
    def run_chain(self, **kw):
        chain = V8Chain(True, **kw)
        answerer = Bell103Engine(False)                      # a 300 bps-only answerer: no V.8, no V.22
        sent = {"done": False}

        def hook(blk):
            if chain.kind == "b103" and chain.connected and not sent["done"]:
                chain.feed_bytes(b"ping\r\n")
                answerer.feed_bytes(b"pong\r\n")
                sent["done"] = True
        got_c, got_a = duplex(chain, answerer, 40, hook)
        return chain, got_c, got_a

    def test_opt_in_falls_back_from_a_failed_v22_attempt_to_300(self):
        chain, got_c, got_a = self.run_chain(b103_fallback=True, v22_attempt_seconds=4.0)
        texts = [str(e) for e in chain.events]
        self.assertTrue(any("falling back to V.22/V.22bis" in t for t in texts), texts)
        self.assertTrue(any("falling back to Bell 103" in t for t in texts), texts)
        self.assertEqual(chain.kind, "b103")
        self.assertTrue(chain.connected)
        self.assertEqual(chain.bit_rate, 300)
        # the 300 bps answerer also decodes some junk from our earlier V.22 attempt, so compare the tail
        self.assertTrue(got_a.endswith(b"ping\r\n"), got_a)
        self.assertTrue(got_c.endswith(b"pong\r\n"), got_c)

    def test_default_never_leaves_v22(self):
        chain, _, _ = self.run_chain()
        self.assertEqual(chain.kind, "v22")
        self.assertFalse(any("Bell 103" in str(e) for e in chain.events))

    def test_a_v22_answerer_is_not_pushed_to_300(self):
        caller = V8Chain(True, b103_fallback=True, v22_attempt_seconds=4.0)
        answerer = v22.V22Modem(False, 2400)
        duplex(caller, answerer, 30)
        self.assertEqual(caller.kind, "v22")
        self.assertTrue(caller.connected)

    def test_the_v22_test_hook_mutes_the_transmitter(self):
        os.environ["V22_TEST_MUTE_TX"] = "1"
        self.addCleanup(os.environ.pop, "V22_TEST_MUTE_TX", None)
        self.assertEqual(set(v22.V22Modem(True, 1200).tx_samples(800)), {0})


if __name__ == "__main__":
    unittest.main()
