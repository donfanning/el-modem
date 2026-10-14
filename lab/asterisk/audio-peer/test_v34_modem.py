"""V.8 plus DCE-surface V.34 caller/answerer loopback."""

from __future__ import annotations

import struct
import unittest

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None


@unittest.skipIf(numpy is None, "numpy not installed")
class V34ModemPairTests(unittest.TestCase):
    def test_dce_wrappers_negotiate_and_exchange_async_data_both_ways(self) -> None:
        from v34_answer_modem import V34AnswerModem
        from v34_modem import V34Modem

        caller = V34Modem(data_rate=9600)
        answerer = V34AnswerModem(data_rate=9600)
        caller_payload = b"caller through DCE wrapper\r\n"
        answer_payload = b"answerer through DCE wrapper\r\n"
        caller.feed_bytes(caller_payload)
        answerer.feed_bytes(answer_payload)
        caller_rx = bytearray()
        answerer_rx = bytearray()

        for _ in range(50 * 50):  # at most 50 seconds of 20 ms audio blocks
            caller_tx = caller.tx_samples(160)
            answerer_tx = answerer.tx_samples(160)
            answerer_rx.extend(answerer.feed_pcm(struct.pack("<160h", *caller_tx)))
            caller_rx.extend(caller.feed_pcm(struct.pack("<160h", *answerer_tx)))
            if (caller.connected and answerer.connected and caller_payload in answerer_rx
                    and answer_payload in caller_rx):
                break

        self.assertTrue(caller.connected, caller.events[-5:])
        self.assertTrue(answerer.connected, answerer.events[-5:])
        self.assertEqual(caller.bit_rate, 9600)
        self.assertEqual(answerer.bit_rate, 9600)
        self.assertIn(caller_payload, answerer_rx)
        self.assertIn(answer_payload, caller_rx)


if __name__ == "__main__":
    unittest.main()
