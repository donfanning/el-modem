"""V.8 plus DCE-surface V.34 caller/answerer loopback."""

from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None


@unittest.skipIf(numpy is None, "numpy not installed")
class V34ModemPairTests(unittest.TestCase):
    def test_diagnostic_answer_tx_cap_preserves_receive_offer(self) -> None:
        from v34_answer_modem import V34AnswerModem

        with patch.dict('os.environ', {'V34_TEST_ANSWER_TX_RATE': '12000'}):
            answerer = V34AnswerModem(data_rate=33600)
        self.assertEqual(answerer.answerer._local_mp['max_a2c'], 5)
        self.assertEqual(answerer.answerer._local_mp['max_c2a'], 14)

    def test_diagnostic_answer_tx_cap_rejects_unsupported_rate(self) -> None:
        from v34_answer_modem import V34AnswerModem

        with patch.dict('os.environ', {'V34_TEST_ANSWER_TX_RATE': '13000'}):
            with self.assertRaisesRegex(ValueError, 'V34_TEST_ANSWER_TX_RATE'):
                V34AnswerModem(data_rate=33600)

    def test_dce_wrappers_exchange_asymmetric_lapm_payloads_exactly(self) -> None:
        from v34_answer_modem import V34AnswerModem
        from v34_modem import V34Modem

        caller = V34Modem(data_rate=33600)
        # Emulate an answerer advertising a 12K transmit ceiling while
        # retaining its 33.6K receive offer, as in the captured asymmetry.
        with patch.dict('os.environ', {'V34_TEST_ANSWER_TX_RATE': '12000'}):
            answerer = V34AnswerModem(data_rate=33600)
        caller_payload = bytes((i * 73 + 19) & 0xFF for i in range(16384))
        answer_payload = bytes((i * 151 + 7) & 0xFF for i in range(16384))
        caller_rx = bytearray()
        answerer_rx = bytearray()

        def tick() -> None:
            caller_tx = caller.tx_samples(160)
            answerer_tx = answerer.tx_samples(160)
            answerer_rx.extend(answerer.feed_pcm(struct.pack('<160h', *caller_tx)))
            caller_rx.extend(caller.feed_pcm(struct.pack('<160h', *answerer_tx)))

        for _ in range(1500):
            tick()
            if caller.connected and answerer.connected:
                break
        self.assertTrue(caller.connected, caller.events[-8:])
        self.assertTrue(answerer.connected, answerer.events[-8:])
        self.assertEqual((caller.bit_rate, caller.rx_bit_rate), (33600, 12000))
        self.assertEqual((answerer.bit_rate, answerer.rx_bit_rate), (12000, 33600))

        caller.start_v42(calling_party=True, rate=caller.bit_rate, frame_size=128)
        answerer.start_v42(calling_party=False, rate=answerer.bit_rate, frame_size=128)
        for _ in range(750):
            tick()
            if caller.v42_mode == answerer.v42_mode == 'lapm':
                break
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ('lapm', 'lapm'))
        caller.feed_bytes(caller_payload)
        answerer.feed_bytes(answer_payload)

        for _ in range(7500):
            tick()
            if caller_rx == answer_payload and answerer_rx == caller_payload:
                break
        self.assertEqual(bytes(caller_rx), answer_payload)
        self.assertEqual(bytes(answerer_rx), caller_payload)
        self.assertEqual((caller._v42.stats['fcs_errors'],
                          answerer._v42.stats['fcs_errors']), (0, 0))

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
