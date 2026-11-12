import random
import unittest

import numpy as np

import v34_negotiation
from v90_analogue_upstream_tx import V90AnalogueUpstreamTransmitter
from v90_upstream_data import V90UpstreamDataReceiver


class V90AnalogueUpstreamTransmitterTests(unittest.TestCase):
    def test_b1_and_payload_round_trip_through_production_answerer_receiver(self):
        profile = v34_negotiation.DirectionParameters(9600, 2400, 1800.0)
        tx = V90AnalogueUpstreamTransmitter(profile)
        rx = V90UpstreamDataReceiver(profile, remote_role="answer")
        rx.start_b1(0)
        gain = 0.67 * np.exp(0.31j)

        b1 = tx.encode_b1()
        self.assertEqual(len(b1), rx.b1_symbols_required)
        for index, symbol in enumerate(b1):
            rx.feed_symbol(index, gain * symbol)
        self.assertEqual(rx.state, rx.DATA)
        self.assertTrue(rx.receiver.diagnostics["qualified"])

        rng = random.Random(9817)
        source = [rng.randrange(2)
                  for _ in range(12 * tx.bits_per_data_frame)]
        symbols = tx.encode_bits(source)
        decoded = []
        for offset, symbol in enumerate(symbols, start=len(b1)):
            decoded.extend(rx.feed_symbol(offset, gain * symbol))
        expected = source[:len(decoded)]
        self.assertGreaterEqual(len(decoded), len(source) - 256)
        self.assertEqual(decoded, expected)

    def test_b1_order_and_complete_frame_boundaries_are_enforced(self):
        profile = v34_negotiation.DirectionParameters(9600, 2400, 1800.0)
        tx = V90AnalogueUpstreamTransmitter(profile)
        with self.assertRaisesRegex(RuntimeError, "cannot precede B1"):
            tx.encode_bits([0] * tx.bits_per_data_frame)
        tx.encode_b1()
        with self.assertRaisesRegex(ValueError, "complete V.34 data frames"):
            tx.encode_bits([0])
        with self.assertRaisesRegex(RuntimeError, "already sent"):
            tx.encode_b1()


if __name__ == "__main__":
    unittest.main()
