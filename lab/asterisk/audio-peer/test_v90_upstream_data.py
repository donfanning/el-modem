"""Synthetic boundary and payload check for the V.90 upstream V.34 stage."""
import random
import unittest
from unittest.mock import patch

import numpy as np

import v34_data
import v34_datarx
import v34_negotiation
import v34_tx
from v90_upstream_data import (V90UpstreamDataReceiver,
                               warmup_v90_receiver_power)


class V90UpstreamDataTests(unittest.TestCase):
    def test_startup_warmup_covers_supported_v90_upstream_profiles(self):
        cached = v34_data.DataTx._estimate_power_cached
        cached.cache_clear()
        warmed = warmup_v90_receiver_power()
        expected = tuple(
            (baud, rate)
            for baud in (v34_tx.SYMBOL_RATES[4], v34_tx.SYMBOL_RATES[5])
            for rate in range(4800, 33601, 2400)
            if v34_negotiation.supported(rate, baud, shaping=False))
        self.assertEqual(warmed, expected)
        self.assertEqual(cached.cache_info().misses, len(expected))

    def test_b1_qualification_then_streaming_v34_payload_bits(self):
        profile = v34_negotiation.DirectionParameters(9600, 2400, 1800.0)
        # V.90 §6.5 uses GPA for its analogue modem, including the calling
        # modem. In the shared V.34 helpers GPA is named the answer role.
        rx = V90UpstreamDataReceiver(profile, remote_role="answer")
        tx = v34_data.DataTx(profile.rate, profile.baud)
        b1 = np.asarray(tx.data_frame(rx.receiver.expected), dtype=np.complex128)
        gain = 0.67 * np.exp(0.31j)
        start = 37
        rx.start_b1(start)

        for index, value in enumerate(b1, start=start):
            self.assertEqual(rx.feed_symbol(index, gain * value), [])
        self.assertEqual(rx.state, rx.DATA)
        self.assertTrue(rx.receiver.diagnostics["qualified"])

        rng = random.Random(9817)
        source_bits = [rng.randrange(2)
                       for _ in range(5 * tx.bits_per_data_frame)]
        data_symbols = []
        frame_bits = tx.bits_per_data_frame
        for offset in range(0, len(source_bits), frame_bits):
            data_symbols.extend(tx.data_frame(source_bits[offset:offset + frame_bits]))
        decoded = []
        for offset, value in enumerate(data_symbols, start=start + len(b1)):
            decoded.extend(rx.feed_symbol(offset, gain * value))
        expected = v34_datarx.descramble_stream(
            source_bits, rx.receiver.bits[-23:], "answer")
        self.assertGreaterEqual(len(decoded), len(expected) - 256)
        mismatches = [index for index, (got, want)
                      in enumerate(zip(decoded, expected)) if got != want]
        self.assertFalse(mismatches,
                         f"{len(mismatches)} bit errors; first at {mismatches[:8]}")


    def test_symbol_gap_fails_closed_after_b1_start(self):
        profile = v34_negotiation.DirectionParameters(9600, 2400, 1800.0)
        rx = V90UpstreamDataReceiver(profile)
        rx.start_b1(10)
        rx.feed_symbol(10, 1 + 1j)
        rx.feed_symbol(12, 1 + 1j)
        self.assertEqual(rx.state, rx.FAILED)
        self.assertIn("symbol gap", rx.failure_reason)


if __name__ == "__main__":
    unittest.main()
