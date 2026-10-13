#!/usr/bin/env python3
"""Self-tests for the V.42 detection-phase patterns."""

import unittest

import spandsp_v22
from v42_detect import (
    AdpDetector, OdpDetector, adp_bits, char_bits, odp_bits,
)
from spandsp_v22 import _AsyncFramer


def decode(bits):
    framer, out = _AsyncFramer(), []
    for bit in bits:
        out += framer.push(bit)
    return bytes(out)


class PatternTests(unittest.TestCase):
    def test_odp_matches_the_spandsp_bit_string(self):
        # "0100010001 11111111 0100010011 11111111"
        expected = [int(c) for c in "0100010001" "11111111" "0100010011" "11111111"]
        self.assertEqual(odp_bits(1), expected)

    def test_adp_matches_the_spandsp_bit_strings(self):
        self.assertEqual(adp_bits(True, 1), [int(c) for c in
                         "0101000101" "11111111" "0110000101" "11111111"])
        self.assertEqual(adp_bits(False, 1), [int(c) for c in
                         "0101000101" "11111111" "0000000001" "11111111"])

    def test_patterns_decode_to_the_expected_characters(self):
        self.assertEqual(decode(odp_bits(3)), bytes([0x11, 0x91] * 3))
        self.assertEqual(decode(adp_bits(True, 3)), b"EC" * 3)
        self.assertEqual(decode(adp_bits(False, 3)), b"E\x00" * 3)
        self.assertEqual(char_bits(0x45), [0, 1, 0, 1, 0, 0, 0, 1, 0, 1])

    def test_detectors(self):
        d = AdpDetector()
        self.assertIsNone(d.feed(b"hello EC")); self.assertIsNone(d.feed(b"x"))
        self.assertEqual(d.feed(b"ECEC"), "v42")
        d2 = AdpDetector()
        self.assertEqual(d2.feed(b"E\x00E\x00"), "no_v42")
        d3 = AdpDetector()
        self.assertIsNone(d3.feed(b"National Institute of Standards"))
        o = OdpDetector()
        self.assertFalse(o.feed(b"\x11"))
        self.assertTrue(o.feed(b"\x91\x11\x91"))


@unittest.skipUnless(spandsp_v22.available(), "libspandsp not installed")
class OverTheLineTests(unittest.TestCase):
    def test_odp_then_adp_over_a_v22bis_link(self):
        import struct
        caller = spandsp_v22.V22bisModem(True, bit_rate=2400, bell212a=False)
        answerer = spandsp_v22.V22bisModem(False, bit_rate=2400, bell212a=False)
        odp, adp = OdpDetector(), AdpDetector()
        sent_odp = replied = False
        for _ in range(50 * 20):
            fa, fb = caller.tx_samples(160), answerer.tx_samples(160)
            got_a = answerer.feed_pcm(struct.pack("<160h", *fa))
            got_c = caller.feed_pcm(struct.pack("<160h", *fb))
            if caller.is_settled() and not sent_odp:
                caller.feed_bits(odp_bits(8))
                sent_odp = True
            if odp.feed(got_a) and not replied:
                answerer.feed_bits(adp_bits(True, 8))
                replied = True
            adp.feed(got_c)
        self.assertTrue(odp.seen)
        self.assertEqual(adp.result, "v42")


if __name__ == "__main__":
    unittest.main()
