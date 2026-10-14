#!/usr/bin/env python3
"""MnpLink (V.42 Annex A, MNP 2-4 class) with two ends wired bit for bit."""
import random
import unittest

from mnp_link import MnpLink, OPT_FIXED_FIELDS, OPT_N401_256, crc16_hdlc, crc16_v42a


def run(a, b, bits, ber=0.0, burst=0, seed=1, feed=None):
    rnd = random.Random(seed)
    got_a, got_b = bytearray(), bytearray()
    for i in range(bits):
        ba, bb = a.tx_bit(), b.tx_bit()
        if ber and rnd.random() < ber:
            ba ^= 1
        if ber and rnd.random() < ber:
            bb ^= 1
        if burst and i % 20000 == 5000:                      # drop a burst of bits (a lost RTP packet)
            for _ in range(burst):
                a.tx_bit(), b.tx_bit()
        a.rx_bit(bb)
        b.rx_bit(ba)
        got_a += a.read()
        got_b += b.read()
        if feed:
            feed(i)
    return bytes(got_a), bytes(got_b)


LEVELS = {
    "mnp2": dict(framing=2, optimize=0),
    "mnp3": dict(framing=3, optimize=0),
    "mnp4": dict(framing=3, optimize=OPT_N401_256 | OPT_FIXED_FIELDS),
}


class MnpTests(unittest.TestCase):
    def test_crcs_have_the_known_check_values(self):
        self.assertEqual(crc16_v42a(b"123456789"), 0xBB3D)           # CRC-16/ARC
        self.assertEqual(crc16_hdlc(b"123456789"), 0x906E)           # CRC-16/X-25

    def test_each_level_establishes_and_exchanges_data(self):
        for name, cfg in LEVELS.items():
            with self.subTest(level=name):
                a = MnpLink(True, 9600, **cfg)
                b = MnpLink(False, 9600, **cfg)
                a.write(b"hello from the caller\r\n")
                b.write(b"hello from the answerer\r\n")
                ga, gb = run(a, b, 9600 * 3)
                self.assertEqual((a.state, b.state), ("lapm", "lapm"))
                self.assertEqual(gb, b"hello from the caller\r\n")
                self.assertEqual(ga, b"hello from the answerer\r\n")
                self.assertEqual((a.framing, a.optimize), (cfg["framing"], cfg["optimize"]))

    def test_negotiation_picks_the_lesser_values(self):
        a = MnpLink(True, 9600, framing=3, k=8, n401=64, optimize=3)
        b = MnpLink(False, 9600, framing=2, k=4, n401=32, optimize=1)
        run(a, b, 9600 * 3)
        self.assertEqual((a.framing, b.framing), (2, 2))
        self.assertEqual((a.k, b.k), (4, 4))
        self.assertEqual((a.optimize, b.optimize), (1, 1))
        self.assertEqual(a.n401, 256)                                # optimization bit 1: 256-octet fields

    def test_bulk_data_both_ways_is_byte_exact(self):
        rnd = random.Random(7)
        da = bytes(rnd.randrange(256) for _ in range(6000))           # includes DLE and flag-like bytes
        db = bytes(rnd.randrange(256) for _ in range(5000))
        for name, cfg in LEVELS.items():
            with self.subTest(level=name):
                a, b = MnpLink(True, 9600, **cfg), MnpLink(False, 9600, **cfg)
                a.write(da)
                b.write(db)
                ga, gb = run(a, b, 9600 * 30)
                self.assertEqual(gb, da)
                self.assertEqual(ga, db)

    def test_bit_errors_and_lost_bursts_are_recovered(self):
        rnd = random.Random(3)
        da = bytes(rnd.randrange(256) for _ in range(4000))
        db = bytes(rnd.randrange(256) for _ in range(4000))
        for name, cfg in LEVELS.items():
            with self.subTest(level=name):
                a, b = MnpLink(True, 9600, **cfg), MnpLink(False, 9600, **cfg)
                a.write(da)
                b.write(db)
                ga, gb = run(a, b, 9600 * 80, ber=2e-4, burst=300)
                self.assertEqual(a.state, "lapm")
                self.assertEqual(gb, da)
                self.assertEqual(ga, db)

    def test_originator_keeps_trying_and_a_silent_peer_never_connects(self):
        a = MnpLink(True, 9600)
        for _ in range(9600 * 6):
            a.tx_bit()
            a.rx_bit(1)
        self.assertEqual(a.state, "idle")
        self.assertGreaterEqual(a._lr_attempts, 3)

    def test_data_written_late_and_in_dribbles_arrives_in_order(self):
        a, b = MnpLink(True, 9600, **LEVELS["mnp4"]), MnpLink(False, 9600, **LEVELS["mnp4"])
        sent = bytearray()

        def feed(i):
            if i % 4000 == 0 and i > 9600:
                chunk = bytes([65 + (i // 4000) % 26]) * 37
                sent.extend(chunk)
                a.write(chunk)

        _, gb = run(a, b, 9600 * 20, feed=feed)
        self.assertEqual(gb, bytes(sent))


if __name__ == "__main__":
    unittest.main()
