#!/usr/bin/env python3
"""V42Layer: two engines wired bit-for-bit, one calling and one answering."""
import unittest
from collections import deque

from v42_layer import V42Layer, _Framer


class BitEngine(V42Layer):
    def __init__(self, rate=9600):
        self.bit_rate = rate
        self._v42_setup()
        self.q = deque()
        self.fr = _Framer()
        self.out = bytearray()

    def _plain_tx_bit(self):
        return self.q.popleft() if self.q else 1

    def _plain_feed_bytes(self, data):
        for v in data:
            self.q.extend([0] + [(v >> i) & 1 for i in range(8)] + [1])

    def _plain_feed_marks(self, n):
        self.q.extend([1] * n)

    def _plain_rx_bits(self, bits):
        return self.fr.push(list(bits))


def pump(a, b, bits):
    for _ in range(bits):
        ba, bb = a.v42_tx_bit(), b.v42_tx_bit()
        a.out += a.v42_rx_bits([bb])
        b.out += b.v42_rx_bits([ba])
        a.out += a.v42_read()
        b.out += b.v42_read()


class V42LayerTests(unittest.TestCase):
    def test_answerer_direct_flags_leave_detection_timer_then_establish_lapm(self):
        from v42_lapm import FLAG_BITS
        a,b=BitEngine(),BitEngine()
        b.start_v42(False,rate=9600)
        b.v42_feed_bytes(b'held answerer payload')
        # A direct-protocol originator sends flags, not ODP. Let detection
        # expire while protocol establishment is still legitimately pending.
        for i in range(9600):
            b.v42_rx_bits([FLAG_BITS[i%8]]);b.v42_tx_bit()
        self.assertEqual(b._v42.state,'idle')
        self.assertFalse(b._v42.adp_seen)
        self.assertEqual(b.v42_mode,'detect')
        a.start_v42(True,rate=9600,detect=False)
        a.v42_feed_bytes(b'direct originator payload')
        pump(a,b,9600*5)
        self.assertEqual((a.v42_mode,b.v42_mode),('lapm','lapm'))
        self.assertEqual(bytes(a.out),b'held answerer payload')
        self.assertEqual(bytes(b.out),b'direct originator payload')

    def test_answerer_flags_without_establishment_still_have_finite_deadline(self):
        from v42_lapm import FLAG_BITS
        b=BitEngine();b.start_v42(False,rate=9600,establish_seconds=2.)
        for i in range(9600*3):
            b.v42_rx_bits([FLAG_BITS[i%8]]);b.v42_tx_bit()
        self.assertEqual(b.v42_mode,'failed')

    def test_isolated_flags_do_not_extend_raw_detection_deadline(self):
        from v42_lapm import FLAG_BITS
        b = BitEngine()
        b.start_v42(False, rate=9600)
        incoming = (FLAG_BITS + [1] * 128) * 4 + [1] * 9600
        for bit in incoming:
            b.v42_rx_bits([bit])
            b.v42_tx_bit()
        self.assertEqual(b.v42_mode, 'failed')
        self.assertEqual(b._v42._phase, 'detect')
        self.assertFalse(b._v42.adp_seen)

    def test_continuous_flags_allow_shared_zero_and_reset_across_gap(self):
        from v42_lapm import FLAG_BITS, _Hdlc
        receiver = _Hdlc()
        for bit in FLAG_BITS + FLAG_BITS[1:] * 3:
            receiver.push(bit)
        self.assertEqual(receiver.continuous_flags, 4)
        for bit in [1] * 128 + FLAG_BITS:
            receiver.push(bit)
        self.assertEqual(receiver.flags, 5)
        self.assertEqual(receiver.continuous_flags, 1)

    def test_both_ends_reach_lapm_and_exchange_data(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600)
        b.start_v42(False, rate=9600)
        a.v42_feed_bytes(b"hello from the caller\r\n")
        b.v42_feed_bytes(b"hello from the answerer\r\n")
        pump(a, b, 9600 * 5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertEqual(bytes(b.out), b"hello from the caller\r\n")
        self.assertEqual(bytes(a.out), b"hello from the answerer\r\n")

    def test_v42bis_is_negotiated_and_compresses_text(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600, compression=(3, 1024, 8))
        b.start_v42(False, rate=9600, compression=(3, 1024, 8))
        text = (b"the quick brown fox jumps over the lazy dog. " * 40) + b"\r\n"
        a.v42_feed_bytes(text)
        b.v42_feed_bytes(text[::-1])
        pump(a, b, 9600 * 8)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertEqual(a.v42_compression, (3, 1024, 8))
        self.assertEqual(b.v42_compression, (3, 1024, 8))
        self.assertEqual(bytes(b.out), text)
        self.assertEqual(bytes(a.out), text[::-1])
        self.assertLess(a._v42.compressed_out, len(text) // 3)          # the line carried far less than the text

    def test_compression_is_dropped_when_the_responder_does_not_accept_it(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600, compression=(3, 1024, 8))
        b.start_v42(False, rate=9600)                                  # accepts nothing
        a.v42_feed_bytes(b"plain " * 30)
        pump(a, b, 9600 * 5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertIsNone(a.v42_compression)
        self.assertEqual(bytes(b.out), b"plain " * 30)

    def test_caller_falls_back_to_raw_async_when_peer_has_no_v42(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600)                 # the peer never starts V.42
        a.v42_feed_bytes(b"raw data please\r\n")
        pump(a, b, 9600 * 3)
        self.assertEqual(a.v42_mode, "failed")
        self.assertIn(b"raw data please\r\n", bytes(b.out))

    def test_plain_mode_is_untouched_async(self):
        a, b = BitEngine(), BitEngine()
        a.v42_feed_bytes(b"no error control\r\n")
        pump(a, b, 400)
        self.assertEqual(bytes(b.out), b"no error control\r\n")

    def test_mnp_protocol_runs_through_the_same_layer(self):
        for level in (2, 3, 4):
            with self.subTest(level=level):
                a, b = BitEngine(), BitEngine()
                a.start_v42(True, rate=9600, protocol="mnp", mnp_level=level)
                b.start_v42(False, rate=9600, protocol="mnp", mnp_level=level)
                a.v42_feed_bytes(b"hello over MNP\r\n")
                pump(a, b, 9600 * 4)
                self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
                self.assertEqual(bytes(b.out), b"hello over MNP\r\n")

    def test_mnp_originator_falls_back_to_raw_when_the_peer_is_silent(self):
        a = BitEngine()
        a.start_v42(True, rate=9600, protocol="mnp", establish_seconds=3.0, detect_seconds=0)
        a.v42_feed_bytes(b"late\r\n")
        for _ in range(9600 * 5):
            a.v42_tx_bit()
            a.v42_rx_bits([1])
        self.assertEqual(a.v42_mode, "failed")



def delayed_pump(a, b, bits, delay):
    """Like pump, with `delay` bits of propagation (plus receive queue) in each direction."""
    to_a, to_b = deque([1] * delay), deque([1] * delay)
    for _ in range(bits):
        ba, bb = a.v42_tx_bit(), b.v42_tx_bit()
        to_a.append(bb)
        to_b.append(ba)
        a.out += a.v42_rx_bits([to_a.popleft()])
        b.out += b.v42_rx_bits([to_b.popleft()])
        a.out += a.v42_read()
        b.out += b.v42_read()


class DetectionWindowTests(unittest.TestCase):
    """dialup.world 2026-10-07: the ADP came back after the 0.73 s detection window had closed and the originator fell back to plain data."""

    def run_delayed(self, one_way_seconds, extra):
        a, b = BitEngine(), BitEngine()
        a.v42_detect_extra_seconds = extra
        b.v42_detect_extra_seconds = extra
        a.start_v42(True, rate=9600)
        b.start_v42(False, rate=9600)
        a.v42_feed_bytes(b"hello from the caller\r\n")
        delayed_pump(a, b, 9600 * 9, int(one_way_seconds * 9600))
        return a, b

    def test_a_return_trip_longer_than_the_default_window_fails_without_the_extension(self):
        a, b = self.run_delayed(0.5, extra=0.0)              # ADP is back after about 1.0 s, window is 0.73 s
        self.assertEqual(a.v42_mode, "failed")

    def test_the_extension_lets_the_same_slow_link_reach_lapm(self):
        a, b = self.run_delayed(0.5, extra=1.5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertEqual(bytes(b.out), b"hello from the caller\r\n")

    def test_the_extension_does_not_change_a_normal_link(self):
        a, b = self.run_delayed(0.0, extra=1.5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))

    def test_a_peer_without_v42_still_falls_back_after_the_longer_window(self):
        a = BitEngine()
        a.v42_detect_extra_seconds = 1.5
        a.start_v42(True, rate=9600)
        for _ in range(9600 * 6):
            a.v42_tx_bit()
            a.v42_rx_bits([1])
        self.assertEqual(a.v42_mode, "failed")
        self.assertGreater(a._v42_detect_deadline, int(2.0 * 9600))

    def test_the_v34_engine_derives_the_extension_from_its_round_trip(self):
        from unittest.mock import patch
        import os
        from v34_modem import V34Modem
        modem = V34Modem(data_rate=9600)
        self.assertAlmostEqual(modem.v42_detect_extra_seconds, V34Modem.V42_RX_LAG_MARGIN_S)
        modem.caller.rtd_samples = 3900                      # 487.5 ms
        self.assertAlmostEqual(modem.v42_detect_extra_seconds, 0.4875 + V34Modem.V42_RX_LAG_MARGIN_S)
        with patch.dict(os.environ, {"V42_DETECT_EXTRA_S": "0"}):
            self.assertEqual(modem.v42_detect_extra_seconds, 0.0)

if __name__ == "__main__":
    unittest.main()
