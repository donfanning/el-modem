"""Batch RX preserves frame boundaries, state changes and exact application bytes."""
import unittest
from test_v42_layer import BitEngine
from v42_lapm import V42Link, FLAG_BITS, RR, DISC, SABME, _stuffed


class BatchReceiveTests(unittest.TestCase):
    def assert_link_equal(self, a, b):
        for name in ('state', 'events', 'stats', 'vs', 'va', 'vr', 'bits_sent',
                     '_need_rej', '_ack_pending', '_t401_at', 'compression'):
            self.assertEqual(getattr(a, name), getattr(b, name), name)
        self.assertEqual(list(a._tx), list(b._tx))
        self.assertEqual(a._hdlc.__dict__, b._hdlc.__dict__)

    def test_payload_and_compression_across_fragmented_establishment(self):
        for compression in (None, (3, 1024, 8)):
            ref = [BitEngine(33600), BitEngine(33600)]
            bat = [BitEngine(33600), BitEngine(33600)]
            messages = [bytes(range(256)) * 6, b'\x7e\xff\x00hello world' * 160]
            for pair in (ref, bat):
                for i, x in enumerate(pair):
                    x.start_v42(i == 0, rate=33600, compression=compression)
                    x.v42_feed_bytes(messages[i])
            # An instance wrapper forces the reference through the original
            # layer's scalar dispatch, including detection and HDLC feed.
            for x in ref:
                single = x._v42.rx_bit
                x._v42.rx_bit = lambda bit, single=single: single(bit)
            for turn in range(250):
                count = (1, 79, 1136, 17)[turn % 4]
                streams = []
                for a, b in zip(ref, bat):
                    bits = a.v42_tx_bits(count)
                    self.assertEqual(bits, b.v42_tx_bits(count))
                    streams.append(bits)
                for pair in (ref, bat):
                    for i, x in enumerate(pair):
                        x.out += x.v42_rx_bits(iter(streams[1-i]))
                        x.out += x.v42_read()
                for a, b in zip(ref, bat):
                    self.assertEqual(a.out, b.out)
                    self.assert_link_equal(a._v42, b._v42)
            for i, x in enumerate(bat):
                self.assertEqual(x.v42_mode, 'lapm')
                self.assertEqual(x.out, messages[1-i])

    def test_disconnect_stops_later_frames_in_same_batch(self):
        a, b = V42Link(True, detect=False), V42Link(True, detect=False)
        a.state = b.state = 'lapm'
        frames = [bytes((a.rsp_addr, RR, 0)),
                  bytes((a.rsp_addr, DISC | 0x10)),
                  bytes((a.rsp_addr, SABME | 0x10)),
                  bytes((a.rsp_addr, 0, 0)) + b'must not arrive']
        bits = FLAG_BITS + sum((_stuffed(f) + FLAG_BITS for f in frames), [])
        for bit in bits:
            a.rx_bit(bit)
        b.rx_bits(iter(bits))
        self.assert_link_equal(a, b)
        self.assertEqual(b.state, 'down')
        self.assertEqual(b.read(), b'')

    def test_corruption_abort_stuffing_and_boundary_splits(self):
        a, b = V42Link(True, detect=False), V42Link(True, detect=False)
        a.state = b.state = 'lapm'
        good = _stuffed(bytes((a.rsp_addr, 0, 0)) + b'\xff\x7e\x00' * 20)
        bad = _stuffed(bytes((a.rsp_addr, RR, 0)))
        bad[3] ^= 1
        bits = FLAG_BITS + good + FLAG_BITS + bad + FLAG_BITS + [1]*10 + FLAG_BITS
        for offset in range(0, len(bits), 13):
            chunk = bits[offset:offset+13]
            for bit in chunk:
                a.rx_bit(bit)
            b.rx_bits(chunk)
            self.assert_link_equal(a, b)
        self.assertEqual(b.read(), b'\xff\x7e\x00' * 20)
        self.assertEqual(b.stats['fcs_errors'], 1)


if __name__ == '__main__':
    unittest.main()
