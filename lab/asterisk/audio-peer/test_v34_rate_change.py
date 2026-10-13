"""Section 11.6 wire tests; these do not qualify a live PCM frontend."""
import unittest

import numpy as np

import v34_data
import v34_mp
import v34_negotiation as negotiation
import v34_tx as tx
from v34_rate_change import MPScanner, RateExchange, RateQamBridge


def rotations(symbols):
    return [tx.rotation_of(z) for z in symbols]


def wire(role, length, mp_type=1, precoder=None):
    trn, sc = tx.trn_symbols(length, role, 4)
    z = tx.trn_last_rotation(trn, 4) if length else 0
    ordinary = v34_mp.build_mp(mp_type, max_c2a=14, max_a2c=14,
                              mask=0x3fff, precoder=precoder)
    prime = v34_mp.build_mp(mp_type, max_c2a=14, max_a2c=14,
                           mask=0x3fff, ack=1, precoder=precoder)
    sy, _ = tx.bit_symbols(ordinary * 3 + prime + [1] * 20, role, sc, z, 4)
    return np.concatenate((trn, sy))


class ScannerTests(unittest.TestCase):
    def test_unknown_training_and_global_rotation(self):
        for role in ('call', 'answer'):
            for length in (0, 1, 11, 1024, 6000):
                for ambiguity in range(4):
                    with self.subTest(role=role, length=length, ambiguity=ambiguity):
                        scanner = MPScanner(role)
                        events = []
                        symbols = wire(role, length)
                        for r in rotations(symbols):
                            events += scanner.feed_rotation((r + ambiguity) % 4, allow_e=True)
                            self.assertLessEqual(len(scanner.buffer), 187)
                            self.assertEqual(len(scanner.history), 23)
                        messages = [e['mp'] for e in events if e['kind'] == 'mp']
                        self.assertGreaterEqual(len(messages), 3)
                        self.assertTrue(messages[-1]['ack'])
                        self.assertEqual(scanner.e_end, len(symbols) * 2)

    def test_e_requires_both_valid_mp_and_permission(self):
        for allowed in (False, True):
            scanner = MPScanner('call')
            sy, _ = tx.bit_symbols([1] * 200, 'call', tx.Scrambler('call'))
            for r in rotations(sy):
                self.assertFalse(scanner.feed_rotation(r, allow_e=allowed))
            self.assertIsNone(scanner.e_end)
        scanner = MPScanner('call')
        for r in rotations(wire('call', 64)):
            scanner.feed_rotation(r, allow_e=False)
        self.assertTrue(scanner.have_mp)
        self.assertIsNone(scanner.e_end)

    def test_bad_crc_followed_by_recovery_and_type_zero(self):
        sc = tx.Scrambler('answer')
        bad = v34_mp.build_mp(0)
        bad[70] ^= 1
        good = v34_mp.build_mp(0, ack=1)
        sy, _ = tx.bit_symbols([0] * 24 + bad + good + [1] * 20, 'answer', sc)
        scanner = MPScanner('answer')
        events = []
        for r in rotations(sy):
            events += scanner.feed_rotation(r, allow_e=True)
        self.assertEqual([e['kind'] for e in events], ['mp', 'e'])
        self.assertEqual(events[0]['mp']['type'], 0)
        self.assertEqual(events[0]['start'], 24 + 88)
        self.assertEqual(scanner.e_end, len(sy) * 2)


class ExchangeTests(unittest.TestCase):
    def make(self, role, cap=33600, **kwargs):
        exchange = RateExchange(negotiation.offer(cap), negotiation.offer(33600),
                            role=role, initiating=role == 'call',
                            tx_baud=tx.SYMBOL_RATES[5], rx_baud=tx.SYMBOL_RATES[5], **kwargs)
        exchange.receive_s()
        return exchange

    def test_two_way_wire_downward_and_upward(self):
        # Fresh transactions with the same original ceiling: do not accidentally
        # cap the upward transaction at the previous negotiated lower rate.
        for cap in (4800, 7200, 9600, 14400, 24000, 28800, 33600):
            a, b = self.make('call', cap), self.make('answer')
            for _ in range(20):
                units = [a.next_unit(), b.next_unit()]
                for own, peer, unit in ((a, b, units[0]), (b, a, units[1])):
                    if unit is None:
                        continue
                    if unit.kind not in ('s', 'sbar', 'b1'):
                        for r in rotations(unit.symbols):
                            peer.feed_rotation(r)
                    own.transmitted(unit)
                if a.tx_ready and b.tx_ready:
                    break
            for exchange in (a, b):
                self.assertIsNone(exchange.failed)
                self.assertTrue(exchange.tx_ready)
                self.assertTrue(exchange.peer_e)
                self.assertEqual(exchange.negotiated.tx.rate, cap)
                self.assertEqual(exchange.negotiated.rx.rate, cap)
                self.assertTrue(exchange.rx_clamped)
                exchange.receive_b1(qualified=True)
                self.assertFalse(exchange.rx_clamped)

    def test_physical_completion_and_fresh_b1(self):
        e = self.make('call')
        for kind in ('s', 'sbar', 'trn', 'mp'):
            unit = e.next_unit()
            self.assertEqual(unit.kind, kind)
            e.transmitted(unit)
        for r in rotations(wire('answer', 64)):
            e.feed_rotation(r)
        # Peer E cannot bypass transmission of our first MP'.
        self.assertFalse(e.prime_sent)
        unit = e.next_unit()
        self.assertEqual(unit.kind, 'mp_prime')
        self.assertFalse(e.prime_sent)
        e.transmitted(unit)
        self.assertTrue(e.prime_sent)
        e.transmitted(e.next_unit())  # E
        b1 = e.next_unit()
        self.assertEqual(b1.kind, 'b1')
        self.assertFalse(e.tx_ready)
        p = e.negotiated.tx
        reference = v34_data.DataTx(p.rate, p.baud,
            precoder=v34_data.precoder_from_mp(p.precoder))
        sc = tx.Scrambler('call')
        expected = np.asarray(reference.data_frame([sc.step(1) for _ in range(reference.bits_per_data_frame)]))
        np.testing.assert_array_equal(b1.symbols, expected * np.sqrt(2 / reference.avg_power))
        e.transmitted(b1)
        self.assertTrue(e.tx_ready)
        self.assertEqual(e.data_sc.reg, sc.reg)
        with self.assertRaises(RuntimeError):
            e.transmitted(b1)

    def test_lookahead_preserves_physical_barrier_and_order(self):
        e = self.make('call')
        queued = [e.next_unit() for _ in range(4)]
        self.assertEqual([u.kind for u in queued], ['s', 'sbar', 'trn', 'mp'])
        with self.assertRaises(RuntimeError):
            e.transmitted(queued[1])
        for unit in queued:
            e.transmitted(unit)
        for r in rotations(wire('answer', 32)):
            e.feed_rotation(r)
        p1, p2 = e.next_unit(), e.next_unit()
        self.assertEqual([p1.kind, p2.kind], ['mp_prime', 'mp_prime'])
        self.assertFalse(e.prime_sent)
        e.transmitted(p1)
        end = e.next_unit()
        self.assertEqual(end.kind, 'e')
        with self.assertRaises(RuntimeError):
            e.transmitted(end)
        e.transmitted(p2)
        e.transmitted(end)
        b1 = e.next_unit()
        self.assertIsNone(e.next_unit())
        self.assertFalse(e.tx_ready)
        e.transmitted(b1)
        self.assertTrue(e.tx_ready)
        limited = self.make('call')
        for _ in range(32):
            limited.next_unit()
        with self.assertRaises(RuntimeError):
            limited.next_unit()

    def test_type_zero_retains_previous_precoder(self):
        old = negotiation.offer(33600)
        old['precoder'] = [323, 0, -162, 16, -8, 36]
        e = RateExchange(negotiation.offer(33600), old, role='call', initiating=True,
                         tx_baud=tx.SYMBOL_RATES[5], rx_baud=tx.SYMBOL_RATES[5])
        e.receive_s()
        for r in rotations(wire('answer', 32, mp_type=0)):
            e.feed_rotation(r)
        self.assertEqual(e.negotiated.tx.precoder, tuple(old['precoder']))

    def test_peer_e_can_replace_lost_prime(self):
        e = self.make('call')
        for _ in range(4):
            e.transmitted(e.next_unit())
        sc = tx.Scrambler('answer')
        ordinary = v34_mp.build_mp(1, **negotiation.offer(33600))
        sy, z = tx.bit_symbols(ordinary * 3, 'answer', sc)
        for r in rotations(sy):
            e.feed_rotation(r)
        prime = e.next_unit()
        self.assertEqual(prime.kind, 'mp_prime')
        e.transmitted(prime)
        sy, _ = tx.bit_symbols([1] * 20, 'answer', sc, z)
        for r in rotations(sy):
            e.feed_rotation(r)
        self.assertFalse(e.peer_prime)
        self.assertTrue(e.peer_e)
        self.assertEqual(e.next_unit().kind, 'e')
        e.receive_b1(qualified=False)
        self.assertTrue(e.rx_clamped)
        with self.assertRaises(RuntimeError):
            e.receive_b1(qualified=True)

    def test_changed_fields_fail_without_unclamping(self):
        e = self.make('call')
        sc = tx.Scrambler('answer')
        bits = v34_mp.build_mp(1, **negotiation.offer(33600)) * 2
        bits += v34_mp.build_mp(1, ack=1, **negotiation.offer(24000))
        sy, _ = tx.bit_symbols(bits, 'answer', sc)
        for r in rotations(sy):
            e.feed_rotation(r)
        self.assertIn('changed coding', e.failed)
        self.assertTrue(e.rx_clamped)
        self.assertIsNone(e.next_unit())

    def test_timeouts_and_b1_barrier(self):
        for initiating in (True, False):
            for cme in (True, False):
                e = RateExchange(negotiation.offer(33600), negotiation.offer(33600),
                    role='call', initiating=initiating, tx_baud=3200, rx_baud=3200,
                    rtd=.5, cme=cme)
                limit = 30 if cme else 2.5 + (2 if initiating else 3) * .5
                self.assertIsNone(e.check_timeout(100))  # transition not transmitted yet
                e.mark_sbar_start(4)
                self.assertIsNone(e.check_timeout(4 + limit - .001))
                self.assertIn('full retrain', e.check_timeout(4 + limit))
        with self.assertRaises(RuntimeError):
            self.make('call').receive_b1(qualified=True)
        for n in (0, 10000):
            with self.assertRaises(ValueError):
                self.make('answer', trn_symbols=n)

    def test_initiator_keeps_receiving_until_peer_s(self):
        e = RateExchange(negotiation.offer(33600), negotiation.offer(33600),
            role='call', initiating=True, tx_baud=3200, rx_baud=3200)
        self.assertFalse(e.rx_clamped)
        with self.assertRaises(RuntimeError):
            e.feed_rotation(0)
        e.receive_s()
        self.assertTrue(e.rx_clamped)
        self.assertEqual(e.feed_rotation(0), [])

    def test_qam_bridge_drains_old_data_without_resetting_shaper(self):
        baud, fc = tx.SYMBOL_RATES[5], 1959.1836734693877
        e = RateExchange(negotiation.offer(24000), negotiation.offer(33600),
            role='call', initiating=True, tx_baud=baud, rx_baud=baud,
            tx_carrier=fc, rx_carrier=fc, trn_symbols=64)
        qam = tx.StreamQam(baud, fc, 1000)
        reference = tx.StreamQam(baud, fc, 1000)
        qam.set_constellation(16)
        reference.set_constellation(16)
        old = np.tile(np.asarray([1 + 3j, -3 + 1j, -1 - 3j, 3 - 1j]), 300)
        qam.push(old)
        reference.push(old)
        np.testing.assert_array_equal(qam.read(13), reference.read(13))
        queue, history, buffered = list(qam.queue), list(qam.history), qam.buf.copy()

        def data_fill(q, exchange):
            encoder = exchange.data_tx
            bits = [exchange.data_sc.step(1) for _ in range(encoder.bits_per_data_frame)]
            q.push(np.asarray(encoder.data_frame(bits)) * np.sqrt(q.power / encoder.avg_power))

        bridge = RateQamBridge(qam, e, data_fill)
        self.assertEqual(qam.queue, queue)
        self.assertEqual(qam.history, history)
        np.testing.assert_array_equal(qam.buf, buffered)
        # Well before the old/new pulse overlap, old PCM is bit-identical.
        np.testing.assert_array_equal(bridge.read(160), reference.read(160))
        self.assertFalse(e.tx_ready)
        e.receive_s()
        for r in rotations(wire('answer', 32)):
            e.feed_rotation(r)
        for _ in range(100):
            self.assertEqual(len(bridge.read(160)), 160)
            if e.tx_ready:
                break
        self.assertTrue(e.tx_ready)
        self.assertEqual(qam.constellation, 16)
        self.assertEqual(qam.power, tx.constellation_power(16))
        self.assertEqual(qam.underrun_symbols if hasattr(qam, 'underrun_symbols') else 0, 0)
        self.assertEqual(e.transition_at, (len(old) + 128) / baud)
        self.assertTrue(e.rx_clamped)  # no receive-side qualification supplied


if __name__ == '__main__':
    unittest.main()
