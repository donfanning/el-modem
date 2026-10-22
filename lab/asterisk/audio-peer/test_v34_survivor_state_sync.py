"""Chosen trellis state must accompany decisions across survivor switches."""
import unittest
import numpy as np
import v34_data as vd
from v34_datarx import DataRx
from v34_viterbi import TrellisDetector
from v34_trellis_native import njit


class SurvivorStateTests(unittest.TestCase):
    def test_selected_state_recovers_recursive_mapper_python(self):
        self._check_selected_state('python')

    @unittest.skipUnless(njit is not None, 'optional Numba backend unavailable')
    def test_selected_state_recovers_recursive_mapper_numba(self):
        self._check_selected_state('numba')

    def _check_selected_state(self, backend):
        tx = vd.DataTx(33600, 3429)
        rng = np.random.default_rng(3400)
        frames = [rng.integers(0, 2, tx.bits_per_data_frame).tolist() for _ in range(8)]
        symbols = [s for bits in frames for s in tx.data_frame(bits)]
        det = TrellisDetector(4*tx.M*2**tx.q, tx.J, tx.P, backend=backend)
        det.emit_states = True
        det.emit_metrics = True
        for a, b in zip(symbols[::2], symbols[1::2]):
            det.feed(a, b)
        det.flush()
        self.assertEqual(len(det.decided_states), len(det.decided))
        self.assertEqual(len(det.decided_metrics), len(det.decided))
        self.assertEqual([m['decision_interval'] for m in det.decided_metrics],
                         sorted(m['decision_interval'] for m in det.decided_metrics))
        self.assertTrue(all(m['survivor_gap'] >= 0 for m in det.decided_metrics))
        self.assertEqual(det.decided_states, [s['state_before'] for s in det.snapshots])
        plain, corrected = DataRx(33600, 3429), DataRx(33600, 3429)
        plain.conv.d = corrected.conv.d = [1, 1, 1, 1]
        plain_bits, corrected_bits = [], []
        for i in range(len(symbols)//8):
            points = [p for pair in det.decided[4*i:4*i+4] for p in pair]
            plain_bits.extend(plain.decode_frame(points)[0])
            corrected_bits.extend(corrected.decode_frame(points,
                trellis_states=det.decided_states[4*i:4*i+4])[0])
        wanted = [bit for frame in frames for bit in frame]
        self.assertEqual(corrected_bits, wanted)
        self.assertNotEqual(plain_bits, wanted)

    def test_invalid_states_do_not_advance_mapper(self):
        rx = DataRx(9600, 3200)
        for states in ([0], [0, 0, 0, 16], [0, 0, 0, 1.5]):
            with self.assertRaises(ValueError):
                rx.decode_frame([1+1j]*8, trellis_states=states)
            self.assertEqual(rx.frame_pos, 0)
            self.assertEqual(rx.m4, 0)

    def test_optional_state_queue_stays_empty_by_default(self):
        det = TrellisDetector(16, 7, 12, backend='python')
        for _ in range(30):
            det.feed(1+1j, 1+1j)
        det.flush()
        self.assertTrue(det.decided)
        self.assertEqual(det.decided_states, [])
