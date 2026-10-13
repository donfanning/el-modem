"""The idle-continuation forward model reproduces the independent B1 reference on B1's own symbols."""
import unittest

import numpy as np

import v34_b1_reference as B
import v34_idle_forward as F


class IdleForwardTests(unittest.TestCase):
    def test_matches_b1_reference_over_b1(self):
        for rate, baud, role in ((33600, 3428.57, 'answer'), (33600, 3428.57, 'call'), (24000, 3428.57, 'answer')):
            ref = np.asarray(B.build(rate, baud, role)['x'])
            x, par = F.build_idle(rate, baud, role, par_frames(rate, baud) + 5)
            self.assertEqual(np.abs(x[:len(ref)] - ref).max(), 0.0, (rate, role))

    def test_continuation_is_not_periodic_garbage_and_has_the_right_length(self):
        x, par = F.build_idle(33600, 3428.57, 'answer', 40)
        self.assertEqual(len(x), 40 * 8)
        self.assertGreater(len(set(np.round(x[120:], 3))), 20)

    def test_residual_blocks(self):
        a = np.ones(480, dtype=complex)
        self.assertEqual(F.residual_by_block(a, a), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(F.residual_by_block(a + 2, a)[0], 2.0)


class TransmitCodingTests(unittest.TestCase):
    """The data-mode transmit coding for the 16/32/64-state trellis and non-linear encoding against an independent reading of the spec."""

    def symbols(self, rate, baud, role, trellis, frames=4, nonlinear=False, bits=None):
        import v34_data as vd
        tx = vd.DataTx(rate, baud, trellis=trellis, nonlinear=nonlinear)
        n = tx.bits_per_data_frame
        bits = bits if bits is not None else B.scrambled_ones(n * frames, role)
        out = []
        for k in range(frames):
            out += tx.data_frame(bits[k * n:(k + 1) * n])
        return np.array(out), tx

    def test_transmitter_matches_the_independent_reference_for_every_trellis(self):
        for rate, baud in ((33600, 3428.57), (31200, 3428.57), (28800, 3200.0)):
            for role in ('answer', 'call'):
                for trellis in (0, 1, 2):
                    sym, tx = self.symbols(rate, baud, role, trellis)
                    x, _ = F.build_idle(rate, baud, role, tx.P * 4, trellis=trellis)
                    self.assertEqual(np.abs(sym - x).max(), 0.0, (rate, role, trellis))

    def test_the_three_encoders_produce_different_streams(self):
        xs = [F.build_idle(33600, 3428.57, 'answer', 60, trellis=t)[0] for t in (0, 1, 2)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertGreater(int((abs(xs[a] - xs[b]) > 1e-9).sum()), 50)

    def test_nonlinear_encoder_follows_equations_9_33_to_9_35(self):
        import random
        for rate, trellis in ((33600, 2), (31200, 2), (33600, 0)):
            r = random.Random(7)
            _, probe = self.symbols(rate, 3428.57, 'answer', trellis, frames=1)
            bits = [r.randrange(2) for _ in range(probe.bits_per_data_frame * 40)]
            lin, tx = self.symbols(rate, 3428.57, 'answer', trellis, frames=40, bits=bits)
            nl, _ = self.symbols(rate, 3428.57, 'answer', trellis, frames=40, nonlinear=True, bits=bits)
            zeta = 0.3125 * abs(lin) ** 2 / tx._linear_power         # average energy of the linear signal
            ref = (1 + zeta / 6 + zeta ** 2 / 120) * lin
            self.assertLess(np.abs(nl - ref).max(), 1e-9, (rate, trellis))


def par_frames(rate, baud):
    return F._PARAMS[round(baud)][1]


if __name__ == '__main__':
    unittest.main()
