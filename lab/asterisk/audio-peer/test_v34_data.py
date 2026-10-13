#!/usr/bin/env python3
"""V.34 data-mode transmitter parameters and mapper checks (offline)."""
import random
import unittest

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

if numpy is not None:
    import v34_data as d


@unittest.skipIf(numpy is None, "numpy not installed")
class FramingTests(unittest.TestCase):
    def test_b_and_swp_match_table_8(self):
        for rate, baud, b, swp in ((4800, 3429, 12, "0421"), (9600, 3429, 23, "14A5"), (4800, 3000, 13, "3DEF"),
                                   (7200, 2800, 21, "15AB"), (9600, 3000, 26, "2D6B"), (2600, 2400, 9, "06DB"),
                                   (4800, 2400, 16, "0FFF"), (9600, 3200, 24, "FFFF")):
            fp = d.frame_params(rate, baud)
            pat = d.swp_pattern(fp["P"], fp["r"])
            self.assertEqual((fp["b"], int("".join(map(str, pat)), 2)), (b, int(swp, 16)), (rate, baud))

    def test_mapping_parameters(self):
        self.assertEqual(d.mapping_params(16, True), {"K": 4, "M": 2, "q": 0, "L": 8})
        self.assertEqual(d.mapping_params(34, False)["K"], 22)


@unittest.skipIf(numpy is None, "numpy not installed")
class MapperTests(unittest.TestCase):
    def test_cached_power_is_exact_and_does_not_share_encoder_state(self):
        h=d.precoder_from_mp([129,188,-297,-160,222,-69])
        cached=d.DataTx._estimate_power_cached
        cached.cache_clear()
        expected=cached.__wrapped__(33600,3429,True,tuple(h),False,1.,0)
        first=d.DataTx(33600,3429,expanded=True,precoder=h)
        self.assertEqual(first.avg_power,expected)
        first.data_frame([1]*first.bits_per_data_frame)
        hits=cached.cache_info().hits
        fresh=d.DataTx(33600,3429,expanded=True,precoder=list(h))
        self.assertEqual(fresh.avg_power,expected)
        self.assertGreater(cached.cache_info().hits,hits)
        self.assertEqual(fresh.x_hist,[0j]*3)
        self.assertEqual(fresh.conv.d,[0]*4)
        self.assertEqual(fresh.m4,0)
        alternate=d.DataTx._estimate_power(33600,3429,True,[0j]*3)
        self.assertEqual(alternate,cached.__wrapped__(33600,3429,True,(0j,0j,0j),False,1.,0))
        # Equal scalar values of differing precision need separate cache keys:
        # NumPy scalar arithmetic can differ from Python complex arithmetic.
        narrow=tuple(numpy.complex64(v) for v in h)
        narrow_expected=cached.__wrapped__(33600,3429,True,narrow,False,1.,0)
        self.assertEqual(d.DataTx._estimate_power(33600,3429,True,narrow),narrow_expected)

    def test_nonlinear_projection_preserves_precoder_history_and_normalizes_power(self):
        h = d.precoder_from_mp([382, 11, -43, 33, 72, 11])
        linear = d.DataTx(33600, 3429, expanded=True, precoder=h)
        nonlinear = d.DataTx(33600, 3429, expanded=True, precoder=h, nonlinear=True)
        rng = random.Random(29)
        energy, count = 0.0, 0
        for _ in range(40):
            bits = [rng.randrange(2) for _ in range(linear.bits_per_data_frame)]
            raw = linear.data_frame(bits)
            projected = nonlinear.data_frame(bits)
            for x, y in zip(raw, projected):
                zeta = 0.3125 * abs(x) ** 2 / linear.avg_power
                expected = x * (1 + zeta / 6 + zeta ** 2 / 120)
                self.assertAlmostEqual(abs(y - expected), 0, places=10)
                energy += abs(y) ** 2
                count += 1
            self.assertEqual(linear.x_hist, nonlinear.x_hist)
        self.assertGreater(nonlinear.avg_power, linear.avg_power)
        self.assertAlmostEqual(energy / count / nonlinear.avg_power, 1.0, delta=0.05)

    def test_quarter_labels_follow_figure_5(self):
        q = d.QUARTER
        self.assertEqual([q[0], q[1], q[2], q[3], q[4], q[5]],
                         [1 + 1j, -3 + 1j, 1 - 3j, -3 - 3j, 1 + 5j, 5 + 1j])
        self.assertEqual(len(set(q)), 416)

    def test_subset_labels_follow_figure_9(self):
        self.assertEqual([d.subset_label(complex(x, y)) for x, y in ((1, 1), (3, 1), (-3, 3), (-1, -1), (3, 3), (-3, -3))],
                         [0b000, 0b111, 0b001, 0b010, 0b010, 0b000])

    def test_shell_mapper_is_injective_and_in_range(self):
        for M, K in ((2, 4), (3, 12), (7, 22)):
            sm = d.ShellMapper(M)
            rnd = random.Random(M)
            seen = {}
            for _ in range(400):
                r0 = rnd.randrange(2 ** K)
                idx = tuple(sm.indices(r0))
                self.assertTrue(all(0 <= v < M for v in idx))
                self.assertEqual(seen.setdefault(idx, r0), r0)

    def test_shell_rank_boundaries_against_independent_polynomial_oracle(self):
        from v34_b1_reference import shell_unrank
        from v34_datarx import inverse_shell
        # Exhaust small alphabets; larger ones exercise every energy boundary,
        # zero-count blocks and ranks beyond 32-bit integer range.
        for m in (1, 2, 3, 7, 16, 20):
            sm = d.ShellMapper(m)
            if m <= 3:
                ranks = range(m ** 8)
            else:
                ranks = {0, m ** 8 - 1}
                for boundary in sm.z8:
                    ranks.update(r for r in (boundary - 1, boundary, boundary + 1)
                                 if 0 <= r < m ** 8)
                rnd = random.Random(m)
                ranks.update(rnd.randrange(m ** 8) for _ in range(256))
            for rank in ranks:
                with self.subTest(m=m, rank=rank):
                    rings = sm.indices(rank)
                    self.assertEqual(rings, shell_unrank(m, rank))
                    self.assertEqual(inverse_shell(sm, rings), rank)

    def test_a_data_frame_has_the_right_size_and_a_sane_constellation(self):
        tx = d.DataTx(4800, 2400)
        rnd = random.Random(2)
        sy = tx.data_frame([rnd.randrange(2) for _ in range(tx.bits_per_data_frame)])
        self.assertEqual(len(sy), 96)
        self.assertLessEqual(len(set(sy)), 8)
        self.assertAlmostEqual(tx.avg_power, 3.5, delta=0.3)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(numpy is None, "numpy not installed")
class RoundTripTests(unittest.TestCase):
    def test_decoder_inverts_the_transmitter_at_several_rates(self):
        import v34_datarx as dr
        for rate, baud, expanded in ((4800, 2400, True), (9600, 3200, False), (14400, 3429, False), (28800, 3429, True),
                                     (4800, 3429, False), (2400, 2400, False)):
            with self.subTest(rate=rate, baud=baud):
                tx = d.DataTx(rate, baud, expanded)
                rx = dr.DataRx(rate, baud, expanded)
                rnd = random.Random(rate + baud)
                for _ in range(3):                                   # three data frames, state carried over
                    bits = [rnd.randrange(2) for _ in range(tx.bits_per_data_frame)]
                    sy = tx.data_frame(bits)
                    got, pos = [], 0
                    for f in range(tx.P):
                        b, worst = rx.decode_frame(sy[8 * f:8 * f + 8])
                        got += b
                        self.assertLess(worst, 1e-9)
                    self.assertEqual(got, bits)
                self.assertEqual(rx.errors, 0)


@unittest.skipIf(numpy is None, "numpy not installed")
class PrecoderTests(unittest.TestCase):
    def test_precoder_coefficients_from_mp(self):
        h = d.precoder_from_mp([5393, 99, -131, 12, 1335, 215])
        self.assertAlmostEqual(h[0].real, 5393 / 16384)
        self.assertAlmostEqual(h[2].imag, 215 / 16384)

    def test_rounding_ties_go_to_the_smaller_magnitude(self):
        self.assertEqual(d._round_half_to_smaller(1.0, 2.0), 0.0)        # exactly half way between 0 and 2
        self.assertEqual(d._round_half_to_smaller(-1.0, 2.0), 0.0)
        self.assertEqual(d._round_half_to_smaller(3.0, 2.0), 2.0)
        self.assertEqual(d._round_half_to_smaller(1.01, 2.0), 2.0)

    def test_zero_precoder_leaves_symbols_on_the_grid_and_nonzero_one_does_not(self):
        rnd = random.Random(4)
        for h, on_grid in (([0j, 0j, 0j], True), ([0.33 + 0.01j, -0.008j, 0.08 + 0.013j], False)):
            tx = d.DataTx(9600, 3000, True, precoder=h)
            bits = [rnd.randrange(2) for _ in range(tx.bits_per_data_frame)]
            sy = tx.data_frame(bits)
            grid = all(abs(v.real - round(v.real)) < 1e-9 and abs(v.imag - round(v.imag)) < 1e-9 for v in sy)
            self.assertEqual(grid, on_grid)
            self.assertEqual(len(sy), 4 * tx.P * 2)
