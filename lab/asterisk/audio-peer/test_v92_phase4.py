"""V.92 Phase 4 analogue-side signals: SUVu, RM / RM', TRN2u."""

import math
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

import v92_cp as cp
import v92_phase3 as p3

if np is not None:
    import v34_g711 as g711
    import v92_pcm_upstream as pu


def bit_value(bits, lo, hi):
    return sum(bits[lo + i] << i for i in range(hi - lo + 1))


class SuvuTests(unittest.TestCase):
    def test_layout_matches_table_27(self):
        bits = cp.build_suvu(level_db=-1.5, wait_for_cpu=True, silent_period=True, ack=True)
        self.assertEqual(len(bits), 52)                       # fill bit at 51
        self.assertEqual(bits[:18], [1] * 17 + [0])
        self.assertEqual(bits[18], 1)                         # SUVu
        self.assertEqual(sum(bits[19:26]), 0)
        self.assertEqual((bits[26], bits[32], bits[33], bits[34], bits[51]), (1, 1, 1, 0, 0))
        self.assertEqual(bit_value(bits, 27, 31), (-6) & 0x1F)    # -1.5 dB = -6 in Q2.2
        parsed = cp.parse_suvu(bits)
        self.assertEqual((parsed["wait_for_cpu"], parsed["silent_period"], parsed["ack"]),
                         (True, True, True))
        self.assertEqual(parsed["level_db"], -1.5)

    def test_unmeasured_level_and_range(self):
        self.assertEqual(bit_value(cp.build_suvu(), 27, 31), 16)
        self.assertIsNone(cp.parse_suvu(cp.build_suvu())["level_db"])
        self.assertEqual(cp.parse_suvu(cp.build_suvu(level_db=3.75))["level_db"], 3.75)
        # -4.00 is the "no measurement" code, so a real -4.00 dB reading is clamped to -3.75.
        self.assertEqual(cp.parse_suvu(cp.build_suvu(level_db=-4.0))["level_db"], -3.75)
        with self.assertRaises(ValueError):
            cp.build_suvu(level_db=4.5)

    def test_faults_rejected(self):
        good = cp.build_suvu(level_db=1.0)
        for index in (18, 22, 30, 34, 40, 51):
            bad = list(good)
            bad[index] ^= 1
            self.assertIsNone(cp.parse_suvu(bad), index)
        self.assertIsNone(cp.parse_suvu(cp.build_cpus(drn=5)))

    def test_parser_caps_lookahead_to_one_frame(self):
        bits = cp.build_suvu(wait_for_cpu=True)
        observed = []
        original = cp._candidates

        def record(candidate_bits):
            observed.append(len(candidate_bits))
            return original(candidate_bits)

        with mock.patch.object(cp, "_candidates", side_effect=record):
            parsed = cp.parse_suvu(bits + [1] * 100_000)
        self.assertTrue(parsed["wait_for_cpu"])
        self.assertEqual(observed, [cp.SINGLE_WORD_FRAME_BITS])


class SuvdTests(unittest.TestCase):
    def test_layout_matches_table_31(self):
        bits = cp.build_suvd(silent_period=True, ack=True)
        self.assertEqual(len(bits), 52)
        self.assertEqual((bits[18], bits[32], bits[33], bits[34], bits[51]), (1, 1, 1, 0, 0))
        self.assertEqual(sum(bits[19:32]), 0)
        parsed = cp.parse_suvd(bits)
        self.assertEqual((parsed["silent_period"], parsed["ack"]), (True, True))
        self.assertIsNone(cp.parse_suvd(cp.build_suvu()))     # SUVu has a nonzero reserved field
        bad = list(bits)
        bad[40] ^= 1
        self.assertIsNone(cp.parse_suvd(bad))

    def test_parser_caps_lookahead_to_one_frame(self):
        bits = cp.build_suvd(ack=True)
        observed = []
        original = cp._candidates

        def record(candidate_bits):
            observed.append(len(candidate_bits))
            return original(candidate_bits)

        with mock.patch.object(cp, "_candidates", side_effect=record):
            parsed = cp.parse_suvd(bits + [1] * 100_000)
        self.assertTrue(parsed["ack"])
        self.assertEqual(observed, [cp.SINGLE_WORD_FRAME_BITS])


class RfTests(unittest.TestCase):
    MASKS = [tuple(range(1, 20 + 3 * i)) for i in range(6)]

    def test_rf_uses_the_highest_ucode_of_each_interval_with_the_sign_pattern(self):
        from v90_dil import g711_ucode_codewords
        for law in ("pcmu", "pcma"):
            table = g711_ucode_codewords(law)
            pattern = p3.rf_codewords(self.MASKS, law)
            self.assertEqual(len(pattern), 12)
            for i, code in enumerate(pattern):
                top = int(table[max(self.MASKS[i % 6])])
                negative = bool(code ^ top)
                self.assertEqual(code, top ^ (0x80 if negative else 0))
                self.assertEqual(negative, i % 4 >= 2)           # + + - - + + - - ...
            inverse = p3.rf_codewords(self.MASKS, law, inverted=True)
            self.assertTrue(all(a ^ b == 0x80 for a, b in zip(pattern, inverse)))
            self.assertEqual(len(p3.rf_codewords(self.MASKS, law, repeats=2)), 24)

    def test_rf_rejects_bad_constellations(self):
        with self.assertRaises(ValueError):
            p3.rf_codewords(self.MASKS[:5], "pcmu")
        with self.assertRaises(ValueError):
            p3.rf_codewords(self.MASKS[:5] + [()], "pcmu")


class Trn2uTests(unittest.TestCase):
    def test_four_point_levels_and_unit_power(self):
        symbols = p3.Trn2uModulator(4).symbols(1200)
        self.assertEqual({round(abs(v) * math.sqrt(5), 6) for v in symbols}, {1.0, 3.0})
        self.assertAlmostEqual(sum(v * v for v in symbols) / len(symbols), 1.0, delta=0.12)

    def test_eight_point_levels_and_unit_power(self):
        symbols = p3.Trn2uModulator(8).symbols(2400)
        self.assertEqual({round(abs(v) * math.sqrt(21), 6) for v in symbols}, {1.0, 3.0, 5.0, 7.0})
        self.assertAlmostEqual(sum(v * v for v in symbols) / len(symbols), 1.0, delta=0.15)

    def test_sign_is_differentially_encoded_from_the_previous_sign(self):
        for points, norm in ((4, math.sqrt(5)), (8, math.sqrt(21))):
            for initial in (0, 1):
                modulator = p3.Trn2uModulator(points, initial_sign=initial)
                symbols = modulator.symbols(240)
                signs = [1 if v < 0 else 0 for v in symbols]
                # Undo the differential encoder: the underlying sign bits must be the
                # scrambler's output for the same input.
                import v34_tx
                scrambler = v34_tx.Scrambler("answer")
                bits = 2 if points == 4 else 3
                expected, previous = [], initial
                for sign in signs:
                    label = 0
                    for position in range(bits):
                        label |= scrambler.step(1) << position
                    expected.append(((label >> (bits - 1)) & 1))
                undone, previous = [], initial
                for sign in signs:
                    undone.append(sign ^ previous)
                    previous = sign
                self.assertEqual(undone, expected, (points, initial))
                self.assertEqual(modulator.last_sign, signs[-1])

    def test_length_and_point_count_are_validated(self):
        with self.assertRaises(ValueError):
            p3.Trn2uModulator(4).symbols(10)
        with self.assertRaises(ValueError):
            p3.Trn2uModulator(16)


@unittest.skipIf(np is None, "numpy not installed")
class RmTests(unittest.TestCase):
    def profile(self):
        codes = np.array([0xFF - i for i in range(20, 120, 3)], dtype=np.uint8)
        levels = tuple(sorted({abs(float(v)) for v in g711.decode(codes, "pcmu")}))
        return pu.PcmUpstreamProfile(constellations=(levels,), interval_set=(0,) * 12,
                                     moduli=(20, 20, 20, 10) * 3, trellis=1)

    def test_rm_patterns_match_tables_25_and_26(self):
        moduli = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
        self.assertEqual(pu.rm_intervals(moduli),
                         [6, 7, 0, 0, 10, 11, 0, 0, 14, 15, 0, 0])
        self.assertEqual(pu.rm_intervals(moduli, prime=True),
                         [0, 0, 8, 9, 0, 0, 12, 13, 0, 0, 16, 17])

    def test_rm_and_rm_prime_survive_the_mapper_trellis_and_a_codec(self):
        profile = self.profile()
        tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
        for repeat in range(6):
            for prime in (False, True):
                ks = pu.rm_intervals(profile.moduli, prime=prime)
                samples = tx.encode_intervals(ks)
                heard = [float(v) for v in g711.decode(
                    g711.encode(np.rint(samples).astype(np.int32), "pcmu"), "pcmu")]
                self.assertEqual(rx.decode_intervals(heard), ks, (repeat, prime))
        self.assertEqual(rx.trellis_violations, 0)

    def test_data_after_rm_is_unaffected(self):
        profile = self.profile()
        tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
        rx.decode_intervals(tx.encode_intervals(pu.rm_intervals(profile.moduli)))
        bits = [1, 0, 1, 1, 0, 0, 1] * (profile.bits_per_frame // 7) + [0] * (profile.bits_per_frame % 7)
        self.assertEqual(rx.decode_frame(tx.encode_frame(bits)), bits)

    def test_intervals_are_validated(self):
        tx = pu.PcmUpstreamTransmitter(self.profile())
        with self.assertRaises(ValueError):
            tx.encode_intervals([0] * 11)
        with self.assertRaises(ValueError):
            tx.encode_intervals([99] + [0] * 11)


class UpstreamDecoderStartTests(unittest.TestCase):
    def test_a_lone_start_up_transient_does_not_set_the_symbol_start(self):
        import numpy as np
        import v92_phase4_controller as c
        lu = 2000.0
        for split in (1, 2, 5):
            d = c._UpstreamDecoder(4, lu)
            x = np.concatenate(([-0.26 * lu, 0.05 * lu], np.zeros(20), [1.34 * lu, -1.34 * lu, 0.45 * lu, 0.45 * lu] * 3))
            d.feed(x[:split])
            d.feed(x[split:])
            self.assertEqual(d.start_raw, 22, split)


if __name__ == "__main__":
    unittest.main()
