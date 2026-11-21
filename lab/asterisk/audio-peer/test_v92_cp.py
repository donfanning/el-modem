"""V.92 CPu/CPt/CPus/CPd framing (ITU-T V.92 Tables 23, 24, 30)."""

import random
import unittest
from unittest import mock

import v92_cp as cp


def bit_value(bits, lo, hi):
    return sum(bits[lo + i] << i for i in range(hi - lo + 1))


UCODES_A = tuple(range(1, 40))
UCODES_B = tuple(range(10, 100, 3))
UCODES_C = (5, 6, 7, 120)


class CpuTests(unittest.TestCase):
    def build(self, **kw):
        base = dict(drn=22, constellations=[UCODES_A], interval_sets=(0,) * 6)
        base.update(kw)
        return cp.build_cpu(**base)

    def test_single_constellation_layout_matches_table_23(self):
        bits = self.build(ack=True, alaw=True, sr=2, lookahead=1, kind=cp.CPU)
        self.assertEqual(len(bits), 290)                  # 289 + delta (0), plus the fill bit
        self.assertEqual(bits[:18], [1] * 17 + [0])
        self.assertEqual(bit_value(bits, 19, 20), 1)      # type CPu
        self.assertEqual(bit_value(bits, 21, 25), 22)     # drn
        self.assertEqual(bit_value(bits, 31, 32), 2)      # Sr
        self.assertEqual((bits[33], bits[35]), (1, 1))    # ack, A-law
        self.assertEqual(bit_value(bits, 49, 50), 1)      # ld
        self.assertEqual(bits[128], 0)
        for start in (34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 272):
            self.assertEqual(bits[start], 0, start)
        self.assertEqual(bits[137], 0)                    # Ucode 0 is not in the set
        self.assertEqual(bit_value(bits, 137, 152) >> 1 & 1, 1)   # Ucode 1 is

    def test_round_trip_cpu_with_several_constellations_and_codec_maps(self):
        sets = [UCODES_A, UCODES_B, UCODES_C]
        for codec in (None, [UCODES_B, UCODES_C, UCODES_A]):
            bits = self.build(constellations=sets, interval_sets=(0, 1, 2, 0, 1, 2),
                              codec_constellations=codec, shaping=(10, -3, 0, 5),
                              trn1d_rms_ratio=0x2468, drn=9)
            parsed = cp.parse_cpu(bits)
            self.assertEqual(parsed["kind"], "cpu")
            self.assertEqual(parsed["interval_sets"], (0, 1, 2, 0, 1, 2))
            self.assertEqual(parsed["constellations"], tuple(map(tuple, sets)))
            self.assertEqual(parsed["codec_constellations"],
                             None if codec is None else tuple(map(tuple, codec)))
            self.assertEqual(parsed["shaping"], (10 / 64, -3 / 64, 0.0, 5 / 64))
            self.assertAlmostEqual(parsed["trn1d_rms_ratio"], 0x2468 / 8192)
            self.assertEqual(parsed["frame_bits"], len(bits))
            delta = 272 if codec is None else 2 * 272 + 136
            self.assertEqual(len(bits), 290 + delta)

    def test_rates_follow_the_table(self):
        self.assertAlmostEqual(cp.parse_cpu(self.build(kind=cp.CPU, drn=22))["rate"], 56000)
        self.assertAlmostEqual(cp.parse_cpu(self.build(kind=cp.CPT, drn=9))["rate"],
                               17 * 8000 / 6)
        self.assertTrue(cp.parse_cpu(self.build(drn=0))["cleardown"])

    def test_faults_are_rejected(self):
        good = self.build()
        self.assertIsNotNone(cp.parse_cpu(good))
        for index in (3, 17, 34, 40, 105, 137, 280):      # data, start and reserved bits
            bad = list(good)
            bad[index] ^= 1
            self.assertIsNone(cp.parse_cpu(bad), index)
        self.assertIsNone(cp.parse_cpu(good[:-1]))        # missing fill bit
        self.assertIsNone(cp.parse_cpu(self.build(drn=0)[:50]))

    def test_scanner_lookahead_is_capped_to_largest_legal_cpu(self):
        masks = [(i,) for i in range(12)]
        bits = self.build(constellations=masks[:6], codec_constellations=masks[6:],
                          interval_sets=(0, 1, 2, 3, 4, 5))
        self.assertEqual(len(bits), cp.MAX_CPU_FRAME_BITS)
        observed_lengths = []
        original = cp._candidates

        def record(candidate_bits):
            observed_lengths.append(len(candidate_bits))
            return original(candidate_bits)

        with mock.patch.object(cp, "_candidates", side_effect=record):
            parsed = cp.parse_cpu(bits + [1] * 100_000)
        self.assertEqual(parsed["frame_bits"], len(bits))
        self.assertEqual(observed_lengths, [cp.MAX_CPU_FRAME_BITS])


class CpusTests(unittest.TestCase):
    def test_layout_and_round_trip(self):
        bits = cp.build_cpus(drn=21, ack=True)
        self.assertEqual(len(bits), 52)                   # fill bit at 51
        self.assertEqual(bit_value(bits, 19, 20), 2)
        self.assertEqual(bit_value(bits, 21, 25), 21)
        self.assertEqual(bits[33], 1)
        parsed = cp.parse_cpus(bits)
        self.assertEqual((parsed["drn"], parsed["ack"]), (21, 1))
        self.assertAlmostEqual(parsed["rate"], 41 * 8000 / 6)
        bad = list(bits)
        bad[40] ^= 1
        self.assertIsNone(cp.parse_cpus(bad))
        self.assertIsNone(cp.parse_cpu(bits))             # not a CPu

    def test_single_word_parser_bounds_scanner_lookahead(self):
        frame = cp.build_cpus(drn=7)
        observed_lengths = []
        original = cp._candidates

        def record(candidate_bits):
            observed_lengths.append(len(candidate_bits))
            return original(candidate_bits)

        with mock.patch.object(cp, "_candidates", side_effect=record):
            parsed = cp.parse_cpus(frame + [1] * 100_000)
        self.assertEqual(parsed["drn"], 7)
        self.assertEqual(observed_lengths, [cp.SINGLE_WORD_FRAME_BITS])


class CpdTests(unittest.TestCase):
    MODULI = (40, 40, 30, 20) * 3
    PRE = dict(z1=[3000, -1000], p1=[8000, -4000, 100], z2=[32767, 5], p2=[])

    def build(self, **kw):
        base = dict(drn=15, trellis=2, gain4=40000, moduli=self.MODULI, precoder=self.PRE,
                    interval_sets=(0, 1, 2, 0, 1, 2),
                    constellations=[[10, 20, 35], [11, 22, 33, 44], [8, 9]])
        base.update(kw)
        return cp.build_cpd(**base)

    def test_layout_matches_table_30(self):
        bits = self.build(ack=True, extend_e2u=True)
        self.assertEqual(bits[:18], [1] * 17 + [0])
        self.assertEqual((bits[19], bits[20], bits[21]), (1, 1, 1))
        self.assertEqual(bit_value(bits, 22, 26), 15)
        self.assertEqual(bit_value(bits, 27, 28), 2)
        self.assertEqual((bits[29], bits[33]), (1, 1))
        self.assertEqual(bit_value(bits, 35, 50), 40000)
        self.assertEqual(bit_value(bits, 52, 59), 40)            # M0
        self.assertEqual(bit_value(bits, 120, 127), 40)          # M8
        self.assertEqual(bit_value(bits, 154, 162), 2)           # LZ1
        self.assertEqual(bit_value(bits, 171, 179), 3)           # LP1
        self.assertEqual(bit_value(bits, 188, 196), 2)           # LZ2
        self.assertEqual(bit_value(bits, 205, 213), 0)           # LP2
        alpha = 17 * (2 + 3 + 2 + 0)
        self.assertEqual(bit_value(bits, 222, 237), 3000)        # z1(1), Q0.15
        self.assertEqual(bits[221 + alpha], 0)
        self.assertEqual(bit_value(bits, 222 + alpha, 225 + alpha), 0)    # intervals 0 & 6
        self.assertEqual(bit_value(bits, 226 + alpha, 229 + alpha), 1)
        self.assertEqual(bit_value(bits, 256 + alpha, 263 + alpha), 3)    # LC1
        self.assertEqual(bit_value(bits, 264 + alpha, 271 + alpha), 4)    # LC2
        self.assertEqual(bit_value(bits, 307 + alpha, 322 + alpha), 10)   # first point
        beta = 17 * (3 + 4 + 2)
        self.assertEqual(len(bits), 324 + alpha + beta)                   # fill bit at 323+a+b

    def test_round_trip_with_all_parts(self):
        parsed = cp.parse_cpd(self.build())
        self.assertEqual(parsed["moduli"], self.MODULI)
        self.assertEqual(parsed["trellis"], 2)
        self.assertAlmostEqual(parsed["rate"], (15 + 17) * 8000 / 6)
        self.assertAlmostEqual(parsed["gain"], 40000 / 65536 / 4)
        self.assertEqual(parsed["precoder"]["z1"], (3000 / 32768, -1000 / 32768))
        self.assertEqual(parsed["precoder"]["p1"], (8000 / 16384, -4000 / 16384, 100 / 16384))
        self.assertEqual(parsed["precoder"]["p2"], ())
        self.assertEqual(parsed["interval_sets"], (0, 1, 2, 0, 1, 2))
        self.assertEqual(parsed["constellations"][:3], ((10, 20, 35), (11, 22, 33, 44), (8, 9)))

    def test_each_part_is_optional_and_removed_when_absent(self):
        full = self.build()
        without_moduli = self.build(moduli=None)
        self.assertEqual(len(full) - len(without_moduli), 6 * 17)
        self.assertEqual(without_moduli[19], 0)
        parsed = cp.parse_cpd(without_moduli)
        self.assertNotIn("moduli", parsed)
        self.assertIn("constellations", parsed)
        minimal = self.build(moduli=None, precoder=None, constellations=None, interval_sets=None)
        parsed = cp.parse_cpd(minimal)
        self.assertEqual(set(parsed) & {"moduli", "precoder", "constellations"}, set())

    def test_faults_and_constraints(self):
        good = self.build()
        for index in (23, 34, 60, 160, 300):
            bad = list(good)
            bad[index] ^= 1
            self.assertIsNone(cp.parse_cpd(bad), index)
        with self.assertRaises(ValueError):
            self.build(constellations=[[10, 20], [], [5]], interval_sets=(0, 0, 2, 0, 0, 2))
        with self.assertRaises(ValueError):
            self.build(constellations=[list(range(1, 66))], interval_sets=(0,) * 6)
        with self.assertRaises(ValueError):
            self.build(drn=20)
        with self.assertRaises(ValueError):
            self.build(interval_sets=(0, 1, 2, 0, 1, 3))      # selects an empty set

    def test_cpd_drives_the_mapper_end_to_end(self):
        import numpy as np
        import v34_g711 as g711
        import v92_pcm_upstream as pu
        codes = np.array([0xFF - i for i in range(1, 49)], dtype=np.uint8)
        levels = [int(v) for v in sorted({abs(int(v)) for v in g711.decode(codes, "pcmu")})]
        bits = cp.build_cpd(drn=8, trellis=1, gain4=0x4000, moduli=(30, 30, 30, 15) * 3,
                            interval_sets=(0,) * 6, constellations=[levels],
                            precoder=dict(z1=[], p1=[], z2=[32767], p2=[]))
        profile = pu.profile_from_cpd(cp.parse_cpd(bits))
        self.assertAlmostEqual(profile.gain, 0x4000 / 65536 / 4, places=9)
        tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
        rng = random.Random(9)
        for _ in range(20):
            data = [rng.randint(0, 1) for _ in range(profile.bits_per_frame)]
            self.assertEqual(rx.decode_frame([v / profile.gain for v in tx.encode_frame(data)]), data)


if __name__ == "__main__":
    unittest.main()


class CpdGainTests(unittest.TestCase):
    def test_with_cpd_gain_replaces_only_the_gain_field(self):
        bits = cp.build_cpd(drn=3, trellis=1, gain4=0x4000, ack=True, moduli=[12, 12, 12, 6] * 3,
                            precoder={"z1": [100], "p1": [], "z2": [32767, -200], "p2": []},
                            interval_sets=(0,) * 6, constellations=[list(range(100, 1300, 100))])
        regained = cp.with_cpd_gain(bits, 261)
        self.assertEqual(len(regained), len(bits))
        before, after = cp.parse_cpd(bits), cp.parse_cpd(regained)
        self.assertEqual(after["gain"], 261 / 65536 / 4)
        self.assertEqual({k: v for k, v in after.items() if k != "gain"},
                         {k: v for k, v in before.items() if k != "gain"})
        self.assertEqual(cp.with_cpd_gain(regained, 0x4000), bits)

    def test_with_cpd_gain_rejects_other_frames_and_a_zero_gain(self):
        bits = cp.build_cpd(drn=3, gain4=0x4000)
        with self.assertRaises(ValueError):
            cp.with_cpd_gain(bits, 0)
        with self.assertRaises(ValueError):
            cp.with_cpd_gain(cp.build_cpu(drn=22, constellations=[UCODES_A], interval_sets=(0,) * 6), 261)
