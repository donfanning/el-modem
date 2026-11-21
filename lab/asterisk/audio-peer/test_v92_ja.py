"""V.92 Ja descriptor (Table 20)."""

import unittest

try:
    import numpy as np  # noqa: F401
except ImportError:
    np = None

if np is not None:
    import v34_info as vi
    import v90_info_reference as v90
    import v92_ja as ja


@unittest.skipIf(np is None, "numpy not installed")
class V92JaTests(unittest.TestCase):
    def test_n_zero_descriptor_is_276_bits(self):
        bits = ja.build_ja_descriptor(N=0)
        self.assertEqual(len(bits), 276)                   # the length V.92 states
        self.assertEqual(bits[:18], [1] * 17 + [0])

    def test_mask_words_sit_between_the_terminal_start_bit_and_the_crc(self):
        rates = [24000, 44000, 45333.333333333336, 48000]
        bits = ja.build_ja_descriptor(N=0, upstream_rates=rates)
        t = 221                                            # 187 + beta (34) + 0
        self.assertEqual(bits[t], 0)
        self.assertEqual(bits[t + 1], 1)                   # 24,000 -> bit 188
        self.assertEqual(bits[t + 16], 1)                  # 44,000 -> bit 203
        self.assertEqual(bits[t + 17], 0)                  # start bit 204
        self.assertEqual((bits[t + 18], bits[t + 20]), (1, 1))   # 45,333 and 48,000 -> 205, 207
        self.assertEqual(sum(bits[t + 21:t + 34]), 0)      # reserved
        self.assertEqual((bits[t + 34], bits[t + 51]), (0, 0))   # start 221, fill 238

    def test_round_trip_with_tables(self):
        fields = dict(N=7, LSP=19, LTP=31, SP=[1, 0] * 9 + [1], TP=[0, 1] * 15 + [1],
                      H=[3, 5, 7, 9, 11, 13, 15, 17], REF=[1, 2, 3, 4, 5, 6, 7, 8],
                      Ucodes=[10, 20, 30, 40, 50, 60, 70])
        rates = [28000 - 4000 + 8000 / 6 * k for k in (0, 3, 9, 14, 18)]
        parsed = ja.parse_ja_descriptor(ja.build_ja_descriptor(upstream_rates=rates, **fields))
        self.assertEqual((parsed["N"], parsed["LSP"], parsed["LTP"]), (7, 19, 31))
        self.assertEqual(parsed["Ucodes"], fields["Ucodes"])
        self.assertEqual([round(r, 3) for r in parsed["upstream_rates"]],
                         [round(r, 3) for r in rates])

    def test_descriptor_prefix_provides_exact_length_for_variable_dil_counts(self):
        for ucodes in ((), (37,), (37, 82), (1, 18, 35, 52, 69, 86, 103)):
            with self.subTest(count=len(ucodes)):
                bits = ja.build_ja_descriptor(N=len(ucodes), Ucodes=list(ucodes),
                                              upstream_rates=(28000, 32000, 36000))
                parsed = ja.parse_ja_descriptor(bits)
                self.assertEqual(ja.descriptor_bits_from_prefix(bits[:50]), len(bits))
                self.assertEqual(parsed["bits"], len(bits))
                self.assertEqual(parsed["Ucodes"], list(ucodes))

    def test_crc_and_framing_faults_are_rejected(self):
        good = ja.build_ja_descriptor(N=2, Ucodes=[5, 6], upstream_rates=[24000 + 8000 / 6 * 4])
        self.assertIsNotNone(ja.parse_ja_descriptor(good))
        for index in (5, 20, 30, 150, 230, 240, 250, 260):
            bad = list(good)
            bad[index] ^= 1
            self.assertIsNone(ja.parse_ja_descriptor(bad), index)
        self.assertIsNone(ja.parse_ja_descriptor(good[:-12]))

    def test_v90_and_v92_descriptors_are_distinct(self):
        v90_bits = v90.build_ja_descriptor(N=0)
        v92_bits = ja.build_ja_descriptor(N=0)
        self.assertIsNone(ja.parse_ja_descriptor(v90_bits))
        read_as_v90 = v90.parse_ja_descriptor(v92_bits)
        self.assertTrue(read_as_v90 is None or not read_as_v90["valid"])

    def test_geometry_matches_the_v90_builders_crc(self):
        # The start-bit set used here must equal the V.90 builder's: recompute its CRC.
        fields = dict(N=5, LSP=19, LTP=31)
        base = v90.build_ja_descriptor(**fields)
        terminal, starts = ja._geometry(5, 19, 31)
        information = [base[i] for i in range(18, terminal) if i not in starts]
        self.assertEqual(sum(b << i for i, b in enumerate(base[terminal + 1:terminal + 17])),
                         vi.crc16(information))

    def test_repeated_compact_descriptors_require_crc_valid_majority(self):
        descriptor = ja.build_ja_descriptor(N=7, LSP=19, LTP=31,
                                            Ucodes=[10, 20, 30, 40, 50, 60, 70],
                                            upstream_rates=(24000, 32000, 40000))
        periods = ja.descriptor_repeat_lengths_from_prefix(descriptor[:50])
        self.assertEqual(periods[-1], len(descriptor))
        self.assertLess(periods[0], len(descriptor))
        compact = periods[0]
        copies = []
        for copy_index in range(5):
            frame = list(descriptor[:compact])
            # Distinct errors in each copy are repaired by majority.  No
            # individual damaged descriptor should pass the ordinary CRC gate.
            for bit_index in (100 + copy_index, 220 + 2 * copy_index):
                frame[bit_index] ^= 1
            self.assertIsNone(ja.parse_ja_descriptor(frame + [0] * (len(descriptor) - len(frame))))
            copies.extend(frame)
        parsed = ja.parse_repeated_ja_descriptor(copies, repeat_bits=compact, repetitions=5)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["upstream_rate_mask"], ja.rate_mask((24000, 32000, 40000)))
        self.assertEqual(parsed["repetitions"], 5)
        self.assertEqual(parsed["repeat_bits"], compact)

    def test_repeated_descriptor_rejects_bad_majority_crc(self):
        descriptor = ja.build_ja_descriptor(N=0, upstream_rates=(24000, 32000))
        frames = []
        for _ in range(5):
            frame = list(descriptor)
            frame[100] ^= 1
            frames.extend(frame)
        self.assertIsNone(ja.parse_repeated_ja_descriptor(
            frames, repeat_bits=len(descriptor), repetitions=5))

    def test_rate_validation(self):
        with self.assertRaises(ValueError):
            ja.rate_mask([23999])
        with self.assertRaises(ValueError):
            ja.rate_mask([49333])
        self.assertEqual(ja.rate_mask([24000, 48000]), 1 | (1 << 18))


if __name__ == "__main__":
    unittest.main()
