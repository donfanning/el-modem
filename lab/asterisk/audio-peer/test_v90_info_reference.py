"""Offline V.90 INFO parsing and waveform checks."""

import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_info as vi
    import v90_info_reference as v90


@unittest.skipIf(np is None, "numpy not installed")
class V90InfoReferenceTests(unittest.TestCase):
    def test_info0d_round_trip_and_crc(self):
        bits = v90.build_info0d(rate3429=1, disallow3429=0, power_reduction=1,
                                max_asym=4, pts1664=1, alaw=1, v90_3429_upstream=1)
        self.assertEqual(len(bits), 62)
        parsed = v90.parse_info0d(bits)
        self.assertEqual(parsed["max_asym"], 4)
        self.assertEqual(parsed["pts1664"], 1)
        self.assertEqual(parsed["alaw"], 1)
        damaged = list(bits)
        damaged[39] ^= 1
        self.assertIsNone(v90.parse_info0d(damaged))

    def test_info1a_v90_selector_is_not_v34_field_interpretation(self):
        bits = v90.build_info1a_v90(md_len=12, uinfo=90, analog_symbol_rate=5,
                                    freq_offset=-17)
        parsed = v90.parse_info1a_v90(bits)
        self.assertEqual(parsed["digital_symbol_rate"], 6)
        self.assertEqual(parsed["analog_symbol_rate"], 5)
        self.assertEqual(parsed["uinfo"], 90)
        self.assertEqual(parsed["freq_offset"], -17)
        self.assertEqual(vi.parse_info1a(bits)["sym_call"], 6)

    def test_waveform_decoder_finds_v90_info_messages(self):
        info0d = v90.build_info0d(rate3429=1, alaw=1)
        info1a = v90.build_info1a_v90(md_len=4, uinfo=100, analog_symbol_rate=4)
        digital = vi.InfoModulator("call").modulate(info0d)
        analog = vi.InfoModulator("answer").modulate(info1a)
        got0d = v90.decode_info0d(np.r_[np.zeros(400), digital, np.zeros(300)], "call")
        got1a = v90.decode_v90_info1a(np.r_[np.zeros(500), analog, np.zeros(300)])
        self.assertEqual(len(got0d), 1)
        self.assertEqual(got0d[0]["alaw"], 1)
        self.assertEqual(len(got1a), 1)
        self.assertEqual(got1a[0]["uinfo"], 100)

    def test_ja_descriptor_round_trip_and_crc_rejection(self):
        vector = v90.build_ja_descriptor(
            N=3, LSP=18, LTP=19,
            SP=[1, 0] * 9, TP=[0, 1, 1] * 6 + [0],
            H=[2, 3, 4, 5, 6, 7, 8, 9], REF=[10, 11, 12, 13, 14, 15, 16, 17],
            Ucodes=[1, 17, 33],
        )
        parsed = v90.parse_ja_descriptor(vector)
        self.assertTrue(parsed["valid"])
        self.assertEqual((parsed["N"], parsed["LSP"], parsed["LTP"]), (3, 18, 19))
        self.assertEqual(parsed["SP"], [1, 0] * 9)
        self.assertEqual(parsed["TP"], [0, 1, 1] * 6 + [0])
        self.assertEqual(parsed["Ucodes"], [1, 17, 33])
        damaged = vector.copy()
        damaged[188] ^= 1
        self.assertFalse(v90.parse_ja_descriptor(damaged)["crc_valid"])

    def test_ja_reserved_bit_50_is_included_in_crc(self):
        vector = v90.build_ja_descriptor(
            N=0, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8, REF=[0] * 8,
            Ucodes=[])
        parsed = v90.parse_ja_descriptor(vector)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["crc"], parsed["crc_expected"])
        # Table 12 calls bit 50 reserved, while V.34 §10.1.2.3.2 excludes
        # start bits (not reserved information bits) from the CRC input.
        damaged = vector.copy()
        damaged[50] = 1
        self.assertFalse(v90.parse_ja_descriptor(damaged)["crc_valid"])
        self.assertFalse(v90.parse_ja_descriptor(damaged)["reserved_bits_zero"])

    def test_ja_crc_known_descriptor_with_reserved_header_zero(self):
        # Spec-layout vector matching the recovered measurement descriptor.
        # 0x4339 is the CRC read from the repeated wire descriptor; bit 50 is
        # zero but still belongs to the CRC-covered information sequence.
        bits = v90.build_ja_descriptor(
            N=120, LSP=12, LTP=11,
            SP=vi._field(0x0A6D, 12), TP=vi._field(0x06B7, 11),
            H=[10] * 8, REF=[0] * 8,
            Ucodes=[2 + (i * 125) // 119 for i in range(120)],
        )
        parsed = v90.parse_ja_descriptor(bits)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["crc_start"], 1242)
        self.assertEqual(parsed["crc"], 0x4339)

    def test_dil_expansion_restarts_patterns_and_selects_uchord(self):
        descriptor = {
            "N": 2, "SP": [1, 0], "TP": [0, 1, 1],
            "H": [0, 0, 0, 0, 0, 0, 0, 0],
            "REF": [2, 0, 0, 0, 0, 0, 0, 0],
            "Ucodes": [1, 17],
        }
        symbols = v90.expand_dil(descriptor)
        self.assertEqual(len(symbols), 12)
        self.assertEqual([x["ucode"] for x in symbols[:6]], [2, 1, 1, 2, 1, 1])
        self.assertEqual([x["sign"] for x in symbols[:6]], [1, -1, 1, -1, 1, -1])
        self.assertEqual([x["ucode"] for x in symbols[6:]], [0, 17, 17, 0, 17, 17])

    def test_v90_ucode_order_matches_g711_table_one_prefix(self):
        self.assertEqual(v90.g711_ucode_codewords("pcmu")[:4].tolist(), [0xFF, 0xFE, 0xFD, 0xFC])
        self.assertEqual(v90.g711_ucode_codewords("pcma")[:4].tolist(), [0xD5, 0xD4, 0xD7, 0xD6])

    def test_dil_linear_samples_apply_law_and_sign(self):
        descriptor = {
            "N": 1, "SP": [1], "TP": [1], "H": [0] * 8,
            "REF": [0] * 8, "Ucodes": [1],
        }
        got = v90.dil_linear_samples(descriptor, "pcma")
        self.assertEqual(got.tolist(), [24] * 6)
        descriptor["SP"] = [0]
        self.assertEqual(v90.dil_linear_samples(descriptor, "pcma").tolist(), [-24] * 6)

    def test_dil_g711_output_preserves_negative_zero_codeword(self):
        descriptor = {
            "N": 1, "SP": [0], "TP": [1], "H": [0] * 8,
            "REF": [0] * 8, "Ucodes": [0],
        }
        self.assertEqual(v90.dil_g711_codewords(descriptor, "pcmu"), b"\x7f" * 6)
        self.assertEqual(v90.dil_linear_samples(descriptor, "pcmu").tolist(), [0] * 6)
        self.assertEqual(v90.dil_g711_codewords(descriptor, "pcma"), b"\x55" * 6)
        self.assertEqual(v90.dil_linear_samples(descriptor, "pcma").tolist(), [-8] * 6)

    def test_dil_validation_allows_only_pcmu_zero_sign_canonicalization(self):
        expected = b"\x7f\x01\xff\x80"
        self.assertTrue(v90.dil_codewords_match(
            expected, b"\xff\x01\x7f\x80", "pcmu"))
        self.assertFalse(v90.dil_codewords_match(
            expected, b"\xff\x01\x7e\x80", "pcmu"))
        self.assertFalse(v90.dil_codewords_match(
            b"\x55", b"\xd5", "pcma"))
        self.assertFalse(v90.dil_codewords_match(
            b"\x7f", b"", "pcmu"))

    def test_dil_codeword_mapping_rejects_unknown_law(self):
        with self.assertRaisesRegex(ValueError, "pcmu.*pcma"):
            v90.dil_g711_codewords({"N": 0, "SP": [], "TP": [], "H": [0] * 8,
                                    "REF": [0] * 8, "Ucodes": []}, "linear")

    def test_dil_stream_repeats_whole_sequence_and_stops_at_segment_boundary(self):
        descriptor = v90.parse_ja_descriptor(v90.build_ja_descriptor(
            N=2, LSP=1, LTP=1, SP=[1], TP=[1], H=[0] * 8,
            REF=[0] * 8, Ucodes=[1, 17]))
        stream = v90.V90DILCodewordStream(descriptor, "pcmu")
        segments = v90.dil_g711_segments(descriptor, "pcmu")
        self.assertEqual(len(segments), 2)
        self.assertEqual(stream.read(6), segments[0])
        self.assertEqual(stream.read(6), segments[1])
        self.assertEqual(stream.completed_passes, 1)
        self.assertEqual(stream.read(6), segments[0])
        self.assertEqual(stream.read(2), segments[1][:2])
        stream.request_stop()
        self.assertEqual(stream.read(6), segments[1][2:])
        self.assertEqual(stream.read(6), b"")
        self.assertEqual(stream.read(6), b"")

    def test_dil_stream_stop_request_at_boundary_does_not_start_next_segment(self):
        descriptor = v90.parse_ja_descriptor(v90.build_ja_descriptor(
            N=2, LSP=1, LTP=1, SP=[1], TP=[1], H=[0] * 8,
            REF=[0] * 8, Ucodes=[1, 17]))
        stream = v90.V90DILCodewordStream(descriptor, "pcma")
        first_segment = v90.dil_g711_segments(descriptor, "pcma")[0]
        self.assertEqual(stream.read(6), first_segment)
        stream.request_stop()
        self.assertEqual(stream.read(6), b"")


if __name__ == "__main__":
    unittest.main()
