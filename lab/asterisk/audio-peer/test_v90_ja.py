"""Incremental V.90 Ja scanner tests."""

import unittest

import v90_info_reference as v90
from v90_ja import V90JaScanner, ja_descriptor_bits_required


class V90JaScannerTests(unittest.TestCase):
    def descriptor_bits(self):
        return v90.build_ja_descriptor(
            N=8, LSP=33, LTP=17, SP=[1, 0] * 16 + [1],
            TP=[0, 1] * 8 + [0], H=[0, 1, 2, 3, 4, 5, 6, 7],
            REF=[7, 6, 5, 4, 3, 2, 1, 0], Ucodes=list(range(8)))

    def test_scans_descriptor_across_arbitrary_chunk_boundaries(self):
        bits = [0, 1, 0] + self.descriptor_bits() + [1, 0, 0]
        scanner = V90JaScanner()
        found = []
        cuts = (1, 17, 50, 52, 101, 205, 271, 400, len(bits))
        start = 0
        for end in cuts:
            if end > start:
                found.extend(scanner.feed(bits[start:end]))
                start = end
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["valid"])
        self.assertEqual(found[0]["start_bit"], 3)
        self.assertEqual(found[0]["Ucodes"], list(range(8)))

    def test_rejects_bad_crc_and_invalid_input(self):
        bits = self.descriptor_bits()
        bits[-3] ^= 1
        scanner = V90JaScanner()
        self.assertEqual(scanner.feed(bits), [])
        with self.assertRaisesRegex(ValueError, "only bits"):
            scanner.feed([0, 1, 2])

    def test_finds_consecutive_descriptors_without_duplicates(self):
        frame = self.descriptor_bits()
        scanner = V90JaScanner()
        found = scanner.feed(frame + frame)
        self.assertEqual([item["start_bit"] for item in found], [0, len(frame)])
        self.assertEqual(scanner.feed([]), [])

    def test_header_length_matches_variable_descriptor_and_rejects_bad_header(self):
        frame = self.descriptor_bits()
        # The fixture builder pads odd-length descriptors to an even bit
        # count; the header-derived parser length excludes that optional pad.
        self.assertEqual(ja_descriptor_bits_required(frame[:51]), len(frame) - 1)
        malformed = frame[:51]
        malformed[17] = 1
        self.assertIsNone(ja_descriptor_bits_required(malformed))


if __name__ == "__main__":
    unittest.main()
