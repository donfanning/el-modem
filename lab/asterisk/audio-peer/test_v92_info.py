"""V.92 INFO bits (ITU-T V.92 Tables 15-19)."""

import unittest

try:
    import numpy as np  # noqa: F401  (v90_info_reference imports it)
except ImportError:
    np = None

if np is not None:
    import v34_info as vi
    import v90_info_reference as v90
    import v92_info as v92


@unittest.skipIf(np is None, "numpy not installed")
class V92InfoTests(unittest.TestCase):
    def test_info0d_flags_round_trip(self):
        for v, s in ((0, 0), (1, 0), (0, 1), (1, 1)):
            bits = v92.build_info0d(v92=bool(v), short_phase2=bool(s), rate3429=1)
            flags = v92.info0d_flags(v90.parse_info0d(bits))
            self.assertEqual(flags, {"v92": bool(v), "short_phase2": bool(s)})

    def test_info0d_bit_positions_match_table_15(self):
        bits = v92.build_info0d(v92=True)
        self.assertEqual((bits[26], bits[27]), (0, 1))     # bit 27 = V.92
        bits = v92.build_info0d(short_phase2=True)
        self.assertEqual((bits[26], bits[27]), (1, 0))     # bit 26 = short Phase 2

    def test_info0a_bit_positions_match_table_16(self):
        bits = v92.build_info0a(v92=True)
        self.assertEqual((bits[26], bits[27]), (1, 0))     # bit 26 = V.92
        bits = v92.build_info0a(short_phase2=True)
        self.assertEqual((bits[26], bits[27]), (0, 1))     # bit 27 = short Phase 2
        parsed = v92.parse_info0a(v92.build_info0a(v92=True, short_phase2=True))
        self.assertEqual(v92.info0a_flags(parsed), {"v92": True, "short_phase2": True})

    def test_info0a_keeps_v90_table_8_fields_outside_the_v92_bits(self):
        # The caller profile and Phase 2 TX use V.90 names; with V.92 on, the
        # INFO0d acknowledgement and bit 19 must survive exactly as in V.90.
        fields = dict(rate3429=1, disallow3429=1, lo3000=1, hi3200=1, ack_info0d=1)
        bits = v92.build_info0a(v92=True, **fields)
        plain = v90.build_info0a(**fields)
        self.assertEqual(bits[12:26] + bits[28:29], plain[12:26] + plain[28:29])
        self.assertEqual((bits[19], bits[26], bits[28]), (1, 1, 1))
        self.assertEqual(v92.build_info0a(**fields), plain)
        with self.assertRaises(ValueError):
            v92.build_info0a(v92=True, ack_infoOd=1)
        with self.assertRaises(ValueError):
            v92.build_info0a(clock=3)

    def test_v90_info0a_parser_still_rejects_the_v92_bits(self):
        self.assertIsNone(v90.parse_info0a(v92.build_info0a(v92=True)))
        self.assertIsNotNone(v90.parse_info0a(v92.build_info0a()))

    def test_negotiation_needs_both_capability_bits(self):
        d = lambda **k: v90.parse_info0d(v92.build_info0d(**k))
        a = lambda **k: v92.parse_info0a(v92.build_info0a(**k))
        self.assertEqual(v92.negotiate(d(v92=True), a(v92=True)),
                         {"v92": True, "short_phase2": False})
        self.assertEqual(v92.negotiate(d(v92=True, short_phase2=True),
                                       a(v92=True, short_phase2=True)),
                         {"v92": True, "short_phase2": True})
        # One side without V.92: the V.90 bits apply, and short Phase 2 is not usable.
        self.assertEqual(v92.negotiate(d(short_phase2=True), a(v92=True, short_phase2=True)),
                         {"v92": False, "short_phase2": False})
        self.assertEqual(v92.negotiate(d(v92=True, short_phase2=True), a(v92=True)),
                         {"v92": True, "short_phase2": False})

    def test_info1d_pcm_upstream_flag_is_bit_70(self):
        blocks = [{}] * 5 + [{"hi_carrier": 1, "preemph": 6, "max_rate": 13}]
        bits = vi.build_info1c(blocks=blocks)
        self.assertEqual(bits[70], 1)
        self.assertTrue(v92.pcm_upstream_supported(v90.parse_info1d(bits)))
        self.assertFalse(v92.pcm_upstream_supported(
            v90.parse_info1d(vi.build_info1c(blocks=[{}] * 6))))

    def test_info1a_pcm_round_trip_and_table_18_layout(self):
        bits = v92.build_info1a_pcm(sections=2, max_coeffs_code=3, max_section_code=1,
                                    md_len=20, uinfo=90)
        self.assertEqual(len(bits), 70)
        parsed = v92.parse_info1a_pcm(bits)
        self.assertEqual(parsed["filter_sections"], ("p1", "p2", "z2"))
        self.assertEqual(parsed["max_total_coeffs"], 384)
        self.assertEqual(parsed["max_section_coeffs"], 192)
        self.assertEqual(parsed["md_symbols"], 20 * 276)
        self.assertEqual((parsed["analog_symbol_rate"], parsed["digital_symbol_rate"]), (6, 6))
        self.assertEqual(sum(bits[40:50]), 10)          # reserved bits are ones
        damaged = list(bits)
        damaged[20] ^= 1
        self.assertIsNone(v92.parse_info1a_pcm(damaged))

    def test_info1a_pcm_rejects_invalid_fields(self):
        with self.assertRaises(ValueError):
            v92.build_info1a_pcm(uinfo=66)
        with self.assertRaises(ValueError):
            v92.build_info1a_pcm(sections=4)

    def test_info1a_pcm_and_v34_forms_are_distinct(self):
        pcm = v92.build_info1a_pcm(uinfo=80)
        v34_form = v90.build_info1a_v90(uinfo=80, analog_symbol_rate=4)
        self.assertIsNotNone(v92.parse_info1a_pcm(pcm))
        self.assertIsNone(v92.parse_info1a_pcm(v34_form))   # analogue rate is not 6


if __name__ == "__main__":
    unittest.main()
