import unittest

from v90_mp_policy import (answerer_mp_profile,
                           select_analog_upstream_profile,
                           select_analog_upstream_rate,
                           select_downstream_drn_from_jd,
                           v90_downstream_rate,
                           validate_answerer_mp_profile)
from v90_mp import V90MpScanner, build_mp, parse_mp

# captured peer's MP from its first native V.90 call (2026-11-10,
# v90-captured_peer-candidate-20261110-1): 197 of 207 repeats were bit-identical;
# this is their majority vote. It is padded to 195 bits, a multiple of the
# 15-bit CPt data frame.
REFERENCE_PEER_MP_TYPE1 = [int(bit) for bit in (
    "11111111111111111010000000110001100011111111111110000100010001000"
    "00000010111111111111011011010111111110110011111111111101101101000"
    "00000001110111000000000000000000000000000111010001000100000000000"
)]


class V90MpPolicyTests(unittest.TestCase):
    def test_explicit_type1_answerer_profile_round_trips_and_keeps_default(self):
        profile = {
            "mp_type": 1,
            "max_a2d_drn": 12,
            "trellis": 0,
            "nonlinear": 1,
            "shaping": 1,
            "rate_mask": 0x1FFF,
            "precoder": [520, -52, -119, 1, 116, 137],
        }
        selected = validate_answerer_mp_profile(profile)
        self.assertEqual(selected["mp_type"], 1)
        self.assertEqual(selected["max_a2d_drn"], 12)
        self.assertEqual(selected["nonlinear"], 1)
        self.assertEqual(selected["shaping"], 1)
        self.assertEqual(selected["rate_mask"], 0x1FFF)
        self.assertEqual(selected["precoder"], (520, -52, -119, 1, 116, 137))
        self.assertEqual(answerer_mp_profile(3428.5714285714)["mp_type"], 0)

    def test_rejects_invalid_explicit_answerer_mp_profiles(self):
        valid = {
            "mp_type": 1, "max_a2d_drn": 12, "trellis": 0,
            "nonlinear": 1, "shaping": 1, "rate_mask": 0x1FFF,
            "precoder": [520, -52, -119, 1, 116, 137],
        }
        for change, expected in (
                ({"trellis": 1}, "only the 16-state"),
                ({"precoder": [1, 2]}, "six signed 16-bit"),
                ({"rate_mask": 1 << 13}, "13 V.90"),
                ({"surprise": 1}, "unknown V.90")):
            candidate = {**valid, **change}
            with self.subTest(change=change), self.assertRaisesRegex(
                    ValueError, expected):
                validate_answerer_mp_profile(candidate)

    def test_parses_captured_peer_type1_mp_with_v34_crc_coverage(self):
        """The CRC covers bits 18..33 and 35..50, as in V.34.

        Our build/parse pair once skipped bits 35 and 50, so paired calls
        agreed with each other and rejected every MP from a real modem.
        """
        parsed = parse_mp(REFERENCE_PEER_MP_TYPE1)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["type"], 1)
        self.assertEqual(parsed["max_a2d_drn"], 12)
        self.assertEqual(parsed["rate_mask"], 0x1FFF)
        self.assertEqual((parsed["nonlinear"], parsed["shaping"], parsed["ack"]),
                         (1, 1, 0))
        self.assertEqual(parsed["precoder"], [546, -12, -165, -13, 91, 119])
        scanner = V90MpScanner()
        found = []
        for _ in range(3):
            found += scanner.feed(REFERENCE_PEER_MP_TYPE1)
        self.assertEqual(len(found), 3)
        self.assertEqual(build_mp(mp_type=1, max_a2d_drn=12, nonlinear=1,
                                  shaping=1, rate_mask=0x1FFF,
                                  precoder=parsed["precoder"]),
                         REFERENCE_PEER_MP_TYPE1[:192])

    def test_type1_mp_sets_precoder_nonlinear_and_shaping(self):
        cp = {"kind": "cp", "rate": 33600,
              "analog_rate_mask": (1 << 0) | (1 << 7) | (1 << 12)}
        mp = {**parse_mp(REFERENCE_PEER_MP_TYPE1), "mp_type": 1}
        profile = select_analog_upstream_profile(cp, mp, baud=3200.0,
                                                 carrier=1828.5714285714)
        self.assertEqual(profile.rate, 21600)
        self.assertTrue(profile.nonlinear)
        self.assertTrue(profile.shaping)
        self.assertEqual(profile.precoder, (546, -12, -165, -13, 91, 119))
        type0 = {**mp, "mp_type": 0, "nonlinear": 0, "shaping": 0}
        profile = select_analog_upstream_profile(cp, type0, baud=3200.0,
                                                 carrier=1828.5714285714)
        self.assertEqual(profile.precoder, (0, 0, 0, 0, 0, 0))
        self.assertFalse(profile.shaping)

    def test_profile_advertises_only_supported_v34_rates_at_3200_baud(self):
        profile = answerer_mp_profile(3200)
        self.assertEqual(profile, {
            "mp_type": 0,
            "max_a2d_drn": 13,
            "trellis": 0,
            "nonlinear": 0,
            "shaping": 0,
            "rate_mask": 0x0FFF,
        })
        parsed = parse_mp(build_mp(**profile))
        self.assertEqual(parsed["max_a2d_drn"], profile["max_a2d_drn"])
        self.assertEqual(parsed["rate_mask"], profile["rate_mask"])

    def test_profile_includes_33600_only_at_supported_top_baud(self):
        profile = answerer_mp_profile(3428.5714285714)
        self.assertEqual(profile["max_a2d_drn"], 14)
        self.assertEqual(profile["rate_mask"], 0x1FFF)

    def test_rejects_baud_with_no_supported_receive_rate(self):
        with self.assertRaisesRegex(ValueError, "no V.90 upstream rate"):
            answerer_mp_profile(1)

    def test_selects_highest_rate_in_cp_mp_intersection_under_mp_ceiling(self):
        cp = {"kind": "cp", "rate": 33600,
              "analog_rate_mask": (1 << 0) | (1 << 7) | (1 << 12)}
        mp = {"rate_mask": 0x1FFF, "max_a2d_drn": 13}
        self.assertEqual(select_analog_upstream_rate(cp, mp), 21600)
        mp["max_a2d_drn"] = 14
        self.assertEqual(select_analog_upstream_rate(cp, mp), 33600)

    def test_rejects_empty_upstream_rate_intersection(self):
        cp = {"kind": "cp", "rate": 33600,
              "analog_rate_mask": 1 << 12}
        mp = {"rate_mask": 0x0FFF, "max_a2d_drn": 14}
        with self.assertRaisesRegex(ValueError, "no common"):
            select_analog_upstream_rate(cp, mp)

    def test_cp_downstream_jd_selector_uses_advertisement_and_local_cap(self):
        self.assertEqual(v90_downstream_rate(1), 28000)
        self.assertEqual(v90_downstream_rate(22), 56000)
        all_rates = (1 << 22) - 1
        self.assertEqual(select_downstream_drn_from_jd(all_rates)["drn"], 22)
        self.assertEqual(
            select_downstream_drn_from_jd(all_rates, profile_cap=7)["drn"], 7)
        self.assertEqual(select_downstream_drn_from_jd(1 << 9)["drn"], 10)
        sparse = (1 << 1) | (1 << 9) | (1 << 20)
        self.assertEqual(
            select_downstream_drn_from_jd(sparse, profile_cap=9)["drn"], 2)
        with self.assertRaisesRegex(ValueError, "rate mask"):
            select_downstream_drn_from_jd(0)
        with self.assertRaisesRegex(ValueError, "no common"):
            select_downstream_drn_from_jd(1 << 10, profile_cap=10)

if __name__ == "__main__":
    unittest.main()
