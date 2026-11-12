import unittest

import v90_info_reference as info
from v90_dil import dil_codewords_match, dil_g711_codewords


class V90DilTests(unittest.TestCase):
    def one_level_descriptor(self, law):
        fields = dict(
            N=1, LSP=1, LTP=1, SP=[0], TP=[0],
            H=[127] + [0] * 7, REF=[0] * 8, Ucodes=[0])
        parsed = info.parse_ja_descriptor(info.build_ja_descriptor(**fields))
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["Ucodes"], [0])
        self.assertEqual(parsed["TP"], [0])
        self.assertEqual(len(dil_g711_codewords(parsed, law)), 768)
        return parsed

    def test_default_one_level_dil_is_exact_g711_for_both_laws(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                expected = dil_g711_codewords(self.one_level_descriptor(law), law)
                self.assertEqual(len(expected), 768)
                self.assertTrue(dil_codewords_match(expected, expected, law))

    def test_pcmu_zero_code_alias_is_the_only_accepted_difference(self):
        self.assertTrue(dil_codewords_match(b"\x7f", b"\xff", "pcmu"))
        self.assertFalse(dil_codewords_match(b"\x7e", b"\xff", "pcmu"))
        self.assertFalse(dil_codewords_match(b"\x7f", b"\xff", "pcma"))


if __name__ == "__main__":
    unittest.main()
