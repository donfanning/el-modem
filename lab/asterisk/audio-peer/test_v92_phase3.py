"""V.92 Phase 3 sequences (Tables 21, 22 and clauses 8.5/8.6)."""

import unittest

import v34_tx
import v92_phase3 as p3


def bits_value(bits, lo, hi):
    return sum(bits[lo + i] << i for i in range(hi - lo + 1))


class JdJpTests(unittest.TestCase):
    def test_jd_layout_and_rates(self):
        mask = p3.jd_rate_mask([28000, 56000, 48000])
        default = p3.parse_jd_jp(p3.build_jd(mask))
        self.assertEqual(default["max_lookahead"], 1)
        bits = p3.build_jd(mask, max_lookahead=2)
        self.assertEqual(len(bits), 72)
        self.assertEqual(bits[18], 1)                          # 28,000
        self.assertEqual(bits[33], 1)                          # 48,000
        self.assertEqual(bits[40], 1)                          # 56,000
        self.assertEqual(bits[47], 0)                          # Jd identifier
        parsed = p3.parse_jd_jp(bits)
        self.assertEqual(parsed["kind"], "jd")
        self.assertEqual(parsed["max_lookahead"], 2)
        self.assertEqual([round(r) for r in parsed["rates"]], [28000, 48000, 56000])

    def test_jd_rate_validation(self):
        with self.assertRaises(ValueError):
            p3.jd_rate_mask([28001])
        with self.assertRaises(ValueError):
            p3.jd_rate_mask([57333])
        with self.assertRaises(ValueError):
            p3.build_jd(1 << 22)

    def test_jp_layout_matches_table_22(self):
        bits = p3.build_jp(0xA55A, training_8point=True)
        self.assertEqual(len(bits), 72)
        self.assertEqual(bits_value(bits, 18, 33), 0xA55A)
        self.assertEqual((bits[47], bits[48], bits[49], bits[50]), (1, 1, 0, 0))
        for index in (17, 34, 51):
            self.assertEqual(bits[index], 0)
        parsed = p3.parse_jd_jp(bits)
        self.assertEqual(parsed["kind"], "jp")
        self.assertEqual(parsed["phase_fraction"], 0xA55A)
        self.assertAlmostEqual(parsed["phase_symbols"], 0xA55A / 65536)
        self.assertTrue(parsed["training_8point"])
        self.assertFalse(parsed["renegotiation_8point"])

    def test_jp_both_constellation_flags(self):
        for training in (False, True):
            for renegotiation in (False, True):
                parsed = p3.parse_jd_jp(p3.build_jp(7, training_8point=training,
                                                    renegotiation_8point=renegotiation))
                self.assertEqual((parsed["training_8point"], parsed["renegotiation_8point"]),
                                 (training, renegotiation))

    def test_faults_rejected(self):
        for build in (lambda: p3.build_jd(p3.jd_rate_mask([36000])), lambda: p3.build_jp(1234)):
            good = build()
            for index in (3, 17, 20, 36, 41, 51, 55, 70):
                bad = list(good)
                bad[index] ^= 1
                self.assertIsNone(p3.parse_jd_jp(bad), index)

    def test_jp_prime_is_twelve_zeros(self):
        self.assertEqual(p3.jp_prime(), [0] * 12)


class SignalTests(unittest.TestCase):
    def test_sign_modulator_is_scrambled_and_differential(self):
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0] * 4
        modulator = p3.SignModulator(initial_sign=1)
        signs = modulator.encode(bits)
        # Undo: differential decode against the previous sign, then descramble (GPC).
        previous, decoded_scrambled = 1, []
        for s in signs:
            decoded_scrambled.append(s ^ previous)
            previous = s
        scrambler = v34_tx.Scrambler("call")
        self.assertEqual(decoded_scrambled, [scrambler.step(b) for b in bits])
        self.assertEqual(modulator.last_sign, signs[-1])

    def test_chained_state_continues_between_sequences(self):
        a = p3.SignModulator(initial_sign=0)
        first = a.encode([1, 1, 0, 1])
        self.assertEqual(a.last_sign, first[-1])
        whole = p3.SignModulator(initial_sign=0).encode([1, 1, 0, 1, 0, 0, 1])
        self.assertEqual(first + a.encode([0, 0, 1]), whole)

    def test_ru_su_patterns(self):
        self.assertEqual(p3.ru(2), [1, 1, 1, -1, -1, -1] * 2)
        self.assertEqual(p3.ru(1, bar=True), [-1, -1, -1, 1, 1, 1])
        su = p3.su(1)
        self.assertAlmostEqual(su[0], 1.5 ** 0.5)
        self.assertEqual((su[1], su[4]), (0.0, 0.0))
        self.assertEqual(p3.su(1, bar=True), [-v for v in su])

    def test_trn1u_comes_from_the_analogue_scrambler_fed_ones(self):
        seq = p3.trn1u(48)
        scrambler = v34_tx.Scrambler("answer")
        expected = [-1.0 if scrambler.step(1) else 1.0 for _ in range(48)]
        self.assertEqual(seq, expected)
        self.assertTrue(set(seq) <= {1.0, -1.0})
        self.assertIn(-1.0, seq)
        with self.assertRaises(ValueError):
            p3.trn1u(10)


if __name__ == "__main__":
    unittest.main()
