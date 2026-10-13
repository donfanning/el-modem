import unittest

from v34_call import P4_TRN_SYMBOLS, p4_trn_symbol_count, withhold_mp


class Phase4TrnLengthTests(unittest.TestCase):
    def test_default_is_the_long_standing_1500_at_every_baud(self):
        for baud in (2400, 2743, 2800, 3000, 3200, 3429):
            self.assertEqual(p4_trn_symbol_count(baud, {}), 1500)
        self.assertEqual(P4_TRN_SYMBOLS, 1500)

    def test_the_variable_lengthens_it(self):
        self.assertEqual(p4_trn_symbol_count(3429, {"V34_P4_TRN_SYMBOLS": "5800"}), 5800)

    def test_it_is_held_between_512_symbols_and_2000_ms(self):
        self.assertEqual(p4_trn_symbol_count(3429, {"V34_P4_TRN_SYMBOLS": "100"}), 512)
        self.assertEqual(p4_trn_symbol_count(3429, {"V34_P4_TRN_SYMBOLS": "99999"}), 6858)      # 2.0 s at 3429 baud
        self.assertEqual(p4_trn_symbol_count(2400, {"V34_P4_TRN_SYMBOLS": "99999"}), 4800)      # 2.0 s at 2400 baud

    def test_a_malformed_value_falls_back_to_the_default(self):
        self.assertEqual(p4_trn_symbol_count(3429, {"V34_P4_TRN_SYMBOLS": "lots"}), 1500)

    def test_the_call_site_uses_the_helper(self):
        import inspect
        import v34_call
        self.assertIn("p4_trn_symbol_count(self.call_baud)", inspect.getsource(v34_call.V34Caller))


class WithholdMpHookTests(unittest.TestCase):
    def test_off_by_default_and_only_exactly_1_turns_it_on(self):
        self.assertFalse(withhold_mp({}))
        self.assertFalse(withhold_mp({"V34_TEST_WITHHOLD_MP": "0"}))
        self.assertFalse(withhold_mp({"V34_TEST_WITHHOLD_MP": "yes"}))
        self.assertTrue(withhold_mp({"V34_TEST_WITHHOLD_MP": "1"}))

    def test_the_mp_start_goes_through_the_hook(self):
        import inspect
        import v34_call
        src = inspect.getsource(v34_call.V34Caller._phase4_rx)
        self.assertIn("if withhold_mp():", src)
        self.assertIn("q.fill = self._fill_mp", src)


if __name__ == "__main__":
    unittest.main()
