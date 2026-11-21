import unittest

import v92_renegotiation as reneg
import v92_session as session
import v90_digital_phase4 as dp4

UP1 = bytes((i * 37 + 5) & 0xFF for i in range(64))
DOWN1 = bytes((i * 91 + 17) & 0xFF for i in range(64))
UP2 = bytes((i * 13 + 101) & 0xFF for i in range(80))
DOWN2 = bytes((i * 53 + 7) & 0xFF for i in range(80))


class RenegotiationTests(unittest.TestCase):
    def test_r_t_pair_is_384_symbols_plus_24_complementary_bar_symbols(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                initial = session.run(law, up_payload=b"u", down_payload=b"d")
                self.assertTrue(initial["ok"], initial.get("reason"))
                profile = initial["profile"]
                ucodes = dp4._highest_power_interval_ucodes(profile, law)
                expected = (dp4._r_signal(ucodes, law) * 16
                            + dp4._r_signal(ucodes, law, r_bar=True))
                actual = reneg.digital_rt_pair(profile, law)
                self.assertEqual(len(actual), 408)
                self.assertEqual(actual, expected)

    def test_a_second_phase_4_with_new_parameters_carries_new_payloads(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 240):
                with self.subTest(law=law, delay=delay):
                    r1 = session.run(law, delay=delay, up_payload=UP1, down_payload=DOWN1)
                    self.assertTrue(r1["ok"], r1.get("reason"))
                    self.assertEqual(r1["up_received"][:len(UP1)], UP1)
                    r2 = reneg.renegotiate(r1, law, delay=delay, up_payload=UP2, down_payload=DOWN2,
                                           drn_down=1, drn_up=8)
                    names = [n for n, _ in r2["events"]]
                    for step in ("rd", "rd-bar", "ru", "ru-bar"):
                        self.assertIn(step, names, r2["events"])
                    self.assertTrue(r2["a2"].connected and r2["d2"].connected, r2["events"])
                    self.assertEqual(r2["up_received"][:len(UP2)], UP2)
                    self.assertEqual(r2["down_received"][:len(DOWN2)], DOWN2)
                    # The parameters really changed: a lower downstream rate (CPu DRN 1 against 2) and a
                    # lower upstream K (CPd DRN 8 against 9).
                    self.assertLess(r2["a2"].data_rx.fb, r1["analogue4"].data_rx.fb)
                    self.assertLess(r2["d2"].data_rx.rx.profile.bits_per_frame,
                                    r1["digital4"].data_rx.rx.profile.bits_per_frame)

    def test_digital_initiated_silent_renegotiation_resumes_exact_payload(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                initial = session.run(law, up_payload=b"initial-up", down_payload=b"initial-down")
                self.assertTrue(initial["ok"], initial.get("reason"))
                resumed = reneg.renegotiate(initial, law, up_payload=UP2, down_payload=DOWN2,
                                            silent_period=True)
                analogue, digital = resumed["a2"], resumed["d2"]
                self.assertIsNotNone(analogue)
                self.assertTrue(analogue.connected, analogue.diagnostics())
                self.assertTrue(digital.connected, digital.diagnostics())
                self.assertEqual(resumed["up_received"][:len(UP2)], UP2)
                self.assertEqual(resumed["down_received"][:len(DOWN2)], DOWN2)
                self.assertIn("SILENT:trn2u-complete", analogue.sent)
                self.assertEqual(analogue._silent_trn_generated, 8004)
                self.assertIn("SILENT:rt-detected", analogue.sent)
                self.assertIn("SILENT:request-cleared", digital.sent)
                self.assertIn("SILENT:rt-sent", digital.sent)

    def test_digital_initiated_silent_renegotiation_early_rt_resumes_payload(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                initial = session.run(law, up_payload=b"initial-up", down_payload=b"initial-down")
                self.assertTrue(initial["ok"], initial.get("reason"))
                resumed = reneg.renegotiate(initial, law, up_payload=UP2, down_payload=DOWN2,
                                            silent_period=True, silent_early_rt_after=4008)
                analogue, digital = resumed["a2"], resumed["d2"]
                self.assertTrue(analogue.connected, analogue.diagnostics())
                self.assertTrue(digital.connected, digital.diagnostics())
                self.assertEqual(resumed["up_received"][:len(UP2)], UP2)
                self.assertEqual(resumed["down_received"][:len(DOWN2)], DOWN2)
                self.assertIn("SILENT:early-rt-wait", digital.sent)
                self.assertIn("SILENT:early-rt-send", digital.sent)
                self.assertIn("SILENT:trn2u-terminated-by-rt", analogue.sent)
                self.assertLess(analogue._silent_trn_generated, 8004)
                self.assertGreaterEqual(analogue._silent_trn_generated, 2400)
                self.assertIn("SILENT:rt-detected", analogue.sent)

    def test_analogue_initiated_silent_renegotiation_resumes_payload(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                initial = session.run(law, up_payload=b"initial-up", down_payload=b"initial-down")
                self.assertTrue(initial["ok"], initial.get("reason"))
                resumed = reneg.renegotiate_analogue_initiated(
                    initial, law, up_payload=UP2, down_payload=DOWN2)
                analogue, digital = resumed["a2"], resumed["d2"]
                self.assertIsNotNone(digital)
                self.assertTrue(analogue.connected, analogue.diagnostics())
                self.assertTrue(digital.connected, digital.diagnostics())
                self.assertEqual(resumed["up_received"][:len(UP2)], UP2)
                self.assertEqual(resumed["down_received"][:len(DOWN2)], DOWN2)
                event_names = [name for name, _ in resumed["events"]]
                self.assertEqual(event_names[:5], ["analogue-ru-start", "ru", "ru-bar",
                                                   "rd", "rd-bar"])
                self.assertIn("SILENT:trn2u-complete", analogue.sent)
                self.assertEqual(analogue._silent_trn_generated, 8004)
                self.assertIn("SILENT:rt-detected", analogue.sent)
                self.assertIn(("suvu", True), digital.silent_period_events)
                self.assertIn(("suvu", False), digital.silent_period_events)
                self.assertIn("SILENT:request-cleared", digital.sent)
                self.assertIn("SILENT:rt-sent", digital.sent)
                self.assertLess(analogue.data_rx.fb, initial["analogue4"].data_rx.fb)
                self.assertLess(digital.data_rx.rx.profile.bits_per_frame,
                                initial["digital4"].data_rx.rx.profile.bits_per_frame)


if __name__ == "__main__":
    unittest.main()
