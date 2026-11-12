"""Tests for the V.90 analogue-caller Phase 3 protocol controller."""

import unittest

import v90_info_reference as v90
from v90_analogue_caller_phase3 import V90AnalogueCallerPhase3


def make_descriptor(n):
    bits = v90.build_ja_descriptor(
        N=n, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
        REF=[0] * 8, Ucodes=[1] * n)
    return bits, v90.parse_ja_descriptor(bits)


class V90AnalogueCallerPhase3Tests(unittest.TestCase):
    def make_controller(self, n=0, *, md_symbols=0):
        bits, descriptor = make_descriptor(n)
        return V90AnalogueCallerPhase3(
            baud=2400 * 10 / 7, ja_bits=bits, descriptor=descriptor,
            md_symbols=md_symbols)

    def test_initial_sequence_obeys_spec_order_and_symbol_durations(self):
        controller = self.make_controller()
        actions = controller.start(1000)
        signals = [action.signal for action in actions]
        self.assertEqual(signals, ["silence", "S", "Sbar", "PP", "TRN",
                                   "Ja-repeat", "listen-sd-transition"])
        self.assertEqual(actions[0].duration_samples, 560)
        self.assertEqual([(a.signal, a.symbols) for a in actions[1:5]],
                         [("S", 128), ("Sbar", 16), ("PP", 288),
                          ("TRN", 512)])
        self.assertEqual(actions[5].symbols, len(controller.ja_bits) // 2)
        symbol_actions = actions[1:5]
        self.assertEqual(sum(a.duration_samples for a in symbol_actions),
                         round(sum(a.symbols for a in symbol_actions) * 8000
                               / controller.baud))
        self.assertEqual(controller.state, controller.WAIT_SD)

    def test_optional_md_has_second_s_pair_before_pp(self):
        controller = self.make_controller(md_symbols=77)
        actions = controller.start(0)
        self.assertEqual([a.signal for a in actions[:8]],
                         ["silence", "S", "Sbar", "MD", "S", "Sbar",
                          "PP", "TRN"])
        self.assertEqual(actions[3].symbols, 77)

    def test_sd_transition_trains_first_2040_trn1d_symbols_then_waits_jd(self):
        controller = self.make_controller()
        controller.start(0)
        actions = controller.feed("sd_transition", controller._ja_start + 120)
        self.assertEqual(controller.state, controller.RX_TRN1D)
        self.assertEqual([action.signal for action in actions],
                         ["stop-Ja", "silence", "receive-trn1d"])
        self.assertEqual(actions[2].symbols, 2040)
        self.assertEqual(actions[2].duration_samples,
                         round(2040 * 8000 / controller.baud))
        actions = controller.feed("trn1d_2040_complete",
                                  actions[2].at + actions[2].duration_samples)
        self.assertEqual(controller.state, controller.WAIT_JD)
        self.assertEqual([a.signal for a in actions], ["listen-jd"])

    def test_jd_jprime_and_zero_dil_reach_phase4(self):
        controller = self.make_controller(n=0)
        controller.start(0)
        controller.feed("sd_transition", controller._ja_start + 1)
        controller.feed("trn1d_2040_complete", controller._cursor)
        actions = controller.feed("jd", controller._cursor + 100)
        self.assertEqual(controller.state, controller.WAIT_JD_PRIME)
        self.assertEqual(actions[0].signal, "S")
        self.assertEqual(actions[0].symbols, 128)
        self.assertEqual(actions[1].signal, "listen-jd-prime")
        actions = controller.feed("jd_prime", actions[0].at +
                                  actions[0].duration_samples + 50)
        self.assertTrue(controller.phase4_ready)
        self.assertEqual([a.signal for a in actions], ["Sbar", "phase4-ready"])

    def test_nonzero_dil_waits_for_exact_requested_octet_count(self):
        controller = self.make_controller(n=1)
        controller.start(0)
        controller.feed("sd_transition", controller._ja_start + 1)
        controller.feed("trn1d_2040_complete", controller._cursor)
        s = controller.feed("jd", controller._cursor + 1)[0]
        controller.feed("jd_prime", s.at + s.duration_samples + 1)
        self.assertEqual(controller.state, controller.RX_DIL)
        expected = controller.dil_symbols
        self.assertGreater(expected, 0)
        self.assertEqual(controller.feed_dil_codewords(b"\x00" * (expected - 1),
                                                       at=5000), [])
        self.assertEqual(controller.state, controller.RX_DIL)
        actions = controller.feed_dil_codewords(b"\x00", at=5001)
        self.assertTrue(controller.phase4_ready)
        self.assertEqual([a.signal for a in actions], ["S", "Sbar",
                                                       "phase4-ready"])
        self.assertEqual([a.symbols for a in actions[:2]], [128, 16])

    def test_missing_sd_or_jd_or_dil_uses_bounded_retrain(self):
        caller = self.make_controller()
        caller.start(0)
        actions = caller.tick(caller._deadline)
        self.assertEqual(caller.state, caller.RETRAIN)
        self.assertEqual([a.signal for a in actions], ["handoff-retrain"])

        caller = self.make_controller()
        caller.start(0)
        caller.feed("sd_transition", caller._ja_start + 1)
        caller.feed("trn1d_2040_complete", caller._cursor)
        caller.feed("jd", caller._cursor + 1)
        self.assertEqual(caller.state, caller.WAIT_JD_PRIME)
        self.assertIsNotNone(caller._deadline)
        actions = caller.tick(caller._deadline)
        self.assertEqual(caller.state, caller.RETRAIN)
        self.assertEqual([a.signal for a in actions], ["handoff-retrain"])

        caller = self.make_controller()
        caller.start(0)
        caller.feed("sd_transition", caller._ja_start + 1)
        actions = caller.tick(caller._deadline)
        self.assertEqual(caller.state, caller.RETRAIN)
        self.assertEqual([a.signal for a in actions], ["handoff-retrain"])

    def test_sd_wait_timeout_is_profile_controlled_and_bounded(self):
        controller = V90AnalogueCallerPhase3(
            baud=2400 * 10 / 7, ja_bits=make_descriptor(0)[0],
            descriptor=make_descriptor(0)[1], sd_timeout_ms=3000)
        controller.start(100)
        self.assertEqual(controller._deadline, controller._ja_start + 24000)
        self.assertEqual(controller.tick(controller._deadline - 1), [])
        self.assertEqual([a.signal for a in controller.tick(controller._deadline)],
                         ["handoff-retrain"])
        with self.assertRaisesRegex(ValueError, "500..10000"):
            V90AnalogueCallerPhase3(
                baud=2400 * 10 / 7, ja_bits=make_descriptor(0)[0],
                descriptor=make_descriptor(0)[1], sd_timeout_ms=10001)

    def test_rejects_unqualified_and_out_of_order_inputs(self):
        bits, descriptor = make_descriptor(0)
        bad = dict(descriptor, valid=False)
        with self.assertRaisesRegex(ValueError, "CRC-valid"):
            V90AnalogueCallerPhase3(baud=2400, ja_bits=bits, descriptor=bad)
        caller = self.make_controller()
        with self.assertRaisesRegex(RuntimeError, "before receive events"):
            caller.feed("sd_transition", 10)
        caller.start(0)
        with self.assertRaisesRegex(ValueError, "invalid in state"):
            caller.feed("jd_prime", caller._cursor)


if __name__ == "__main__":
    unittest.main()
