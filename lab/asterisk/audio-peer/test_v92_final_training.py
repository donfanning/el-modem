"""V.92 final-training exchange rules (clause 9.6)."""

import unittest

import v92_final_training as ft


def kinds(trace):
    return [m.kind for m in trace]


class FinalTrainingTests(unittest.TestCase):
    def test_error_free_exchange_connects_at_every_latency(self):
        for latency in range(0, 8):
            result = ft.run(latency)
            self.assertTrue(result["connected"], latency)
            for name, trace in result["traces"].items():
                self.assertEqual(kinds(trace).count("CP"), 1, (latency, name))
                self.assertEqual(kinds(trace).count("E"), 1, (latency, name))
                self.assertIn("B1", kinds(trace), (latency, name))

    def test_ordering_trn_suv_cp_e_b1_and_acks(self):
        trace = ft.run(2)["traces"]["digital"]
        order = [m.kind for m in trace]
        # TRN first, then SUV, a single CP, E, then B1 only.
        self.assertEqual(order[0], "TRN")
        self.assertLess(order.index("SUV"), order.index("CP"))
        self.assertLess(order.index("CP"), order.index("E"))
        self.assertLess(order.index("E"), order.index("B1"))
        self.assertEqual(set(order[order.index("B1"):]), {"B1"})
        # Nothing carries an acknowledge bit before the peer's CP could have arrived.
        first_ack = next(i for i, m in enumerate(trace) if m.ack)
        peer_cp_slot = kinds(ft.run(2)["traces"]["analogue"]).index("CP")
        self.assertGreater(first_ack, peer_cp_slot)

    def test_e_is_never_sent_without_acknowledge_evidence(self):
        result = ft.run(3)
        for name, party in result["parties"].items():
            self.assertTrue(party.sent_ack, name)
            self.assertTrue(party.saw_ack or party.saw_e, name)

    def test_analogue_starts_suv_early_when_the_digital_modem_does(self):
        # Digital is ready after 4 slots; the analogue modem's own TRN2u minimum is 12.
        trace = ft.run(1, digital_trn=4, analogue_trn=12)["traces"]["analogue"]
        first_suv = kinds(trace).index("SUV")
        self.assertLess(first_suv, 12)
        # The digital modem, by contrast, always waits out its minimum.
        digital = ft.run(1, digital_trn=6, analogue_trn=12)["traces"]["digital"]
        self.assertGreaterEqual(kinds(digital).index("SUV"), 6)

    def test_a_lost_first_cp_is_repeated_and_the_call_still_connects(self):
        # Figure 14: the first CPu never reaches the digital modem.
        seen = []

        def drop(sender, slot, message):
            if sender == "analogue" and message.kind == "CP" and not seen:
                seen.append(slot)
                return True
            return False

        result = ft.run(2, drop=drop)
        self.assertTrue(result["connected"])
        self.assertGreaterEqual(kinds(result["traces"]["analogue"]).count("CP"), 2)
        # The repeat came a round trip plus allowance after the first: 2 * (2 + 1) + 2 = 8 slots.
        positions = [i for i, m in enumerate(result["traces"]["analogue"]) if m.kind == "CP"]
        self.assertEqual(positions[1] - positions[0], 8)

    def test_a_dead_return_path_never_connects_and_never_sends_e_blindly(self):
        result = ft.run(1, drop=lambda s, slot, m: s == "analogue", max_slots=80)
        self.assertFalse(result["connected"])
        # The digital modem keeps sending SUV; it must not send E without an acknowledgement.
        self.assertNotIn("E", kinds(result["traces"]["digital"]))
        self.assertNotIn("B1", kinds(result["traces"]["digital"]))

    def test_the_repeat_wait_scales_with_the_round_trip_delay(self):
        # A CP must not be repeated before its acknowledgement could have come back.
        for latency in (0, 3, 7):
            result = ft.run(latency)
            for trace in result["traces"].values():
                self.assertEqual(kinds(trace).count("CP"), 1, latency)

    def test_bad_roles_are_rejected(self):
        with self.assertRaises(ValueError):
            ft.Party("modem", trn_slots=1, ack_wait=1)


if __name__ == "__main__":
    unittest.main()
