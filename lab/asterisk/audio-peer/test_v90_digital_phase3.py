"""Tests for the event-driven V.90 digital-answerer Phase 3 controller."""

import unittest

import v90_info_reference as v90
from v90_digital_phase3 import V90DigitalAnswererPhase3


class V90DigitalAnswererPhase3Tests(unittest.TestCase):
    def descriptor(self, n):
        return v90.parse_ja_descriptor(v90.build_ja_descriptor(
            N=n, LSP=1, LTP=1, SP=[1], TP=[1], H=[0] * 8,
            REF=[0] * 8, Ucodes=[1] * n))

    def reach_jd(self, controller):
        controller.start()
        initial_len = 384 + 48 + controller.trn1d_symbols
        self.assertEqual(len(controller.read(initial_len)), initial_len)
        self.assertEqual(controller.state, controller.JD)

    def test_zero_dil_proceeds_to_phase4_after_jprime_without_extra_transition(self):
        controller = V90DigitalAnswererPhase3(90, "pcmu", self.descriptor(0), 1)
        self.reach_jd(controller)
        first_jd = controller.read(1)
        self.assertEqual(len(first_jd), 1)
        controller.notify_s_transition()
        rest = controller.read(83)
        self.assertEqual(len(rest), 83)  # 71 Jd + 12 J'_d
        self.assertTrue(controller.phase4_ready)
        self.assertEqual(controller.read(100), b"")

    def test_dil_stops_only_at_the_current_segment_boundary(self):
        descriptor = self.descriptor(1)
        controller = V90DigitalAnswererPhase3(90, "pcma", descriptor, 1)
        self.reach_jd(controller)
        controller.notify_s_transition()
        # Complete the current 72-symbol Jd and 12-symbol J'_d.
        self.assertEqual(len(controller.read(84)), 84)
        self.assertEqual(controller.state, controller.DIL)
        prefix = controller.read(2)
        self.assertEqual(len(prefix), 2)
        controller.notify_s_transition()
        tail = controller.read(20)
        self.assertEqual(len(tail), 4)
        self.assertTrue(controller.phase4_ready)
        self.assertEqual(len(prefix + tail), 6)

    def test_rejects_out_of_order_events_and_repeated_start(self):
        controller = V90DigitalAnswererPhase3(90, "pcmu", self.descriptor(0), 1)
        with self.assertRaisesRegex(RuntimeError, "before Phase 3"):
            controller.notify_s_transition()
        controller.start()
        with self.assertRaisesRegex(RuntimeError, "already started"):
            controller.start()

    def test_ja_descriptor_can_arrive_before_jprime_finishes(self):
        controller = V90DigitalAnswererPhase3(90, "pcmu", None, 1)
        self.reach_jd(controller)
        controller.notify_s_transition()
        self.assertEqual(len(controller.read(71)), 71)  # Finish current Jd.
        self.assertEqual(controller.state, controller.JD)
        controller.set_dil_descriptor(self.descriptor(0))
        self.assertEqual(len(controller.read(13)), 13)
        self.assertTrue(controller.phase4_ready)

    def test_rejects_ja_descriptor_after_dil_start(self):
        controller = V90DigitalAnswererPhase3(90, "pcmu", None, 1)
        self.reach_jd(controller)
        controller.notify_s_transition()
        controller.read(71)
        controller.set_dil_descriptor(self.descriptor(1))
        controller.read(13)
        self.assertEqual(controller.state, controller.DIL)
        with self.assertRaisesRegex(RuntimeError, "after the DIL decision"):
            controller.set_dil_descriptor(self.descriptor(0))

    def test_no_s_deadline_is_anchored_at_trn1d_and_includes_rtd(self):
        rtd_samples = 2400
        controller = V90DigitalAnswererPhase3(
            90, "pcmu", self.descriptor(0), 1,
            round_trip_delay_samples=rtd_samples)
        self.reach_jd(controller)
        deadline = (controller._trn1d_start_sample + 5100 * 8
                    + rtd_samples)
        remaining = deadline - controller._samples_emitted

        self.assertEqual(len(controller.read(remaining - 1)), remaining - 1)
        self.assertEqual(controller.state, controller.JD)
        self.assertEqual(len(controller.read(100)), 1)
        self.assertEqual(controller.state, controller.FAILED)
        self.assertIn("5100 ms plus RTD", controller.failure_reason)
        self.assertEqual(controller.read(100), b"")

    def test_s_detected_before_deadline_finishes_current_jd(self):
        controller = V90DigitalAnswererPhase3(
            90, "pcmu", self.descriptor(0), 1,
            round_trip_delay_samples=800)
        self.reach_jd(controller)
        deadline = controller._trn1d_start_sample + 5100 * 8 + 800
        remaining = deadline - controller._samples_emitted - 1
        self.assertEqual(len(controller.read(remaining)), remaining)
        controller.notify_s_transition()
        self.assertEqual(controller.state, controller.JD)
        self.assertTrue(controller.read(1000))
        self.assertTrue(controller.phase4_ready)
        self.assertIsNone(controller.failure_reason)


if __name__ == "__main__":
    unittest.main()
