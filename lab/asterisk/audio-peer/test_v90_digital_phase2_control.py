import unittest

from v90_digital_phase2_adapter import V90DigitalPhase2Adapter
from v90_digital_phase2_control import V90DigitalPhase2Controller
from v90_digital_phase2_rx import Phase2RxEvent


class DigitalPhase2ControlTests(unittest.TestCase):
    def _to_second_probe(self):
        c = V90DigitalPhase2Controller(max_l2_samples=8000)
        actions = c.start(600)
        self.assertEqual([(a.at, a.signal, a.after) for a in actions], [
            (600, "info0d", None), (None, "tone_b", "info0d-complete")])
        c.feed("info0a", 700, payload={"ack_info0d": 0})
        first = c.feed("tone_a_reversal", 1000, now=1010)
        self.assertEqual((first[0].at, first[0].latest), (1320, 1328))
        self.assertEqual(first[1].at, 1400)
        self.assertEqual(c.feed("tone_a_reversal", 3000, now=3010), [])
        self.assertEqual(c.rtd_samples, 1360)
        c.feed("l1_complete", 4500)
        c.feed("l2_complete", 8500)
        c.feed("tone_a", 9000)
        return c

    def test_phase2_action_order_and_info1a_mode_selection(self):
        c = self._to_second_probe()
        actions = c.feed("tone_a_reversal_after_l2", 9500, now=9510)
        self.assertEqual([(a.signal, a.at) for a in actions], [
            ("reverse-tone-b", 9820), ("l1", 9900), ("l2", 11180)])
        self.assertEqual(actions[2].value, 4000)
        c.feed("tone_a", 12000)
        actions = c.feed("l2_echo_complete", 13000)
        self.assertEqual([(a.signal, a.at, a.after) for a in actions], [
            ("info1d", 13000, None), ("silence", None, "info1d-complete")])
        actions = c.feed("info1a", 14000, payload={"mode": 6})
        self.assertEqual(actions[0].signal, "handoff-v90-phase3")
        self.assertTrue(c.done)

    def test_retrain_continuation_uses_retained_info0a_and_40ms_a_to_b_turnaround(self):
        c = V90DigitalPhase2Controller()
        reversal_at = 10000
        actions = c.start_retrain_continuation(
            reversal_at, now=reversal_at + 160,
            info0a={"rate3429": 1, "disallow3429": 1})
        self.assertEqual(c.info0a, {"rate3429": 1, "disallow3429": 1})
        self.assertEqual(c.state, c.WAIT_A_REV2)
        self.assertEqual(c._a_rev_count, 1)
        self.assertEqual(c.b_reversal_at, reversal_at + 40 * 8)
        self.assertEqual([(a.at, a.signal, a.latest) for a in actions], [
            (reversal_at + 40 * 8, "reverse-tone-b", reversal_at + 41 * 8),
            (reversal_at + 50 * 8, "silence", None),
        ])

    def test_missing_info1a_accepts_late_info_or_runs_bounded_tone_a_retrain(self):
        c = V90DigitalPhase2Controller(rtd_samples=2400,
                                       info1a_timeout_retrain=True)
        c.start(0)
        c.state = c.WAIT_INFO1A
        complete_at = 10000
        c.info1d_transmitted(complete_at)
        deadline = complete_at + 5600 + 2400
        self.assertEqual(c._info1a_deadline_at, deadline)
        self.assertEqual(c.tick(deadline - 1), [])
        self.assertEqual(c.state, c.WAIT_INFO1A)
        self.assertEqual(c.tick(deadline), [])
        self.assertEqual(c.state, c.WAIT_INFO1A_RETRAIN)

        # The frame scanner remains usable after the response window expires.
        late = c.feed("info1a", deadline + 1, payload={"mode": 5})
        self.assertEqual(late[0].signal, "handoff-v34-call-phase3")
        self.assertTrue(c.done)

        c = V90DigitalPhase2Controller(rtd_samples=2400,
                                       info1a_timeout_retrain=True)
        c.start(0)
        c.state = c.WAIT_INFO1A
        c.info1d_transmitted(complete_at)
        c.tick(deadline)
        silence_and_tone = c.feed("tone_a", deadline + 120,
                                  now=deadline + 160)
        self.assertEqual(c.state, c.WAIT_RETRAIN_TONE_B)
        self.assertEqual([(a.at, a.signal) for a in silence_and_tone], [
            (deadline + 160, "silence"),
            (deadline + 160 + 560, "tone_b"),
        ])
        self.assertEqual(c.tick(deadline + 160 + 559), [])
        self.assertEqual(c.state, c.WAIT_RETRAIN_TONE_B)
        self.assertEqual(c.tick(deadline + 160 + 560), [])
        self.assertEqual(c.state, c.WAIT_RETRAIN_A_REVERSAL)

        first = c.feed("retrain_tone_a_reversal", deadline + 900,
                       now=deadline + 920)
        self.assertEqual(c.state, c.WAIT_RETRAIN_A_REV2)
        self.assertEqual([(a.at, a.signal) for a in first], [
            (deadline + 1220, "reverse-tone-b"),
            (deadline + 1300, "silence"),
        ])
        self.assertEqual(c._a_rev_count, 1)

    def test_retrain_response_buffers_tone_a_reversal_during_70ms_silence(self):
        c = V90DigitalPhase2Controller(rtd_samples=2400,
                                       info1a_timeout_retrain=True)
        c.start(0)
        c.state = c.WAIT_INFO1A
        complete_at = 10000
        c.info1d_transmitted(complete_at)
        deadline = complete_at + 5600 + 2400
        c.tick(deadline)
        c.feed("tone_a", deadline + 120, now=deadline + 160)
        tone_b_at = deadline + 160 + 560
        reversal_at = deadline + 120 + 50 * 8

        # The analogue modem's >=50 ms reversal occurs before our 70 ms
        # response silence ends. Do not discard it or reverse a Tone B that
        # has not yet started.
        self.assertEqual(c.state, c.WAIT_RETRAIN_TONE_B)
        self.assertEqual(c.feed("retrain_tone_a_reversal", reversal_at,
                                now=reversal_at + 40), [])
        self.assertEqual(c._pending_retrain_a_reversal[0], reversal_at)
        self.assertEqual(c.tick(tone_b_at - 1), [])
        actions = c.tick(tone_b_at)

        # Preserve the standard 40 ms A-to-B reversal time; the B transition
        # simply lands 15 ms into the returned Tone B after the 70 ms silence.
        self.assertEqual([(action.at, action.signal) for action in actions], [
            (reversal_at + 40 * 8, "reverse-tone-b"),
            (reversal_at + 50 * 8, "silence"),
        ])
        self.assertEqual(c.b_reversal_at, reversal_at + 40 * 8)
        self.assertEqual(c.state, c.WAIT_RETRAIN_A_REV2)

    def test_adapter_keeps_tone_a_reversal_detection_live_during_response_silence(self):
        c = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(c, rx_to_media=lambda sample: sample)
        c.state = c.WAIT_RETRAIN_TONE_B
        c.info0a = {"rate3429": 1}
        event = Phase2RxEvent("tone_a_reversal", 800, 920, {})
        self.assertEqual(adapter._expected_signal(), "tone_a")
        self.assertEqual(adapter._route(event),
                         ("retrain_tone_a_reversal", {}))

    def test_info1a_timeout_retrain_is_disabled_by_default(self):
        c = V90DigitalPhase2Controller()
        c.start(0)
        c.state = c.WAIT_INFO1A
        c.info1d_transmitted(1000)
        self.assertIsNone(c._info1a_deadline_at)
        self.assertEqual(c.tick(1000 + 10 * 8000), [])
        self.assertEqual(c.state, c.WAIT_INFO1A)

    def test_missing_second_tone_a_reversal_runs_900ms_rtd_recovery(self):
        c = self._to_second_probe()
        deadline = (c._second_initial_a_reversal_at
                    + int(.900 * 8000) + c.rtd_samples)
        self.assertEqual(c._missing_tone_a_reversal_deadline_at, deadline)
        self.assertEqual(c.tick(deadline - 1), [])
        actions = c.tick(deadline)
        self.assertEqual([(a.signal, a.at) for a in actions], [
            ("reverse-tone-b", deadline + 320),
            ("l1", deadline + 400),
            ("l2", deadline + 1680),
        ])
        self.assertEqual(actions[0].latest, deadline + 328)
        self.assertEqual(c.state, c.TX_PROBE)
        self.assertIsNone(c._missing_tone_a_reversal_deadline_at)

        # §9.2.1.2.4 conditions the receiver for Tone A and continues through
        # §9.2.1.1.7. INFO1d still requires both Tone A and the local L2 echo.
        c.feed("tone_a", deadline + 3000)
        echo_at = c.l2_started_at + 400 * 8
        info = c.feed("l2_echo_complete", echo_at)
        self.assertEqual([action.signal for action in info],
                         ["info1d", "silence"])
        self.assertEqual(c.state, c.WAIT_INFO1A)

    def test_tone_a_reversal_at_recovery_deadline_takes_recovery_branch(self):
        c = self._to_second_probe()
        deadline = c._missing_tone_a_reversal_deadline_at
        actions = c.feed("tone_a_reversal_after_l2", deadline, now=deadline)
        self.assertEqual([action.signal for action in actions],
                         ["reverse-tone-b", "l1", "l2"])
        self.assertEqual(c.state, c.TX_PROBE)

    def test_tone_a_recovery_after_timely_l2_echo_does_not_reject_echo(self):
        c = self._to_second_probe()
        c.feed("tone_a_reversal_after_l2", 9500, now=9510)
        echo_at = c.l2_started_at + 400 * 8
        c.feed("l2_echo_complete", echo_at)

        # The peer's bounded missing-Tone-B recovery can begin after the
        # digital side's timely local echo. The 550 ms limit is on echo time,
        # not on the later Tone A receive event that completes the exchange.
        tone_a_at = c.l2_started_at + 600 * 8
        actions = c.feed("tone_a", tone_a_at)
        self.assertEqual(c.state, c.WAIT_INFO1A)
        self.assertEqual([a.signal for a in actions], ["info1d", "silence"])
        self.assertEqual(actions[0].at, tone_a_at)

    def test_repeated_info0a_requests_deferred_acknowledged_info0d(self):
        c = V90DigitalPhase2Controller()
        c.start(0)
        self.assertEqual(c.feed("info0a", 10, payload={"ack_info0d": 0}), [])
        actions = c.feed("info0a", 20, payload={"ack_info0d": 0})
        self.assertEqual(
            [(action.signal, action.value, action.at, action.after)
             for action in actions],
            [("info0d", 1, None, "current-info0d-complete"),
             ("tone_b", None, None, "info0d-complete")])

    def test_repeated_info0a_restarts_tone_a_ranging_reversal_count(self):
        c = V90DigitalPhase2Controller()
        c.start(0)
        c.feed("info0a", 100, payload={"ack_info0d": 0})

        # The first A reversal gets the normal B reversal response.
        first = c.feed("tone_a_reversal", 1000, now=1010)
        self.assertEqual([action.signal for action in first],
                         ["reverse-tone-b", "silence"])
        self.assertEqual(c._a_rev_count, 1)

        # A repeated INFO0a enters §9.2.1.2.1 recovery. The B reversal that
        # arrives after the recovery response must be treated as the first
        # reversal of the restarted §9.2.1.1.3 exchange, not as the second.
        retry = c.feed("info0a", 1800, payload={"ack_info0d": 0})
        self.assertEqual([action.signal for action in retry],
                         ["info0d", "tone_b"])
        self.assertEqual(c._a_rev_count, 0)
        self.assertIsNone(c.b_reversal_at)
        restarted_first = c.feed("tone_a_reversal", 3000, now=3010)
        self.assertEqual([action.signal for action in restarted_first],
                         ["reverse-tone-b", "silence"])
        self.assertEqual(c._a_rev_count, 1)
        self.assertEqual(c.state, c.WAIT_A_REV2)

        # Only a later reversal after the fresh B reversal opens first-probe
        # reception and starts a new RTD measurement.
        self.assertEqual(c.feed("tone_a_reversal", 5000, now=5010), [])
        self.assertEqual(c.state, c.RX_PROBE)
        self.assertEqual(c.rtd_samples, 1360)

    def test_repeated_info0a_during_first_probe_aborts_and_restarts_ranging(self):
        c = V90DigitalPhase2Controller()
        c.start(0)
        c.feed("info0a", 100, payload={"ack_info0d": 0})
        c.feed("tone_a_reversal", 1000)
        c.feed("tone_a_reversal", 5000)
        self.assertEqual(c.state, c.RX_PROBE)

        # A real SmartLink call repeats INFO0a after we have already entered
        # first-probe receive. Its INFO0a frames carry ack=0 until the server
        # acknowledges the repeated exchange.
        actions = c.feed("info0a", 6000, payload={"ack_info0d": 0})
        self.assertEqual([action.signal for action in actions], ["info0d", "tone_b"])
        self.assertEqual(c.state, c.WAIT_A_REV2)
        self.assertEqual(c._a_rev_count, 0)
        self.assertFalse(c._first_probe_l1_done)
        self.assertIsNone(c.l2_started_at)
        self.assertIsNone(c._missing_tone_a_reversal_deadline_at)

        first = c.feed("tone_a_reversal", 7000)
        self.assertEqual([action.signal for action in first],
                         ["reverse-tone-b", "silence"])
        self.assertEqual(c.state, c.WAIT_A_REV2)
        self.assertEqual(c._a_rev_count, 1)

    def test_smartlink_recorded_info0a_recovery_trace_does_not_skip_ranging(self):
        """The saved SmartLink trace's post-recovery A reversal is only #1.

        The 2026-11-10 `slm-fu` run used the older answerer image and entered
        RX_PROBE on this reversal. The production controller now resets the
        ranging count when repeated INFO0a starts the recovery exchange.
        """
        c = V90DigitalPhase2Controller()
        c.start(38_400)
        c.feed("info0a", 38_420, now=39_701,
               payload={"ack_info0d": 0})
        first = c.feed("tone_a_reversal", 41_607, now=41_621)
        self.assertEqual([action.signal for action in first],
                         ["reverse-tone-b", "silence"])
        self.assertEqual(c.state, c.WAIT_A_REV2)
        self.assertEqual(c._a_rev_count, 1)

        # These are the CRC-valid repeated INFO0a observations in the saved
        # answerer trace, including their receive/ transmit clock positions.
        repeats = (
            (42_887, 44_502), (43_540, 44_502),
            (44_192, 46_102), (44_846, 46_102),
            (45_499, 46_102), (46_152, 47_703),
        )
        for media_at, tx_now in repeats:
            actions = c.feed("info0a", media_at, now=tx_now,
                             payload={"ack_info0d": 0})
            self.assertEqual([action.signal for action in actions],
                             ["info0d", "tone_b"])
            self.assertEqual(c.state, c.WAIT_A_REV2)
            self.assertEqual(c._a_rev_count, 0)
            self.assertIsNone(c.b_reversal_at)

        # The recorded second A reversal is the first one after recovery. The
        # paired RX replay finds steady 2400 Hz here, not an L1 comb, so moving
        # to receive-first-probe at this point is a premature state transition.
        actions = c.feed("tone_a_reversal", 51_206, now=51_222)
        self.assertEqual([action.signal for action in actions],
                         ["reverse-tone-b", "silence"])
        self.assertEqual(c.state, c.WAIT_A_REV2)
        self.assertEqual(c._a_rev_count, 1)
        self.assertIsNone(c._second_initial_a_reversal_at)

    def test_l1_precedes_l2_and_late_local_echo_fails(self):
        c = V90DigitalPhase2Controller(rtd_samples=100)
        c.start(0)
        c.feed("info0a", 1, payload={})
        c.feed("tone_a_reversal", 10)
        c.feed("tone_a_reversal", 1000)
        with self.assertRaisesRegex(ValueError, "L2 cannot complete"):
            c.feed("l2_complete", 1200)

        c.feed("l1_complete", 1200)
        c.feed("l2_complete", 2000)
        c.feed("tone_a", 2100)
        c.feed("tone_a_reversal_after_l2", 2200)
        c.feed("tone_a", 3000)
        actions = c.feed("l2_echo_complete",
                         c.l2_started_at + int(.550 * 8000) + c.rtd_samples + 1)
        self.assertEqual(c.state, c.FAILED)
        self.assertEqual([a.signal for a in actions], ["retrain-timeout"])

    def _late_echo(self, excess, tolerance):
        c = V90DigitalPhase2Controller(rtd_samples=100, l2_echo_tolerance_samples=tolerance)
        c.start(0)
        c.feed("info0a", 1, payload={})
        c.feed("tone_a_reversal", 10)
        c.feed("tone_a_reversal", 1000)
        c.feed("l1_complete", 1200)
        c.feed("l2_complete", 2000)
        c.feed("tone_a", 2100)
        c.feed("tone_a_reversal_after_l2", 2200)
        c.feed("tone_a", 3000)
        actions = c.feed("l2_echo_complete",
                         c.l2_started_at + int(.550 * 8000) + c.rtd_samples + excess)
        return c, [a.signal for a in actions]

    def test_echo_limit_is_exact_by_default_and_tolerance_is_opt_in(self):
        # The exact limit passes; one sample over fails; the default tolerance is 0.
        c, signals = self._late_echo(0, 0)
        self.assertNotEqual(c.state, c.FAILED)
        c, signals = self._late_echo(1, 0)
        self.assertEqual((c.state, signals), (c.FAILED, ["retrain-timeout"]))
        # With a tolerance of N samples, N over passes and N+1 over still fails.
        for tolerance in (1, 8):
            c, signals = self._late_echo(tolerance, tolerance)
            self.assertNotEqual(c.state, c.FAILED, tolerance)
            self.assertNotIn("retrain-timeout", signals)
            c, signals = self._late_echo(tolerance + 1, tolerance)
            self.assertEqual((c.state, signals), (c.FAILED, ["retrain-timeout"]), tolerance)
        with self.assertRaises(ValueError):
            V90DigitalPhase2Controller(l2_echo_tolerance_samples=-1)

    def test_missing_tone_a_after_l2_times_out_at_650ms_plus_rtd(self):
        c = self._to_second_probe()
        c.feed("tone_a_reversal_after_l2", 9500, now=9510)
        deadline = c._tone_a_deadline_at
        self.assertEqual(deadline,
                         c.l2_started_at + int(.650 * 8000) + c.rtd_samples)
        self.assertEqual(c.tick(deadline - 1), [])
        actions = c.tick(deadline)
        self.assertEqual(c.state, c.FAILED)
        self.assertEqual([a.signal for a in actions], ["retrain-timeout"])
        self.assertEqual(actions[0].at, deadline)

    def test_timely_tone_a_disarms_its_deadline_while_waiting_for_l2_echo(self):
        c = self._to_second_probe()
        c.feed("tone_a_reversal_after_l2", 9500, now=9510)
        deadline = c._tone_a_deadline_at
        self.assertIsNotNone(deadline)

        c.feed("tone_a", deadline - 1, now=deadline + 20)
        self.assertTrue(c._tone_a_during_second_l2)
        self.assertIsNone(c._tone_a_deadline_at)
        self.assertEqual(c.state, c.TX_PROBE)
        self.assertEqual(c.tick(deadline), [])
        self.assertEqual(c.state, c.TX_PROBE)

    def test_tone_a_received_after_its_deadline_starts_retrain_failure(self):
        c = self._to_second_probe()
        c.feed("tone_a_reversal_after_l2", 9500, now=9510)
        deadline = c._tone_a_deadline_at
        actions = c.feed("tone_a", deadline + 1, now=deadline + 20)
        self.assertEqual(c.state, c.FAILED)
        self.assertEqual([action.signal for action in actions],
                         ["retrain-timeout"])

    def test_timely_tone_a_does_not_waive_late_l2_echo_limit(self):
        c = self._to_second_probe()
        c.feed("tone_a_reversal_after_l2", 9500, now=9510)
        l2_start = c.l2_started_at
        tone_a_deadline = c._tone_a_deadline_at
        c.feed("tone_a", tone_a_deadline - 1500,
               now=tone_a_deadline - 1480)

        # A timely Tone A disarms only the 650 ms Tone-A detection timer.
        # The independent 550 ms + RTD echo limit still rejects a late echo.
        echo_at = l2_start + int(.550 * 8000) + c.rtd_samples + 1
        actions = c.feed("l2_echo_complete", echo_at, now=echo_at + 20)
        self.assertEqual(c.state, c.FAILED)
        self.assertEqual([action.signal for action in actions],
                         ["retrain-timeout"])

    def test_events_before_start_and_out_of_order_are_rejected(self):
        c = V90DigitalPhase2Controller()
        with self.assertRaisesRegex(RuntimeError, "start Phase 2"):
            c.feed("tone_a", 0)
        c.start(0)
        with self.assertRaisesRegex(ValueError, "invalid in state"):
            c.feed("tone_a", 1)


if __name__ == "__main__":
    unittest.main()
