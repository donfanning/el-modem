"""Retrain response in phases 3-4: the answerer abandons the exchange and returns to tone A (lab/V34_LCP1_REPLAY.md)."""
import os
import unittest
from unittest.mock import patch

import numpy as np

import v34_ans
import v34_call

FS = 8000


def tone_a(n, phase):
    t = (np.arange(n) + phase) / FS
    return 1400.0 * np.sin(2 * np.pi * 2400.0 * t) + 300.0 * np.sin(2 * np.pi * 1800.0 * t)


def loop(seconds=16.0, switch_stage=None, delay_ms=25.0, rate=4800):
    """Caller <-> answerer loopback. Once the caller reaches `switch_stage` it hears steady tone A instead of the answerer."""
    call = v34_call.V34Caller(symbol_rates=[2400], data_rate=rate)
    ans = v34_ans.V34Answerer(symbol_rates=[2400], data_rate=rate)
    d = int(delay_ms * 8)
    to_ans, to_call = [np.zeros(d)], [np.zeros(d)]
    switched_at, reached, phase = None, None, 0
    for tick in range(int(seconds * 50)):
        to_ans.append(call.tx(160))
        to_call.append(ans.tx(160))
        xa = np.concatenate(to_ans)[:160]
        xc = np.concatenate(to_call)[:160]
        to_ans = [np.concatenate(to_ans)[160:]]
        to_call = [np.concatenate(to_call)[160:]]
        if switch_stage and switched_at is None and call.stage == switch_stage:
            switched_at = tick * 0.02
        if switched_at is not None:
            xc = tone_a(160, phase)
            phase += 160
        call.rx(xc)
        ans.rx(xa)
        if reached is None and call.stage == "data" and ans.stage == "data":
            reached = tick * 0.02
    return call, ans, switched_at, reached


class LoopbackTests(unittest.TestCase):
    def test_healthy_call_never_retrains(self):
        call, ans, _, reached = loop(seconds=14.0)
        self.assertIsNotNone(reached)
        self.assertEqual(call._phase_retrains, 0)

    def test_tone_a_during_phase_four_is_answered_with_a_retrain(self):
        exercised = 0
        for stage in ("p4_trn", "phase3"):
            call, ans, switched, _ = loop(seconds=15.0, switch_stage=stage)
            if switched is None:
                continue                                          # the loopback passes through that stage too quickly to catch it
            exercised += 1
            self.assertEqual(call._phase_retrains, 1, stage)
            events = [e for _, e in call.events if "returned to tone A" in e]
            self.assertEqual(len(events), 1, stage)
            self.assertIn(call.stage, ("info0_ack", "b_reversal_wait", "b_done"), stage)   # ranging restarted, waiting for the A reversals
            self.assertEqual(call.data_rate, call._data_rate_initial)
            self.assertIsNone(call.negotiated)
        self.assertGreaterEqual(exercised, 1)

    def test_switch_disables_it(self):
        with patch.dict(os.environ, {"V34_PHASE_RETRAIN": "0"}):
            call, _, switched, _ = loop(seconds=12.0, switch_stage="p4_trn")
        self.assertIsNotNone(switched)
        self.assertEqual(call._phase_retrains, 0)
        self.assertEqual(call.stage, "p4_trn")


class UnitTests(unittest.TestCase):
    def caller(self, coherence, windows=400):
        call = v34_call.V34Caller()
        call._coh = [coherence] * windows
        call._lvl = [1000.0] * windows
        call._w2400 = [1j] * windows
        call._wins = windows
        call.stage = "p4_trn"
        return call

    def test_broadband_phase_four_signal_does_not_trigger(self):
        self.assertFalse(self.caller(0.06)._phase_retrain_due())

    def test_steady_tone_must_persist(self):
        call = self.caller(0.06)
        n = -(-int(call.PHASE_RETRAIN_TONE_S * FS) // v34_call.WIN)
        call._coh[-(n // 2):] = [0.9] * (n // 2)                  # only half the required time
        self.assertFalse(call._phase_retrain_due())
        call._coh[-n:] = [0.9] * n
        self.assertTrue(call._phase_retrain_due())

    def test_retrains_are_bounded_and_only_before_e(self):
        call = self.caller(0.9)
        fired = 0
        for _ in range(10):
            call.stage = "p4_mp"
            fired += call._phase_retrain_due()
        self.assertEqual(fired, call.PHASE_RETRAIN_MAX)
        call = self.caller(0.9)
        call.e_sent = True
        self.assertFalse(call._phase_retrain_due())

    def test_other_stages_are_left_alone(self):
        call = self.caller(0.9)
        for stage in ("info0", "probing", "toneB2", "info1c_sent", "data"):
            self.assertNotIn(stage, call.PHASE_RETRAIN_STAGES)



class ToneSkipTests(unittest.TestCase):
    """Once tone A has been steady for 0.2 s the phase-4 tracking (and its TRN-end rescan) is skipped."""

    def caller(self, coherence, windows=400):
        call = v34_call.V34Caller()
        call._coh = [coherence] * windows
        call._lvl = [1000.0] * windows
        call._w2400 = [1j] * windows
        call._wins = windows
        return call

    def test_tone_a_steady_threshold(self):
        call = self.caller(0.06)
        self.assertFalse(call._tone_a_steady(0.2))
        n = -(-int(0.2 * FS) // v34_call.WIN)
        call._coh[-(n - 1):] = [0.9] * (n - 1)
        self.assertFalse(call._tone_a_steady(0.2))
        call._coh[-n:] = [0.9] * n
        self.assertTrue(call._tone_a_steady(0.2))
        self.assertFalse(call._tone_a_steady(0.6))                # shorter than the retrain persistence

    def test_phase4_step_skips_the_receiver_only_while_the_tone_lasts(self):
        from types import SimpleNamespace
        fed = []
        rcv = SimpleNamespace(idx=[], trn_end=None, _end_est=None, frames=[], b1_symbol=None, e_span=None,
                              eq=SimpleNamespace(slips=0), feed=lambda parse_mp: fed.append(1))
        for coherence, expect_feeds in ((0.06, 1), (0.9, 0)):
            call = self.caller(coherence)
            call.stage, call._p4rcv, call._p4R = "p4_trn", rcv, SimpleNamespace(update_baseband=lambda x: None, _z=None)
            call._p4_x0, call._p4_last, call.rx_t, call._rx, call.ans_baud = 0, 0, 10 * FS, np.zeros(16), 3428.57
            fed.clear()
            try:
                call._phase4_rx()
            except AttributeError:
                pass                                              # the code after the feed needs a real receiver; only the feed matters
            self.assertEqual(len(fed), expect_feeds, coherence)

    def test_the_retrain_still_fires_with_the_guard(self):
        call, ans, switched, _ = loop(seconds=15.0, switch_stage="p4_trn")
        if switched is None:
            self.skipTest("the loopback passed through p4_trn too quickly to switch")
        self.assertEqual(call._phase_retrains, 1)


class AfterETests(unittest.TestCase):
    """Opt-in V34_RETRAIN_AFTER_E=1: tone A after our E, before the answerer's E/B1 was found, is a retrain too (gu4)."""

    def caller(self, coherence=0.9):
        call = UnitTests.caller(self, coherence)
        call.stage, call.e_sent, call._rx_b1 = "data", True, None
        return call

    def test_off_by_default(self):
        self.assertFalse(self.caller()._phase_retrain_due())

    def test_fires_after_e_without_b1(self):
        with patch.dict(os.environ, {"V34_RETRAIN_AFTER_E": "1"}):
            call = self.caller()
            self.assertTrue(call._awaiting_b1_after_e())
            self.assertTrue(call._phase_retrain_due())
            self.assertFalse(call.e_sent)
            self.assertEqual(call._phase_retrains, 1)
            self.assertIn("returned to tone A during data", call.events[-1][1] if call.events else "")

    def test_not_once_b1_was_found_or_in_other_stages(self):
        with patch.dict(os.environ, {"V34_RETRAIN_AFTER_E": "1"}):
            call = self.caller()
            call._rx_b1 = 9766
            self.assertFalse(call._awaiting_b1_after_e())
            self.assertFalse(call._phase_retrain_due())
            call = self.caller()
            call.stage = "p4_mp"
            self.assertFalse(call._awaiting_b1_after_e())
            self.assertFalse(call._phase_retrain_due())        # e_sent still blocks it outside the window

    def test_broadband_signal_does_not_trigger(self):
        with patch.dict(os.environ, {"V34_RETRAIN_AFTER_E": "1"}):
            self.assertFalse(self.caller(0.06)._phase_retrain_due())

    def test_bounded_and_switchable(self):
        with patch.dict(os.environ, {"V34_RETRAIN_AFTER_E": "1"}):
            call = self.caller()
            fired = 0
            for _ in range(10):
                call.stage, call.e_sent, call._rx_b1 = "data", True, None
                fired += call._phase_retrain_due()
            self.assertEqual(fired, call.PHASE_RETRAIN_MAX)
        with patch.dict(os.environ, {"V34_RETRAIN_AFTER_E": "1", "V34_PHASE_RETRAIN": "0"}):
            self.assertFalse(self.caller()._phase_retrain_due())


if __name__ == "__main__":
    unittest.main()
