"""Answerer retrain response: waveform, session ownership and bounded requests."""
import os
import unittest
from unittest.mock import patch
import numpy as np
from v34_answer_modem import V34AnswerModem
from v34_ans import V34Answerer, ToneASrc


class AnswerRetrainTests(unittest.TestCase):
    def tone(self, frequency=1200):
        return 900*np.sin(2*np.pi*frequency*np.arange(160)/8000)

    def test_response_silence_and_tone_then_ranging(self):
        answerer=V34Answerer(data_rate=33600)
        capabilities=dict(rate3429=1)
        answerer.respond_to_retrain(capabilities)
        capabilities['rate3429']=0
        self.assertEqual(answerer.info0c['rate3429'],1)
        np.testing.assert_array_equal(answerer.tx(560),np.zeros(560))
        self.assertEqual(answerer.stage,'retrain_silence')
        signal=answerer.tx(400)
        self.assertIsInstance(answerer._src,ToneASrc)
        self.assertEqual(answerer.stage,'toneA')
        self.assertEqual(answerer._t_a,560)
        self.assertGreater(float(np.mean(signal**2)),10000)
        for _ in range(6):answerer.rx(self.tone())
        self.assertEqual(answerer.stage,'wait_first_b')
        self.assertGreaterEqual(answerer.a_rev1_t,960)

    def test_first_call_lost_reversal_reconditions_twice_then_fails(self):
        # No respond_to_retrain(): exercise the newly enabled first-call path.
        answerer=V34Answerer()
        for retry in range(3):
            answerer.stage='wait_first_b'
            answerer._a_rev1_rx=answerer.rx_t
            answerer.rx(np.zeros(16000))
            self.assertEqual(answerer.stage,'toneA' if retry<2 else 'failed')
        self.assertEqual(answerer.failed_reason,'first B reversal timeout')
        self.assertEqual(answerer._retrain_first_b_retries,2)

    def test_lost_reversal_reconditions_receiver_with_bounded_retries(self):
        answerer=V34Answerer()
        answerer.respond_to_retrain(dict(rate3429=1))
        self.assertEqual(answerer._info0c_at,-8000)
        for retry in range(3):
            answerer.stage='wait_first_b'
            answerer._a_rev1_rx=answerer.rx_t
            answerer.rx(np.zeros(16000))
            self.assertEqual(answerer.stage,'toneA' if retry<2 else 'failed')
        self.assertEqual(answerer._retrain_first_b_retries,2)

    def test_request_replaces_only_core_after_consecutive_tone(self):
        with patch.dict(os.environ,{'V34_CALLER_RETRAIN':'1'}):
            modem=V34AnswerModem(data_rate=33600,with_v8=False)
            modem.state='data';modem.answerer.info0c=dict(rate3429=1)
            old=modem.answerer;v42=modem._v42
            for _ in range(3):modem._watch_for_caller_retrain(self.tone())
            self.assertIs(modem.answerer,old)
            modem._watch_for_caller_retrain(self.tone())
            self.assertIsNot(modem.answerer,old)
            self.assertIs(modem._v42,v42)
            self.assertTrue(modem.connected)
            self.assertTrue(modem._retraining)
            self.assertEqual(modem._answerer_args['data_rate'],33600)
            self.assertEqual(modem._phase,'p2')
            self.assertEqual(modem.answerer.stage,'retrain_silence')
            self.assertEqual(modem._retrains,1)
            modem._watch_for_caller_retrain(self.tone())
            self.assertEqual(modem._retrains,1)

    def test_noise_other_tones_silence_and_disabled_do_not_request(self):
        self.assertFalse(V34AnswerModem._is_tone_b(self.tone(2400)))
        self.assertFalse(V34AnswerModem._is_tone_b(np.zeros(160)))
        self.assertFalse(V34AnswerModem._is_tone_b(np.random.default_rng(1).normal(0,900,160)))
        with patch.dict(os.environ,{'V34_CALLER_RETRAIN':'0'}):
            modem=V34AnswerModem(with_v8=False);modem.state='data'
            for _ in range(10):modem._watch_for_caller_retrain(self.tone())
            self.assertEqual(modem._retrains,0)

    def test_interrupted_tone_and_request_limit(self):
        with patch.dict(os.environ,{'V34_CALLER_RETRAIN':'1'}):
            modem=V34AnswerModem(with_v8=False);modem.state='data'
            modem.answerer.info0c=dict(rate3429=1)
            for _ in range(3):modem._watch_for_caller_retrain(self.tone())
            modem._watch_for_caller_retrain(np.zeros(160))
            modem._watch_for_caller_retrain(self.tone())
            self.assertEqual(modem._retrains,0)
            modem._retrain_budget.used=modem.RETRAIN_MAX
            for _ in range(4):modem._watch_for_caller_retrain(self.tone())
            self.assertEqual(modem.state,'failed')
            self.assertIn('caller retrained again',modem.failed_reason)
