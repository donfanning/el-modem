"""Retrain response when no INFO1a follows INFO1c and the answerer still sends tone A (V.34 11.2.2.1.6, 11.5.1.2)."""
import os
import unittest
from unittest.mock import patch
import numpy as np

import v34_ans
import v34_call

FS = 8000


def loop(seconds=14.0, delay_ms=25.0, blank_to_answerer=None, rate=4800):
    """Caller <-> answerer; `blank_to_answerer(call, chunk)` -> chunk lets a test drop what the answerer hears."""
    call = v34_call.V34Caller(symbol_rates=[2400], data_rate=rate)
    ans = v34_ans.V34Answerer(symbol_rates=[2400], data_rate=rate)
    d = int(delay_ms * 8)
    to_ans, to_call = [np.zeros(d)], [np.zeros(d)]
    reached = None
    for tick in range(int(seconds * 50)):
        to_ans.append(call.tx(160)); to_call.append(ans.tx(160))
        xa = np.concatenate(to_ans)[:160]; xc = np.concatenate(to_call)[:160]
        to_ans = [np.concatenate(to_ans)[160:]]; to_call = [np.concatenate(to_call)[160:]]
        if blank_to_answerer:
            xa = blank_to_answerer(call, xa)
        call.rx(xc); ans.rx(xa)
        if reached is None and call.stage == 'data' and ans.stage == 'data':
            reached = tick * 0.02
    return call, ans, reached


def lose_info1c():
    """Silence everything the answerer hears from the moment the caller sends INFO1c until 0.5 s after it ends."""
    state = {}

    def blank(call, chunk):
        if call.stage == 'info1c_sent' and 'until' not in state:
            state['until'] = call.info1c_sent_t + FS // 2
        if 'until' in state and call.t < state['until']:
            return np.zeros_like(chunk)
        return chunk
    return blank


class Info1RetryTests(unittest.TestCase):
    def test_healthy_call_never_retries(self):
        call, ans, reached = loop()
        self.assertIsNotNone(reached)
        self.assertEqual(call._info1_retries, 0)

    def test_lost_info1c_makes_the_caller_answer_the_retrain(self):
        call, ans, reached = loop(blank_to_answerer=lose_info1c(), seconds=10.0)
        self.assertGreaterEqual(call._info1_retries, 1)
        self.assertLessEqual(call._info1_retries, call.INFO1_RETRY_MAX)
        self.assertTrue(any('answering the retrain' in e for _, e in call.events))

    def test_switch_disables_it(self):
        with patch.dict(os.environ, {'V34_INFO1_RETRY': '0'}):
            call, ans, reached = loop(blank_to_answerer=lose_info1c(), seconds=10.0)
        self.assertEqual(call._info1_retries, 0)

    def test_no_retry_inside_the_wait_or_without_tone_a(self):
        call = v34_call.V34Caller()
        call.stage = 'info1c_sent'
        call.info1c_sent_t = 10000
        call.rtd_samples = 800
        call._tone_a_on = lambda: True
        call.rx_t = 10000 + int(0.65 * FS)
        self.assertFalse(call._info1_retrain_due())
        call._tone_a_on = lambda: False
        call.rx_t = 10000 + int(1.5 * FS)
        self.assertFalse(call._info1_retrain_due())
        call._tone_a_on = lambda: True
        self.assertTrue(call._info1_retrain_due())
        self.assertEqual(call.stage, 'info0_ack')

    def test_retries_are_bounded(self):
        call = v34_call.V34Caller()
        call.info1c_sent_t = 0
        call._tone_a_on = lambda: True
        call.rx_t = 5 * FS
        n = 0
        for _ in range(10):
            call.stage = 'info1c_sent'
            n += call._info1_retrain_due()
        self.assertEqual(n, call.INFO1_RETRY_MAX)


if __name__ == '__main__':
    unittest.main()
