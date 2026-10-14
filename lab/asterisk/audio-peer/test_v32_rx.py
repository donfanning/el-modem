#!/usr/bin/env python3
"""Loopback test: V32CallerTx (answerer role, 8 kHz) -> V32Rx -> bytes."""
import unittest

from v32_caller import V32CallerTx
from v32_caller import V32_TCM_9600, rate_word
from v32_rx import V32Rx, deframe_async, expected_trn, trellis_bytes


class LoopbackTests(unittest.TestCase):
    def test_text_survives_8khz_audio_loopback(self):
        text = b"Hello, V.32 receiver!\r\n"
        tx = V32CallerTx(role="answerer", sample_rate=8000, trn_length=2400, my_rates=(4800,))
        tx.start_training_now(gap_symbols=40)
        # silence, then training, then E + data
        audio = [0] * 4000
        audio += tx.tx_samples(int(8000 * (40 + 256 + 16 + 2400) / 2400))
        tx.request_e()
        audio += tx.tx_samples(8000 // 2)
        tx.feed_bytes(text)                     # data well after E, as on a real link
        audio += tx.tx_samples(8000 * 2)
        rx = V32Rx(8000, tap=5)
        trn_start = (4000 / 8000) + (40 + 256 + 16) / 2400
        bits, diag = rx.run(audio, trn_start, search_s=0.2, trn_len=1800)
        self.assertGreater(diag["corr"], 0.9)
        self.assertEqual(deframe_async(bits).count(text), 1)

    def test_7200_trellis_text_survives_loopback(self):
        text = b"Trellis coded 7200 bps!\r\n"
        tx = V32CallerTx(role="answerer", sample_rate=8000, trn_length=2400, my_rates=(7200,))
        tx.start_training_now(gap_symbols=40)
        audio = [0] * 4000
        audio += tx.tx_samples(int(8000 * (40 + 256 + 16 + 2400) / 2400))
        audio += tx.tx_samples(8000 // 4)       # a few rate words first (the TRN edge is noisy)
        tx.request_e()
        audio += tx.tx_samples(8000 // 2)
        tx.feed_bytes(text)
        audio += tx.tx_samples(8000 * 2)
        rx = V32Rx(8000, tap=5)
        trn_start = (4000 / 8000) + (40 + 256 + 16) / 2400
        bits, diag = rx.run(audio, trn_start, search_s=0.2, trn_len=1800)
        self.assertGreater(diag["corr"], 0.9)
        self.assertEqual(trellis_bytes(diag["ys"], bits, 0xF000 | rate_word(7200)).count(text), 1)

    def test_9600_trellis_text_survives_loopback(self):
        text = b"Trellis coded 9600 bps, V.32!\r\n"
        tx = V32CallerTx(role="answerer", sample_rate=8000, trn_length=2400, my_rates=(9600,))
        tx.start_training_now(gap_symbols=40)
        audio = [0] * 4000
        audio += tx.tx_samples(int(8000 * (40 + 256 + 16 + 2400) / 2400))
        audio += tx.tx_samples(8000 // 4)
        tx.request_e()
        audio += tx.tx_samples(8000 // 2)
        tx.feed_bytes(text)
        audio += tx.tx_samples(8000 * 2)
        rx = V32Rx(8000, tap=5, points=list(V32_TCM_9600.values()))
        trn_start = (4000 / 8000) + (40 + 256 + 16) / 2400
        bits, diag = rx.run(audio, trn_start, search_s=0.2, trn_len=1800, mu_dd=0.0)
        self.assertGreater(diag["corr"], 0.9)
        self.assertEqual(
            trellis_bytes(diag["ys"], bits, 0xF000 | rate_word(9600), rate=9600).count(text), 1)


if __name__ == "__main__":
    unittest.main()
