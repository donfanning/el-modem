"""Long-channel recovery and training/streaming alignment for the span candidate."""
import unittest

import numpy as np

from v34_rx import Phase3Rx, SymbolEq


class Phase4SpanTests(unittest.TestCase):
    def channel(self, span):
        rng = np.random.default_rng(8)
        symbols = rng.choice([-1, 1], 2000) + 1j * rng.choice([-1, 1], 2000)
        clean = np.repeat(symbols, 2)
        received = clean.copy()
        # An echo nine symbols earlier lies outside the smaller centered window.
        received[18:] += .35 * clean[:-18]
        receiver = Phase3Rx(4000, 1200, 'answer')
        receiver.lag, receiver.delay, receiver._z = 0, 0, received
        receiver.train_known(symbols[72:672], 64, span=span)
        return receiver, symbols

    def test_longer_window_recovers_held_out_delayed_echo(self):
        errors = {}
        for span in (12, 24):
            receiver, symbols = self.channel(span)
            # No validation symbols enter the tap fit.
            target = symbols[808:1308]
            residual = receiver.equalize(800, 500) - target
            errors[span] = np.mean(abs(residual)**2) / np.mean(abs(target)**2)
        self.assertLess(errors[24], .02)
        self.assertLess(errors[24], errors[12] / 4)

    def test_wider_streaming_window_matches_training_matrix(self):
        receiver, _ = self.channel(24)
        equalizer = SymbolEq(receiver, 800)
        window = equalizer.window(800)
        self.assertEqual(window.shape, (48,))
        np.testing.assert_allclose(window, receiver._design(800, 1)[0], atol=1e-12)
        self.assertAlmostEqual(abs(window @ equalizer.h - receiver.equalize(800, 1)[0]), 0)


if __name__ == '__main__':
    unittest.main()
