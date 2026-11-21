import unittest

import numpy as np

import v92_equaliser as eq


def trn(n, lu=1000.0, seed=3):
    rng = np.random.default_rng(seed)
    return lu * np.where(rng.integers(0, 2, n) == 1, -1.0, 1.0)


class EqualiserTests(unittest.TestCase):
    def test_it_inverts_a_dispersive_line_with_unit_gain(self):
        for h in ([1.0, 0.45, -0.2, 0.1], [0.0, 0.0, 0.9, -0.3, 0.2], [0.7, 0.5, 0.3]):
            with self.subTest(h=h):
                s = trn(4000)
                r = np.convolve(s, h)[:len(s)]
                w, d, residual = eq.train(r, s)
                y = eq.StreamFilter(w).apply(r)
                err = y[200 + d:] - s[200:len(s) - d]
                self.assertLess(np.sqrt(np.mean(err ** 2)) / 1000.0, 0.05)
                # Another symbol sequence through the same line comes back at the same level and delay.
                s2 = trn(2000, lu=2000.0, seed=9)
                y2 = eq.StreamFilter(w).apply(np.convolve(s2, h)[:len(s2)])
                err2 = y2[200 + d:] - s2[200:len(s2) - d]
                self.assertLess(np.sqrt(np.mean(err2 ** 2)) / 2000.0, 0.05)

    def test_a_pure_delay_is_found(self):
        s = trn(3000)
        r = np.concatenate((np.zeros(17), s))[:len(s)]
        w, d, residual = eq.train(r, s)
        self.assertEqual(d, 17)
        self.assertLess(residual, 1.0)


    def test_affine_training_recovers_a_constant_output_bias(self):
        rng = np.random.default_rng(11)
        s = np.r_[np.full(1500, 1000.0), np.full(1500, -1000.0)]
        rng.shuffle(s)
        r = np.convolve(s, [1.0, 0.3, -0.1])[:len(s)] + 35.0
        w, delay, residual, bias = eq.train_affine(r, s)
        self.assertEqual(delay, 0)
        # FIR tap DC response and the explicit intercept share a component;
        # the model need not assign the entire offset to the intercept.
        self.assertLess(bias, -10.0)
        self.assertLess(residual, 1.0)

    def test_chunking_does_not_change_the_output(self):
        w = np.array([1.0, -0.4, 0.1, 0.05])
        x = np.random.default_rng(1).normal(0, 300, 1000)
        whole = eq.StreamFilter(w).apply(x)
        f = eq.StreamFilter(w)
        out = []
        for i in range(0, 1000, 160):
            out.append(f.apply(x[i:i + 160]))
        self.assertTrue(np.allclose(np.concatenate(out), whole))

    def test_adaptive_filter_matches_the_fixed_filter_and_tracks_a_drifting_line(self):
        w = np.array([1.0, -0.4, 0.1, 0.05])
        x = np.random.default_rng(1).normal(0, 300, 1000)
        fixed, adaptive = eq.StreamFilter(w), eq.AdaptiveStreamFilter(w, 0.05)
        for i in range(0, 1000, 160):
            self.assertTrue(np.allclose(adaptive.apply(x[i:i + 160]), fixed.apply(x[i:i + 160])))
        self.assertFalse(adaptive.adapt(0, np.zeros(4), np.ones(4, bool), 1.0))      # too few samples
        self.assertTrue(adaptive.adapt(0, np.zeros(40), np.ones(40, bool), 1.0))
        self.assertTrue(np.array_equal(adaptive.w, w))
        # The line's response moves from h0 to h1 while the filter, started on h0's inverse, follows it.
        rng = np.random.default_rng(5)
        h0, h1 = np.array([1.0, 0.3]), np.array([0.8, 0.45])
        s = trn(40000, seed=7)
        ramp = np.linspace(0.0, 1.0, len(s))
        r = s * (h0[0] + ramp * (h1[0] - h0[0])) + np.r_[0.0, s[:-1]] * (h0[1] + ramp * (h1[1] - h0[1]))
        r += rng.normal(0, 5, len(s))
        start, _, _ = eq.train(r[:3000], s[:3000], taps=16, max_delay=0)
        fixed, adaptive = eq.StreamFilter(start), eq.AdaptiveStreamFilter(start, 0.2)
        tail = []
        for i in range(0, len(s), 160):
            y = adaptive.apply(r[i:i + 160])
            adaptive.adapt(0, s[i:i + 160] - y, np.ones(len(y), bool), 1.0)
            tail.append(y)
        late = slice(len(s) - 4000, len(s))
        adaptive_err = np.sqrt(np.mean((np.concatenate(tail)[late] - s[late]) ** 2))
        fixed_err = np.sqrt(np.mean((fixed.apply(r)[late] - s[late]) ** 2))
        self.assertLess(adaptive_err, 0.15 * fixed_err)
        with self.assertRaises(ValueError):
            eq.AdaptiveStreamFilter(w, 0.0)


        with self.assertRaises(ValueError):
            eq.train(np.zeros(10), np.zeros(11))
        with self.assertRaises(ValueError):
            eq.train(np.zeros(100), np.zeros(100))


if __name__ == "__main__":
    unittest.main()
