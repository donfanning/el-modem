"""Bootstrap batching must preserve the original seeded confidence gate."""
import unittest
import numpy as np
from v34_duplex_model import bootstrap_benefit


def reference(benefit):
    intervals = {}
    for width in (8, 16, 32):
        rng = np.random.default_rng(340170+width)
        draws = []
        for _ in range(400):
            starts = rng.integers(0, len(benefit), int(np.ceil(len(benefit)/width)))
            samples = (starts[:, None]+np.arange(width)) % len(benefit)
            draws.append(float(np.mean(benefit[samples.ravel()[:len(benefit)]])))
        intervals[str(width)] = np.percentile(draws, [1, 99]).tolist()
    return intervals


class BootstrapTests(unittest.TestCase):
    def test_seeded_intervals_are_exact(self):
        rng = np.random.default_rng(33600)
        for n in (1, 7, 32, 63, 500, 1000):
            values = rng.normal(size=n)
            for benefit in (values, values+.5, np.zeros(n), values[::2]):
                self.assertEqual(bootstrap_benefit(benefit), reference(benefit))


if __name__ == '__main__':
    unittest.main()
