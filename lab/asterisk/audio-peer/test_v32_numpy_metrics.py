"""Opt-in shared V32 distances must preserve the reference receiver decisions."""
import math
import os
import random
import unittest
from unittest.mock import patch

from v32_caller import V32_TCM, ZTAB4
from v32_rx import ViterbiStream, slice_to

try:
    import numpy
except ImportError:
    numpy = None


@unittest.skipIf(numpy is None, 'optional NumPy not installed')
class SharedDistanceTests(unittest.TestCase):
    def test_metrics_decisions_and_complete_survivor_state(self):
        for rate in (7200, 9600, 12000, 14400):
            with self.subTest(rate=rate):
                reference = ViterbiStream(rate, backend='python')
                shared = ViterbiStream(rate, backend='numpy')
                points = ZTAB4 if rate == 7200 else list(V32_TCM[rate].values())
                rng = random.Random(3400)
                symbols = [complex(rng.uniform(-7, 7), rng.uniform(-7, 7)) for _ in range(512)]
                symbols += list(points)
                # Every pair midpoint includes exact decision boundaries and
                # interior ties. Preserve the reference's first-point choice.
                symbols += [(a+b)/2 for a in points for b in points]
                for z in symbols:
                    metrics, desired = shared.metrics_and_decision(z)
                    self.assertEqual(metrics, reference._metrics(z))
                    self.assertEqual(desired, slice_to(z, points))
                    self.assertEqual(shared.push(z, metrics=metrics), reference.push(z))
                    self.assertEqual(shared.sv, reference.sv)
                    self.assertEqual(shared.paths, reference.paths)
                    self.assertEqual(shared.prev, reference.prev)

    def test_both_sides_of_near_ties_and_large_finite_samples(self):
        shared = ViterbiStream(14400, backend='numpy')
        reference = ViterbiStream(14400, backend='python')
        points = list(V32_TCM[14400].values())
        for a, b in zip(points, points[1:]):
            midpoint = (a+b)/2
            for epsilon in (0, 1e-15, -1e-15, 1e-10, -1e-10):
                z = midpoint + complex(epsilon, -epsilon)
                metrics, desired = shared.metrics_and_decision(z)
                self.assertEqual(metrics, reference._metrics(z))
                self.assertEqual(desired, slice_to(z, points))
        for z in (complex(1e150, -1e150), complex(1e-150, -1e-150)):
            metrics, desired = shared.metrics_and_decision(z)
            self.assertEqual(metrics, reference._metrics(z))
            self.assertEqual(desired, slice_to(z, points))

    def test_default_and_explicit_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ViterbiStream(14400).backend, 'python')
        with patch.dict(os.environ, {'V32_NUMPY_METRICS': '1'}):
            self.assertEqual(ViterbiStream(14400).backend, 'numpy')
            for rate in (7200, 9600, 12000):
                self.assertEqual(ViterbiStream(rate).backend, 'python')
            self.assertEqual(ViterbiStream(14400, backend='python').backend, 'python')
        with self.assertRaises(ValueError):
            ViterbiStream(14400, backend='other')


if __name__ == '__main__':
    unittest.main()
