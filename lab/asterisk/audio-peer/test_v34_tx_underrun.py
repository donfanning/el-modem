import unittest
import v34_tx


class UnderrunCounterTests(unittest.TestCase):
    def test_empty_queue_is_counted_and_filled_queue_is_not(self):
        q = v34_tx.StreamQam(2400, 1800, 4)
        q.fill = None
        q.read(160)
        self.assertGreater(getattr(q, 'underrun_symbols', 0), 0)
        q2 = v34_tx.StreamQam(2400, 1800, 4)
        q2.push([1 + 0j] * 400)
        q2.fill = None
        q2.read(160)
        self.assertEqual(getattr(q2, 'underrun_symbols', 0), 0)


if __name__ == '__main__':
    unittest.main()
