#!/usr/bin/env python3
"""Our V.34 caller against our V.34 answerer (in-process, no V.8): complete start-up and data both ways."""
import unittest

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

if numpy is not None:
    import v34_loop


@unittest.skipIf(numpy is None, "numpy not installed")
class LoopTests(unittest.TestCase):
    def check(self, **kw):
        call, ans = v34_loop.run(**kw)
        self.assertEqual((call.stage, ans.stage), ("data", "data"), (call.events[-3:], ans.events[-3:]))
        self.assertEqual(bytes(ans.rx_bytes), b"HELLO FROM THE CALLER\r\n" * 3)
        self.assertEqual(bytes(call.rx_bytes), b"HELLO FROM THE ANSWERER\r\n" * 3)
        return call, ans

    def test_4800_at_2400_baud(self):
        self.check(seconds=18, rate=4800, syms=(2400,))

    def test_14400_at_3200_baud_with_delay_and_noise(self):
        self.check(seconds=18, rate=14400, syms=(3200,), delay_ms=60, noise=30)

    def test_33600_at_3429_baud(self):
        call, ans = self.check(seconds=18, rate=33600, syms=(3429,))
        self.assertEqual(call.data_rate, ans.data_rate)


if __name__ == "__main__":
    unittest.main()
