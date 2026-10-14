#!/usr/bin/env python3
"""Back-to-back V.8: our caller wrapper against SpanDSP's own answerer."""
import unittest

import spandsp_v8 as v8


@unittest.skipUnless(v8.available(), "libspandsp not installed")
class V8Tests(unittest.TestCase):
    def test_caller_and_answerer_agree_on_v32(self):
        caller = v8.V8Negotiator(True, modulations=v8.V8_MOD_V32)
        answerer = v8.V8Negotiator(False, modulations=v8.V8_MOD_V32 | v8.V8_MOD_V34 | v8.V8_MOD_V22)
        for _ in range(2000):                                   # 20 ms chunks = 40 s max
            a = answerer.tx(160)
            caller.rx(a)
            c = caller.tx(160)
            answerer.rx(c)
            if caller.status == 2 and answerer.status == 2:
                break
        self.assertEqual((caller.status_name, answerer.status_name), ("v8-call", "v8-call"))
        self.assertTrue(caller.result["modulations"] & v8.V8_MOD_V32)
        self.assertEqual(caller.result["call_function"], v8.V8_CALL_V_SERIES)


if __name__ == "__main__":
    unittest.main()
