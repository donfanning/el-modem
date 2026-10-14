import unittest

import real_call_ladder as L

NIST_OK = ("init: b'ATE0X4\\r\\r\\nOK\\r\\n'\n"
           "[ 12.9s] b'\\r\\nCONNECT 1200\\r\\n'\n"
           "[ 20.9s] b'\\r\\nNational Institute of Standards and Technology\\r\\nTelephone Time Service, Generator 3a\\r\\n'\n"
           "[ 22.1s] b'Enter the question mark character for HELP\\r\\n61321 26-10-08 10:10:55 25 0 +.0 045.0 UTC(NIST) *\\r\\n'\n"
           "TOTAL BYTES after CONNECT: 200\nhangup confirmed: True\n")
NOISE = ("init: b'ATE0X4\\r\\r\\nOK\\r\\n'\n[ 18.8s] b'\\r\\nCONNECT 2400\\r\\n'\n"
         "[ 22.0s] b'M\\xe0?????????????|\\xc7\\xd9M\\xd5B\\x13\\xa0 \\xf60\\x9d\\t-?7\\x01\\x02d\\x012\\x81\\xd0\\x92\\xe0Q\\xe31\\x08@\\x06\\x05,\\xbb\\xcb\\\\$=\\xd9\\x1d\\xec'\n"
         "[ 23.0s] b'\\x94\\xef\\xeavc\\x05\\x14Z8s\\x0f\\xd5t\\xd7\\x1a\\t\\x8e\\x10>y\\x16,m\\x96?<\\x8d\\x88\\xa2P5\\xb3-\\x89$'\n"
         "[ 47.2s] b'\\r\\nNO CARRIER\\r\\n'\nhangup confirmed: True\n")
NO_ANSWER = "init: b'ATE0X4\\r\\r\\nOK\\r\\n'\n[100.1s] b''\nhangup confirmed: True\n"


class JudgeTests(unittest.TestCase):
    def test_readable_text_after_connect_is_success(self):
        v = L.judge(NIST_OK)
        self.assertTrue(v["ok"], v)
        self.assertTrue(v["connected"])

    def test_connect_followed_by_noise_is_failure(self):
        v = L.judge(NOISE)
        self.assertTrue(v["connected"])
        self.assertFalse(v["ok"], v)

    def test_no_connect_is_failure(self):
        self.assertFalse(L.judge(NO_ANSWER)["ok"])

    def test_too_little_text_is_failure(self):
        short = "[ 1.0s] b'\\r\\nCONNECT 300\\r\\n'\n[ 2.0s] b'hi\\r\\n'\n"
        self.assertFalse(L.judge(short)["ok"])


class LadderTests(unittest.TestCase):
    def make_runner(self, outputs):
        calls = []

        def runner(name, number, init, env, secs):
            calls.append((name, number, init, env, secs))
            return outputs[len(calls) - 1]
        return runner, calls

    def test_stops_at_the_first_rung_that_works(self):
        runner, calls = self.make_runner([NOISE, NIST_OK, NIST_OK])
        res = L.run_ladder("nist", "13034944774", runner)
        self.assertEqual([r["rung"] for r in res], [1, 2])
        self.assertEqual([r["ok"] for r in res], [False, True])
        self.assertEqual(len(calls), 2)                          # rung 3 was never dialled
        self.assertEqual(calls[1][2], "ATE0X4+MS=B212,0,1200,1200")
        self.assertEqual(calls[0][0], "nist-rung1")

    def test_falls_through_to_300_and_respects_the_call_cap(self):
        runner, calls = self.make_runner([NOISE, NO_ANSWER, NIST_OK])
        res = L.run_ladder("n", "1", runner)
        self.assertEqual([r["rung"] for r in res], [1, 2, 3])
        self.assertEqual(calls[2][2], "ATE0X4+MS=B103,0,300,300")
        runner2, calls2 = self.make_runner([NOISE, NOISE, NOISE])
        res2 = L.run_ladder("n", "1", runner2, max_calls=2)
        self.assertEqual(len(calls2), 2)
        self.assertFalse(res2[-1]["ok"])

    def test_start_rung_skips_the_ones_above(self):
        runner, calls = self.make_runner([NIST_OK])
        L.run_ladder("n", "1", runner, start=3)
        self.assertEqual(calls[0][2], "ATE0X4+MS=B103,0,300,300")


if __name__ == "__main__":
    unittest.main()
