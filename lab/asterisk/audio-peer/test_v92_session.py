import unittest

import v92_session as session

UP = bytes((i * 37 + 5) & 0xFF for i in range(96))
DOWN = bytes((i * 91 + 17) & 0xFF for i in range(96))


class WholeSessionTests(unittest.TestCase):
    def test_phase_3_phase_4_and_a_payload_each_way(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 240, 800):
                with self.subTest(law=law, delay=delay):
                    r = session.run(law, delay=delay, up_payload=UP, down_payload=DOWN)
                    self.assertTrue(r["ok"], r.get("reason"))
                    self.assertEqual(r["digital3"].cpt["drn"], 2)
                    self.assertEqual(r["profile"]["kind"], "cpt")
                    self.assertTrue(r["analogue4"].connected and r["digital4"].connected)
                    self.assertEqual(r["up_received"][:len(UP)], UP)
                    self.assertEqual(r["down_received"][:len(DOWN)], DOWN)


if __name__ == "__main__":
    unittest.main()
