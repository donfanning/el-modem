"""Idealised paired V.92 PCM-upstream session."""

import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v92_pair_session as session


def telephone_line():
    """Smooth low-loss line: a few taps of dispersion, a precursor and a weak echo, with no
    deep spectral nulls (a band-limited line with them is the open problem in the gap map)."""
    h = np.zeros(24)
    h[:6] = [0.12, 0.72, 0.30, -0.16, 0.08, -0.03]
    h[17] = 0.04
    return h


@unittest.skipIf(np is None, "numpy not installed")
class PairSessionTests(unittest.TestCase):
    PAYLOAD = bytes(range(256)) * 4

    def test_clean_line_both_laws(self):
        for law in ("pcmu", "pcma"):
            result = session.run(telephone_line(), self.PAYLOAD, law=law)
            self.assertTrue(result["connected"], (law, result["events"]))
            self.assertEqual(result["b1_errors"], 0, law)
            self.assertEqual(result["trellis_violations"], 0, law)
            self.assertTrue(result["payload_exact"], law)
            self.assertGreaterEqual(result["rate"], 24000)
            self.assertLessEqual(result["rate"], 48000)

    def test_events_follow_the_start_up_order(self):
        result = session.run(telephone_line(), b"hello world", law="pcmu")
        names = [name for _, name, _ in result["events"]]
        order = ["negotiated", "INFO1a", "channel estimated", "Jd", "Jp", "CPt"]
        self.assertEqual([n for n in names if n in order], order)
        tail = [n for n in names if n in ("CPd", "B1u")]
        self.assertEqual(tail[::2], ["CPd"] * (len(tail) // 2))      # each CPd is followed
        self.assertEqual(tail[1::2], ["B1u"] * (len(tail) // 2))     # by its B1u
        self.assertLess(names.index("CPt"), names.index("CPd"))

    def test_noise_costs_rate_never_correctness(self):
        for law in ("pcmu", "pcma"):
            for noise in (0.0, 40.0, 150.0):
                result = session.run(telephone_line(), self.PAYLOAD, law=law, noise_rms=noise)
                self.assertTrue(result["connected"], (law, noise, result["events"]))
                self.assertTrue(result["payload_exact"], (law, noise))

    def test_failed_b1u_is_followed_by_a_more_conservative_cpd(self):
        result = session.run(telephone_line(), self.PAYLOAD, law="pcma")
        retries = [e for e in result["events"] if e[1] == "B1u failed, renegotiating"]
        cpds = [e for e in result["events"] if e[1] == "CPd"]
        self.assertEqual(len(cpds), len(retries) + 1)
        rates = [e[2]["rate"] for e in cpds]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_hopeless_line_declines_pcm_upstream(self):
        result = session.run(telephone_line(), b"x", law="pcmu", noise_rms=4000.0)
        self.assertFalse(result["connected"])

    def test_a_degrading_line_is_renegotiated_down_without_a_new_training(self):
        payload = bytes(range(200)) * 4
        result = session.run(telephone_line(), payload, law="pcmu", noise_rms=0.0,
                             later_segments=[(payload, 200.0)])
        self.assertTrue(result["connected"], result["events"])
        first, second = result["segments"]
        self.assertTrue(first["payload_exact"] and second["payload_exact"])
        self.assertGreaterEqual(second["renegotiations"], 1)
        self.assertLess(second["rate"], first["rate"])
        names = [e[1] for e in result["events"]]
        self.assertIn("rate renegotiation", names)
        # One channel estimate only: renegotiation does not repeat Phase 3 training.
        self.assertEqual(names.count("channel estimated"), 1)

    def test_a_steady_line_needs_no_renegotiation(self):
        payload = bytes(range(100)) * 4
        result = session.run(telephone_line(), payload, law="pcma", noise_rms=40.0,
                             later_segments=[(payload, 40.0)])
        self.assertTrue(result["payload_exact"])
        self.assertEqual(result["segments"][1]["renegotiations"], 0)

    def test_all_trellis_selections(self):
        for selection in (0, 1, 2):
            result = session.run(telephone_line(), b"trellis selection test" * 20,
                                 law="pcmu", trellis=selection)
            self.assertTrue(result["payload_exact"], selection)


if __name__ == "__main__":
    unittest.main()
