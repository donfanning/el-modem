import unittest

import v92_cp
import v92_phase4_downstream as down


def profile(law, drn=2):
    masks = tuple(tuple(range(16 + 16 * index, 32 + 16 * index)) for index in range(6))
    return {"kind": "cpt", "codec": law, "ack": 0, "rate": (drn + 8) * 8000 / 6,
            "spectral_sign_bits": 0, "lookahead": 0, "shape": (),
            "interval_constellations": (0, 1, 2, 3, 4, 5),
            "constellations": masks, "codec_constellations": masks}


def cpd():
    return v92_cp.build_cpd(
        drn=12, trellis=1, gain4=40000, ack=True,
        moduli=[7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12],
        precoder={"z1": [100, -200], "p1": [300], "z2": [], "p2": [-50, 25]},
        interval_sets=(0, 1, 0, 1, 0, 1), constellations=[[1, 5, 9, 20], [2, 6, 10]])


class DownstreamPhase4Tests(unittest.TestCase):
    def test_suvd_and_cpd_come_back_under_both_laws(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                p = profile(law)
                wire = down.build_downstream(
                    [v92_cp.build_suvd(silent_period=True), v92_cp.build_suvd(), cpd()], p, law)
                self.assertEqual(len(wire) % 6, 0)
                frames = down.recover_downstream(wire, p, law)
                self.assertEqual([f["kind"] for f in frames], ["suvd", "suvd", "cpd"])
                self.assertTrue(frames[0]["silent_period"])
                self.assertFalse(frames[1]["silent_period"])
                self.assertEqual(frames[2]["drn"], 12)
                self.assertTrue(frames[2]["ack"])

    def test_trn2d_alone_has_no_frames_and_lengths_are_checked(self):
        p = profile("pcmu")
        self.assertEqual(down.recover_downstream(down.build_downstream([], p, "pcmu"), p, "pcmu"), [])
        with self.assertRaises(ValueError):
            down.build_downstream([], p, "pcmu", trn_symbols=1000)

    def test_a_rate_that_needs_shaping_is_refused(self):
        p = profile("pcmu")
        p["spectral_sign_bits"] = 1
        p["shape"] = (1.0, 0.0, 0.0, 0.0)
        with self.assertRaises(NotImplementedError):
            down.recover_downstream(b"\xff" * 12, p, "pcmu")


if __name__ == "__main__":
    unittest.main()
