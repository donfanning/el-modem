import dataclasses
import random
import unittest

import numpy as np

import v34_g711
import v34_tx
import v92_cp
import v92_data_mode as dm
import v92_pair_session as session
import v92_pcm_upstream as pu
import v92_pcm_upstream_rx as rx


def cpu_frame(law="pcmu", drn=4):
    masks = [tuple(range(16 + 16 * i, 32 + 16 * i)) for i in range(6)]
    return v92_cp.build_cpu(drn=drn, alaw=(law == "pcma"), constellations=masks,
                            interval_sets=(0, 1, 2, 3, 4, 5))


def cpd_frame():
    sets = [[int(v) for v in session.positive_levels("pcmu")[2:50]]]
    return v92_cp.build_cpd(drn=9, trellis=1, gain4=0x4000, moduli=[30, 30, 30, 15] * 3,
                            interval_sets=(0,) * 6, constellations=sets,
                            precoder={"z1": [], "p1": [], "z2": [], "p2": []})


class DataModeTests(unittest.TestCase):
    def test_downstream_b1d_and_data_from_a_parsed_cpu(self):
        payload = random.Random(1).randbytes(120)
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                parsed = v92_cp.parse_cpu(cpu_frame(law))
                profile = dm.downstream_profile_from_cpu(parsed)
                self.assertEqual(profile["codec"], law)
                result = dm.decode_downstream(profile, dm.encode_downstream(profile, payload))
                self.assertEqual(result["b1_errors"], 0)
                self.assertEqual(result["payload"][:len(payload)], payload)

    def test_upstream_data_from_a_parsed_cpd_through_the_streaming_receiver(self):
        parsed = v92_cp.parse_cpd(cpd_frame())
        # The CPd's gain G is the analogue modem's output gain, which a real line undoes; this test
        # line has unity gain, so G is set to 1 and everything else comes from the CPd.
        profile = dataclasses.replace(pu.profile_from_cpd(parsed), gain=1.0)
        tx = pu.PcmUpstreamTransmitter(profile)
        k = profile.bits_per_frame
        payload = random.Random(3).randbytes(150)
        scr = v34_tx.Scrambler("answer")
        data = [(b >> i) & 1 for b in payload for i in range(8)]
        data += [0] * (-len(data) % k)
        scrambled = [scr.step(b) for b in [1] * (dm.B1D_FRAMES * k) + data]
        samples = []
        for i in range(0, len(scrambled), k):
            samples.extend(tx.encode_frame(scrambled[i:i + k]))
        wire = (v34_g711.encode(np.rint(samples), "pcmu").astype(np.uint8) & 0xFF).tobytes()
        receiver = rx.PcmUpstreamStreamReceiver(profile, "pcmu")
        receiver.feed(wire)
        self.assertEqual(receiver.state, receiver.DATA, receiver.failure)
        self.assertEqual(receiver.take_bytes()[:len(payload)], payload)

    def test_a_cleardown_cpu_and_a_wrong_frame_are_refused(self):
        with self.assertRaises(ValueError):
            dm.downstream_profile_from_cpu(v92_cp.parse_cpu(cpu_frame(drn=0)))
        with self.assertRaises(ValueError):
            dm.downstream_profile_from_cpu({"kind": "cpd"})


if __name__ == "__main__":
    unittest.main()
