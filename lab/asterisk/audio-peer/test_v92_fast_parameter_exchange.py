import unittest

import numpy as np
import v34_g711
import v92_fast_parameter_exchange as fpe
import v92_fpe_signals as fpe_signals
import v92_pcm_upstream as pu
import v92_session as session

UP1 = bytes((i * 37 + 5) & 0xFF for i in range(64))
DOWN1 = bytes((i * 91 + 17) & 0xFF for i in range(64))
UP2 = bytes((i * 11 + 3) & 0xFF for i in range(72))
DOWN2 = bytes((i * 29 + 201) & 0xFF for i in range(72))


class FastParameterExchangeTests(unittest.TestCase):
    def test_rm_rf_then_new_parameters_and_new_payloads(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 1, 7, 11, 240, 241, 247, 251):
                with self.subTest(law=law, delay=delay):
                    r1 = session.run(law, delay=delay, up_payload=UP1, down_payload=DOWN1)
                    self.assertTrue(r1["ok"], r1.get("reason"))
                    r2 = fpe.exchange(r1, law, delay=delay, up_payload=UP2, down_payload=DOWN2)
                    names = [n for n, _ in r2["events"]]
                    for step in ("rm", "rm-prime", "rf", "rf-bar"):
                        self.assertIn(step, names, r2["events"])
                    rf_at = next(value for name, value in r2["events"] if name == "rf")
                    rm_at = next(value for name, value in r2["events"] if name == "rm")
                    self.assertEqual(rm_at % 12, delay % 12)
                    self.assertEqual((rf_at - delay) % 12, 0)
                    self.assertTrue(r2["a2"].connected and r2["d2"].connected, r2["events"])
                    self.assertEqual(r2["d2"].data_rx.old_frames_remaining, 0)
                    self.assertEqual(r2["d2"].data_rx.fb1_errors, 0)
                    self.assertTrue(r2["d2"].data_rx.qualified)
                    self.assertEqual(r2["d2"].data_rx.b1_errors, 0)
                    self.assertEqual(r2["a2"].sent.count("FPE:FB1u"), 1)
                    self.assertLess(r2["a2"].sent.index("E"), r2["a2"].sent.index("FPE:FB1u"))
                    self.assertEqual(r2["up_received"][:len(UP2)], UP2)
                    self.assertEqual(r2["down_received"][:len(DOWN2)], DOWN2)

    def test_digital_initiated_exchange_keeps_frame_phase_and_payload(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 1, 7, 11, 240, 241, 247, 251):
                with self.subTest(law=law, delay=delay):
                    r1 = session.run(law, delay=delay, up_payload=UP1, down_payload=DOWN1)
                    self.assertTrue(r1["ok"], r1.get("reason"))
                    r2 = fpe.exchange_digital_initiated(
                        r1, law, delay=delay, up_payload=UP2, down_payload=DOWN2)
                    names = [name for name, _ in r2["events"]]
                    for step in ("rf", "rf-bar", "rm", "rm-prime"):
                        self.assertIn(step, names, r2["events"])
                    rf_at = next(value for name, value in r2["events"] if name == "rf")
                    rm_at = next(value for name, value in r2["events"] if name == "rm")
                    self.assertEqual((rf_at - delay) % 12, 0)
                    self.assertEqual((rm_at - 2 * delay) % 12, 0)
                    self.assertTrue(r2["a2"].connected and r2["d2"].connected,
                                    r2["events"])
                    self.assertEqual(r2["d2"].data_rx.old_frames_remaining, 0)
                    self.assertEqual(r2["d2"].data_rx.fb1_errors, 0)
                    self.assertTrue(r2["d2"].data_rx.qualified)
                    self.assertEqual(r2["d2"].data_rx.b1_errors, 0)
                    self.assertEqual(r2["a2"].sent.count("FPE:FB1u"), 1)
                    self.assertLess(r2["a2"].sent.index("E"), r2["a2"].sent.index("FPE:FB1u"))
                    self.assertEqual(r2["up_received"][:len(UP2)], UP2)
                    self.assertEqual(r2["down_received"][:len(DOWN2)], DOWN2)

    def test_rf_is_the_documented_pattern(self):
        r1 = session.run("pcmu", delay=0, up_payload=UP1, down_payload=DOWN1)
        both, rf, rbar = fpe.rf_signals(fpe.dm_profile(r1), "pcmu")
        self.assertEqual(len(both), 12 * (fpe.R_FRAMES + fpe.RBAR_FRAMES))
        self.assertEqual(both[:12], rf)
        self.assertEqual(both[-12:], rbar)
        self.assertNotEqual(rf, rbar)

    def test_streaming_rm_detector_handles_packet_and_frame_phase(self):
        for law in ("pcmu", "pcma"):
            r1 = session.run(law, delay=0, up_payload=UP1, down_payload=DOWN1)
            profile = r1["digital4"].data_rx.rx.profile
            tx = pu.PcmUpstreamTransmitter(profile)
            rm = pu.rm_intervals(profile.moduli)
            rmp = pu.rm_intervals(profile.moduli, prime=True)
            signal = np.concatenate([
                np.asarray(tx.encode_intervals(rm), dtype=float) for _ in range(fpe.R_FRAMES)
            ] + [
                np.asarray(tx.encode_intervals(rmp), dtype=float) for _ in range(fpe.RBAR_FRAMES)
            ])
            pcm = v34_g711.decode(v34_g711.encode(np.rint(signal).astype(np.int32), law), law)
            for lead in (0, 1, 7, 11, 159, 241):
                for size in (160, 53, 17):
                    for sample_phase in (lead % 12, None):
                        with self.subTest(law=law, lead=lead, chunk=size,
                                          sample_phase=sample_phase):
                            detector = fpe_signals.RmSequenceDetector(
                                profile, sample_phase=sample_phase)
                            stream = np.concatenate((np.zeros(lead), pcm))
                            found = []
                            for at in range(0, len(stream), size):
                                found.extend(detector.feed(stream[at:at + size]))
                            self.assertEqual(
                                found, [("rm", lead),
                                        ("rm-prime", lead + fpe.R_FRAMES * 12)])


if __name__ == "__main__":
    unittest.main()
