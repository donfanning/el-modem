"""Final-training gain recovery scored on separate B1 segments."""
import dataclasses
import json
import random
import unittest

import numpy as np

import v34_g711 as g711
import v92_pcm_upstream as pu
import v92_phase4_controller as ctl
from v92_pcm_upstream_rx import PcmUpstreamStreamReceiver


def profile(law):
    levels = sorted({int(x) for x in g711.decode(np.arange(256), law) if x > 0})[6:102:2]
    return pu.PcmUpstreamProfile((tuple(levels),), (0,) * 12,
                                 (30, 30, 30, 15) * 3, trellis=1, bits=50)


def signal(prof, payload):
    bits = ctl._scrambled_ones_and_payload(prof.bits_per_frame, payload)
    tx = pu.PcmUpstreamTransmitter(prof, record=True)
    out = []
    for i in range(0, len(bits), prof.bits_per_frame):
        out.extend(tx.encode_frame(bits[i:i + prof.bits_per_frame]))
    return np.asarray(out), tx.levels


def receive(prof, law, samples):
    rx = ctl._UpstreamDataRx(prof, law, calibrate_gain=True)
    for i in range(0, len(samples), 160):
        rx.feed(samples[i:i + 160])
    for _ in range(64):
        if not rx._held:
            break
        rx.feed([])
    return rx


class B1GainTests(unittest.TestCase):
    def test_known_gain_recovers_b1_and_unseen_payload_under_both_laws(self):
        payload = random.Random(1942).randbytes(1024)
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                prof = profile(law)
                samples, _ = signal(prof, payload)
                samples *= 0.94
                uncorrected = PcmUpstreamStreamReceiver(prof, law)
                uncorrected.feed(bytes(g711.encode(np.rint(samples), law)))
                self.assertGreater(uncorrected.b1_errors, 0)
                corrected = receive(prof, law, samples)
                self.assertTrue(corrected.gain_calibration["accepted"])
                self.assertAlmostEqual(corrected.gain, 0.94, delta=0.005)
                self.assertTrue(corrected.qualified)
                self.assertEqual(corrected.b1_errors, 0)
                self.assertEqual(corrected.payload()[:len(payload)], payload)
                self.assertFalse(corrected._held)

    def test_gain_that_hurts_unused_b1_symbols_is_rejected(self):
        prof = profile("pcmu")
        samples, _ = signal(prof, b"")
        split = ctl._UpstreamDataRx.SEARCH_FRAMES * pu.INTERVALS
        samples[:split] *= 0.94
        samples[split:] *= 1.06
        rx = receive(prof, "pcmu", samples)
        self.assertFalse(rx.gain_calibration["accepted"])
        self.assertGreater(rx.gain_calibration["validation_after"],
                           rx.gain_calibration["validation_before"])
        self.assertEqual(rx.gain, 1.0)
        self.assertFalse(rx.qualified)

    def test_silence_cannot_create_a_gain_and_diagnostics_remain_json(self):
        rx = receive(profile("pcmu"), "pcmu", np.zeros(576))
        self.assertFalse(rx.gain_calibration["candidate_valid"])
        self.assertFalse(rx.gain_calibration["accepted"])
        self.assertEqual(rx.gain, 1.0)
        json.dumps(rx.gain_calibration, allow_nan=False)

    def test_cpd_gain_scale_and_viterbi_recover_pcm_upstream_b1(self):
        gain = 768 / (4 * 65536)
        levels = (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236)
        prof = pu.PcmUpstreamProfile((levels,), (0,) * 12, (10, 10, 10, 5) * 3,
                                     trellis=1, bits=36, gain=gain)
        samples, _ = signal(prof, b"")
        rx = ctl._UpstreamDataRx(prof, "pcmu", rx_filter_mode="trn2u-fit",
                                 decoder_mode="viterbi32", cpd_gain=gain)
        from v92_equaliser import StreamFilter
        rx.install_trn2u_filter(
            StreamFilter([1.0]), delay=0, gain=pu.cpd_receive_scale(gain),
            target_rms=float(np.sqrt(np.mean(np.asarray(levels, dtype=float) ** 2))) * gain,
            training_target_rms=1.0, fit_residual=0.07,
            fractional_delay_samples=None)
        for start in range(0, len(samples), 160):
            rx.feed(samples[start:start + 160])
        self.assertTrue(rx.qualified)
        self.assertEqual(rx.b1_errors, 0)
        self.assertEqual(rx.rx.trellis_violations, 0)

    def test_trn2u_fit_rx_probe_is_opt_in_and_processes_linear_samples(self):
        prof = profile("pcmu")
        payload = b"probe-payload"
        samples, _ = signal(prof, payload)

        default_rx = ctl._UpstreamDataRx(prof, "pcmu")
        self.assertEqual(default_rx.rx_filter_diagnostic["mode"], "off")
        for start in range(0, len(samples), 160):
            default_rx.feed(samples[start:start + 160])
        self.assertTrue(default_rx.qualified)
        self.assertEqual(default_rx.b1_errors, 0)

        probe_rx = ctl._UpstreamDataRx(prof, "pcmu", rx_filter_mode="trn2u-fit",
                                      fitted_fractional_delay_samples=0.655)
        from v92_equaliser import StreamFilter
        target_rms = float(np.sqrt(np.mean(np.asarray(prof.constellations[0]) ** 2)))
        probe_rx.install_trn2u_filter(
            StreamFilter([1.0]), delay=0, gain=1.0, target_rms=target_rms,
            training_target_rms=1.0, fit_residual=0.07,
            fractional_delay_samples=0.655)
        for start in range(0, len(samples), 160):
            probe_rx.feed(samples[start:start + 160])
        self.assertEqual(probe_rx.rx_filter_diagnostic["state"], "active")
        self.assertEqual(probe_rx.rx_filter_diagnostic["coefficients"], [1.0])
        self.assertEqual(probe_rx.rx_filter_diagnostic["delay_samples"], 0)
        self.assertFalse(probe_rx.rx_filter_diagnostic["fractional_delay_applied"])
        self.assertTrue(probe_rx.qualified)
        self.assertEqual(probe_rx.b1_errors, 0)
        self.assertEqual(probe_rx.payload()[:len(payload)], payload)

    def test_trn2u_nlms_tracks_a_slow_gain_drift_that_breaks_the_fixed_filter(self):
        prof = profile("pcmu")
        payload = bytes(range(256)) * 12
        samples, _ = signal(prof, payload)
        drifted = samples * np.linspace(1.0, 1.06, len(samples))
        target_rms = float(np.sqrt(np.mean(np.asarray(prof.constellations[0]) ** 2)))
        from v92_equaliser import StreamFilter
        results = {}
        for mu in (0.0, 0.05):
            rx = ctl._UpstreamDataRx(prof, "pcmu", rx_filter_mode="trn2u-fit", nlms_mu=mu)
            rx.install_trn2u_filter(
                StreamFilter([1.0]), delay=0, gain=1.0, target_rms=target_rms,
                training_target_rms=1.0, fit_residual=0.07, fractional_delay_samples=None)
            for start in range(0, len(drifted), 160):
                rx.feed(drifted[start:start + 160])
            results[mu] = rx
        self.assertNotEqual(results[0.0].payload()[:len(payload)], payload)
        tracked = results[0.05]
        self.assertTrue(tracked.qualified)
        self.assertEqual(tracked.payload()[:len(payload)], payload)
        stats = tracked.rx_filter_diagnostic["nlms"]
        self.assertEqual(stats["mu"], 0.05)
        self.assertGreater(stats["updates"], 0)
        self.assertGreater(stats["tap_change"], 0.0)
        self.assertIsNone(results[0.0].rx_filter_diagnostic["nlms"])

    def test_trn2u_nlms_step_size_is_validated(self):
        prof = profile("pcmu")
        with self.assertRaises(ValueError):
            ctl._UpstreamDataRx(prof, "pcmu", rx_filter_mode="trn2u-fit", nlms_mu=-0.1)
        import os
        from unittest import mock
        for raw, ok in (("0.05", 0.05), ("0", 0.0), ("x", None), ("2", None)):
            with mock.patch.dict(os.environ, {"EL_MODEM_V92_TRN2U_NLMS_MU": raw}):
                if ok is None:
                    with self.assertRaises(ValueError):
                        ctl._trn2u_nlms_mu()
                else:
                    self.assertEqual(ctl._trn2u_nlms_mu(), ok)

    def test_prefilter_omission_does_not_change_constellation_reference(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                prof = dataclasses.replace(profile(law), precoder_z1=(0.01,),
                                           precoder_p1=(0.1, -0.02),
                                           prefilter_z2=(0.5, 0.1, 0.03), prefilter_p2=(0.1,))
                _, expected = signal(prof, b"")
                rx = ctl._UpstreamDataRx(prof, law, calibrate_gain=True)
                for _ in range(12):
                    rx._advance_gain_reference()
                self.assertEqual(rx._gain_tx.levels, expected)

    def test_fast_alignment_retains_every_exhaustive_zero_and_exact_payload(self):
        payload = random.Random(1943).randbytes(1024)
        for law in ("pcmu", "pcma"):
            for scale, prefix in ((1.0, 0), (0.94, 17), (0.94, 90)):
                with self.subTest(law=law, scale=scale, prefix=prefix):
                    prof = profile(law)
                    samples, _ = signal(prof, payload)
                    samples = np.concatenate((np.zeros(prefix), samples * scale))
                    octets = bytes(g711.encode(np.rint(samples), law))
                    rx = ctl._UpstreamDataRx(prof, law, align_search=prefix + 8)
                    for _ in range(12):
                        rx._advance_gain_reference()
                    scores = [(rx._score(octets, skip), skip) for skip in range(rx._search)]
                    zeroes = {skip for score, skip in scores if score == 0}
                    self.assertIn(prefix, zeroes)
                    self.assertTrue(zeroes.issubset(rx._zero_candidates(octets)))
                    for i in range(0, len(samples), 160):
                        rx.feed(samples[i:i + 160])
                    self.assertEqual(rx.alignment, min(scores)[1])
                    self.assertLessEqual(rx.gain_calibration["scored_alignments"],
                                         ctl._UpstreamDataRx.SCORES_PER_CALL)
                    self.assertTrue(rx.qualified)
                    self.assertEqual(rx.payload()[:len(payload)], payload)

    def test_no_zero_candidate_uses_complete_bounded_search(self):
        prof = profile("pcmu")
        rx = ctl._UpstreamDataRx(prof, "pcmu", align_search=11)
        # Silence has no valid B1 alignment. It must still receive the full
        # fallback search, with no more than the declared trials per callback.
        octets = bytes(g711.encode(np.zeros(800), "pcmu"))
        for _ in range(12):
            rx._advance_gain_reference()
        self.assertEqual(rx._zero_candidates(octets), [])
        expected = min((rx._score(octets, skip), skip) for skip in range(rx._search))[1]
        previous = 0
        for _ in range(16):
            rx.feed(np.zeros(160))
            self.assertLessEqual(len(rx._scores) - previous, rx.SCORES_PER_CALL)
            previous = len(rx._scores)
            if rx.alignment is not None:
                break
        self.assertEqual(len(rx._scores), rx._search)
        self.assertEqual(rx.alignment, expected)
        self.assertFalse(rx.qualified)


if __name__ == "__main__":
    unittest.main()
