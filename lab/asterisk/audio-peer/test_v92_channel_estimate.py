"""TRN1u channel estimation and the TRN1u -> design -> data pipeline."""

import random
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_g711 as g711
    import v92_channel_estimate as est
    import v92_phase3 as p3
    import v92_pcm_upstream as pu
    import v92_precoder_design as design
    from test_v92_precoder_design import levels

LU = 3000.0
TRN_SYMBOLS = 2040


def line():
    n = np.arange(41) - 20.0
    h = (0.85 * np.sinc(0.85 * n) - 0.075 * np.sinc(0.075 * n)) * np.hamming(41) * 0.7
    h = np.concatenate([h, np.zeros(10)])
    return h + 0.05 * np.roll(h, 9)


def through_codec(samples):
    return g711.decode(g711.encode(np.rint(samples).astype(np.int32), "pcmu"), "pcmu").astype(float)


@unittest.skipIf(np is None, "numpy not installed")
class ChannelEstimateTests(unittest.TestCase):
    def sent(self):
        return LU * np.array(p3.trn1u(TRN_SYMBOLS))

    def test_exact_recovery_without_quantisation(self):
        h = line()
        sent = self.sent()
        out = est.estimate(design.apply_channel(sent, h), sent, taps=len(h))
        self.assertLess(np.max(np.abs(out["h"] - h)), 1e-9)
        self.assertLess(out["residual_rms"], 1e-6)

    def test_mu_law_quantisation_is_small_measurement_noise(self):
        h = line()
        sent = self.sent()
        received = through_codec(design.apply_channel(sent, h))
        out = est.estimate(received, sent, taps=len(h))
        self.assertLess(np.max(np.abs(out["h"] - h)), 0.02)
        self.assertLess(out["residual_rms"] / out["signal_rms"], 0.02)

    def test_a_pure_delay_appears_as_leading_zero_taps(self):
        sent = self.sent()
        h = np.concatenate([np.zeros(5), [0.6, 0.2, -0.1]])
        out = est.estimate(design.apply_channel(sent, h), sent, taps=12)
        self.assertLess(np.max(np.abs(out["h"][:5])), 1e-9)
        self.assertAlmostEqual(out["h"][5], 0.6, places=9)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            est.estimate([1.0, 2.0], [1.0], taps=1)
        with self.assertRaises(ValueError):
            est.estimate(np.ones(20), np.ones(20), taps=16)

    def test_trn1u_to_design_to_data_pipeline(self):
        true = line()
        sent = self.sent()
        received = through_codec(design.apply_channel(sent, true))
        h = est.estimate(received, sent, taps=64)["h"]
        plan = design.design(h, prefilter_taps=64, postcursors=16, noise_floor=1e-3)
        self.assertTrue(plan["stable"])
        set_ = levels(count=120, step=5, start=30)
        profile = pu.PcmUpstreamProfile(
            constellations=(set_,), interval_set=(0,) * 12, moduli=(10, 10, 10, 5) * 3,
            precoder_p1=plan["p1"], prefilter_z2=plan["z2"])
        tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
        rng = random.Random(4)
        frames, sent_bits, wire = 30, [], []
        for _ in range(frames + 12):
            bits = [rng.randint(0, 1) for _ in range(profile.bits_per_frame)]
            sent_bits.append(bits)
            wire.extend(tx.encode_frame(bits))
        # The real channel is the TRUE response; the design only knew the estimate.
        received = through_codec(design.apply_channel(wire, true))
        aligned = received[plan["delay"]:plan["delay"] + 12 * frames]
        decoded = [rx.decode_frame([float(v) for v in aligned[12 * i:12 * i + 12]])
                   for i in range(frames)]
        self.assertEqual(decoded, sent_bits[:frames])


if __name__ == "__main__":
    unittest.main()
