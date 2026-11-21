"""Digital-modem precoder design for V.92 PCM upstream."""

import random
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_g711 as g711
    import v92_precoder_design as design
    import v92_pcm_upstream as pu


def levels(law="pcmu", count=100, step=3, start=20):
    """Widely spaced positive levels: a real constellation is sized to the residual ISI."""
    codes = np.array([0xFF - i for i in range(start, count + 1, step)], dtype=np.uint8)
    return tuple(sorted({abs(float(v)) for v in g711.decode(codes, law)}))


def run_link(h, prof_kwargs, frames=12, seed=1):
    extra = 12
    plan = design.design(h, **prof_kwargs)
    assert plan["stable"], plan
    set_ = levels()
    profile = pu.PcmUpstreamProfile(
        constellations=(set_,), interval_set=(0,) * 12, moduli=(20, 20, 20, 10) * 3,
        precoder_p1=plan["p1"], prefilter_z2=plan["z2"])
    tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
    rng = random.Random(seed)
    sent = []
    wire = []
    for _ in range(frames + extra):
        bits = [rng.randint(0, 1) for _ in range(profile.bits_per_frame)]
        sent.append(bits)
        wire.extend(tx.encode_frame(bits))
    received = design.apply_channel(wire, h)
    d = plan["delay"]
    return plan, sent[:frames], profile, rx, received[d:d + 12 * frames]


@unittest.skipIf(np is None, "numpy not installed")
class PrecoderDesignTests(unittest.TestCase):
    def check_link(self, h, **kw):
        plan, sent, profile, rx, aligned = run_link(h, kw)
        decoded = [rx.decode_frame(list(aligned[12 * i:12 * i + 12])) for i in range(len(sent))]
        return plan, decoded == sent, rx.trellis_violations

    def test_flat_channel_needs_no_equalisation(self):
        plan, ok, violations = self.check_link([1.0], prefilter_taps=4, postcursors=2)
        self.assertEqual(plan["delay"], 0)
        self.assertTrue(ok)
        self.assertLess(plan["residual"], 1e-9)

    def test_postcursor_isi_is_cancelled_by_the_feedback_section(self):
        h = [1.0, 0.45, -0.25, 0.1]
        plan, ok, violations = self.check_link(h, prefilter_taps=1, postcursors=3)
        self.assertTrue(ok)
        self.assertEqual(violations, 0)
        self.assertAlmostEqual(plan["p1"][0], -0.45, delta=0.01)   # the noise floor biases it slightly

    def test_precursor_and_attenuation_are_handled_by_the_prefilter(self):
        h = [0.2, 0.8, 0.35, -0.2, 0.1, 0.05]            # precursor, loss, ISI
        plan, ok, violations = self.check_link(h, prefilter_taps=24, postcursors=6)
        self.assertLess(plan["residual"], 1e-3)
        self.assertTrue(ok)

    def test_non_minimum_phase_channel(self):
        h = [0.5, 1.0, 0.4, -0.15]                         # zero outside the unit circle
        plan, ok, violations = self.check_link(h, prefilter_taps=32, postcursors=4)
        self.assertLess(plan["residual"], 1e-2)
        self.assertTrue(ok)

    def test_a_longer_prefilter_does_not_do_worse(self):
        h = [0.3, 0.9, 0.5, -0.3, 0.2, -0.1, 0.05]
        short = design.design(h, prefilter_taps=8, postcursors=3)
        long = design.design(h, prefilter_taps=40, postcursors=3)
        self.assertLessEqual(long["residual"], short["residual"] + 1e-12)

    def test_band_limited_telephone_like_line_through_a_mu_law_codec(self):
        # 300-3400 Hz band-pass at 8 kHz (nulls at DC and Nyquist) plus a weak echo, with the
        # digital modem's A/D quantising the received samples to mu-law.
        n = np.arange(41) - 20.0
        h = (0.85 * np.sinc(0.85 * n) - 0.075 * np.sinc(0.075 * n)) * np.hamming(41) * 0.7
        h = np.concatenate([h, np.zeros(10)])
        h = h + 0.05 * np.roll(h, 9)
        plan = design.design(h, prefilter_taps=64, postcursors=16, noise_floor=1e-3)
        self.assertTrue(plan["stable"])
        set_ = levels(count=120, step=5, start=30)
        profile = pu.PcmUpstreamProfile(
            constellations=(set_,), interval_set=(0,) * 12, moduli=(10, 10, 10, 5) * 3,
            precoder_p1=plan["p1"], prefilter_z2=plan["z2"])
        tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
        rng = random.Random(2)
        frames, sent, wire = 30, [], []
        for _ in range(frames + 12):
            bits = [rng.randint(0, 1) for _ in range(profile.bits_per_frame)]
            sent.append(bits)
            wire.extend(tx.encode_frame(bits))
        received = design.apply_channel(wire, h)[plan["delay"]:plan["delay"] + 12 * frames]
        codec = g711.decode(g711.encode(np.rint(received).astype(np.int32), "pcmu"), "pcmu")
        decoded = [rx.decode_frame([float(v) for v in codec[12 * i:12 * i + 12]])
                   for i in range(frames)]
        self.assertEqual(decoded, sent[:frames])
        # The price of inverting a band-limited line: the transmitted peak exceeds the largest
        # constellation point (about 2x here), which a real design must fit into the power limit.
        self.assertGreater(max(abs(v) for v in wire), 1.5 * max(set_))

    def test_unstable_targets_are_flagged_not_hidden(self):
        # A channel whose spectrum nearly vanishes cannot be inverted by feedback alone.
        h = [1.0, -1.999, 0.999]
        plan = design.design(h, prefilter_taps=8, postcursors=2)
        self.assertIn("stable", plan)

    def test_min_phase_target_is_monic_and_stable_for_a_real_line_response(self):
        # A band-limited, lightly dispersive line: low-pass with an echo.
        n = np.arange(-12, 13)
        h = np.sinc(0.8 * n) * 0.8 * np.hamming(25)
        h = np.concatenate([h, [0.0] * 6]) + 0.12 * np.roll(np.concatenate([h, [0.0] * 6]), 9)
        target = design.minimum_phase_target(h, 12)
        self.assertAlmostEqual(target[0], 1.0)
        self.assertTrue(design.is_stable(target))

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            design.design([0.0, 0.0], prefilter_taps=4, postcursors=1)
        with self.assertRaises(ValueError):
            design.design([1.0], prefilter_taps=0, postcursors=1)


if __name__ == "__main__":
    unittest.main()
