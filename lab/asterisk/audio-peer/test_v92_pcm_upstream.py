"""V.92 PCM-upstream mapper (ITU-T V.92 clause 6.4)."""

import random
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_g711 as g711
    import v92_pcm_upstream as pu


def mulaw_levels(count=64, start=1):
    codes = np.array([0xFF - i for i in range(start, start + count)], dtype=np.uint8)
    return tuple(float(v) for v in g711.decode(codes, "pcmu"))


def profile(**kw):
    sets = (mulaw_levels(64, 1), mulaw_levels(48, 20), mulaw_levels(32, 40))
    base = dict(constellations=sets, interval_set=(0, 1, 2, 0, 1, 2) * 2,
                moduli=(40, 40, 30, 20, 40, 40, 30, 20, 40, 40, 30, 20))
    base.update(kw)
    return pu.PcmUpstreamProfile(**base)


@unittest.skipIf(np is None, "numpy not installed")
class V92PcmUpstreamTests(unittest.TestCase):
    def test_cpd_receive_scale_inverts_the_transmit_gain(self):
        gain = 768 / (4 * 65536)
        self.assertAlmostEqual(pu.cpd_receive_scale(gain), 341.3333333333, places=8)
        with self.assertRaises(ValueError):
            pu.cpd_receive_scale(0.0)

    def test_modulus_encoder_matches_the_clause_6_4_1_algorithm(self):
        moduli = [3, 5, 2, 7] + [1] * 8
        total = 3 * 5 * 2 * 7
        # R = 150: s=1 (> (210-1)/2), d(f-1)=0 -> R0=150, then successive divisions.
        ks, state = pu.modulus_encode([(150 >> i) & 1 for i in range(8)], moduli, total, 0)
        self.assertEqual(ks[:4], [150 % 3, (150 // 3) % 5, (150 // 15) % 2, (150 // 30) % 7])
        self.assertEqual(state, 1)
        # With d(f-1)=1 the label is mirrored: R0 = M-1-R.
        ks, state = pu.modulus_encode([(150 >> i) & 1 for i in range(8)], moduli, total, 1)
        r0 = total - 1 - 150
        self.assertEqual(ks[:4], [r0 % 3, (r0 // 3) % 5, (r0 // 15) % 2, (r0 // 30) % 7])
        self.assertEqual(state, 0)           # s=1 xor d(f-1)=1

    def test_modulus_round_trip_with_differential_sign(self):
        rng = random.Random(7)
        moduli = [6, 7, 5, 4, 6, 7, 5, 4, 3, 3, 2, 2]
        total = 1
        for m in moduli:
            total *= m
        count = total.bit_length() - 1
        tx = rx = 0
        for _ in range(200):
            bits = [rng.randint(0, 1) for _ in range(count)]
            ks, tx = pu.modulus_encode(bits, moduli, total, tx)
            decoded, rx = pu.modulus_decode(ks, moduli, total, rx, count)
            self.assertEqual(decoded, bits)
            self.assertEqual(tx, rx)

    def test_round_trip_all_trellis_selections(self):
        for selection in (0, 1, 2):
            prof = profile(trellis=selection)
            tx, rx = pu.PcmUpstreamTransmitter(prof), pu.PcmUpstreamReceiver(prof)
            rng = random.Random(selection)
            for _ in range(60):
                bits = [rng.randint(0, 1) for _ in range(prof.bits_per_frame)]
                self.assertEqual(rx.decode_frame(tx.encode_frame(bits)), bits)
            self.assertEqual(rx.trellis_violations, 0)

    def test_fourth_point_parity_follows_the_trellis_output(self):
        prof = profile()
        tx = pu.PcmUpstreamTransmitter(prof)
        rng = random.Random(3)
        conv = pu.convolutional_encoder(prof.trellis)
        for _ in range(20):
            bits = [rng.randint(0, 1) for _ in range(prof.bits_per_frame)]
            samples = tx.encode_frame(bits)
            etas = []
            for i, value in enumerate(samples):
                levels = prof.constellations[prof.interval_set[i]]
                etas.append(min(range(-len(levels), len(levels)),
                                key=lambda e: abs(prof.level(i, e) - value)))
            for t in range(3):
                group = etas[4 * t:4 * t + 4]
                self.assertEqual(group[3] & 1, (sum(group[:3]) + conv.y0) & 1)
                pu._trellis_step(conv, group)

    def test_corrupting_a_point_is_flagged_by_the_trellis_check(self):
        prof = profile()
        tx, rx = pu.PcmUpstreamTransmitter(prof), pu.PcmUpstreamReceiver(prof)
        samples = tx.encode_frame([1, 0] * (prof.bits_per_frame // 2) +
                                  [0] * (prof.bits_per_frame % 2))
        levels = prof.constellations[prof.interval_set[3]]
        eta = min(range(-len(levels), len(levels)),
                  key=lambda e: abs(prof.level(3, e) - samples[3]))
        # Move the fourth point to a neighbour of the other parity.
        samples[3] = prof.level(3, eta + 1 if eta + 1 < len(levels) else eta - 1)
        try:
            rx.decode_frame(samples)
        except ValueError:
            pass
        self.assertGreaterEqual(rx.trellis_violations, 1)

    def test_selected_point_is_the_smallest_in_the_class_without_a_precoder(self):
        prof = profile()
        tx = pu.PcmUpstreamTransmitter(prof)
        samples = tx.encode_frame([0] * prof.bits_per_frame)
        # All-zero bits give R=0, so K0..K10 = 0; class {z*M} contains eta=0 -> the
        # smallest-magnitude positive point of the interval's set.
        self.assertEqual(samples[0], prof.constellations[prof.interval_set[0]][0])

    def test_expanded10_minimum_energy_representative_table(self):
        levels = (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236)
        prof = pu.PcmUpstreamProfile(
            constellations=(levels,), interval_set=(0,) * 12,
            moduli=(10, 10, 10, 5) * 3, trellis=1, bits=36)
        tx = pu.PcmUpstreamTransmitter(prof)

        # On the first three trellis positions, eta is K or K-10. The
        # minimum-energy point must mirror K=5..9 into negative levels.
        expected_plain = (96, 292, 492, 716, 988, -988, -716, -492, -292, -96)
        for k, expected in enumerate(expected_plain):
            options = tx._candidates(0, k, [])
            chosen = min(options, key=lambda eta: (abs(prof.level(0, eta)),
                                                   abs(prof.level(0, eta)), eta,
                                                   prof.level(0, eta)))
            self.assertEqual(prof.level(0, chosen), expected)

        # Slot 3 uses b=2*K+p, with candidates {b-10,b}. This table
        # catches the easy-to-miss K1 and K2 parity cases explicitly.
        expected_slot3 = (
            (96, 292), (492, 716), (988, -988),
            (-716, -492), (-292, -96),
        )
        for k, pair in enumerate(expected_slot3):
            for parity, expected in enumerate(pair):
                options = tx._candidates(3, k, [parity])
                b = 2 * k + parity
                self.assertEqual(options, [b - 10, b])
                chosen = min(options, key=lambda eta: (abs(prof.level(3, eta)),
                                                       abs(prof.level(3, eta)), eta,
                                                       prof.level(3, eta)))
                self.assertEqual(prof.level(3, chosen), expected)

    def test_precoder_and_prefilter_invert_the_channel(self):
        # x(1-P1) = u(1+Z1): a channel C = (1-P1)/(1+Z1) after the prefilter returns u.
        z1, p1 = (0.25, -0.125), (0.5, -0.25)
        prof = profile(precoder_z1=z1, precoder_p1=p1, prefilter_z2=(1.0,))
        tx = pu.PcmUpstreamTransmitter(prof)
        rng = random.Random(11)
        sent_u, received = [], []
        r_hist, v_hist = [], []
        for _ in range(8):
            bits = [rng.randint(0, 1) for _ in range(prof.bits_per_frame)]
            for v in tx.encode_frame(bits):
                # Channel: r(n) = v(n) - sum p1 r(n-k) ... realises C = (1-P1)/(1+Z1)
                num = v - sum(c * v_hist[-k] for k, c in enumerate(p1, 1) if len(v_hist) >= k)
                r = num - sum(c * r_hist[-k] for k, c in enumerate(z1, 1) if len(r_hist) >= k)
                v_hist.append(v)
                r_hist.append(r)
                received.append(r)
            sent_u = tx._u
        # Every received value must be a legal constellation point (the precoder aims
        # the channel output at u(n)).
        legal = {round(l, 6) for s in prof.constellations for l in s}
        legal |= {-x for x in legal}
        self.assertTrue(all(round(abs(r), 6) in {abs(x) for x in legal} for r in received))

    def test_survives_a_g711_codec_in_both_laws(self):
        # The analogue modem's levels are G.711 decoder outputs, so the network's
        # encoder returns the same codewords and the digital modem recovers every bit.
        for law, top in (("pcmu", 0xFF), ("pcma", 0xD5)):
            codes = np.array([top - 2 * i for i in range(1, 49)], dtype=np.uint8) \
                if law == "pcma" else np.array([top - i for i in range(1, 49)], dtype=np.uint8)
            levels = tuple(sorted({abs(float(v)) for v in g711.decode(codes, law)}))
            prof = pu.PcmUpstreamProfile(
                constellations=(levels,), interval_set=(0,) * 12,
                moduli=(30, 30, 30, 15) * 3, trellis=1)
            tx, rx = pu.PcmUpstreamTransmitter(prof), pu.PcmUpstreamReceiver(prof)
            rng = random.Random(5)
            for _ in range(40):
                bits = [rng.randint(0, 1) for _ in range(prof.bits_per_frame)]
                linear = np.rint(tx.encode_frame(bits)).astype(np.int32)
                wire = g711.encode(linear, law)
                back = [float(v) for v in g711.decode(wire, law)]
                self.assertEqual(back, [float(v) for v in linear])
                self.assertEqual(rx.decode_frame(back), bits)

    def test_profile_rejects_moduli_that_do_not_fit(self):
        with self.assertRaises(ValueError):
            profile(moduli=(200,) + (40,) * 11)           # more than the 128 points
        with self.assertRaises(ValueError):
            profile(moduli=(40, 40, 30, 90) + (40, 40, 30, 20) * 2)   # 2*M > points at k=3
        with self.assertRaises(ValueError):
            profile(interval_set=(0,) * 11 + (9,))

    def test_frame_length_is_enforced(self):
        prof = profile()
        with self.assertRaises(ValueError):
            pu.PcmUpstreamTransmitter(prof).encode_frame([0] * (prof.bits_per_frame - 1))


if __name__ == "__main__":
    unittest.main()
