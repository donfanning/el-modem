"""Tests for the opt-in digital-answerer Phase 4 transmit sequence."""
import unittest
from unittest.mock import patch

import numpy as np

import v90_digital_phase4 as phase4
from v90_digital_phase4_tx import V90DigitalPhase4Transmitter


def _profile(kind: str, law: str = "pcma", drn: int = 9, *, ack: int = 0,
             lookahead: int = 0, shape=(1.0, 0.0, 0.0, 0.0)) -> dict:
    # Sixteen choices per interval can represent the 17-bit CP data frame.
    masks = tuple(tuple(range(16 + 16 * index, 32 + 16 * index))
                  for index in range(6))
    return {
        "kind": kind, "codec": law,
        "ack": ack,
        "rate": (drn + (8 if kind == "cpt" else 20)) * 8000 / 6,
        "spectral_sign_bits": 1, "lookahead": lookahead,
        "shape": shape,
        "interval_constellations": (0, 1, 2, 3, 4, 5),
        "constellations": masks,
        "codec_constellations": masks,
    }


class _ReferenceGpc:
    """List-based recurrence retained here as an independent test oracle."""

    def __init__(self):
        self.history = [0] * 23

    def step(self, bit):
        output = int(bit) ^ self.history[17] ^ self.history[22]
        self.history = [output, *self.history[:22]]
        return output


class V90DigitalPhase4TransmitterTests(unittest.TestCase):
    def test_lookahead_one_matches_exhaustive_two_frame_path_metric(self):
        profile = _profile("cpt", lookahead=1,
                           shape=(0.25, -0.25, 0.125, -0.375))
        probe = phase4.V90DigitalPhase4Encoder(profile, "pcma")
        rng = np.random.default_rng(9201)
        frame_bits = probe.s + probe.k
        inputs = [rng.integers(0, 2, frame_bits).tolist() for _ in range(2)]
        frames = [probe._prepare_shaped_frame(frame) for frame in inputs]

        a1, a2, b1, b2 = probe.shape_filter

        def options(q, t):
            if q == 0:
                return (("A", 0, tuple(t)),
                        ("B", 1, tuple(bit ^ 1 for bit in t)))
            return (("C", 0, tuple(bit ^ int(i % 2 == 0)
                                    for i, bit in enumerate(t))),
                    ("D", 1, tuple(bit ^ int(i % 2 == 1)
                                    for i, bit in enumerate(t))))

        def metric(signs_by_frame):
            x1 = y1 = v1 = 0.0
            total = 0.0
            for frame, signs in zip(frames, signs_by_frame):
                for sign, magnitude in zip(signs, frame.magnitudes):
                    x = magnitude if sign else -magnitude
                    y = x - b1 * x1 + a1 * y1
                    v = y - b2 * y1 + a2 * v1
                    total += v * v
                    x1, y1, v1 = x, y, v
            return total

        candidates = []
        for first_name, q1, first_signs in options(0, frames[0].t):
            for second_name, _, second_signs in options(q1, frames[1].t):
                candidates.append((metric((first_signs, second_signs)),
                                   (first_name, second_name), first_signs))
        _, _, expected_signs = min(candidates, key=lambda item: (item[0], item[1]))

        encoder = phase4.V90DigitalPhase4Encoder(profile, "pcma")
        wire = encoder.encode_bits([bit for frame in inputs for bit in frame])
        self.assertEqual(len(wire), 6)
        expected = bytes(int(encoder.ucode_octets[points[index]])
                          ^ (0 if sign else 0x80)
                          for points, index, sign in zip(
                              encoder.code_sets, frames[0].indices, expected_signs))
        self.assertEqual(wire, expected)

    def test_lookahead_pipeline_flushes_exact_frame_count_and_keeps_legal_rules(self):
        for depth in (1, 2, 3):
            with self.subTest(depth=depth):
                profile = _profile("cpt", lookahead=depth,
                                   shape=(0.25, -0.25, 0.125, -0.375))
                encoder = phase4.V90DigitalPhase4Encoder(profile, "pcma")
                frame_bits = encoder.s + encoder.k
                rng = np.random.default_rng(9000 + depth)
                bits = rng.integers(0, 2, frame_bits * 64).tolist()
                output = encoder.encode_bits(bits)
                self.assertEqual(len(output), (64 - depth) * 6)
                output += encoder.flush_lookahead()
                self.assertEqual(len(output), 64 * 6)
                self.assertEqual(len(encoder._shape_pipeline), 0)

    def test_lookahead_one_reduces_average_shaping_metric(self):
        shape = (0.25, -0.25, 0.125, -0.375)
        rng = np.random.default_rng(9210)
        base = phase4.V90DigitalPhase4Encoder(
            _profile("cpt", lookahead=0, shape=shape), "pcma")
        bits = rng.integers(0, 2, (1200, base.s + base.k)).tolist()

        def waveform(profile):
            encoder = phase4.V90DigitalPhase4Encoder(profile, "pcma")
            wire = encoder.encode_bits([bit for frame in bits for bit in frame])
            wire += encoder.flush_lookahead()
            return phase4.g711.decode(np.frombuffer(wire, dtype=np.uint8), "pcma")

        ld0 = waveform(_profile("cpt", lookahead=0, shape=shape))
        ld1 = waveform(_profile("cpt", lookahead=1, shape=shape))

        def energy(samples):
            a1, a2, b1, b2 = shape
            x1 = y1 = v1 = total = 0.0
            for sample in samples:
                x = float(sample)
                y = x - b1 * x1 + a1 * y1
                v = y - b2 * y1 + a2 * v1
                total += v * v
                x1, y1, v1 = x, y, v
            return total / len(samples)

        self.assertLessEqual(energy(ld1), energy(ld0))

    def test_trn2d_first_frame_magnitudes_are_independent_of_lookahead(self):
        rng = np.random.default_rng(9271)
        base = phase4.V90DigitalPhase4Encoder(_profile("cpt", lookahead=0), "pcma")
        frame_bits = base.s + base.k
        frames = rng.integers(0, 2, (4, frame_bits)).tolist()

        first_magnitudes = []
        for depth in range(4):
            encoder = phase4.V90DigitalPhase4Encoder(
                _profile("cpt", lookahead=depth), "pcma")
            bits = [bit for frame in frames for bit in frame]
            wire = encoder.encode_bits(bits)
            # Four input frames are enough to release frame zero even at ld=3.
            self.assertGreaterEqual(len(wire), 6)
            first = phase4.g711.decode(
                np.frombuffer(wire[:6], dtype=np.uint8), "pcma")
            first_magnitudes.append(np.abs(first))

        for magnitudes in first_magnitudes[1:]:
            np.testing.assert_array_equal(magnitudes, first_magnitudes[0])

    def test_s_r_zero_accepts_lookahead_field_without_changing_output(self):
        base = {**_profile("cpt", lookahead=0),
                "spectral_sign_bits": 0, "shape": ()}
        profile = {**base, "lookahead": 1}
        a = phase4.V90DigitalPhase4Encoder(base, "pcma")
        b = phase4.V90DigitalPhase4Encoder(profile, "pcma")
        bits = [int((i * 3) % 7 < 3) for i in range(400)]
        self.assertEqual(a.encode_bits(bits), b.encode_bits(bits))

    def test_packed_gpc_preserves_trn2d_and_continuing_mp_codewords(self):
        profile = _profile("cpt")
        mp_bits = [int((index * 7 + 3) % 11 < 5) for index in range(17)]
        fast = phase4.V90DigitalPhase4Encoder(profile, "pcma")
        fast_trn = fast.trn2d(2040)
        fast_mp = fast.encode_bits(mp_bits)

        with patch.object(phase4, "_Gpc", _ReferenceGpc):
            reference = phase4.V90DigitalPhase4Encoder(profile, "pcma")
            reference_trn = reference.trn2d(2040)
            reference_mp = reference.encode_bits(mp_bits)

        self.assertEqual(fast_trn, reference_trn)
        self.assertEqual(fast_mp, reference_mp)

    def test_phase4_signals_keep_exact_g711_lengths_and_order(self):
        tx = V90DigitalPhase4Transmitter(
            uinfo=78, law="pcma",
            mp={"max_a2d_drn": 14, "trellis": 0, "rate_mask": 0x1FFF})
        ri = tx.start()
        self.assertEqual(tx.state, tx.WAIT_CPT)
        self.assertEqual(len(ri), 192)
        self.assertEqual(len(tx.repeat_ri()), 24)

        prefix = tx.receive_cpt(_profile("cpt"))
        self.assertEqual(tx.state, tx.WAIT_CP)
        self.assertEqual(len(prefix), 24 + 2040)
        self.assertEqual(
            prefix[:24],
            phase4.rbar_i_codewords(78, "pcma", duration_symbols=24),
        )

        mp1 = tx.next_mp()
        mp2 = tx.next_mp()
        # Table 16 fills MP to whole CPt mapper frames (S+K = 16 here: 96 bits).
        self.assertEqual((len(mp1), len(mp2)), (36, 36))
        self.assertEqual(tx.encoder._pending_bits, [])
        self.assertEqual(tx.mp_frames_sent, 2)

        tx.receive_cp(_profile("cp", drn=2))
        self.assertEqual(tx.state, tx.WAIT_CP_PRIME)
        prime = tx.next_mp()
        self.assertEqual(len(prime), 36)
        self.assertEqual(tx.mp_prime_frames_sent, 1)

        tail = tx.receive_cp_prime(_profile("cp", drn=2, ack=1))
        self.assertEqual(tx.state, tx.DATA_READY)
        # Ed contributes two CPt-mapped six-symbol frames. B1d is 48 CP-mapped
        # six-symbol data frames, with fresh mapper memories.
        self.assertEqual(len(tail), 12 + 48 * 6)
        self.assertEqual(len(tail), 300)
        with self.assertRaisesRegex(RuntimeError, "MP is only sent"):
            tx.next_mp()

    def test_profile_law_and_message_order_are_enforced(self):
        tx = V90DigitalPhase4Transmitter(uinfo=78, law="pcmu")
        tx.start()
        with self.assertRaisesRegex(RuntimeError, "already started"):
            tx.start()
        with self.assertRaisesRegex(ValueError, "expected a parsed V.90 CPT"):
            tx.receive_cpt(_profile("cp", law="pcmu"))
        with self.assertRaisesRegex(ValueError, "law differs"):
            tx.receive_cpt(_profile("cpt", law="pcma"))
        tx.receive_cpt(_profile("cpt", law="pcmu"))
        with self.assertRaisesRegex(RuntimeError, "before sending an MP"):
            tx.receive_cp(_profile("cp", law="pcmu"))

    def test_end_signal_is_latched_until_first_mp_prime_is_sent(self):
        tx = V90DigitalPhase4Transmitter(uinfo=78, law="pcma")
        tx.start()
        tx.receive_cpt(_profile("cpt"))
        with self.assertRaisesRegex(RuntimeError, "after CP"):
            tx.receive_cp_prime(_profile("cp", drn=2, ack=1))
        tx.next_mp()
        tx.receive_cp(_profile("cp", drn=2))
        cp_prime = _profile("cp", drn=2, ack=1)
        cp_prime["start_bit"] = 1234
        self.assertEqual(tx.receive_cp_prime(cp_prime), b"")
        prime_and_tail = tx.next_mp()
        self.assertEqual(len(prime_and_tail), 36 + 300)
        self.assertEqual(tx.state, tx.DATA_READY)

    def test_acknowledged_cp_or_e_completes_training(self):
        for end_signal in ("cp-prime", "e"):
            with self.subTest(end_signal=end_signal):
                tx = V90DigitalPhase4Transmitter(uinfo=78, law="pcma")
                tx.start()
                tx.receive_cpt(_profile("cpt"))
                tx.next_mp()
                cp = _profile("cp", drn=2)
                tx.receive_cp(cp)
                if end_signal == "cp-prime":
                    tail = tx.receive_cp_prime(_profile("cp", drn=2, ack=1))
                else:
                    tail = tx.receive_e()
                self.assertEqual(tail, b"")
                prime_and_tail = tx.next_mp()
                self.assertEqual(len(prime_and_tail), 336)
                self.assertEqual(tx.state, tx.DATA_READY)

    def test_cp_prime_must_ack_selected_cp_profile(self):
        tx = V90DigitalPhase4Transmitter(uinfo=78, law="pcma")
        tx.start()
        tx.receive_cpt(_profile("cpt"))
        tx.next_mp()
        cp = _profile("cp", drn=2)
        tx.receive_cp(cp)
        tx.next_mp()
        with self.assertRaisesRegex(ValueError, "parameters differ"):
            tx.receive_cp_prime(_profile("cp", drn=3, ack=1))


if __name__ == "__main__":
    unittest.main()
