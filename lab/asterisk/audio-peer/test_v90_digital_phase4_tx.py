"""Tests for the opt-in digital-answerer Phase 4 transmit sequence."""
import unittest
from unittest.mock import patch

import v90_digital_phase4 as phase4
from v90_digital_phase4_tx import V90DigitalPhase4Transmitter


def _profile(kind: str, law: str = "pcma", drn: int = 9, *, ack: int = 0) -> dict:
    # Sixteen choices per interval can represent the 17-bit CP data frame.
    masks = tuple(tuple(range(16 + 16 * index, 32 + 16 * index))
                  for index in range(6))
    return {
        "kind": kind, "codec": law,
        "ack": ack,
        "rate": (drn + (8 if kind == "cpt" else 20)) * 8000 / 6,
        "spectral_sign_bits": 1, "lookahead": 0,
        "shape": (1.0, 0.0, 0.0, 0.0),
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
        self.assertEqual((len(mp1), len(mp2)), (30, 30))
        self.assertEqual(tx.mp_frames_sent, 2)

        tx.receive_cp(_profile("cp", drn=2))
        self.assertEqual(tx.state, tx.WAIT_CP_PRIME)
        prime = tx.next_mp()
        self.assertEqual(len(prime), 30)
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
        self.assertEqual(len(prime_and_tail), 30 + 300)
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
                self.assertEqual(len(prime_and_tail), 330)
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
