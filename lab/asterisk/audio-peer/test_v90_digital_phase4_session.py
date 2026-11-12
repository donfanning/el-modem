"""Clocked-session checks for V.90 digital answerer Phase 4."""
import unittest

from v90_digital_phase4_session import V90DigitalAnswererPhase4Session


def _profile(kind: str, *, law: str = "pcma", ack: int = 0, start_bit: int = 0,
             drn: int | None = None) -> dict:
    if drn is None:
        drn = 9 if kind == "cpt" else 2
    maps = tuple(tuple(range(16 + 16 * i, 32 + 16 * i)) for i in range(6))
    return {
        "kind": kind, "ack": ack, "start_bit": start_bit, "codec": law,
        "drn": drn, "rate": (drn + (8 if kind == "cpt" else 20)) * 8000 / 6,
        "spectral_sign_bits": 1, "lookahead": 0, "shape": (1., 0., 0., 0.),
        "interval_constellations": (0, 1, 2, 3, 4, 5),
        "constellations": maps, "codec_constellations": maps,
    }


class V90DigitalPhase4SessionTests(unittest.TestCase):
    def test_clocked_output_and_full_control_sequence_to_b1d(self):
        session = V90DigitalAnswererPhase4Session(
            uinfo=78, law="pcma", constellation=4,
            mp={"max_a2d_drn": 14, "rate_mask": 0x1FFF})
        with self.assertRaisesRegex(RuntimeError, "has not started"):
            session.read_g711(160)
        with self.assertRaisesRegex(RuntimeError, "before B1d"):
            session.write_data_bits([1] * 22)

        session.start()
        self.assertEqual(len(session.read_g711(192)), 192)
        self.assertEqual(session.state, session.transmitter.WAIT_CPT)

        cpt = _profile("cpt")
        self.assertTrue(session.receive_profile(cpt))
        self.assertFalse(session.receive_profile({**cpt, "start_bit": 1100}))
        # Ri(24T) + TRN2d(2040T) drain before the first MP is generated.
        self.assertEqual(len(session.read_g711(2064)), 2064)
        self.assertEqual(session.transmitter.mp_frames_sent, 0)
        self.assertEqual(len(session.read_g711(30)), 30)

        cp = _profile("cp")
        self.assertTrue(session.receive_profile(cp))
        self.assertEqual(session.state, session.transmitter.WAIT_CP_PRIME)
        self.assertEqual(len(session.read_g711(30)), 30)  # MP′

        cp_prime = _profile("cp", ack=1, start_bit=9876)
        self.assertTrue(session.receive_profile(cp_prime))
        self.assertFalse(session.receive_profile(cp_prime))
        self.assertEqual(len(session.read_g711(300)), 300)  # Ed + 48 B1d frames
        self.assertEqual(session.state, session.transmitter.DATA_READY)
        data_octets = session.write_data_bits([1, 0] * 11)
        self.assertEqual(data_octets, 6)
        self.assertEqual(len(session.read_g711(6)), 6)
        self.assertEqual(session.read_g711(160), b"")  # byte/LAPM framing is separate

        requested = []
        def bit_source(count):
            requested.append(count)
            return [1] * count
        self.assertEqual(len(session.read_g711(160, data_bit_source=bit_source)), 160)
        self.assertEqual(requested, [27 * 22])  # 27 six-symbol frames cover 160 samples.

    def test_e_can_arrive_before_mp_prime_and_is_latched(self):
        session = V90DigitalAnswererPhase4Session(
            uinfo=78, law="pcmu", constellation=16)
        session.start()
        session.read_g711(192)
        session.receive_profile(_profile("cpt", law="pcmu"))
        session.read_g711(2064)
        session.read_g711(30)  # MP
        session.receive_profile(_profile("cp", law="pcmu"))
        session.receive_e()  # upstream E detector has validated 20 bits
        self.assertFalse(session.receive_profile(
            _profile("cp", law="pcmu", ack=1)))  # equivalent CP′ after E
        self.assertEqual(session.state, session.transmitter.WAIT_CP_PRIME)
        self.assertEqual(len(session.read_g711(330)), 330)  # MP′ + Ed + B1d
        self.assertEqual(session.state, session.transmitter.DATA_READY)

    def test_repeated_cp_prime_is_idempotent_before_mp_prime_finishes(self):
        session = V90DigitalAnswererPhase4Session(
            uinfo=78, law="pcma", constellation=4)
        session.start()
        session.read_g711(192)
        session.receive_profile(_profile("cpt"))
        session.read_g711(2064)
        session.read_g711(30)
        session.receive_profile(_profile("cp"))
        cp_prime = _profile("cp", ack=1, start_bit=2000)
        self.assertTrue(session.receive_profile(cp_prime))
        self.assertFalse(session.receive_profile({**cp_prime, "start_bit": 4000}))
        self.assertEqual(len(session.read_g711(330)), 330)
        self.assertEqual(session.state, session.transmitter.DATA_READY)

    def test_rejects_cp_prime_that_changes_selected_cp(self):
        session = V90DigitalAnswererPhase4Session(
            uinfo=78, law="pcma", constellation=4)
        session.start()
        session.read_g711(192)
        session.receive_profile(_profile("cpt"))
        session.read_g711(2064)
        session.read_g711(30)
        session.receive_profile(_profile("cp"))
        session.read_g711(30)
        with self.assertRaisesRegex(ValueError, "parameters differ"):
            session.receive_profile(_profile("cp", ack=1, drn=3))


if __name__ == "__main__":
    unittest.main()
