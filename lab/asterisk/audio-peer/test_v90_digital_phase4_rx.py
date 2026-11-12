"""Tests for the exact-codeword V.90 Phase 4 receive mapper."""
import random
import unittest

from v90_digital_phase4 import V90DigitalPhase4Encoder
from v90_digital_phase4_rx import V90DigitalPhase4Receiver
from v90_dil import g711_ucode_codewords


def _profile(law: str) -> dict:
    # Six 16-point interval maps represent the selected 21-bit frame rate.
    maps = tuple(tuple(range(16 + 16 * index, 32 + 16 * index))
                 for index in range(6))
    return {
        "kind": "cp", "codec": law, "ack": 0,
        "rate": 21 * 8000 / 6,
        "spectral_sign_bits": 0, "lookahead": 0,
        "interval_constellations": tuple(range(6)),
        "constellations": maps,
        "codec_constellations": maps,
    }


class V90DigitalPhase4ReceiverTests(unittest.TestCase):
    def test_decodes_b1d_and_streaming_user_bits_for_both_laws(self):
        rng = random.Random(9001)
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                profile = _profile(law)
                encoder = V90DigitalPhase4Encoder(profile, law)
                frame_bits = encoder.s + encoder.k
                source = ([1] * (48 * frame_bits)
                          + [rng.randrange(2) for _ in range(120 * frame_bits)])
                codewords = encoder.encode_bits(source)
                self.assertEqual(len(codewords), 168 * 6)

                receiver = V90DigitalPhase4Receiver(profile, law)
                decoded = []
                offset = 0
                for width in (1, 2, 11, 3, 29, 7, 5, 41):
                    decoded.extend(receiver.decode_codewords(
                        codewords[offset:offset + width]))
                    offset += width
                decoded.extend(receiver.decode_codewords(codewords[offset:]))
                self.assertEqual(decoded, source)
                self.assertEqual(receiver._pending, [])

    def test_rejects_illegal_constellation_point(self):
        law = "pcmu"
        profile = _profile(law)
        receiver = V90DigitalPhase4Receiver(profile, law)
        table = g711_ucode_codewords(law)
        illegal = int(table[0])
        legal = int(table[16])
        frame = bytes([illegal] + [legal] * 5)
        with self.assertRaisesRegex(ValueError, "not legal in its CP interval"):
            receiver.decode_codewords(frame)

    def test_rejects_shaped_profile_until_inverse_is_supported(self):
        profile = _profile("pcma")
        profile["spectral_sign_bits"] = 1
        with self.assertRaisesRegex(NotImplementedError, "S_r=0 only"):
            V90DigitalPhase4Receiver(profile, "pcma")


if __name__ == "__main__":
    unittest.main()
