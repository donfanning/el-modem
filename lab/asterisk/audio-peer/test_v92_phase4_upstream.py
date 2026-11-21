import unittest

import numpy as np

import v34_g711
import v92_cp
import v92_phase3
import v92_phase4_upstream as up


def cpu_bits(drn=14):
    return v92_cp.build_cpu(drn=drn, constellations=[(0, 3, 6, 9, 12, 15)], interval_sets=(0,) * 6)


class UpstreamPhase4Tests(unittest.TestCase):
    def test_trn2u_matches_the_existing_modulator(self):
        for points in (4, 8):
            expected = v92_phase3.Trn2uModulator(points).symbols(120)
            coder = up.SymbolCoder(points)
            got = coder.encode([1] * (120 * coder.bits))
            self.assertTrue(np.allclose(got, expected), points)

    def test_coder_round_trips_arbitrary_bits_and_keeps_state_across_calls(self):
        rng = np.random.default_rng(3)
        for points in (4, 8):
            bits = list(rng.integers(0, 2, 360).astype(int))
            tx, rx = up.SymbolCoder(points), up.SymbolCoder(points)
            half = (len(bits) // 2) // tx.bits * tx.bits
            symbols = tx.encode(bits[:half]) + tx.encode(bits[half:half + (len(bits) - half)
                                                              // tx.bits * tx.bits])
            decoded = rx.decode(symbols[:50]) + rx.decode(symbols[50:])
            self.assertEqual(decoded, bits[:len(decoded)], points)

    def test_suvu_and_cpu_survive_the_line_under_both_laws_and_constellations(self):
        suvu = v92_cp.build_suvu(level_db=-2.25, wait_for_cpu=True, silent_period=True)
        for points in (4, 8):
            for law in ("pcmu", "pcma"):
                with self.subTest(points=points, law=law):
                    burst = up.build_burst([suvu, suvu, cpu_bits()], points=points, lu=2000.0)
                    self.assertEqual(len(burst) % 12, 0)
                    line = v34_g711.roundtrip(np.rint(burst), law).astype(float)
                    frames = up.recover_frames(line, points=points, lu=2000.0)
                    self.assertEqual([f["kind"] for f in frames], ["suvu", "suvu", "cpu"])
                    self.assertEqual(frames[0]["level_db"], -2.25)
                    self.assertTrue(frames[0]["wait_for_cpu"] and frames[0]["silent_period"])
                    self.assertEqual(frames[2]["drn"], 14)
                    self.assertEqual(frames[2]["constellations"], ((0, 3, 6, 9, 12, 15),))

    def test_a_different_trn2u_length_and_initial_sign(self):
        for trn, sign in ((12, 0), (600, 1)):
            burst = up.build_burst([v92_cp.build_suvu()], trn_symbols=trn, initial_sign=sign)
            frames = up.recover_frames(burst, initial_sign=sign)
            self.assertEqual([f["kind"] for f in frames], ["suvu"])

    def test_trn2u_sign_seed_is_required_to_decode_the_all_ones_training(self):
        for points in (4, 8):
            coder = up.SymbolCoder(points, initial_sign=1)
            sent = [1] * (120 * coder.bits)
            symbols = coder.encode(sent)
            decoded = up.SymbolCoder(points, initial_sign=1).decode(symbols)
            wrong_seed = up.SymbolCoder(points, initial_sign=0).decode(symbols)
            self.assertEqual(decoded, sent)
            self.assertNotEqual(wrong_seed[:24], sent[:24])

    def test_trn2u_alone_has_no_frames(self):
        self.assertEqual(up.recover_frames(up.build_burst([])), [])

    def test_silent_renegotiation_second_trn2u_preserves_scrambler_and_sign_state(self):
        """Amendment 1 requires the second TRN2u to continue the first coder state."""
        for points in (4, 8):
            with self.subTest(points=points):
                coder = up.SymbolCoder(points)
                group = 12 * coder.bits

                def frame_bits(frame):
                    return list(frame) + [0] * (-len(frame) % group)

                frames = [v92_cp.build_suvu(ack=True), cpu_bits(14)]
                symbols = coder.encode([1] * (12000 * coder.bits))
                symbols.extend(coder.encode(frame_bits(frames[0])))
                symbols.extend(coder.encode(frame_bits(frames[1])))
                # E2u is a frame of scrambled zeroes; the following TRN2u is
                # deliberately sent by the same coder without resetting it.
                symbols.extend(coder.encode([0] * group))
                symbols.extend(coder.encode([1] * (8004 * coder.bits)))
                symbols.extend(coder.encode(frame_bits(frames[0])))
                symbols.extend(coder.encode(frame_bits(frames[1])))

                decoded = up.SymbolCoder(points).decode(np.asarray(symbols))
                offset = 12000 * coder.bits
                recovered = []
                for _ in range(2):
                    for parser, kind in ((v92_cp.parse_suvu, "suvu"),
                                         (v92_cp.parse_cpu, "cpu")):
                        parsed = parser(decoded[offset:])
                        self.assertIsNotNone(parsed, (points, kind, offset))
                        recovered.append(parsed["kind"])
                        offset += int(parsed["frame_bits"])
                        offset += -offset % group
                    offset += group              # E2u
                    offset += 8004 * coder.bits  # continuous second TRN2u
                self.assertEqual(recovered, ["suvu", "cpu", "suvu", "cpu"])

    def test_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            up.SymbolCoder(5)
        with self.assertRaises(ValueError):
            up.build_burst([], trn_symbols=100)
        with self.assertRaises(ValueError):
            up.SymbolCoder(8).encode([1, 0])


if __name__ == "__main__":
    unittest.main()
