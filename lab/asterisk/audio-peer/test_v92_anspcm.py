import math
import unittest

import numpy as np

import v34_g711
import v92_anspcm as an

# The first eight printed entries of Table 7 (-9.5 dBm0): mu-law and A-law codewords for k = 0..7.
TABLE7_HEAD = {"pcmu": bytes.fromhex("A15822C2A33825B0"), "pcma": bytes.fromhex("887608EE89130F9B")}


class AnspcmTests(unittest.TestCase):
    def test_the_first_printed_entries(self):
        for law, head in TABLE7_HEAD.items():
            self.assertEqual(an.codewords(-9.5, law)[:8], head, law)

    def test_period_and_tone_frequency(self):
        for law in ("pcmu", "pcma"):
            x = v34_g711.decode(np.frombuffer(an.stream(301 * 8, -12.0, law), dtype=np.uint8), law)
            spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            peak_hz = np.argmax(spectrum) * 8000 / len(x)
            self.assertAlmostEqual(peak_hz, 8000 * 79 / 301, delta=8000 / len(x) * 2)    # about 2100 Hz
            self.assertEqual(an.stream(602, -12.0, law)[:301], an.stream(602, -12.0, law)[301:])

    def test_levels_step_down_with_the_transmit_level(self):
        rms = {}
        for level in an.SCL:
            x = v34_g711.decode(np.frombuffer(an.codewords(level, "pcmu"), dtype=np.uint8), "pcmu")
            rms[level] = float(np.sqrt(np.mean(x.astype(float) ** 2)))
        order = [rms[l] for l in (-9.5, -12.0, -15.0, -18.0)]
        self.assertEqual(order, sorted(order, reverse=True))
        # 2.5 dB steps, 3 dB for the last one (1334, 1000, 708, 500): within a G.711 step or two.
        self.assertAlmostEqual(20 * math.log10(order[0] / order[1]), 2.5, delta=0.25)
        self.assertAlmostEqual(20 * math.log10(order[2] / order[3]), 3.0, delta=0.25)

    def test_a_phase_reversal_every_3612_symbols(self):
        for law in ("pcmu", "pcma"):
            s = an.stream(3612 * 2 + 301, -15.0, law)
            first, second, third = s[:3612], s[3612:7224], s[7224:]
            lin = lambda b: v34_g711.decode(np.frombuffer(b, dtype=np.uint8), law).astype(int)
            self.assertTrue(np.array_equal(lin(second), -lin(first)))         # inverted
            self.assertTrue(np.array_equal(lin(third), lin(first)[:len(third)]))   # and back
            # A stream cut across the boundary: the 12 symbols before it plain, the rest reversed.
            cut = an.stream(100, -15.0, law, start_symbol=3600)
            self.assertEqual(cut[12:], an.stream(88, -15.0, law, start_symbol=3612))
            self.assertEqual(cut[:12], an.stream(12, -15.0, law, start_symbol=3600))
            period = an.codewords(-15.0, law)                    # 3612 = 12 periods, so the block restarts at 0
            self.assertEqual(an.stream(4, -15.0, law, start_symbol=3612),
                             bytes(b ^ 0x80 for b in period[:4]))

    def test_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            an.codewords(-10.0, "pcmu")
        with self.assertRaises(ValueError):
            an.codewords(-9.5, "alaw")


if __name__ == "__main__":
    unittest.main()
