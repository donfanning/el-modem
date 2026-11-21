import unittest

import numpy as np

import v34_g711
import v92_anspcm
import v92_quick_connect as qc
import v92_short_phase1 as sp1
from v92_short_phase1 import Party, run


def events(p, name):
    return [(t, v) for t, k, v in p.events if k == name]


def first_tx(p, label):
    return next(t for t, k, v in p.events if k == "tx" and v == label)


class ShortPhase1Tests(unittest.TestCase):
    def test_analogue_calls_a_digital_answerer_figure_3(self):
        a = Party("analogue", calling=True, uqts_code=0b0100)
        d = Party("digital", calling=False, level_code=0b10)
        run(a, d, latency_ms=10)
        self.assertIsNotNone(a.phase2_at)
        self.assertIsNotNone(d.phase2_at)
        # The answerer is silent for 200 ms, then ANSam; the caller answers after hearing it for 1 s.
        self.assertEqual(first_tx(d, "ANSam"), 200)
        self.assertGreaterEqual(first_tx(a, "QC1a") - 200, sp1.ANSAM_DETECT_MS)
        # QCA1d follows QC1a, then 75 ms of silence, QTS then ANSpcm on the digital side.
        t_qca = first_tx(d, "QCA1d")
        t_qts = first_tx(d, "QTS")
        self.assertEqual(t_qts - (t_qca + sp1.QCA_MS), sp1.SILENCE_MS)
        self.assertEqual(first_tx(d, "ANSpcm") - t_qts, 102)
        # The analogue modem stopped CM when it heard QCA1d, recovered V from QTS, and sent TONEq after 1 s of ANSpcm.
        self.assertTrue(events(a, "stop-cm"))
        self.assertEqual(events(a, "qts")[0][1], qc.UQTS_UCODES[0b0100])         # Ucode 67 round-tripped via QC1a
        self.assertGreaterEqual(first_tx(a, "TONEq") - first_tx(d, "ANSpcm"), sp1.ANSPCM_DETECT_MS)
        # Both go to Phase 2 75 ms after the tone stops.
        self.assertLessEqual(abs(a.phase2_at - d.phase2_at), 2 * 10 + 5)
        self.assertEqual(d.ansp_level, -15.0)                                      # LM = 10 in QCA1d

    def test_digital_calls_an_analogue_answerer_figure_4(self):
        d = Party("digital", calling=True, level_code=0b00)
        a = Party("analogue", calling=False, uqts_code=0b1011)
        run(d, a, latency_ms=10)
        self.assertIsNotNone(d.phase2_at)
        self.assertIsNotNone(a.phase2_at)
        self.assertGreaterEqual(first_tx(d, "QC1d") - 200, sp1.ANSAM_DETECT_MS)
        self.assertEqual(events(a, "qts")[0][1], qc.UQTS_UCODES[0b1011])           # 82
        self.assertEqual(d.ansp_level, -9.5)
        # The calling digital modem starts QTS 75 ms after it hears QCA1a (no QCA of its own to send).
        self.assertGreaterEqual(first_tx(d, "QTS") - first_tx(a, "QCA1a"), sp1.QCA_MS - sp1.EDGE_MS)

    def test_no_toneq_within_two_seconds_falls_back_to_ansam_and_v8(self):
        a = Party("analogue", calling=True, deaf_to_toneq=True)
        d = Party("digital", calling=False)
        run(a, d, latency_ms=10, max_ms=8000)
        self.assertEqual(d.fallback, "v8")
        t_fb = events(d, "fallback")[0][0]
        self.assertGreaterEqual(t_fb - (first_tx(d, "QCA1d") + sp1.QCA_MS), sp1.FALLBACK_MS)
        self.assertIsNone(d.phase2_at)

    def test_the_qts_signal_is_the_documented_pattern(self):
        for law in ("pcmu", "pcma"):
            q = sp1.qts_codewords(67, law)
            self.assertEqual(len(q), 768 + 48)
            self.assertEqual(sp1.check_qts(q, law), 67)
            self.assertIsNone(sp1.check_qts(q[:-6] + q[:6], law))
            lin = v34_g711.decode(np.frombuffer(q, dtype=np.uint8), law).astype(int)
            v = lin[0]
            self.assertEqual(list(lin[:6]), [v, lin[1], v, -v, -lin[1], -v])           # +V +0 +V -V -0 -V
            self.assertEqual(list(lin[768:774]), [-v, -lin[1], -v, v, lin[1], v])      # QTS-bar
            self.assertEqual(q[:6] * 128, q[:768])

    def test_ansp_cm_level_codes_select_the_anspcm_level_and_the_tone_is_2100_hz(self):
        for code, level in qc.LEVELS_DBM0.items():
            for law in ("pcmu", "pcma"):
                tone = v92_anspcm.stream(8000, level, law)
                self.assertAlmostEqual(sp1.tone_hz(tone, law), 2100.0, delta=10.0)


if __name__ == "__main__":
    unittest.main()
