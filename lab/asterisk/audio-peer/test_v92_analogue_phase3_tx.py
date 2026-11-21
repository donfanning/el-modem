import unittest

import numpy as np

import v34_g711
import v34_tx
import v92_analogue_phase3_tx as tx
import v92_ja
from v92_phase3_rx import PatternTracker


def descriptor():
    return v92_ja.build_ja_descriptor(N=0, upstream_rates=(28000.0, 32000.0))


class AnaloguePhase3TxTests(unittest.TestCase):
    def test_segment_layout_and_lengths(self):
        plan = tx.build(descriptor())
        b = plan.boundaries
        self.assertEqual(b["silence"], (0, 560))
        self.assertEqual(b["ru"], (560, 560 + 384))
        self.assertEqual(b["ru-bar"], (944, 944 + 24))
        self.assertEqual(b["trn1u"][1] - b["trn1u"][0], 2052)
        self.assertEqual(b["trn1u"][0], 968)
        self.assertEqual((b["ja"][1] - b["ja"][0]) % 12, 0)
        self.assertEqual(plan.length, b["ja"][1])

    def test_ru_levels_and_trn1u_are_plus_minus_lu(self):
        plan = tx.build(descriptor(), lu=700.0)
        x = plan.samples
        b = plan.boundaries
        self.assertEqual(set(np.abs(x[b["ru"][0]:b["ja"][1]])), {700.0})
        self.assertTrue(np.all(x[:560] == 0))
        self.assertEqual(list(x[560:566]), [700, 700, 700, -700, -700, -700])
        self.assertEqual(list(x[944:950]), [-700, -700, -700, 700, 700, 700])

    def test_trn1u_is_the_scrambled_ones_sequence(self):
        plan = tx.build(descriptor())
        b = plan.boundaries
        scrambler = v34_tx.Scrambler("answer")
        expected = [-1.0 if scrambler.step(1) else 1.0 for _ in range(2052)]
        got = list(plan.samples[b["trn1u"][0]:b["trn1u"][1]] / 1000.0)
        self.assertEqual(got, expected)

    def test_ja_starts_with_24_ones_then_the_descriptor_and_round_trips(self):
        d = descriptor()
        plan = tx.build(d)
        self.assertEqual(plan.ja_bits[:24], (1,) * 24)
        self.assertEqual(list(plan.ja_bits[24:24 + len(d)]), d)
        b = plan.boundaries
        scrambler = v34_tx.Scrambler("answer")
        trn = [scrambler.step(1) for _ in range(2052)]
        ja = plan.samples[b["ja"][0]:b["ja"][1]]
        back = tx.recover_ja_bits(ja, trn[-1], trn)
        self.assertEqual(tuple(back), plan.ja_bits)

    def test_the_digital_detector_sees_ru_then_the_flip_through_g711(self):
        plan = tx.build(descriptor(), lu=900.0)
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                x = v34_g711.roundtrip(np.rint(plan.samples), law).astype(float)
                tracker = PatternTracker()
                events = []
                for i in range(0, len(x), 160):
                    events += tracker.feed(x[i:i + 160])
                names = [n for n, _ in events]
                self.assertEqual(names[:2], ["ru", "ru-bar"])
                self.assertLessEqual(abs(events[0][1] - 560), 6)
                self.assertLessEqual(abs(events[1][1] - 944), 6)

    def test_bad_inputs(self):
        d = descriptor()
        with self.assertRaises(NotImplementedError):
            tx.build(d, md_symbols=276)
        with self.assertRaises(ValueError):
            tx.build(d, trn1u_symbols=2028)
        with self.assertRaises(ValueError):
            tx.build(d, trn1u_symbols=2055)
        with self.assertRaises(ValueError):
            tx.build(d, ja_bits=30)
        with self.assertRaises(ValueError):
            tx.build([])
        with self.assertRaises(ValueError):
            tx.build(d, lu=0)


if __name__ == "__main__":
    unittest.main()
