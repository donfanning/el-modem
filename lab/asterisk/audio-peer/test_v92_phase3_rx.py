import unittest

import numpy as np

import v34_g711
import v92_phase3 as p3
from v92_phase3_rx import PatternTracker


def line(symbols, *, gain=300.0, law="pcmu", noise=0.0, lowpass=0.0, seed=1):
    x = np.asarray(symbols, dtype=float) * gain
    if lowpass:
        y = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = lowpass * acc + (1 - lowpass) * v
            y[i] = acc
        x = y
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, len(x))
    return v34_g711.roundtrip(np.rint(x), law).astype(float)


def run(tracker, x, chunk=160):
    events = []
    for i in range(0, len(x), chunk):
        events.extend(tracker.feed(x[i:i + chunk]))
    return events


def fractional_delay(x, delay):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        current = x[i]
        previous = x[i - 1] if i >= 1 else 0.0
        older = x[i - 2] if i >= 2 else 0.0
        out[i] = (((delay - 1.0) * (delay - 2.0) / 2.0) * current
                  - delay * (delay - 2.0) * previous
                  + (delay * (delay - 1.0) / 2.0) * older)
    return out


class PatternTrackerTests(unittest.TestCase):
    def test_batched_classifier_matches_single_window_reference(self):
        rng = np.random.default_rng(90210)
        blocks = rng.normal(0.0, 350.0, (512, 6))
        for idx, pattern in enumerate((
                p3.ru(1), p3.ru(1, bar=True), p3.su(1), p3.su(1, bar=True))):
            blocks[idx] = np.asarray(pattern, dtype=float) * 300.0
        # DC and silence have zero centered energy; weak patterns must remain
        # rejected even when their normalized correlation is perfect.
        blocks[4:7] = np.asarray([np.zeros(6), np.full(6, 200.0),
                                 np.asarray(p3.ru(1)) * 10.0])
        for families in (("ru",), ("su",), ("ru", "su")):
            tracker = PatternTracker(families=families)
            batch = tracker._classify_many(blocks)
            expected = [tracker._classify(block) for block in blocks]
            actual = [None if int(value) < 0 else
                      ("ru", "ru-bar", "su", "su-bar")[int(value)]
                      for value in batch]
            self.assertEqual(actual, expected)

    def test_ru_then_ru_bar_is_reported_at_the_flip_under_both_laws(self):
        for law in ("pcmu", "pcma"):
            for lead in (0, 1, 3, 5, 17):
                with self.subTest(law=law, lead=lead):
                    sig = [0.0] * lead + p3.ru(64) + p3.ru(64, bar=True) + [0.0] * 120
                    events = run(PatternTracker(), line(sig, law=law))
                    self.assertEqual([n for n, _ in events], ["ru", "ru-bar"])
                    self.assertIn((events[0][1] - lead) % 6, (0, 3))
                    self.assertLessEqual(abs(events[0][1] - lead), 6 * 1)
                    self.assertLessEqual(abs(events[1][1] - (lead + 384)), 6)

    def test_su_and_su_bar(self):
        sig = [0.0] * 40 + p3.su(40) + p3.su(40, bar=True) + [0.0] * 60
        events = run(PatternTracker(), line(sig, gain=400.0))
        self.assertEqual([n for n, _ in events], ["su", "su-bar"])
        self.assertLessEqual(abs(events[0][1] - 40), 6)
        self.assertLessEqual(abs(events[1][1] - (40 + 240)), 6)

    def test_fractional_su_tracker_finds_both_edges_across_sample_phases(self):
        for delay in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
            for lead in (0, 1, 2, 3, 4, 5):
                with self.subTest(delay=delay, lead=lead):
                    symbols = ([0.0] * lead + p3.su(40)
                               + p3.su(40, bar=True) + [0.0] * 60)
                    received = fractional_delay(line(symbols, gain=400.0), delay)
                    events = run(PatternTracker(families=("su",), fractional_timing=True),
                                 received)
                    self.assertEqual([name for name, _ in events], ["su", "su-bar"])
                    self.assertLessEqual(abs(events[0][1] - lead), 6, events)
                    self.assertLessEqual(abs(events[1][1] - (lead + 240)), 6, events)

    def test_gain_noise_and_a_band_limited_line(self):
        for gain in (60.0, 300.0, 3000.0):
            with self.subTest(gain=gain):
                sig = [0.0] * 11 + p3.ru(80) + p3.ru(80, bar=True)
                events = run(PatternTracker(), line(sig, gain=gain, noise=gain * 0.15,
                                                    lowpass=0.5))
                self.assertEqual([n for n, _ in events], ["ru", "ru-bar"])

    def test_chunk_size_does_not_change_the_events(self):
        sig = [0.0] * 7 + p3.ru(50) + p3.ru(50, bar=True)
        x = line(sig)
        reference = run(PatternTracker(), x, chunk=len(x))
        for chunk in (1, 13, 160, 1000):
            self.assertEqual(run(PatternTracker(), x, chunk=chunk), reference, chunk)

    def test_silence_noise_and_data_like_signals_do_not_trigger(self):
        rng = np.random.default_rng(7)
        for name, x in (
                ("silence", np.zeros(16000)),
                ("alaw idle", np.full(16000, 8.0)),
                ("noise", rng.normal(0, 200, 40000)),
                ("random pcm", rng.integers(-3000, 3000, 40000).astype(float)),
                ("random binary", rng.choice([-300.0, 300.0], 40000)),
                ("tone 1200", 300 * np.cos(2 * np.pi * 1200 * np.arange(16000) / 8000)),
                ("tone 2400", 300 * np.cos(2 * np.pi * 2400 * np.arange(16000) / 8000)),
                ("phase 2 style tone 600", 300 * np.cos(2 * np.pi * 600 * np.arange(16000) / 8000)),
        ):
            with self.subTest(name):
                self.assertEqual(run(PatternTracker(), x), [])

    def test_a_short_burst_is_not_enough(self):
        sig = [0.0] * 30 + p3.ru(5) + [0.0] * 120
        self.assertEqual(run(PatternTracker(), line(sig)), [])

    def test_the_pattern_is_lost_after_silence_and_found_again(self):
        sig = p3.ru(30) + [0.0] * 600 + p3.su(30)
        events = run(PatternTracker(), line(sig))
        self.assertEqual([n for n, _ in events], ["ru", "su"])

    def test_the_expected_family_decides_the_name(self):
        # Ru and Su correlate at 0.82, so each family is restricted to what the call expects.
        sig = [0.0] * 20 + p3.su(60) + p3.su(60, bar=True)
        x = line(sig, lowpass=0.7, gain=3000.0)
        su_only = run(PatternTracker(families=("su",)), x)
        self.assertEqual(su_only, [("su", 20), ("su-bar", 380)])
        self.assertEqual([n for n, _ in run(PatternTracker(families=("ru",)), line(
            [0.0] * 20 + p3.ru(60) + p3.ru(60, bar=True), gain=300.0))], ["ru", "ru-bar"])
        # The other family correlates 0.82: at the default 0.8 it is accepted under the expected
        # name; rejecting it needs a stricter threshold, which a clean Su still passes.
        ru_x = line([0.0] * 20 + p3.ru(60) + p3.ru(60, bar=True))
        self.assertEqual(run(PatternTracker(families=("su",), min_correlation=0.9), ru_x), [])
        su_x = line([0.0] * 20 + p3.su(60) + p3.su(60, bar=True))
        self.assertEqual([n for n, _ in run(PatternTracker(families=("su",), min_correlation=0.9),
                                            su_x)], ["su", "su-bar"])

    def test_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            PatternTracker(confirm=1)
        with self.assertRaises(ValueError):
            PatternTracker().feed([float("nan")])
        with self.assertRaises(ValueError):
            PatternTracker(families=())
        with self.assertRaises(ValueError):
            PatternTracker(families=("xx",))


if __name__ == "__main__":
    unittest.main()
