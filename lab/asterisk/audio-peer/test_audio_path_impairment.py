import unittest

from audio_path_impairment import AudioPathImpairment


class AudioPathImpairmentTests(unittest.TestCase):
    def test_seeded_noise_is_repeatable_and_clipping_is_counted(self):
        kwargs = dict(gain_db=6, dc_offset=100, noise_rms=20,
                      clip_level=1000, seed=42)
        left = AudioPathImpairment(**kwargs)
        right = AudioPathImpairment(**kwargs)
        samples = [0, 100, 500, 1000, 2000, -2000]
        self.assertEqual(left.process(samples), right.process(samples))
        self.assertEqual(left.report(), right.report())
        self.assertEqual(left.report()["samples"], len(samples))
        self.assertGreater(left.report()["clipped_samples"], 0)

    def test_rejects_non_finite_or_out_of_range_settings(self):
        for kwargs in (dict(gain_db=19), dict(dc_offset=4097),
                       dict(noise_rms=-1), dict(noise_rms=8193),
                       dict(clip_level=255), dict(gain_db=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AudioPathImpairment(**kwargs)


if __name__ == "__main__":
    unittest.main()
