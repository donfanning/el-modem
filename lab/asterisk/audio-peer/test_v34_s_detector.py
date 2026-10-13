"""detect_s_flip: sensitivity on a weak (PSTN-like) S, and no false alarms on noise or training.

A live dialup.world call showed the answerer's phase-4 S scoring about 0.71 against the detector's
40-symbol reference, only 0.012 above the old 0.70 threshold, so the live caller never answered. The
threshold is now S_FLIP_THRESHOLD (0.65). These tests pin that a weak S scored between the old and new
thresholds is detected by the default and not by the old threshold, while noise and TRN-like symbols
at the same level are not detected. See lab/V34_PHASE3_STALL_DIAGNOSIS.md.
"""
import math
import unittest

import numpy as np

import v34_rx as rx
import v34_tx as tx

FS = 8000
BAUD = 3428.5714285714284
FC = 1959.1836734693877
RMS = 1500.0


def peak_score(x):
    """Peak normalised correlation with the detector's own reference (ref_len 40, 8-symbol leads)."""
    spl = FS/BAUD
    n_ref = int((40+16)*spl)
    ref = tx.qam_modulate(tx.s_signal(40), BAUD, 0, 1, delay=0.0, baseband=True)[:n_ref]
    z = rx.to_baseband(x, FC, max(1700.0, .62*BAUD))
    nfft = 1 << int(math.ceil(math.log2(len(z)+len(ref))))
    c = np.fft.ifft(np.fft.fft(z, nfft)*np.conj(np.fft.fft(ref, nfft)))[:len(z)-len(ref)+1]
    e = np.convolve(np.abs(z)**2, np.ones(len(ref)), mode='valid')
    return float(np.max(np.abs(c)/np.sqrt(e*np.sum(np.abs(ref)**2)+1e-9)))


def s_burst(noise_std, seed=1, lead_s=0.08, tail_s=0.12):
    """Passband S (128T) then S-bar (16T) at RMS 1500, inside white noise of the given std."""
    rng = np.random.default_rng(seed)
    symbols = np.concatenate([tx.s_signal(128), tx.s_signal(16, True)])
    burst = tx.qam_modulate(symbols, BAUD, FC, RMS)
    x = np.concatenate([np.zeros(int(lead_s*FS)), burst, np.zeros(int(tail_s*FS))])
    return x+rng.normal(0, noise_std, len(x))


def noise_for_score(target, seed=1):
    """Bisect the noise std so the S peak score lands on `target`."""
    lo, hi = 1.0, 6000.0
    for _ in range(40):
        mid = math.sqrt(lo*hi)
        if peak_score(s_burst(mid, seed)) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo*hi)


class SDetectorTests(unittest.TestCase):
    def test_cached_reference_preserves_operator_and_is_immutable(self):
        for baud in (2400., 3200., BAUD):
            for length in (40, 64):
                ref = rx._s_flip_reference(baud, length)
                count = int((length + 16) * FS / baud)
                expected = tx.qam_modulate(tx.s_signal(length), baud, 0, 1,
                                           delay=0., baseband=True)[:count]
                np.testing.assert_array_equal(ref, expected)
                self.assertIs(rx._s_flip_reference(baud, length), ref)
                with self.assertRaises(ValueError):
                    ref[0] = 0

    def test_threshold_constant_is_between_old_and_false_alarm_edge(self):
        self.assertEqual(rx.S_FLIP_THRESHOLD, 0.65)
        self.assertEqual(rx.detect_s_flip.__defaults__[-1], rx.S_FLIP_THRESHOLD)

    def test_clean_s_is_detected_by_default(self):
        result = rx.detect_s_flip(s_burst(5.0), FC, BAUD)
        self.assertIsNotNone(result)
        self.assertIn(result[0], ('S', 'flip'))

    def test_weak_s_between_old_and_new_threshold_is_detected_only_by_the_default(self):
        found = 0
        for seed in range(1, 6):
            std = noise_for_score(0.675, seed)
            x = s_burst(std, seed)
            score = peak_score(x)
            self.assertTrue(.65 < score < .70, score)
            if rx.detect_s_flip(x, FC, BAUD, threshold=0.70) is None and rx.detect_s_flip(x, FC, BAUD) is not None:
                found += 1
        self.assertGreaterEqual(found, 4, 'a 0.675 S must be missed by 0.70 and found by the default')

    def test_noise_alone_is_not_detected(self):
        for seed in range(1, 21):
            noise = np.random.default_rng(seed).normal(0, 1500, int(0.4*FS))
            self.assertIsNone(rx.detect_s_flip(noise, FC, BAUD), seed)

    def test_training_like_symbols_are_not_detected(self):
        for seed in range(1, 11):
            rng = np.random.default_rng(100+seed)
            points = np.array([1+1j, -1+1j, -1-1j, 1-1j])
            symbols = points[rng.integers(0, 4, 400)]
            x = tx.qam_modulate(symbols, BAUD, FC, RMS)[:int(0.4*FS)]
            self.assertIsNone(rx.detect_s_flip(x+rng.normal(0, 30, len(x)), FC, BAUD), seed)


if __name__ == '__main__':
    unittest.main()
