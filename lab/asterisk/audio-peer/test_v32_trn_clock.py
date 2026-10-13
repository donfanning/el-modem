import unittest
from types import SimpleNamespace
import numpy as np
from v32_rx import expected_trn, GPA
from v32_trn_clock import TrainingClock, training_delay


class TrainingClockTest(unittest.TestCase):
    def waveform(self, ppm):
        # Independent continuous raised-cosine synthesis, evaluated with the
        # imposed clock. No production resampler/correlation helper is used.
        reference = np.asarray(expected_trn(7000,GPA))
        time = np.arange(14040)/2*(1+ppm*1e-6)
        index = np.floor(time).astype(int)[:,None]+np.arange(-7,9)
        delta = time[:,None]-index
        beta = .3
        denominator = 1-(2*beta*delta)**2
        with np.errstate(divide='ignore',invalid='ignore'):
            pulse = np.sinc(delta)*np.cos(np.pi*beta*delta)/denominator
        pulse = np.where(abs(denominator)<1e-10,np.pi/4*np.sinc(1/(2*beta)),pulse)
        values = reference[np.clip(index,0,len(reference)-1)]
        values = np.where((index>=0)&(index<len(reference)),values,0)
        u = np.sum(values*pulse,axis=1)*1.3*np.exp(.7j)
        return SimpleNamespace(_s0=0,_ph=0,ubase=0,u=u.tolist(),_rx_tap=GPA,
                               _rx_mode='train1',bb=SimpleNamespace(step=5/3))

    def test_clock_sign_scale_and_null_from_independent_waveform(self):
        for ppm in (-129.,0.,129.):
            with self.subTest(ppm=ppm):
                engine = self.waveform(ppm)
                clock = TrainingClock()
                for done in range(1,5001):
                    clock.feed(engine,done)
                fit, = clock.fits
                self.assertAlmostEqual(fit['residual_clock_ppm'],ppm,delta=2.)
                self.assertEqual(fit['applied'],bool(ppm))
                self.assertAlmostEqual(engine.bb.step/(5/3),1/(1+ppm*1e-6),delta=2e-6)
                self.assertEqual(clock.report()['retained_rows'],0)
                self.assertEqual(clock.report()['retained_reference_symbols'],0)
                self.assertEqual(clock.report()['peak_rows'],6)

    def test_missing_and_silent_waveforms_do_not_fit(self):
        engine = self.waveform(129)
        reference = np.asarray(expected_trn(7000,GPA))
        engine.u = []
        self.assertIsNone(training_delay(engine,2048,reference))
        engine.u = [0j]*14040
        self.assertIsNone(training_delay(engine,2048,reference))
        clock = TrainingClock()
        for done in range(1,5001):clock.feed(engine,done)
        self.assertFalse(clock.fits[0]['applied'])
        self.assertEqual(engine.bb.step,5/3)

    def test_short_training_release_and_new_stage(self):
        engine = self.waveform(129)
        clock = TrainingClock()
        for done in range(1,1800):clock.feed(engine,done)
        clock.release()
        self.assertEqual(clock.report()['retained_reference_symbols'],0)
        self.assertFalse(clock.fits)
        engine._rx_mode='train2'
        for done in range(1,5001):clock.feed(engine,done)
        self.assertTrue(clock.fits[0]['applied'])

    def test_fit_is_available_before_short_answerer_training_ends(self):
        for ppm in (-100.,100.):
            with self.subTest(ppm=ppm):
                engine=self.waveform(ppm);clock=TrainingClock()
                for done in range(1,2977):clock.feed(engine,done)
                fit,=clock.fits
                self.assertEqual(fit['symbol'],2304)
                self.assertTrue(fit['applied'])
                self.assertAlmostEqual(fit['residual_clock_ppm'],ppm,delta=2.)


if __name__ == '__main__':unittest.main()
