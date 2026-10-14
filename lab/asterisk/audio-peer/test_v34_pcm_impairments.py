"""Independent analytic signal checks for continuous PCM impairments."""
import unittest
import numpy as np
from v34_stream_impair import StreamImpairment


def process(signal,chunk,**kwargs):
    stream=StreamImpairment(**kwargs);stream.latency=512-(63 if kwargs.get('phase_jitter_hz') else 0)
    out=np.concatenate([stream.process(signal[i:i+chunk]) for i in range(0,len(signal),chunk)])
    return out,stream


class PCMImpairmentTests(unittest.TestCase):
    def tone(self):
        return np.rint(10000*np.cos(2*np.pi*1500*np.arange(16000)/8000)).astype(np.int16)

    def test_zero_jitter_matched_delay_is_sample_exact(self):
        signal=self.tone()
        a,_=process(signal,160)
        b,stream=process(signal,160,phase_jitter_hz=60)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(b[512:],signal[:-512])
        self.assertEqual(stream.provenance()['latency_samples'],512)

    def test_phase_jitter_agrees_with_independent_analytic_tone(self):
        signal=self.tone();out,_=process(signal,160,phase_jitter_hz=60,phase_jitter_deg=6)
        # Closed-form phase-modulated carrier, independent of the streaming
        # Hilbert filter. Exclude startup and compare in original PCM units.
        n=np.arange(len(out))-512
        expected=10000*np.cos(2*np.pi*1500*n/8000+np.deg2rad(6)*np.sin(2*np.pi*60*n/8000))
        self.assertLess(np.sqrt(np.mean((out[1000:]-expected[1000:])**2)),3.)

    def test_chunk_sizes_do_not_change_phase_noise_or_gain(self):
        signal=self.tone()
        for params in (dict(phase_jitter_hz=60,phase_jitter_deg=6),
                       dict(gain_hit_db=1),dict(noise_rms=16,seed=777)):
            expected,_=process(signal,80,**params)
            for chunk in (160,320):
                actual,_=process(signal,chunk,**params)
                np.testing.assert_array_equal(actual,expected)

    def test_gain_ratio_activation_and_clipping_evidence(self):
        signal=self.tone();stream=StreamImpairment(active=False,gain_hit_db=1)
        stream.latency=512
        a=stream.process(signal[:1600]);stream.activate();b=stream.process(signal[1600:])
        self.assertEqual(stream.provenance()['gain_output_activation_sample'],1600)
        np.testing.assert_array_equal(a[512:],signal[:1088])
        np.testing.assert_array_equal(b,np.rint(signal[1088:-512]*10**(1/20)).astype(np.int16))
        self.assertEqual(stream.clipped_samples,0)
        clipped=StreamImpairment(gain_hit_db=3);clipped.latency=64
        clipped.process(np.full(1000,32767));self.assertGreater(clipped.clipped_samples,0)

    def test_noise_rms_and_bounds(self):
        out,_=process(np.zeros(16000),160,noise_rms=16,seed=777)
        self.assertAlmostEqual(np.std(out),16,delta=.3)
        for params in (dict(phase_jitter_deg=3),dict(phase_jitter_hz=301),dict(gain_hit_db=4)):
            with self.assertRaises(ValueError):StreamImpairment(**params)

    def test_delayed_phase_activation_matches_analytic_onset(self):
        signal=self.tone();stream=StreamImpairment(active=False,phase_jitter_hz=60,phase_jitter_deg=3)
        stream.latency=449
        first=stream.process(signal[:1600]);stream.activate()
        out=np.r_[first,stream.process(signal[1600:])]
        n=np.arange(len(out));phase=np.zeros(len(out))
        active=n>=2049
        phase[active]=np.deg2rad(3)*np.sin(2*np.pi*60*(n[active]-2049)/8000)
        expected=10000*np.cos(2*np.pi*1500*(n-512)/8000+phase)
        self.assertLess(np.sqrt(np.mean((out[1000:]-expected[1000:])**2)),3.)
        self.assertEqual(stream.provenance()['phase_output_activation_sample'],2049)


if __name__=='__main__':unittest.main()
