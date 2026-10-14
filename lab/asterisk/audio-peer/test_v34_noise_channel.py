"""Independent injected channel must be causal and preserve chunk boundaries."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_noise_channel import ColoredNoiseChannel


class ColoredNoiseTests(unittest.TestCase):
    def test_chunking_preserves_exact_pcm_and_inactive_path(self):
        whole=ColoredNoiseChannel(64.,1246);chunked=ColoredNoiseChannel(64.,1246)
        signal=np.arange(1024,dtype=np.int16)
        expected=whole.process(signal,True)
        actual=chunked.process(signal[:160],True)+chunked.process(signal[160:512],True)+chunked.process(signal[512:],True)
        self.assertEqual(actual,expected)
        self.assertEqual(chunked.process(signal,False),signal.tolist())
        self.assertEqual(chunked.samples,1024)

    def test_fixed_stationary_rms_and_no_estimator_truth(self):
        channel=ColoredNoiseChannel(64.,1246)
        output=np.asarray(channel.process(np.zeros(80000),True))
        self.assertAlmostEqual(float(np.sqrt(np.mean(output[1000:]**2)))/64.,1.,delta=.05)
        self.assertEqual(channel.clipped,0)
        self.assertFalse(channel.report()['parameters_shared_with_estimator'])


if __name__=='__main__':unittest.main()
