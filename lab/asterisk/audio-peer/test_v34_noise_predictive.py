"""Independent forward-channel checks for the offline predictive detector."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_noise_predictive import detect
from b1_noise_kernel import available
from b1_path_cost_audit import waveform
import v34_b1_reference as independent
import v34_receive as receive
from v34_negotiation import DirectionParameters


class NoisePredictiveTests(unittest.TestCase):
    def fixture(self):
        profile=DirectionParameters(rate=33600,baud=24000/7,carrier=1959.1836734693877,
            trellis=0,shaping=True,nonlinear=True,precoder=(129,188,-297,-160,222,-69))
        reference=independent.build(profile.rate,profile.baud,'answer',True,profile.precoder)
        receiver=receive.Receiver(profile,'answer');receiver.energy=675.
        wave=receive.projection(np.asarray(reference['x']),receiver.energy)
        return receiver,reference,wave

    def test_noiseless_precoded_nonlinear_sequence_and_cost(self):
        receiver,truth,wave=self.fixture()
        points,cost=detect(wave,receiver,[.6-.12j,.12+.06j,-.08+.03j])
        np.testing.assert_array_equal(points,truth['u'])
        self.assertLess(cost,1e-15)

    def test_returned_cost_uses_selected_paths_own_histories(self):
        receiver,truth,wave=self.fixture()
        rng=np.random.default_rng(340615)
        observed=wave+.7*(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave)))
        coef=np.array([.6-.12j,.12+.06j,-.08+.03j])
        points,cost=detect(observed,receiver,coef)
        linear,_=waveform(points,receiver.h,receiver.tx.w)
        residual=observed-receive.projection(linear,receiver.energy)
        expected=0.
        for i,e in enumerate(residual):
            predicted=sum(coef[k-1]*residual[i-k] for k in (1,2,3) if i>=k)
            expected+=abs(e-predicted)**2
        self.assertAlmostEqual(cost,expected,places=9)

    def test_wrong_predictor_order_is_rejected(self):
        receiver,_,wave=self.fixture()
        with self.assertRaisesRegex(ValueError,'three-coefficient'):
            detect(wave,receiver,[.5])

    def test_native_layouts_match_reference_for_all_modulo_offsets(self):
        from b1_noise_kernel import _tables
        receiver,_,_=self.fixture()
        detector=receiver.detector()
        points,classes,counts,transitions=_tables(len(detector.pts))
        np.testing.assert_array_equal(points,detector.pts)
        np.testing.assert_array_equal(transitions,detector.next)
        for re in range(8):
            for im in range(8):
                ref=detector._layout(complex(re,im))[0]
                for label in range(8):
                    for parity in range(2):
                        c=label*2+parity
                        np.testing.assert_array_equal(classes[re,im,c,:counts[re,im,c]],ref[label][parity])

    @unittest.skipUnless(available(), 'optional Numba backend not installed')
    def test_native_and_reference_noisy_path_cost_and_ties(self):
        receiver,_,wave=self.fixture()
        coef=np.array([.6-.12j,.12+.06j,-.08+.03j])
        rng=np.random.default_rng(340616)
        for samples in (wave, wave+.9*(rng.normal(size=len(wave))+1j*rng.normal(size=len(wave))),
                        np.zeros(8,complex)):
            reference,cost=detect(samples,receiver,coef,backend='python')
            native,ncost=detect(samples,receiver,coef,backend='numba')
            np.testing.assert_array_equal(native,reference)
            np.testing.assert_allclose(ncost,cost,rtol=1e-12,atol=1e-12)

    @unittest.skipUnless(available(), 'optional Numba backend not installed')
    def test_native_signed_half_quantization(self):
        from b1_noise_kernel import _quantize
        for step in (2**-7,2.,4.):
            for v in (-3.5,-2.5,-.5,0.,.5,2.5,3.5):
                z=complex(v*step,-v*step)
                self.assertEqual(_quantize(z,step),independent.quantize(z,step))

    @unittest.skipUnless(available(), 'optional Numba backend not installed')
    def test_native_input_validation(self):
        receiver,_,wave=self.fixture()
        with self.assertRaisesRegex(ValueError,'pairs'):
            detect(wave[:-1],receiver,[0.,0.,0.],backend='numba')
        with self.assertRaisesRegex(ValueError,'finite'):
            detect(np.array([np.nan,0j]),receiver,[0.,0.,0.],backend='numba')


if __name__=='__main__':unittest.main()
