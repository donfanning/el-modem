"""Independent domain, startup, modulo expansion and payload gates."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_whitening_probe import independent_gates, whitening, expanded_points, reconstruct_mapper, ChannelOutputDetector
import v34_b1_reference as independent
import v34_data as vd
from v34_datarx import DataRx


class B1WhiteningProbeTests(unittest.TestCase):
    def test_independent_b1_and_random_bits_with_both_modulo_steps(self):
        # Stronger than the corpus's small MP coefficients: exercise nonzero c
        # and channel y points outside the original mapper constellation.
        for rate,baud,nonlinear in ((14400,3200,False),(33600,3428.5714285714,True)):
            profile=dict(rate=rate,baud=baud,carrier=1800.,trellis=0,shaping=True,
                         nonlinear=nonlinear,precoder=[8192,4096,-4096,2048,2048,-1024])
            rows=independent_gates(profile,state=(1,0,1,1))
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['bit_errors']==0 for r in rows))
            self.assertGreater(max(r['fir_rounding_max_coordinate'] for r in rows),0.)

    def test_expanded_alphabet_contains_actual_transmitted_channel_points(self):
        rate,baud=33600,3428.5714285714
        mp=[16384,8192,-8192,4096,4096,-2048]
        h=vd.precoder_from_mp(mp); data=DataRx(rate,baud,True,h)
        ref=independent.build(rate,baud,'answer',True,mp)
        points,bound=expanded_points(data._pts,h,data.w)
        self.assertGreater(bound,0.)
        self.assertTrue(set(ref['y']).issubset(set(points)))
        self.assertTrue(any(y not in set(data._pts) for y in ref['y']))
        u,x=reconstruct_mapper(ref['y'],h,data.w)
        np.testing.assert_array_equal(u,ref['u'])
        np.testing.assert_allclose(x,ref['x'],atol=1e-12)

    def test_whitening_is_linear_with_positive_coefficients_and_zero_history(self):
        x=np.array([1+2j,3-1j,-2+.5j]); h=[.2+.1j,-.15j,.3]
        expected=np.array([x[0],x[1]+h[0]*x[0],x[2]+h[0]*x[1]+h[1]*x[0]])
        np.testing.assert_allclose(whitening(x,h),expected)
        np.testing.assert_allclose(whitening(2j*x,h),2j*expected)

    def test_channel_parity_is_label_low_bit_and_has_no_early_feedback(self):
        data=DataRx(14400,3200,True,[.5+.25j,-.25j,.1])
        det=ChannelOutputDetector(data._pts,data.h,data.w,data.J,data.P)
        np.testing.assert_array_equal(det.par,det.lab&1)
        det.feed(3.1+1.2j,5.3-3.1j)
        self.assertEqual(det.x_hist,[0j]*3)


if __name__=='__main__': unittest.main()
