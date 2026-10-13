"""Startup-only gradient mode is isolated and handed off without an NCO reset."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from v34_receive_dsp import ReceiveDsp, configured


class TrnGradientTests(unittest.TestCase):
    def test_gradient_is_opt_in_and_restores_mode_once(self):
        eq=SimpleNamespace(k=32,timing_mode='centroid',_gradient_rows=[1],dt=3.,vel=.0007)
        dsp=ReceiveDsp(SimpleNamespace(),trn_gradient=True)
        dsp.before_symbol(SimpleNamespace(eq=eq,k_trn=16))
        self.assertEqual(eq.timing_mode,'gradient')
        self.assertEqual(eq._gradient_rows,[])
        eq._gradient_rows.append(2)
        dsp.before_symbol(SimpleNamespace(eq=eq,k_trn=16))
        self.assertEqual(eq._gradient_rows,[2])
        dsp.before_b1(eq,16)
        self.assertEqual(eq.timing_mode,'centroid')
        self.assertEqual((eq.dt,eq.vel),(3.,.0007))
        self.assertEqual(eq._gradient_rows,[])
        eq.timing_mode='data-selected'
        dsp.before_b1(eq,16)
        self.assertEqual(eq.timing_mode,'data-selected')
        self.assertTrue(dsp.report()['startup_timing_restored'])

    def test_auto_waits_for_crc_grace_and_repeated_errors(self):
        dsp=ReceiveDsp(SimpleNamespace(),trn_gradient='auto')
        receiver=SimpleNamespace(trn_end=None,mismatch=[0]*600+[1])
        self.assertFalse(dsp._gradient_ready(receiver))
        receiver.mismatch.extend([0]*63)
        self.assertFalse(dsp._gradient_ready(receiver))
        receiver.mismatch.append(1)
        self.assertTrue(dsp._gradient_ready(receiver))
        receiver.trn_end=600
        self.assertFalse(dsp._gradient_ready(receiver))

    def test_auto_preserves_clean_and_acquisition_transients(self):
        for errors in ([0]*1200,[1]*600):
            dsp=ReceiveDsp(SimpleNamespace(),trn_gradient='auto')
            self.assertFalse(dsp._gradient_ready(SimpleNamespace(trn_end=None,mismatch=errors)))
            self.assertIsNone(dsp._gradient_first_error)

    def test_default_and_per_call_isolation(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertIsNone(configured(SimpleNamespace()))
        with patch.dict(os.environ,{'V34_TRN_GRADIENT':'1'},clear=True):
            dsp=configured(SimpleNamespace())
        other=ReceiveDsp(SimpleNamespace())
        eq=SimpleNamespace(k=32,timing_mode='centroid',_gradient_rows=[])
        dsp.before_symbol(SimpleNamespace(eq=eq,k_trn=16))
        self.assertIsNone(other._startup_timing_mode)
        self.assertTrue(dsp.trn_gradient)
        self.assertIsNone(dsp.clock)


if __name__=='__main__':unittest.main()
