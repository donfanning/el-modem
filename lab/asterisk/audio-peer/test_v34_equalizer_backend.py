"""Experimental selection is explicit; arithmetic parity preserves loop ordering."""
import copy
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
from v34_equalizer_backend import selected,warmup
from v34_rx import Phase3Rx,SymbolEq,decide4

class BackendSelectionTests(unittest.TestCase):
    def test_default_does_not_import_extension(self):
        with patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{'el_modem_dsp':None}):
            self.assertIsNone(selected());self.assertEqual(warmup(),'legacy')

    def test_unknown_or_missing_requested_backend_fails(self):
        with patch.dict(os.environ,{'V34_EQUALIZER_BACKEND':'typo'}):
            with self.assertRaises(ValueError):selected()
        with patch.dict(os.environ,{'V34_EQUALIZER_BACKEND':'rust'}),patch.dict(sys.modules,{'el_modem_dsp':None}):
            with self.assertRaises(ImportError):warmup()

    @unittest.skipUnless(importlib.util.find_spec('el_modem_dsp'),'experimental wheel not installed')
    def test_selected_rust_tracks_acquisition_buffer_change_and_callbacks(self):
        rng=np.random.default_rng(741)
        R=Phase3Rx(3200,1800,'answer');R._z=rng.normal(size=3000)+1j*rng.normal(size=3000)
        R._span=24;R.taps=np.zeros(48,complex);R.taps[24]=1;R.lag=100;R.delay=0
        for adapt,nlms in ((True,True),(False,True),(True,False)):
            with patch.dict(os.environ,{'V34_EQUALIZER_BACKEND':'legacy'}):slow=SymbolEq(copy.deepcopy(R),20)
            with patch.dict(os.environ,{'V34_EQUALIZER_BACKEND':'rust'}):
                fast=SymbolEq(copy.deepcopy(R),20);self.assertEqual(warmup(),'rust')
            self.assertEqual(fast.arithmetic_backend,'rust')
            for eq in (slow,fast):
                eq.adapt=adapt;eq.adapt_nlms=nlms;eq.ktp=.0001;eq.kti=.000001;eq.cdead=.01;eq.calpha=.1
                eq._native_feedback=eq._native_interp=eq._traj=False
            calls=[0,0];observed=[[],[]]
            for index,eq in enumerate((slow,fast)):
                def decision(y):calls[index]+=1;return decide4(y)
                for ordinal in range(400):
                    if ordinal==100:
                        eq.R._z=eq.R._z[32:].copy();eq.R._bb_origin=32
                    observed[index].append(eq.step(decide=decision))
            self.assertEqual(calls,[400,400]);self.assertEqual(slow.slips,fast.slips)
            np.testing.assert_allclose(observed[0],observed[1],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(slow.h,fast.h,rtol=1e-10,atol=1e-10)
            for key in ('dt','vel','phase','freq','cf','c0'):
                self.assertEqual(getattr(slow,key) is None,getattr(fast,key) is None)
                if getattr(slow,key) is not None:self.assertAlmostEqual(getattr(slow,key),getattr(fast,key),places=10)

if __name__=='__main__':unittest.main()
