"""Prediction must use past TRN, correct block transfer and a real gate."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_trn_predictor import train,forecast,FIT,HOLD,install
import v34_rx as rx
import v34_receive as receive


class TRNPredictorTests(unittest.TestCase):
    def signal(self):
        rng=np.random.default_rng(332);t=np.arange(5600)/3200
        known=rng.choice([1+1j,3+1j,1+3j,3+3j],len(t))
        periodic=np.deg2rad(3)*np.sin(2*np.pi*60*t+.3)
        return t,known,periodic

    def test_weighted_training_recovers_unaveraged_future_phase(self):
        t,known,periodic=self.signal()
        model=train(known*np.exp(1j*(.2+.04*t+periodic)),known,t)
        self.assertTrue(model['predictive_lead'])
        future=np.arange(5700,5900)/3200
        expected=np.deg2rad(3)*np.sin(2*np.pi*60*future+.3)
        np.testing.assert_allclose(forecast(model,future),expected,atol=3e-5)

    def test_prediction_block_changes_gate_not_training_coefficients(self):
        t,known,periodic=self.signal();y=known*np.exp(1j*(.2+periodic))
        first=train(y,known,t)
        changed=y.copy();changed[HOLD[0]:HOLD[1]]=known[HOLD[0]:HOLD[1]]*np.exp(.2j)
        second=train(changed,known,t)
        np.testing.assert_array_equal(first['coefficients'],second['coefficients'])
        self.assertFalse(second['predictive_lead'])
        np.testing.assert_array_equal(forecast(second,t),np.zeros(len(t)))

    def test_constant_phase_does_not_trigger_numerical_lead(self):
        t,known,_=self.signal()
        model=train(known*np.exp(.2j),known,t)
        self.assertFalse(model['predictive_lead'])

    def test_window_correction_preserves_feedback_coordinates_and_no_model_parity(self):
        saved=(rx.Phase4Receiver.feed,rx.SymbolEq.step,rx.SymbolEq.window,
               receive.initialize,receive.Receiver.qualify)
        try:
            n=np.arange(4096);r=SimpleNamespace(_span=12,spl=2.5,lag=50,LEAD=8,delay=0,
                taps=np.ones(24,complex),_z=np.exp(.12j*n),baud=3200,fc=1900)
            eq=rx.SymbolEq(r,40);before=eq.window(40).copy();h=eq.h.copy()
            install(predict_data=True,correct_refinement=True)
            np.testing.assert_array_equal(eq.window(40),before)
            coeff=np.zeros(10);coeff[4]=.05
            model=dict(predictive_lead=True,origin=0.,coefficients=coeff.tolist(),k_trn=0)
            eq._trn_window_phase_model=model
            phi=forecast(model,np.array([40/3200]))[0]
            np.testing.assert_allclose(eq.window(40),before*np.exp(-1j*phi),atol=1e-12)
            # A target in the corrected output coordinates produces zero NLMS
            # and PLL error; no second output rotation or target rotation.
            y=eq.step(decide=lambda value:value)
            self.assertAlmostEqual(abs(y-(before@h)*np.exp(-1j*phi)),0.,places=11)
            np.testing.assert_array_equal(eq.h,h)
            self.assertEqual(eq.phase,0.)
        finally:
            rx.Phase4Receiver.feed,rx.SymbolEq.step,rx.SymbolEq.window,receive.initialize,receive.Receiver.qualify=saved


if __name__=='__main__':unittest.main()
