"""Lab predictive B1 must preserve state at the existing streaming handoff."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_noise_live_probe import LivePredictor
import v34_receive as receive
from v34_negotiation import DirectionParameters


class NoiseLiveTests(unittest.TestCase):
    def fixture(self):
        profile=DirectionParameters(rate=33600,baud=24000/7,carrier=1959.1836734693877,
            trellis=0,shaping=True,nonlinear=True,precoder=(129,188,-297,-160,222,-69))
        receiver=receive.Receiver(profile,'answer')
        return profile,receive.projection(receiver.linear,675.)*(.11+.002j)

    def test_predictive_b1_reconstructs_streaming_state_and_restores_hook(self):
        profile,observed=self.fixture()
        baseline=receive.Receiver(profile,'answer');baseline.qualify(observed)
        original=receive.Receiver.qualify
        duplex=SimpleNamespace(finish=lambda receiver:None)
        probe=LivePredictor(duplex)
        try:
            probe.model=dict(accepted=True)
            probe.coefficients=np.array([.6-.12j,.12+.06j,-.08+.03j])
            receiver=receive.Receiver(profile,'answer');receiver.qualify(observed)
            self.assertEqual(receiver.diagnostics['bit_errors'],0)
            self.assertTrue(receiver.diagnostics['lab_noise_predictive']['applied'])
            self.assertEqual(receiver.bits,baseline.bits)
            self.assertEqual(receiver.drx.conv.d,baseline.drx.conv.d)
            self.assertEqual(receiver.drx.z,baseline.drx.z)
            self.assertEqual(receiver.drx.frame_pos,baseline.drx.frame_pos)
            self.assertEqual(receiver.drx.m4,baseline.drx.m4)
            np.testing.assert_array_equal(receiver.drx.x_hist,baseline.drx.x_hist)
            np.testing.assert_array_equal(receiver.slicer_hist,receiver.drx.x_hist)
            np.testing.assert_array_equal(receiver.vit.x_hist,receiver.drx.x_hist)
            np.testing.assert_array_equal(receiver.vit.metric,baseline.vit.metric)
        finally:probe.close()
        self.assertIs(receive.Receiver.qualify,original)

    def test_rejected_training_leaves_production_decisions_unchanged(self):
        profile,observed=self.fixture()
        probe=LivePredictor(SimpleNamespace(finish=lambda receiver:None))
        try:
            probe.model=dict(accepted=False)
            receiver=receive.Receiver(profile,'answer');receiver.qualify(observed)
            self.assertEqual(receiver.diagnostics['bit_errors'],0)
            self.assertFalse(receiver.diagnostics['lab_noise_predictive']['applied'])
            self.assertEqual(probe.calls,[])
        finally:probe.close()

    def test_corrected_b1_uses_correct_precoder_history_at_data_boundary(self):
        profile,observed=self.fixture()
        coefficients=np.array([.6-.12j,.12+.06j,-.08+.03j])
        rng=np.random.default_rng(340611)
        innovations=.45*(rng.normal(size=len(observed))+1j*rng.normal(size=len(observed)))
        noise=np.zeros(len(observed),complex)
        for i in range(len(noise)):
            noise[i]=innovations[i]+sum(coefficients[k-1]*noise[i-k] for k in (1,2,3) if i>=k)
        observed=observed+noise*(.11+.002j)
        baseline=receive.Receiver(profile,'answer');baseline.qualify(observed)
        self.assertGreater(baseline.diagnostics['bit_errors'],0)
        probe=LivePredictor(SimpleNamespace(finish=lambda receiver:None))
        try:
            # Known injected predictor is a detector/handoff positive control,
            # not a claim to have tested live coefficient estimation here.
            probe.model=dict(accepted=True);probe.coefficients=coefficients
            receiver=receive.Receiver(profile,'answer');receiver.qualify(observed)
            self.assertEqual(receiver.diagnostics['bit_errors'],0)
            np.testing.assert_array_equal(receiver.drx.x_hist,receiver.linear[-3:][::-1])
            np.testing.assert_array_equal(receiver.vit.x_hist,receiver.drx.x_hist)
            self.assertEqual(receiver.diagnostics['lab_noise_predictive']['baseline_bit_errors'],
                             baseline.diagnostics['bit_errors'])
        finally:probe.close()


if __name__=='__main__':unittest.main()
