"""Input fades must not destroy trained feedback state or sample-clock accounting."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from v34_rx import InputPowerGuard,Phase3Rx,SymbolEq


class InputPowerGuardTests(unittest.TestCase):
    def test_fade_mode_holds_partial_power_and_keeps_silence_profile_unchanged(self):
        guard=InputPowerGuard('fade')
        for _ in range(64):guard.observe_training(100.)
        guard.arm()
        self.assertTrue(guard.hold(24.))
        for _ in range(20):self.assertTrue(guard.hold(49.))
        for _ in range(7):self.assertTrue(guard.hold(50.))
        self.assertFalse(guard.hold(50.))
        self.assertEqual(guard.report()['mode'],'fade')
        self.assertEqual(InputPowerGuard().ENTER_RATIO,1e-4)
        with self.assertRaises(ValueError):InputPowerGuard('unknown')
        eq=self.equalizer()
        with patch.dict('os.environ',{'V34_INPUT_POWER_GUARD':'fade'}):
            configured=SymbolEq(eq.R,0)
        self.assertEqual(configured._input_guard.mode,'fade')

    def trained(self):
        guard=InputPowerGuard()
        for _ in range(64):guard.observe_training(100.)
        guard.observe_training(1e20)
        guard.arm()
        return guard

    def test_training_reference_is_robust_and_short_training_bypasses(self):
        guard=self.trained();self.assertEqual(guard.reference,100.)
        guard.observe_training(1e-30);self.assertEqual(guard.reference,100.)
        short=InputPowerGuard()
        for _ in range(31):short.observe_training(100.)
        short.arm();self.assertFalse(short.hold(0.));self.assertFalse(short.report()['reference_available'])

    def test_return_requires_consecutive_power_above_hysteresis(self):
        guard=self.trained()
        self.assertTrue(guard.hold(0.))
        for _ in range(7):self.assertTrue(guard.hold(.05))
        self.assertTrue(guard.hold(.02))  # above entry, below exit: restart confirmation
        for _ in range(7):self.assertTrue(guard.hold(.05))
        self.assertFalse(guard.hold(.05))
        self.assertEqual(guard.report()['events'],1)
        self.assertEqual(guard.report()['power_returns'],1)
        self.assertFalse(guard.report()['feedback_held'])
        self.assertTrue(guard.hold(float('nan')))
        self.assertEqual(guard.report()['events'],2)

    def equalizer(self,native=False,mode='centroid'):
        R=Phase3Rx(3200,1800,'answer');R._span=12;R.lag=0;R.delay=0
        R.taps=np.zeros(24,complex);R.taps[12]=1;R._z=np.ones(2000,complex)
        with patch.dict('os.environ',{'V34_INPUT_POWER_GUARD':'1'}):eq=SymbolEq(R,0,timing_mode=mode)
        eq._native_feedback=native
        eq.window=lambda k:np.ones(24,complex)
        # Learn the reference through the actual pre-data step path.
        for _ in range(512):eq.step(decide=lambda y:y)
        eq.begin_data_tracking(False)
        self.assertTrue(eq._input_guard.report()['reference_available'])
        return eq

    def test_low_input_freezes_feedback_but_advances_both_oscillators(self):
        for native in (False,True):
            for mode in ('centroid','gradient'):
                with self.subTest(native=native,mode=mode):
                    eq=self.equalizer(native,mode)
                    eq.freq=.001;eq.vel=1e-5;eq.dt=.03;eq.phase=.4
                    eq.ktp=.02;eq.kti=.0001;eq.cf=.7
                    eq._gradient_rows=[('old-domain',)]
                    before=eq.h.copy();k=eq.k;n=eq._n
                    eq.window=lambda k:np.full(24,1e-6+0j)
                    for _ in range(240):eq.step(decide=lambda y:1+1j)
                    np.testing.assert_array_equal(eq.h,before)
                    self.assertEqual(eq.k,k+240);self.assertEqual(eq._n,n)
                    self.assertAlmostEqual(eq.dt,.03+240*1e-5)
                    self.assertAlmostEqual(eq.phase,.4+240*.001)
                    self.assertEqual(eq.freq,.001);self.assertEqual(eq.vel,1e-5)
                    self.assertEqual(eq.cf,.7);self.assertEqual(eq._gradient_rows,[])
                    self.assertEqual(eq._input_guard.held_symbols,240)
                    self.assertFalse(eq.timing_snapshot()['tap_adaptation'])
                    # On returning power, feedback cannot restart on the first window.
                    eq.window=lambda k:np.ones(24,complex)
                    for _ in range(7):eq.step(decide=lambda y:y)
                    self.assertTrue(eq._input_guard.holding)
                    eq.step(decide=lambda y:y)
                    self.assertFalse(eq._input_guard.holding)
                    self.assertEqual(eq._input_guard.power_returns,1)
                    self.assertEqual(eq.k,k+248)

    def test_default_remains_disabled_and_clean_power_has_exact_pcm_state_parity(self):
        eq=self.equalizer();plain=copy.deepcopy(eq);plain._input_guard=None
        for _ in range(100):
            self.assertEqual(eq.step(decide=lambda y:1+1j),plain.step(decide=lambda y:1+1j))
        np.testing.assert_array_equal(eq.h,plain.h)
        for key in ('dt','vel','phase','freq','slips'):self.assertEqual(getattr(eq,key),getattr(plain,key))
        self.assertEqual(eq._input_guard.events,0)
        with patch.dict('os.environ',{'V34_INPUT_POWER_GUARD':'0'}):
            default=SymbolEq(eq.R,0)
        self.assertIsNone(default._input_guard)


if __name__=='__main__':unittest.main()
