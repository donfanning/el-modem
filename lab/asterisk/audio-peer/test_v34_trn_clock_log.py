"""The TRN clock gate decision is logged exactly once, at the B1 handoff."""
from types import SimpleNamespace
import unittest
import numpy as np
from v34_receive_dsp import ReceiveDsp


def dsp_with(model, taps=None):
    events = []
    engine = SimpleNamespace(_ev=events.append)
    dsp = ReceiveDsp(engine, trn_clock=True)
    dsp.clock.model, dsp.clock.taps = model, taps
    return dsp, events


class ClockLogTests(unittest.TestCase):
    def eq(self):
        return SimpleNamespace(k=8591+16, dt=2.756, vel=0., h=np.ones(24), _grid='old')

    def test_accepted_is_logged_once_with_improvement_and_offset(self):
        model = dict(accepted=True, offset_samples=1.9, samples_per_symbol_correction=3e-4,
                     anchor_trn_symbol=5500., validation_improvement=.7878)
        dsp, events = dsp_with(model, np.zeros(24, complex))
        dsp.before_b1(self.eq(), 16)
        dsp.before_b1(self.eq(), 16)
        self.assertEqual(len(events), 1)
        self.assertIn('accepted', events[0])
        self.assertIn('78.8%', events[0])
        self.assertIn('applied at symbol', events[0])

    def test_rejected_is_logged_as_bypassed_and_leaves_state(self):
        dsp, events = dsp_with(dict(accepted=False, validation_improvement=-.035))
        eq = self.eq()
        dsp.before_b1(eq, 16)
        self.assertEqual(len(events), 1)
        self.assertIn('bypassed', events[0])
        self.assertIn('-3.5%', events[0])
        self.assertEqual(eq.dt, 2.756)
        self.assertEqual(eq._grid, 'old')

    def test_error_reason_is_logged(self):
        dsp, events = dsp_with(dict(accepted=False, reason='insufficient guarded TRN'))
        dsp.before_b1(self.eq(), 16)
        self.assertIn('insufficient guarded TRN', events[0])

    def test_engine_without_event_hook_is_safe(self):
        dsp = ReceiveDsp(SimpleNamespace(), trn_clock=True)
        dsp.clock.model = dict(accepted=False)
        dsp.before_b1(self.eq(), 16)


if __name__ == '__main__':
    unittest.main()
