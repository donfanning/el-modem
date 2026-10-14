"""Actual V.32 evidence adapter: sample time, LAPM delivery and discontinuities."""
import unittest
from types import SimpleNamespace as NS
from v32_stream import V32Modem


class V32BudgetTests(unittest.TestCase):
    def modem(self):
        m=V32Modem(use_v8=False,echo=False)
        m.connected=True;m.state='data';m.bit_rate=9600
        m._rx_power=m._ref_power;m.v42_mode='lapm'
        m._v42=NS(state='lapm',stats=dict(hdlc_bad=0))
        m._retrain_budget.take();m._retrain_budget.take()
        m._retrain_attempts=2
        return m

    def tick(self,m,delivered=128):
        m._observe_retrain_budget(8000,delivered)

    def test_sustained_actual_delivery_renews_used_budget_not_attempt_serial(self):
        m=self.modem()
        for _ in range(9):self.tick(m)
        self.assertEqual(m._retrain_budget.used,2)
        self.tick(m)
        self.assertEqual(m._retrain_budget.used,0)
        self.assertEqual(m.retrain_stats['attempts'],2)
        self.assertTrue(m.request_retrain())
        self.assertEqual(m.retrain_stats['attempts'],3)
        self.assertEqual(m.retrain_stats['budget']['total'],3)

    def test_raw_mnp_bad_frames_training_candidate_carrier_and_concealment_reject_proof(self):
        for defect in ('raw','mnp','bad','training','candidate','carrier','quiet','concealed'):
            m=self.modem()
            for _ in range(9):self.tick(m)
            if defect=='raw':m.v42_mode='off'
            elif defect=='mnp':m.v42_mode='mnp';m._v42.state='mnp'
            elif defect=='bad':m._v42.stats['hdlc_bad']=1
            elif defect=='training':m._retraining=True
            elif defect=='candidate':m._request_tone.run=1
            elif defect=='carrier':m._rx_power=0
            elif defect=='quiet':m._quiet_blocks=1
            elif defect=='concealed':m.note_concealed_rx(8000)
            self.tick(m)
            self.assertEqual(m._retrain_budget.used,2,defect)
            self.assertEqual(m._retrain_budget.samples,0,defect)

    def test_fresh_line_transport_reset_and_delivery_gap_restart_proof(self):
        for defect in ('line','transport','gap'):
            m=self.modem()
            for _ in range(7 if defect=='gap' else 9):self.tick(m)
            if defect=='line':m.eq=object()
            elif defect=='transport':m.note_transport_discontinuity_rx()
            else:
                self.tick(m,0);self.tick(m,0)
            self.tick(m)
            self.assertEqual(m._retrain_budget.used,2,defect)

    def test_transmit_and_idle_time_do_not_replenish_budget(self):
        m=self.modem()
        m.tx_samples(16000)
        self.assertEqual(m._retrain_budget.samples,0)
        for _ in range(12):self.tick(m,0)
        self.assertEqual(m._retrain_budget.used,2)
