"""Successive unstable retrains remain bounded; proven delivery renews them."""
import unittest
from types import SimpleNamespace as NS
from v34_retrain_budget import RetrainBudget, observe


class BudgetTests(unittest.TestCase):
    def tick(self, b, core, **kw):
        args=dict(core=core,samples=8000,fs=8000,healthy=True,delivered=128,bad_frames=0)
        args.update(kw)
        return b.observe(**args)

    def test_third_attempt_without_recovery_rejected(self):
        b=RetrainBudget()
        self.assertTrue(b.take());self.assertTrue(b.take());self.assertFalse(b.take())
        self.assertEqual(b.report()['total'],2)

    def test_only_sustained_validated_delivery_renews(self):
        b=RetrainBudget();core=object();b.take();b.take()
        for _ in range(9):self.assertFalse(self.tick(b,core))
        self.assertFalse(b.take())
        self.assertTrue(self.tick(b,core))
        self.assertTrue(b.take())
        self.assertEqual((b.used,b.total,b.rearms),(1,3,1))

    def test_idle_or_brief_payload_alone_does_not_renew(self):
        for delivered,samples in ((0,8000),(1024,160)):
            b=RetrainBudget();b.take();b.take();core=object()
            for _ in range(11):self.assertFalse(self.tick(b,core,delivered=delivered,samples=samples))
            self.assertFalse(b.take())

    def test_early_payload_then_silent_loss_cannot_replenish_budget(self):
        b=RetrainBudget();b.take();b.take();core=object()
        self.assertFalse(self.tick(b,core,delivered=1024))
        for _ in range(9):self.assertFalse(self.tick(b,core,delivered=0))
        self.assertFalse(b.take())
        for _ in range(9):self.assertFalse(self.tick(b,core))
        self.assertTrue(self.tick(b,core))
        self.assertTrue(b.take())

    def test_bad_frame_hold_and_new_receive_epoch_break_stability(self):
        for defect in ('bad','held','epoch'):
            b=RetrainBudget();b.take();b.take();core=object()
            for _ in range(9):self.tick(b,core)
            if defect=='bad':self.assertFalse(self.tick(b,core,bad_frames=1))
            if defect=='held':self.assertFalse(self.tick(b,core,healthy=False))
            if defect=='epoch':self.assertFalse(self.tick(b,object()))
            self.assertFalse(b.take())

    def modem(self):
        core=NS(stage='data',_rx_state={},rx_qualification=dict(standard_qualification=True,bit_errors=0))
        m=NS(connected=True,_retraining=False,fs=8000,v42_mode='lapm',
             _v42=NS(state='lapm',stats=dict(hdlc_bad=0)),_retrain_budget=RetrainBudget(),_ev=lambda text:None)
        m._retrain_budget.take();m._retrain_budget.take()
        return m,core

    def test_engine_evidence_rejects_raw_bad_b1_pending_exchange_and_concealment(self):
        for defect in ('raw','b1','exchange','concealed','held','retraining','lost','unlocked'):
            m,c=self.modem()
            if defect=='raw':m.v42_mode='off'
            elif defect=='b1':c.rx_qualification['bit_errors']=1
            elif defect=='exchange':c._connected_rate=NS(exchange=object())
            elif defect=='held':c._p4rcv=NS(eq=NS(_input_guard=NS(holding=True)))
            elif defect=='retraining':m._retraining=True
            elif defect=='lost':c.lost_too_long=True
            elif defect=='unlocked':c._rx_state=None
            for _ in range(11):observe(m,c,8000,b'x'*128,concealed=defect=='concealed')
            self.assertFalse(m._retrain_budget.take(),defect)

    def test_engine_valid_delivery_renews_and_replacement_clears_progress(self):
        m,c=self.modem()
        for _ in range(9):observe(m,c,8000,b'x'*128)
        c._rx_state={}
        observe(m,c,8000,b'x'*128)
        self.assertFalse(m._retrain_budget.take())
        for _ in range(9):observe(m,c,8000,b'x'*128)
        self.assertTrue(m._retrain_budget.take())
