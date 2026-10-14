"""Same-rate resynchronization must retain both negotiated directions."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from v34_negotiation import offer, resolve
from v34_rate_live import ConnectedRate

class ResyncOfferTests(unittest.TestCase):
    def test_scalar_ceiling_never_raises_original_lower_direction(self):
        original=offer(33600);original['max_c2a']=10
        peer=offer(33600)
        p=resolve(original,peer,'call',24000/7,24000/7)
        engine=SimpleNamespace(_local_mp=copy.deepcopy(original),mp_ans=peer,
            negotiated=p,rtd_samples=0,_qam=None,_ev=lambda event:None)
        ctrl=ConnectedRate.__new__(ConnectedRate);ctrl.engine=engine;ctrl.role='call'
        with patch('v34_rate_live.RateQamBridge'):ctrl._start(True,31200)
        self.assertEqual((ctrl.exchange.local['max_c2a'],ctrl.exchange.local['max_a2c']),(10,13))
        self.assertEqual(engine._local_mp,original)

    def test_directional_receive_cap_keeps_transmit_and_original_offer(self):
        for role in ('call','answer'):
            original=offer(33600);peer=offer(33600)
            p=resolve(original,peer,role,24000/7,24000/7)
            engine=SimpleNamespace(_local_mp=copy.deepcopy(original),mp_ans=peer,mp_call=peer,
                negotiated=p,rtd_samples=0,_qam=None,_ev=lambda event:None)
            ctrl=ConnectedRate.__new__(ConnectedRate);ctrl.engine=engine;ctrl.role=role
            with patch('v34_rate_live.RateQamBridge'):ctrl._start(True,rates=(33600,31200))
            result=resolve(ctrl.exchange.local,peer,role,p.tx.baud,p.rx.baud)
            self.assertEqual((result.tx.rate,result.rx.rate),(33600,31200))
            self.assertEqual(engine._local_mp,original)
            for rates in ((33600,36000),(33600,31000),(0,31200)):
                with self.assertRaises(ValueError):ctrl._start(True,rates=rates)
            with self.assertRaises(ValueError):ctrl._start(True,31200,rates=(33600,31200))

    def check_role(self, role):
        original=offer(33600);active=offer(33600);active['max_c2a']=10
        peer=offer(33600)
        p=resolve(active,peer,role,3428.5714285714284,3428.5714285714284,1959.,1959.)
        engine=SimpleNamespace(_local_mp=copy.deepcopy(original),mp_ans=peer,mp_call=peer,
            negotiated=p,rtd_samples=0,_qam=None,_ev=lambda event:None)
        ctrl=ConnectedRate.__new__(ConnectedRate);ctrl.engine=engine;ctrl.role=role
        with patch('v34_rate_live.RateQamBridge'):
            ctrl._start(True,resync=True)
        self.assertEqual(engine._local_mp,original)
        self.assertEqual((ctrl.exchange.local['max_c2a'],ctrl.exchange.local['max_a2c']),(10,14))
        result=resolve(ctrl.exchange.local,peer,role,p.tx.baud,p.rx.baud,p.tx.carrier,p.rx.carrier)
        self.assertEqual((result.tx.rate,result.rx.rate),(p.tx.rate,p.rx.rate))
        self.assertNotEqual(p.tx.rate,p.rx.rate)

    def test_caller_retains_lower_tx_and_higher_rx(self):self.check_role('call')
    def test_answerer_retains_higher_tx_and_lower_rx(self):self.check_role('answer')
