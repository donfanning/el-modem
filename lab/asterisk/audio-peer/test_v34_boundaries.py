"""E provenance is independent of parser latency and receive block boundaries."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from v34_rx import Phase4Receiver
from v34_mp import build_mp
from v34_negotiation import final_mp

class BoundaryTests(unittest.TestCase):
    def receiver(self,c16,bits):
        r=object.__new__(Phase4Receiver)
        r.bits=bits;r.pos=0;r.frames=[];r.mp_spans=[];r.e_span=None
        r.c16=c16;r.k_trn=16;r.trn_end=700;r.eq=SimpleNamespace(k=2000)
        r.b1_symbol=None
        return r

    def test_retrospective_e_uses_bit_position(self):
        for c16 in (False,True):
            mp=build_mp(1,ack=1)
            r=self.receiver(c16,mp+[1]*20+[0]*40)
            r._parse(None)
            self.assertEqual(r.b1_symbol,716+(len(mp)+20)//(4 if c16 else 2))
            self.assertNotEqual(r.b1_symbol,r.eq.k)
            self.assertEqual(r.e_span['end_bit'],len(mp)+20)

    def test_incremental_and_buffered_parsing_agree(self):
        bits=build_mp(0)+build_mp(1,ack=1)+[1]*20
        expected=716+len(bits)//4
        for chunk in (1,3,7,20,80,len(bits)):
            r=self.receiver(True,[])
            for start in range(0,len(bits),chunk):
                r.bits.extend(bits[start:start+chunk]);r._parse(None)
            self.assertEqual(r.b1_symbol,expected)
            self.assertEqual(len(r.mp_spans),2)

    def test_old_s_search_window_keeps_absolute_sample_anchor(self):
        from v34_call import V34Caller
        engine=V34Caller()
        engine.call_baud=engine.ans_baud=24000/7
        engine.ans_fc=1959.
        engine.p3_tx_started=0;engine._p3_last=0;engine.rx_t=12000
        engine._rx=np.zeros(12000);engine._p3_search_lo=0
        # Search visits 10000, 8000, 6000; the match is in the third window.
        with patch('v34_rx.detect_s_flip',side_effect=[None,None,('flip',123)]), \
                patch.object(engine,'_send_j_prime') as send:
            engine._phase3_wait_s()
        self.assertEqual(engine._p4_flip_abs,6123)
        send.assert_called_once()

    def test_flip_center_retains_sbar_acquisition_context(self):
        from v34_call import V34Caller
        from v34_tx import s_signal, trn_symbols, qam_modulate
        engine=V34Caller(data_rate=33600)
        engine.call_baud=engine.ans_baud=24000/7
        engine.call_fc=engine.ans_fc=1959.
        trn,_=trn_symbols(1000,'answer',constellation=16)
        signal=qam_modulate(np.r_[s_signal(128),s_signal(16,True),trn],
            engine.ans_baud,engine.ans_fc,1000.)
        engine._rx=signal;engine.rx_t=len(signal);engine._p4_last=0
        engine._p4_flip_abs=round((128+8)*8000/engine.ans_baud)
        engine.stage='p4_trn';engine._p4rcv=None;engine._rx_p4_c=16
        engine._phase4_rx()
        self.assertIsNotNone(engine._p4rcv)
        self.assertLess(engine._p4_x0,engine._p4_flip_abs-8*8000/engine.ans_baud)

    def test_final_acknowledged_request(self):
        first={'ack':0,'max_c2a':14}
        ack={'ack':1,'max_c2a':13}
        self.assertEqual(final_mp([first,ack,'E']),ack)
        self.assertEqual(final_mp([first,ack,first]),ack)
        self.assertIsNone(final_mp(['E']))

if __name__=='__main__':unittest.main()
