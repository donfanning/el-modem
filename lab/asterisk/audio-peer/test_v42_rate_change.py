"""Physical rate changes preserve link data and timer duration, not old bit budgets."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v42_lapm import V42Link
from mnp_link import MnpLink
from test_v42_layer import BitEngine, pump


class RateChangeTests(unittest.TestCase):
    def test_lapm_pending_timer_keeps_seconds_up_and_down(self):
        for old,new in [(7200,33600),(33600,7200)]:
            with self.subTest(old=old,new=new):
                link=V42Link(True,detect=False,bit_rate=old)
                link.bits_sent=old*20
                link._t401_at=link.bits_sent+old//2
                link._tx.extend([1,0,1]);link._txq.extend(b'queued')
                queue=list(link._tx);before=link.bits_sent;flags=link._establish_at
                link.set_bit_rate(new)
                self.assertAlmostEqual((link._t401_at-before)/new,.5,delta=1/new)
                self.assertAlmostEqual(link._t401_bits/new,
                                       max(1.5,2.2*(link.n401+8)*8/new+1.0),delta=1/new)
                self.assertEqual(link.bits_sent,before)
                self.assertEqual(link._establish_at,flags)
                self.assertEqual(list(link._tx),queue)
                self.assertEqual(link._txq,b'queued')
                link.set_bit_rate(new)
                self.assertEqual(link._t401_at,before+new//2)
                link._t401_at=before-10
                link.set_bit_rate(old)
                self.assertEqual(link._t401_at,before)
                link._t401_at=None;link.set_bit_rate(new)
                self.assertIsNone(link._t401_at)

    def test_detection_deadlines_preserve_remaining_time_and_odp_bits(self):
        link=V42Link(True,bit_rate=9600,t400=.75)
        link.bits_sent=2400
        link.set_bit_rate(24000)
        self.assertEqual(link._t400_at,2400+12000)
        self.assertEqual(link._answer_detect_at,2400+30000)
        self.assertEqual(link.bits_sent%36,2400%36)

    def test_mnp_timers_preserve_remaining_seconds_and_credit(self):
        for old,new in [(7200,33600),(33600,7200)]:
            with self.subTest(old=old,new=new):
                link=MnpLink(True,bit_rate=old)
                link._bits=old*20
                for name in ('_lr_due','_resp_t401','_t401','_t402','_t404'):
                    setattr(link,name,link._bits+old//2)
                link._out=[[1,b'unacked',2]];link._txbuf.extend(b'queued')
                credit=link._s_k;clock=link._bits
                link.set_bit_rate(new)
                for name in ('_lr_due','_resp_t401','_t401','_t402','_t404'):
                    self.assertAlmostEqual((getattr(link,name)-clock)/new,.5,delta=1/new)
                self.assertEqual(link._out,[[1,b'unacked',2]])
                self.assertEqual(link._txbuf,b'queued');self.assertEqual(link._s_k,credit)
                self.assertEqual(link._est_t401,new*2)
                link._t401=None;link.set_bit_rate(old);self.assertIsNone(link._t401)

    def test_layer_rebases_detection_and_updates_fallback_parameters(self):
        engine=BitEngine(9600);engine.start_v42(True,rate=9600)
        engine._v42_bits=2400;engine._v42.bits_sent=2400
        previous=engine._v42;detect=engine._v42_detect_deadline;establish=engine._v42_establish_deadline
        engine.set_v42_rate(24000)
        self.assertIs(engine._v42,previous)
        self.assertAlmostEqual((engine._v42_detect_deadline-2400)/24000,(detect-2400)/9600,delta=1/24000)
        self.assertAlmostEqual((engine._v42_establish_deadline-2400)/24000,(establish-2400)/9600,delta=1/24000)
        self.assertEqual(engine._v42_params[0],24000)
        engine=BitEngine();engine.set_v42_rate(24000);self.assertIsNone(engine._v42)

    def test_compressed_pending_payload_survives_upward_then_downward_change(self):
        a,b=BitEngine(7200),BitEngine(7200)
        for engine,calling in [(a,True),(b,False)]:
            engine.start_v42(calling,rate=7200,compression=(3,1024,8))
        pump(a,b,7200*3)
        self.assertEqual((a.v42_mode,b.v42_mode),('lapm','lapm'))
        text=(b'rate changes must preserve the compression dictionary and pending bytes. '*70)
        original=(a._v42,b._v42);dictionaries=(a._v42._enc,b._v42._dec)
        a.v42_feed_bytes(text);b.v42_feed_bytes(text[::-1]);pump(a,b,400)
        self.assertGreater(a._v42.pending()+b._v42.pending(),0)
        for rate in (33600,7200):
            a.set_v42_rate(rate);b.set_v42_rate(rate);pump(a,b,rate*3)
            a.v42_feed_bytes(text);b.v42_feed_bytes(text[::-1])
        pump(a,b,7200*5)
        self.assertEqual(bytes(b.out),text*3);self.assertEqual(bytes(a.out),text[::-1]*3)
        self.assertEqual((a._v42,b._v42),original)
        self.assertEqual((a._v42._enc,b._v42._dec),dictionaries)
        self.assertEqual((a.v42_mode,b.v42_mode),('lapm','lapm'))

    def test_both_modem_roles_update_retained_link_at_retrain_completion(self):
        from v34_modem import V34Modem
        from v34_answer_modem import V34AnswerModem
        for cls,name in [(V34Modem,'caller'),(V34AnswerModem,'answerer')]:
            with self.subTest(role=name),patch.dict('os.environ',{'V34_CALLER_RETRAIN':'0'}):
                modem=cls(with_v8=False);modem.state='data';modem.bit_rate=7200
                modem._retraining=True;modem.start_v42(True,rate=7200,detect=False)
                previous=modem._v42
                core=SimpleNamespace(rx=lambda x:None,events=[],failed_reason=None,answerer_retrain=False,
                    lost_too_long=False,stage='data',_rx_state=object(),data_rate=16800,rx_rate=14400)
                setattr(modem,name,core)
                modem.feed_pcm(bytes(320))
                self.assertFalse(modem._retraining)
                self.assertIs(modem._v42,previous)
                self.assertEqual(previous.rate,16800)
                self.assertEqual(modem.rx_bit_rate,14400)


if __name__=='__main__':unittest.main()
