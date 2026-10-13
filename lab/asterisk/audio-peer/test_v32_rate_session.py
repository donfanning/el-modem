"""Actual paired PCM rate changes with retained tracking and in-flight LAPM."""
import random
import unittest
from test_v32_retrain import Pair
from v32_stream import V32Modem


class RateSessionTests(unittest.TestCase):
    def test_unanswered_exchange_tx_deadline_fails_cleanly_in_both_roles(self):
        from v32_answer import V32Answerer
        from v32_caller import rate_word
        for cls in (V32Modem,V32Answerer):
            with self.subTest(role=cls.__name__):
                m=cls(use_v8=False,echo=False)
                m.state='data';m.connected=True;m.v42_mode='lapm'
                m.tx.state='data';m._tx_started=True
                m._r1=m._r2=rate_word(9600)
                m.tx.feed_bytes(b'pending application bytes')
                pending=list(m.tx._tx_bytes)
                self.assertTrue(m.enable_rate_changes())
                self.assertTrue(m.request_rate_change(9600))
                m._rrp.tx.timeout_symbols=64
                for _ in range(10):
                    samples=m.tx_samples(160)
                    self.assertEqual(len(samples),160)
                    if m.state=='failed':break
                self.assertEqual(m.state,'failed')
                self.assertIn('rate-change transmit timeout',m.failed_reason)
                self.assertFalse(m.connected or m._rrp.active)
                self.assertEqual(m._rate_change_history[-1]['status'],'failed')
                self.assertEqual(list(m.tx._tx_bytes),pending)
                before=m._txn
                self.assertEqual(m.tx_samples(160),[0]*160)
                self.assertEqual(m._txn,before+160)
                self.assertEqual(m._retrain_attempts,0)

    def test_receive_deadline_drops_carrier_and_does_not_report_completion(self):
        from v32_caller import rate_word
        m=V32Modem(use_v8=False,echo=False)
        m.state='data';m.connected=True;m.v42_mode='lapm';m.tx.state='data'
        m._r1=rate_word(9600)
        self.assertTrue(m.enable_rate_changes());self.assertTrue(m.request_rate_change(9600))
        m._t=10*m.fs+1;m._rrp.poll()
        self.assertEqual(m.state,'failed')
        self.assertEqual(m.failed_reason,'V32 rate-change timeout')
        self.assertFalse(m.connected or m._rrp.active)
        self.assertEqual(m._rate_change_history[-1]['status'],'failed')
        self.assertEqual(m.tx_samples(160),[0]*160)

    def test_changed_repeated_wire_offer_fails_without_receive_exception(self):
        from itertools import islice
        from v32_answer import V32Answerer
        from v32_caller import RATE_BITS,rate_word
        from v32_rate_change import RateChangeTx,RateControlReader
        from v32_rate_session import RateSession
        for role,cls,other in (('call',V32Modem,'answer'),('answer',V32Answerer,'call')):
            with self.subTest(role=role):
                m=cls(use_v8=False,echo=False)
                m.state='data';m.connected=True;m.v42_mode='lapm'
                m.tx.state='data'
                session=m._rrp=RateSession(m)
                session.active=True;session.started=m._t;session.mode='control'
                session.events.append(dict(status='pending'))
                session.control=RateControlReader(role,rate_word(*RATE_BITS))
                tx=RateChangeTx(other,RATE_BITS,14400)
                list(islice(tx.symbols(lambda:1),64))
                # The generator is suspended on its last preamble yield;
                # seed the following control stream from that physical point.
                tx.encoder.q=3 if other=='call' else 0
                index=0
                for word in (rate_word(9600),rate_word(9600),
                             rate_word(12000),rate_word(12000)):
                    for z in tx._control_word(word):
                        session.symbol(z,index);index+=1
                self.assertEqual(m.state,'failed')
                self.assertFalse(m.connected or session.active)
                self.assertIn('received offer changed',m.failed_reason)
                self.assertEqual(session.events[-1]['status'],'failed')
                self.assertIsNone(session.data)

    def test_pcm_no_common_rate_clears_both_roles_without_retraining(self):
        p=Pair();p.until(p.ready)
        p.c.start_v42(True,detect=False);p.a.start_v42(False,detect=False)
        p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
        p.c._configured_rates=(9600,);p.a._configured_rates=(14400,)
        for m in (p.c,p.a):self.assertTrue(m.enable_rate_changes())
        self.assertTrue(p.c.request_rate_change(9600))
        for _ in range(200):
            if all(m.state=='failed' for m in (p.c,p.a)):break
            p.step()
        for m in (p.c,p.a):
            self.assertEqual(m.state,'failed')
            self.assertEqual(m.failed_reason,'V32 rate exchange cleared')
            self.assertEqual(m._rate_change_history[-1]['status'],'cleared')
            self.assertFalse(m.connected or m._rrp.active)
            self.assertEqual(m._retrain_attempts,0)

    def test_disabled_and_unresolved_link_cannot_start_exchange(self):
        m=V32Modem(use_v8=False,echo=False)
        self.assertFalse(m.enable_rate_changes());self.assertFalse(m.request_rate_change(9600))
        m.connected=True;m.state='data'
        self.assertFalse(m.enable_rate_changes())

    def test_pcm_down_up_retains_frontend_links_and_exact_inflight_bytes(self):
        for first_role in ('call','answer'):
            with self.subTest(first_role=first_role):
                p=Pair();p.until(p.ready)
                p.c.start_v42(True,detect=False);p.a.start_v42(False,detect=False)
                p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
                links=(p.c._v42,p.a._v42);equalizers=(p.c.eq,p.a.eq)
                expected_c=bytearray();expected_a=bytearray()
                def queue(seed,size):
                    c=random.Random(seed).randbytes(size);a=random.Random(seed+1).randbytes(size)
                    expected_c.extend(c);expected_a.extend(a);p.c.feed_bytes(c);p.a.feed_bytes(a)
                def payload():
                    p.until(lambda:len(p.got_a)>=len(expected_c) and len(p.got_c)>=len(expected_a),carrier=True)
                    self.assertEqual(p.got_a,expected_c);self.assertEqual(p.got_c,expected_a)
                queue(501,4096);payload()
                for m in (p.c,p.a):self.assertTrue(m.enable_rate_changes())
                first,second=(p.c,p.a) if first_role=='call' else (p.a,p.c)
                queue(503,8192)
                p.until(lambda:len(p.got_a)>=4400 and len(p.got_c)>=4400)
                self.assertTrue(first.request_rate_change(9600))
                self.assertFalse(first.request_rate_change(4800))
                p.until(lambda:all(not m._rrp.active and m.bit_rate==9600 for m in (p.c,p.a)),carrier=True)
                payload()
                queue(505,8192)
                p.until(lambda:len(p.got_a)>=12600 and len(p.got_c)>=12600)
                self.assertTrue(second.request_rate_change(14400))
                p.until(lambda:all(not m._rrp.active and m.bit_rate==14400 for m in (p.c,p.a)),carrier=True)
                payload()
                for m,link,eq in zip((p.c,p.a),links,equalizers):
                    self.assertIs(m._v42,link);self.assertIs(m.eq,eq)
                    self.assertEqual([e['rate'] for e in m._rrp.events],[9600,14400])
                    self.assertTrue(all(e['status']=='complete' for e in m._rrp.events))
                    self.assertTrue(all(e['rx_convention']=='reset' for e in m._rrp.events))
                    self.assertEqual(m._retrain_attempts,0)

    def test_full_retrain_interrupts_pending_exchange_and_reinstalls_controller(self):
        from v32_rate_session import RateSession
        m=V32Modem(use_v8=False,echo=False)
        m.state='data';m.connected=True;m.bit_rate=9600;m.v42_mode='lapm'
        self.assertTrue(m.enable_rate_changes())
        old=m._rrp;old.active=True;old.events.append(dict(status='pending'))
        history=m._rate_change_history
        self.assertTrue(m._restart_line('peer request'))
        self.assertIsNone(m._rrp)
        self.assertEqual(history[-1]['status'],'interrupted')
        self.assertIs(m._rate_change_history,history)
        self.assertTrue(m._rate_changes_enabled)
        m.state='data';m._retrain_completed()
        self.assertIsInstance(m._rrp,RateSession)
        self.assertIsNot(m._rrp,old)
        self.assertIs(m._rrp.events,history)
        self.assertEqual(m._rrp.gain,1+0j)
        self.assertIsNone(m._rrp.detector.last_index)
        m._rrp.active=True;history.append(dict(status='pending'))
        m._fail('carrier lost')
        self.assertFalse(m._rrp.active)
        self.assertEqual(history[-1],dict(status='failed',reason='carrier lost'))

    def test_pcm_fast_full_fast_keeps_history_and_exact_lapm_payload(self):
        p=Pair();p.until(p.ready)
        p.c.start_v42(True,detect=False);p.a.start_v42(False,detect=False)
        p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
        for m in (p.c,p.a):self.assertTrue(m.enable_rate_changes())
        links=(p.c._v42,p.a._v42)
        expected_c=bytearray();expected_a=bytearray()
        for epoch in range(4):
            c=random.Random(601+epoch).randbytes(4096)
            a=random.Random(701+epoch).randbytes(4096)
            expected_c.extend(c);expected_a.extend(a)
            p.c.feed_bytes(c);p.a.feed_bytes(a)
            p.until(lambda:len(p.got_a)>=len(expected_c) and len(p.got_c)>=len(expected_a),carrier=True)
            self.assertEqual(p.got_a,expected_c);self.assertEqual(p.got_c,expected_a)
            if epoch==0:
                self.assertTrue(p.c.request_rate_change(9600))
                p.until(lambda:all(not m._rrp.active and m.bit_rate==9600 for m in (p.c,p.a)),carrier=True)
            elif epoch==1:
                self.assertTrue(p.a.request_retrain(12000))
                p.until(p.ready,carrier=True)
                self.assertEqual((p.c.bit_rate,p.a.bit_rate),(12000,12000))
                self.assertTrue(all(m._rrp is not None for m in (p.c,p.a)))
            elif epoch==2:
                self.assertTrue(p.a.request_rate_change(14400))
                p.until(lambda:all(not m._rrp.active and m.bit_rate==14400 for m in (p.c,p.a)),carrier=True)
        for m,link in zip((p.c,p.a),links):
            self.assertIs(m._v42,link)
            self.assertEqual([e['rate'] for e in m._rate_change_history],[9600,14400])
            self.assertEqual(m._retrain_attempts,1)

    def test_simultaneous_requests_choose_common_rate_with_exact_inflight_payload(self):
        p=Pair();p.until(p.ready)
        p.c.start_v42(True,detect=False);p.a.start_v42(False,detect=False)
        p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
        for m in (p.c,p.a):self.assertTrue(m.enable_rate_changes())
        c=random.Random(803).randbytes(8192);a=random.Random(804).randbytes(8192)
        p.c.feed_bytes(c);p.a.feed_bytes(a)
        p.until(lambda:len(p.got_c)>=300 and len(p.got_a)>=300)
        self.assertTrue(p.c.request_rate_change(9600));self.assertTrue(p.a.request_rate_change(12000))
        p.until(lambda:all(not m._rrp.active and m.bit_rate==9600 for m in (p.c,p.a)),carrier=True)
        p.until(lambda:len(p.got_a)>=len(c) and len(p.got_c)>=len(a),carrier=True)
        self.assertEqual(p.got_a,c);self.assertEqual(p.got_c,a)
        for m in (p.c,p.a):
            self.assertEqual(m._rate_change_history[-1]['origin'],'local')
            self.assertEqual(m._rate_change_history[-1]['rx_convention'],'reset')
            self.assertEqual(m._retrain_attempts,0)

    def test_compression_dictionaries_survive_fast_down_and_up(self):
        p=Pair();p.until(p.ready)
        p.c.start_v42(True,detect=False,compression=(3,512,8))
        p.a.start_v42(False,detect=False,compression=(3,512,8))
        p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
        dictionaries=[(m._v42._enc,m._v42._dec) for m in (p.c,p.a)]
        for m in (p.c,p.a):self.assertTrue(m.enable_rate_changes())
        c=random.Random(901).randbytes(8192);a=random.Random(902).randbytes(8192)
        p.c.feed_bytes(c);p.a.feed_bytes(a)
        p.until(lambda:len(p.got_c)>=300 and len(p.got_a)>=300)
        self.assertTrue(p.c.request_rate_change(9600))
        p.until(lambda:all(not m._rrp.active and m.bit_rate==9600 for m in (p.c,p.a)),carrier=True)
        p.until(lambda:len(p.got_c)>=2000 and len(p.got_a)>=2000)
        self.assertTrue(p.a.request_rate_change(14400))
        p.until(lambda:all(not m._rrp.active and m.bit_rate==14400 for m in (p.c,p.a)),carrier=True)
        p.until(lambda:len(p.got_a)>=len(c) and len(p.got_c)>=len(a),carrier=True)
        self.assertEqual(p.got_a,c);self.assertEqual(p.got_c,a)
        for m,(enc,dec) in zip((p.c,p.a),dictionaries):
            self.assertIs(m._v42._enc,enc);self.assertIs(m._v42._dec,dec)
            self.assertEqual(m._retrain_attempts,0)
