"""Policy decisions without channel labels; commands are physically tested elsewhere."""
from collections import deque
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

from v34_auto_rate import AutomaticRate, RatePolicyConfig
from v34_negotiation import offer, resolve


class Owner:
    def __init__(self, role='call', tx=33600, rx=33600, baud=24000/7):
        original=offer(33600);active=offer(33600);peer=offer(33600)
        active['max_c2a'],active['max_a2c']=((tx//2400,rx//2400) if role=='call' else (rx//2400,tx//2400))
        p=resolve(active,peer,role,baud,baud)
        self.monitor=NS(windows=deque(maxlen=32),profile=dict(baud=baud,nonlinear=False,precoded=False),
                        invalid_batches=0,discarded_symbols=0)
        self.rx=NS(_rx_state={'quality':self.monitor},_local_mp=original,negotiated=p,
                   _connected_rate=NS(exchange=None),_p4rcv=NS(eq=NS(_input_guard=NS(holding=False))))
        setattr(self.rx,'mp_ans' if role=='call' else 'mp_call',peer)
        self.eng=NS(bit_rate=p.tx.rate,rx_bit_rate=p.rx.rate,connected=True,_retraining=False,
                    _v42=NS(stats=dict(hdlc_good=0,hdlc_bad=0)))
        self.pending=None;self.calls=[];self.done=None;self.accept=True

    def engine(self):return self.eng
    def core(self,engine):return self.rx
    def request(self,parameter,**kwargs):
        self.calls.append((parameter,kwargs))
        if self.accept:self.pending=(None,dict(origin='automatic'))
        return self.accept
    def poll(self):
        if self.done is None:return None
        result=self.done;self.done=None;self.pending=None
        return result
    def finish(self,tx,rx,status='rate-exchanged'):
        self.eng.bit_rate,self.eng.rx_bit_rate=tx,rx
        self.rx.negotiated=NS(asymmetric=self.rx.negotiated.asymmetric,
            tx=NS(rate=tx,baud=self.monitor.profile['baud'],shaping=False),
            rx=NS(rate=rx,baud=self.monitor.profile['baud'],shaping=False))
        self.monitor=NS(windows=deque(maxlen=32),profile=dict(self.monitor.profile),
                        invalid_batches=0,discarded_symbols=0)
        self.rx._rx_state={'quality':self.monitor}
        self.done=dict(status=status,tx_rate=tx,rx_rate=rx)


class AutomaticRateTests(unittest.TestCase):
    def test_symmetric_peer_requires_both_rates_to_rise_together(self):
        owner,policy=self.setup_policy(tx=31200,rx=31200)
        owner.rx.negotiated=replace(owner.rx.negotiated,asymmetric=False)
        for _ in range(8):self.observe(owner,policy,.1)
        self.assertEqual(owner.calls[-1],(5,dict(origin='automatic',rates=(33600,33600))))

    def setup_policy(self,**kw):
        owner=Owner(**kw)
        cfg=RatePolicyConfig(bad_seconds=.5,up_seconds=2.,trial_seconds=1.,cooldown_seconds=.2,
                             up_backoff_seconds=5.)
        policy=AutomaticRate(owner,cfg)
        self.now=0.;self.interval=0
        policy.tick(active=True,loaded=True,now=0.)
        return owner,policy

    def observe(self,owner,policy,cost,seconds=.25,**kw):
        self.now+=seconds;self.interval+=1
        owner.monitor.windows.append(dict(end_interval=self.interval,
            symbols=seconds*owner.monitor.profile['baud'],path_cost_per_symbol=cost))
        return policy.tick(active=kw.pop('active',True),loaded=kw.pop('loaded',True),now=self.now,**kw)

    def test_persistent_loss_lowers_rx_only_in_either_asymmetric_role(self):
        for role in ('call','answer'):
            owner,policy=self.setup_policy(role=role,tx=24000)
            self.observe(owner,policy,.5)
            self.assertFalse(owner.calls)
            self.observe(owner,policy,.5)
            self.assertEqual(owner.calls,[(4,dict(origin='automatic',rates=(24000,31200)))])

    def test_transient_or_idle_clean_does_not_oscillate(self):
        owner,policy=self.setup_policy(rx=31200)
        self.observe(owner,policy,.5)
        self.observe(owner,policy,.1)
        for _ in range(20):self.observe(owner,policy,.1,loaded=False)
        self.assertFalse(owner.calls)
        for _ in range(8):self.observe(owner,policy,.1)
        self.assertEqual(owner.calls[-1][0],5)

    def test_bad_upward_trial_restores_exact_last_good_pair_and_backs_off(self):
        owner,policy=self.setup_policy(tx=24000,rx=31200)
        for _ in range(8):self.observe(owner,policy,.1)
        self.assertEqual(policy.trial_from,(24000,31200))
        owner.finish(24000,33600)
        policy.tick(active=True,loaded=True,now=self.now)
        policy.tick(active=True,loaded=True,now=self.now)
        for _ in range(3):self.observe(owner,policy,.5)
        self.assertEqual(owner.calls[-1],(4,dict(origin='automatic',rates=(24000,31200))))
        self.assertIsNone(policy.trial_from)
        self.assertGreater(policy.next_up,self.now)

    def test_peer_declines_up_no_repeat_until_backoff(self):
        owner,policy=self.setup_policy(rx=31200)
        for _ in range(8):self.observe(owner,policy,.1)
        owner.finish(33600,31200)
        policy.tick(active=True,loaded=True,now=self.now)
        policy.tick(active=True,loaded=True,now=self.now)
        for _ in range(12):self.observe(owner,policy,.1)
        self.assertEqual(len(owner.calls),1)
        self.assertIsNone(policy.trial_from)

    def test_unloaded_upward_trial_expires_to_prior_rates(self):
        owner,policy=self.setup_policy(rx=31200)
        for _ in range(8):self.observe(owner,policy,.1)
        owner.finish(33600,33600)
        policy.tick(active=True,loaded=True,now=self.now)
        policy.tick(active=True,loaded=True,now=self.now)
        policy.tick(active=True,loaded=False,now=self.now+31.)
        self.assertEqual(owner.calls[-1],(4,dict(origin='automatic',rates=(33600,31200))))
        self.assertIsNone(policy.trial_from)

    def test_stable_upward_trial_becomes_last_good(self):
        owner,policy=self.setup_policy(rx=31200)
        for _ in range(8):self.observe(owner,policy,.1)
        owner.finish(33600,33600)
        policy.tick(active=True,loaded=True,now=self.now)
        policy.tick(active=True,loaded=True,now=self.now)
        for _ in range(6):self.observe(owner,policy,.1)
        self.assertEqual(policy.last_good,(33600,33600))
        self.assertIsNone(policy.trial_from)
        self.assertEqual(len(owner.calls),1)

    def test_holds_transport_and_host_mode_break_persistence(self):
        for obstruction in ('hold','transport','host','discard'):
            owner,policy=self.setup_policy()
            self.observe(owner,policy,.5)
            if obstruction=='hold':owner.rx._p4rcv.eq._input_guard.holding=True
            if obstruction=='discard':owner.monitor.discarded_symbols+=2
            self.observe(owner,policy,.5,active=obstruction!='host',transport_ok=obstruction!='transport')
            owner.rx._p4rcv.eq._input_guard.holding=False
            if obstruction=='discard':
                self.assertEqual(len(owner.calls),0)
            else:
                self.observe(owner,policy,.5)
                self.assertFalse(owner.calls,obstruction)

    def test_retrain_only_and_disabled_modes(self):
        owner,policy=self.setup_policy()
        for _ in range(5):self.observe(owner,policy,.5,mode=0)
        self.assertFalse(owner.calls)
        for _ in range(2):self.observe(owner,policy,.5,mode=1)
        self.assertEqual(owner.calls[-1][0],1)

    def test_crc_backstop_counts_control_frames_without_timer_inference(self):
        owner,policy=self.setup_policy()
        for _ in range(2):
            owner.eng._v42.stats['hdlc_good']+=3;owner.eng._v42.stats['hdlc_bad']+=1
            self.observe(owner,policy,.1)
        self.assertEqual(owner.calls[-1][0],4)

    def test_original_mask_and_baud_limit_constrain_upward_ladder(self):
        owner,policy=self.setup_policy(rx=31200)
        owner.rx._local_mp['mask'] &= ~(1 << 13)
        self.assertIsNone(policy._next_rx(owner.rx,1))
        self.assertEqual(policy._next_rx(owner.rx,-1),28800)

    def test_host_transaction_is_never_consumed_and_overrides_up_trial(self):
        owner,policy=self.setup_policy(rx=31200)
        policy.trial_from=(33600,31200)
        owner.pending=(None,dict(origin='host'));owner.done={'sentinel':True}
        policy.tick(active=True,loaded=True,now=0.)
        self.assertEqual(owner.done,{'sentinel':True})
        self.assertIsNone(policy.trial_from)

    def test_refusal_and_request_history_are_bounded(self):
        owner,policy=self.setup_policy();owner.accept=False
        for i in range(50):policy._request(float(i),4,'test',(33600,31200))
        self.assertEqual(len(owner.calls),8)
        self.assertLessEqual(len(policy.history),32)
        self.assertEqual(len(policy.requests),8)


if __name__=='__main__':unittest.main()
