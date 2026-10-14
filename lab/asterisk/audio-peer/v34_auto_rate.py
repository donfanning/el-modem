"""Experimental quality-driven rate control; no payload oracle/channel truth.

Production enables this only with V34_AUTO_RATE=1. Linear unprecoded physical
thresholds are development settings, not universal BER estimates. HDLC loss
corroborates/backstops them. Timers, RTP damage and power holds are not noise.
"""
from collections import deque
from dataclasses import dataclass
import math
import time

from v34_negotiation import supported


@dataclass(frozen=True)
class RatePolicyConfig:
    bad_cost: float = .30
    clean_cost: float = .18
    bad_seconds: float = 1.0
    up_seconds: float = 60.0
    trial_seconds: float = 10.0
    trial_timeout_seconds: float = 30.0
    cooldown_seconds: float = 5.0
    up_backoff_seconds: float = 60.0
    crc_bad_fraction: float = .05
    crc_min_frames: int = 4

    def __post_init__(self):
        values=(self.bad_cost,self.clean_cost,self.bad_seconds,self.up_seconds,
                self.trial_seconds,self.trial_timeout_seconds,self.cooldown_seconds,self.up_backoff_seconds)
        if (any(not math.isfinite(v) or v<=0 for v in values)
                or self.trial_timeout_seconds<=self.trial_seconds
                or self.clean_cost>=self.bad_cost or not 0<self.crc_bad_fraction<1
                or self.crc_min_frames<1):
            raise ValueError('invalid automatic-rate configuration')


class AutomaticRate:
    def __init__(self, control, config=None):
        self.control = control
        self.config = config or RatePolicyConfig()
        self.epoch = None
        self.cursor = None
        self.good_seconds = self.bad_seconds = 0.
        self.next_action = self.next_up = 0.
        self.up_backoff = self.config.up_backoff_seconds
        self.last_good = self.trial_from = None
        self.trial_clean = 0.
        self.trial_deadline = None
        self.history = deque(maxlen=32)
        self.requests = deque(maxlen=8)
        self._crc = (0, 0)
        self._crc_pending = [0, 0]
        self.reason = 'waiting for data'
        self.mode = 0

    def _record(self, now, event, **fields):
        self.history.append(dict(monotonic=now, event=event, **fields))

    def _reset_window(self, quality, counters):
        self.good_seconds = self.bad_seconds = 0.
        self.cursor = quality.windows[-1]['end_interval'] if quality and quality.windows else None
        self._crc = counters
        self._crc_pending = [0, 0]

    @staticmethod
    def _rates(engine):
        return (engine.bit_rate, engine.rx_bit_rate)

    def _next_rx(self, core, direction):
        role = 'call' if hasattr(core, 'mp_ans') else 'answer'
        key = 'max_a2c' if role == 'call' else 'max_c2a'
        p = core.negotiated.rx
        original = core._local_mp
        rate = p.rate + direction * 2400
        while 2400 <= rate <= original[key]*2400:
            symmetric=not core.negotiated.asymmetric
            tx_key='max_c2a' if role=='call' else 'max_a2c'
            tx=core.negotiated.tx
            tx_ok=(not symmetric or (rate<=original[tx_key]*2400 and supported(rate,tx.baud,tx.shaping)))
            if original['mask'] & (1 << (rate//2400-1)) and supported(rate,p.baud,p.shaping) and tx_ok:
                return rate
            rate += direction * 2400
        return None

    @staticmethod
    def _pair(core, engine, rx):
        return (engine.bit_rate,rx) if core.negotiated.asymmetric else (rx,rx)

    def _request(self, now, parameter, reason, rates=None):
        # At most eight requests per five minutes, including refusals. Bounded
        # history, but no lifetime cap that disables a long-lived modem.
        while self.requests and now-self.requests[0]>=300:
            self.requests.popleft()
        if len(self.requests)==8:
            self.next_action=max(self.next_action,self.requests[0]+300)
            self.reason='request frequency limit'
            return False
        self.requests.append(now)
        accepted=self.control.request(parameter,origin='automatic',rates=rates)
        self._record(now,'request' if accepted else 'refused',parameter=parameter,
                     reason=reason,rates=list(rates) if rates else None)
        self.good_seconds=self.bad_seconds=0.
        self._crc_pending=[0,0]
        self.next_action=now+(self.config.cooldown_seconds if accepted else 30.)
        return accepted

    def _backoff(self, now, reason):
        self.next_up=now+self.up_backoff
        self._record(now,'up-backoff',seconds=self.up_backoff,reason=reason)
        self.up_backoff=min(600.,2*self.up_backoff)

    def tick(self, *, active, mode=2, loaded=False, transport_ok=True, now=None):
        """Called by the serial/receive owner, never by the TX thread.

        ``loaded`` is real application activity/queued traffic, not expected
        bytes. ``transport_ok`` is continuity evidence, never impairment truth.
        Return automatic completion only; leave host transactions to the DCE.
        """
        now=time.monotonic() if now is None else now
        self.mode=mode
        if mode!=2:self.trial_from=None
        pending=self.control.pending
        if pending is not None:
            if pending[-1].get('origin')!='automatic':
                self.reason='host control active'
                if self.trial_from is not None:
                    self._record(now,'upward trial cancelled by host control')
                    self.trial_from=None
                self.good_seconds=self.bad_seconds=0.
                return None
            result=self.control.poll()
            if result is None:
                self.reason='automatic exchange active'
                return None
            self._record(now,'completed',status=result['status'],
                         tx_rate=result['tx_rate'],rx_rate=result['rx_rate'])
            if self.trial_from is not None:
                if (result['status']!='rate-exchanged'
                        or result['rx_rate']<=self.trial_from[1]):
                    self._backoff(now,'upward trial declined or fell back to retrain')
                    self.trial_from=None
                else:
                    self.trial_clean=0.
                    self.trial_deadline=now+self.config.trial_timeout_seconds
            self.next_action=max(self.next_action,now+self.config.cooldown_seconds)
            return result
        eng=self.control.engine()
        core=self.control.core(eng)
        state=getattr(core,'_rx_state',None) or {}
        quality=state.get('quality')
        link=getattr(eng,'_v42',None)
        stats=getattr(link,'stats',{})
        counters=(stats.get('hdlc_good',0),stats.get('hdlc_bad',0))
        rate=getattr(core,'_connected_rate',None)
        guard=getattr(getattr(getattr(core,'_p4rcv',None),'eq',None),'_input_guard',None)
        if (not active or mode not in (1,2) or core is None or quality is None
                or not getattr(eng,'connected',False) or getattr(eng,'_retraining',False)
                or getattr(rate,'exchange',None) is not None or getattr(guard,'holding',False)
                or not transport_ok or quality.invalid_batches):
            if mode==0:self.trial_from=None
            self.reason='suspended'
            self._reset_window(quality,counters)
            return None
        token=(id(core),id(quality))
        if token!=self.epoch:
            self.epoch=token
            if self.trial_from is not None and eng.rx_bit_rate<=self.trial_from[1]:
                self._backoff(now,'peer changed rate during upward trial')
                self.trial_from=None
            self._reset_window(quality,counters)
            self.reason='new data epoch'
            return None
        if (mode==2 and self.trial_from is not None and self.trial_deadline is not None
                and now>=max(self.trial_deadline,self.next_action)):
            self._backoff(now,'upward trial not qualified before deadline')
            if self._request(now,4,'unqualified upward trial deadline',self.trial_from):
                self.trial_from=None
            return None
        # Preserve bad-frame evidence between the much shorter DCE polls and
        # physical windows; counters can reset on a new retained-link instance.
        for i,value in enumerate(counters):
            self._crc_pending[i]+=max(0,value-self._crc[i])
        self._crc=counters
        fresh=[r for r in quality.windows if self.cursor is None or r['end_interval']>self.cursor]
        if not fresh:return None
        self.cursor=fresh[-1]['end_interval']
        if now<self.next_action:
            self._reset_window(quality,counters)
            self.reason='cooldown'
            return None
        good,bad=self._crc_pending;self._crc_pending=[0,0]
        crc_bad=(bad>0 and good+bad>=self.config.crc_min_frames
                 and bad/(good+bad)>=self.config.crc_bad_fraction)
        # A hold/discard can occur and end between polls. Do not interpret the
        # windows on opposite sides as one persistent clean or bad interval.
        if getattr(self,'_discarded',None)!=quality.discarded_symbols:
            self._discarded=quality.discarded_symbols
            self.good_seconds=self.bad_seconds=0.
        physical=not (quality.profile['nonlinear'] or quality.profile['precoded'])
        for row in fresh:
            seconds=row['symbols']/quality.profile['baud']
            poor=(physical and row['path_cost_per_symbol']>=self.config.bad_cost) or crc_bad
            clean=(physical and row['path_cost_per_symbol']<=self.config.clean_cost and bad==0)
            self.bad_seconds=self.bad_seconds+seconds if poor else 0.
            self.good_seconds=self.good_seconds+seconds if clean and loaded else 0.
            if self.trial_from is not None:
                self.trial_clean=self.trial_clean+seconds if clean and loaded else 0.
            if self.bad_seconds>=self.config.bad_seconds:
                self.reason='persistent degradation'
                if mode==1:
                    self._request(now,1,self.reason)
                elif self.trial_from is not None:
                    restore=self.trial_from
                    self._backoff(now,'upward trial degraded')
                    if self._request(now,4,'restore last good rate',restore):self.trial_from=None
                else:
                    rx=self._next_rx(core,-1)
                    if rx is None:self._request(now,1,'minimum supported rate; retrain')
                    else:self._request(now,4,self.reason,self._pair(core,eng,rx))
                return None
        if self.trial_from is not None and self.trial_clean>=self.config.trial_seconds:
            self.last_good=self._rates(eng);self.trial_from=None
            self.up_backoff=self.config.up_backoff_seconds
            self._record(now,'upward trial qualified',rates=list(self.last_good))
        if self.trial_from is None and self.good_seconds>=self.config.up_seconds:
            self.last_good=self._rates(eng)
            rx=self._next_rx(core,1) if mode==2 and now>=self.next_up else None
            if rx is not None:
                prior=self.last_good
                if self._request(now,5,'sustained clean loaded interval',self._pair(core,eng,rx)):
                    self.trial_from=prior;self.trial_clean=0.
        self.reason='monitoring'
        return None

    def report(self):
        return dict(experimental=True,mode=self.mode,reason=self.reason,good_seconds=self.good_seconds,
                    bad_seconds=self.bad_seconds,last_good=self.last_good,
                    trial_from=self.trial_from,next_up_monotonic=self.next_up,
                    trial_deadline_monotonic=self.trial_deadline,
                    nonlinear_or_precoded_physical_policy_qualified=False,
                    history=list(self.history),config=dict(self.config.__dict__))
