"""Experimental live section-8 receive/controller, explicitly installed in LAPM.

Not enabled by a DCE default. Frontend clocks/taps and application link survive;
control recognition never uses an expected payload or recorded reference.
"""
from collections import deque
from v32_caller import POINTS,RATE_BITS,rate_word,TCM_BITS,ZTAB4,V32_TCM
from v32_rx import TLEN
from v32_rate_change import (RateChangeTx,RateControlReader,RateDataReader,
                            RatePreambleDetector,parse_offer)


class RateSession:
    def __init__(self,engine):
        self.engine=engine;self.gain=1+0j;self.mode='data';self.active=False
        self.detector=RatePreambleDetector(self.role)
        self.history=deque(maxlen=88);self.hold=deque();self.current_index=0
        self.seed_history=deque(maxlen=88);self.prior_seed=None
        self.control=None;self.data=None;self.tx=None
        self.events=engine._rate_change_history;self.started=None
        self.rx_ready=False

    @property
    def role(self):return 'call' if self.engine.tx.role=='caller' else 'answer'

    def request(self,desired):
        m=self.engine
        if self.active or m.state!='data' or not m.connected or m._retraining or m.v42_mode!='lapm':return False
        if desired not in m._configured_rates:return False
        peer=m._r1 if self.role=='call' else m._r2
        try:parse_offer(peer)
        except (ValueError,TypeError):return False
        self.tx=RateChangeTx(self.role,m._configured_rates,desired)
        m.tx.start_rate_change(self.tx)
        self.active=True;self.started=m._t;self.rx_ready=False
        self.events.append(dict(origin='local',requested_rate=desired,previous_rate=m.bit_rate,
                                start_sample=m._t,status='pending'))
        return True

    def emit(self,bits):
        if not bits:return
        latency=TLEN-1 if self.engine._trellis is not None else 0
        self.hold.append((self.current_index-latency,list(bits)))
        self.flush(self.current_index-80)

    def deliver(self,bits):
        self.engine._out+=self.engine.v42_rx_bits(bits)

    def flush(self,before):
        while self.hold and self.hold[0][0]<before:self.deliver(self.hold.popleft()[1])

    def symbol(self,yr,index,metrics=None):
        self.current_index=index
        if self.mode=='control':self.control_symbol(yr);self.poll();return
        if self.mode=='cleardown':
            terminal=getattr(self.engine.tx,'rate_change_terminal_symbol',None)
            if (self.tx.cleared and terminal is not None and
                    self.engine.tx._n*self.engine.tx._g_sample>=terminal*self.engine.tx._g_symbol):
                self.events[-1].update(status='cleared');self.active=False
                self.engine._fail('V32 rate exchange cleared')
            else:self.poll()
            return
        if self.mode=='newdata':
            self.emit(self.data.feed(yr))
            self.rx_ready=self.data.physical_ready
            self.poll();return
        self.history.append((index,yr))
        event=self.detector.feed(yr,index)
        if event is not None:
            self.flush(event['start_symbol'])
            self.hold.clear()
            self.prior_seed=next((seed for k,seed in self.seed_history
                                  if k==event['start_symbol']-1),None)
            local=self.tx.word if self.active else rate_word(*self.engine._configured_rates)
            self.control=RateControlReader(self.role,local)
            self.mode='control'
            if not self.active:
                self.active=True;self.started=self.engine._t;self.rx_ready=False
                self.events.append(dict(origin='peer',previous_rate=self.engine.bit_rate,
                                        start_sample=self.engine._t,status='pending'))
            relative=event['gain'];self.gain*=relative
            self.events[-1]['preamble']=dict(start_symbol=event['start_symbol'],end_symbol=event['end_symbol'],
                                             residual=event['residual'])
            for k,y in self.history:
                if k>=event['end_symbol']:self.control_symbol(y/relative)
            return
        if self.data is None:self.engine._symbol(yr,metrics)
        else:self.emit(self.data.feed(yr))
        self.remember_seed(index)
        self.poll()

    def remember_seed(self,index):
        trellis=self.engine._trellis
        if trellis is None or trellis.rate not in TCM_BITS:return
        state=min(range(8),key=lambda i:trellis.sv[i]);path=trellis.paths[state]
        if not path:return
        k=path[-1]>>8
        previous=((k&1)<<1)|(k>>1)
        self.seed_history.append((index,(tuple(trellis.sv),previous)))

    def control_symbol(self,yr):
        try:
            events=self.control.feed(yr)
        except ValueError as exc:
            self.engine._fail('V32 invalid rate control: '+str(exc))
            return
        for event in events:
            if event['type']=='offer':
                if self.tx is None:
                    m=self.engine
                    self.tx=RateChangeTx(self.role,m._configured_rates,max(m._configured_rates),
                                         initiator=False,peer_word=event['word'])
                    m.tx.start_rate_change(self.tx)
                else:self.tx.accept_peer_offer(event['word'])
            elif event['type']=='e':
                if not event['rate']:
                    self.mode='cleardown';return
                self.data=RateDataReader(self.control,prior_seed=self.prior_seed)
                m=self.engine;m._trellis=self.data.trellis;m._desc=self.data.descrambler
                m._rate=event['rate'];m._prev_quad=self.data.previous
                m._dd_points=(POINTS if event['rate']==4800 else ZTAB4 if event['rate']==7200
                              else list(V32_TCM[event['rate']].values()))
                self.mode='newdata'

    def poll(self):
        if not self.active:return
        m=self.engine
        if m._t-self.started>10*m.fs:
            self.events[-1].update(status='failed',reason='V32 rate-change timeout')
            self.active=False;m._fail('V32 rate-change timeout');return
        data_symbol=getattr(m.tx,'rate_change_data_symbol',None)
        if (self.rx_ready and self.tx is not None and self.tx.tx_completed and data_symbol is not None
                and m.tx._n*m.tx._g_sample>=data_symbol*m.tx._g_symbol):
            m.bit_rate=self.data.rate;m.set_v42_rate(m.bit_rate)
            self.events[-1].update(status='complete',rate=m.bit_rate,completed_sample=m._t)
            self.events[-1].update(rx_convention=self.data.convention,
                                   rx_convention_scores=self.data.convention_scores)
            self.active=False;self.mode='data';self.tx=None
            self.detector=RatePreambleDetector(self.role);self.history.clear()
            self.seed_history.clear()
