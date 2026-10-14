"""V.32bis section-8 transmit exchange and decoded rate-word validation.

Symbol-domain component, not an enabled live rate-change controller. Its owner
must detect the received preamble, preserve the frontend, decode E, wait 24 RX
symbols and only then report completion. No full retrain is substituted here.
"""
from v32_caller import (A, C, RATE_BITS, RATE_WORD_BASE, Scrambler,
                        Differential4800Encoder, TrellisEncoder,
                        Trellis7200Encoder, TCM_BITS, GPC, GPA)
from collections import deque
import math
import numpy as np

CLEAR_RATE_WORD = RATE_WORD_BASE & ~0x0800


def ending_word(rate):
    return 0xf000 | (RATE_WORD_BASE | RATE_BITS[rate] if rate else CLEAR_RATE_WORD)


class RatePreambleDetector:
    """Bounded symbol-domain candidate detector, requiring later R validation.

    The 64T short reversal is scored in the existing POINTS domain, with an
    eight-symbol preceding guard against sustained full-retrain AA/AC. A local
    score minimum is reported after it passes, retaining at most eight symbols
    of delay. No rate, payload or expected bits enter this detector.
    """
    def __init__(self, receiver_role, maximum_residual=.05):
        if receiver_role not in ('call','answer') or not 0<maximum_residual<1:
            raise ValueError('valid receiver role and residual bound required')
        self.pattern=np.asarray([A,C]*28+[C,A]*4 if receiver_role=='call' else [A]*56+[C]*8)
        self.reference_energy=float(np.vdot(self.pattern,self.pattern).real)
        self.maximum_residual=maximum_residual
        self.history=deque(maxlen=72);self.last_index=None
        self.candidate=None;self.delay=0;self.cooldown=0

    def feed(self, symbol, index):
        if not isinstance(index,int) or (self.last_index is not None and index!=self.last_index+1):
            raise ValueError('contiguous integer symbol indices required')
        if not math.isfinite(symbol.real) or not math.isfinite(symbol.imag):
            raise ValueError('finite symbols required')
        self.last_index=index;self.history.append(symbol)
        if self.cooldown:
            self.cooldown-=1;return None
        if len(self.history)<72:return None
        data=np.asarray(self.history);y=data[-64:]
        energy=float(np.vdot(y,y).real)
        gain=np.vdot(self.pattern,y)/self.reference_energy
        residual=float(np.sum(abs(y-gain*self.pattern)**2)/max(energy,1e-20))
        preceding=data[:8]
        extended=float(np.sum(abs(preceding-gain*self.pattern[:8])**2)
                       /max(float(np.vdot(preceding,preceding).real),1e-20))
        valid=(residual<=self.maximum_residual and .5<=abs(gain)<=1.5
               and abs(math.atan2(gain.imag,gain.real))<=math.pi/4
               and extended>self.maximum_residual)
        if valid:
            event=dict(start_symbol=index-63,end_symbol=index+1,gain=complex(gain),
                       residual=residual,preceding_residual=extended)
            if self.candidate is None:
                self.candidate=event;self.delay=0
            elif residual<self.candidate['residual']:
                self.candidate=event
        if self.candidate is not None:
            self.delay+=1
            if not valid or self.delay>=8:
                event=self.candidate;self.candidate=None;self.cooldown=64
                event['reported_symbol']=index
                return event
        return None


def parse_offer(word):
    """Validate section-5.3 R4/R5; ignore B13/B14, reject legacy capability."""
    if not isinstance(word, int) or not 0 <= word <= 0xffff:
        raise ValueError('16-bit rate word required')
    if word & 0xf111 != 0x0111:
        raise ValueError('invalid rate-word synchronization bits')
    # Table 5 note 3: B4-B6/B9/B10/B12 all zero means cleardown,
    # which must be recognized before rejecting missing capability bits.
    if word & 0x0880 != 0x0880 and word & 0x0e68:
        raise ValueError('peer does not advertise V32bis rate renegotiation')
    return word & ~0x0006


def offered_rates(word):
    word=parse_offer(word)
    return tuple(rate for rate in sorted(RATE_BITS) if word & RATE_BITS[rate])


class RateWordReader:
    """Locate two consecutive equal decoded 16-bit words at any bit offset.

    Input must already be differential-decoded and descrambled from the end of
    a validated preamble. This class does not invent a symbol clock or reference.
    """
    def __init__(self):
        self.shift=0;self.count=0;self.last=None
        self.last_end=None

    def feed(self, bits):
        events=[]
        for bit in bits:
            if bit not in (0,1):raise ValueError('binary decoded bits required')
            self.shift=((self.shift<<1)|bit)&0xffffffff;self.count+=1
            if self.count<32:continue
            try:
                a=parse_offer(self.shift>>16);b=parse_offer(self.shift&0xffff)
            except ValueError:continue
            if a==b and a!=self.last:
                self.last=a;self.last_end=self.count;events.append(a)
        return events


class RateControlReader:
    """Decode R4/R5 and E after an externally validated incoming preamble.

    Symbols are equalized in the existing POINTS coordinate domain. This is
    not a preamble detector or a data decoder. The owner must switch to the new
    data constellation on the E event and enforce the separate 24T RX clamp.
    """
    def __init__(self, receiver_role, local_word):
        if receiver_role not in ('call','answer'):
            raise ValueError('call/answer role required')
        self.local_word=parse_offer(local_word)
        self.previous=0 if receiver_role=='call' else 3
        self.descrambler=Scrambler(GPA if receiver_role=='call' else GPC)
        self.reader=RateWordReader();self.peer_word=None
        self.selected_rate=None;self.shift=0;self.expected_e=None;self.seen_e=False

    def feed(self, symbol):
        from v32_rx import slice4
        from v32_caller import NR_DIFF
        if self.seen_e:
            raise RuntimeError('E decoded; owner must change RX mode before more symbols')
        point=slice4(symbol)
        dibit=NR_DIFF[(self.previous<<2)|point];self.previous=point
        events=[]
        for scrambled in (dibit>>1,dibit&1):
            bit=self.descrambler.rev(scrambled)
            self.shift=((self.shift<<1)|bit)&0xffff
            offers=self.reader.feed([bit])
            for word in offers:
                if self.peer_word is not None and word!=self.peer_word:
                    raise ValueError('received offer changed during an active exchange')
                self.peer_word=word
                common=set(offered_rates(word)).intersection(offered_rates(self.local_word))
                self.selected_rate=max(common) if common else 0
                self.expected_e=ending_word(self.selected_rate)
                events.append(dict(type='offer',word=word,rate=self.selected_rate))
        if (self.expected_e is not None and self.shift & ~0x0006 == self.expected_e
                and (self.reader.count-self.reader.last_end)%16==0):
            self.seen_e=True
            events.append(dict(type='e',word=self.expected_e,rate=self.selected_rate))
        return events


class RateDataReader:
    """New-rate bit decoder retaining R/E descrambler history and 24T clamp.

    Physical symbol readiness and traceback-delayed idle-group removal are
    independent. Only decoded payload bits are returned, never the 24T ones.
    """
    def __init__(self, control, prior_seed=None):
        from v32_rx import ViterbiStream
        if not control.seen_e or not control.selected_rate:
            raise ValueError('matching non-cleardown E required before data mode')
        self.rate=control.selected_rate;self.descrambler=control.descrambler
        self.previous=control.previous
        self.trellis=None if self.rate==4800 else ViterbiStream(self.rate)
        self.nbits=self.rate//2400;self.symbols_seen=0
        self.remaining_idle_groups=24;self.idle_errors=0
        self.convention='reset';self.convention_scores={}
        self._prior_seed=None;self._idle_observations=[]
        if prior_seed is not None and self.rate in TCM_BITS:
            scores,previous=prior_seed
            if (len(scores)!=8 or any(not math.isfinite(v) or v<0 for v in scores)
                    or not isinstance(previous,int) or not 0<=previous<4):
                raise ValueError('finite eight-state metrics and differential value required')
            self._prior_seed=(tuple(scores),previous);self.convention='pending'

    @property
    def physical_ready(self):
        return self.symbols_seen>=24 and self.convention!='pending'

    def _choose_convention(self):
        """Compare known idle models in one observed-power coordinate domain.

        Continue only on a strong, bounded improvement. Ambiguous or poor fits
        retain canonical reset; no gain fit, payload or traceback decisions
        choose the convention. These engineering bounds are explicit and
        remain confined to the experimental live rate-change controller.
        """
        scores,previous=self._prior_seed
        state=min(range(8),key=lambda i:scores[i])
        observed=np.asarray(self._idle_observations)
        energy=max(float(np.vdot(observed,observed).real),1e-20)
        losses={}
        for name,t,y in [('reset',0,0),('continue',state,previous)]:
            scr=Scrambler(self.descrambler.tap);scr.wd=self.descrambler.wd
            encoder=TrellisEncoder(self.rate);encoder.t=t;encoder.y=y
            reference=[]
            for _ in range(24):
                group=0
                for _ in range(self.nbits):group=2*group+scr.fwd(1)
                reference.append(encoder.encode(group))
            losses[name]=float(np.sum(abs(observed-np.asarray(reference))**2)/energy)
        reset,continued=losses['reset'],losses['continue']
        use_continued=(continued<=.05 and reset>=2*continued and reset-continued>=.01)
        self.convention='continue' if use_continued else 'reset'
        self.convention_scores=dict(normalizer='same observed idle energy',symbols=24,
            reset=reset,continued=continued,continued_state=state,continued_previous=previous,
            maximum_continued_loss=.05,minimum_loss_ratio=2.,minimum_loss_advantage=.01)
        if use_continued:
            self.trellis.sv=list(scores);self.trellis.prev=previous

    def feed(self, symbol):
        if not math.isfinite(symbol.real) or not math.isfinite(symbol.imag):
            raise ValueError('finite new-rate symbols required')
        self.symbols_seen+=1
        if self.convention=='pending':
            self._idle_observations.append(symbol)
            if self.symbols_seen<24:return []
            self._choose_convention()
            for y in self._idle_observations:
                if self._decode(y):raise RuntimeError('idle replay released payload')
            self._idle_observations.clear()
            return []
        return self._decode(symbol)

    def _decode(self, symbol):
        from v32_rx import slice4
        from v32_caller import NR_DIFF
        if self.trellis is None:
            point=slice4(symbol);group=NR_DIFF[(self.previous<<2)|point];self.previous=point
        else:
            group=self.trellis.push(symbol)
            if group is None:return []
        bits=[self.descrambler.rev((group>>s)&1) for s in range(self.nbits-1,-1,-1)]
        if self.remaining_idle_groups:
            self.remaining_idle_groups-=1;self.idle_errors+=sum(bit!=1 for bit in bits)
            return []
        return bits


class RateChangeTx:
    """One finite exchange, retaining application queues via a lazy bit source.

    A responder is constructed only after a validated R4. An initiator waits
    for validated R5; both send at least 64T of their own offer. Completion here
    means symbol generation only, not PCM transmission, remote E receipt or
    end-to-end rate completion; pulse shapers generate symbols ahead of time.
    """
    def __init__(self, role, enabled_rates, desired_rate, *, initiator=True,
                 peer_word=None, timeout_symbols=24000):
        if role not in ('call','answer'):raise ValueError('call/answer role required')
        enabled=tuple(sorted(set(enabled_rates)))
        if not enabled or any(r not in RATE_BITS for r in enabled) or desired_rate not in enabled:
            raise ValueError('desired rate must be in the supported enabled profile')
        if not isinstance(timeout_symbols,int) or timeout_symbols<64:
            raise ValueError('timeout must allow at least 64 rate symbols')
        if not initiator and peer_word is None:
            raise ValueError('responder requires a validated R4')
        self.role=role;self.initiator=bool(initiator);self.enabled=enabled
        self.word=RATE_WORD_BASE
        for r in enabled:
            if r<=desired_rate:self.word|=RATE_BITS[r]
        self.peer_word=None;self.selected_rate=None
        self.timeout_symbols=timeout_symbols;self.rate_symbols=0
        self.state='pending';self.tx_completed=False;self.cleared=False
        self.scrambler=Scrambler(GPC if role=='call' else GPA)
        self.encoder=Differential4800Encoder();self._started=False
        if peer_word is not None:self.accept_peer_offer(peer_word)

    def accept_peer_offer(self, word):
        word=parse_offer(word)
        if self.peer_word is not None and word!=self.peer_word:
            raise ValueError('peer offer changed during an active exchange')
        if self.tx_completed or self.cleared:
            raise ValueError('exchange already terminal')
        self.peer_word=word
        common=set(offered_rates(self.word)).intersection(offered_rates(word))
        self.selected_rate=max(common) if common else 0

    def _control_word(self, word):
        for shift in range(14,-1,-2):
            hi=self.scrambler.fwd((word>>(shift+1))&1)
            lo=self.scrambler.fwd((word>>shift)&1)
            yield self.encoder.encode((hi<<1)|lo)

    def symbols(self, bit_source):
        if self._started:raise RuntimeError('exchange stream can only start once')
        self._started=True;self.state='preamble'
        self.scrambler.reset()
        if self.role=='call':
            for _ in range(56):yield A
            for _ in range(8):yield C
            self.encoder.q=3  # Final CC symbol: C.
        else:
            for _ in range(28):yield A;yield C
            for _ in range(4):yield C;yield A
            self.encoder.q=0  # Final CA symbol: A.
        self.state='rate'
        while self.rate_symbols<64 or self.selected_rate is None:
            if self.rate_symbols+8>self.timeout_symbols:
                self.state='failed'
                raise TimeoutError('no valid R5 within transmit exchange deadline')
            for symbol in self._control_word(self.word):
                self.rate_symbols+=1
                yield symbol
        self.state='e'
        e=ending_word(self.selected_rate)
        if not self.selected_rate:
            # No common rate/cleardown: at least 64T of E, no payload consumed.
            for _ in range(8):yield from self._control_word(e)
            self.state='cleared';self.cleared=True
            return
        yield from self._control_word(e)
        if self.selected_rate in TCM_BITS:
            data_encoder=TrellisEncoder(self.selected_rate);nbits=TCM_BITS[self.selected_rate]
        elif self.selected_rate==7200:
            data_encoder=Trellis7200Encoder(0);nbits=3
        else:
            data_encoder=self.encoder;nbits=2
        self.state='ones'
        for _ in range(24):
            bits=0
            for _ in range(nbits):bits=(bits<<1)|self.scrambler.fwd(1)
            yield data_encoder.encode(bits)
        self.state='data';self.tx_completed=True
        while True:
            bits=0
            for _ in range(nbits):
                bit=bit_source()
                if bit not in (0,1):raise ValueError('binary application bit required')
                bits=(bits<<1)|self.scrambler.fwd(bit)
            yield data_encoder.encode(bits)
