"""Opt-in validity of equalizer feedback on known concealed input samples.

Coordinates are this receive core's raw PCM sample clock, not RTP timestamps.
The decoder and both NCOs still consume all samples/symbols. No payload truth.
"""
from collections import deque
import math


def feedback_support(eq):
    """Conservative inclusive raw-input support of the just-served window.

    81-tap centered FIR, 12-tap sinc interpolation, all T/2 equalizer taps.
    Include +/-0.05-sample direct timing derivatives. Trajectory derivatives
    use the original grid's source support, not its corrected virtual position.
    """
    R=eq.R
    offset=R._pcm_source_offset
    if eq._tj_k==eq.k and eq._g_traj:
        x=eq._tj_i-eq.MARGIN-eq.span
        first=R.lag+(R.LEAD+eq._g_k0+R.delay)*R.spl+eq._g_dt+x*(R.spl+eq._g_vel)/2
        last=first+(eq.L-1)*(R.spl+eq._g_vel)/2
    else:
        center=R.lag+(R.LEAD+eq.k+R.delay)*R.spl+eq.dt
        first=center-eq.span*R.spl/2
        last=first+(eq.L-1)*R.spl/2
    origin=getattr(R,'_bb_origin',0)
    left=max(origin,math.floor(min(first,last)-.05)-5)
    right=min(origin+len(R._z)-1,math.floor(max(first,last)+.05)+6)
    return offset+left-40,offset+right+40


class FeedbackValidity:
    MAX_RANGES=128

    def __init__(self,mode):
        if mode not in ('observe','hold','hold-dsp','hold-quality','hold-carrier','hold-timing'):raise ValueError('unknown V34_GAP_FEEDBACK mode')
        self.mode=mode
        # hold = both; hold-dsp holds only equalizer/carrier/timing feedback; hold-quality only makes the
        # data-quality monitor discard the tainted batch (DSP feedback runs as in observe). Lab split of 'hold'.
        self.hold_dsp=mode in ('hold','hold-dsp')
        self.hold_carrier=mode in ('hold','hold-dsp','hold-carrier')   # taps + carrier loop
        self.hold_timing=mode in ('hold','hold-dsp','hold-timing')     # sampling-window recovery
        self.discard_quality=mode in ('hold','hold-quality')
        self.discard_marks=0
        self.ranges=deque()
        self.registered_gaps=self.registered_samples=0
        self.tainted_symbols=self.held_symbols=0
        self.tap_updates=self.phase_updates=self.velocity_updates=0
        self.tap_delta_squared=0.
        self.first_tainted=self.last_tainted=None

    def add(self,start,count):
        if not isinstance(start,int) or not isinstance(count,int) or start<0 or count<=0:
            raise ValueError('positive raw PCM concealment range required')
        self.registered_gaps+=1;self.registered_samples+=count
        end=start+count
        if self.ranges and start<=self.ranges[-1][1]:
            old=self.ranges.pop();start=min(start,old[0]);end=max(end,old[1])
        self.ranges.append((start,end))
        # Overflow is conservative: retain all invalid samples inside an envelope,
        # never silently drop a still-live gap when the receiver is backlogged.
        if len(self.ranges)>self.MAX_RANGES:
            a=self.ranges.popleft();b=self.ranges.popleft()
            self.ranges.appendleft((a[0],max(a[1],b[1])))

    def prune(self,first_raw_sample):
        while self.ranges and self.ranges[0][1]<first_raw_sample-81:
            self.ranges.popleft()

    def tainted(self,eq):
        if not self.ranges or not hasattr(eq.R,'_pcm_source_offset'):return False
        left,right=feedback_support(eq)
        bad=any(start<=right and end>left for start,end in self.ranges)
        if bad:
            self.tainted_symbols+=1
            if self.first_tainted is None:self.first_tainted=dict(symbol=eq.k,raw_support=[left,right])
            self.last_tainted=dict(symbol=eq.k,raw_support=[left,right])
        return bad

    def record(self,before,eq,held):
        taps,phase,freq,velocity=before
        delta=float(sum(abs(eq.h-taps)**2))
        self.tap_delta_squared+=delta
        self.tap_updates+=int(delta>0)
        self.phase_updates+=int(abs(eq.phase-phase-freq)>1e-12 or eq.freq!=freq)
        self.velocity_updates+=int(eq.vel!=velocity)
        self.held_symbols+=int(held)
        self.discard_marks+=int(self.discard_quality)

    def report(self):
        return dict(mode=self.mode,coordinate_domain='receive-core raw PCM samples',
            registered_gaps=self.registered_gaps,registered_samples=self.registered_samples,
            retained_ranges=len(self.ranges),tainted_symbols=self.tainted_symbols,
            held_symbols=self.held_symbols,quality_discard_marks=self.discard_marks,
            hold_dsp=self.hold_dsp,hold_carrier=self.hold_carrier,hold_timing=self.hold_timing,discard_quality=self.discard_quality,tap_updates_on_taint=self.tap_updates,
            carrier_updates_on_taint=self.phase_updates,timing_velocity_updates_on_taint=self.velocity_updates,
            tap_delta_squared=self.tap_delta_squared,tap_delta_units='sum squared complex tap coefficient changes',
            first_tainted=self.first_tainted,last_tainted=self.last_tainted,
            decoder_samples_preserved=True,payload_truth_used=False)


def note_concealed(core,start,count):
    if getattr(core,'_rx_state',None) is None:return
    eq=getattr(getattr(core,'_p4rcv',None),'eq',None)
    guard=getattr(eq,'_transport_guard',None)
    if guard is not None:guard.add(int(start),int(count))


def held_symbols(eq):
    return (getattr(getattr(eq,'_input_guard',None),'held_symbols',0)
            +getattr(getattr(eq,'_transport_guard',None),'discard_marks',0))
