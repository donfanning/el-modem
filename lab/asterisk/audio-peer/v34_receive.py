"""Shared V.34 16-state receive calibration and B1 qualification.

Waveform symbols x' and mapper points u are separate domains. Calibration uses
only the first half of B1; held-out decisions never receive expected symbols.
"""
from dataclasses import asdict, dataclass, replace
import math
import os
import numpy as np
import v34_data as vd
import v34_datarx as vr
import v34_b1_reference
import v34_viterbi as vv
from v34_quality import DataQuality, observe as observe_quality


DUAL_CONVENTION='dual-convention'
CONTINUED_STATE=(0,1,0,0)


@dataclass(frozen=True)
class SearchPolicy:
    """Independent offline hypotheses; the default is the normative B1 check."""
    label: str = 'standard'
    rate_search: bool = False
    initial_states: tuple = ((0, 0, 0, 0),)
    boundary_offsets: tuple = (0,)

    def __post_init__(self):
        if not self.initial_states or any(len(s)!=4 or any(b not in (0,1) for b in s)
                                          for s in self.initial_states):
            raise ValueError('initial states must be four binary delay elements')
        max_offset = 128 if self.label.endswith('-diagnostic') else 0
        if not self.boundary_offsets or any(not isinstance(k,int) or abs(k)>max_offset for k in self.boundary_offsets):
            raise ValueError(f'boundary offsets exceed the {max_offset}-symbol policy limit')
        if self.label=='standard' and (self.rate_search or self.initial_states!=((0,0,0,0),) or self.boundary_offsets!=(0,)):
            raise ValueError('standard policy fixes negotiated rate, zero resets and E boundary')

    @classmethod
    def dual_convention(cls):
        """Live B1 policy: accept either differential-encoder start convention.

        10.1.3.1 zeroes the differential encoder before B1 (convention 'reset'). Some
        transmitters (observed in the Rockwell/Conexant-hardware family, see
        lab/V34_B1_INITIAL_ROTATION.md) instead continue it from the last E symbol, so Z0 is
        odd about half the time. With the trellis state zero, an odd Z0 equals initial state
        [0,1,0,0] up to a quarter turn that the complex B1 gain absorbs, so the second state is
        the 'continued' convention. Only these two states are searched, the rate and boundary stay
        normative, and the choice is ranked on first-half calibration alone (ties go to zero),
        never on B1 bit errors or held-out data.
        """
        return cls(DUAL_CONVENTION, False, ((0,0,0,0),(0,1,0,0)), (0,))

    @classmethod
    def diagnostic(cls, axis='combined', boundary_radius=8):
        if axis not in ('rate','reset','boundary','combined'):
            raise ValueError('unknown diagnostic axis')
        if not isinstance(boundary_radius, int) or not 0 <= boundary_radius <= 128:
            raise ValueError('diagnostic boundary radius must be an integer in 0..128')
        states=tuple(tuple((s>>k)&1 for k in range(4)) for s in range(16))
        return cls(axis+'-diagnostic', axis in ('rate','combined'),
                   states if axis in ('reset','combined') else ((0,0,0,0),),
                   tuple(range(-boundary_radius,boundary_radius+1)
                         ) if axis in ('boundary','combined') else (0,))


def projection(x, energy):
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError('nonlinear energy must be finite and positive')
    z = .3125 * np.abs(x)**2 / energy
    return x * (1 + z/6 + z*z/120)


def inverse_projection(y, energy):
    """Invert the strictly increasing radial polynomial, preserving phase."""
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError('nonlinear energy must be finite and positive')
    if np.ndim(y) == 0:
        r = abs(y)
        if not r:
            return 0j
        a = .3125/(6*energy); b = .3125**2/(120*energy**2)
        x = min(r,(r/a)**(1/3),(r/b)**.2)
        for _ in range(10):
            x -= (x + a*x**3 + b*x**5-r)/(1+3*a*x*x+5*b*x**4)
        return y*(x/r)
    r = np.abs(y)
    a = .3125/(6*energy); b = .3125**2/(120*energy**2)
    x = np.minimum(r,np.minimum((r/a)**(1/3),(r/b)**.2))
    for _ in range(10):
        x -= (x+a*x**3+b*x**5-r)/(1+3*a*x*x+5*b*x**4)
    return y*np.divide(x,r,out=np.ones_like(r),where=r!=0)


def calibrate(observed, linear, nonlinear, *, fit_indices=None):
    """Fit positive nonlinear energy and complex gain on the training half only."""
    n = len(linear)//2
    indices = np.arange(n) if fit_indices is None else np.asarray(fit_indices)
    if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(linear))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('calibration indices must be distinct symbols inside B1')
    x,y = np.asarray(linear)[indices],np.asarray(observed)[indices]
    def fit(log_energy):
        e = math.exp(log_energy)
        ref = projection(x,e) if nonlinear else x
        g = np.vdot(ref,y)/np.vdot(ref,ref)
        return float(np.mean(np.abs(y-g*ref)**2)), complex(g), e
    if not nonlinear:
        _,g,e = fit(0)
    else:
        # Wide physical search relative to the measured known-sequence energy.
        # Refuse boundary solutions rather than quietly assuming our TX estimate.
        center = math.log(float(np.mean(np.abs(x)**2)))
        lo,hi = center-12,center+12
        ratio = (math.sqrt(5)-1)/2
        a,b = hi-ratio*(hi-lo),lo+ratio*(hi-lo)
        fa,fb = fit(a)[0],fit(b)[0]
        for _ in range(90):
            if fa < fb:
                hi,b,fb = b,a,fa; a=hi-ratio*(hi-lo); fa=fit(a)[0]
            else:
                lo,a,fa = a,b,fb; b=lo+ratio*(hi-lo); fb=fit(b)[0]
        _,g,e = fit((lo+hi)/2)
        if abs(math.log(e)-center)>11.9:
            raise ValueError('B1 nonlinear normalization is not identifiable')
    if abs(g)<1e-12 or not np.isfinite(g):
        raise ValueError('invalid B1 gain')
    ref = projection(np.asarray(linear),e) if nonlinear else np.asarray(linear)
    nmse = float(np.mean(np.abs(observed[n:]/g-ref[n:])**2)/np.mean(np.abs(ref[n:])**2))
    return g,e,nmse,ref


def refine_taps(receiver, windows, observed, prior):
    """Production first-half refinement, explicitly tied to this receiver's state.

    Offline state comparisons must call this for every hypothesis, rather than
    reusing outputs that were fitted against a different B1 reference.
    """
    first=len(receiver.linear)//2
    A=np.asarray(windows)[:first]
    ridge=1e-4*float(np.linalg.norm(A,ord=2)**2)
    normal=A.conj().T@A+ridge*np.eye(A.shape[1])
    def fit(log_energy):
        energy=math.exp(log_energy)
        ref=projection(receiver.linear,energy) if receiver.profile.nonlinear else receiver.linear
        g=np.vdot(ref[:first],observed[:first])/np.vdot(ref[:first],ref[:first])
        target=g*ref[:first]
        taps=np.linalg.solve(normal,A.conj().T@target+ridge*prior)
        error=float(np.mean(abs(A@taps-target)**2)/np.mean(abs(target)**2))
        return error,taps
    if not receiver.profile.nonlinear:
        return fit(0)
    center=math.log(float(np.mean(abs(receiver.linear)**2)))
    lo,hi=center-4,center+4
    ratio=(math.sqrt(5)-1)/2
    a,b=hi-ratio*(hi-lo),lo+ratio*(hi-lo)
    fa,fb=fit(a)[0],fit(b)[0]
    for _ in range(48):
        if fa<fb:
            hi,b,fb=b,a,fa; a=hi-ratio*(hi-lo); fa=fit(a)[0]
        else:
            lo,a,fa=a,b,fb; b=lo+ratio*(hi-lo); fb=fit(b)[0]
    return fit((lo+hi)/2)


def calibration_scores(observed, linear, nonlinear):
    """Score all state/boundary hypotheses on their first halves, in parallel.

    Returns [state, boundary]. Broadcasting avoids thousands of Python golden
    searches when a recorded peer's rate and effective state are unknown.
    """
    n=linear.shape[1]//2
    x=np.asarray(linear[:,:n])[:,None,:]
    y=np.asarray(observed[:,:n])[None,:,:]
    center=np.log(np.mean(abs(x)**2,axis=-1))
    shape=(x.shape[0],y.shape[1])
    def fit(log_energy):
        energy=np.exp(log_energy)[...,None]
        z=.3125*abs(x)**2/energy
        ref=x*(1+z/6+z*z/120) if nonlinear else np.broadcast_to(x,shape+(n,))
        gain=np.sum(ref.conj()*y,axis=-1)/np.sum(abs(ref)**2,axis=-1)
        mse=np.mean(abs(y-gain[...,None]*ref)**2,axis=-1)
        return mse,gain
    if nonlinear:
        lo=np.broadcast_to(center-12,shape).copy();hi=np.broadcast_to(center+12,shape).copy()
        ratio=(math.sqrt(5)-1)/2
        a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo)
        fa,fb=fit(a)[0],fit(b)[0]
        for _ in range(90):
            left=fa<fb
            hi=np.where(left,b,hi);lo=np.where(left,lo,a)
            # Recompute both to keep the vectorized search easy to audit.
            a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo)
            fa,fb=fit(a)[0],fit(b)[0]
        optimum=(lo+hi)/2
        mse,gain=fit(optimum)
        invalid=abs(optimum-center)>11.9
    else:
        mse,gain=fit(np.zeros(shape));invalid=np.zeros(shape,dtype=bool)
    scores=mse/np.mean(abs(y)**2,axis=-1)
    scores[invalid | (abs(gain)<1e-12) | ~np.isfinite(gain) | ~np.isfinite(scores)]=np.inf
    return scores


class Receiver:
    def __init__(self, profile, remote_role, initial_state=(0,0,0,0)):
        if profile.trellis != 0:
            raise ValueError('unsupported receive trellis: only 16-state is qualified')
        self.profile, self.role = profile, remote_role
        self.initial_state=tuple(initial_state)
        self.h = vd.precoder_from_mp(profile.precoder)
        self.tx = vd.DataTx(profile.rate,profile.baud,expanded=profile.shaping,precoder=self.h)
        self.drx = vr.DataRx(profile.rate,profile.baud,expanded=profile.shaping,precoder=self.h)
        self.tx.conv.d=list(initial_state)
        self.drx.conv.d=list(initial_state)
        independent = v34_b1_reference.build(profile.rate, profile.baud, remote_role,
            profile.shaping, profile.precoder, initial_state)
        self.expected = independent["bits"]
        self.expected_mapper = np.asarray(independent["u"])
        self.linear = np.asarray(independent["x"])
        self.energy=1.; self.slicer_hist=[0j]*3
        self.diagnostics={'profile':asdict(profile),'qualified':False,'initial_trellis_state':list(initial_state),'standard_initialization':not any(initial_state)}

    def detector(self, start=False, trace=False):
        d=self.drx
        detector=vv.TrellisDetector(4*d.M*2**d.q,d.J,d.P,
            start_state=tuple(d.conv.d) if start else self.initial_state,
            m4_start=d.m4 if start else 0,precoder=self.h,w=self.tx.w,
            nonlinear=self.profile.nonlinear,energy=self.energy,
            erasure_hold=os.environ.get('V34_GAP_TRELLIS_HOLD')=='1',
            x_hist=d.x_hist if start else None,trace=trace)
        # Qualification diagnostics retain every interval. Ordinary native
        # streaming uses only bounded traceback; opt-in tracing still records.
        if start and not trace:
            detector.record_snapshots=False
        if start:
            detector.emit_states = os.environ.get('V34_SURVIVOR_STATE_SYNC') == '1'
            detector.emit_metrics = os.environ.get('V34_DATA_STATE_TRACE') == '1'
        return detector

    def qualify(self, observed, *, trace=False):
        g,self.energy,nmse,ref=calibrate(np.asarray(observed),self.linear,self.profile.nonlinear)
        self.b1_gain=g
        det=self.detector(trace=trace)
        self.b1_detector=det
        for a,b in np.asarray(observed).reshape(-1,2)/g:
            det.feed(a,b)
        det.flush()
        points=[x for pair in det.decided for x in pair]
        bits=[]
        for i in range(0,len(points),8):
            bits.extend(self.drx.decode_frame(points[i:i+8])[0])
        errors=sum(a!=b for a,b in zip(bits,self.expected)) + abs(len(bits)-len(self.expected))
        # Mapper disagreements must compare mapper u with mapper u, not
        # nonlinear/precoded waveform x'. Distances below use the detector's
        # actual feedback history and forward projection for both candidates.
        # A locally closer point alone does not prove a path-metric bug.
        _n=min(len(points),len(self.expected_mapper))
        _bad=[i for i in range(_n) if points[i]!=self.expected_mapper[i]]
        _obs=np.asarray(observed).reshape(-1)
        _ev=[]
        for i in _bad[:8]:
            rcv=complex(_obs[i]/g) if i < len(_obs) else 0j
            sn=det.snapshots[i//2]
            delta=complex(*sn['delta'+str(i%2)])
            true_candidate=self.expected_mapper[i]+delta
            decided_candidate=points[i]+delta
            if self.profile.nonlinear:
                true_candidate=complex(projection(true_candidate,self.energy))
                decided_candidate=complex(projection(decided_candidate,self.energy))
            _ev.append(dict(symbol=i, received=[rcv.real,rcv.imag],
                true=[self.expected_mapper[i].real,self.expected_mapper[i].imag],
                decided=[points[i].real,points[i].imag],
                true_candidate_waveform=[true_candidate.real,true_candidate.imag],
                decided_candidate_waveform=[decided_candidate.real,decided_candidate.imag],
                d_true=float(abs(rcv-true_candidate)), d_decided=float(abs(rcv-decided_candidate)),
                d_expected_waveform=float(abs(rcv-ref[i]))))
        # At the first divergence, what did the trellis survive to, and were the true label/parity
        # admissible? snapshots[i//2] is the interval that produced symbol i.
        first_div = None
        if _bad and hasattr(det, "snapshots"):
            i = _bad[0]
            sn = det.snapshots[i // 2] if i // 2 < len(det.snapshots) else None
            if sn is not None:
                first_div = dict(symbol=i, interval=i // 2, snapshot=sn)
        # Positions of the individual bit errors, so a failure can be read by shape as well as by
        # count. A count alone cannot separate a decision path that is wrong from the first symbol
        # (rotation, mapping convention, differential offset) from one that degrades as the burst
        # proceeds (precoder inverse truncation, nonlinear calibration drift), which is the
        # difference between correcting the frontend and correcting a filter.
        self.diagnostics.update(qualified=errors==0,bit_errors=errors,bit_count=len(self.expected),
            held_out_nmse=nmse,energy=self.energy,gain_real=g.real,gain_imag=g.imag,
            error_bit_indices=[i for i in range(min(len(bits),len(self.expected))) if bits[i]!=self.expected[i]],
            length_mismatch=abs(len(bits)-len(self.expected)),
            first_decoded_point_deltas=_ev,
            point_distance_domain='forward waveform with detector feedback history',
            mapper_symbol_errors=len(_bad),first_divergence=first_div)
        self.bits=bits
        self.slicer_hist=list(self.drx.x_hist)
        self.vit=self.detector(start=True)
        return g,ref

    def decide(self, y):
        """Forward waveform target for NLMS/PLL; separate from delayed trellis decisions."""
        x=inverse_projection(y,self.energy) if self.profile.nonlinear else y
        if any(self.h):
            p=vd._round_complex(sum(x*h for x,h in zip(self.slicer_hist,self.h)),2**-7)
            c=vd._round_complex(p,2**self.tx.w)
        else:p=c=0j
        u=self.drx.nearest_point(x+p-c)
        target=u+c-p
        self.slicer_hist=[target]+self.slicer_hist[:2]
        return complex(projection(target,self.energy)) if self.profile.nonlinear else target


def live_policy():
    """Default for live calls: both differential-encoder conventions. V34_B1_CONVENTIONS=standard
    restores the zero-reset-only policy."""
    if os.environ.get('V34_B1_CONVENTIONS','dual')=='standard':
        return SearchPolicy()
    return SearchPolicy.dual_convention()


def initialize(engine, eq, remote_role, baud, policy=None):
    """Acquire B1 once; return False until enough samples exist."""
    if getattr(engine,"rx_qualification",{}).get("reason") or getattr(engine,"failed_reason",None):
        return False
    profile=engine.negotiated.rx
    if profile.trellis != 0:
        reason='unsupported receive trellis: only 16-state is qualified'
        engine.rx_qualification={'profile':asdict(profile),'qualified':False,'reason':reason}
        engine.failed_reason=reason
        return False
    # Legacy simulated replay remains compatible; passive observation passes an
    # explicit policy and never uses this switch to select its qualification.
    recorded=getattr(engine,'_recorded_receive',False)
    policy=policy or getattr(engine,'_receive_search_policy',None) or (SearchPolicy.diagnostic() if recorded else live_policy())
    profiles=[profile]
    if policy.rate_search:
        mask=engine._local_mp['mask'] & (getattr(engine,'mp_ans',None) or getattr(engine,'mp_call',None))['mask']
        profiles=[replace(profile,rate=rate) for rate in range(2400,profile.rate+1,2400)
                  if mask & (1 << (rate//2400-1)) and __import__('v34_negotiation').supported(rate,profile.baud,profile.shaping)]
    margin=max(abs(k) for k in policy.boundary_offsets)
    n=max(8*vd.frame_params(p.rate,p.baud)["P"] for p in profiles)
    if eq.available()<engine._rx_b1+n+margin+8:
        return False
    dsp=getattr(engine,'receive_dsp',None)
    if dsp is not None and hasattr(dsp,'before_b1') and not getattr(engine, '_connected_rate_b1', False):
        dsp.before_b1(eq, engine._p4rcv.k_trn)
    snapshot=eq.__dict__.copy()
    timing_start=eq.timing_snapshot()
    eq.adapt=False; eq.k=engine._rx_b1-margin; eq.phase-=margin*eq.freq
    samples,windows,centers=[],[],[]
    for _ in range(n+2*margin):
        centers.append(eq.R.lag+(eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl+eq.dt)
        windows.append(eq.window(eq.k).copy()*np.exp(-1j*eq.phase))
        samples.append(eq.step(decide=lambda y:y))
    observed_all=np.array(samples)
    windows=np.asarray(windows)
    source_offset=getattr(engine,'_p4_x0',None)
    if source_offset is None:
        source_offset=engine._p3['x0']
    if getattr(engine,'_receive_trace',False):
        bb_origin=getattr(eq.R, '_bb_origin', 0)
        pcm_origin=getattr(engine, '_rx_origin', 0)
        lo=max(bb_origin,int(np.floor(min(centers)-eq.span*eq.R.spl/2))-48)
        hi=min(bb_origin+len(eq.R._z),int(np.ceil(max(centers)+eq.span*eq.R.spl/2))+49)
        engine._b1_inputs=dict(windows=windows,observed_before=observed_all,
            centers=np.array(centers)+source_offset,taps_before=snapshot['h'].copy(),
            raw=engine._rx[source_offset+lo-pcm_origin:source_offset+hi-pcm_origin].copy(),
            baseband=eq.R._z[lo-bb_origin:hi-bb_origin].copy(),baseband_offset=lo,
            source_offset=source_offset,frequency=eq.R.fc,baud=eq.R.baud,
            lag=eq.R.lag,delay=eq.R.delay,dt=snapshot['dt'],phase=snapshot['phase'],
            frequency_loop=snapshot['freq'],b1_symbol=engine._rx_b1,margin=margin)
        engine._b1_inputs['timing_start']=np.array([timing_start['sample_position'],
            timing_start['samples_per_symbol'],timing_start['timing_velocity'],timing_start['carrier_phase']])
    candidates=[]
    states=policy.initial_states
    candidate_scores=[]
    for candidate_profile in profiles:
        linear=np.asarray([v34_b1_reference.build(candidate_profile.rate,
            candidate_profile.baud,remote_role,candidate_profile.shaping,
            candidate_profile.precoder,state)['x'] for state in states])
        length=linear.shape[1]
        observations=np.asarray([observed_all[margin+offset:margin+offset+length]
            for offset in policy.boundary_offsets])
        scores=calibration_scores(observations,linear,candidate_profile.nonlinear)
        state_index,offset_index=(int(v) for v in np.unravel_index(np.argmin(scores),scores.shape))
        shift=margin+policy.boundary_offsets[offset_index]
        candidates.append((float(scores[state_index,offset_index]),candidate_profile,
            states[state_index],shift,observations[offset_index]))
        candidate_scores.append(dict(rate=candidate_profile.rate,
            scores=[[float(v) if np.isfinite(v) else None for v in row] for row in scores]))
    search_evidence=dict(search_policy=asdict(policy),normative_rate=profile.rate,
        rate_rule='V.34 11.4.1.1.3 / 11.4.1.2.3: highest mutually enabled rate',
        calibration_candidates=candidate_scores,selection_uses_held_out=False)
    if not candidates or not np.isfinite(min(c[0] for c in candidates)):
        engine.rx_qualification={'profile':asdict(profile),'qualified':False,'reason':'no identifiable B1 profile',**search_evidence}
        engine._b1_linear=np.asarray(v34_b1_reference.build(profiles[-1].rate,
            baud,remote_role,profile.shaping,profile.precoder)['x'])
        engine._b1_observed=observed_all[margin:margin+len(engine._b1_linear)]
        engine.failed_reason='no identifiable B1 profile'
        return False
    # Every allowed rate gets the same state/boundary search, ranked only on
    # calibration samples. Expected held-out bits never select a candidate.
    score,selected_profile,state,shift,observed=min(candidates,key=lambda q:q[0])
    if (os.environ.get('V34_GAP_TRELLIS_ERASURE')=='1'
            and any(selected_profile.precoder)):
        reason='V34 gap trellis erasure experiment requires a zero precoder'
        engine.rx_qualification={'profile':asdict(selected_profile),'qualified':False,
            'reason':reason,'search_evidence':search_evidence}
        engine.failed_reason=reason
        return False
    receiver=Receiver(selected_profile,remote_role,state)
    receiver.diagnostics.update(search_evidence)
    receiver.diagnostics.update(differential_convention='continued' if any(state) else 'reset',
        convention_policy=policy.label,boundary_offset=shift-margin,calibration_nmse=score,
        rate_candidates=len(profiles),state_candidates=len(states),boundary_candidates=len(policy.boundary_offsets))
    refined_h=None
    if score>1e-4:
        # Train taps on the calibration half only. Re-fit radial energy jointly
        # with the linear channel; a channel mismatch must not masquerade as
        # nonlinear expansion. Held-out windows are never used in the fit.
        W=windows[shift:shift+len(receiver.linear)]
        refined_score,refined_h=refine_taps(receiver,W,observed,snapshot['h'])
        observed=W@refined_h
        receiver.diagnostics['tap_calibration_nmse']=refined_score
    engine._b1_observed=observed
    engine._b1_linear=receiver.linear
    if getattr(engine,'_receive_trace',False):
        engine._b1_inputs.update(observed_after=observed,linear=receiver.linear,
            chosen_shift=shift,taps_after=refined_h if refined_h is not None else snapshot['h'])
    try:
        g,ref=receiver.qualify(observed)
        dsp=getattr(engine,'receive_dsp',None)
        if dsp is not None:dsp.qualify(receiver,observed,g)
    except ValueError as exc:
        engine.rx_qualification={**receiver.diagnostics,'reason':str(exc)}
        engine.failed_reason=str(exc)
        return False
    # Restore and advance exactly to the selected B1/data boundary.
    eq.__dict__.update(snapshot)
    eq.adapt=False; eq.k=engine._rx_b1-margin; eq.phase-=margin*eq.freq
    for _ in range(shift+len(receiver.linear)):
        eq.step(decide=lambda y:y)
    if policy.rate_search and receiver.diagnostics['qualified']:
        engine.rx_rate=receiver.profile.rate
        engine.negotiated=replace(engine.negotiated,rx=receiver.profile)
    engine.rx_qualification=receiver.diagnostics
    engine.rx_qualification['standard_qualification']=bool(receiver.diagnostics['qualified']
        and policy.label in ('standard',DUAL_CONVENTION) and selected_profile.rate==profile.rate
        and not any(state) and shift==margin)
    engine._dbg_b1=(observed.copy(),ref.copy(),g)
    engine._ev(f"B1: errors={receiver.diagnostics['bit_errors']} held-out NMSE={receiver.diagnostics['held_out_nmse']:.6g}")
    if not receiver.diagnostics['qualified']:
        engine.failed_reason='B1 receive qualification failed'
        return False
    eq.h=(refined_h if refined_h is not None else eq.h)/g
    eq.begin_data_tracking(precoded=bool(any(receiver.h)))
    engine._decide=receiver.decide
    eq.adapt=True
    receiver.diagnostics['timing_at_data_start']=eq.timing_snapshot()
    engine._rx_state={'drx':receiver.drx,'receiver':receiver,'vit':receiver.vit,
        'hist':receiver.bits[-23:],'bits':[],'ys':[],'ys_tainted':[],
        'pend':[],'frames':0,'ones':[0,0]}
    if os.environ.get('V34_QUALITY_MONITOR') == '1' or os.environ.get('V34_AUTO_RATE') == '1':
        from v34_transport import held_symbols
        engine._rx_state['quality'] = DataQuality(receiver.vit, receiver.profile, baud,
            held_symbols=held_symbols(eq))
    return True
