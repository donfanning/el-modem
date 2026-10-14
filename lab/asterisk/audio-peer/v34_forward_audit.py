"""Offline raw-PCM pulse identification. Never used by the live receiver.

The pulse absorbs fixed delay, phase, transmit shaping and channel response.
Clock slope and carrier error are fitted to TRN alone. B1 fits only complex
gain and, if enabled, nonlinear energy. Unknown neighbouring symbols are
excluded rather than silently replaced by zero in scored windows.
"""
from dataclasses import dataclass
import numpy as np

FS = 8000
SPAN = 8
KNOTS = 65


def basis(samples, symbols, origin, spl, span=SPAN, resolution=4):
    """Piecewise linear complex pulse at quarter-symbol knots, +/-8T."""
    samples = np.asarray(samples)
    symbols = np.asarray(symbols, complex)
    coordinate = (samples-origin)/spl
    knots=2*span*resolution+1
    matrix = np.zeros((len(samples), knots), complex)
    rows = np.arange(len(samples))
    for offset in range(-span, span+2):
        k = np.floor(coordinate).astype(int)+offset
        t = (coordinate-k+span)*resolution
        j = np.floor(t).astype(int)
        fraction = t-j
        valid = (k >= 0) & (k < len(symbols)) & (j >= 0) & (j < knots-1)
        r = rows[valid]; index = j[valid]; value = symbols[k[valid]]
        matrix[r,index] += value*(1-fraction[valid])
        matrix[r,index+1] += value*fraction[valid]
    return matrix


def nmse(observed, predicted):
    return float(np.mean((observed-predicted)**2)/max(np.mean(observed**2),1e-30))


@dataclass
class PulseModel:
    origin: float
    spl: float
    carrier: float
    pulse: np.ndarray
    training_nmse: float
    held_out_training_nmse: float
    carrier_offset_hz: float
    clock_ppm: float
    span: int = SPAN
    resolution: int = 4
    selection: dict = None

    def envelope(self, samples, symbols, symbol_origin):
        return basis(samples,symbols,symbol_origin,self.spl,self.span,self.resolution) @ self.pulse * np.exp(
            2j*np.pi*self.carrier*(np.asarray(samples)-self.origin)/FS)


def fit_pulse(pcm, symbols, origin, nominal_spl, carrier, ridge=1e-5,select_model=True,extended=False):
    """Fit on 60% of TRN, select on 20%, validate on the final 20%.

    A bounded coordinate search estimates frequency (+/-10 Hz) and clock
    slope (+/-500 ppm). Validation samples never select those parameters.
    """
    pcm = np.asarray(pcm,float); symbols=np.asarray(symbols,complex)
    guard=(32 if extended else 16) if select_model else SPAN
    stop=len(symbols)-guard-2
    if stop < 200: raise ValueError('insufficient phase-four TRN')
    split=int(stop*.6);split2=int(stop*.8)
    train=np.arange(int(np.ceil(origin+(guard+2)*nominal_spl)),
                    int(np.floor(origin+(split-guard)*nominal_spl)))
    selection=np.arange(int(np.ceil(origin+(split+guard)*nominal_spl)),
                       int(np.floor(origin+(split2-guard)*nominal_spl)))
    valid=np.arange(int(np.ceil(origin+(split2+guard)*nominal_spl)),
                    int(np.floor(origin+stop*nominal_spl)))
    if len(valid)<100 or len(selection)<100:raise ValueError('insufficient held-out TRN')
    if train[0]<0 or valid[-1]>=len(pcm):raise ValueError('TRN outside PCM')
    # Cap cost without moving either signal on its recording clock.
    train=train[::max(1,len(train)//2200)]
    def solve(hz,ppm):
        spl=nominal_spl*(1+ppm*1e-6)
        b=basis(train,symbols,origin,spl)*np.exp(2j*np.pi*(carrier+hz)*(train-origin)/FS)[:,None]
        design=np.column_stack((b.real,-b.imag))
        gram=design.T@design
        penalty=ridge*np.trace(gram)/len(gram)
        coeff=np.linalg.solve(gram+penalty*np.eye(len(gram)),design.T@pcm[train])
        return nmse(pcm[train],design@coeff),coeff[:KNOTS]+1j*coeff[KNOTS:]
    hz=ppm=0.; best,pulse=solve(hz,ppm)
    for dh,dp in [(2.,200.),(.5,50.),(.1,10.),(.02,2.)]:
        for _ in range(12):
            candidates=[]
            for a,b in [(hz-dh,ppm),(hz+dh,ppm),(hz,ppm-dp),(hz,ppm+dp)]:
                if abs(a)<=10 and abs(b)<=500:
                    score,h=solve(a,b);candidates.append((score,a,b,h))
            score,a,b,h=min(candidates,key=lambda x:x[0])
            if score>=best:break
            best,hz,ppm,pulse=score,a,b,h
    model=PulseModel(origin,nominal_spl*(1+ppm*1e-6),carrier+hz,pulse,best,0.,hz,ppm)
    candidates=[]
    spans=(8,16,24,32) if extended else (8,12,16)
    for span,resolution,regularization in ([(s,r,v) for s in spans for r in (4,8)
            for v in (1e-5,1e-4,1e-3)] if select_model else [(SPAN,4,ridge)]):
        def fitted(a,b):
            spl=nominal_spl*(1+b*1e-6)
            field=basis(train,symbols,origin,spl,span,resolution)*np.exp(
                2j*np.pi*(carrier+a)*(train-origin)/FS)[:,None]
            design=np.column_stack((field.real,-field.imag));gram=design.T@design
            penalty=regularization*np.trace(gram)/len(gram)
            normal=gram+penalty*np.eye(len(gram))
            coef=np.linalg.solve(normal,design.T@pcm[train])
            knots=2*span*resolution+1;p=coef[:knots]+1j*coef[knots:]
            m=PulseModel(origin,spl,carrier+a,p,nmse(pcm[train],design@coef),0,a,b,span,resolution)
            return m,design,normal
        m,design,normal=fitted(hz,ppm)
        steps=0
        if extended:
            # Eliminate pulse coefficients from the clock/carrier derivatives.
            # Training PCM alone enters this variable-projection search.
            for _ in range(6):
                prediction=m.envelope(train,symbols,origin)
                dh=(2j*np.pi*(train-origin)/FS*prediction).real
                moved=basis(train,symbols,origin,m.spl*(1+2e-6),span,resolution)@m.pulse
                moved*=np.exp(2j*np.pi*m.carrier*(train-origin)/FS)
                dp=(moved.real-prediction.real)/2
                jac=np.column_stack((dh,dp))
                jac-=design@np.linalg.solve(normal,design.T@jac)
                delta=np.linalg.lstsq(jac,pcm[train]-prediction.real,rcond=1e-8)[0]
                delta=np.clip(delta,[-.5,-100],[.5,100])
                accepted=False
                for fraction in (1.,.5,.25):
                    a=float(np.clip(m.carrier_offset_hz+fraction*delta[0],-10,10))
                    b=float(np.clip(m.clock_ppm+fraction*delta[1],-500,500))
                    other,d,g=fitted(a,b)
                    if other.training_nmse < m.training_nmse*(1-1e-9):
                        m,design,normal=other,d,g;steps+=1;accepted=True;break
                if not accepted:break
        score=nmse(pcm[selection],m.envelope(selection,symbols,origin).real)
        candidates.append((m,dict(span=span,resolution=resolution,ridge=regularization,
            knots=len(m.pulse),selection_nmse=score,training_nmse=m.training_nmse,
            clock_ppm=m.clock_ppm,carrier_offset_hz=m.carrier_offset_hz,joint_fit_steps=steps)))
    minimum=min(c[1]['selection_nmse'] for c in candidates)
    model,selected=min((c for c in candidates if c[1]['selection_nmse']<=minimum*1.01),
                      key=lambda c:(c[1]['knots'],-c[1]['ridge']))
    model.selection=dict(selected=selected,candidates=[c[1] for c in candidates],
        fit_fraction=.6,selection_fraction=.2,validation_fraction=.2,
        carrier_clock_fit=('first 60%, variable projection per candidate' if extended else
            'first 60%, baseline pulse; frozen across model candidates'),support_guard_symbols=guard,uses_b1=False)
    prediction=model.envelope(valid,symbols,origin).real
    model.held_out_training_nmse=nmse(pcm[valid],prediction)
    return model,dict(training_samples=train,selection_samples=selection,validation_samples=valid,
                      selection_prediction=model.envelope(selection,symbols,origin).real,
                      validation_prediction=prediction,validation_residual=pcm[valid]-prediction)


def compare_b1(pcm,model,symbols,origin,nonlinear=False,context=None):
    """Fit gain/energy on first-half PCM; score second half without refitting.

    Both halves exclude pulse support across their shared boundary. A known
    MP/E prefix permits scoring the start; unknown following data is excluded.
    """
    symbols=np.asarray(symbols,complex); mid=len(symbols)//2
    span=model.span
    # Wider pulses leave little interior PCM in short B1 frames. Move the
    # calibration/validation split earlier, preserving eight late symbols and
    # the full support gap; never use following unknown data for calibration.
    mid=min(mid,len(symbols)-2*span-8)
    if context is not None and len(context)<span:raise ValueError('insufficient MP/E pulse context')
    samples=np.arange(int(np.ceil(origin+(0 if context is not None else span)*model.spl)),
                      int(np.floor(origin+(len(symbols)-span)*model.spl)))
    t=(samples-origin)/model.spl
    calibration=(t>=8)&(t<mid-span);held_out=t>=mid+span
    if not calibration.any() or not held_out.any():raise ValueError('B1 too short for disjoint pulse windows')
    if samples[0]<0 or samples[-1]>=len(pcm):raise ValueError('B1 outside PCM')
    observed=np.asarray(pcm,float)[samples]
    prefix=np.zeros(len(samples)) if context is None else model.envelope(
        samples,context,origin-len(context)*model.spl).real
    base=model.envelope(samples,symbols,origin)
    quadratic=model.envelope(samples,symbols*(.3125/6)*abs(symbols)**2,origin) if nonlinear else 0
    quartic=model.envelope(samples,symbols*(.3125**2/120)*abs(symbols)**4,origin) if nonlinear else 0
    def evaluate(energy):
        envelope=base+quadratic/energy+quartic/energy**2
        design=np.column_stack((envelope.real,-envelope.imag))
        coeff=np.linalg.lstsq(design[calibration],(observed-prefix)[calibration],rcond=None)[0]
        prediction=design@coeff+prefix
        return nmse(observed[calibration],prediction[calibration]),prediction,coeff
    energies=np.geomspace(max(np.mean(abs(symbols)**2)*.15,1e-6),
                         max(np.mean(abs(symbols)**2)*5,1e-5),49) if nonlinear else [1.]
    candidates=[(evaluate(e),e) for e in energies]
    (score,prediction,coeff),energy=min(candidates,key=lambda item:item[0][0])
    residual=observed-prediction
    intervals=[]
    for k in range(0,len(symbols),2):
        mask=(calibration|held_out|(t<8)) & (t>=k) & (t<k+2)
        if mask.any():intervals.append(dict(interval_4d=k//2,
            section='calibration' if np.any(mask & calibration) else ('early_holdout' if k<8 else 'held_out'),
            first_sample=int(samples[mask][0]),last_sample=int(samples[mask][-1]),
            nmse=nmse(observed[mask],prediction[mask])))
    return dict(calibration_nmse=score,held_out_nmse=nmse(observed[held_out],prediction[held_out]),
                gain_real=float(coeff[0]),gain_imag=float(coeff[1]),energy=float(energy) if nonlinear else None,
                selection_uses_held_out=False,intervals=intervals,
                energy_at_search_boundary=bool(nonlinear and energy in (energies[0],energies[-1])),
                excluded_start_symbols=span if context is None else 0,excluded_end_symbols=span,
                excluded_half_boundary_symbols=2*span,early_symbols_used_for_calibration=False,
                calibration_split_symbol=mid,calibration_pcm_samples=int(calibration.sum()),
                held_out_pcm_samples=int(held_out.sum()),
                first_eight_nmse=nmse(observed[t<8],prediction[t<8]) if np.any(t<8) else None),dict(
                samples=samples,prediction=prediction,residual=residual,
                calibration=calibration,held_out=held_out)


INVERSE_RIDGES=(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1)


def inverse_system(pcm,model,origin,count,context):
    """Raw linear system with unknown B1 and unknown following data.

    No expected B1 values enter the matrix or right-hand side. Future unknown
    symbols are nuisance variables; zero singular values are retained/reported.
    """
    span=model.span
    if len(context)<span:raise ValueError('insufficient inverse prefix')
    samples=np.arange(int(np.ceil(origin)),int(np.floor(origin+(count+span)*model.spl)))
    if samples[0]<0 or samples[-1]>=len(pcm):raise ValueError('inverse window outside PCM')
    unknown=count+2*span
    relative=(samples[:,None]-origin)/model.spl-np.arange(unknown)[None,:]
    knots=np.linspace(-span,span,len(model.pulse))
    pulse=np.interp(relative,knots,model.pulse.real,left=0,right=0)+1j*np.interp(
        relative,knots,model.pulse.imag,left=0,right=0)
    field=pulse*np.exp(2j*np.pi*model.carrier*(samples-model.origin)/FS)[:,None]
    matrix=np.column_stack((field.real,-field.imag))
    prefix=model.envelope(samples,context,origin-len(context)*model.spl).real
    y=np.asarray(pcm,float)[samples]-prefix
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    return dict(samples=samples,matrix=matrix,rhs=y,u=u,s=s,vh=vh,
                unknown=unknown,count=count,prefix=prefix)


def solve_inverse(system,ridge):
    s=system['s']; filt=s/(s*s+ridge*s[0]**2)
    coef=system['vh'].T@(filt*(system['u'].T@system['rhs']))
    unknown=system['unknown'];observed=(coef[:unknown]+1j*coef[unknown:])[:system['count']]
    rank=int(np.count_nonzero(s>s[0]*1e-8))
    information=dict(ridge=float(ridge),rank=rank,unknown_real_parameters=2*unknown,
        samples=len(system['samples']),condition=float(s[0]/max(s[-1],1e-30)),
        effective_condition=float(s[0]/s[rank-1]) if rank else None,
        mean_noise_amplification=float(np.sum(filt*filt)/(2*system['count'])),
        waveform_nmse=nmse(system['rhs'],system['matrix']@coef),
        expected_b1_used_for_inversion=False)
    return observed,information


def select_inverse(pcm,model,trn,fit_fraction=.6):
    """Choose ridge using two known TRN blocks within the fitting section."""
    count=min(128,(int(len(trn)*fit_fraction)-4*model.span)//2)
    if count<64:raise ValueError('insufficient TRN for inverse selection')
    starts=[2*model.span,int(len(trn)*fit_fraction)-count-2*model.span]
    scores={ridge:[] for ridge in INVERSE_RIDGES}
    for k in starts:
        system=inverse_system(pcm,model,model.origin+k*model.spl,count,trn[k-model.span:k])
        for ridge in INVERSE_RIDGES:
            x,_=solve_inverse(system,ridge)
            scores[ridge].append(float(np.mean(abs(x-trn[k:k+count])**2)/np.mean(abs(trn[k:k+count])**2)))
    selected=min(INVERSE_RIDGES,key=lambda r:np.mean(scores[r]))
    return selected,dict(selection_uses_b1=False,blocks=starts,block_symbols=count,
                         candidates=[dict(ridge=r,trn_symbol_nmse=float(np.mean(scores[r]))) for r in INVERSE_RIDGES])


def decode_symbols(observed,profile,sender,state=(0,0,0,0)):
    """Diagnostic decoding of measurements, never substitution by expected u."""
    from v34_receive import Receiver
    from v34_negotiation import DirectionParameters
    receiver=Receiver(DirectionParameters(**profile),sender,state)
    try:
        gain,_=receiver.qualify(observed)
        detector=Receiver(DirectionParameters(**profile),sender,state)
        detector.energy=receiver.energy;vit=detector.detector()
        for a,b in np.asarray(observed).reshape(-1,2)/gain:vit.feed(a,b)
        vit.flush();points=np.asarray([x for pair in vit.decided for x in pair])
        from v34_b1_reference import build
        ref=build(profile['rate'],profile['baud'],sender,profile['shaping'],profile['precoder'],state)
        errors=np.flatnonzero(points!=ref['u'])
        return dict(**receiver.diagnostics,mapper_errors=len(errors),
                    first_mapper_error=int(errors[0]) if len(errors) else None,
                    zero_reset_b1_decoded=not any(state) and receiver.diagnostics['qualified'],
                    standard_qualification=False,expected_mapper_substituted=False),points
    except ValueError as exc:
        return dict(qualified=False,reason=str(exc),standard_qualification=False),np.array([],complex)
