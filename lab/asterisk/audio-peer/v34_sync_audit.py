"""Independent timing/channel diagnostics; expected B1 never selects parameters."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from v34_forward_audit import FS, nmse, decode_symbols


def local_alignment(pcm, model, symbols, origin, samples):
    """Relative timing against one fixed TRN pulse; phase/gain are nuisance terms."""
    samples=np.asarray(samples,int);y=pcm[samples];offset=0.
    def fitted(t):
        wave=model.envelope(samples,symbols,origin+t)
        a=np.column_stack((wave.real,-wave.imag))
        g=np.linalg.lstsq(a,y,rcond=1e-8)[0]
        return wave,a,g,a@g
    wave,a,g,prediction=fitted(offset)
    for _ in range(8):
        plus=model.envelope(samples,symbols,origin+offset+.01)
        minus=model.envelope(samples,symbols,origin+offset-.01)
        derivative=((plus-minus)/.02*(g[0]+1j*g[1])).real
        derivative-=a@np.linalg.lstsq(a,derivative,rcond=1e-8)[0]
        step=float(np.dot(derivative,y-prediction)/max(np.dot(derivative,derivative),1e-30))
        step=float(np.clip(step,-.25*model.spl,.25*model.spl))
        accepted=False
        for fraction in (1.,.5,.25):
            new=float(np.clip(offset+step*fraction,-.5*model.spl,.5*model.spl))
            w,b,h,p=fitted(new)
            if nmse(y,p)<nmse(y,prediction):
                offset,wave,a,g,prediction=new,w,b,h,p;accepted=True;break
        if not accepted:break
    return dict(center_sample=float(np.mean(samples)),timing_offset_samples=offset,
        carrier_phase=float(np.angle(g[0]+1j*g[1])),gain=float(np.hypot(*g)),nmse=nmse(y,prediction),
        timing_information=float(np.dot(derivative,derivative)/max(np.dot(y,y),1e-30)),
        timing_at_bound=bool(abs(offset)>=.499*model.spl))


def trajectory(pcm,model,symbols,origin,kind):
    rows=[]
    for k in range(model.span, len(symbols)-128-model.span+1,64):
        samples=np.arange(int(np.ceil(origin+k*model.spl)),int(np.floor(origin+(k+128)*model.spl)))
        row=local_alignment(pcm,model,symbols,origin,samples)
        rows.append(dict(kind=kind,symbol_start=k,**row))
    return rows


def slope(rows):
    usable=[r for r in rows if not r['timing_at_bound'] and r['timing_information']>1e-8]
    if len(usable)<3:return dict(identifiable=False,reason='fewer than three usable timing windows')
    t=np.array([r['center_sample'] for r in usable]);a=np.column_stack((np.ones(len(t)),t-t[0]))
    delay=np.array([r['timing_offset_samples'] for r in usable]);phase=np.unwrap([r['carrier_phase'] for r in usable])
    dt=np.linalg.lstsq(a,delay,rcond=None)[0];dp=np.linalg.lstsq(a,phase,rcond=None)[0]
    return dict(identifiable=True,anchor_sample=float(t[0]),timing_intercept=float(dt[0]),
        additional_clock_ppm=float(dt[1]*1e6),additional_carrier_hz=float(dp[1]*FS/(2*np.pi)),
        phase_intercept=float(dp[0]),timing_fit_rms=float(np.sqrt(np.mean((delay-a@dt)**2))),
        phase_fit_rms=float(np.sqrt(np.mean((phase-a@dp)**2))),windows=len(usable))


def band_edges(pcm,carrier,baud,lo,hi):
    """Godard-style band-edge cross product; relative phase, not absolute timing.

    Two complex resonators and removal of their symbol-rate beat expose cyclic
    timing phase. Channel phase biases absolute phase; coherence gates usefulness.
    """
    alpha=.98;states=[0j,0j];frequencies=[carrier-baud/2,carrier+baud/2]
    poles=[alpha*np.exp(2j*np.pi*f/FS) for f in frequencies]
    rows=[];products=[];powers=[];block=max(128,int(128*FS/baud))
    for n in range(max(0,lo-1024),min(len(pcm),hi)):
        for i in (0,1):states[i]=poles[i]*states[i]+(1-alpha)*pcm[n]
        if n<lo:continue
        products.append(states[1]*states[0].conjugate()*np.exp(-2j*np.pi*baud*n/FS))
        powers.append([abs(v)**2 for v in states])
        if len(products)==block:
            mean=np.mean(products);power=np.mean(powers,axis=0)
            coherence=float(abs(mean)/max(np.sqrt(np.prod(power)),1e-30))
            rows.append(dict(center_sample=n-block/2,phase=float(np.angle(mean)),coherence=coherence,
                observable=bool(coherence>.1)))
            products=[];powers=[]
    return dict(method='band-edge cyclic cross-product',frequencies_hz=frequencies,rows=rows,
        absolute_timing_identified=False,observable_windows=sum(r['observable'] for r in rows))


def residual_spectrum(residual,opposite):
    window=256;hop=128;frames=[];cross=[];other=[]
    for k in range(0,len(residual)-window+1,hop):
        w=np.hanning(window);a=np.fft.rfft(residual[k:k+window]*w);b=np.fft.rfft(opposite[k:k+window]*w)
        frames.append(abs(a)**2);other.append(abs(b)**2);cross.append(a*b.conjugate())
    if not frames:return dict(identifiable=False)
    power=np.mean(frames,axis=0);other=np.mean(other,axis=0);cross=np.mean(cross,axis=0)
    coherence=abs(cross)**2/np.maximum(power*other,1e-30)
    indexes=np.argsort(power)[-8:][::-1];f=np.fft.rfftfreq(window,1/FS)
    return dict(identifiable=True,windows=len(frames),peaks=[dict(hz=float(f[i]),power=float(power[i]),
        opposite_lane_coherence=float(coherence[i])) for i in indexes],
        interpretation='Shared-time spectral association; does not establish echo or random noise')


def equalizer_ablation(pcm,model,trn,b1_origin,profile,sender,original_clock,span=12,ridge=1e-3):
    from v34_rx import Phase3Rx,SymbolEq
    from v34_receive import initialize,SearchPolicy
    from v34_negotiation import DirectionParameters
    r=Phase3Rx(profile['baud'],model.carrier,sender);r.spl=model.spl
    r.lag=model.origin-r.LEAD*r.spl;r.delay=0;r._z=r.update_baseband(pcm)
    # Equalizers share precisely the same samples in every span/ridge trial.
    first=50;last=int(len(trn)*.6)-50
    r.train_known(trn[first:last],first,span=span,reg=ridge)
    # Integer E boundary is invariant. Each clock maps that symbol index to PCM.
    k=round((original_clock[1]-original_clock[0])/(FS/profile['baud']))
    eq=SymbolEq(r,k);p=DirectionParameters(**profile)
    engine=SimpleNamespace(negotiated=SimpleNamespace(rx=p),_rx_b1=k,_rx=pcm,_p4_x0=0,
        _receive_trace=True,rx_qualification={},failed_reason=None,_ev=lambda _:None)
    valid=initialize(engine,eq,sender,profile['baud'],SearchPolicy());trace=getattr(engine,'_b1_inputs',{})
    validation_start=int(len(trn)*.8)+50;validation_end=len(trn)-50
    validation=(r.equalize(validation_start,validation_end-validation_start)
                if validation_end>validation_start else np.array([]))
    select_start=int(len(trn)*.6)+50;select_end=int(len(trn)*.8)-50
    selected=r.equalize(select_start,select_end-select_start) if select_end>select_start else np.array([])
    selection_mse=float(np.mean(abs(selected-trn[select_start:select_end])**2)/
        np.mean(abs(trn[select_start:select_end])**2)) if len(selected) else None
    mse=float(np.mean(abs(validation-trn[validation_start:validation_end])**2)/
        np.mean(abs(trn[validation_start:validation_end])**2)) if len(validation) else None
    result=dict(initialized=bool(valid),qualification=engine.rx_qualification,
        span=span,taps=2*span,ridge=ridge,trn_validation_symbol_nmse=mse,
        trn_selection_symbol_nmse=selection_mse,tap_noise_gain=float(np.vdot(r.taps,r.taps).real),full_streaming_acquisition=False)
    for stage in ('before','after'):
        if 'observed_'+stage in trace:result[stage],_=decode_symbols(trace['observed_'+stage],profile,sender)
    return result
