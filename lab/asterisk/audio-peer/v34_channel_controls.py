"""Offline channel/noise controls derived from TRN, never from B1 decisions."""
from types import SimpleNamespace
import numpy as np
from v34_forward_audit import FS,fit_pulse,select_inverse,inverse_system,solve_inverse,decode_symbols


def bandlimited_sample(pcm,positions,half=32):
    """Windowed-sinc real PCM resampling; linear interpolation alters RF gain."""
    out=np.empty(len(positions));offsets=np.arange(-half+1,half+1)
    for start in range(0,len(positions),4096):
        p=positions[start:start+4096];integer=np.floor(p).astype(int)
        distance=offsets[None,:]-(p-integer)[:,None]
        weights=np.sinc(distance)*(.5+.5*np.cos(np.pi*distance/half))
        weights/=weights.sum(axis=1)[:,None]
        indexes=np.clip(integer[:,None]+offsets[None,:],0,len(pcm)-1)
        out[start:start+len(p)]=np.sum(pcm[indexes]*weights,axis=1)
    return out


def spectrum(model,frequencies):
    t=np.arange(-int(np.ceil(model.span*model.spl)),int(np.ceil(model.span*model.spl))+1)
    knots=np.linspace(-model.span,model.span,len(model.pulse))
    pulse=np.interp(t/model.spl,knots,model.pulse.real,left=0,right=0)+1j*np.interp(
        t/model.spl,knots,model.pulse.imag,left=0,right=0)
    return np.exp(-2j*np.pi*np.asarray(frequencies)[:,None]*t/FS)@pulse


def channel_kernel(source,target):
    """Fit a finite real RF filter to TRN pulses, avoiding spectral seams.

    A 33-tap centered FIR limits added pulse support. Report mismatch rather
    than treating an arbitrary complex pulse ratio as a physical real filter.
    """
    extent=int(np.ceil(max(source.span*source.spl,target.span*target.spl)))+16
    t=np.arange(-extent,extent+1);offsets=np.arange(-16,17)
    def pulse(model,positions):
        knots=np.linspace(-model.span,model.span,len(model.pulse))
        return (np.interp(positions/model.spl,knots,model.pulse.real,left=0,right=0)+1j*np.interp(
            positions/model.spl,knots,model.pulse.imag,left=0,right=0))*np.exp(2j*np.pi*model.carrier*positions/FS)
    a=pulse(source,t);b=pulse(target,t)
    b*=np.sqrt(np.vdot(a,a).real/np.vdot(b,b).real)*np.exp(-1j*np.angle(np.vdot(a,b)))
    design=pulse(source,t[:,None]-offsets[None,:])
    real=np.vstack((design.real,design.imag));rhs=np.r_[b.real,b.imag]
    gram=real.T@real;ridge=1e-3*np.trace(gram)/len(gram)
    prior=np.zeros(33);prior[16]=1
    kernel=np.linalg.solve(gram+ridge*np.eye(33),real.T@rhs+ridge*prior)
    mismatch=float(np.mean(abs(design@kernel-b)**2)/np.mean(abs(b)**2))
    return kernel,dict(taps=33,pulse_transfer_nmse=mismatch,derived_from='TRN pulses only')


def transfer(source,target):
    kernel,_=channel_kernel(source,target);frequencies=np.linspace(0,FS/2,2049)
    response=np.exp(-2j*np.pi*frequencies[:,None]*np.arange(-16,17)/FS)@kernel
    return frequencies,response


def impair(pcm,source,target,residual,reference_rms,condition,seed):
    """Apply a measured filter/drift and/or bootstrap TRN residual blocks.

    Residual bootstrapping is an alternative noise explanation, not proof that
    the measured residual is stochastic. Preserve the source on its clock.
    """
    pcm=np.asarray(pcm,float);out=pcm.copy()
    if condition in ('channel','combined'):
        kernel,_=channel_kernel(source,target)
        filtered=np.convolve(pcm,kernel,mode='same');size=1<<(len(pcm)-1).bit_length()
        # Analytic modulation applies independent carrier error, then clock
        # resampling follows the specified symbol drift around the TRN anchor.
        analytic_filter=np.zeros(size);analytic_filter[0]=1;analytic_filter[1:size//2]=2;analytic_filter[size//2]=1
        analytic=np.fft.ifft(np.fft.fft(filtered,n=size)*analytic_filter)[:len(pcm)]
        n=np.arange(len(pcm));clock=1+target.clock_ppm*1e-6
        # Compensate the carrier shift introduced by clock resampling, so
        # imposed carrier error and symbol drift remain independent.
        offset=(source.carrier+target.carrier_offset_hz)*clock-source.carrier
        out=(analytic*np.exp(2j*np.pi*offset*(n-source.origin)/FS)).real
        positions=source.origin+(n-source.origin)/(1+target.clock_ppm*1e-6)
        out=bandlimited_sample(out,positions)
    if condition in ('noise','combined'):
        rng=np.random.default_rng(seed);residual=np.asarray(residual,float)-np.mean(residual)
        if len(residual)<64:raise ValueError('insufficient TRN residual for block bootstrap')
        starts=rng.integers(0,len(residual)-32+1,size=(len(pcm)+31)//32)
        noise=residual[starts[:,None]+np.arange(32)].ravel()[:len(pcm)]
        lo=int(source.origin+32*source.spl);hi=int(source.origin+300*source.spl)
        scale=np.sqrt(np.mean(pcm[lo:hi]**2))/reference_rms
        out+=noise*scale
    if condition not in ('clean','channel','noise','combined'):raise ValueError('unknown condition')
    return out


def frontend(pcm,model,trn,b1_origin,profile,sender):
    """TRN-trained T/2 diagnostic plus the production B1 tap refinement.

    This uses observed anchors, not a complete streaming handshake replay.
    Carrier/clock come from TRN; no expected B1 enters initial equalizer taps.
    """
    from v34_rx import Phase3Rx,SymbolEq
    from v34_receive import initialize,SearchPolicy
    from v34_negotiation import DirectionParameters
    r=Phase3Rx(profile['baud'],model.carrier,sender);r.spl=model.spl
    r.lag=model.origin-r.LEAD*r.spl;r.delay=0
    r._z=r.update_baseband(pcm)
    first=20;last=int(len(trn)*.6)-20
    r.train_known(trn[first:last],first)
    k=round((b1_origin-model.origin)/model.spl);eq=SymbolEq(r,k)
    p=DirectionParameters(**profile)
    engine=SimpleNamespace(negotiated=SimpleNamespace(rx=p),_rx_b1=k,_rx=pcm,_p4_x0=0,
        _receive_trace=True,rx_qualification={},failed_reason=None,_ev=lambda message:None)
    initialized=initialize(engine,eq,sender,profile['baud'],SearchPolicy())
    trace=getattr(engine,'_b1_inputs',{})
    result=dict(initialized=bool(initialized),after=engine.rx_qualification,
                full_streaming_acquisition=False)
    for stage in ('before','after'):
        key='observed_'+stage
        if key in trace:
            result[stage],_=decode_symbols(trace[key],profile,sender)
    return result


def evaluate(pcm,source,trn,b1_origin,context,profile,sender):
    model,_=fit_pulse(pcm,trn,source.origin,FS/profile['baud'],profile['carrier'])
    origin=model.origin+(b1_origin-source.origin)/source.spl*model.spl
    from v34_b1_reference import build
    count=len(build(profile['rate'],profile['baud'],sender,profile['shaping'],profile['precoder'])['x'])
    ridge,selection=select_inverse(pcm,model,trn)
    system=inverse_system(pcm,model,origin,count,context)
    symbols,information=solve_inverse(system,ridge)
    qualification,_=decode_symbols(symbols,profile,sender)
    return dict(training_validation_nmse=model.held_out_training_nmse,model_selection=model.selection,
        fitted_carrier_offset_hz=model.carrier_offset_hz,fitted_clock_ppm=model.clock_ppm,
        inverse=dict(**information,selection=selection,qualification=qualification),
        frontend=frontend(pcm,model,trn,origin,profile,sender),
        standard_qualification=False,impaired_payload_verified=False)


def timing_cache_probe():
    """Demonstrate cached fractional-position behavior without changing DSP."""
    from v34_rx import SymbolEq,interp
    n=np.arange(2048);z=np.exp(1j*(.22*n+.0001*n*n))
    r=SimpleNamespace(_span=12,spl=2.5,lag=50,LEAD=8,delay=0,taps=np.ones(24),_z=z,baud=3200,fc=1900)
    eq=SymbolEq(r,40);eq.window(40);eq.dt+=.25
    positions=r.lag+(r.LEAD+eq.k)*r.spl+eq.dt+eq._offs
    desired=interp(z,positions);cached=eq.window(40).copy()
    eq._grid=None;rebuilt=eq.window(40)
    return dict(fractional_shift_samples=.25,cached_nmse=float(np.mean(abs(cached-desired)**2)),
                rebuilt_nmse=float(np.mean(abs(rebuilt-desired)**2)),causal_for_reference_failures=False)
