"""Per-call opt-in TRN cancellation and B1 prediction, without global hooks.

The source is committed emitted PCM. Models never receive channel truth or
payload/B1 labels. Rejected or unavailable models preserve ordinary reception.
"""
import copy
import os
import time
import threading
import numpy as np
import v34_tx as tx
import v34_duplex_model as model


class ReceiveDsp:
    def __init__(self, engine, *, cancel=False, predictive=False, trn_clock=False, trn_gradient=False):
        if (trn_clock or trn_gradient) and (cancel or predictive):
            raise ValueError('TRN clock currently requires ordinary detection without cancellation/prediction')
        self.trn_gradient=trn_gradient
        self._gradient_first_error=None
        self._gradient_activation=None
        self._startup_timing_mode=None
        self._startup_timing_restored=False
        self.engine=engine;self.enabled=cancel;self.predictive=predictive
        self.source=np.zeros(1024);self.length=0;self.source_origin=0
        self.received=0;self.rows=[];self.clone=None;self.model=None;self.noise_model=None
        self.activation=None;self.clipped=0;self.calls=[];self.fallbacks=[]
        self._source_lock=threading.RLock();self._fitting=False
        from v34_trn_clock import TrnClock
        self.clock=TrnClock() if trn_clock else None
        engine.receive_dsp=self

    def commit_tx(self, samples):
        """Append only samples already emitted; sample counts are never rewound."""
        if self.enabled or self.predictive:
            with self._source_lock:self._commit_tx(samples)

    def _commit_tx(self,samples):
        values=np.asarray(samples,float)
        required=self.length-self.source_origin+len(values)
        if required>len(self.source):
            grown=np.zeros(max(required,2*len(self.source)))
            grown[:self.length-self.source_origin]=self.source[:self.length-self.source_origin]
            self.source=grown
        self.source[self.length-self.source_origin:required]=values
        self.length+=len(values)
        if self.model is not None and not self._fitting and required>20096:
            drop=required-16000
            self.source=self.source[drop:required].copy();self.source_origin+=drop

    def process(self, pcm):
        with self._source_lock:return self._process(pcm)

    def _process(self,pcm):
        start=self.received;self.received+=len(pcm)
        if not self.enabled or not self.model or not self.model.get('accepted'):
            return np.asarray(pcm)
        if self.activation is None:self.activation=start
        delay=self.model['delay_samples'];taps=np.asarray(self.model['taps'])
        absolute=np.arange(start,start+len(pcm))[:,None]-delay-np.arange(len(taps))
        if absolute.max()>=self.length or np.any((absolute>=0)&(absolute<self.source_origin)):
            self.fallbacks.append('TX reference unavailable at receive frame')
            self.model['accepted']=False
            return np.asarray(pcm)
        indices=absolute-self.source_origin
        echo=(self.source[np.clip(indices,0,len(self.source)-1)]*(absolute>=0))@taps
        fade=np.clip((np.arange(start,start+len(pcm))-self.activation)/128,0,1)
        result=np.asarray(pcm,float)-fade*echo
        self.clipped+=int(np.count_nonzero((result>32767)|(result<-32768)))
        return np.clip(np.rint(result),-32768,32767).astype('<i2')

    def _gradient_ready(self, receiver):
        if self.trn_gradient != 'auto':return True
        # Give a genuine MP enough symbols to confirm its CRC before treating
        # repeated loss of known TRN labels as tracking failure.
        if receiver.trn_end is not None:return False
        mismatch=receiver.mismatch
        n=len(mismatch)
        if n<512:return False
        if self._gradient_first_error is None:
            if mismatch[-1] and len(mismatch)>=65 and not any(mismatch[-65:-1]):
                self._gradient_first_error=n-1
            return False
        grace=64 if getattr(receiver,'c16',True) else 128
        return n-self._gradient_first_error>=grace and sum(mismatch[-128:])>=2

    def before_symbol(self, receiver):
        if self.trn_gradient and self._startup_timing_mode is None and self._gradient_ready(receiver):
            self._gradient_activation=receiver.eq.k-receiver.k_trn
            self._startup_timing_mode=receiver.eq.timing_mode
            receiver.eq.timing_mode='gradient'
            # A centered 81-tap FIR revises its trailing 40 samples. Timing
            # scores, like data decisions, require settled filtered input.
            receiver.eq._filter_future=40
            import os
            receiver.eq._native_interp=os.environ.get('V34_GRADIENT_NATIVE','1')=='1'
            receiver.eq._gradient_rows=[]
        if self.clock is not None:self.clock.before_symbol(receiver)
        if not (self.enabled or self.predictive):return
        eq=receiver.eq;index=eq.k-receiver.k_trn
        if self.model is not None:return
        if index==1024:
            self.clone=copy.copy(eq);self.clone.h=eq.h.copy();self.clone.adapt=False;self.clone._grid=None
            source_offset=getattr(self.engine,'_p4_x0',None)
            if source_offset is None:source_offset=self.engine._p3['x0']
            self.offset=self.received-self.engine.rx_t+source_offset
            self.h=self.clone.h.copy();self.fc=eq.R.fc;self.spl=eq.R.spl
        if self.clone is not None and index<model.HOLD[1]:
            if self.clone.k!=eq.k:raise ValueError('frozen TRN observation lost symbol alignment')
            center=self.clone.timing_snapshot()['sample_position'];rot=np.exp(-1j*self.clone.phase)
            frozen=self.clone.step(decide=lambda y:y)
            self.rows.append((index,frozen,center+self.offset,rot))

    def after_feed(self, receiver):
        if self.clock is not None:self.clock.after_feed(receiver)
        if not (self.enabled or self.predictive):return
        if self.model is not None:return
        if receiver.trn_end is not None and receiver.trn_end<model.HOLD[1]+80:
            self.model=dict(accepted=False,reason='insufficient guarded TRN')
            self.rows=[];self.clone=None
            return
        if len(receiver.idx)<model.HOLD[1]+80 or self.clone is None:return
        started=time.monotonic()
        self._fitting=True
        try:
            with self._source_lock:
                if self.source_origin:raise ValueError('training TX history was trimmed')
                self._training_source=self.source[:self.length].copy()
            rows=np.asarray(self.rows)
            trace=dict(symbol=rows[:,0].real.astype(int),observed=rows[:,1],center=rows[:,2].real,rotation=rows[:,3])
            known,_=tx.trn_symbols(model.HOLD[1]+80,receiver.role,16 if receiver.c16 else 4)
            # Some zero-delay windows reach beyond emitted TX history even
            # though later, causal delays have complete support. Exclude only
            # those unavailable delays; keep the predeclared blocks and gates.
            hold=(trace['symbol']>=model.HOLD[0])&(trace['symbol']<model.HOLD[1])
            support=len(self.h)*self.spl/4+46
            minimum_delay=max(0,int(np.floor(trace['center'][hold].max()+support
                -len(self._training_source)))+1)
            self.model=model.train_model(self._training_source,trace,known,self.fc,self.spl,self.h,self.offset,
                                         minimum_delay=minimum_delay)
            self.model.update(training_complete_receive_sample=self.received,last_training_symbol=4399,
                              training_uses_future_pcm=False,fit_seconds=time.monotonic()-started)
            if self.predictive:self._train_noise(receiver,trace,known)
        except (ValueError,np.linalg.LinAlgError) as error:
            self.model=dict(accepted=False,reason=str(error))
            self.noise_model=dict(accepted=False,reason=str(error))
        finally:
            self.rows=[];self.clone=None
            self._fitting=False
            self._training_source=None

    def _train_noise(self, receiver, trace, known):
        ids=trace['symbol'];fit=(ids>=model.FIT[0])&(ids<model.FIT[1])
        hold=(ids>=model.HOLD[0])&(ids<model.HOLD[1])
        if fit.sum()!=1000 or hold.sum()!=1000:raise ValueError('incomplete predictive TRN blocks')
        nuisance=model.own_design(known,ids)
        own=np.linalg.lstsq(nuisance[fit],trace['observed'][fit],rcond=1e-10)[0]
        residual=trace['observed']-nuisance@own
        if self.enabled and self.model['accepted']:
            import v34_rx as rx
            source=rx.to_baseband(self._training_source,self.fc,max(1700,.62*8000/self.spl))
            rotation=trace['rotation']*np.exp(2j*np.pi*self.fc*self.offset/8000)
            features=model.echo_features(source,trace['center'],self.h,self.spl,rotation,self.fc,self.model['delay_samples'])
            residual-=features@np.asarray(self.model['taps'])
        early,later=residual[fit],residual[hold]
        def design(e):return np.column_stack([e[3-k:len(e)-k] for k in (1,2,3)])
        coefficients=np.linalg.lstsq(design(early),early[3:],rcond=1e-10)[0]
        innovation=later[3:]-design(later)@coefficients
        baseline=float(np.mean(abs(later[3:])**2));improvement=1-float(np.mean(abs(innovation)**2))/max(baseline,1e-30)
        poles=float(max(abs(np.roots(np.r_[1,-coefficients]))))
        intervals=model.bootstrap_benefit(abs(later[3:])**2-abs(innovation)**2)
        accepted=improvement>=.1 and poles<.98 and all(v[0]>0 for v in intervals.values())
        self.coefficients=coefficients
        self.noise_model=dict(accepted=bool(accepted),improvement_fraction=improvement,maximum_pole_radius=poles,
            coefficients=[[v.real,v.imag] for v in coefficients],last_training_symbol=4399,
            pcm_support_gap_samples=self.model['pcm_support_gap_samples'],
            uses_b1_symbols=False,uses_b1_errors=False,uses_payload_or_injection_truth=False,
            training_uses_future_pcm=False)

    def before_b1(self, eq, k_trn):
        if self.trn_gradient and not self._startup_timing_restored:
            if self._startup_timing_mode is not None:
                eq.timing_mode=self._startup_timing_mode
                eq._gradient_rows=[]
            self._startup_timing_restored=True
        if self.clock is None:return
        self.clock.apply(eq, k_trn)
        m=self.clock.model or {}
        log=getattr(self.engine,'_ev',None)
        if log is None or getattr(self,'_clock_logged',False):return
        self._clock_logged=True
        improvement=m.get('validation_improvement')
        detail=''if improvement is None else f' validation improvement {100*improvement:.1f}%'
        if m.get('accepted'):
            gate=''
            if m.get('gate') == 'replaced':
                gate=f" (gate: fitted velocity {m['gate_fitted_velocity']:.2e} replaced by carrier-derived {m['gate_carrier_velocity']:.2e})"
            log(f"TRN clock fit accepted{detail}; applied at symbol {m.get('applied_at_symbol')}"
                f" offset {m.get('applied_timing_offset', float('nan')):.3f} velocity {m.get('applied_velocity', float('nan')):.2e}"+gate)
        else:
            note=''
            if m.get('gate') == 'fallback':
                note=(f"; applied anyway at symbol {m.get('applied_at_symbol')} offset {m.get('applied_timing_offset', float('nan')):.3f}"
                      f" with the carrier-derived velocity {m['gate_carrier_velocity']:.2e} (V34_CLOCK_FIT_GATE)")
            log(f"TRN clock fit bypassed{detail}"+(f" ({m['reason']})" if m.get('reason') else ' (gate rejected)')+note)

    def qualify(self, receiver, observed, gain):
        if not self.predictive or not self.noise_model or not self.noise_model.get('accepted'):return
        import v34_datarx as datarx
        from v34_noise_predictive import detect
        started=time.monotonic();baseline=receiver.diagnostics['bit_errors']
        points,cost=detect(np.asarray(observed)/gain,receiver,self.coefficients)
        receiver.drx=datarx.DataRx(receiver.profile.rate,receiver.profile.baud,
                                 expanded=receiver.profile.shaping,precoder=receiver.h)
        receiver.drx.conv.d=list(receiver.initial_state)
        bits=[]
        for i in range(0,len(points),8):bits.extend(receiver.drx.decode_frame(points[i:i+8])[0])
        errors=sum(a!=b for a,b in zip(bits,receiver.expected))+abs(len(bits)-len(receiver.expected))
        for key in ('first_decoded_point_deltas','first_divergence','point_distance_domain'):
            receiver.diagnostics.pop(key,None)
        info=dict(applied=True,baseline_bit_errors=baseline,innovation_cost=cost,
                  seconds=time.monotonic()-started,data_detector='ordinary streaming',uses_expected_branch_targets=False)
        receiver.diagnostics.update(qualified=errors==0,bit_errors=errors,
            mapper_symbol_errors=int(np.sum(points!=receiver.expected_mapper)),
            error_bit_indices=[i for i,(a,b) in enumerate(zip(bits,receiver.expected)) if a!=b],
            length_mismatch=abs(len(bits)-len(receiver.expected)),noise_predictive=info)
        receiver.bits=bits;receiver.slicer_hist=list(receiver.drx.x_hist);receiver.vit=receiver.detector(start=True)
        self.calls.append(info)

    def report(self):
        return dict(trn_gradient=self.trn_gradient, gradient_activation_trn_symbol=self._gradient_activation,
            startup_timing_restored=self._startup_timing_restored, trn_clock=self.clock.report() if self.clock else None,
            enabled=self.enabled,predictive=self.predictive,model=self.model,noise_model=self.noise_model,
            qualifications=self.calls,activation_sample=self.activation,clipped_samples=self.clipped,
            tx_samples=self.length,retained_tx_samples=self.length-self.source_origin,
            fallbacks=self.fallbacks,uses_payload_or_injection_truth=False)


def configured(engine):
    cancel=os.environ.get('V34_DUPLEX_CANCEL')=='1';predictive=os.environ.get('V34_B1_NOISE_PREDICTIVE')=='1'
    clock=os.environ.get('V34_TRN_CLOCK')=='1'
    value=os.environ.get('V34_TRN_GRADIENT','0')
    gradient='auto' if value=='auto' else value=='1'
    return ReceiveDsp(engine,cancel=cancel,predictive=predictive,trn_clock=clock,trn_gradient=gradient) if cancel or predictive or clock or gradient else None
