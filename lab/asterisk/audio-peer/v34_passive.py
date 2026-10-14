"""Passive two-channel V.34 observation on the original WAV sample clock.

No live modem objects or synthesized local transmitter. Each direction has a
receive filter, but both use one recording and one ordered exchange timeline.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import wave
import numpy as np
import v34_tx as tx
import v34_rx as rx
import v34_negotiation as negotiation
import v34_receive as receive
import v34_data as mapper
from v34_info import InfoDemodulator
from v34_b1_reference import build as b1_reference
from v34_data import frame_params
from v34_ans import ToneTracker

FS=8000
ROLES=('call','answer')


def load_stereo(path, chunk_samples=160):
    path=Path(path)
    with wave.open(str(path)) as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype()) != (FS,2,2,'NONE'):
            raise ValueError('requires original uncompressed stereo 8000 Hz signed 16-bit PCM')
        if chunk_samples < 1:raise ValueError('chunk size must be positive')
        chunks=[]
        while True:
            raw=w.readframes(chunk_samples)
            if not raw:break
            chunks.append(np.frombuffer(raw,dtype='<i2').reshape(-1,2))
        data=np.concatenate(chunks) if chunks else np.empty((0,2),dtype='<i2')
    return data,hashlib.sha256(path.read_bytes()).hexdigest()


def event(sender,kind,start,end=None,**evidence):
    return dict(sender=sender,kind=kind,start_sample=float(start),
                end_sample=float(start if end is None else end),evidence=evidence)


def symbol_sample(lane,k):
    r=lane['R']
    return lane['offset']+r.lag+(r.LEAD+k+r.delay)*r.spl


def information(data):
    events=[]
    for col,sender in enumerate(ROLES):
        for kind in ('info0','info1c' if sender=='call' else 'info1a'):
            frames=InfoDemodulator(sender,phases=8).decode(data[:,col],kind)
            kept=[]
            for f in sorted(frames,key=lambda f:f['_start_sample']):
                fields={k:v for k,v in f.items() if not k.startswith('_')}
                if kept and f['_start_sample']-kept[-1]['start_sample']<FS*.015:
                    continue
                kept.append(event(sender,kind,f['_start_sample'],f['_end_sample'],
                    fields=fields,crc_valid=True,timing_uncertainty_samples=FS/600))
            events.extend(kept)
    return sorted(events,key=lambda e:(e['start_sample'],e['sender'],e['kind']))


def prelude(data):
    """Context labels from spectral evidence; no regional dialtone assumption."""
    events=[];window=320;hop=40
    low=(697,770,852,941);high=(1209,1336,1477,1633)
    digits=('123A','456B','789C','*0#D')
    for col,sender in enumerate(ROLES):
        x=data[:,col].astype(float)
        freqs=low+high+(2100,1200,2400,350,440,425)
        t=np.arange(window)/FS
        basis=np.exp(-2j*np.pi*np.array(freqs)[:,None]*t)*np.hanning(window)
        runs={};finished=[]
        for start in range(0,max(0,len(x)-window+1),hop):
            seg=x[start:start+window];rms=float(np.sqrt(np.mean(seg**2)))
            labels=set()
            if rms>60:
                magnitude=abs(basis@seg)/np.sum(np.hanning(window))
                li=int(np.argmax(magnitude[:4]));hi=int(np.argmax(magnitude[4:8]))
                if magnitude[li]>.3*rms and magnitude[4+hi]>.3*rms:
                    labels.add('DTMF '+digits[li][hi])
                for i,f in enumerate(freqs[8:],8):
                    if magnitude[i]>.45*rms:labels.add('tone '+str(f)+' Hz')
            for label in list(runs):
                if label not in labels:
                    first,last=runs.pop(label)
                    if last+window-first>=320:finished.append((label,first,last+window))
            for label in labels:
                first,_=runs.get(label,(start,start));runs[label]=(first,start)
        finished.extend((label,first,last+window) for label,(first,last) in runs.items())
        for label,start,end in finished:
            if end-start>=320:events.append(event(sender,label,start,end,context_only=True,
                timing_uncertainty_samples=window))
        for f in ((1200,2400) if sender=='call' else (2100,2400)):
            tracker=ToneTracker(f);tracker.feed(x)
            events.extend(event(sender,'phase reversal',sample,frequency=f,
                timing_uncertainty_samples=16) for sample in tracker.reversals)
    return events


def pair_attempts(events):
    """Pair observed INFO1 exchanges; never borrow parameters across retries."""
    calls=[e for e in events if e['kind']=='info1c']
    answers=[e for e in events if e['kind']=='info1a']
    pairs=[];used=set()
    for answer in answers:
        candidates=[e for e in calls if e['start_sample']<=answer['start_sample'] and
                    answer['start_sample']-e['start_sample']<10*FS]
        if not candidates:continue
        call=max(candidates,key=lambda e:e['start_sample'])
        key=(call['start_sample'],answer['start_sample'])
        if key in used:continue
        used.add(key);pairs.append((call,answer))
    return pairs


def train_phase3(data,sender,parameters,lo,hi,events):
    col=ROLES.index(sender);baud,fc=parameters[sender]
    offset=max(0,int(lo)-80)
    x=data[offset:int(hi),col].astype(float)
    r=rx.Phase3Rx(baud,fc,sender)
    got=r.acquire(x,search=min(len(x),6*FS))
    if got is None:raise ValueError(f'{sender}: no observed S/S-bar/PP')
    r._z=r.update_baseband(x)
    if r.lag+(r.LEAD+432+700+16)*r.spl>=len(x):
        raise ValueError(f'{sender}: truncated phase-three training')
    mse,trn_role=r.train_trn_auto()
    follower=rx.Phase3Receiver(r,role=trn_role);follower.feed()
    lane=dict(R=r,offset=offset,phase3=follower,role=sender,training_role=trn_role)
    for kind,start,end in [('S',0,128),('S-bar',128,144),('PP',144,432)]:
        events.append(event(sender,kind,symbol_sample(lane,start),symbol_sample(lane,end),
            phase=3,correlation=got['metric']))
    events.append(event(sender,'TRN',symbol_sample(lane,432),
        symbol_sample(lane,432+(follower.j[3] if follower.j else len(follower.idx))),
        phase=3,training_nmse=mse,scrambler_role=trn_role))
    if follower.j is None:raise ValueError(f'{sender}: no observed J')
    j,seen,pattern,split=follower.j
    events.append(event(sender,'J',symbol_sample(lane,432+j),
        symbol_sample(lane,432+j+seen),pattern=pattern,confirmed_symbols=seen))
    lane['j_pattern']=pattern
    return lane


def caller_phase4(lane,constellation,events,limit):
    follower=lane['phase3'];eq=follower.eq
    scanner=rx.JPrimeScanner(follower.idx,'call',follower.j[3],follower.j[2])
    while scanner.found is None and eq.k<eq.available() and symbol_sample(lane,eq.k)<limit:
        y=eq.step(decide=rx.decide4)
        follower.idx.append(int(round(-np.angle(y/tx.P0)/(math.pi/2)))%4)
        if len(follower.idx)%8==0:scanner.update(follower.idx)
    if scanner.found is None:raise ValueError('call: no observed J-prime')
    jp=follower.k_trn+scanner.found
    events.append(event('call','J-prime',symbol_sample(lane,jp),symbol_sample(lane,jp+8)))
    receiver=rx.Phase4Receiver(lane['R'],jp+8,'call',eq=eq,constellation=constellation)
    receiver.feed(None);lane['phase4']=receiver
    return lane


def answer_phase4(data,lane,constellation,caller,hi,events):
    baud,fc=lane['R'].baud,lane['R'].fc
    # Bound by recorded phase-three J and the recorded caller's training start.
    # Include pulse/FIR context; use the recorded S-bar/TRN correlation itself.
    lo=max(symbol_sample(lane,432+lane['phase3'].j[0]),
           symbol_sample(caller,0)-FS*.1)
    offset=max(0,int(lo)-80)
    x=data[offset:int(hi),1].astype(float)
    r=rx.Phase3Rx(baud,fc,'answer')
    trn,_=tx.trn_symbols(800,'answer',constellation=constellation)
    seq=np.r_[tx.s_signal(16,True),trn[:200]]
    got=r.acquire(x,search=min(len(x),10*FS),seq=seq,min_metric=.7)
    if got is None:raise ValueError('answer: no observed phase-four S-bar/TRN')
    r._z=r.update_baseband(x)
    mse=r.train_known(np.r_[tx.s_signal(16,True),trn[:700]],0,
                      span=rx.phase4_equalizer_span())
    receiver=rx.Phase4Receiver(r,16,'answer',constellation=constellation)
    if baud>=3200:
        receiver.eq.ktp,receiver.eq.kti,receiver.eq.mu=.003,5e-8,.1
    receiver.feed(None)
    p4=dict(R=r,offset=offset,phase4=receiver,role='answer')
    sbar=symbol_sample(p4,0)
    start=max(0,int(symbol_sample(caller,432+caller['phase3'].j[0])-FS*.1))
    for at in range(start,max(start+1,int(sbar)),FS//8):
        segment=data[at:min(int(sbar)+80,at+FS//4),1]
        found=rx.detect_s_flip(segment,fc,baud)
        if found is not None:
            sample=at+found[1]
            events.append(event('answer','S' if found[0]=='S' else 'S-to-S-bar',sample,
                phase=4,observed=True,timing_uncertainty_samples=r.spl*3))
            break
    events.append(event('answer','S-bar',symbol_sample(p4,0),symbol_sample(p4,16),
                        phase=4,correlation=got['metric'],training_nmse=mse))
    return p4


def phase4_events(lane,events):
    rcv=lane['phase4'];sender=lane['role'];bpn=4 if rcv.c16 else 2
    if rcv.trn_end is None:raise ValueError(f'{sender}: no CRC-valid MP after training')
    events.append(event(sender,'TRN',symbol_sample(lane,rcv.k_trn),
        symbol_sample(lane,rcv.k_trn+rcv.trn_end),phase=4))
    for span in rcv.mp_spans:
        events.append(event(sender,"MP-prime" if span['fields']['ack'] else 'MP',
            symbol_sample(lane,rcv.k_trn+rcv.trn_end+span['start_bit']/bpn),
            symbol_sample(lane,rcv.k_trn+rcv.trn_end+span['end_bit']/bpn),
            fields=span['fields'],crc_valid=True,bit_span=[span['start_bit'],span['end_bit']]))
    if rcv.e_span is None:raise ValueError(f'{sender}: no observed E')
    span=rcv.e_span
    events.append(event(sender,'E',symbol_sample(lane,rcv.k_trn+rcv.trn_end+span['start_bit']/bpn),
        symbol_sample(lane,rcv.b1_symbol),bit_span=[span['start_bit'],span['end_bit']],
        processing_symbol=span['processing_symbol'],b1_symbol=rcv.b1_symbol))
    mp=negotiation.final_mp(rcv.frames)
    if mp is None or not mp.get('ack'):raise ValueError(f'{sender}: no acknowledged MP before E')
    return mp


def qualify(data,lane,other_mp,own_mp,parameters,diagnostic=False,diagnostic_axis='combined'):
    sender=lane['role'];other='answer' if sender=='call' else 'call'
    remote_baud,remote_fc=parameters[sender];local_baud,local_fc=parameters[other]
    negotiated=negotiation.resolve(other_mp,own_mp,other,
        local_baud,remote_baud,local_fc,remote_fc)
    engine=SimpleNamespace(negotiated=negotiated,_local_mp=other_mp,
        mp_ans=own_mp if sender=='answer' else None,mp_call=own_mp if sender=='call' else None,
        _receive_trace=True,_p4_x0=lane['offset'],
        _rx=data[:,ROLES.index(sender)].astype(float),_rx_b1=lane['phase4'].b1_symbol,
        rx_qualification={},failed_reason=None,_ev=lambda message:None)
    eq=deepcopy(lane['phase4'].eq)
    policy=receive.SearchPolicy.diagnostic(diagnostic_axis) if diagnostic else receive.SearchPolicy()
    ready=receive.initialize(engine,eq,sender,remote_baud,policy=policy)
    if not engine.rx_qualification:
        engine.rx_qualification=dict(profile=asdict(negotiated.rx),qualified=False,reason='truncated B1/context')
    return dict(status='PASS' if ready else 'FAIL',qualification=engine.rx_qualification,
        reason=engine.failed_reason or (None if ready else 'truncated B1/context'),
        inputs=getattr(engine,'_b1_inputs',None))


def stage_evidence(inputs,qualification,sender):
    """Received decisions compared with independent domains, never substituted."""
    if inputs is None or 'observed_after' not in inputs or 'bit_errors' not in qualification:
        return dict(status='unidentified',reason=qualification.get('reason','no decodable B1 profile'))
    profile=negotiation.DirectionParameters(**qualification['profile'])
    state=qualification['initial_trellis_state']
    expected=b1_reference(profile.rate,profile.baud,sender,profile.shaping,profile.precoder,state,trace=True)
    decoder=receive.Receiver(profile,sender,state)
    g,ref=decoder.qualify(inputs['observed_after'])
    detector=decoder.detector();detector.energy=decoder.energy
    for a,b in (inputs['observed_after']/g).reshape(-1,2):detector.feed(a,b)
    detector.flush()
    points=np.array([p for pair in detector.decided for p in pair])
    differences=np.flatnonzero(points!=np.asarray(expected['u']))
    bits=np.flatnonzero(np.asarray(decoder.bits)!=np.asarray(expected['bits']))
    normalized=inputs['observed_after']/g
    residual=abs(normalized-ref)
    linearized=receive.inverse_projection(normalized,decoder.energy) if profile.nonlinear else normalized
    history=[0j]*3;line=[];cs=[];states=[]
    conv=mapper.ConvEncoder16();conv.d=list(state)
    for index,u in enumerate(points):
        if index%2==0:states.append(tuple(conv.d))
        prediction=mapper._round_complex(sum(x*h for x,h in zip(history,decoder.h)),2**-7)
        c=mapper._round_complex(prediction,2**decoder.tx.w)
        x=u+c-prediction;line.append(x);cs.append(c);history=[x]+history[:2]
        if index%2:
            label=mapper.TABLE13[mapper.subset_label(points[index-1]+cs[-2])][mapper.subset_label(u+c)]
            conv.step(label)
    state_errors=np.flatnonzero(np.any(np.asarray(states)!=np.asarray(expected['states']),axis=1))
    line_errors=np.flatnonzero(np.asarray(line)!=np.asarray(expected['x']))
    shift=int(inputs['chosen_shift']);centers=inputs['centers'][shift:shift+len(points)]
    for row in expected['alignment']:
        first=2*row['interval_4d']
        row['source_sample_centers']=centers[first:first+2].tolist()
        row['received_u']=[[float(v.real),float(v.imag)] for v in points[first:first+2]]
        row['received_linearized']=[[float(v.real),float(v.imag)] for v in linearized[first:first+2]]
        row['received_state_before']=list(states[row['interval_4d']])
        lo,hi=row['pair_bit_span'];row['received_pair_bits']=decoder.bits[lo:hi]
        slo,shi=row['shell_bit_span'];row['received_shell_bits']=decoder.bits[slo:shi]
    # Waveform error is continuous: preserve all residuals instead of inventing
    # an arbitrary threshold that declares an analog symbol exactly correct.
    return dict(status='measured',mapper_symbol_errors=len(differences),
        first_mapper_symbol_error=int(differences[0]) if len(differences) else None,
        first_bit_error=int(bits[0]) if len(bits) else None,
        precoder_line_symbol_errors=len(line_errors),
        first_precoder_line_symbol_error=int(line_errors[0]) if len(line_errors) else None,
        trellis_state_errors=len(state_errors),
        first_trellis_state_error=int(state_errors[0]) if len(state_errors) else None,
        linearized_residual_by_symbol=abs(linearized-np.asarray(expected['x'])).tolist(),
        specification=expected['specification'],framing=expected['parameters'],
        equation_alignment=expected['alignment'],
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_by_symbol=residual.tolist(),
        source_sample_centers=inputs['centers'][int(inputs['chosen_shift']):
            int(inputs['chosen_shift'])+len(points)].tolist(),
        calibrated_on_first_half=True,oracle_substitution=False)


def analyze(data,source_hash=None,chunk_samples=160,channel_order=ROLES,diagnostic_axis='combined'):
    """Both channels share sample zero. Acquisition uses fixed offline buffers.

    chunk_samples describes ingestion only; analysis never uses delivery chunks
    as physical event boundaries. This prevents local scheduling from moving a
    recorded transition or changing which protocol attempt owns a frame.
    """
    if chunk_samples<1 or set(channel_order)!=set(ROLES):raise ValueError('invalid ingestion options')
    data=np.asarray(data)
    if data.ndim!=2 or data.shape[1]!=2:raise ValueError('requires two synchronized channels')
    infos=information(data);events=list(infos)+prelude(data);attempts=[];traces={}
    pairs=pair_attempts(infos)
    for index,(c,a) in enumerate(pairs):
        attempt=dict(index=index,start_sample=c['start_sample'],status='FAIL',directions={})
        start_event=len(events)
        try:
            cf,af=c['evidence']['fields'],a['evidence']['fields']
            cb,ab=tx.SYMBOL_RATES[af['sym_call']],tx.SYMBOL_RATES[af['sym_ans']]
            parameters=dict(call=(cb,tx.CARRIERS[round(cb)][af['hi_carrier']]),
                answer=(ab,tx.CARRIERS[round(ab)][cf['blocks'][af['sym_ans']]['hi_carrier']]))
            hi=int(pairs[index+1][0]['start_sample']) if index+1<len(pairs) else len(data)
            lanes={}
            for sender in channel_order:
                lanes[sender]=train_phase3(data,sender,parameters,a['end_sample'],hi,events)
            caller=caller_phase4(lanes['call'],16 if lanes['answer']['j_pattern']=='J_16PT' else 4,events,hi)
            answer=answer_phase4(data,lanes['answer'],16 if lanes['call']['j_pattern']=='J_16PT' else 4,caller,hi,events)
            lanes=dict(call=caller,answer=answer)
            messages={sender:phase4_events(lanes[sender],events) for sender in ROLES}
            for sender in ROLES:
                other='answer' if sender=='call' else 'call'
                strict=qualify(data[:hi],lanes[sender],messages[other],messages[sender],parameters)
                diagnostic=qualify(data[:hi],lanes[sender],messages[other],messages[sender],parameters,True,diagnostic_axis)
                strict['stages']=stage_evidence(strict['inputs'],strict['qualification'],sender)
                diagnostic['stages']=stage_evidence(diagnostic['inputs'],diagnostic['qualification'],sender)
                traces[f'{index}.{sender}.standard']=strict.pop('inputs')
                traces[f'{index}.{sender}.diagnostic']=diagnostic.pop('inputs')
                attempt['directions'][sender]=dict(standard=strict,diagnostic=diagnostic)
                n=8*frame_params(strict['qualification']['profile']['rate'],parameters[sender][0])['P']
                b1=lanes[sender]['phase4'].b1_symbol
                events.append(event(sender,'B1',symbol_sample(lanes[sender],b1),symbol_sample(lanes[sender],b1+n),
                    standard=strict,diagnostic=diagnostic,
                    complete_in_recording=symbol_sample(lanes[sender],b1+n)<=hi))
                if symbol_sample(lanes[sender],b1+n)<=hi:
                    events.append(event(sender,'data-boundary',symbol_sample(lanes[sender],b1+n),
                        payload_verified=False,b1_verified=strict['status']=='PASS'))
            attempt['status']='PASS' if all(v['standard']['status']=='PASS' for v in attempt['directions'].values()) else 'FAIL'
        except (ValueError,KeyError,IndexError) as exc:
            attempt['reason']=str(exc)
        for observed in events[start_event:]:observed['attempt']=index
        c['attempt']=index;a['attempt']=index
        attempt['event_count']=len(events)-start_event
        attempts.append(attempt)
    relationships=[]
    for attempt in attempts:
        index=attempt['index'];begin=pairs[index][0]['start_sample']
        end=pairs[index+1][0]['start_sample'] if index+1<len(pairs) else len(data)
        selected=[e for e in events if begin<=e['start_sample']<end]
        def last(sender,kinds,before=float('inf')):
            choices=[e for e in selected if e['sender']==sender and e['kind'] in kinds and e['end_sample']<=before]
            return max(choices,key=lambda e:e['end_sample']) if choices else None
        jp=last('call',('J-prime',))
        answer_sbar=[e for e in selected if e['sender']=='answer' and e['kind']=='S-bar' and e['evidence'].get('phase')==4]
        if jp and answer_sbar:
            sbar=answer_sbar[-1]
            relationships.append(dict(attempt=index,kind='recorded phase-four timing',
                call_jprime_start_sample=jp['start_sample'],answer_sbar_start_sample=sbar['start_sample'],
                answer_sbar_minus_call_jprime_samples=sbar['start_sample']-jp['start_sample'],
                causal_order_assumed=False,transport_delay_corrected=False))
        for sender in ROLES:
            opposite='answer' if sender=='call' else 'call'
            for current in [e for e in selected if e['sender']==sender and e['kind']=='MP-prime']:
                previous=last(opposite,('MP',),current['start_sample'])
                if previous:
                    relationships.append(dict(attempt=index,kind='MP acknowledgement',sender=sender,
                        request_sender=opposite,request_start_sample=previous['start_sample'],
                        acknowledgement_start_sample=current['start_sample'],
                        separation_samples=current['start_sample']-previous['start_sample'],
                        transport_delay_corrected=False))
            e=last(sender,('E',));b1=last(sender,('B1',))
            if e and b1:
                relationships.append(dict(attempt=index,kind='E to B1',sender=sender,
                    e_end_sample=e['end_sample'],b1_start_sample=b1['start_sample'],
                    contiguous=abs(e['end_sample']-b1['start_sample'])<1e-6))
    
    for i,e in enumerate(sorted(events,key=lambda e:(e['start_sample'],e['sender'],e['kind']))):
        e['id']=i
    events.sort(key=lambda e:(e['start_sample'],e['sender'],e['kind']))
    return dict(sample_rate=FS,samples=len(data),audio_sha256=source_hash,
        channel_map=dict(call=0,answer=1),clock='original shared WAV sample index',
        simulated_transmitter=False,events=events,relationships=relationships,attempts=attempts,
        limitations=[] if attempts else ['no paired CRC-valid INFO1 exchange'],
        status='PASS' if attempts and all(a['status']=='PASS' for a in attempts) else 'FAIL'),traces
