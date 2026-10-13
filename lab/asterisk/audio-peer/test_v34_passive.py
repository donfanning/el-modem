"""Passive duplex observation: original clock, recorded parameters, no local TX."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import wave
import numpy as np
import v34_tx as tx
from v34_info import InfoModulator,build_info1c,build_info1a
from v34_mp import build_mp
from v34_data import DataTx
from v34_passive import analyze,load_stereo,event,pair_attempts


def recording(call_rate=9600,answer_rate=7200,call_cap=4,answer_cap=3,asymmetric=1):
    baud=tx.SYMBOL_RATES[5];fc=tx.CARRIERS[3429][0]
    pcm=np.zeros((32000,2))
    def put(col,start,x):pcm[round(start*8000):round(start*8000)+len(x),col]+=x
    blocks=[dict(hi_carrier=0,preemph=0,max_rate=call_cap) for _ in range(6)]
    put(0,.4,InfoModulator('call',amplitude=1500).modulate(build_info1c(blocks=blocks)))
    put(1,.7,InfoModulator('answer',amplitude=1500).modulate(build_info1a(max_rate=call_cap,sym_ans=5,sym_call=5)))
    mask=(1<<max(call_cap,answer_cap))-1
    requests=dict(call=dict(max_c2a=call_cap,max_a2c=answer_cap,mask=mask,asym=asymmetric),
                  answer=dict(max_c2a=call_cap,max_a2c=max(call_cap,answer_cap),mask=mask,asym=asymmetric))
    expected={}
    for role,col,start in [('answer',1,1.),('call',0,1.8)]:
        trn,sc=tx.trn_symbols(800,role)
        j,z=tx.j_symbols(tx.J_16PT,8,role,sc,tx.trn_last_rotation(trn,4))
        prefix=np.r_[tx.s_signal(128),tx.s_signal(16,True),tx.pp_signal(),trn,j]
        if role=='answer':
            put(col,start,tx.qam_modulate(prefix,baud,fc,1500))
            prefix=np.r_[tx.s_signal(128),tx.s_signal(16,True)]
            start=2.3
        else:
            jp,z=tx.j_symbols(tx.J_PRIME,1,role,sc,z)
            prefix=np.r_[prefix,jp]
        training,sc=tx.trn_symbols(1000,role,constellation=16)
        z=tx.trn_last_rotation(training,16);parts=[prefix,training]
        for ack in [0]*10+[1]*3:
            symbols,z=tx.bit_symbols(build_mp(1,**requests[role],ack=ack),role,sc,z,constellation=16)
            parts.append(symbols)
        e,z=tx.bit_symbols([1]*20,role,sc,z,constellation=16);parts.append(e)
        rate=call_rate if role=='call' else answer_rate
        mapper=DataTx(rate,baud);bsc=tx.Scrambler(role)
        b1=np.array(mapper.data_frame([bsc.step(1) for _ in range(mapper.bits_per_data_frame)]))
        b1*=np.sqrt(10/mapper.avg_power)
        before=sum(len(part) for part in parts)
        expected[role]=dict(rate=rate,b1_center=start*8000+(8+before)*8000/baud)
        symbols=np.r_[*parts,b1,b1]
        put(col,start,tx.qam_modulate(symbols,baud,fc,1500,power=10))
    return np.rint(pcm).astype('<i2'),expected

class PassiveTests(unittest.TestCase):
    def test_nonmaximum_rate_is_diagnostic_only(self):
        pcm,_=recording(call_cap=5)
        result,_=analyze(pcm,diagnostic_axis='rate')
        d=result['attempts'][0]['directions']['call']
        self.assertEqual(d['standard']['qualification']['profile']['rate'],12000)
        self.assertEqual(d['standard']['status'],'FAIL')
        q=d['diagnostic']['qualification']
        self.assertEqual(q['profile']['rate'],9600)
        self.assertEqual(q['bit_errors'],0)
        self.assertEqual(q['initial_trellis_state'],[0,0,0,0])
        self.assertEqual(q['boundary_offset'],0)
        self.assertFalse(q['standard_qualification'])

    def test_symmetric_rate_uses_both_directional_limits(self):
        pcm,_=recording(call_rate=7200,asymmetric=0)
        result,_=analyze(pcm)
        self.assertEqual(result['status'],'PASS')
        for d in result['attempts'][0]['directions'].values():
            self.assertEqual(d['standard']['qualification']['profile']['rate'],7200)

    def test_shared_stereo_ingestion(self):
        samples=np.arange(2000,dtype='<i2').reshape(-1,2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stereo.wav'
            with wave.open(str(path),'wb') as w:
                w.setparams((2,2,8000,len(samples),'NONE','not compressed'));w.writeframes(samples.tobytes())
            for chunk in (80,160,320):
                actual,_=load_stereo(path,chunk);np.testing.assert_array_equal(actual,samples)

    def test_pairing_uses_same_observed_attempt(self):
        events=[event('call','info1c',100,200),event('answer','info1a',300,400),
                event('call','info1c',1000,1100),event('answer','info1a',1200,1300)]
        pairs=pair_attempts(events)
        self.assertEqual([(c['start_sample'],a['start_sample']) for c,a in pairs],[(100,300),(1000,1200)])

    def test_missing_negotiation_is_not_synthesized(self):
        result,_=analyze(np.zeros((2000,2),dtype='<i2'))
        self.assertEqual(result['status'],'FAIL');self.assertEqual(result['attempts'],[])

    def test_truncated_b1_cannot_pass_or_invent_data(self):
        pcm,expected=recording()
        cut=round(expected['answer']['b1_center']+40)
        result,_=analyze(pcm[:cut])
        self.assertEqual(result['status'],'FAIL')
        self.assertFalse(any(e['kind']=='data-boundary' and e['sender']=='answer'
                             for e in result['events']))
        self.assertEqual(result['attempts'][0]['directions']['answer']['standard']['status'],'FAIL')

    def test_retry_with_missing_answer_does_not_borrow_future_info(self):
        observed=[event('call','info1c',100,200),event('call','info1c',1000,1100),
                  event('answer','info1a',1200,1300)]
        pairs=pair_attempts(observed)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0][0]['start_sample'],1000)

    def test_original_stereo_excerpt_on_shared_clock(self):
        source=Path(__file__).parent/'fixtures/v34-reference/passive/ktx-stereo-handshake.wav'
        pcm,digest=load_stereo(source,80)
        result,_=analyze(pcm,digest,80)
        reverse_pcm,reverse_digest=load_stereo(source,320)
        reverse,_=analyze(reverse_pcm,reverse_digest,320,('answer','call'))
        self.assertEqual(result,reverse)
        attempt=result['attempts'][0]
        self.assertNotIn('reason',attempt)
        self.assertEqual(attempt['directions']['answer']['standard']['qualification']['bit_errors'],0)
        self.assertEqual(attempt['directions']['call']['diagnostic']['qualification']['bit_errors'],0)
        self.assertEqual(attempt['directions']['call']['standard']['status'],'FAIL')
        self.assertTrue(any(r['kind']=='E to B1' and r['contiguous']
                            for r in result['relationships']))
        for sender,seconds in [('answer',16.8680416667),('call',17.0059270833)]:
            b1=next(e for e in result['events'] if e['kind']=='B1' and e['sender']==sender)
            self.assertAlmostEqual(b1['start_sample']+79000,seconds*8000,delta=4)

    def test_two_recorded_attempts_keep_separate_boundaries(self):
        pcm,_=recording()
        combined=np.r_[pcm,np.zeros((8000,2),dtype='<i2'),pcm]
        result,_=analyze(combined)
        self.assertEqual([a['status'] for a in result['attempts']],['PASS','PASS'])
        for sender in ('call','answer'):
            boundaries=[e for e in result['events'] if e['kind']=='B1' and e['sender']==sender]
            self.assertEqual([e['attempt'] for e in boundaries],[0,1])
            self.assertAlmostEqual(boundaries[1]['start_sample']-boundaries[0]['start_sample'],40000,delta=3)
        repeated=[e for e in result['events'] if e['kind']=='MP' and e['attempt']==0]
        self.assertGreater(len(repeated),2)

    def test_known_stereo_handshake_and_directional_rates(self):
        pcm,expected=recording()
        with patch('v34_call.V34Caller',side_effect=AssertionError('simulated caller')), \
                patch('v34_ans.V34Answerer',side_effect=AssertionError('simulated answer')):
            result,_=analyze(pcm)
        self.assertEqual(len(result['attempts']),1,result)
        attempt=result['attempts'][0]
        self.assertNotIn('reason',attempt,attempt)
        for sender,values in attempt['directions'].items():
            q=values['standard']['qualification']
            self.assertEqual(q['profile']['rate'],expected[sender]['rate'])
            self.assertEqual(q.get('bit_errors'),0,q)
            stages=values['standard']['stages']
            for key in ('mapper_symbol_errors','precoder_line_symbol_errors','trellis_state_errors'):
                self.assertEqual(stages[key],0,stages)
            b1=next(e for e in result['events'] if e['sender']==sender and e['kind']=='B1')
            self.assertAlmostEqual(b1['start_sample'],expected[sender]['b1_center'],delta=3)
        self.assertEqual(result['status'],'PASS')
        reverse,_=analyze(pcm,channel_order=('answer','call'),chunk_samples=320)
        self.assertEqual(result,reverse)

if __name__=='__main__':unittest.main()
