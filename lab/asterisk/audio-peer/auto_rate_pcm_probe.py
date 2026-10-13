#!/usr/bin/env python3
"""Causal own-DSP development channel, not independent or real-time gates.

Noise is introduced after verified baseline traffic, only in the DUT receive
direction. The policy gets received-byte activity, decoder metrics and CRC
counters, never impairment schedule or expected payload. Short dwell settings
exercise mechanics; production dwell remains 60 seconds.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import struct

import numpy as np
from v34_modem import V34Modem
from v34_answer_modem import V34AnswerModem
from v34_online_control import OnlineControl
from v34_auto_rate import AutomaticRate, RatePolicyConfig


def run(role='call', snr_db=30., seed=3522, protocol='lapm', noise_seconds=6.):
    os.environ.update(V34_CONNECTED_RATE='1',V34_AUTO_RATE='1',V34_PHASE4_EQ_SPAN='24',
                      V34_NATIVE_TRELLIS='1',V34_SURVIVOR_STATE_SYNC='1',
                      V34_CALLER_RETRAIN='1',V34_ANSWERER_RETRAIN='1')
    caller,answer=V34Modem(data_rate=33600),V34AnswerModem(data_rate=33600)
    dut=caller if role=='call' else answer
    owner=OnlineControl(lambda:dut)
    config=RatePolicyConfig(up_seconds=5.,trial_seconds=3.,cooldown_seconds=1.)
    policy=AutomaticRate(owner,config)
    c_rx,a_rx=bytearray(),bytearray()
    rng=np.random.default_rng(seed+10000)
    reference=[]
    noise_sigma=None
    snapshots=[]
    last_monitor=None
    quality_reports=[]
    logical_time=0.
    loaded_until=0.
    completions=[]

    def tick(noisy=False, automatic=False):
        nonlocal logical_time,loaded_until,last_monitor
        ct=np.asarray(caller.tx_samples(160),dtype=np.int16)
        at=np.asarray(answer.tx_samples(160),dtype=np.int16)
        incoming=at if role=='call' else ct
        if noise_sigma is None and dut.connected:
            reference.extend(incoming.astype(float).tolist())
            del reference[:-8000]
        if noisy:
            x=incoming.astype(float)+rng.normal(0,noise_sigma,len(incoming))
            if np.any(abs(x)>32767):raise RuntimeError('noise fixture clipped PCM')
            incoming=np.rint(x).astype(np.int16)
            if role=='call':at=incoming
            else:ct=incoming
        ar=answer.feed_pcm(struct.pack('<160h',*ct))
        cr=caller.feed_pcm(struct.pack('<160h',*at))
        a_rx.extend(ar);c_rx.extend(cr)
        logical_time+=.02
        if cr if role=='call' else ar:loaded_until=logical_time+.5
        core=dut.caller if role=='call' else dut.answerer
        monitor=(getattr(core,'_rx_state',None) or {}).get('quality')
        if monitor is not None and monitor is not last_monitor:
            if last_monitor is not None:quality_reports.append(last_monitor.report())
            last_monitor=monitor
        if automatic:
            result=policy.tick(active=True,loaded=logical_time<loaded_until,now=logical_time)
            if result is not None:completions.append(result)
        if dut.state=='failed':raise RuntimeError(dut.failed_reason)
        if caller.state=='failed' or answer.state=='failed':raise RuntimeError('peer failed')

    for _ in range(1500):
        tick()
        if caller.connected and answer.connected:break
    if not caller.connected or not answer.connected:raise RuntimeError('startup did not connect')
    if protocol=='lapm':
        caller.start_v42(True,rate=33600);answer.start_v42(False,rate=33600)
        for _ in range(500):
            tick()
            if caller.v42_mode==answer.v42_mode=='lapm':break
        if caller.v42_mode!=answer.v42_mode or caller.v42_mode!='lapm':raise RuntimeError('LAPM not established')
    bits=random.Random(seed)
    c_head=bytes(bits.randrange(256) for _ in range(4096))
    a_head=bytes(bits.randrange(256) for _ in range(4096))
    caller.feed_bytes(c_head);answer.feed_bytes(a_head)
    for _ in range(600):
        tick()
        if len(c_rx)>=4096 and len(a_rx)>=4096:break
    if bytes(c_rx)!=a_head or bytes(a_rx)!=c_head:raise RuntimeError('baseline payload mismatch')
    if not reference:raise RuntimeError('no measured signal reference')
    signal_power=float(np.mean(np.square(reference)))
    noise_sigma=(signal_power/10**(snr_db/10))**.5
    c_data=bytes(bits.randrange(256) for _ in range(131072))
    a_data=bytes(bits.randrange(256) for _ in range(131072))
    caller.feed_bytes(c_data);answer.feed_bytes(a_data)
    for i in range(5000):
        # One second clean, six seconds impaired, then recovery. Coordinates
        # and transmitted samples are preserved; no erasure or retiming.
        tick(noisy=1<=i*.02<1+noise_seconds,automatic=True)
        if i%50==0:
            snapshots.append(dict(seconds=i*.02,tx_rate=dut.bit_rate,rx_rate=dut.rx_bit_rate,
                reason=policy.reason,good_seconds=policy.good_seconds,bad_seconds=policy.bad_seconds,
                quality=last_monitor.report()['path_cost_per_symbol'] if last_monitor else None,
                blocks=dict(getattr(getattr(dut,'_v42',None),'stats',{}))))
        if len(c_rx)>=len(a_head+a_data) and len(a_rx)>=len(c_head+c_data):break
    if last_monitor is not None:quality_reports.append(last_monitor.report())
    return dict(scope='own-DSP causal PCM development; no independent/realtime claim',
        role=role,protocol=protocol,seed=seed,snr_db=snr_db,signal_power_pcm_squared=signal_power,
        noise_component_rms_pcm=noise_sigma,config=asdict(config),
        caller_bytes=len(c_rx),answer_bytes=len(a_rx),
        noise_seconds=noise_seconds,expected_bytes=135168,
        caller_payload_exact=bytes(c_rx)==a_head+a_data,answer_payload_exact=bytes(a_rx)==c_head+c_data,
        caller_prefix_exact=bytes(c_rx)==(a_head+a_data)[:len(c_rx)],
        answer_prefix_exact=bytes(a_rx)==(c_head+c_data)[:len(a_rx)],
        caller_clean_tail_exact=bytes(c_rx).endswith(a_data[-4096:]),
        answer_clean_tail_exact=bytes(a_rx).endswith(c_data[-4096:]),
        completed=completions,policy=policy.report(),quality_epochs=quality_reports,snapshots=snapshots,
        final_rates=dict(tx=dut.bit_rate,rx=dut.rx_bit_rate),
        engine_events=list(dut.events),channel_truth_passed_to_policy=False,external_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['call','answer'],default='call')
    p.add_argument('--protocol',choices=['lapm','raw'],default='lapm')
    p.add_argument('--snr-db',type=float,default=30.)
    p.add_argument('--seed',type=int,default=3522)
    p.add_argument('--noise-seconds',type=float,default=6.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.noise_seconds<=0:p.error('noise duration must be positive')
    result=run(a.role,a.snr_db,a.seed,a.protocol,a.noise_seconds)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['role','snr_db','caller_payload_exact','answer_payload_exact','final_rates']}))


if __name__=='__main__':main()
