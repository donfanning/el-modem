#!/usr/bin/env python3
"""Paired El-Modem DCE/RTP probe with controlled loopback packet loss.

Each OnlineAudioSession runs in a separate process so its RTP sender and DSP
do not contend for one interpreter lock. A local UDP relay drops a configured
number of contiguous PCMU packets in each direction after LAPM starts.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import selectors
import socket
import time


def _modem_worker(conn, local_port, relay_port, role, rate, recovery):
    for key in ('V34_GAP_FEEDBACK','V34_GAP_TRELLIS_ERASURE','V34_GAP_TRELLIS_HOLD'):
        os.environ.pop(key,None)
    os.environ['V34_GAP_RECOVERY']='1' if recovery else '0'
    from modem_dce import OnlineAudioSession

    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1',local_port));sock.settimeout(.003)
    tx_role='originate' if role=='caller' else 'answer'
    rx_role='answer' if role=='caller' else 'originate'
    session=OnlineAudioSession(sock,('127.0.0.1',relay_port),tx_role,rx_role,
        line_mode='v34',v34_rate=rate,v34_min_rate=rate,v42=True,
        carrier_loss_seconds=15.0)
    rx=bytearray();payload=None;expected_payload=None;stopping=False
    last_status=0.0
    try:
        session.start()
        while not stopping:
            now=time.monotonic();session.poll(now)
            while conn.poll():
                command=conn.recv()
                if command[0]=='send':
                    payload,expected_payload=command[1],command[2]
                    session.feed_terminal_bytes(payload)
                elif command[0]=='stop':
                    stopping=True
            try:packet,source=sock.recvfrom(4096)
            except socket.timeout:packet=None
            if packet:
                rx.extend(session.handle_rtp_datagram(source,packet))
            if now-last_status>=.5:
                modem=session._v22
                core=getattr(modem,'caller',None) or getattr(modem,'answerer',None)
                eq=getattr(getattr(core,'_p4rcv',None),'eq',None)
                guard=getattr(eq,'_transport_guard',None)
                link=getattr(modem,'_v42',None)
                conn.send(dict(role=role,connected=bool(getattr(modem,'connected',False)),
                    v34_state=getattr(modem,'state',None),
                    tx_rate=getattr(modem,'bit_rate',None),rx_rate=getattr(modem,'rx_bit_rate',None),
                    v42_mode=getattr(modem,'v42_mode',None),lapm_state=getattr(link,'state',None),
                    lapm_stats=dict(getattr(link,'stats',{}) or {}),
                    lapm_events=list(getattr(link,'events',())[-24:]),
                    lapm_detail=({
                        'va': getattr(link,'va',None), 'vs': getattr(link,'vs',None),
                        'vr': getattr(link,'vr',None),
                        'unacked_frames': len(getattr(link,'_unacked',())),
                        'retransmit_frames': len(getattr(link,'_retx',())),
                        'queued_dte_bytes': len(getattr(link,'_txq',())),
                        'pending_output_bytes': len(getattr(link,'_out',())),
                        'poll_out': getattr(link,'_poll_out',None),
                        'retry_count': getattr(link,'_tries',None),
                        'peer_busy': getattr(link,'_peer_busy',None),
                        'local_busy': getattr(link,'_local_busy',None),
                    } if link is not None else None),
                    b1_errors=(getattr(core,'rx_qualification',{}) or {}).get('bit_errors'),
                    timing_slips=getattr(eq,'slips',None),
                    gap_feedback=guard.report() if guard is not None else None,
                    received_bytes=len(rx),received_sha256=hashlib.sha256(rx).hexdigest(),
                    expected_bytes=len(expected_payload) if expected_payload is not None else 0,
                    expected_sha256=(hashlib.sha256(expected_payload).hexdigest()
                                     if expected_payload is not None else None),
                    exact_payload=expected_payload is not None and rx==expected_payload,
                    rtp=dict(session._rtp_counts),events=list(getattr(modem,'events',())[-8:])))
                last_status=now
    finally:
        session.stop();sock.close();conn.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery',choices=('off','on'),required=True)
    parser.add_argument('--rate',type=int,default=33600)
    parser.add_argument('--bytes',type=int,default=32768)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--erasure-ms',type=int,default=80)
    parser.add_argument('--caller-drop-after-ms',type=int,default=2000)
    parser.add_argument('--answerer-drop-after-ms',type=int,default=3000)
    parser.add_argument('--timeout-seconds',type=float,default=90)
    parser.add_argument('--output')
    args=parser.parse_args()
    if (args.rate<=0 or args.bytes<=0 or args.erasure_ms<0 or args.erasure_ms%20
            or args.caller_drop_after_ms<0 or args.caller_drop_after_ms%20
            or args.answerer_drop_after_ms<0 or args.answerer_drop_after_ms%20
            or args.timeout_seconds<=0):
        parser.error('positive rate/bytes/timeout and nonnegative 20-ms-aligned loss settings required')

    ctx=mp.get_context('spawn')
    relay_socks=[];children=[];pipes=[]
    selector=selectors.DefaultSelector()
    for _ in range(2):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.bind(('127.0.0.1',0));sock.setblocking(False)
        relay_socks.append(sock);selector.register(sock,selectors.EVENT_READ)
    ports=[s.getsockname()[1] for s in relay_socks]
    modem_ports=[]
    for _ in range(2):
        probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        probe.bind(('127.0.0.1',0));modem_ports.append(probe.getsockname()[1]);probe.close()
    for role,index in (('caller',0),('answerer',1)):
        parent,child=ctx.Pipe()
        proc=ctx.Process(target=_modem_worker,args=(child,modem_ports[index],ports[index],
            role,args.rate,args.recovery=='on'))
        proc.start();child.close();pipes.append(parent);children.append(proc)

    sources=[None,None];drop_at={'c2a':None,'a2c':None}
    dropped={'c2a':0,'a2c':0};dropped_goal=args.erasure_ms//20
    relayed={'c2a':0,'a2c':0};latest={}
    data_start=None;result_error=None
    payloads=[random.Random(args.seed).randbytes(args.bytes),
              random.Random(args.seed+1).randbytes(args.bytes)]
    deadline=time.monotonic()+args.timeout_seconds
    try:
        while time.monotonic()<deadline:
            for key,events in selector.select(.004):
                side=0 if key.fileobj is relay_socks[0] else 1
                packet,source=key.fileobj.recvfrom(4096)
                sources[side]=source
                direction='c2a' if side==0 else 'a2c'
                destination=sources[1-side]
                if direction in drop_at and drop_at[direction] is not None and dropped[direction]<dropped_goal and time.monotonic()>=drop_at[direction]:
                    dropped[direction]+=1
                    if dropped[direction]==dropped_goal:drop_at[direction]=None
                    continue
                if destination is not None:
                    # Send from the destination modem's relay socket so its
                    # source learning pins replies to its own relay ingress.
                    relay_socks[1-side].sendto(packet,destination)
                    relayed[direction]+=1
            for conn in pipes:
                while conn.poll():
                    row=conn.recv();latest[row['role']]=row
            if data_start is None and all(latest.get(role,{}).get('v42_mode')=='lapm'
                                          for role in ('caller','answerer')):
                for i,conn in enumerate(pipes):conn.send(('send',payloads[i],payloads[1-i]))
                data_start=time.monotonic()
                drop_at={'c2a':data_start+args.caller_drop_after_ms/1000,
                         'a2c':data_start+args.answerer_drop_after_ms/1000}
            if data_start is not None:
                if any(latest.get(role,{}).get('v42_mode') in ('failed','down')
                       for role in ('caller','answerer')):
                    result_error='a paired endpoint left V.42/LAPM'
                    break
                if all(latest.get(role,{}).get('exact_payload') is True
                       for role in ('caller','answerer')) and all(
                           count==dropped_goal for count in dropped.values()):
                    break
            elif any(latest.get(role,{}).get('v34_state')=='failed'
                     for role in ('caller','answerer')):
                result_error='V.34 startup failed'
                break

        if data_start is None and result_error is None:result_error='V.34/V.42 startup timed out'
        if data_start is not None and result_error is None:
            if not all(latest.get(role,{}).get('exact_payload') is True
                       for role in ('caller','answerer')):
                result_error='duplex payload did not complete exactly'
            elif any(count!=dropped_goal for count in dropped.values()):
                result_error='payload completed before all scheduled drops'
        result=dict(status='PASS' if result_error is None else 'FAIL',
            recovery=args.recovery,rate=args.rate,payload_bytes_each_way=args.bytes,seed=args.seed,
            elapsed_seconds=round(time.monotonic()-(data_start or deadline-args.timeout_seconds),2),
            loss=dict(method='loopback UDP RTP relay drops contiguous packets',
                frame_ms=20,duration_ms=args.erasure_ms,
                caller_to_answer_offset_ms=args.caller_drop_after_ms,
                answer_to_caller_offset_ms=args.answerer_drop_after_ms,
                dropped=dict(dropped),relayed=dict(relayed)),
            endpoints={role:latest.get(role) for role in ('caller','answerer')},
            scope='paired El-Modem OnlineAudioSession/DCE, one modem process per endpoint, loopback UDP; no Asterisk, codec transcoding, external route or independent modem',
            failure=result_error)
        encoded=json.dumps(result,indent=2)+'\n'
        if args.output:
            from pathlib import Path
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(encoded)
        print(encoded,end='')
        return 0 if result_error is None else 1
    finally:
        for conn in pipes:
            if conn.closed:continue
            try:conn.send(('stop',))
            except (BrokenPipeError,EOFError,OSError):pass
            conn.close()
        for proc in children:
            proc.join(timeout=2)
            if proc.is_alive():proc.terminate();proc.join(timeout=1)
        selector.close()
        for sock in relay_socks:sock.close()


if __name__=='__main__':
    raise SystemExit(main())
