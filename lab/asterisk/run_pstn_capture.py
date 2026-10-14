#!/usr/bin/env python3
"""One short, fully recorded PSTN startup probe or independent local control.

No login, signup, menus, PPP payloads or automatic retries. The local control
uses a short independently known payload. Both paths preserve failed calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0,str(Path(__file__).parent/'audio-peer'))
from capture_io import TranscriptSocket
from capture_tools import PacketCapture, cli, hash_manifest, inspect_pcap, pcm_inventory
from verify_v34_dsp_capture import verify
from v34_g711 import decode

ROOT=Path(__file__).resolve().parent.parent.parent
BASELINE='sha256:131ebfa935f62982712cd31a0f66a778069f9da7a7e62373c276d1d577382064'


def connection_observed(result):
    if result.get('scope') == 'local control':
        rows=result.get('control_matrix',{}).get('results',[])
        return bool(rows and all(row.get('result',{}).get('dce_connect','').startswith('CONNECT ')
                                for row in rows))
    return bool(result.get('session',{}).get('connected'))


def read_until(connection, predicate, deadline):
    data=bytearray()
    while time.monotonic()<deadline:
        if select.select([connection],[],[],min(.1,max(0,deadline-time.monotonic())))[0]:
            part=connection.recv(65536)
            if not part:
                raise ConnectionError('DTE closed')
            data.extend(part)
            if predicate(bytes(data)):
                break
    return bytes(data)


EMSI_INQ=b'**EMSI_INQC816\r'         # FSC-0056 inquiry (CRC-16 of 'EMSI_INQ'); the answerer's EMSI_REQ carries CRC A77E


def probe(connection, number, observe_seconds=30, whole_seconds=90, prompts=None, replies=None, error_control=3, modulation_profile=None):
    """Observe a session; only Hayes commands and the explicitly listed replies are sent, never login guesses.

    ``replies`` is a list of (regex, bytes): when the regex first appears in the received data the bytes are sent once."""
    result=dict(connected=False,stop_reason=None,payload_verified=False,
                session_scope='startup/banner/prompt observation only')
    online=False
    try:
        commands=['ATZ','ATE0','AT&K0',f'AT\\N{int(error_control)}','AT%C0','ATS0=0']
        if modulation_profile:commands.append('AT+MS='+modulation_profile)
        result['modulation_profile']=modulation_profile
        for command in commands:
            connection.sendall(command.encode()+b'\r')
            answer=read_until(connection,lambda b:b'\r\nOK\r\n' in b or b'ERROR' in b,time.monotonic()+3)
            if b'OK' not in answer:
                raise RuntimeError('DCE initialization rejected or timed out: '+command)
        started=time.monotonic()
        # Reserve ten seconds inside the whole-call limit for escape/hangup.
        active_deadline=started+whole_seconds-10
        connection.sendall(('ATDT'+number+'\r').encode())
        answer=read_until(connection,lambda b:bool(re.search(
            rb'(?:CONNECT[^\r\n]*\r\n|NO CARRIER|NO ANSWER|BUSY|NO DIALTONE|ERROR)',b)),
            min(started+60,active_deadline))
        match=re.search(rb'CONNECT[^\r\n]*\r\n',answer)
        if match is None:
            result.update(stop_reason='startup_result_or_timeout',result_bytes=repr(answer))
            return result
        online=True
        result.update(connected=True,connect_result=match[0].decode('ascii','replace').strip(),
                      connect_seconds=time.monotonic()-started)
        initial=answer[match.end():]
        patterns=[re.compile(value.encode(),re.I) for value in (prompts or [])]
        def finished(data):
            return b'NO CARRIER' in data or any(pattern.search(data) for pattern in patterns)
        received=initial
        if replies:
            result['session_scope']='startup/banner/prompt observation with listed replies (no login guesses)'
            pending=[(re.compile(value.encode() if isinstance(value,str) else value,re.I),data) for value,data in replies]
            sent=result.setdefault('replies_sent',[])
            end=min(time.monotonic()+observe_seconds,active_deadline)
            while time.monotonic()<end and not finished(received):
                received+=read_until(connection,lambda b:False,min(time.monotonic()+.5,end))
                for item in list(pending):
                    if item[0].search(received):
                        connection.sendall(item[1]);pending.remove(item)
                        sent.append(dict(trigger=item[0].pattern.decode('latin1'),sent=item[1].decode('latin1'),
                                         seconds_after_connect=round(time.monotonic()-started-result['connect_seconds'],2),
                                         received_before=len(received)))
                        result['bytes_before_last_reply']=len(received)
        elif not finished(received):
            received+=read_until(connection,lambda b:finished(initial+b),
                min(time.monotonic()+observe_seconds,active_deadline))
        if b'NO CARRIER' in received:
            online=False; result['stop_reason']='remote_disconnect_or_DCE_carrier_result'
        elif any(pattern.search(received) for pattern in patterns):
            result['stop_reason']='target_prompt'
        else:
            result['stop_reason']='observation_timeout'
        result['post_connect_bytes']=len(received)
        return result
    finally:
        result['hangup_confirmed']=False
        try:
            escaped=not online
            if online:
                # Guard time allows real Hayes escape; drain bytes into transcript.
                read_until(connection,lambda b:False,time.monotonic()+1.3)
                connection.sendall(b'+++')
                answer=read_until(connection,lambda b:b'OK' in b or b'NO CARRIER' in b,time.monotonic()+4)
                escaped=b'OK' in answer or b'NO CARRIER' in answer
                if not escaped:
                    result['hangup_reason']='escape not acknowledged; runner must stop owned DCE'
            if escaped:
                connection.sendall(b'ATH0\r')
                answer=read_until(connection,lambda b:b'OK' in b or b'NO CARRIER' in b,time.monotonic()+3)
                result['hangup_confirmed']=b'OK' in answer or b'NO CARRIER' in answer
        except (OSError,ConnectionError):
            result['hangup_reason']='DTE closed; runner verifies PBX channel cleanup'


def _copy(container, source, destination):
    with Path(destination).open('wb') as file:
        subprocess.run(['docker','exec',container,'cat',source],stdout=file,stderr=subprocess.PIPE,check=True)


def collect_audio(container, stem, directory, modem_port=16112):
    directory=Path(directory); report={}
    pcm=pcm_inventory(directory)
    for direction in ('local','remote','mix'):
        target=directory/('pbx_'+direction+'.ulaw')
        exists=subprocess.run(['docker','exec',container,'test','-f',stem+'_'+direction+'.ulaw']).returncode==0
        if exists:
            _copy(container,stem+'_'+direction+'.ulaw',target)
        else:
            target.touch()
        report[direction]=dict(bytes=target.stat().st_size,original_file_exists=exists)
    left=decode(np.frombuffer((directory/'pbx_local.ulaw').read_bytes(),np.uint8),'pcmu').astype('<i2')
    right=decode(np.frombuffer((directory/'pbx_remote.ulaw').read_bytes(),np.uint8),'pcmu').astype('<i2')
    length=max(len(left),len(right)); stereo=np.zeros((length,2),dtype='<i2')
    stereo[:len(left),0]=left;stereo[:len(right),1]=right
    with wave.open(str(directory/'pbx_duplex.wav'),'wb') as file:
        file.setparams((2,2,8000,0,'NONE','not compressed'));file.writeframes(stereo.tobytes())
    report.update(channels=['local-originating modem','remote-answering modem'],sample_rate=8000,
        timeline='Asterisk MixMonitor r/t common sample timeline; synchronization padding enabled',
        derived_end_padding=dict(local=length-len(left),remote=length-len(right)),
        originals_preserved=True)
    packets=[json.loads(line) for line in (directory/'rtp.timeline.jsonl').read_text().splitlines()]
    report['packet_mapping']={}
    for direction,audio in (('local',left),('remote',right)):
        index={}
        for row in packets:
            # Modem-side leg only; the other leg is retained in the same PCAP.
            source,destination=row['stream'].split('>')
            relevant=(source.endswith(':'+str(modem_port)) if direction=='local'
                      else destination.split('/')[0].endswith(':'+str(modem_port)))
            if not relevant:
                continue
            value=decode(np.frombuffer(bytes.fromhex(row['pcmu_payload_hex']),np.uint8),'pcmu').astype('<i2')
            digest=hashlib.sha256(value.tobytes()).hexdigest()
            index.setdefault(digest,[]).append({k:row[k] for k in
                ('stream','sequence','timestamp','arrival_wall_time','pcap_record_offset')})
        unmapped=ambiguous=silent=matched=0
        with (directory/('pbx_'+direction+'.packets.jsonl')).open('w') as file:
            for start in range(0,len(audio),160):
                values=audio[start:start+160]
                candidates=index.get(hashlib.sha256(values.tobytes()).hexdigest(),[])
                is_silent=bool(np.max(abs(values.astype(int)))<=8)
                if candidates:
                    matched+=1;ambiguous+=int(len(candidates)>1)
                elif is_silent:
                    silent+=1
                else:
                    unmapped+=1
                file.write(json.dumps(dict(pbx_sample=start,samples=len(values),
                    packet_candidates=candidates,near_silence=is_silent,
                    origin=('wire' if candidates else 'synchronization_silence_or_unmapped'),
                    ambiguity='identical packet audio; no unique time assignment' if len(candidates)>1 else None))+'\n')
        report['packet_mapping'][direction]=dict(matched_frames=matched,
            ambiguous_frames=ambiguous,unmatched_near_silence_frames=silent,
            unmatched_active_frames=unmapped,valid=unmapped==0 and (matched>0 or not index),
            missing_original_when_wire_audio_exists=bool(index and not report[direction]['original_file_exists']))
    report['alignment_rule']='per-frame wire provenance; no fixed offset assumed for PBX padding'
    (directory/'audio.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def run(args):
    out=args.output.resolve()
    if out.exists():
        raise ValueError('a fresh output directory is required')
    out.mkdir(parents=True,mode=0o700)
    call_id='capture-'+hashlib.sha256(str(out).encode()).hexdigest()[:12]
    journal=[]; packet=PacketCapture(out,args.pbx_container)
    result=dict(schema=1,call_id=call_id,scope='local control' if args.local_control else 'VSP/PSTN startup observation',
        rate_cap=args.rate,phase4_eq_span=args.phase4_eq_span,
        qualified_baseline_image=BASELINE,status='PREPARING',
        capture_complete=False,payload_verified=False,external_call_placed=False)
    owned=[];dut=None;connection=None;logger=False;packet_started=False
    logger_name=call_id+'.log'
    pbx_stem='/var/spool/asterisk/'+call_id
    def command(text):
        answer=cli(text,args.pbx_container);journal.append(dict(command=text,response=answer))
        return answer
    try:
        counts=command('core show channels count')
        if '0 active channels' not in counts or '0 active calls' not in counts:
            raise RuntimeError('local capture needs an idle PBX to associate both media legs')
        result['pbx_settings']=command('core show settings')
        result['dut_image_id']=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.dut_image],text=True).strip()
        # Record the exact packaged code. The local gate additionally compares it
        # against its frozen checkout snapshot before placing its control call.
        script="import pathlib,hashlib,json;root=pathlib.Path('/app');print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}))"
        result['packaged_source']=json.loads(subprocess.check_output(['docker','run','--rm','--entrypoint','python3',args.dut_image,'-c',script],text=True))
        expected={}
        for folder,origin in (('',ROOT/'lab/asterisk'),('audio-peer/',ROOT/'lab/asterisk/audio-peer')):
            for path in origin.glob('*.py'):
                if path.is_file():
                    expected[folder+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if any(result['packaged_source'].get(name)!=digest for name,digest in expected.items()):
            raise RuntimeError('diagnostic image does not match current source; rebuild before dialing')
        answer=command('logger add channel '+logger_name+' debug,verbose,notice,warning,error,dtmf')
        logger=True
        subprocess.run(['docker','exec',args.pbx_container,'test','-f','/var/log/asterisk/'+logger_name],check=True)
        result['packet_collector']=packet.start();packet_started=True
        if args.local_control:
            control=out/'control'
            command_line=local_control_command(args,control)
            with (out/'control-runner.log').open('w') as log:
                code=subprocess.run(command_line,stdout=log,stderr=subprocess.STDOUT,timeout=210).returncode
            result.update(control_exit_code=code,status='CONTROL_FINISHED')
            result['control_matrix']=json.loads((control/'matrix.json').read_text())
            rows=result['control_matrix'].get('results',[])
            result['payload_verified']=bool(code==0 and rows and all(
                row.get('exit_code')==0 and row['verification']['valid']
                and row['verification']['realtime_valid'] for row in rows))
            pbx_stem='/var/spool/asterisk/v34-dce-'+hashlib.sha256(str(control).encode()).hexdigest()[:8]
        else:
            targets=json.loads((ROOT/'lab/V34_PSTN_TARGETS.json').read_text())
            target=next(t for t in targets['targets'] if t['id']==args.target)
            if target['transport']!='vsp-pstn':
                raise ValueError('this runner requires a VSP/PSTN target')
            result['target']=target
            extension='619'+str(int(call_id[-6:],16)%1000).zfill(3)
            if "'"+extension+"'" in command('dialplan show '+extension+'@from-lab'):
                raise RuntimeError('temporary capture extension already owned')
            steps=[('NoOp',call_id+'_${UNIQUEID}_${CHANNEL(linkedid)}'),('Set','CALLERID(num)=${LAB_CALLERID}'),
                ('MixMonitor',pbx_stem+'_mix.ulaw,t('+pbx_stem+'_remote.ulaw)r('+pbx_stem+'_local.ulaw)'),
                ('Dial','PJSIP/'+target['dial_number']+'@voipms,60'),('Hangup','')]
            for priority,(application,data) in enumerate(steps,1):
                answer=command(f'dialplan add extension {extension},{priority},{application}'+('('+data+')' if data else '')+' into from-lab')
                if "added into 'from-lab' context" not in answer:
                    raise RuntimeError('failed to install capture route')
                if extension not in owned:owned.append(extension)
            dut=call_id+'-dut'
            docker_command=dut_docker_command(args,dut,out)
            result['dut_command']=docker_command
            subprocess.run(docker_command,capture_output=True,check=True)
            # The DTE client lives inside the same network namespace as the DCE.
            # Use an exec'd observation driver, retaining every byte to the mount.
            driver=['docker','exec',dut,'python3','/app/run_pstn_capture.py','--observe-dte',
                '--number',extension,'--output','/captures/session.json']
            if getattr(args,'automode',False):driver+=['--automode','--rate',str(args.rate)]
            hold=float(getattr(args,'hold_seconds',0) or 0)
            if hold>0:         # long session: no prompt stop, hold the link idle for ``hold`` seconds after CONNECT
                driver+=['--observe-seconds',str(hold),'--whole-seconds',str(hold+70)]
            else:
                for pattern in args.prompt or [r'(?:login|username|user name)\s*:']:
                    driver+=['--prompt',pattern]
            if getattr(args,'reply_emsi',False):driver+=['--reply-emsi']
            if getattr(args,'error_control',3)!=3:driver+=['--error-control',str(args.error_control)]      # the session subprocess sends the AT init
            for item in getattr(args,'reply',None) or []:driver+=['--reply',item]
            if packet.process.poll() is not None:
                raise RuntimeError('packet collector stopped before the dial command')
            with (out/'session-driver.log').open('w') as log:
                result['driver_exit_code']=subprocess.run(driver,stdout=log,stderr=subprocess.STDOUT,timeout=110+float(getattr(args,'hold_seconds',0) or 0)).returncode
            result['session']=json.loads((out/'session.json').read_text())
            result['status']='SESSION_DRIVER_ERROR' if result['session'].get('error') else 'OBSERVATION_FINISHED'
    except Exception as error:
        result.update(status='ERROR',reason=str(error))
    finally:
        if connection is not None:
            connection.close()
        if dut:
            subprocess.run(['docker','stop','-t','5',dut],capture_output=True)
            with (out/'dce.log').open('w') as file:
                subprocess.run(['docker','logs',dut],stdout=file,stderr=subprocess.STDOUT)
            subprocess.run(['docker','rm',dut],capture_output=True)
        try:
            result['final_pbx_channels']=command('core show channels count')
            if '0 active channels' not in result['final_pbx_channels']:
                result.setdefault('collection_errors',[]).append('PBX channels remain; recording not finalized')
            if packet.process is not None:
                packet.stop()
            if logger:
                command('logger remove channel '+logger_name)
                _copy(args.pbx_container,'/var/log/asterisk/'+logger_name,out/'pbx.log')
            if packet_started:
                result['pcap']=inspect_pcap(out/'wire.pcap',out/'rtp.timeline.jsonl')
                log=(out/'tcpdump.log').read_text()
                drops=re.search(r'(\d+) packets dropped by kernel',log)
                result['pcap']['kernel_drops']=int(drops[1]) if drops else None
                if result.get('target'):
                    number=result['target']['dial_number']
                    result['external_call_placed']=any(message['first_line'].startswith('INVITE sip:'+number+'@')
                        for message in result['pcap'].get('sip_summary',[]))
            result['pcm']=pcm_inventory(out)
            result['dsp_replay']=[]
            for path in out.rglob('*.dsp.json'):
                replay=verify(str(path)[:-5]);result['dsp_replay'].append(dict(path=str(path),report=replay))
            if result['status']!='ERROR':
                result['pbx_audio']=collect_audio(args.pbx_container,pbx_stem,out)
            metrics=list(out.rglob('*.metrics.json'))+([out/'metrics.json'] if (out/'metrics.json').exists() else [])
            result['metrics']=[json.loads(path.read_text()) for path in metrics]
            receiver_created=bool(list(out.rglob('*.dsp.ready.json')))
            result['stage_coverage']=dict(receiver_created=receiver_created,
                local_pcm='recorded' if result['pcm']['captures'] else 'not reached',
                live_dsp='recorded' if result['dsp_replay'] else 'not reached')
            local_complete=(result['pcm']['valid'] and result['dsp_replay']
                and all(r['report']['valid'] for r in result['dsp_replay']) and result['metrics']) if receiver_created else (
                    bool(list(out.rglob('*.dsp.armed.json'))) and not result.get('session',{}).get('connected',False))
            result['capture_complete']=bool(packet_started and result.get('pcap',{}).get('valid')
                and result['pcap'].get('kernel_drops')==0 and local_complete
                and result.get('pbx_audio') and all(a['valid'] for a in
                    result['pbx_audio'].get('packet_mapping',{}).values())
                and not result.get('collection_errors'))
            result['outcome']=dict(connect=connection_observed(result),
                target_prompt_observed=result.get('session',{}).get('stop_reason')=='target_prompt',
                qualification=[metric.get('rx_qualification') for metric in result['metrics']],
                modem_failure_reasons=[metric.get('engine_failed_reason') for metric in result['metrics']])
        except Exception as error:
            result.setdefault('collection_errors',[]).append(str(error))
        for extension in owned:
            command('dialplan remove extension '+extension+'@from-lab')
        (out/'pbx-commands.json').write_text(json.dumps(journal,indent=2)+'\n')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'manifest.json').write_text(json.dumps(hash_manifest(out),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),status=result['status'],capture_complete=result['capture_complete'],
                         payload_verified=result['payload_verified']),indent=2))
    return int(not result['capture_complete'] or result['status'] in ('ERROR','SESSION_DRIVER_ERROR')
               or (args.local_control and not result['payload_verified']))


def observe(args):
    result={};connection=None
    try:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:
                connection=socket.create_connection(('127.0.0.1',9104),timeout=.2);break
            except OSError:
                time.sleep(.1)
        if connection is None:
            raise RuntimeError('DCE not ready')
        if not (args.output.parent/'call.dsp.armed.json').exists():
            raise RuntimeError('live capture exporter not armed before dialing')
        connection=TranscriptSocket(connection,args.output.parent/'dte')
        result=probe(connection,args.number,observe_seconds=args.observe_seconds,whole_seconds=args.whole_seconds,prompts=args.prompt,replies=reply_list(args),error_control=args.error_control,
                     modulation_profile=f'V34,1,300,{args.rate}' if args.automode else None)
    except Exception as error:
        result=dict(error=str(error),payload_verified=False)
    finally:
        if connection is not None:connection.close()
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    return int(bool(result.get('error')))


def dut_docker_command(args,dut,out):
    """The ``docker run`` command for the DUT of a real-call observation (kept separate so it can be tested)."""
    optin_env,optin_args=optin_settings(args)
    return ['docker','run','-d','--name',dut,'--network','container:asterisk-audio-peer-1',
        '-e','OPENBLAS_NUM_THREADS=1','-e','MODEM_DEBUG=1','-e','V42_DEBUG=1',
        '-e','V34_RATE='+str(args.rate),*(['-e','V34_SYMS=3429'] if args.rate>=24000 else []),'-e','EL_MODEM_RTP_JITTER_MS=40',
        *[x for item in optin_env for x in ('-e',item)],'-v',str(out)+':/captures',args.dut_image,
        '--serial-tcp','127.0.0.1:9104','--line-mode','ms' if getattr(args,'automode',False) else 'v34','--v34-native-trellis',*optin_args,
        '--sip-host','0.0.0.0','--sip-port','5092','--media-host','172.29.103.20','--media-port','16112',
        '--dial-host','172.29.103.10','--timing-trace','--metrics-file','/captures/metrics.json',
        '--capture','/captures/call.pcm','--capture-seconds',str(max(120,int(float(getattr(args,'hold_seconds',0) or 0))+70)),'--dsp-capture','/captures/call.dsp']


def reply_list(args):
    """Replies to send during the observation: --reply-emsi and repeated --reply 'REGEX=TEXT' (TEXT with \\r, \\n escapes)."""
    out=[]
    if getattr(args,'reply_emsi',False):
        out.append((r'EMSI_REQ',EMSI_INQ))
    for item in getattr(args,'reply',None) or []:
        pattern,_,text=item.partition('=')
        out.append((pattern,text.encode('latin1').decode('unicode_escape').encode('latin1')))
    return out


def optin_settings(args):
    """Extra DUT environment and arguments for the opt-in receiver features; empty by default (the qualified behaviour)."""
    env, extra = [], []
    if getattr(args,'opt_in',None) == 'clock':
        extra += ['--v34-trn-clock']                      # only the TRN clock fit: no native paths, no gradient timing
    if getattr(args,'opt_in',None) == 'tested':
        extra += ['--v34-trn-gradient-auto','--v34-trn-clock']
        env += ['V34_TRN_CLOCK_NATIVE=1','V34_TRAJECTORY_GRID=1','V34_NATIVE_FEEDBACK=1','V34_TIMING_MODE=data-gradient']
    if getattr(args,'info_hole_tolerant',False):
        env += ['V34_INFO_HOLE_TOLERANT=1']
    if getattr(args,'gc_pause',False):
        extra += ['--gc-pause']                           # no automatic GC during the call (lab/V34_PHASE4_TX_GAP_GC.md)
    if getattr(args,'v34_preimport',False):
        extra += ['--v34-preimport']                      # DCE imports the V.34 data-link modules before the call (lab/V34_FULL_DCE_STALL_REPLAY.md)
    if getattr(args,'phase4_eq_span',12) == 24:
        env += ['V34_PHASE4_EQ_SPAN=24']
    env += list(getattr(args,'dut_env',None) or [])
    return env, extra


DUT_ENV_PREFIXES=('V34_','V42_','EL_MODEM_')


def dut_env_arg(text):
    """argparse type for --dut-env: NAME=VALUE with a modem-feature prefix (so a typo cannot redirect capture paths or the shell)."""
    name,sep,value=text.partition('=')
    if not sep or not name.isidentifier() or not name.startswith(DUT_ENV_PREFIXES):
        raise argparse.ArgumentTypeError(f'expected NAME=VALUE with NAME starting {"/".join(DUT_ENV_PREFIXES)}, got {text!r}')
    return text


def local_control_command(args,control):
    """Qualify the same receiver options and thread cap as the PSTN DUT."""
    env,extra=optin_settings(args)
    control_flags={'--v34-trn-clock':'--trn-clock',
                   '--v34-trn-gradient-auto':'--trn-gradient-auto',
                   '--v34-preimport':'--v34-preimport', '--gc-pause':'--gc-pause'}
    command=[sys.executable,str(ROOT/'lab/slmodem-peer/run_v34_dce_gates.py'),
        '--output',str(control),'--dut-image',args.dut_image,
        '--native-trellis','--timing-trace','--capture-diagnostics',
        '--expected-convention-policy','dual-convention',
        '--rates',str(args.rate),'--roles','call','--protocols','lapm',
        '--caller-transport','sip','--seeds','3400','--bytes','16384',
        *[control_flags[item] for item in extra]]
    for item in ['OPENBLAS_NUM_THREADS=1',*env]:
        command += ['--dut-env',item]
    if getattr(args,'automode',False):command.append('--automode')
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--automode',action='store_true',help='all-in-one AT+MS=V34,1,300,RATE rather than forced V34')
    parser.add_argument('--opt-in',choices=['tested','clock'],default=None,help='clock: only the TRN clock fit (lab/V34_OPTIN24_CALL.md). tested: enable the validated opt-in receiver set: TRN clock fit, trajectory grid, startup/data gradient timing, native feedback (lab/V34_CONSOLIDATION_2026-10-07.md)')
    parser.add_argument('--info-hole-tolerant',action='store_true',help='accept an INFO1a whose CRC fails only over RTP loss-concealment evidence (logged unverified)')
    parser.add_argument('--v34-preimport',action='store_true',help='start the DUT with --v34-preimport (import the V.34 data-link modules before accepting the call)')
    parser.add_argument('--gc-pause',action='store_true',help='start the DUT with --gc-pause (no automatic garbage collection during the call)')
    parser.add_argument('--phase4-eq-span',type=int,choices=[12,24],default=12,
        help='phase-four equalizer span (default: 12; 24 is locally qualified on clean short calls only)')
    parser.add_argument('--dut-env',action='append',default=[],type=dut_env_arg,metavar='NAME=VALUE',
        help='extra environment for the DUT container, repeatable (names starting V34_, V42_ or EL_MODEM_), e.g. V42_ASSUME_BIS=1')
    parser.add_argument('--error-control',type=int,choices=[0,3],default=3,
        help='AT\\N setting for the call: 3 (default) = V.42/LAPM with ODP detection, 0 = no error control (no ODP is sent)')
    parser.add_argument('--local-control',action='store_true')
    parser.add_argument('--target',choices=['dialup-world-pstn','astound-new-york-pstn','astound-queens-pstn'],default='dialup-world-pstn')
    parser.add_argument('--rate',type=int,choices=[9600,14400,19200,24000,28800,31200,33600],default=24000,
        help='data rate to offer; below 24000 the symbol rate is left to the answerer (V34_SYMS unset), 24000 and above offer 3429 only')
    parser.add_argument('--dut-image',default='el-modem-native-pstn-diagnostic')
    parser.add_argument('--pbx-container',default='asterisk-asterisk-1')
    parser.add_argument('--prompt',action='append',help='observed prompt regex; no reply is sent')
    parser.add_argument('--reply-emsi',action='store_true',help='when the answerer sends EMSI_REQ, reply once with the EMSI inquiry **EMSI_INQC816 (no data or credentials)')
    parser.add_argument('--reply',action='append',help='REGEX=TEXT: send TEXT once when REGEX appears in the received data (escapes like \\r allowed)')
    parser.add_argument('--hold-seconds',type=float,default=0,help='long-session mode: do not stop at the login prompt; hold the link idle this many seconds after CONNECT (no input is sent), then escape and hang up. Lengthens the call limits and the PCM capture cap to match')
    parser.add_argument('--observe-seconds',type=float,default=30,help=argparse.SUPPRESS)
    parser.add_argument('--whole-seconds',type=float,default=90,help=argparse.SUPPRESS)
    parser.add_argument('--observe-dte',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--number',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.observe_dte:
        return observe(args)
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
