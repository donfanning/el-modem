"""Packaged multi-worker local SIP/payload gate; no PBX or external dialing."""
import argparse
import array
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import termios
import time
import tty
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'lab'/'asterisk'))
from modem_supervisor import Supervisor


def run(executable,output,serial_client='terminal',pairs=2,stagger_calls=False):
    if pairs not in (1,2):raise ValueError('pairs must be 1 or 2')
    output.mkdir(parents=True,exist_ok=False)
    supervisor=Supervisor(output/'config.json');supervisor.acquire()
    transcript={}; responses={}; negotiated={}; ports={}; connected_lines={}; processes=[]
    for index in range(pairs*2):
        spec=supervisor.add(serial='driver')
        spec.sip_port=25200+index*2;spec.media_port=25400+index*4
    supervisor.save()
    def poll():
        supervisor.poll()
        for ident,data in supervisor.events:transcript.setdefault(ident,bytearray()).extend(bytes.fromhex(data))
        supervisor.events.clear()
        for ident,fd in ports.items():
            try:
                data=os.read(fd,65536)
                if data:transcript.setdefault(ident,bytearray()).extend(data)
            except BlockingIOError:pass
        for ident,state in supervisor.states.items():
            if state.get('state')=='failed':raise RuntimeError(f'{ident}: {state}')
    def wait(predicate,seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            poll()
            if predicate():return
            time.sleep(.01)
        raise RuntimeError('bounded wait timed out')
    def request(ident,action,**fields):
        key=supervisor.request(ident,action,**fields)
        wait(lambda:key in supervisor.responses,10)
        reply=supervisor.responses.pop(key)
        if not reply['ok']:raise RuntimeError(reply['error'])
        return reply.get('value')
    def write(ident,data):
        if serial_client=='terminal':
            request(ident,'terminal.write',hex=data.hex());return
        offset=0;deadline=time.monotonic()+10
        while offset<len(data):
            try:offset+=os.write(ports[ident],data[offset:])
            except BlockingIOError:pass
            poll()
            if time.monotonic()>deadline:raise TimeoutError('external serial write timed out')
            if offset<len(data):time.sleep(.01)
    try:
        with patch.object(sys,'frozen',True,create=True),patch.object(sys,'executable',str(executable.resolve())):
            for spec in supervisor.specs:supervisor.start(spec.id)
        processes=[proc for proc,channel in supervisor.workers.values()]
        wait(lambda:all(s.get('state')=='idle' for s in supervisor.states.values()) and len(supervisor.states)==pairs*2,120)
        for spec in supervisor.specs:
            if serial_client=='terminal':request(spec.id,'terminal.open')
            else:
                fd=os.open(supervisor.states[spec.id]['serial'],os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY|os.O_CLOEXEC)
                ports[spec.id]=fd
                tty.setraw(fd)
                attrs=termios.tcgetattr(fd);attrs[4]=attrs[5]=termios.B115200
                attrs[2]|=termios.CLOCAL|termios.CRTSCTS
                termios.tcsetattr(fd,termios.TCSANOW,attrs)
                fcntl.ioctl(fd,termios.TIOCMBIS,array.array('I',[termios.TIOCM_DTR|termios.TIOCM_RTS]))
        calls=[(supervisor.specs[index*2],supervisor.specs[index*2+1]) for index in range(pairs)]
        for caller,answer in calls:
            request(caller.id,'set',name='sip.server',value='127.0.0.1')
            request(caller.id,'set',name='sip.port',value=str(answer.sip_port))
            request(answer.id,'set',name='s0',value=1)
        import re
        for caller,answer in calls:
            write(caller.id,b'ATDT1\r')
            if stagger_calls:
                wait(lambda:all(supervisor.states.get(spec.id,{}).get('state')=='connected'
                                and re.search(rb'CONNECT [0-9]+(?:/[A-Z0-9]+)?\r\n',transcript.get(spec.id,b''))
                                for spec in (caller,answer)),120)
        wait(lambda:all(s.get('state')=='connected' for s in supervisor.states.values()),120)
        # Keep negotiation results and clear command echoes before exact payload.
        wait(lambda:all(re.search(rb'CONNECT [0-9]+(?:/[A-Z0-9]+)?\r\n',transcript.get(spec.id,b'')) for spec in supervisor.specs),10)
        quiet_until=time.monotonic()+.3
        while time.monotonic()<quiet_until:poll();time.sleep(.01)
        for ident,data in transcript.items():(output/(ident+'.startup.dte.raw')).write_bytes(data)
        for spec in supervisor.specs:negotiated[spec.id]=dict(supervisor.states[spec.id]);transcript[spec.id]=bytearray()
        for ident,fd in ports.items():
            bits=array.array('I',[0]);fcntl.ioctl(fd,termios.TIOCMGET,bits,True)
            connected_lines[ident]=dict(DCD=bool(bits[0]&termios.TIOCM_CAR),DSR=bool(bits[0]&termios.TIOCM_DSR),
                CTS=bool(bits[0]&termios.TIOCM_CTS),RI=bool(bits[0]&termios.TIOCM_RNG))
            if not all(connected_lines[ident][key] for key in ('DCD','DSR','CTS')) or connected_lines[ident]['RI']:
                raise RuntimeError(f'{ident}: connected native hardware signals disagree {connected_lines[ident]}')
        # Opposite, distinctive printable bodies are safe for ASCII DTE/flow control.
        payloads={spec.id:(f'{spec.id}:'.encode()+b'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\r\n')*400 for spec in supervisor.specs}
        payloads={ident:data[:16384] for ident,data in payloads.items()}
        for offset in range(0,16384,512):
            for spec in supervisor.specs:write(spec.id,payloads[spec.id][offset:offset+512])
            time.sleep(.005)
        opposite={left.id:right.id for left,right in calls}
        opposite.update({right.id:left.id for left,right in calls})
        wait(lambda:all(len(transcript.get(spec.id,b''))>=16384 for spec in supervisor.specs),90)
        for spec in supervisor.specs:
            expected=payloads[opposite[spec.id]];actual=bytes(transcript[spec.id])
            if actual!=expected:raise RuntimeError(f'{spec.id}: exact payload mismatch ({len(actual)} bytes)')
            responses[spec.id]=dict(bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest(),state=negotiated[spec.id])
        report=dict(status='PASS',cases=responses,external_calls=0,artifact_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    serial_client=serial_client,connected_hardware_lines=connected_lines,
                    pairs=pairs,workers=pairs*2,stagger_calls=stagger_calls,
                    scope=f'{pairs*2} frozen workers/source supervisor; {pairs} local calls, not independent interoperability or TUI rendering')
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=str(exc),states=supervisor.states,lengths={k:len(v) for k,v in transcript.items()}),indent=2)+'\n')
        raise
    finally:
        for ident,data in transcript.items():(output/(ident+'.data.dte.raw')).write_bytes(data)
        for fd in ports.values():os.close(fd)
        supervisor.close()
        if any(proc.poll() is None for proc in processes):raise RuntimeError('worker teardown failed')
    report['cleanup']=True
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{pairs*2} packaged workers / {pairs} simultaneous SIP calls / exact bidirectional 16KB PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--serial-client',choices=('terminal','external'),default='terminal',
                        help='internal ASCII terminal IPC or an ordinary native serial application with hardware flow control')
    parser.add_argument('--pairs',type=int,choices=(1,2),default=2,
                        help='number of concurrent local modem-to-modem calls (2 workers per call)')
    parser.add_argument('--stagger-calls',action='store_true',
                        help='wait for each call to connect before starting the next')
    args=parser.parse_args();run(args.executable,args.output,args.serial_client,args.pairs,args.stagger_calls)
