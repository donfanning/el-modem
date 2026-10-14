"""Exercise an actual packaged supervisor/worker over local TCP, with no calls."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time


def port(kind=socket.SOCK_STREAM):
    with socket.socket(type=kind) as reserve:
        reserve.bind(('127.0.0.1',0));return reserve.getsockname()[1]


def qualify(executable):
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);serial=port();sip=port(socket.SOCK_DGRAM);media=port(socket.SOCK_DGRAM)
        while media==sip:media=port(socket.SOCK_DGRAM)
        config=root/'config.json'
        config.write_text(json.dumps(dict(schema=1,modems=[dict(id='modem-1',name='Smoke',sip_host='127.0.0.1',sip_port=sip,media_host='127.0.0.1',media_port=media,serial='tcp',serial_port=serial,enabled=True)])))
        connection=None;checks=[]
        with (root/'supervisor.log').open('wb') as log:
            options={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {}
            proc=subprocess.Popen([str(Path(executable).resolve()),'--headless','--config',str(config)],stdout=log,stderr=log,**options)
            try:
                deadline=time.monotonic()+120
                while time.monotonic()<deadline:
                    if proc.poll() is not None:raise RuntimeError('supervisor exited: '+(root/'supervisor.log').read_text(errors='replace'))
                    try:connection=socket.create_connection(('127.0.0.1',serial),timeout=1);break
                    except OSError:time.sleep(.1)
                if connection is None:raise RuntimeError('packaged worker did not publish TCP endpoint')
                checks.append('packaged supervisor launched a worker and published TCP DTE')
                connection.settimeout(5);connection.sendall(b'ATI3\r');data=bytearray()
                while b'OK\r\n' not in data:
                    block=connection.recv(4096)
                    if not block:raise RuntimeError('worker closed DTE')
                    data.extend(block)
                    if len(data)>65536:raise RuntimeError('unbounded DTE response')
                if b'donfanning/el-modem' not in data:raise RuntimeError('wrong modem identity')
                checks.append('worker responds to Hayes identity command')
                connection.close();connection=None
                proc.send_signal(signal.CTRL_BREAK_EVENT if os.name=='nt' else signal.SIGTERM)
                proc.wait(timeout=20)
                if proc.returncode:raise RuntimeError('unclean supervisor exit '+str(proc.returncode))
                checks.append('cooperative supervisor/worker shutdown exits cleanly')
            finally:
                if connection is not None:connection.close()
                if proc.poll() is None:
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                    else:proc.kill()
                    proc.wait(timeout=10)
            worker_log=root/'instances/modem-1/worker.log'
            if worker_log.exists() and 'Traceback' in worker_log.read_text(errors='replace'):
                raise RuntimeError('worker traceback: '+worker_log.read_text(errors='replace'))
        return dict(status='PASS',checks=checks,scope='native packaged TCP DTE/control startup and cleanup; no SIP/PSTN call or virtual COM qualification')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('executable');p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();report=qualify(a.executable);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
