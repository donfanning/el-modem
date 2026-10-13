"""Packaged supervisor/TUI entrypoint, with legacy DCE CLI compatibility."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

_PACKAGING=Path(__file__).resolve().parents[2]/'packaging'
if _PACKAGING.is_dir(): sys.path.insert(0,str(_PACKAGING))

def worker(path,fd):
    import signal
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    from native_runtime import NATIVE_ENV
    from modem_settings import ENV
    document=json.loads(Path(path).read_text()); spec=document['spec']; directory=Path(document['directory'])
    for key in list(os.environ):
        if key.startswith(('V34_','MODEM_','EL_MODEM_')): del os.environ[key]
    os.environ.update(NATIVE_ENV)
    os.environ['EL_MODEM_FACTORY_NATIVE']='1'
    boot={'instance.name':'name','serial.backend':'serial','serial.tcp_port':'serial_port',
          'sip.bind_host':'sip_host','sip.bind_port':'sip_port','sip.media_host':'media_host','sip.media_port':'media_port'}
    for name,key in boot.items(): os.environ[ENV[name][0]]=str(spec[key])
    profile=directory/'profiles.json'
    if profile.exists():
        saved=json.loads(profile.read_text()).get('profiles',{}).get('0',{}).get('extended_settings',{})
        for name,value in saved.items():
            if name in ENV: os.environ[ENV[name][0]]=str(value)
    for name,key in boot.items():
        value=os.environ[ENV[name][0]]
        spec[key]=int(value) if key.endswith('port') else value
    from modem_supervisor import ModemSpec
    ModemSpec(**spec).validate()
    from modem_control import WorkerControl
    from modem_serial import DriverSerial, LegacySerial
    from modem_dce import main
    link=str(directory/'serial')
    if Path(link).exists() or Path(link).is_symlink():
        raise ValueError(f'refusing to replace existing serial path: {link}')
    control=WorkerControl(fd)
    serial=None
    def create_serial():
        nonlocal serial
        serial=(DriverSerial(link) if spec['serial']=='driver' else
                LegacySerial(spec['serial'],link=link,address=f'127.0.0.1:{spec["serial_port"]}'))
        return serial
    try:
        prefix=str(directory/'captures'/f'run-{time.time_ns()}'/'call')
        Path(prefix).parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        sys.argv=['el-modem','--line-mode','ms','--sip-host',spec['sip_host'],'--sip-port',str(spec['sip_port']),
                  '--media-host',spec['media_host'],'--media-port',str(spec['media_port']),
                  '--profile-file',str(profile),'--metrics-file',prefix+'.metrics.json']
        if os.environ.get('EL_MODEM_RECORDING','1')=='1':
            sys.argv += ['--capture',prefix,'--dsp-capture',prefix+'.dsp','--capture-seconds',os.environ.get('EL_MODEM_RECORDING_SECONDS','600')]
        # Publish the serial endpoint only after imports and native warm-up.
        main(serial_backend=create_serial,control=control)
        serial=None # DCE owns cleanup after successful entry.
    finally:
        if serial is not None:
            try: serial.close()
            except OSError: pass
        if not control.channel.closed: control.close()

def main():
    if '--worker' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--control-fd',type=int,required=True)
        a=p.parse_args();worker(a.worker,a.control_fd);return
    supervisor_flags={'--tui','--headless','--config','--modems','--serial-backend','--help','-h'}
    if len(sys.argv)>1 and not any(x.split('=')[0] in supervisor_flags for x in sys.argv[1:]):
        from modem_dce import main as legacy
        legacy();return
    p=argparse.ArgumentParser(prog='el-modem',description='EL-MODEM multi-modem dashboard and native DSP runtime')
    ui=p.add_mutually_exclusive_group();ui.add_argument('--tui',action='store_true');ui.add_argument('--headless',action='store_true')
    p.add_argument('--config',default=str(Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'el-modem'/'config.json'))
    p.add_argument('--modems',type=int);p.add_argument('--serial-backend',choices=('driver','pty','tcp'),default='driver')
    a=p.parse_args()
    if a.modems is not None and a.modems<1: p.error('--modems must be positive')
    if not a.headless and not sys.stdin.isatty(): p.error('interactive terminal required; use --headless')
    from modem_supervisor import Supervisor
    supervisor=Supervisor(a.config)
    try:
        supervisor.acquire()
        count=a.modems if a.modems is not None else max(1,len(supervisor.specs))
        while len(supervisor.specs)<count: supervisor.add(serial=a.serial_backend)
        if a.modems is not None and len(supervisor.specs)>count: p.error('configuration contains more modems; remove them in the TUI')
        for spec in supervisor.specs:
            if not spec.enabled: continue
            try: supervisor.start(spec.id)
            except (OSError,ValueError) as exc: supervisor.states[spec.id]=dict(state='failed',error=str(exc),lights={})
        if a.headless:
            import signal
            signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            while True:
                supervisor.poll()
                if not supervisor.workers: raise RuntimeError('no modem workers running')
                time.sleep(.05)
        else:
            from modem_tui import ModemApp
            ModemApp(supervisor).run()
    except KeyboardInterrupt: pass
    finally: supervisor.close()

if __name__=='__main__': main()
