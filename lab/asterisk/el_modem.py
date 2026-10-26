"""Packaged supervisor/TUI entrypoint, with legacy DCE CLI compatibility."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

_PACKAGING=Path(__file__).resolve().parents[2]/'packaging'
if _PACKAGING.is_dir(): sys.path.insert(0,str(_PACKAGING))


WORKER_PASSTHROUGH_ENV=(
    'V34_GAP_FEEDBACK', 'V34_GAP_RECOVERY', 'V34_EQUALIZER_BACKEND',
    'V34_NATIVE_FEEDBACK', 'V34_SYMBOL_INTERP_NATIVE', 'V32_NUMPY_METRICS',
    'EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS', 'EL_MODEM_TIMING_TRACE',
    'EL_MODEM_TX_WATCHDOG', 'EL_MODEM_RECORDING', 'EL_MODEM_RECORDING_SECONDS',
)  # explicitly opted-in experiments/diagnostics; all other modem env is scrubbed
WORKER_SCRUB_PREFIXES=('V34_','V42_','MODEM_','EL_MODEM_')


def worker_passthrough_environment(environ=None):
    """Return the small, explicit set of parent settings workers may inherit."""
    environ = os.environ if environ is None else environ
    return {key: environ[key] for key in WORKER_PASSTHROUGH_ENV if key in environ}

def worker_runtime_environment(environ, profile_settings=None):
    """Build an isolated modem environment, then apply explicit per-modem settings."""
    from native_runtime import NATIVE_ENV
    from modem_settings import ENV
    result={key:value for key,value in environ.items()
            if not key.startswith(WORKER_SCRUB_PREFIXES)}
    result.update(NATIVE_ENV)
    result.update(worker_passthrough_environment(environ))
    for name,value in (profile_settings or {}).items():
        if name in ENV:
            result[ENV[name][0]]=str(value)
    result['EL_MODEM_FACTORY_NATIVE']='1'
    return result

def worker_profile_settings(directory):
    """Read the selected modem's persisted extended settings, if present."""
    profile=Path(directory)/'profiles.json'
    if not profile.exists():
        return {}
    return json.loads(profile.read_text()).get('profiles',{}).get('0',{}).get('extended_settings',{})


def worker(path,fd=None):
    import signal
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    from modem_settings import ENV
    document=json.loads(Path(path).read_text()); spec=document['spec']; directory=Path(document['directory'])

    profile=directory/'profiles.json'
    runtime_env=worker_runtime_environment(os.environ.copy(),worker_profile_settings(directory))
    os.environ.clear()
    os.environ.update(runtime_env)

    boot={'instance.name':'name','serial.backend':'serial','serial.tcp_port':'serial_port',
          'serial.device':'serial_device','serial.baud':'serial_baud',
          'sip.bind_host':'sip_host','sip.bind_port':'sip_port','sip.media_host':'media_host','sip.media_port':'media_port'}
    for name,key in boot.items(): os.environ[ENV[name][0]]=str(spec[key])
    for name,key in boot.items():
        value=os.environ[ENV[name][0]]
        spec[key]=int(value) if key.endswith(('port','baud')) else value
    from modem_supervisor import ModemSpec
    ModemSpec(**spec).validate()
    from modem_control import WorkerControl
    from modem_serial import DriverSerial, LegacySerial
    from modem_dce import main
    link=str(directory/'serial')
    if Path(link).exists() or Path(link).is_symlink():
        raise ValueError(f'refusing to replace existing serial path: {link}')
    if fd is None:
        import socket
        endpoint=document['control']
        connection=socket.create_connection(('127.0.0.1',endpoint['port']),timeout=10)
        connection.sendall(endpoint['token'].encode())
        control=WorkerControl(connection)
    else:control=WorkerControl(fd)
    serial=None
    def create_serial():
        nonlocal serial
        if spec['serial']=='com':
            from modem_serial_com import ComSerial
            serial=ComSerial(device=spec['serial_device'],baud=spec['serial_baud'])
        elif spec['serial']=='pipe':
            from modem_serial_pipe import PipeSerial
            serial=PipeSerial(device=spec['serial_device'],baud=spec['serial_baud'])
        elif spec['serial']=='device':
            from modem_serial_device import DeviceSerial
            serial=DeviceSerial(device=spec['serial_device'],baud=spec['serial_baud'])
        else:
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
    # CTRL_BREAK permits orderly cleanup of Windows process groups.
    import signal
    if hasattr(signal,'SIGBREAK'):
        signal.signal(signal.SIGBREAK,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if '--worker' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--control-fd',type=int)
        a=p.parse_args();worker(a.worker,a.control_fd);return
    supervisor_flags={'--tui','--headless','--config','--modems','--serial-backend','--serial-device','--serial-baud','--help','-h'}
    if len(sys.argv)>1 and not any(x.split('=')[0] in supervisor_flags for x in sys.argv[1:]):
        from modem_dce import main as legacy
        legacy();return
    p=argparse.ArgumentParser(prog='el-modem',description='EL-MODEM multi-modem dashboard and native DSP runtime')
    ui=p.add_mutually_exclusive_group();ui.add_argument('--tui',action='store_true');ui.add_argument('--headless',action='store_true')
    p.add_argument('--config',default=str(Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'el-modem'/'config.json'))
    from modem_serial import default_backend
    serial_choices=('tcp','pipe','com') if sys.platform=='win32' else ('driver','pty','tcp','device') if sys.platform.startswith('linux') else ('pty','tcp','device')
    p.add_argument('--modems',type=int);p.add_argument('--serial-backend',choices=serial_choices,default=default_backend())
    p.add_argument('--serial-device',default='',help='COM port, Windows pipe name, or POSIX tty path (e.g. /dev/ttyUSB0 or /dev/cu.usbserial-*)')
    p.add_argument('--serial-baud',type=int,default=115200,help='DTE serial baud rate for COM, pipe, or device backends')
    a=p.parse_args()
    if a.modems is not None and a.modems<1: p.error('--modems must be positive')
    if not a.headless and not sys.stdin.isatty(): p.error('interactive terminal required; use --headless')
    from modem_supervisor import Supervisor
    supervisor=Supervisor(a.config)
    try:
        supervisor.acquire()
        count=a.modems if a.modems is not None else max(1,len(supervisor.specs))
        while len(supervisor.specs)<count: supervisor.add(serial=a.serial_backend,device=a.serial_device,baud=a.serial_baud)
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
