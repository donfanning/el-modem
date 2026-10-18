"""One process per modem, with private configuration and bounded IPC."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from modem_serial import default_backend
from modem_control import Channel
from modem_storage import atomic_json

@dataclass
class ModemSpec:
    id: str
    name: str
    sip_host: str = '127.0.0.1'
    sip_port: int = 5062
    media_host: str = '127.0.0.1'
    media_port: int = 16000
    serial: str = field(default_factory=default_backend)
    serial_port: int = 9100
    serial_device: str = ''
    serial_baud: int = 115200
    enabled: bool = True

    def validate(self):
        import re
        if not re.fullmatch(r'modem-[1-9][0-9]*',self.id): raise ValueError('invalid modem ID')
        if not self.name or len(self.name)>80 or any(ord(c)<32 for c in self.name): raise ValueError('invalid modem name')
        if self.serial not in ('driver','pty','tcp','com','pipe','device'): raise ValueError('unknown serial backend')
        if self.serial=='device':
            if os.name!='posix': raise ValueError('native serial-device backend requires Linux or macOS')
            from modem_serial_device import normalize_device, SUPPORTED_BAUDS
            try:self.serial_device=normalize_device(self.serial_device)
            except ValueError:raise
            if isinstance(self.serial_baud,bool) or self.serial_baud not in SUPPORTED_BAUDS:
                raise ValueError('unsupported native serial-device baud rate')
        if self.serial=='com':
            from modem_serial_com import normalize_com_device, BAUD_MINIMUM, BAUD_MAXIMUM
            try: normalize_com_device(self.serial_device)
            except ValueError: raise ValueError('invalid COM device (expected COMn, e.g. COM3)')
            if isinstance(self.serial_baud,bool) or not isinstance(self.serial_baud,int) or not BAUD_MINIMUM<=self.serial_baud<=BAUD_MAXIMUM:
                raise ValueError('invalid serial baud (expected %d..%d)' % (BAUD_MINIMUM, BAUD_MAXIMUM))
        if self.serial=='pipe':
            from modem_serial_pipe import normalize_pipe_name, BAUD_MINIMUM, BAUD_MAXIMUM
            try: normalize_pipe_name(self.serial_device)
            except ValueError: raise ValueError('invalid pipe name (expected \\\\.\\pipe\\ElModemSerial)')
            if isinstance(self.serial_baud,bool) or not isinstance(self.serial_baud,int) or not BAUD_MINIMUM<=self.serial_baud<=BAUD_MAXIMUM:
                raise ValueError('invalid serial baud (expected %d..%d)' % (BAUD_MINIMUM, BAUD_MAXIMUM))
        for value in (self.sip_port,self.media_port,self.serial_port):
            if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=65535: raise ValueError('invalid port')
        for host in (self.sip_host,self.media_host): socket.inet_aton(host)
        return self

class Supervisor:
    def __init__(self, path):
        self.path=Path(path)
        self.root=self.path.parent/'instances'
        self.specs=[]; self.workers={}; self.states={}; self.responses={}; self.events=[]; self.sequence=0
        self.lock=None
        if self.path.exists():
            data=json.loads(self.path.read_text())
            if data.get('schema')!=1: raise ValueError('unsupported configuration schema')
            self.specs=[ModemSpec(**s).validate() for s in data['modems']]
        self.validate()
    BOOT = {'instance.name':'name','serial.backend':'serial','serial.tcp_port':'serial_port',
            'serial.device':'serial_device','serial.baud':'serial_baud',
            'sip.bind_host':'sip_host','sip.bind_port':'sip_port','sip.media_host':'media_host','sip.media_port':'media_port'}
    def effective(self, spec):
        from dataclasses import replace
        profile=self.root/spec.id/'profiles.json'
        changes={}
        if profile.exists():
            saved=json.loads(profile.read_text()).get('profiles',{}).get('0',{}).get('extended_settings',{})
            for name,key in self.BOOT.items():
                if name in saved: changes[key]=int(saved[name]) if key.endswith(('port','baud')) else saved[name]
        return replace(spec,**changes).validate()
    def configure(self, spec, changes):
        from dataclasses import replace
        candidate=replace(spec,**changes).validate()
        position=self.specs.index(spec)
        profile=self.root/spec.id/'profiles.json'
        previous=json.loads(profile.read_text()) if profile.exists() else None
        if previous is not None:
            import copy
            updated=copy.deepcopy(previous)
            settings=updated.setdefault('profiles',{}).setdefault('0',{}).setdefault('extended_settings',{})
            for name,key in self.BOOT.items():
                if key in changes:settings[name]=str(changes[key])
            atomic_json(profile,updated)
        self.specs[position]=candidate
        try:self.save()
        except BaseException:
            self.specs[position]=spec
            if previous is not None:atomic_json(profile,previous)
            raise
    def validate(self):
        ids=set(); ports=set()
        for spec in self.specs:
            spec.validate()
            try: spec=self.effective(spec)
            except (ValueError,TypeError,OSError): pass
            if spec.id in ids: raise ValueError('duplicate modem ID')
            ids.add(spec.id)
            for host,port in ((spec.sip_host,spec.sip_port),(spec.media_host,spec.media_port)):
                if (host,port) in ports: raise ValueError('colliding SIP/RTP ports')
                ports.add((host,port))
            if spec.serial=='tcp':
                key=('tcp',spec.serial_port)
                if key in ports: raise ValueError('colliding serial ports')
                ports.add(key)
            if spec.serial=='com':
                key=('com',spec.serial_device.strip().upper())
                if key in ports: raise ValueError('colliding serial devices')
                ports.add(key)
            if spec.serial=='device':
                key=('device',os.path.realpath(spec.serial_device))
                if key in ports:raise ValueError('colliding serial devices')
                ports.add(key)
            if spec.serial=='pipe':
                from modem_serial_pipe import normalize_pipe_name
                key=('pipe',normalize_pipe_name(spec.serial_device).casefold())
                if key in ports:
                    raise ValueError('colliding serial pipes; each modem needs a distinct driver pipe endpoint')
                ports.add(key)
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.lock=open(str(self.path)+'.lock','a+b')
        try:
            if sys.platform=='win32':
                import msvcrt
                self.lock.seek(0);self.lock.write(b'0');self.lock.flush();self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.lock.close();self.lock=None; raise ValueError('configuration is already owned by another supervisor')
    def save(self): self.validate(); atomic_json(self.path,dict(schema=1,modems=[asdict(s) for s in self.specs]))
    def _bindable(self, port, stream):
        """True when the OS will actually let us bind this TCP/UDP port.

        Windows reserves whole ranges (Hyper-V/WSL NAT, virtual adapters,
        exclusive-address holders) and refuses those binds with WSAEACCES
        (10013) rather than address-in-use. That error is invisible to a
        collision check, so the choice has to be probed.
        """
        for addr in ('127.0.0.1','0.0.0.0'):
            probe=socket.socket(socket.AF_INET,socket.SOCK_STREAM if stream else socket.SOCK_DGRAM)
            try:
                probe.bind((addr,port)); return True
            except OSError:
                continue
            finally:
                probe.close()
        return False
    def _pick_port(self, base, step, number, stream, kind):
        """Walk the documented ladder from its default until the OS allows a bind.

        Keeps the published defaults wherever they are free, and only moves on
        when a port is refused. The documented ladder is tried in full before
        falling back to an ephemeral port, so a crowded low range cannot walk
        this into another modem's ports, and a host that reserves the whole
        range still gets a usable port instead of a hard failure.
        """
        ceiling=base+step*64
        for port in range(base+number*step,ceiling,step):
            if self._bindable(port,stream): return port
        # Whole ladder refused. Let the OS hand out a free high port rather
        # than refusing to start; the chosen port is recorded in the config.
        probe=socket.socket(socket.AF_INET,socket.SOCK_STREAM if stream else socket.SOCK_DGRAM)
        try:
            probe.bind(('',0))
            return probe.getsockname()[1]
        finally:
            probe.close()
    def add(self, name=None, serial=None, device='', baud=115200):
        serial=serial or default_backend()
        if serial=='com' and not str(device or '').strip():
            raise ValueError('com backend needs an explicit serial device (e.g. COM3); create the instance in the TUI or pass --serial-device')
        if serial=='pipe':
            # Validate early so a typo surfaces here rather than in the worker.
            from modem_serial_pipe import normalize_pipe_name
            normalize_pipe_name(device)
        taken={s.id for s in self.specs}; number=1
        while f'modem-{number}' in taken or (self.root/f'modem-{number}').exists(): number+=1
        # Probe each port so a host with reserved ranges still starts. Only the
        # serial and media ports need this: SIP uses 5060+2n, which stays clear.
        media_port=self._pick_port(15996,4,number,False,'media')
        serial_port=self._pick_port(9099,1,number,True,'serial') if serial=='tcp' else 9099+number
        spec=ModemSpec(f'modem-{number}',name or f'Modem {number}',sip_port=5060+number*2,
                       media_port=media_port,serial_port=serial_port,serial=serial,
                       serial_device=str(device or '').strip(),serial_baud=baud).validate()
        self.specs.append(spec)
        try: self.save()
        except BaseException: self.specs.remove(spec); raise
        return spec
    def start(self, ident):
        if ident in self.workers: raise ValueError('modem is already running')
        spec=self.effective(next(s for s in self.specs if s.id==ident))
        self.validate()
        directory=self.root/ident
        directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        if (directory/'serial').exists() or (directory/'serial').is_symlink():
            raise ValueError('serial path already exists; refusing to replace it')
        # Check binding before spawning; worker still handles a subsequent race.
        for kind,host,port in ((socket.SOCK_DGRAM,spec.sip_host,spec.sip_port),
                               (socket.SOCK_DGRAM,spec.media_host,spec.media_port)):
            with socket.socket(socket.AF_INET,kind) as probe: probe.bind((host,port))
        if spec.serial=='driver' and not os.access('/dev/ttyPipe',os.R_OK|os.W_OK):
            raise ValueError('serial driver unavailable: install the documented serialsim prerequisite and device permissions, or explicitly choose PTY/TCP')
        if spec.serial=='com':
            if sys.platform!='win32':
                raise ValueError('COM serial backend requires Windows; choose PTY/TCP')
            from modem_serial_com import probe_device
            try: probe_device(spec.serial_device)
            except (OSError, ValueError) as exc:
                raise ValueError('COM port unavailable: %s' % exc)
        if spec.serial=='pipe':
            if sys.platform!='win32':
                raise ValueError('pipe serial backend requires Windows; choose PTY/TCP')
            from modem_serial_pipe import probe_device as probe_pipe
            try:
                # Name validation only: the driver attaches asynchronously and
                # may not be running yet, which is not a start-up failure.
                probe_pipe(spec.serial_device)
            except (OSError, ValueError) as exc:
                raise ValueError('pipe unavailable: %s' % exc)
        if spec.serial=='device':
            import stat
            try:
                st=os.stat(spec.serial_device)
                if not stat.S_ISCHR(st.st_mode):raise ValueError('not a character device')
                fd=os.open(spec.serial_device,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK)
                os.close(fd)
            except (OSError,ValueError) as exc:
                raise ValueError('serial device unavailable: %s' % exc)
        document=dict(spec=asdict(spec),directory=str(directory.resolve()))
        argv=([sys.executable] if getattr(sys,'frozen',False) else
              [sys.executable,str(Path(__file__).with_name('el_modem.py'))])
        argv+=['--worker',str((directory/'worker.json').resolve())]
        log=open(directory/'worker.log','ab',buffering=0)
        left=right=listener=proc=None
        try:
            if sys.platform=='win32':
                import secrets
                import hmac
                listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(4);listener.settimeout(30)
                token=secrets.token_hex(32)
                document['control']=dict(port=listener.getsockname()[1],token=token)
                atomic_json(directory/'worker.json',document)
                proc=subprocess.Popen(argv,stdout=log,stderr=log)
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    listener.settimeout(max(.01,deadline-time.monotonic()))
                    connection,_=listener.accept();connection.settimeout(min(2,max(.01,deadline-time.monotonic())))
                    auth=bytearray()
                    try:
                        while len(auth)<64:
                            block=connection.recv(64-len(auth))
                            if not block:break
                            auth.extend(block)
                        if hmac.compare_digest(bytes(auth),token.encode()):left=connection;break
                    except OSError:pass
                    connection.close()
                if left is None:raise TimeoutError('worker control authentication timed out')
            else:
                left,right=socket.socketpair()
                atomic_json(directory/'worker.json',document)
                proc=subprocess.Popen(argv+['--control-fd',str(right.fileno())],
                                      pass_fds=(right.fileno(),),stdout=log,stderr=log)
        except BaseException:
            if left is not None:left.close()
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
        finally:
            if right is not None:right.close()
            if listener is not None:listener.close()
            log.close()
        self.workers[ident]=(proc,Channel(left)); self.states[ident]=dict(state='starting',lights={},owner='none')
    def request(self,ident,action,**fields):
        if ident not in self.workers: raise ValueError('modem is not running')
        if self.states.get(ident,{}).get('state')=='starting': raise ValueError('modem is warming up; wait until ready')
        self.sequence+=1; request=self.sequence
        self.workers[ident][1].send(dict(id=request,action=action,**fields)); return request
    def poll(self):
        for ident,(proc,channel) in list(self.workers.items()):
            try:
                channel.flush()
                for message in channel.receive():
                    if message.get('event')=='status': self.states[ident]=message['value']
                    elif message.get('event')=='terminal': self.events.append((ident,message['hex']))
                    else: self.responses[message['id']]=message
                if len(self.events)>256: raise BufferError('terminal consumer is too slow')
                if proc.poll() is not None or channel.closed:
                    self.states[ident]=dict(state='failed',error=f'worker exited ({proc.poll()}); see {self.root/ident/"worker.log"}',lights={})
                    self.stop(ident,preserve_state=True)
            except (OSError,ValueError,BufferError) as exc:
                self.states[ident]=dict(state='failed',error=str(exc),lights={}); self.stop(ident,preserve_state=True)
    def stop(self,ident,preserve_state=False):
        item=self.workers.pop(ident,None)
        if item:
            proc,channel=item
            if proc.poll() is None:
                # Windows terminate() cannot run Python cleanup handlers.
                # Ask the DCE loop to hang up/archive and exit first on every OS.
                try:
                    channel.send(dict(action='shutdown'));channel.flush()
                    proc.wait(timeout=8)
                except (OSError,BufferError,subprocess.TimeoutExpired):
                    proc.terminate()
                    try:proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
            channel.close()
            serial_link=self.root/ident/'serial'
            if serial_link.is_symlink():serial_link.unlink()
        if not preserve_state: self.states[ident]=dict(state='stopped',lights={},owner='none')
    def remove(self,ident):
        self.stop(ident); self.specs=[s for s in self.specs if s.id!=ident];self.save()
        # Retain profiles and recordings for recovery; IDs cannot silently reuse them.
    def close(self):
        for ident in list(self.workers): self.stop(ident)
        if self.lock: self.lock.close(); self.lock=None
