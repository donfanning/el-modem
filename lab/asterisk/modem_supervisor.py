"""One process per modem, with private configuration and bounded IPC."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
    serial: str = 'driver'
    serial_port: int = 9100
    enabled: bool = True

    def validate(self):
        import re
        if not re.fullmatch(r'modem-[1-9][0-9]*',self.id): raise ValueError('invalid modem ID')
        if not self.name or len(self.name)>80 or any(ord(c)<32 for c in self.name): raise ValueError('invalid modem name')
        if self.serial not in ('driver','pty','tcp'): raise ValueError('unknown serial backend')
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
            'sip.bind_host':'sip_host','sip.bind_port':'sip_port','sip.media_host':'media_host','sip.media_port':'media_port'}
    def effective(self, spec):
        from dataclasses import replace
        profile=self.root/spec.id/'profiles.json'
        changes={}
        if profile.exists():
            saved=json.loads(profile.read_text()).get('profiles',{}).get('0',{}).get('extended_settings',{})
            for name,key in self.BOOT.items():
                if name in saved: changes[key]=int(saved[name]) if key.endswith('port') else saved[name]
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
    def acquire(self):
        import fcntl
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.lock=open(str(self.path)+'.lock','a')
        try: fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.lock.close();self.lock=None; raise ValueError('configuration is already owned by another supervisor')
    def save(self): self.validate(); atomic_json(self.path,dict(schema=1,modems=[asdict(s) for s in self.specs]))
    def add(self, name=None, serial='driver'):
        taken={s.id for s in self.specs}; number=1
        while f'modem-{number}' in taken or (self.root/f'modem-{number}').exists(): number+=1
        spec=ModemSpec(f'modem-{number}',name or f'Modem {number}',sip_port=5060+number*2,
                       media_port=15996+number*4,serial_port=9099+number,serial=serial).validate()
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
        left,right=socket.socketpair()
        atomic_json(directory/'worker.json',dict(spec=asdict(spec),directory=str(directory.resolve())))
        argv=([sys.executable] if getattr(sys,'frozen',False) else
              [sys.executable,str(Path(__file__).with_name('el_modem.py'))])
        log=open(directory/'worker.log','ab',buffering=0)
        try:
            proc=subprocess.Popen(argv+['--worker',str((directory/'worker.json').resolve()),'--control-fd',str(right.fileno())],
                                  pass_fds=(right.fileno(),),stdout=log,stderr=log)
        except BaseException:
            left.close(); raise
        finally: right.close();log.close()
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
                proc.terminate()
                try: proc.wait(timeout=8)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
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
