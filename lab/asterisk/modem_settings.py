"""Shared Hayes/control configuration; no DSP or UI dependencies."""
import json
import os
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Setting:
    name: str
    kind: str
    apply: str = 'immediate'
    command: str = ''
    secret: bool = False
    choices: tuple = ()
    minimum: int = 0
    maximum: int = 65535

    def parse(self, value):
        if self.kind == 'int':
            if isinstance(value, bool):
                raise ValueError('integer required')
            value = int(str(value))
            if not self.minimum <= value <= self.maximum:
                raise ValueError(f'expected {self.minimum}..{self.maximum}')
        else:
            value = str(value)
            if len(value) > 512 or any(ord(c) < 32 for c in value):
                raise ValueError('invalid text')
        if self.choices and value not in self.choices:
            raise ValueError(f'expected one of {self.choices}')
        return value

NETWORK = {
 'sip.server':100, 'sip.port':101, 'sip.user':102, 'sip.password':103,
 'sip.registrar':104, 'sip.caller_number':105, 'sip.contact_host':106,
 'iax.server':110, 'iax.port':111, 'iax.user':112, 'iax.password':113,
 'iax.caller_number':114, 'iax.register':115, 'network.trunk':120,
}
FIELDS = {'modem.echo':'echo', 'modem.quiet':'quiet', 'modem.verbose':'verbose',
 'modem.dtr':'dtr_behavior', 'modem.carrier':'carrier_detect',
 'modem.flow_control':'flow_control', 'modem.error_control':'error_control',
 'modem.compression':'compression', 'modem.auto_rate':'auto_rate_mode', 'modem.result_mode':'result_mode'}
ENV = {
 'v42.cerror_recovery':('V42_CERROR_RECOVERY','0',('0','1')),
 'dsp.eq_span':('V34_PHASE4_EQ_SPAN','24',()),
 'dsp.trn_clock':('V34_TRN_CLOCK','1',('0','1')),
 'dsp.clock_fit_gate':('V34_CLOCK_FIT_GATE','1',('0','1')),
 'dsp.survivor_sync':('V34_SURVIVOR_STATE_SYNC','1',('0','1')),
 'dsp.l2_guard':('V34_L2_GUARD','1',('0','1')),
 'dsp.retrain_after_e':('V34_RETRAIN_AFTER_E','1',('0','1')),
 'dsp.input_power_guard':('V34_INPUT_POWER_GUARD','fade',('off','fade')),
 'dsp.answerer_retrain':('V34_ANSWERER_RETRAIN','1',('0','1')),
 'dsp.caller_retrain':('V34_CALLER_RETRAIN','1',('0','1')),
 'dsp.native_feedback':('V34_NATIVE_FEEDBACK','1',('0','1')),
 'dsp.native_interp':('V34_SYMBOL_INTERP_NATIVE','1',('0','1')),
 'dsp.native_clock':('V34_TRN_CLOCK_NATIVE','1',('0','1')),
 'dsp.capture_async':('MODEM_CAPTURE_ASYNC','1',('0','1')),
 'dsp.packet_poll':('V34_TRACKING_PACKET_POLL','1',('0','1')),
 'dsp.deadline_drain':('EL_MODEM_RTP_DEADLINE_DRAIN','1',('0','1')),
 'dsp.training_stride':('EL_MODEM_DSP_TRAINING_STRIDE','4',()),
 'dsp.timing_mode':('V34_TIMING_MODE','centroid',('data-gradient','centroid')),
 'dsp.trajectory_grid':('V34_TRAJECTORY_GRID','1',('0','1')),
 'dsp.connected_rate':('V34_CONNECTED_RATE','1',('0','1')),
 'dsp.quality_monitor':('V34_QUALITY_MONITOR','1',('0','1')),
 'dsp.auto_rate':('V34_AUTO_RATE','1',('0','1')),
 'dsp.native_trellis':('V34_NATIVE_TRELLIS','1',('0','1')),
 'runtime.preimport':('V34_PREIMPORT','1',('0','1')),
 'runtime.gc_pause':('EL_MODEM_GC_PAUSE','1',('0','1')),
 'runtime.blas_threads':('OPENBLAS_NUM_THREADS','1',()),
 'recording.enabled':('EL_MODEM_RECORDING','1',('0','1')),
 'recording.seconds':('EL_MODEM_RECORDING_SECONDS','600',()),
 'instance.name':('EL_MODEM_NAME','Modem',()),
 'serial.backend':('EL_MODEM_SERIAL_BACKEND','driver',('driver','pty','tcp','com','pipe','device')),
 'serial.tcp_port':('EL_MODEM_SERIAL_PORT','9100',()),
 'serial.device':('EL_MODEM_SERIAL_DEVICE','',()),
 'serial.baud':('EL_MODEM_SERIAL_BAUD','115200',()),
 'sip.bind_host':('EL_MODEM_SIP_BIND_HOST','127.0.0.1',()),
 'sip.bind_port':('EL_MODEM_SIP_BIND_PORT','5062',()),
 'sip.media_host':('EL_MODEM_MEDIA_HOST','127.0.0.1',()),
 'sip.media_port':('EL_MODEM_MEDIA_PORT','16000',()),
}
LIMITS = {'modem.echo':1,'modem.quiet':1,'modem.verbose':1,'modem.dtr':3,
 'modem.carrier':1,'modem.flow_control':6,'modem.error_control':3,
 'modem.compression':1,'modem.auto_rate':2,'modem.result_mode':4}

def registry(dce):
    result = {name:Setting(name,'int',maximum=LIMITS[name]) for name in FIELDS}
    result.update({name:Setting(name,'text','idle',f'S{reg}',reg in (103,113)) for name,reg in NETWORK.items()})
    result['network.trunk']=Setting('network.trunk','text','idle','S120',choices=('0','1'))
    result.update({f's{reg}':Setting(f's{reg}','int',command=f'S{reg}',maximum=255)
                   for reg in dce.registers if reg != 1})
    result.update({f'modem.{name}':Setting(f'modem.{name}','text','idle',command='+MS')
                   for name in ('modulation','automatic','min_rate','max_rate')})
    result.update({name:Setting(name,'text','restart',choices=choices)
                   for name,(_,_,choices) in ENV.items()})
    for name in ('dsp.eq_span','dsp.training_stride','runtime.blas_threads'):
        result[name] = Setting(name,'int','restart',minimum=1,maximum=128)
    for name in ('serial.tcp_port','sip.bind_port','sip.media_port'):
        result[name]=Setting(name,'int','restart',minimum=1,maximum=65535)
    import sys
    serial_backends=(('tcp','pipe','com') if sys.platform=='win32' else
                     ('driver','pty','tcp','device') if sys.platform.startswith('linux') else
                     ('pty','tcp','device'))
    result['serial.backend']=Setting('serial.backend','text','restart',choices=serial_backends)
    result['serial.baud']=Setting('serial.baud','int','restart',minimum=300,maximum=921600)
    result['recording.seconds'] = Setting('recording.seconds','int','restart',minimum=1,maximum=86400)
    return result

MS = {'modem.modulation':'mod','modem.automatic':'auto','modem.min_rate':'min','modem.max_rate':'max'}

def read(dce, name):
    if name in FIELDS: return int(getattr(dce,FIELDS[name]))
    if name in NETWORK: return dce.net[NETWORK[name]]
    if name in MS: return dce.ms[MS[name]]
    if name in ENV:
        env,default,_ = ENV[name]
        return dce.extended_settings.get(name,os.environ.get(env,default))
    if re.fullmatch(r's\d+',name) and name in registry(dce): return dce.registers[int(name[1:])]
    raise ValueError('unknown setting')

def set_value(dce, name, value):
    spec = registry(dce).get(name)
    if spec is None: raise ValueError('unknown setting')
    value = spec.parse(value)
    if name in ('sip.bind_host','sip.media_host'):
        import socket
        try: socket.inet_aton(value)
        except OSError as exc: raise ValueError('IPv4 bind address required') from exc
    if name == 'instance.name' and not value: raise ValueError('name cannot be empty')
    busy = dce.online or dce.outgoing_pending or dce.incoming_ringing or dce._answering
    if busy: raise ValueError('settings cannot change during an active call')
    if name in NETWORK:
        if not dce._set_net(NETWORK[name],str(value)): raise ValueError('invalid network setting')
    elif name in FIELDS: setattr(dce,FIELDS[name],value)
    elif name in MS:
        candidate = dict(dce.ms)
        candidate[MS[name]] = value
        if name != 'modem.modulation': candidate[MS[name]] = int(value)
        # The Hayes +MS parser is the authoritative profile validator.
        command = 'AT+MS={mod},{auto},{min},{max}'.format(**candidate)
        quiet, verbose = dce.quiet, dce.verbose
        dce.quiet, dce.verbose = False, True
        try: response = dce._execute(command)
        finally: dce.quiet, dce.verbose = quiet, verbose
        if b'ERROR' in response: raise ValueError('invalid modulation profile')
    elif name in ENV: dce.extended_settings[name] = str(value)
    else: dce.registers[int(name[1:])] = value
    return spec.apply

def snapshot(dce):
    out=[]
    for name,spec in registry(dce).items():
        value=read(dce,name)
        out.append(dict(name=name,kind=spec.kind,apply=spec.apply,command=spec.command,
                        secret=spec.secret,choices=spec.choices,value=('********' if value else '') if spec.secret else value,
                        active=os.environ.get(ENV[name][0],ENV[name][1]) if name in ENV else value))
        if spec.secret: out[-1]['active']=out[-1]['value']
    return out

def hayes(dce,line):
    suffix=line[6:].strip()
    try:
        if suffix=='?': lines=[json.dumps(x,ensure_ascii=True) for x in snapshot(dce)]
        elif suffix=='=?': lines=[x['name'] for x in snapshot(dce)]
        elif suffix.startswith('='):
            args=json.loads('['+suffix[1:]+']')
            if len(args)!=2: raise ValueError('two parameters required')
            timing=set_value(dce,str(args[0]).lower(),args[1]); lines=[f'APPLY: {timing}']
        else: raise ValueError('invalid +CFG command')
        return b''.join(dce._line(x) for x in lines)+dce._result('OK')
    except (ValueError,TypeError,OverflowError): return dce._result('ERROR')
