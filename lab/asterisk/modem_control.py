"""Bounded local IPC. Worker commands are executed only in the DCE loop."""
import json
import os
import socket
import time

class Channel:
    LIMIT = 262144
    def __init__(self, sock):
        self.sock=sock; sock.setblocking(False)
        self.input=bytearray(); self.output=bytearray(); self.closed=False
    def send(self, message):
        data=json.dumps(message,separators=(',',':')).encode()+b'\n'
        if len(self.output)+len(data)>self.LIMIT: raise BufferError('control consumer is too slow')
        self.output.extend(data)
    def flush(self):
        if not self.output: return
        try:
            sent=self.sock.send(self.output); del self.output[:sent]
        except BlockingIOError: pass
    def receive(self):
        try:
            data=self.sock.recv(65536)
            if not data: self.closed=True; return []
            self.input.extend(data)
        except BlockingIOError: pass
        if len(self.input)>self.LIMIT: raise ValueError('control message exceeds limit')
        result=[]
        for _ in range(16):
            position=self.input.find(b'\n')
            if position<0: break
            result.append(json.loads(self.input[:position])); del self.input[:position+1]
        return result
    def close(self): self.sock.close(); self.closed=True

class WorkerControl:
    def __init__(self, fd):
        self.channel=Channel(fd if isinstance(fd,socket.socket) else socket.socket(fileno=fd))
        self.next_status=0.; self.terminal=None; self.terminal_backend=None; self.pending=bytearray()
        self.rd_until=0.; self.sd_until=0.; self.counts=(0,0)
    def poll(self, dce, ser, registration, session):
        from modem_settings import snapshot, set_value
        now=time.monotonic()
        for message in self.channel.receive():
            try:
                action=message['action']
                if action=='shutdown':raise KeyboardInterrupt()
                elif action=='settings': value=snapshot(dce)
                elif action=='set': value=set_value(dce,message['name'],message['value'])
                elif action=='save':
                    dce._store_profile(0); value='saved'
                elif action=='hangup':
                    dce._execute('ATH0'); dce.notify_call_progress('NO CARRIER'); value='hung up'
                elif action=='terminal.open':
                    if self.terminal is not None: raise ValueError('terminal already open')
                    self.terminal=ser.open_terminal(); self.terminal_backend=ser; value='acquired'
                elif action=='terminal.close':
                    self.close_terminal(); value='released'
                elif action=='terminal.write':
                    if self.terminal is None: raise ValueError('terminal does not own modem')
                    data=bytes.fromhex(message['hex'])
                    if len(data)>4096 or len(self.pending)+len(data)>65536: raise ValueError('terminal input queue full')
                    self.pending.extend(data); value=len(data)
                else: raise ValueError('unknown action')
                self.channel.send(dict(id=message.get('id'),ok=True,value=value))
            except (ValueError,KeyError,TypeError,OSError) as exc:
                self.channel.send(dict(id=message.get('id'),ok=False,error=str(exc)))
        if self.channel.closed: raise KeyboardInterrupt()
        if self.terminal is not None:
            if self.pending:
                try:
                    n=self.terminal.send(self.pending) if isinstance(self.terminal,socket.socket) else os.write(self.terminal,self.pending); del self.pending[:n]
                except BlockingIOError: pass
            try:
                data=self.terminal.recv(4096) if isinstance(self.terminal,socket.socket) else os.read(self.terminal,4096)
                if data: self.channel.send(dict(event='terminal',hex=data.hex()))
            except BlockingIOError: pass
        if now>=self.next_status and not self.channel.output:
            self.next_status=now+.1
            call=dce._call or {}
            counts=(call.get('chars_received',0),call.get('chars_sent',0))
            if counts[0]!=self.counts[0]: self.rd_until=now+.15
            if counts[1]!=self.counts[1]: self.sd_until=now+.15
            self.counts=counts
            carrier=bool(dce.online or (session and session.negotiation_complete))
            signals=ser.signals()
            signals.update(AA=bool(dce.registers[0]),CD=carrier if dce.carrier_detect else True,
                           OH=bool(carrier or dce.outgoing_pending or dce._answering),
                           RD=now<self.rd_until,SD=now<self.sd_until,MR=True,
                           RI=bool(dce.incoming_ringing))
            value=dict(state='connected' if carrier else 'dialing' if dce.outgoing_pending else 'ringing' if dce.incoming_ringing else 'idle',
                       lights=signals,serial=ser.name,owner='terminal' if self.terminal is not None else 'external' if ser.client_present() else 'none',
                       registration='registered' if registration and registration.registered else 'pending' if registration else 'disabled',
                       error_control=dce.error_control,compression=dce.compression,profile=dict(dce.ms),
                       call={key:call[key] for key in ('speed','suffix','modulation','chars_sent','chars_received') if key in call},
                       recording=os.environ.get('MODEM_CAPTURE',''),
                       health=self.health(session))
            self.channel.send(dict(event='status',value=value))
        self.channel.flush()
    @staticmethod
    def health(session):
        if session is None: return dict(mean_dsp_ms=0, maximum_sender_lateness_ms=0)
        stats=session.realtime
        mean=sum(stats.sums.get(key,0)/max(1,stats.counts.get(key,0))
                 for key in ('data_tx_seconds','data_rx_seconds'))
        return dict(mean_dsp_ms=round(mean*1000,3),
                    maximum_sender_lateness_ms=round(stats.maxima.get('data_sender_lateness_seconds',0)*1000,3))
    def close_terminal(self):
        if self.terminal is not None:
            try:
                release=getattr(self.terminal_backend,'release_terminal',None)
                if release is not None: release(self.terminal)
            finally: (self.terminal.close() if isinstance(self.terminal,socket.socket) else os.close(self.terminal))
        self.terminal=None; self.terminal_backend=None; self.pending.clear()
    def close(self): self.close_terminal(); self.channel.close()
