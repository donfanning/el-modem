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
    def poll(self, dce, ser, registration, session, iax_registered=False,
             iax_configured=False):
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
            if getattr(self.terminal_backend,'terminal_is_internal',False):
                if self.pending:
                    self.terminal_backend.terminal_write(self.pending)
                    self.pending.clear()
                data=self.terminal_backend.terminal_read()
                if data:self.channel.send(dict(event='terminal',hex=data.hex()))
            elif self.pending:
                try:
                    n=self.terminal.send(self.pending) if isinstance(self.terminal,socket.socket) else os.write(self.terminal,self.pending); del self.pending[:n]
                except BlockingIOError: pass
            if not getattr(self.terminal_backend,'terminal_is_internal',False):
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
            # MR means registered to Asterisk, on either transport. It was a
            # hardcoded True, so it showed lit whether or not anything had
            # registered, and could never distinguish SIP from IAX.
            registered=bool(registration and registration.registered) or bool(iax_registered)
            # HS/DC/EC were left unwired in the dashboard mapping. Semantics follow the
            # owner's chosen panel: replace RS/SYN with DC/EC, retain ARQ.
            #   HS  high speed: the requested threshold is >= 9600 bps.
            #   DC  data compression: V.42bis in use, not merely supported.
            #   EC  error correction: V.42 or MNP 4 in use. The V.92 calls
            #       this EC; ARQ was the older name for the same thing.
            # The engine is reached the way modulation_name and link_stats
            # reach it - through _auto when the autodetect line is in play,
            # because AutoCallerLine surfaces bit_rate itself and its state
            # is a plain string, not an engine. Before it picks a standard
            # there is no line, and its bit_rate would dereference a None
            # engine.
            auto=getattr(session,'_auto',None)
            if auto is not None:
                # AutoCallerLine exposes bit_rate and connected itself. Its
                # state is a plain string ("listen"|"v22"|"bell103") rather
                # than an engine, and before it picks a standard there is no
                # line at all. It holds no v42_mode of its own, so the error
                # control state comes from the V.22 engine inside it.
                eng=None if getattr(auto,'state','')=='listen' else auto
                error_engine=getattr(auto,'_v22',None) if getattr(auto,'state','')=='v22' else None
            else:
                eng=getattr(session,'_v22',None)
                eng=getattr(eng,'engine',None) or eng      # V8Chain handover
                error_engine=eng
            v42=getattr(error_engine,'v42_mode','off')
            try: rate=int(getattr(eng,'bit_rate',0) or 0)
            except Exception: rate=0
            compress=v42=='lapm' and bool(getattr(error_engine,'v42_compression',False))
            terminal_present=getattr(ser, 'terminal_present', ser.client_present)()
            if self.terminal is not None:
                owner='dashboard'
            elif terminal_present is None:
                owner='unknown'
            else:
                owner='external' if terminal_present else 'none'
            signals.update(AA=bool(dce.registers[0]),CD=carrier if dce.carrier_detect else True,
                           OH=bool(carrier or dce.outgoing_pending or dce._answering),
                           RD=now<self.rd_until,SD=now<self.sd_until,MR=registered,
                           HS=rate>=9600,DC=compress,EC=v42=='lapm',ARQ=v42=='lapm',
                           RI=bool(dce.incoming_ringing))
            value=dict(state='connected' if carrier else 'dialing' if dce.outgoing_pending else 'ringing' if dce.incoming_ringing else 'idle',
                       lights=signals,serial=ser.name,port=getattr(ser,'presented_port',lambda:None)() or ser.name,owner=owner,
                       registration=('registered' if registered else
                                     'pending' if registration or iax_configured else 'disabled'),
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
            finally:
                if not getattr(self.terminal_backend,'terminal_is_internal',False):
                    self.terminal.close() if isinstance(self.terminal,socket.socket) else os.close(self.terminal)
        self.terminal=None; self.terminal_backend=None; self.pending.clear()
    def close(self): self.close_terminal(); self.channel.close()
