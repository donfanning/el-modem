"""Driver-backed modem serial endpoint and explicit legacy adapters.

Windows native COM lives in modem_serial_com.py (same interface plus real
modem-control lines); this module keeps the Linux driver and the
PTY/TCP legacy adapters."""
import array
import os
import select
import socket
import time
import sys
if os.name == 'posix':
    import fcntl
    import termios
    import tty
from serial_port import PtySerial, TcpSerial


def default_backend():
    return 'driver' if sys.platform.startswith('linux') else 'tcp' if sys.platform=='win32' else 'pty'


# Pinned serialsim UAPI plus EL-MODEM prerequisite-driver open-count extension.
ALLOC=0x5391
SET_NULL=0x54e4
SET_REMOTE=0x54e5
GET_REMOTE=0x800454e9
GET_OPEN=0x800454ec

class DriverSerial:
    def __init__(self, link):
        if not sys.platform.startswith("linux"):raise ValueError("serialsim driver backend requires Linux; choose PTY/TCP")
        self.allocator=os.open('/dev/ttyPipe',os.O_RDWR|os.O_CLOEXEC)
        try:
            self.number=fcntl.ioctl(self.allocator,ALLOC,0)
            self.client=f'/dev/ttyPipeA{self.number}'
            endpoint=f'/dev/ttyPipeB{self.number}'
            deadline=time.monotonic()+3
            while not os.access(endpoint,os.R_OK|os.W_OK) or not os.access(self.client,os.R_OK|os.W_OK):
                if time.monotonic()>deadline: raise OSError('udev did not grant access to serialsim device pair')
                time.sleep(.01)
            self.fd=os.open(endpoint,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK|os.O_CLOEXEC)
            tty.setraw(self.fd)
            attrs=termios.tcgetattr(self.fd); attrs[4]=attrs[5]=termios.B115200
            attrs[2]|=termios.CLOCAL; termios.tcsetattr(self.fd,termios.TCSANOW,attrs)
            fcntl.ioctl(self.fd,SET_NULL,0)
            self.name=self.client; self.link=link
            self.pending=bytearray(); self.previous_dtr=False
            self.get_open_count() # Require patched driver rather than invent ownership.
            if link: os.symlink(self.client,link); self.name=link
            self.update_lines(False,False,True,False)
        except BaseException:
            if hasattr(self,'fd'): os.close(self.fd)
            os.close(self.allocator); raise
    def get_open_count(self):
        count=array.array('I',[0]); fcntl.ioctl(self.fd,GET_OPEN,count,True); return count[0]
    def client_present(self): return self.get_open_count()>0
    def remote(self):
        bits=array.array('I',[0]); fcntl.ioctl(self.fd,GET_REMOTE,bits,True); return bits[0]
    def signals(self):
        bits=self.remote()
        return dict(TR=bool(bits & termios.TIOCM_DTR),RTS=bool(bits & termios.TIOCM_RTS),CTS=bool(bits & termios.TIOCM_CTS))
    def update_lines(self,carrier,ring,ready,hardware,backlog=0):
        # Receiver backpressure controls peer CTS; RTS controls output flushing.
        cts=ready and (not hardware or backlog < 40960)
        mask=termios.TIOCM_CAR|termios.TIOCM_DSR|termios.TIOCM_CTS|termios.TIOCM_RNG
        value=(termios.TIOCM_CAR if carrier else 0)|(termios.TIOCM_DSR if ready else 0)|(termios.TIOCM_RNG if ring else 0)|(termios.TIOCM_CTS if cts else 0)
        fcntl.ioctl(self.fd,SET_REMOTE,(mask<<16)|value)
        self.hardware=hardware
    def open_terminal(self):
        if self.client_present(): raise ValueError('serial port is owned by an external application')
        fd=os.open(self.client,os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY|os.O_CLOEXEC)
        tty.setraw(fd)
        attrs=termios.tcgetattr(fd); attrs[4]=attrs[5]=termios.B115200
        attrs[2]|=termios.CLOCAL|termios.CRTSCTS; termios.tcsetattr(fd,termios.TCSANOW,attrs)
        return fd
    def fds(self): return [self.fd]
    def write_fds(self):
        return [self.fd] if self.pending and (not self.hardware or self.remote() & termios.TIOCM_RTS) else []
    def service(self,readable,writable=()):
        if self.fd in writable: self._flush()
        dtr=bool(self.remote() & termios.TIOCM_DTR)
        dropped=self.previous_dtr and not dtr; self.previous_dtr=dtr
        data=b''
        if self.fd in readable:
            try: data=os.read(self.fd,4096)
            except BlockingIOError: pass
        return data,dropped
    def _flush(self):
        if self.hardware and not self.remote() & termios.TIOCM_RTS: return
        try:
            n=os.write(self.fd,self.pending); del self.pending[:n]
        except BlockingIOError: pass
    def write(self,data):
        if len(self.pending)+len(data)>65536: raise BufferError('serial receive queue exhausted')
        self.pending.extend(data); self._flush()
    def close(self):
        try: self.update_lines(False,False,False,False)
        finally:
            os.close(self.fd); os.close(self.allocator)
            if self.link and os.path.islink(self.link): os.unlink(self.link)

class LegacySerial:
    def __init__(self,kind,link=None,address=None):
        self.kind=kind
        if kind=='pty': self.backend=PtySerial(link)
        else:
            host,port=address.rsplit(':',1); self.backend=TcpSerial(host,int(port),exclusive=True)
        self.name=self.backend.name
    def __getattr__(self,name): return getattr(self.backend,name)
    def client_present(self):
        if self.kind=='tcp': return bool(self.backend._conn)
        poll=select.poll();poll.register(self.backend.master,select.POLLHUP)
        return not any(flags & select.POLLHUP for _,flags in poll.poll(0))
    def signals(self): return dict(TR=self.client_present(),RTS=None,CTS=None)
    def update_lines(self,*args): pass
    def open_terminal(self):
        if self.client_present(): raise ValueError('serial port is owned by an external application')
        if self.kind=='pty':
            fd=os.open(self.backend.name,os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY)
            try:
                tty.setraw(fd)
                fcntl.ioctl(fd,termios.TIOCEXCL)
                return fd
            except BaseException:
                os.close(fd)
                raise
        host,port=self.backend.name[6:].rsplit(':',1)
        conn=socket.create_connection((host,int(port))); conn.setblocking(False); return conn
    def release_terminal(self,fd):
        if self.kind=='pty': fcntl.ioctl(fd,termios.TIOCNXCL)
