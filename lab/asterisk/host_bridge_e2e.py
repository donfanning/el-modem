#!/usr/bin/env python3
"""Host-side end to end: two serial bridges (127.0.0.1:9100 and :9101) give two host PTYs; one "terminal" dials the
other through Asterisk, they exchange data, the caller closes its port while online, and the far side must see
NO CARRIER (hangup on terminal close). Needs the lab up in a mode that connects quickly (any). Exit 0 on success."""
import os, sys, time, select, subprocess, signal
here = os.path.dirname(os.path.abspath(__file__))
link0, link1 = "/tmp/softmodem0", "/tmp/softmodem1"
procs=[subprocess.Popen([sys.executable,os.path.join(here, "serial_bridge.py"),"client","--connect",f"127.0.0.1:{p}","--baud-control-port",str(c),"--link",l]) for p,c,l in ((9100,9110,link0),(9101,9111,link1))]
time.sleep(1.5)
def op(p): return os.open(p, os.O_RDWR|os.O_NOCTTY)
def wait(fd,pat,t):
    out=b""; end=time.time()+t
    while time.time()<end and pat not in out:
        if select.select([fd],[],[],0.2)[0]:
            try: out+=os.read(fd,4096)
            except OSError: break
    return out
try:
    b=op(link1); os.write(b,b"ATZ\r"); print("B:",wait(b,b"OK",3)); os.write(b,b"ATS0=1\r"); print("B:",wait(b,b"OK",3))
    a=op(link0); os.write(a,b"ATZ\r"); print("A:",wait(a,b"OK",3)); os.write(a,b"ATDT110\r"); print("A:",wait(a,b"CONNECT",40))
    print("B:",wait(b,b"CONNECT",20))
    os.write(a,b"hello from the host terminal\r\n"); got=wait(b,b"terminal",10); print("B got:",got); ok=b"hello from the host terminal" in got
    os.close(a); print("A terminal closed while online"); nc=wait(b,b"NO CARRIER",15); print("B:",nc); ok=ok and b"NO CARRIER" in nc; sys.exit(0 if ok else 1)
finally:
    for p in procs: p.send_signal(signal.SIGINT)
