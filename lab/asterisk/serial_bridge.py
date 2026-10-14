#!/usr/bin/env python3
"""Carry a modem serial port over TCP so a terminal program on the host (minicom, picocom, screen ...) can use the
softmodem that lives in a container, with hangup-on-close preserved end to end.

  container:  modem_dce.py --link /tmp/modem0 --tcp-listen 0.0.0.0:9100        (uses serve_pty_over_tcp)
  host:       serial_bridge.py client --connect 127.0.0.1:9100 --link ~/.softmodem0
              minicom -D ~/.softmodem0

A client connecting opens the modem's PTY, a client disconnecting closes it (which the DCE sees as a dropped DTR and
answers per &D). On the host the bridge creates its own PTY and connects to the modem only while a terminal program
has that PTY open, so quitting minicom closes the TCP connection and the call is hung up. Raw bytes, no telnet.
"""

from __future__ import annotations

import argparse
import errno
import os
import pty
import select
import socket
import sys
import threading
import time
import tty


def _pump(tty_fd: int, sock: socket.socket, tty_closed) -> None:
    """Copy both ways until the socket closes or ``tty_closed()`` says the terminal went away."""
    while True:
        readable, _, _ = select.select([tty_fd, sock], [], [], 0.1)
        if tty_fd in readable:
            try:
                data = os.read(tty_fd, 4096)
            except BlockingIOError:
                data = b""
            except OSError:
                return
            if data:
                try:
                    sock.sendall(data)
                except OSError:
                    return
        elif tty_closed():
            return
        if sock in readable:
            try:
                data = sock.recv(4096)
            except OSError:
                return
            if not data:
                return
            while data:
                try:
                    n = os.write(tty_fd, data)
                    data = data[n:]
                except BlockingIOError:
                    select.select([], [tty_fd], [], 0.1)
                except OSError:
                    return


def serve_pty_over_tcp(path: str, host: str, port: int, stop: threading.Event | None = None) -> threading.Thread:
    """Listen on host:port; while a client is connected, hold ``path`` (the modem's PTY slave) open for it."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(0.5)

    def loop() -> None:
        while stop is None or not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            except OSError:
                conn.close()
                continue
            try:
                tty.setraw(fd)
                _pump(fd, conn, lambda: False)
            finally:
                os.close(fd)                    # the modem sees the port close: a dropped DTR
                conn.close()
        srv.close()

    t = threading.Thread(target=loop, name="serial-tcp", daemon=True)
    t.start()
    return t


def run_client(link: str, host: str, port: int, stop: threading.Event | None = None) -> None:
    """Host side: own PTY at ``link``; connect to the modem only while a terminal program has the PTY open."""
    master, slave = pty.openpty()
    name = os.ttyname(slave)
    tty.setraw(slave)
    os.close(slave)                              # with no terminal attached the master reads EIO
    os.set_blocking(master, False)
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(name, link)
    print(f"serial bridge: {link} -> {host}:{port}", flush=True)
    try:
        while stop is None or not stop.is_set():
            r, _, _ = select.select([master], [], [], 0)
            attached = True
            if r:
                try:
                    first = os.read(master, 4096)
                except BlockingIOError:
                    first = b""
                except OSError as exc:
                    attached = exc.errno != errno.EIO
                    first = b""
            else:
                first = b""
            if not attached:
                time.sleep(0.1)
                continue
            try:
                sock = socket.create_connection((host, port), timeout=3)
            except OSError as exc:
                print(f"serial bridge: modem not reachable ({exc}); retrying", file=sys.stderr, flush=True)
                time.sleep(1.0)
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if first:
                sock.sendall(first)

            def gone() -> bool:
                rr, _, _ = select.select([master], [], [], 0)
                if not rr:
                    return False
                try:
                    os.read(master, 0)
                    return False
                except OSError as e:
                    return e.errno == errno.EIO

            _pump(master, sock, gone)
            sock.close()                         # terminal closed (or modem gone): hang the call up
    finally:
        os.close(master)
        if os.path.islink(link):
            os.unlink(link)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("client", help="host side: create a local PTY wired to the modem's TCP port")
    c.add_argument("--connect", default="127.0.0.1:9100", metavar="HOST:PORT")
    c.add_argument("--link", default=os.path.expanduser("~/.softmodem0"))
    args = ap.parse_args()
    host, port = args.connect.rsplit(":", 1)
    try:
        run_client(args.link, host, int(port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
