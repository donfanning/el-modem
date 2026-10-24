#!/usr/bin/env python3
"""Carry a modem serial port over TCP so a terminal program on the host (minicom, picocom, screen ...) can use the
softmodem that lives in a container, with hangup-on-close preserved end to end.

  container:  modem_dce.py --link /tmp/modem0 --tcp-listen 0.0.0.0:9100 --tcp-baud-control
  host:       serial_bridge.py client --connect 127.0.0.1:9100 --baud-control-port 9110 --link ~/.softmodem0
              minicom -D ~/.softmodem0

A client connecting opens the modem's PTY, a client disconnecting closes it (which the DCE sees as a dropped DTR and
answers per &D). On the host the bridge creates its own PTY and connects to the modem only while a terminal program
has that PTY open, so quitting minicom closes the TCP connection and the call is hung up. Raw bytes, no telnet.
For baud synchronization, both sides opt into a versioned control socket on an explicitly exposed port; older
raw-only clients remain supported. The direct ``--serial-tcp`` backend is a separate byte-only interface.
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

from serial_port import (get_tty_baud, set_tty_baud, get_tty_character_format,
                         set_tty_character_format)
from serial_metadata import encode_baud, encode_character_format, take_metadata_frames


def _pump(tty_fd: int, sock: socket.socket, tty_closed, *, baud_sock=None,
          baud_listener=None, send_initial_baud=False) -> None:
    """Copy bytes and optionally mirror tty baud over a separate control socket."""
    last_baud = None
    last_format = None
    baud_input = bytearray()
    baud_output = bytearray()
    next_baud_poll = 0.0

    def queue_baud(rate):
        if rate and 0 < rate <= 4_000_000:
            baud_output.extend(encode_baud(rate))

    def queue_format(value):
        if value is not None:
            baud_output.extend(encode_character_format(*value))

    def close_baud_socket():
        nonlocal baud_sock
        if baud_sock is not None:
            try:
                baud_sock.close()
            except OSError:
                pass
            baud_sock = None

    if send_initial_baud and baud_sock is not None:
        last_baud = get_tty_baud(tty_fd)
        queue_baud(last_baud)

    while True:
        now = time.monotonic()
        if baud_sock is not None and now >= next_baud_poll:
            next_baud_poll = now + 0.05
            local_baud = get_tty_baud(tty_fd)
            if local_baud and local_baud != last_baud:
                queue_baud(local_baud)
                last_baud = local_baud
            local_format = get_tty_character_format(tty_fd)
            if local_format and local_format != last_format:
                queue_format(local_format)
                last_format = local_format
        readers = [tty_fd, sock]
        if baud_sock is not None:
            readers.append(baud_sock)
        if baud_listener is not None and baud_sock is None:
            readers.append(baud_listener)
        writers = [baud_sock] if baud_sock is not None and baud_output else []
        readable, writable, _ = select.select(readers, writers, [], 0.05)
        if baud_listener is not None and baud_listener in readable:
            try:
                baud_sock, _ = baud_listener.accept()
                baud_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                baud_sock.setblocking(False)
                baud_input.clear()
                baud_output.clear()
                # The client side sends the initial DTE setting. The server
                # waits for it, then reports later changes made by the DCE.
                last_baud = get_tty_baud(tty_fd)
                last_format = get_tty_character_format(tty_fd)
            except OSError:
                pass
        if baud_sock is not None and baud_sock in writable:
            try:
                sent = baud_sock.send(baud_output)
                del baud_output[:sent]
            except (BlockingIOError, OSError):
                pass
        if baud_sock is not None and baud_sock in readable:
            try:
                chunk = baud_sock.recv(128)
                if not chunk:
                    close_baud_socket()
                    baud_input.clear()
                    baud_output.clear()
                    last_baud = get_tty_baud(tty_fd)
                else:
                    baud_input.extend(chunk)
                    for kind, value, aux in take_metadata_frames(baud_input):
                        if kind == "baud" and value and set_tty_baud(tty_fd, value):
                            # A received setting is already synchronized;
                            # don't echo it back as a local change.
                            last_baud = value
                        elif kind == "format" and set_tty_character_format(tty_fd, value, aux):
                            last_format = (value, aux)
            except BlockingIOError:
                pass
            except OSError:
                close_baud_socket()
                baud_input.clear()
                baud_output.clear()
                last_baud = get_tty_baud(tty_fd)
                last_format = get_tty_character_format(tty_fd)
        if tty_fd in readable:
            try:
                data = os.read(tty_fd, 4096)
            except BlockingIOError:
                data = b""
            except OSError:
                close_baud_socket()
                return
            if data:
                try:
                    sock.sendall(data)
                except OSError:
                    close_baud_socket()
                    return
        elif tty_closed():
            close_baud_socket()
            return
        if sock in readable:
            try:
                data = sock.recv(4096)
            except OSError:
                close_baud_socket()
                return
            if not data:
                close_baud_socket()
                return
            while data:
                try:
                    n = os.write(tty_fd, data)
                    data = data[n:]
                except BlockingIOError:
                    select.select([], [tty_fd], [], 0.1)
                except OSError:
                    close_baud_socket()
                    return


def serve_pty_over_tcp(path: str, host: str, port: int, stop: threading.Event | None = None,
                       *, baud_control_port: int | None = None) -> threading.Thread:
    """Listen for raw serial; optionally mirror tty baud over a separate control port."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(0.5)
    baud_srv = None
    if baud_control_port is not None:
        baud_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        baud_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            baud_srv.bind((host, baud_control_port))
            baud_srv.listen(1)
            baud_srv.setblocking(False)
        except BaseException:
            baud_srv.close()
            srv.close()
            raise

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
                _pump(fd, conn, lambda: False, baud_listener=baud_srv)
            finally:
                os.close(fd)                    # the modem sees the port close: a dropped DTR
                conn.close()
        if baud_srv is not None:
            baud_srv.close()
        srv.close()

    t = threading.Thread(target=loop, name="serial-tcp", daemon=True)
    t.start()
    return t


def run_client(link: str, host: str, port: int, stop: threading.Event | None = None,
               *, baud_control_port: int | None = None) -> None:
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
            baud_sock = None
            if baud_control_port is not None:
                try:
                    baud_sock = socket.create_connection((host, baud_control_port), timeout=0.5)
                    baud_sock.setblocking(False)
                except OSError:
                    baud_sock = None          # raw serial remains available without the side channel
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

            _pump(master, sock, gone, baud_sock=baud_sock, send_initial_baud=True)
            if baud_sock is not None:
                baud_sock.close()
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
    c.add_argument("--baud-control-port", type=int,
                   help="optional separate TCP port for versioned DTE baud synchronization")
    args = ap.parse_args()
    host, port = args.connect.rsplit(":", 1)
    try:
        run_client(args.link, host, int(port), baud_control_port=args.baud_control_port)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
