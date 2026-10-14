#!/usr/bin/env python3
"""serial_bridge: data both ways through TCP, and closing the terminal reaches the modem as a port close."""
import errno
import os
import pty
import select
import socket
import tempfile
import threading
import time
import unittest

import serial_bridge


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def read_some(fd, t=2.0, want_eio=False):
    """Bytes read from ``fd``; EIO (no peer has the port open) is waited out unless ``want_eio`` (then it is returned)."""
    end, out = time.time() + t, b""
    while time.time() < end and not out:
        if select.select([fd], [], [], 0.1)[0]:
            try:
                out += os.read(fd, 100)
            except OSError as e:
                if want_eio:
                    return e.errno
                time.sleep(0.02)
    return out


class BridgeTests(unittest.TestCase):
    def test_server_side_carries_data_and_close_is_seen_by_the_modem(self):
        master, slave = pty.openpty()
        path = os.ttyname(slave)
        os.close(slave)
        port = free_port()
        stop = threading.Event()
        serial_bridge.serve_pty_over_tcp(path, "127.0.0.1", port, stop)
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=3)
            c.sendall(b"ATZ\r")
            self.assertEqual(read_some(master), b"ATZ\r")
            os.write(master, b"OK\r\n")
            c.settimeout(2)
            self.assertEqual(c.recv(100), b"OK\r\n")
            c.close()
            self.assertEqual(read_some(master, want_eio=True), errno.EIO)          # the modem sees the port close
        finally:
            stop.set()

    def test_host_side_connects_only_while_a_terminal_has_the_port_open(self):
        port = free_port()
        accepted = []
        srv = socket.socket()
        srv.bind(("127.0.0.1", port))
        srv.listen(2)
        srv.settimeout(0.3)
        stop = threading.Event()
        link = os.path.join(tempfile.mkdtemp(), "tty")
        t = threading.Thread(target=serial_bridge.run_client, args=(link, "127.0.0.1", port, stop), daemon=True)
        t.start()
        try:
            for _ in range(50):
                if os.path.islink(link):
                    break
                time.sleep(0.05)
            time.sleep(0.5)
            with self.assertRaises(socket.timeout):                 # nobody has the port open: no connection
                srv.accept()
            term = os.open(link, os.O_RDWR | os.O_NOCTTY)
            conn, _ = srv.accept()
            os.write(term, b"AT\r")
            conn.settimeout(2)
            self.assertEqual(conn.recv(100), b"AT\r")
            conn.sendall(b"OK\r\n")
            self.assertEqual(read_some(term), b"OK\r\n")
            os.close(term)                                          # minicom quits
            conn.settimeout(3)
            self.assertEqual(conn.recv(100), b"")                   # the bridge dropped the connection
        finally:
            stop.set()
            t.join(2)
            srv.close()


if __name__ == "__main__":
    unittest.main()
