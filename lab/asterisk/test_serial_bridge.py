#!/usr/bin/env python3
"""serial_bridge: data both ways through TCP, and closing the terminal reaches the modem as a port close."""
import errno
import os
import pty
import select
import socket
import sys
import tempfile
import threading
import time
import unittest

import serial_bridge
from serial_port import (PtySerial, get_tty_baud, set_tty_baud,
                         get_tty_character_format, set_tty_character_format)
from serial_port import TcpSerial


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
    @unittest.skipUnless(os.name == "posix", "POSIX termios only")
    def test_mirrors_character_format_in_both_directions(self):
        modem = PtySerial()
        port = free_port()
        stop = threading.Event()
        server = serial_bridge.serve_pty_over_tcp(
            modem.name, "127.0.0.1", port, stop, baud_control_port=port + 1)
        link = os.path.join(tempfile.mkdtemp(), "tty")
        client = threading.Thread(target=serial_bridge.run_client,
                                  args=(link, "127.0.0.1", port, stop),
                                  kwargs={"baud_control_port": port + 1}, daemon=True)
        client.start()
        term = None
        try:
            deadline = time.monotonic() + 3
            while not os.path.islink(link) and time.monotonic() < deadline:
                time.sleep(0.02)
            term = os.open(link, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            # Linux PTYs commonly coerce CS7 to CS8; 8N2 is accepted here.
            self.assertTrue(set_tty_character_format(term, 1, 1))  # 8N2
            deadline = time.monotonic() + 3
            while modem.get_dte_character_format() != (1, 1) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(modem.get_dte_character_format(), (1, 1))

            self.assertTrue(modem.set_dte_character_format(3, 1))  # 8N1
            deadline = time.monotonic() + 3
            while get_tty_character_format(term) != (3, 1) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(get_tty_character_format(term), (3, 1))
        finally:
            if term is not None:
                os.close(term)
            stop.set()
            client.join(2)
            server.join(2)
            modem.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux termios2 custom rates only")
    def test_direct_tcp_serial_mirrors_custom_baud_in_both_directions(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        while port >= 65535:
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()
        server = TcpSerial("127.0.0.1", port, baud_control_port=port + 1)
        stop = threading.Event()
        link = os.path.join(tempfile.mkdtemp(), "tty")

        def service_tcp_serial():
            while not stop.is_set():
                readable, writable, _ = select.select(server.fds(), server.write_fds(), [], 0.05)
                server.service(readable, writable)

        modem_thread = threading.Thread(target=service_tcp_serial, daemon=True)
        client = threading.Thread(target=serial_bridge.run_client,
                                  args=(link, "127.0.0.1", port, stop),
                                  kwargs={"baud_control_port": port + 1}, daemon=True)
        modem_thread.start()
        client.start()
        term = None
        try:
            deadline = time.monotonic() + 3
            while not os.path.islink(link) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(os.path.islink(link))
            term = os.open(link, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            self.assertTrue(set_tty_baud(term, 1212))
            deadline = time.monotonic() + 3
            while server.get_dte_baud() != 1212 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(server.get_dte_baud(), 1212)

            self.assertTrue(server.set_dte_baud(1170))
            deadline = time.monotonic() + 3
            while get_tty_baud(term) != 1170 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(get_tty_baud(term), 1170)

            self.assertTrue(set_tty_character_format(term, 1, 1))  # 8N2
            deadline = time.monotonic() + 3
            while server.get_dte_character_format() != (1, 1) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(server.get_dte_character_format(), (1, 1))
            self.assertTrue(server.set_dte_character_format(3, 1))  # 8N1
            deadline = time.monotonic() + 3
            while get_tty_character_format(term) != (3, 1) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(get_tty_character_format(term), (3, 1))
        finally:
            if term is not None:
                os.close(term)
            stop.set()
            client.join(2)
            modem_thread.join(2)
            server.close()
            self.assertFalse(client.is_alive())
            self.assertFalse(modem_thread.is_alive())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux termios2 custom rates only")
    def test_mirrors_custom_bell_baud_in_both_directions(self):
        modem = PtySerial()
        port = free_port()
        while port == 65535:
            port = free_port()
        stop = threading.Event()
        server = serial_bridge.serve_pty_over_tcp(
            modem.name, "127.0.0.1", port, stop, baud_control_port=port + 1)
        link = os.path.join(tempfile.mkdtemp(), "tty")
        client = threading.Thread(target=serial_bridge.run_client,
                                  args=(link, "127.0.0.1", port, stop),
                                  kwargs={"baud_control_port": port + 1}, daemon=True)
        client.start()
        term = None
        try:
            deadline = time.monotonic() + 3
            while not os.path.islink(link) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(os.path.islink(link))
            term = os.open(link, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            self.assertTrue(set_tty_baud(term, 1212))
            deadline = time.monotonic() + 3
            while modem.get_dte_baud() != 1212 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(modem.get_dte_baud(), 1212)

            self.assertTrue(modem.set_dte_baud(1170))
            deadline = time.monotonic() + 3
            while get_tty_baud(term) != 1170 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(get_tty_baud(term), 1170)
        finally:
            if term is not None:
                os.close(term)
            stop.set()
            client.join(2)
            server.join(2)
            self.assertFalse(client.is_alive(), "serial bridge client did not stop")
            self.assertFalse(server.is_alive(), "serial bridge server did not observe terminal close")
            modem.close()

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
        conn = None
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
            if conn is not None:
                conn.close()
            srv.close()


if __name__ == "__main__":
    unittest.main()
