import os
import select
import socket
import sys
import time
import unittest

from serial_port import PtySerial, TcpSerial


def pump(ser, t=0.3):
    out, dropped = b"", False
    end = time.monotonic() + t
    while time.monotonic() < end:
        r, w, _ = select.select(ser.fds(), ser.write_fds(), [], 0.05)
        d, dr = ser.service(r, w)
        out += d
        dropped = dropped or dr
    return out, dropped


class TcpSerialTests(unittest.TestCase):
    def setUp(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.ser = TcpSerial("127.0.0.1", self.port)

    def tearDown(self):
        self.ser.close()

    def test_data_both_ways_and_disconnect_is_a_drop(self):
        c = socket.create_connection(("127.0.0.1", self.port))
        c.sendall(b"ATZ\r")
        self.assertEqual(pump(self.ser), (b"ATZ\r", False))
        self.ser.write(b"OK\r\n")
        c.settimeout(1)
        self.assertEqual(c.recv(10), b"OK\r\n")
        c.close()
        self.assertEqual(pump(self.ser), (b"", True))
        self.ser.write(b"ignored")                       # no client: must not raise

    def test_new_client_replaces_old(self):
        a = socket.create_connection(("127.0.0.1", self.port))
        pump(self.ser, 0.1)
        b = socket.create_connection(("127.0.0.1", self.port))
        _, dropped = pump(self.ser, 0.2)
        self.assertTrue(dropped)
        b.sendall(b"AT\r")
        self.assertEqual(pump(self.ser)[0], b"AT\r")
        a.close()
        b.close()

    def test_backpressure_preserves_binary_output(self):
        c = socket.socket()
        c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        c.connect(("127.0.0.1", self.port))
        try:
            pump(self.ser, 0.05)
            self.ser._conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            # Fill the nonblocking TCP send queue before asking the modem
            # backend to write. This reliably creates pending output across
            # OSes with different loopback autotuning and buffer minima.
            filler = b'F' * 65536
            filler_bytes = 0
            blocked = False
            for _ in range(1024):
                try:
                    sent = self.ser._conn.send(filler)
                except BlockingIOError:
                    blocked = True
                    break
                filler_bytes += sent
                if sent == 0:
                    break
            self.assertGreater(filler_bytes, 0, 'test must fill the socket send queue')
            self.assertTrue(blocked, 'socket send queue did not reach backpressure')
            payload = bytes(range(256)) * 4096
            self.ser.write(payload)
            self.assertTrue(self.ser.write_fds(), 'test must exercise queued output')
            c.setblocking(False)
            received = bytearray()
            filler_remaining = filler_bytes
            end = time.monotonic() + 15
            while len(received) < len(payload) and time.monotonic() < end:
                r, w, _ = select.select([c], self.ser.write_fds(), [], .01)
                self.ser.service([], w)
                if r:
                    data = c.recv(65536)
                    if filler_remaining:
                        skipped = min(filler_remaining, len(data))
                        filler_remaining -= skipped
                        data = data[skipped:]
                    received.extend(data)
            self.assertEqual(filler_remaining, 0)
            self.assertEqual(received, payload)
            self.assertEqual(self.ser.write_fds(), [])
        finally:
            c.close()


@unittest.skipIf(os.name == "nt", "PTY is POSIX only")
class PtySerialTests(unittest.TestCase):
    def test_sets_v250_character_format_in_termios(self):
        import termios

        ser = PtySerial()
        try:
            formats, _parities = ser.supported_dte_character_formats()
            self.assertIn(1, formats)
            self.assertTrue(ser.set_dte_character_format(1, 1))  # 8N2
            attrs = termios.tcgetattr(ser.master)
            self.assertEqual(attrs[2] & termios.CSIZE, termios.CS8)
            self.assertFalse(attrs[2] & termios.PARENB)
            self.assertTrue(attrs[2] & termios.CSTOPB)
            self.assertTrue(ser.set_dte_character_format(3, 3))  # 8N1
            attrs = termios.tcgetattr(ser.master)
            self.assertEqual(attrs[2] & termios.CSIZE, termios.CS8)
            self.assertFalse(attrs[2] & termios.PARENB)
            self.assertFalse(attrs[2] & termios.CSTOPB)
            if 4 in formats:
                self.assertTrue(ser.set_dte_character_format(4, 1))  # 7N2
                self.assertEqual(termios.tcgetattr(ser.master)[2] & termios.CSIZE, termios.CS7)
            if 5 not in formats:
                self.assertFalse(ser.set_dte_character_format(5, 1))
        finally:
            ser.close()

    def test_sets_standard_dte_baud(self):
        ser = PtySerial()
        fd = os.open(ser.name, os.O_RDWR | os.O_NOCTTY)
        try:
            self.assertIn(9600, ser.supported_dte_baud_rates())
            self.assertTrue(ser.set_dte_baud(9600))
            self.assertEqual(ser.get_dte_baud(), 9600)
            self.assertFalse(ser.set_dte_baud(12345))
        finally:
            os.close(fd)
            ser.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux termios2 only")
    def test_sets_custom_bell_dte_baud(self):
        ser = PtySerial()
        fd = os.open(ser.name, os.O_RDWR | os.O_NOCTTY)
        try:
            supported = ser.supported_dte_baud_rates()
            self.assertIn(1170, supported)
            self.assertIn(1212, supported)
            self.assertTrue(ser.set_dte_baud(1170))
            self.assertEqual(ser.get_dte_baud(), 1170)
            self.assertTrue(ser.set_dte_baud(1212))
            self.assertEqual(ser.get_dte_baud(), 1212)
            self.assertFalse(ser.set_dte_baud(1169))
            self.assertFalse(ser.set_dte_baud(1213))
        finally:
            os.close(fd)
            ser.close()

    def test_reports_configured_dte_baud(self):
        import termios

        ser = PtySerial()
        fd = os.open(ser.name, os.O_RDWR | os.O_NOCTTY)
        try:
            attrs = termios.tcgetattr(fd)
            attrs[4] = termios.B1200
            attrs[5] = termios.B1200
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            self.assertEqual(ser.get_dte_baud(), 1200)
        finally:
            os.close(fd)
            ser.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux termios2 only")
    def test_reports_custom_termios2_baud(self):
        import fcntl
        import struct

        ser = PtySerial()
        fd = os.open(ser.name, os.O_RDWR | os.O_NOCTTY)
        try:
            attrs = bytearray(44)  # Linux struct termios2, NCCS=19
            fcntl.ioctl(fd, 0x802C542A, attrs, True)  # TCGETS2
            cflag = struct.unpack_from("=I", attrs, 8)[0]
            struct.pack_into("=I", attrs, 8, (cflag & ~0x100F) | 0x1000)  # CBAUD/BOTHER
            struct.pack_into("=II", attrs, 36, 1212, 1212)
            fcntl.ioctl(fd, 0x402C542B, attrs, True)  # TCSETS2
            self.assertEqual(ser.get_dte_baud(), 1212)
        finally:
            os.close(fd)
            ser.close()

    def test_open_data_close(self):
        ser = PtySerial()
        try:
            fd = os.open(ser.name, os.O_RDWR | os.O_NOCTTY)
            os.write(fd, b"AT\r")
            self.assertEqual(pump(ser)[0], b"AT\r")
            os.close(fd)
            self.assertTrue(pump(ser, 0.5)[1])
        finally:
            ser.close()


if __name__ == "__main__":
    unittest.main()
