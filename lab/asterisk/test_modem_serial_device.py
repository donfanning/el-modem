"""Native POSIX tty BREAK marker decoding; hardware is not required."""
import unittest
import os
import pty
import sys
import termios
from unittest import mock

import modem_serial_device
from modem_dce import AtDce
from modem_serial_device import DeviceSerial, decode_parmrk


class ParityMarkTests(unittest.TestCase):
    def test_break_marker_is_removed_and_reported(self):
        data, breaks, state = decode_parmrk(b"before\xff\x00\x00after")
        self.assertEqual((data, breaks, state), (b"beforeafter", 1, 0))

    def test_literal_ff_and_error_markers_remain_byte_data(self):
        data, breaks, state = decode_parmrk(b"a\xff\xffb\xff\x00\x7fc")
        self.assertEqual((data, breaks, state), (b"a\xffb\x7fc", 0, 0))

    def test_marker_can_be_split_across_reads(self):
        first = decode_parmrk(b"x\xff\x00")
        second = decode_parmrk(b"\x00y", first[2])
        self.assertEqual(first, (b"x", 0, 2))
        self.assertEqual(second, (b"y", 1, 0))


class NativeDtrTransitionTests(unittest.TestCase):
    def test_unknown_initial_dtr_never_invents_a_drop(self):
        serial = object.__new__(DeviceSerial)
        serial.fd = 91
        serial._terminal_active = False
        serial._previous_dtr = None
        observed = iter((None, True, True, False, False, None, True, False))
        with mock.patch.object(serial, "_get_line", side_effect=lambda _line: next(observed)):
            drops = [serial.service([], [])[1] for _ in range(8)]
        self.assertEqual(drops, [False, False, False, True, False, False, False, True])


class NativeDteSettingsTests(unittest.TestCase):
    def test_linux_native_serial_applies_custom_bell_dte_rates(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("custom BOTHER rates are Linux termios2 only")
        master, slave = pty.openpty()
        serial = None
        try:
            serial = DeviceSerial(os.ttyname(slave), baud=1212)
            self.assertEqual(serial.get_dte_baud(), 1212)
            self.assertIn(1170, serial.supported_dte_baud_rates())
            self.assertIn(1212, serial.supported_dte_baud_rates())
            self.assertTrue(serial.set_dte_baud(1170))
            self.assertEqual(serial.get_dte_baud(), 1170)
            self.assertTrue(serial.set_dte_baud(1200))
            self.assertEqual(serial.get_dte_baud(), 1200)
        finally:
            if serial is not None:
                serial.close()
            os.close(master)
            os.close(slave)

    def test_native_serial_applies_bell212_online_framing(self):
        master, slave = pty.openpty()
        serial = None
        try:
            serial = DeviceSerial(os.ttyname(slave), baud=1200)
            self.assertTrue(serial.set_bell212_dte_format(10, "E"))  # 8N1
            cflag = termios.tcgetattr(serial.fd)[2]
            self.assertEqual(cflag & termios.CSIZE, termios.CS8)
            self.assertFalse(cflag & termios.PARENB)
            self.assertFalse(cflag & termios.CSTOPB)
            # A PTY may not implement parity or seven-bit UART framing. When
            # it accepts one, verify the exact physical-format bits it reports.
            if serial.set_bell212_dte_format(11, "O"):
                cflag = termios.tcgetattr(serial.fd)[2]
                self.assertTrue(cflag & termios.PARENB)
                self.assertTrue(cflag & termios.PARODD)
            self.assertTrue(serial.set_dte_character_format(3, 3))  # restore 8N1
        finally:
            if serial is not None:
                serial.close()
            os.close(master)
            os.close(slave)

    def test_native_serial_exposes_and_applies_ipr_and_icf_settings(self):
        master, slave = pty.openpty()
        serial = None
        try:
            serial = DeviceSerial(os.ttyname(slave), baud=1200)
            formats, parities = serial.supported_dte_character_formats()
            dce = AtDce(
                set_dte_baud=serial.set_dte_baud,
                get_dte_baud=serial.get_dte_baud,
                dte_baud_rates=serial.supported_dte_baud_rates(),
                set_dte_character_format=serial.set_dte_character_format,
                dte_character_formats=formats,
                dte_parity_values=parities,
            )
            self.assertEqual(serial.get_dte_baud(), 1200)
            self.assertIn(9600, serial.supported_dte_baud_rates())
            self.assertTrue(dce.feed(b"AT+IPR=9600\r").endswith(b"OK\r\n"))
            self.assertTrue(serial.set_dte_baud(dce.take_pending_dte_baud()))
            self.assertEqual(serial.get_dte_baud(), 9600)

            self.assertIn(3, formats)
            self.assertTrue(dce.feed(b"AT+ICF=1,3\r").endswith(b"OK\r\n"))  # 8N2
            self.assertTrue(serial.set_dte_character_format(*dce.take_pending_dte_character_format()))
            if 1 in formats:
                self.assertTrue(dce.feed(b"AT+ICF=3,3\r").endswith(b"OK\r\n"))
                self.assertTrue(serial.set_dte_character_format(*dce.take_pending_dte_character_format()))
            parity_formats = [code for code in (2, 5) if code in formats]
            if parity_formats:
                self.assertTrue(serial.set_dte_character_format(parity_formats[0], parities[0]))
        finally:
            if serial is not None:
                serial.close()
            os.close(master)
            os.close(slave)


class PosixBreakQueueTests(unittest.TestCase):
    def test_output_pending_includes_application_and_tty_queues(self):
        ser = object.__new__(DeviceSerial)
        ser.fd = 42
        ser._pending = bytearray(b"abc")
        ser._output_shadow = bytearray(b"abcdefghij")

        def ioctl(_fd, _request, queued, _mutate):
            queued[0] = 7

        with mock.patch.object(modem_serial_device.termios, "TIOCOUTQ", 0x5411,
                               create=True), \
             mock.patch.object(modem_serial_device.fcntl, "ioctl", side_effect=ioctl):
            self.assertEqual(ser.output_pending(), 10)

    def test_destructive_break_clears_application_and_tty_queues(self):
        ser = object.__new__(DeviceSerial)
        ser.fd = 42
        ser._pending = bytearray(b"queued")
        ser._output_shadow = bytearray(b"queued")
        with mock.patch.object(modem_serial_device.termios, "TCOFLUSH", 1, create=True), \
             mock.patch.object(modem_serial_device.termios, "tcflush") as flush:
            self.assertTrue(ser.discard_output())
        self.assertEqual(ser._pending, bytearray())
        flush.assert_called_once_with(42, 1)

    def test_break_reordering_preserves_pending_bytes_for_post_break_send(self):
        ser = object.__new__(DeviceSerial)
        ser.fd = 42
        ser._pending = bytearray(b"ef")
        ser._output_shadow = bytearray(b"abcdef")
        ser._can_reorder_break = True

        def ioctl(_fd, _request, queued, _mutate):
            queued[0] = 4

        with mock.patch.object(modem_serial_device.termios, "TIOCOUTQ", 0x5411,
                               create=True), \
             mock.patch.object(modem_serial_device.termios, "TCOFLUSH", 1, create=True), \
             mock.patch.object(modem_serial_device.fcntl, "ioctl", side_effect=ioctl), \
             mock.patch.object(modem_serial_device.termios, "tcflush") as flush:
            self.assertTrue(ser.place_break_before_pending_output())
        self.assertEqual(ser._pending, bytearray(b"abcdef"))
        self.assertEqual(ser._output_shadow, bytearray(b"abcdef"))
        flush.assert_called_once_with(42, 1)

    def test_queued_output_is_not_flushed_during_line_break(self):
        ser = object.__new__(DeviceSerial)
        ser.fd = 42
        ser._pending = bytearray(b"after-break")
        ser._break_active = True
        ser._hardware_flow = False
        written = []
        with mock.patch.object(modem_serial_device.os, "write",
                               side_effect=lambda fd, data: written.append((fd, bytes(data))) or len(data)) as write:
            ser._flush()
            write.assert_not_called()
            ser._break_active = False
            ser._flush()
        self.assertEqual(written, [(42, b"after-break")])


if __name__ == "__main__":
    unittest.main()
