"""Windows COM backend tests; fully mocked, no hardware or Windows needed."""
import os
import sys
import time
import unittest
from unittest.mock import patch

from modem_serial_com import (
    BAUD_MAXIMUM,
    BAUD_MINIMUM,
    COM_CTS_BACKLOG_BITS,
    MS_CTS_ON,
    MS_DSR_ON,
    ComSerial,
    check_baud,
    normalize_com_device,
)


class FakeKernel:
    """Records line changes; scripts status, reads and writes."""

    def __init__(self):
        self.status = 0
        self.fail_status = False
        self.read_chunks = []
        self.fail_read = False
        self.written = []
        self.write_partial = None
        self.fail_write = False
        self.dtr_calls = []
        self.rts_calls = []
        self.configured = []
        self.opened = []
        self.closed = []
        self.cancelled = 0
        self._next_handle = 1

    def create_file(self, path):
        self.opened.append(path)
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def close_handle(self, handle):
        self.closed.append(handle)

    def configure(self, handle, baud):
        self.configured.append((handle, baud))

    def set_dtr(self, handle, on):
        self.dtr_calls.append(bool(on))

    def set_rts(self, handle, on):
        self.rts_calls.append(bool(on))

    def modem_status(self, handle):
        if self.fail_status:
            raise OSError("GetCommModemStatus failed")
        return self.status

    def read(self, handle, size, timeout_s):
        if self.fail_read:
            raise OSError("ReadFile failed")
        if self.read_chunks:
            return self.read_chunks.pop(0)
        time.sleep(min(timeout_s, 0.01))
        return b""

    def write(self, handle, data, timeout_s):
        if self.fail_write:
            raise OSError("WriteFile failed")
        if self.write_partial is not None:
            part = self.write_partial
            self.write_partial = None
            self.written.append(data[:part])
            return part
        self.written.append(bytes(data))
        return len(data)

    def cancel(self, handle):
        self.cancelled += 1


def make_ser(kernel=None, **kwargs):
    kernel = kernel or FakeKernel()
    kwargs.setdefault("device", "COM42")
    ser = ComSerial(_kernel=kernel, **kwargs)
    return ser, kernel


class DeviceValidationTests(unittest.TestCase):
    def test_normalizes(self):
        self.assertEqual(normalize_com_device("COM42"), ("COM42", "\\\\.\\COM42"))
        self.assertEqual(normalize_com_device("com7"), ("COM7", "\\\\.\\COM7"))
        self.assertEqual(normalize_com_device("\\\\.\\COM10"), ("COM10", "\\\\.\\COM10"))
        self.assertEqual(normalize_com_device("  COM1  "), ("COM1", "\\\\.\\COM1"))

    def test_rejects(self):
        for bad in ("", "COM", "COM0", "COM257", "COM1000", "SER1", "COM4x", None, 42):
            with self.assertRaises(ValueError, msg=repr(bad)):
                normalize_com_device(bad)

    def test_baud(self):
        self.assertEqual(check_baud(115200), 115200)
        self.assertEqual(check_baud("9600"), 9600)
        for bad in (0, -1, True, False, "fast", "", 299, BAUD_MAXIMUM + 1, 1.5, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                check_baud(bad)

    def test_init_validates_before_open(self):
        kernel = FakeKernel()
        with self.assertRaises(ValueError):
            ComSerial(device="NOPE", _kernel=kernel)
        with self.assertRaises(ValueError):
            ComSerial(device="COM42", baud=7, _kernel=kernel)
        self.assertEqual(kernel.opened, [])

    def test_requires_windows_without_injected_kernel(self):
        with patch.object(os, "name", "posix"):
            with self.assertRaises(ValueError):
                ComSerial(device="COM42")


class SignalMappingTests(unittest.TestCase):
    def test_initial_lines(self):
        ser, kernel = make_ser()
        try:
            # update_lines(False, False, True, False): DTR low (no carrier),
            # RTS high (ready, no hardware flow).
            self.assertEqual(kernel.dtr_calls, [False])
            self.assertEqual(kernel.rts_calls, [True])
            self.assertEqual(ser.baud, 115200)
            self.assertEqual(ser.name, "COM42")
            self.assertTrue(ser.is_com)
        finally:
            ser.close()

    def test_signals_follow_peer(self):
        kernel = FakeKernel()
        kernel.status = MS_DSR_ON | MS_CTS_ON
        ser, kernel = make_ser(kernel)
        try:
            self.assertEqual(ser.signals(), {"TR": True, "RTS": True, "CTS": True})
            kernel.status = 0
            self.assertEqual(ser.signals(), {"TR": False, "RTS": False, "CTS": True})
        finally:
            ser.close()

    def test_update_lines_matrix(self):
        kernel = FakeKernel()
        ser, _ = make_ser(kernel)
        try:
            kernel.dtr_calls.clear()
            kernel.rts_calls.clear()
            # carrier -> DTR; backlog below threshold with hardware -> RTS stays up
            ser.update_lines(True, False, True, True, backlog=0)
            self.assertEqual(kernel.dtr_calls, [True])
            self.assertEqual(kernel.rts_calls, [True])
            # backlog at/above threshold -> RTS drops (peer CTS falls)
            ser.update_lines(True, False, True, True, backlog=COM_CTS_BACKLOG_BITS)
            self.assertEqual(kernel.dtr_calls, [True, True])
            self.assertEqual(kernel.rts_calls, [True, False])
            self.assertFalse(ser.signals()["CTS"])
            # not ready -> RTS drops regardless of backlog
            ser.update_lines(True, False, False, False, backlog=0)
            self.assertEqual(kernel.rts_calls, [True, False, False])
            # ring is recorded but has no pin action on com0com (option B)
            rts_before = list(kernel.rts_calls)
            dtr_before = list(kernel.dtr_calls)
            ser.update_lines(False, True, True, False, backlog=0)
            self.assertTrue(ser._ring)
            self.assertEqual(kernel.rts_calls, rts_before + [True])
            self.assertEqual(kernel.dtr_calls, dtr_before + [False])
            ser.update_lines(False, False, True, False, backlog=0)
            self.assertFalse(ser._ring)
        finally:
            ser.close()

    def test_client_present_is_peer_dtr(self):
        kernel = FakeKernel()
        ser, _ = make_ser(kernel)
        try:
            self.assertFalse(ser.client_present())
            kernel.status = MS_DSR_ON
            self.assertTrue(ser.client_present())
        finally:
            ser.close()


class DataPathTests(unittest.TestCase):
    def test_service_delivers_queued_bytes(self):
        ser, _ = make_ser()
        try:
            ser._rx.put(b"AT\r")
            ser._signal_wakeup()
            data, dropped = ser.service([ser._wake_r], [])
            self.assertEqual(data, b"AT\r")
            self.assertFalse(dropped)
        finally:
            ser.close()

    def test_dtr_drop_reports_once(self):
        kernel = FakeKernel()
        kernel.status = MS_DSR_ON
        ser, _ = make_ser(kernel)
        try:
            _, dropped = ser.service([], [])
            self.assertFalse(dropped)
            kernel.status = 0
            _, dropped = ser.service([], [])
            self.assertTrue(dropped)
            _, dropped = ser.service([], [])
            self.assertFalse(dropped)
        finally:
            ser.close()

    def test_cts_gating_holds_and_releases(self):
        kernel = FakeKernel()
        ser, _ = make_ser(kernel)
        try:
            ser.update_lines(False, False, True, True, backlog=0)
            kernel.status = 0  # peer RTS low: do not send
            ser.write(b"HELLO")
            self.assertEqual(kernel.written, [])
            self.assertEqual(ser.write_fds(), [])
            kernel.status = MS_CTS_ON  # peer RTS high: flush
            self.assertTrue(ser.write_fds())
            _, _ = ser.service([], [ser._wake_r])
            self.assertEqual(b"".join(kernel.written), b"HELLO")
            self.assertEqual(ser.write_fds(), [])
        finally:
            ser.close()

    def test_partial_write_keeps_remainder(self):
        kernel = FakeKernel()
        ser, _ = make_ser(kernel)
        try:
            kernel.write_partial = 2
            ser.write(b"ABCDE")
            self.assertEqual(b"".join(kernel.written), b"AB")
            self.assertEqual(bytes(ser.pending), b"CDE")
            _, _ = ser.service([], [ser._wake_r])
            self.assertEqual(b"".join(kernel.written), b"ABCDE")
        finally:
            ser.close()

    def test_write_overflow(self):
        ser, _ = make_ser()
        try:
            with self.assertRaises(BufferError):
                ser.write(b"x" * 65537)
        finally:
            ser.close()

    def test_dead_port(self):
        kernel = FakeKernel()
        ser, _ = make_ser(kernel)
        try:
            kernel.fail_write = True
            with self.assertRaises(OSError):
                ser.write(b"AT\r")
            with self.assertRaises(OSError):
                ser.write(b"AT\r")
        finally:
            ser.close()

    def test_open_terminal_refuses(self):
        ser, _ = make_ser()
        try:
            with self.assertRaises(ValueError):
                ser.open_terminal()
        finally:
            ser.close()

    def test_close_releases(self):
        kernel = FakeKernel()
        ser, _ = make_ser(kernel)
        handle = ser._handle
        ser.close()
        self.assertIn(handle, kernel.closed)
        self.assertEqual(kernel.dtr_calls[-1], False)
        self.assertEqual(kernel.rts_calls[-1], False)


class SupervisorValidationTests(unittest.TestCase):
    def test_com_spec_validation(self):
        from modem_supervisor import ModemSpec
        ModemSpec("modem-1", "M1", serial="com", serial_device="COM42").validate()
        ModemSpec("modem-1", "M1", serial="com", serial_device="COM42",
                  serial_baud=9600).validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="com", serial_device="").validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="com", serial_device="SER9").validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="com", serial_device="COM42",
                      serial_baud=7).validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="bogus").validate()

    def test_duplicate_devices_rejected(self):
        import tempfile
        from pathlib import Path
        from modem_supervisor import Supervisor
        with tempfile.TemporaryDirectory() as folder:
            supervisor = Supervisor(Path(folder) / "modems.json")
            supervisor.specs.append(
                __import__("modem_supervisor").ModemSpec("modem-1", "M1", serial="com",
                                                         serial_device="COM42").validate())
            supervisor.specs.append(
                __import__("modem_supervisor").ModemSpec("modem-2", "M2", serial="com",
                                                         serial_device="com42").validate())
            with self.assertRaises(ValueError):
                supervisor.validate()

    def test_add_requires_device(self):
        import tempfile
        from pathlib import Path
        from modem_supervisor import Supervisor
        with tempfile.TemporaryDirectory() as folder:
            supervisor = Supervisor(Path(folder) / "modems.json")
            with self.assertRaises(ValueError):
                supervisor.add(serial="com")


if __name__ == "__main__":
    unittest.main()
