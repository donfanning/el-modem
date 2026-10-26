"""Pipe backend tests.

The mocked cases run everywhere and cover the state machine, buffering and
supervisor validation. The Win32 cases are the ones that actually matter for
this backend: they prove a real client can connect to the pipe El-Modem
creates, that bytes cross in both directions, and that the server re-arms
after the client goes away.
"""
import os
import socket
import sys
import time
import unittest
from unittest.mock import patch

import ctypes
from ctypes import wintypes

from modem_serial_pipe import (
    DEFAULT_PIPE_NAME,
    PENDING_LIMIT,
    PIPE_SDDL,
    PipeDisconnected,
    PipeSerial,
    normalize_pipe_name,
    probe_device,
)

POLL_S = 0.01
POLL_TIMEOUT_S = 3.0

_name_counter = [0]


def _next_name():
    _name_counter[0] += 1
    return _name_counter[0]


class _OVERLAPPED(ctypes.Structure):
    """Local copy so the client side does not depend on the module's internals."""
    _fields_ = [
        ("Internal", ctypes.c_void_p),
        ("InternalHigh", ctypes.c_void_p),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


def wait_for(predicate, timeout=POLL_TIMEOUT_S):
    """Poll until ``predicate()`` is true. Returns whether it became true.

    The backend moves state between threads, so tests assert on convergence
    rather than on an exact interleaving.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_S)
    return bool(predicate())


class FakePipeKernel:
    """Scripts connection, reads and writes without touching Windows."""

    def __init__(self):
        self.created = []
        self.closed = []
        self.cancelled = []
        self.disconnected = []
        self.written = []
        self.connect_calls = 0
        self.auto_connect = False
        self.read_chunks = []
        self.write_partial = None
        self.fail_write = False
        self.fail_create = False
        self._next_handle = 1

    def create_pipe(self, name, size=65536):
        if self.fail_create:
            raise OSError("cannot create pipe")
        self.created.append(name)
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def connect(self, handle, timeout_s):
        self.connect_calls += 1
        if self.auto_connect:
            return True
        time.sleep(min(timeout_s, 0.01))
        return False

    def read(self, handle, size, timeout_s):
        if self.read_chunks:
            item = self.read_chunks.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        time.sleep(min(timeout_s, 0.01))
        return b""

    def write(self, handle, data, timeout_s):
        if self.fail_write:
            raise OSError("WriteFile failed")
        if self.write_partial is not None:
            part = self.write_partial
            self.write_partial = None
            self.written.append(bytes(data[:part]))
            return part
        self.written.append(bytes(data))
        return len(data)

    def disconnect(self, handle):
        self.disconnected.append(handle)

    def cancel(self, handle):
        self.cancelled.append(handle)

    def close(self, handle):
        self.closed.append(handle)


def make_ser(kernel=None, **kwargs):
    kernel = kernel or FakePipeKernel()
    ser = PipeSerial(_kernel=kernel, **kwargs)
    return ser, kernel


class NameValidationTests(unittest.TestCase):
    def test_defaults_to_the_driver_name(self):
        self.assertEqual(normalize_pipe_name(""), DEFAULT_PIPE_NAME)
        self.assertEqual(normalize_pipe_name(None), DEFAULT_PIPE_NAME)
        self.assertEqual(normalize_pipe_name("   "), DEFAULT_PIPE_NAME)
        self.assertEqual(probe_device(""), DEFAULT_PIPE_NAME)

    def test_accepts_bare_and_full(self):
        self.assertEqual(normalize_pipe_name("ElModemSerial"), DEFAULT_PIPE_NAME)
        self.assertEqual(normalize_pipe_name("  Other  "), "\\\\.\\pipe\\Other")
        self.assertEqual(normalize_pipe_name("\\\\.\\pipe\\Abc-1_2.3"), "\\\\.\\pipe\\Abc-1_2.3")

    def test_rejects(self):
        for bad in ("\\\\.\\pipe\\", "\\\\.\\mailslot\\x", "bad name", "\\\\.\\pipe\\a\\b",
                    "\\\\.\\pipe\\" + "x" * 65, 42, b"ElModemSerial"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                normalize_pipe_name(bad)


class ConstructionTests(unittest.TestCase):
    def test_validates_before_creating_anything(self):
        kernel = FakePipeKernel()
        with self.assertRaises(ValueError):
            PipeSerial(device="bad name", _kernel=kernel)
        with self.assertRaises(ValueError):
            PipeSerial(baud=7, _kernel=kernel)
        self.assertEqual(kernel.created, [])

    def test_requires_windows_without_injected_kernel(self):
        with patch.object(os, "name", "posix"):
            with self.assertRaises(ValueError):
                PipeSerial()

    def test_sddl_is_protected_and_authenticated_users_only(self):
        self.assertTrue(PIPE_SDDL.startswith("D:P"), PIPE_SDDL)
        self.assertIn("(A;;GRGW;;;AU)", PIPE_SDDL)

    def test_reports_com_to_the_dce(self):
        ser, _ = make_ser()
        try:
            self.assertTrue(ser.is_com)
            self.assertEqual(ser.name, DEFAULT_PIPE_NAME)
            self.assertEqual(ser.baud, 115200)
        finally:
            ser.close()


class PresenceTests(unittest.TestCase):
    def test_absent_until_the_driver_connects(self):
        ser, kernel = make_ser()
        try:
            self.assertFalse(ser.client_present())
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present), "driver never registered as present")
        finally:
            ser.close()

    def test_signals_report_unknown_flow_lines(self):
        ser, kernel = make_ser()
        try:
            # No modem-control line exists on a byte pipe, so none of them may
            # report a boolean. TR included: it used to follow driver presence,
            # on the reasoning that the driver connecting means a terminal is
            # using the port. It lit before ZOC was ever started. "Can use" is
            # not "is using" -- the driver attaches as soon as the modem serves
            # the pipe, and opening COM3 puts nothing on the wire, so a
            # terminal attaching is not observable from this side at all.
            self.assertEqual(ser.signals(), dict(TR=None, RTS=None, CTS=None))
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
            # Presence is still tracked, for the dashboard's "who owns the
            # port", but it is not a signal line and must not light one.
            self.assertEqual(ser.signals(), dict(TR=None, RTS=None, CTS=None))
        finally:
            ser.close()

    def test_update_lines_never_enables_hardware_flow(self):
        ser, _ = make_ser()
        try:
            ser.update_lines(True, True, True, True, backlog=0)
            self.assertTrue(ser._ring)
            # Gating on a CTS the peer cannot drive would stall the transmitter.
            self.assertFalse(ser.hardware)
            ser.update_lines(False, False, True, False, backlog=0)
            self.assertFalse(ser._ring)
        finally:
            ser.close()

    def test_rearms_after_the_driver_disconnects(self):
        ser, kernel = make_ser()
        try:
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
            # Stop auto-connecting before dropping the peer, otherwise the
            # server re-arms and reconnects faster than the poll can observe
            # the absent phase.
            kernel.auto_connect = False
            kernel.read_chunks.append(PipeDisconnected("peer went away"))
            self.assertTrue(wait_for(lambda: not ser.client_present()), "presence never cleared")
            self.assertTrue(wait_for(lambda: len(kernel.created) > 1), "server never re-armed")
        finally:
            ser.close()


class DataPathTests(unittest.TestCase):
    def test_service_delivers_bytes_from_the_driver(self):
        ser, kernel = make_ser()
        try:
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
            kernel.read_chunks.extend([b"AT", b"\r\n"])
            self.assertTrue(wait_for(lambda: ser._rx.qsize() == 2))
            data, dropped = ser.service([ser._wake_r], [])
            self.assertEqual(data, b"AT\r\n")
            self.assertFalse(dropped)
        finally:
            ser.close()

    def test_write_reaches_the_pipe(self):
        ser, kernel = make_ser()
        try:
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
            ser.write(b"CONNECT 9600\r\n")
            self.assertEqual(b"".join(kernel.written), b"CONNECT 9600\r\n")
            self.assertEqual(ser.write_fds(), [])
        finally:
            ser.close()

    def test_partial_write_keeps_remainder(self):
        ser, kernel = make_ser()
        try:
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
            kernel.write_partial = 2
            ser.write(b"ABCDE")
            self.assertEqual(bytes(ser.pending), b"CDE")
            _, _ = ser.service([], [ser._wake_r])
            self.assertEqual(b"".join(kernel.written), b"ABCDE")
        finally:
            ser.close()

    def test_backlog_is_held_while_the_driver_is_away(self):
        ser, kernel = make_ser()
        try:
            ser.write(b"HELD")
            self.assertEqual(bytes(ser.pending), b"HELD")
            self.assertEqual(kernel.written, [])
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
            ser.write(b"")  # nudge; pending is already non-empty
            self.assertEqual(b"".join(kernel.written), b"HELD")
        finally:
            ser.close()

    def test_write_overflow(self):
        ser, _ = make_ser()
        try:
            with self.assertRaises(BufferError):
                ser.write(b"x" * (PENDING_LIMIT + 1))
        finally:
            ser.close()

    def test_dead_pipe(self):
        ser, kernel = make_ser()
        try:
            kernel.fail_write = True
            kernel.auto_connect = True
            self.assertTrue(wait_for(ser.client_present))
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

    def test_close_stops_the_reader_and_closes_the_wake_sockets(self):
        kernel = FakePipeKernel()
        kernel.auto_connect = True
        ser, _ = make_ser(kernel)
        self.assertTrue(wait_for(ser.client_present))
        wake_r, wake_w = ser._wake_r, ser._wake_w
        ser.close()
        self.assertTrue(kernel.closed, "no pipe instance was closed")
        # Both wake sockets are closed and the reader thread has exited, so the
        # worker can be torn down without a lingering thread.
        self.assertEqual(wake_r.fileno(), -1)
        self.assertEqual(wake_w.fileno(), -1)
        self.assertFalse(ser._reader.is_alive())


@unittest.skipUnless(sys.platform == "win32", "requires Windows named pipes")
class Win32IntegrationTests(unittest.TestCase):
    """Exercises the real kernel32 path: a genuine client connects to the
    pipe El-Modem creates, bytes cross both ways, and the server re-arms."""

    # A byte-stream pipe blocks indefinitely on a synchronous read, so the
    # client is opened overlapped and every transfer is bounded. A test that
    # hangs is far less useful than one that fails.
    GENERIC_READ_WRITE = 0xC0000000
    OPEN_EXISTING = 3
    FILE_FLAG_OVERLAPPED = 0x40000000
    ERROR_IO_PENDING = 997
    WAIT_OBJECT_0 = 0

    def setUp(self):
        # A unique name per test: the server is a single-instance pipe, so a
        # leftover instance from a previous test would make CreateNamedPipe
        # fail with ERROR_PIPE_BUSY.
        self.NAME = "\\\\.\\pipe\\ElModemTest%d_%d" % (os.getpid(), _next_name())
        self.ser = PipeSerial(device=self.NAME)
        self.addCleanup(self.ser.close)
        self.ctypes, self.wintypes, self.k32 = self._bind()

    @staticmethod
    def _bind():
        """Fully typed kernel32 handles.

        Without argtypes ctypes marshals a Python str as char*, which a W
        function misreads; every name below would come out as garbage.
        """
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
        k32.WaitNamedPipeW.restype = wintypes.BOOL
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.HANDLE]
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateEventW.restype = wintypes.HANDLE
        k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_char_p, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
        k32.ReadFile.restype = wintypes.BOOL
        k32.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_char_p, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
        k32.WriteFile.restype = wintypes.BOOL
        k32.GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED),
                                            ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        k32.GetOverlappedResult.restype = wintypes.BOOL
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED)]
        k32.CancelIoEx.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        return ctypes, wintypes, k32

    def _overlapped(self):
        event = self.k32.CreateEventW(None, True, False, None)
        overlapped = _OVERLAPPED()
        overlapped.hEvent = event
        return overlapped, event

    def _await(self, overlapped, event, done, handle, what, timeout_s=10.0):
        ctypes, wintypes, k32 = self.ctypes, self.wintypes, self.k32
        waited = k32.WaitForSingleObject(event, int(timeout_s * 1000))
        if waited != self.WAIT_OBJECT_0:
            k32.CancelIoEx(handle, ctypes.byref(overlapped))
            self.fail("%s timed out after %ss" % (what, timeout_s))
        if not k32.GetOverlappedResult(handle, ctypes.byref(overlapped),
                                       ctypes.byref(done), False):
            self.fail("%s failed: %s" % (what, ctypes.WinError(ctypes.get_last_error())))

    def _client(self, timeout_s=10.0):
        ctypes, wintypes, k32 = self.ctypes, self.wintypes, self.k32
        # WaitNamedPipeW returns immediately with FILE_NOT_FOUND when the name
        # does not exist yet, so its timeout argument cannot cover the server's
        # startup. Poll instead, the same way the driver's own broker retries.
        deadline = time.monotonic() + timeout_s
        while True:
            if k32.WaitNamedPipeW(self.NAME, 200):
                handle = k32.CreateFileW(self.NAME, self.GENERIC_READ_WRITE, 0, None,
                                         self.OPEN_EXISTING, self.FILE_FLAG_OVERLAPPED, None)
                if handle != wintypes.HANDLE(-1).value and handle:
                    return handle
            if time.monotonic() >= deadline:
                self.fail("server never published %s (dead=%s): %s"
                          % (self.NAME, self.ser._dead, ctypes.WinError(ctypes.get_last_error())))
            time.sleep(POLL_S)

    def _client_read(self, handle, size=256, timeout_s=10.0):
        ctypes, wintypes, k32 = self.ctypes, self.wintypes, self.k32
        buffer = ctypes.create_string_buffer(size)
        overlapped, event = self._overlapped()
        done = wintypes.DWORD()
        try:
            if not k32.ReadFile(handle, buffer, size, ctypes.byref(done), ctypes.byref(overlapped)):
                if ctypes.get_last_error() != self.ERROR_IO_PENDING:
                    self.fail("client ReadFile failed: %s" % ctypes.WinError(ctypes.get_last_error()))
                self._await(overlapped, event, done, handle, "client read", timeout_s)
            return buffer.raw[:done.value]
        finally:
            k32.CloseHandle(event)

    def _client_write(self, handle, data, timeout_s=10.0):
        ctypes, wintypes, k32 = self.ctypes, self.wintypes, self.k32
        overlapped, event = self._overlapped()
        done = wintypes.DWORD()
        try:
            if not k32.WriteFile(handle, data, len(data), ctypes.byref(done), ctypes.byref(overlapped)):
                if ctypes.get_last_error() != self.ERROR_IO_PENDING:
                    self.fail("client WriteFile failed: %s" % ctypes.WinError(ctypes.get_last_error()))
                self._await(overlapped, event, done, handle, "client write", timeout_s)
            return done.value
        finally:
            k32.CloseHandle(event)

    def test_client_connects_and_data_flows_both_ways(self):
        handle = self._client()
        try:
            self.assertTrue(wait_for(self.ser.client_present), "driver never seen as present")
            # Driver -> terminal
            self.ser.write(b"\r\nCONNECT 9600\r\n")
            self.assertEqual(self._client_read(handle), b"\r\nCONNECT 9600\r\n")
            # Terminal -> driver
            self.assertEqual(self._client_write(handle, b"ATD5551212\r\n"), 12)
            self.assertTrue(wait_for(lambda: not self.ser._rx.empty()),
                            "server never received the client's bytes")
            data, dropped = self.ser.service([self.ser._wake_r], [])
            self.assertEqual(data, b"ATD5551212\r\n")
            self.assertFalse(dropped)
        finally:
            self.k32.CloseHandle(handle)

    def test_server_rearms_after_the_client_leaves(self):
        handle = self._client()
        self.assertTrue(wait_for(self.ser.client_present))
        self.k32.CloseHandle(handle)
        self.assertTrue(wait_for(lambda: not self.ser.client_present()),
                        "presence never cleared after client exit")
        # A second client must be able to attach to the fresh instance.
        second = self._client()
        try:
            self.assertTrue(wait_for(self.ser.client_present), "server did not re-arm")
            self.ser.write(b"OK\r\n")
            self.assertEqual(self._client_read(second), b"OK\r\n")
        finally:
            self.k32.CloseHandle(second)


class SupervisorValidationTests(unittest.TestCase):
    def test_pipe_spec_validation(self):
        from modem_supervisor import ModemSpec
        ModemSpec("modem-1", "M1", serial="pipe").validate()
        ModemSpec("modem-1", "M1", serial="pipe", serial_device="Other").validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="pipe", serial_device="bad name").validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="pipe", serial_baud=7).validate()
        with self.assertRaises(ValueError):
            ModemSpec("modem-1", "M1", serial="bogus").validate()

    def test_add_defaults_to_the_driver_pipe_name(self):
        import tempfile
        from pathlib import Path
        from modem_supervisor import Supervisor
        with tempfile.TemporaryDirectory() as folder:
            supervisor = Supervisor(Path(folder) / "modems.json")
            spec = supervisor.add(serial="pipe")
            self.assertEqual(spec.serial_device, "")
            with self.assertRaises(ValueError):
                supervisor.add(serial="pipe", device="bad name")


if __name__ == "__main__":
    unittest.main()