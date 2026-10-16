"""Named-pipe serial backend for the bundled Windows virtual-COM driver.

``PipeSerial`` is the counterpart to ``ComSerial`` when El-Modem is reached
through our own UMDF2 virtual-COM driver instead of a physical modem or a
third-party virtual COM pair. Ownership is inverted compared with ``com``:

* With ``--serial-backend com`` El-Modem *owns* one end of a COM pair and the
  terminal opens the peer port.
* With ``--serial-backend pipe`` the driver *is* the COM port. Terminals
  (PuTTY, ZOC, SyncTerm, TTerm, Terminate) open the presented COM port
  directly, and the driver brokers between that port and this pipe. El-Modem
  therefore creates the pipe as the **server** and the driver connects as a
  **client**.

Consequences of that inversion, all deliberate:

* The pipe carries bytes only. No modem-control lines exist on it, so
  ``update_lines`` is a no-op and RTS/CTS report ``None`` rather than a
  fabricated boolean. ``&K1``/``&K3`` therefore have no effect here, exactly
  as on the PTY/TCP adapters, and ``hardware`` stays False so the DCE never
  waits on a CTS gate that cannot exist.
* ``client_present()`` means "the driver has connected", which is the honest
  translation of "a terminal is using the port": with no driver there is no
  COM port at all, so no terminal can be attached.
* The dashboard ASCII terminal cannot attach to a pipe, so ``open_terminal``
  raises. That reflects the real arrangement rather than papering over it: the
  terminal side of this backend is a COM port, reached by a terminal program.
* Output queued while the driver is away is held, not dropped, and flushed on
  reconnect. A modem hanging up mid-response is handled by the DCE's own
  session teardown; discarding bytes here would corrupt a legitimate reply to
  a terminal that reconnects.

Security: the driver runs inside ``WUDFHost.exe`` as SYSTEM in session 0 while
El-Modem runs unprivileged in the interactive session, so the pipe is created
with an explicit DACL granting read/write to Authenticated Users. The default
DACL would normally already cover SYSTEM, but stating it explicitly removes
the dependence on token defaults that differ between an elevated and an
unelevated shell.

Reconnection: the driver retries ``CreateFile`` every 100 ms, so the server
simply recreates its instance each time the client goes away. A fresh instance
per cycle avoids the undefined state that a cancelled overlapped
``ConnectNamedPipe`` leaves behind.

Interface matches ``modem_serial_com.ComSerial``: ``fds()``/``write_fds()``
return the socket-pair wake end (a pipe handle cannot be watched by
``select()``), ``service(readable, writable)`` returns ``(data, dropped)``,
``write(data)`` queues output.
"""
import ctypes
import os
import queue
import re
import socket
import threading
from ctypes import wintypes

# Defined here rather than imported. These were taken from modem_serial_com,
# which is not part of this tree, so the public preview exported a module that
# could not be imported at all. modem_supervisor reuses them from here.
BAUD_MINIMUM = 300
BAUD_MAXIMUM = 921600


def check_baud(baud):
    """Validate a serial baud rate. Accepts ints and digit strings."""
    if isinstance(baud, bool):
        raise ValueError("invalid serial baud (expected %d..%d)" % (BAUD_MINIMUM, BAUD_MAXIMUM))
    if isinstance(baud, str):
        if not baud.strip().isdigit():
            raise ValueError("invalid serial baud (expected %d..%d)" % (BAUD_MINIMUM, BAUD_MAXIMUM))
        baud = int(baud.strip())
    if not isinstance(baud, int) or not BAUD_MINIMUM <= baud <= BAUD_MAXIMUM:
        raise ValueError("invalid serial baud (expected %d..%d)" % (BAUD_MINIMUM, BAUD_MAXIMUM))
    return baud


DEFAULT_PIPE_NAME = "\\\\.\\pipe\\ElModemSerial"
PIPE_NAME_PATTERN = re.compile(r"^\\\\\.\\pipe\\([A-Za-z0-9_.\-]{1,64})$")
# Gating threshold in transmitter backlog bits, same units and value as
# DriverSerial and ComSerial so &K1/3 read alike across backends.
PIPE_CTS_BACKLOG_BITS = 40960
PENDING_LIMIT = 65536
PIPE_BUFFER = 65536

PIPE_ACCESS_DUPLEX = 0x00000003
FILE_FLAG_OVERLAPPED = 0x40000000
PIPE_TYPE_BYTE = 0x00000000
PIPE_READMODE_BYTE = 0x00000000
PIPE_WAIT = 0x00000000

SDDL_REVISION_1 = 1
ERROR_FILE_NOT_FOUND = 2
ERROR_BROKEN_PIPE = 109
ERROR_NO_DATA = 232
ERROR_PIPE_BUSY = 231
ERROR_PIPE_NOT_CONNECTED = 233
ERROR_OPERATION_ABORTED = 995
ERROR_IO_PENDING = 997
ERROR_PIPE_CONNECTED = 535
DISCONNECT_ERRORS = (ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED, ERROR_NO_DATA)
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
CANCEL_GRACE_MS = 2000

ACCEPT_POLL_S = 0.2
READ_POLL_S = 0.15
WRITE_TIMEOUT_S = 0.25

# Protected DACL (no inheritance) granting read/write to authenticated users.
# Create-instance is deliberately absent: the driver only ever opens the pipe
# that already exists.
PIPE_SDDL = "D:P(A;;GRGW;;;AU)"


class PipeDisconnected(OSError):
    """The peer closed the pipe; the server should re-arm a fresh instance."""


def normalize_pipe_name(name):
    """Return a full pipe path. Accepts a bare name (``ElModemSerial``) or
    ``\\\\.\\pipe\\ElModemSerial``. Empty means the driver's default."""
    if name is None:
        return DEFAULT_PIPE_NAME
    if not isinstance(name, str):
        raise ValueError("invalid pipe name (expected \\\\.\\pipe\\ElModemSerial)")
    text = name.strip()
    if not text:
        return DEFAULT_PIPE_NAME
    if "\\" not in text:
        text = "\\\\.\\pipe\\" + text
    if not PIPE_NAME_PATTERN.match(text):
        raise ValueError("invalid pipe name (expected \\\\.\\pipe\\ElModemSerial)")
    return text


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_void_p),
        ("InternalHigh", ctypes.c_void_p),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _PipeKernel:
    """Thin ctypes binding over kernel32/advapi32. Windows only."""

    def __init__(self):
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        a32 = ctypes.WinDLL("advapi32", use_last_error=True)
        k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.DWORD, ctypes.POINTER(_SECURITY_ATTRIBUTES)]
        k32.CreateNamedPipeW.restype = wintypes.HANDLE
        k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED)]
        k32.ConnectNamedPipe.restype = wintypes.BOOL
        k32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
        k32.DisconnectNamedPipe.restype = wintypes.BOOL
        k32.SetNamedPipeHandleState.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                                                ctypes.POINTER(wintypes.DWORD),
                                                ctypes.POINTER(wintypes.DWORD)]
        k32.SetNamedPipeHandleState.restype = wintypes.BOOL
        k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
        k32.ReadFile.restype = wintypes.BOOL
        k32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
        k32.WriteFile.restype = wintypes.BOOL
        k32.GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED),
                                            ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        k32.GetOverlappedResult.restype = wintypes.BOOL
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateEventW.restype = wintypes.HANDLE
        k32.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED)]
        k32.CancelIoEx.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.LocalFree.argtypes = [wintypes.HLOCAL]
        k32.LocalFree.restype = wintypes.HLOCAL
        a32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.ULONG)]
        a32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
        self._k32 = k32
        self._a32 = a32
        self._invalid = wintypes.HANDLE(-1).value
        self._descriptor = None

    def _security_attributes(self):
        if self._descriptor is None:
            descriptor = wintypes.LPVOID()
            if not self._a32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    PIPE_SDDL, SDDL_REVISION_1, ctypes.byref(descriptor), None):
                raise ctypes.WinError(ctypes.get_last_error(),
                                      "ConvertStringSecurityDescriptorToSecurityDescriptor")
            self._descriptor = descriptor
        attributes = _SECURITY_ATTRIBUTES()
        attributes.nLength = ctypes.sizeof(_SECURITY_ATTRIBUTES)
        attributes.lpSecurityDescriptor = self._descriptor
        attributes.bInheritHandle = False
        return attributes

    def _event(self):
        event = self._k32.CreateEventW(None, True, False, None)
        if event == self._invalid or not event:
            raise ctypes.WinError(ctypes.get_last_error(), "CreateEventW")
        return event

    def create_pipe(self, name, size=PIPE_BUFFER):
        attributes = self._security_attributes()
        handle = self._k32.CreateNamedPipeW(
            name,
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            1, size, size, 0, ctypes.byref(attributes))
        if handle == self._invalid or not handle:
            error = ctypes.get_last_error()
            if error == ERROR_PIPE_BUSY:
                raise OSError("pipe %s is already served by another process" % name)
            raise OSError("cannot create pipe %s: %s" % (name, ctypes.WinError(error)))
        # Pin byte-read mode explicitly. A client may switch the handle to
        # message mode, and message framing would defeat the driver's byte
        # pump, so reassert byte mode on every fresh instance.
        mode = wintypes.DWORD(PIPE_READMODE_BYTE)
        if not self._k32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None):
            self._k32.CloseHandle(handle)
            raise OSError("cannot set byte mode on %s: %s"
                          % (name, ctypes.WinError(ctypes.get_last_error())))
        return handle

    def connect(self, handle, timeout_s):
        """Wait up to ``timeout_s`` for a client.

        Returns True when a client attached, False on timeout. On timeout the
        overlapped connect is deliberately left pending; the caller discards
        the instance, which is cleaner than cancelling a connect whose
        post-cancellation state MSDN calls undefined.
        """
        event = self._event()
        overlapped = _OVERLAPPED()
        overlapped.hEvent = event
        try:
            ok = self._k32.ConnectNamedPipe(handle, ctypes.byref(overlapped))
            error = 0 if ok else ctypes.get_last_error()
            if ok or error == ERROR_PIPE_CONNECTED:
                return True
            if error != ERROR_IO_PENDING:
                raise OSError("ConnectNamedPipe failed: %s" % ctypes.WinError(error))
            waited = self._k32.WaitForSingleObject(event, max(0, int(timeout_s * 1000)))
            if waited == WAIT_TIMEOUT:
                return False
            if waited != WAIT_OBJECT_0:
                raise OSError("ConnectNamedPipe wait failed: %s"
                              % ctypes.WinError(ctypes.get_last_error()))
            return True
        finally:
            self._k32.CloseHandle(event)

    def read(self, handle, size, timeout_s):
        buf = ctypes.create_string_buffer(size)
        overlapped = _OVERLAPPED()
        overlapped.hEvent = self._event()
        try:
            done = wintypes.DWORD()
            if not self._k32.ReadFile(handle, buf, size, None, ctypes.byref(overlapped)):
                error = ctypes.get_last_error()
                if error in DISCONNECT_ERRORS:
                    raise PipeDisconnected("pipe peer closed the connection")
                if error != ERROR_IO_PENDING:
                    raise OSError("ReadFile failed: %s" % ctypes.WinError(error))
                waited = self._k32.WaitForSingleObject(overlapped.hEvent,
                                                       max(0, int(timeout_s * 1000)))
                if waited == WAIT_TIMEOUT:
                    self._k32.CancelIoEx(handle, ctypes.byref(overlapped))
                    self._k32.WaitForSingleObject(overlapped.hEvent, CANCEL_GRACE_MS)
                    return b""
                if waited != WAIT_OBJECT_0:
                    raise OSError("ReadFile wait failed: %s"
                                  % ctypes.WinError(ctypes.get_last_error()))
            if not self._k32.GetOverlappedResult(handle, ctypes.byref(overlapped),
                                                  ctypes.byref(done), False):
                error = ctypes.get_last_error()
                if error in DISCONNECT_ERRORS:
                    raise PipeDisconnected("pipe peer closed the connection")
                raise OSError("GetOverlappedResult failed: %s" % ctypes.WinError(error))
            # A byte-stream pipe completing with zero bytes means the writer
            # side went away.
            if done.value == 0:
                raise PipeDisconnected("pipe peer closed the connection")
            return bytes(buf.raw[:done.value])
        finally:
            self._k32.CloseHandle(overlapped.hEvent)

    def write(self, handle, data, timeout_s):
        overlapped = _OVERLAPPED()
        overlapped.hEvent = self._event()
        try:
            done = wintypes.DWORD()
            if not self._k32.WriteFile(handle, data, len(data), None, ctypes.byref(overlapped)):
                error = ctypes.get_last_error()
                if error in DISCONNECT_ERRORS:
                    raise PipeDisconnected("pipe peer closed the connection")
                if error != ERROR_IO_PENDING:
                    raise OSError("WriteFile failed: %s" % ctypes.WinError(error))
                waited = self._k32.WaitForSingleObject(overlapped.hEvent,
                                                       max(0, int(timeout_s * 1000)))
                if waited == WAIT_TIMEOUT:
                    self._k32.CancelIoEx(handle, ctypes.byref(overlapped))
                    self._k32.WaitForSingleObject(overlapped.hEvent, CANCEL_GRACE_MS)
                    self._k32.GetOverlappedResult(handle, ctypes.byref(overlapped),
                                                  ctypes.byref(done), False)
                    return done.value
                if waited != WAIT_OBJECT_0:
                    raise OSError("WriteFile wait failed: %s"
                                  % ctypes.WinError(ctypes.get_last_error()))
            if not self._k32.GetOverlappedResult(handle, ctypes.byref(overlapped),
                                                  ctypes.byref(done), False):
                error = ctypes.get_last_error()
                if error in DISCONNECT_ERRORS:
                    raise PipeDisconnected("pipe peer closed the connection")
                raise OSError("GetOverlappedResult failed: %s" % ctypes.WinError(error))
            return done.value
        finally:
            self._k32.CloseHandle(overlapped.hEvent)

    def disconnect(self, handle):
        if handle is None:
            return
        # Best effort: a never-connected instance reports ERROR_PIPE_NOT_CONNECTED.
        self._k32.DisconnectNamedPipe(handle)

    def cancel(self, handle):
        if handle is None:
            return
        try:
            self._k32.CancelIoEx(handle, None)
        except OSError:
            pass

    def close(self, handle):
        if handle is None:
            return
        # Cancel first: closing a handle with I/O still outstanding risks the
        # handle number being recycled under the pending operation.
        self.cancel(handle)
        if not self._k32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error(), "CloseHandle")


def probe_device(device):
    """Validate a pipe name for the supervisor. Deliberately does not create
    the pipe: the driver may attach seconds later, so refusing to start would
    be wrong. Returns the normalized name."""
    return normalize_pipe_name(device)


class PipeSerial:
    """Byte-pipe serial endpoint for the virtual-COM driver. See module docstring."""

    # Terminals genuinely attach over a COM port, so ATI4 should report COM.
    is_com = True

    def __init__(self, device=DEFAULT_PIPE_NAME, baud=115200, _kernel=None):
        name = normalize_pipe_name(device)
        baud = check_baud(baud)
        if _kernel is None:
            if os.name != "nt":
                raise ValueError("pipe serial backend requires Windows; choose PTY/TCP")
            kernel = _PipeKernel()
        else:
            kernel = _kernel
        self._kernel = kernel
        self._handle = None
        self._present = False
        self._dead = False
        self._ring = False
        # Guards _handle: the reader thread publishes and withdraws it while the
        # DCE's main loop writes through it.
        self._lock = threading.Lock()
        # No CTS/RTS exists on a byte pipe, so the DCE must never be told that
        # hardware flow control is in force.
        self.hardware = False
        self.name = name
        self.device = name
        self.baud = baud
        self.pending = bytearray()
        self._rx = queue.Queue()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._pump, name="pipeserial", daemon=True)
        self._reader.start()

    # ----- presence and modem-control lines -----

    def client_present(self):
        """The driver has connected, i.e. a terminal can be using the port."""
        return bool(self._present)

    def signals(self):
        """TR follows driver presence. RTS/CTS are genuinely unknown, so they
        report None rather than a fabricated boolean."""
        return dict(TR=self.client_present(), RTS=None, CTS=None)

    def update_lines(self, carrier, ring, ready, hardware, backlog=0):
        """No modem-control lines exist on a byte pipe.

        The ring state is recorded for the dashboard, and ``hardware`` is
        ignored rather than obeyed: gating on a CTS the peer cannot drive would
        stall the transmitter forever.
        """
        self._ring = bool(ring)

    # ----- byte pump -----

    def _signal_wakeup(self):
        try:
            self._wake_w.send(b"\0")
        except (BlockingIOError, OSError):
            pass

    def _drain_wakeup(self):
        try:
            while self._wake_r.recv(65536):
                pass
        except (BlockingIOError, OSError):
            pass

    def _pump(self):
        handle = None
        try:
            while not self._stop.is_set():
                # A fresh instance per connection cycle: the driver's retry
                # loop finds it within ~100 ms of us going away.
                handle = self._kernel.create_pipe(self.name, PIPE_BUFFER)
                try:
                    if self._kernel.connect(handle, ACCEPT_POLL_S):
                        # Publish the handle only once a client is actually
                        # attached. Publishing it earlier would let a concurrent
                        # write target an instance with no peer, where those
                        # bytes would be silently dropped.
                        with self._lock:
                            self._handle = handle
                            self._present = True
                        self._signal_wakeup()
                        self._serve(handle)
                finally:
                    # Withdraw the handle before tearing the instance down, so
                    # _flush can never write into a handle being closed.
                    with self._lock:
                        self._handle = None
                        self._present = False
                    self._kernel.disconnect(handle)
                    self._kernel.close(handle)
                    handle = None
                    self._signal_wakeup()
        except OSError:
            self._dead = True
            self._signal_wakeup()
        finally:
            with self._lock:
                self._present = False
                self._handle = None
            if handle is not None:
                try:
                    self._kernel.disconnect(handle)
                    self._kernel.close(handle)
                except OSError:
                    pass
            self._signal_wakeup()

    def _serve(self, handle):
        while not self._stop.is_set():
            try:
                data = self._kernel.read(handle, 4096, READ_POLL_S)
            except PipeDisconnected:
                return
            except OSError:
                self._dead = True
                self._signal_wakeup()
                return
            if data:
                self._rx.put(data)
                self._signal_wakeup()

    def _flush(self):
        if not self.pending or self._dead:
            return
        with self._lock:
            handle = self._handle
        if handle is None:
            # Driver is away. Hold the backlog (bounded by PENDING_LIMIT) so a
            # reply in flight is not truncated by a brief reconnect.
            return
        try:
            sent = self._kernel.write(handle, bytes(self.pending), WRITE_TIMEOUT_S)
        except PipeDisconnected:
            return
        except OSError:
            self._dead = True
            raise
        del self.pending[:sent]

    # ----- DriverSerial-compatible interface -----

    def fds(self):
        return [self._wake_r]

    def write_fds(self):
        return [self._wake_r] if self.pending else []

    def service(self, readable, writable=()):
        if self._wake_r in writable:
            try:
                self._flush()
            except OSError:
                pass
        data = b""
        if self._wake_r in readable:
            self._drain_wakeup()
            while True:
                try:
                    data += self._rx.get_nowait()
                except queue.Empty:
                    break
        return data, False

    def write(self, data):
        if self._dead:
            raise OSError("pipe is no longer usable: %s" % self.name)
        if len(self.pending) + len(data) > PENDING_LIMIT:
            raise BufferError("serial receive queue exhausted")
        self.pending.extend(data)
        self._flush()

    def open_terminal(self):
        raise ValueError("pipe backend cannot host the dashboard terminal: attach a terminal"
                         " program to the COM port presented by the virtual-COM driver")

    def close(self):
        self._stop.set()
        self._signal_wakeup()
        # Deliberately no CancelIoEx here. Every pump wait is already bounded
        # (READ_POLL_S, ACCEPT_POLL_S), so the thread notices the stop flag
        # within ~200ms and closes the instance it created itself. Cancelling
        # from this thread would race a handle the pump may have already closed,
        # and a stale CancelIoEx can target an unrelated, reused handle.
        self._reader.join(timeout=3)
        for sock in (self._wake_r, self._wake_w):
            try:
                sock.close()
            except OSError:
                pass