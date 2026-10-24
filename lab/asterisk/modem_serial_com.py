"""Windows native-COM serial backend (our own code, standard library only).

``ComSerial`` owns one end of an already-installed virtual COM pair
(com0com) and presents Hayes-compatible modem-control lines to the DCE:
real CTS/RTS hardware flow control, DTR presence/drop, and DCD carrier.
It is the Windows counterpart to ``DriverSerial`` in modem_serial.py,
which drives the separately installed serialsim prerequisite on Linux.

Required pair wiring (set once, as administrator, when the pair is
created; the modem itself runs unprivileged and never touches setup):

    setupc.exe install PortName=COM41,dsr=on PortName=COM42

i.e. the *terminal* side reports DSR always-on (the modem is always
present) while its DCD follows our DTR (carrier) and its CTS follows
our RTS (flow). Our side keeps com0com defaults (our CTS follows the
peer's RTS, our DSR/DCD follow the peer's DTR) so peer presence and
RTS gating read correctly. See packaging/com0com/README.md.

Deliberate limits (option B):

* RI has no pin action on com0com: two DCE outputs (DTR, RTS) cannot
  drive three independent peer inputs (DCD, CTS, RI) with static
  wiring, and the idle state rules out every multiplexing scheme.
  Incoming-ring detection still reaches terminals as the Hayes ``RING``
  result code. ``update_lines`` records the ring state so a future
  DCE-capable presenting driver lights the pin with no DCE changes.
* The dashboard ASCII terminal cannot take over a COM port whose peer
  is unknown; ``open_terminal`` raises instead of guessing.
* Baud is nominal on a virtual port (bytes move at memory speed, not
  at the configured rate); the setting controls the DCB and reporting.
* COM BREAK uses CE_BREAK polling and SetCommBreak/ClearCommBreak. TCP, pipe,
  and PTY byte transports have no equivalent signal and must not encode BREAK
  as a special data byte.

Interface matches serial_port.py plus the DriverSerial signal methods:
``fds()``/``write_fds()`` return selectable objects (a socket-pair wake
end fed by the overlapped-reader thread, since select() cannot watch a
COM handle); ``service(readable, writable)`` returns ``(data, dropped)``;
``write(data)`` queues output; ``signals()``/``update_lines()`` carry
the modem-control lines; ``client_present()`` reports peer DTR.
"""
import ctypes
import os
import queue
import re
import socket
import threading
import time
from ctypes import wintypes

COM_DEVICE_PATTERN = r"COM([1-9][0-9]{0,2})"
BAUD_MINIMUM = 300
BAUD_MAXIMUM = 921600
# Gating threshold in transmitter backlog bits; same units and value as
# DriverSerial (modem_serial.py), so &K1/3 behave the same on both.
COM_CTS_BACKLOG_BITS = 40960
PENDING_LIMIT = 65536
CE_BREAK = 0x0010

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_OVERLAPPED = 0x40000000
PURGE_TXCLEAR = 0x4
PURGE_RXCLEAR = 0x8
PURGE_TXABORT = 0x1
SET_RTS = 3
CLR_RTS = 4
SET_DTR = 5
CLR_DTR = 6
MS_CTS_ON = 0x10
MS_DSR_ON = 0x20
MS_RING_ON = 0x40
MS_RLSD_ON = 0x80
DTR_CONTROL_ENABLE = 1
RTS_CONTROL_ENABLE = 1
ERROR_IO_PENDING = 997
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
INFINITE = 0xFFFFFFFF


def normalize_com_device(device):
    """Return ``(short, path)`` for a COM device, e.g. ``COM42`` ->
    ``("COM42", "\\\\.\\COM42")``. Raises ValueError otherwise."""
    text = str(device or "").strip().upper()
    for prefix in ("\\\\.\\", "\\\\?\\"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    match = re.fullmatch(COM_DEVICE_PATTERN, text)
    if not match:
        raise ValueError("invalid COM device (expected COMn, e.g. COM42)")
    number = int(match.group(1))
    if number > 256:
        raise ValueError("invalid COM device (expected COM1..COM256)")
    short = "COM%d" % number
    return short, "\\\\.\\" + short


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


class _DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("fBinary", wintypes.DWORD, 1),
        ("fParity", wintypes.DWORD, 1),
        ("fOutxCtsFlow", wintypes.DWORD, 1),
        ("fOutxDsrFlow", wintypes.DWORD, 1),
        ("fDtrControl", wintypes.DWORD, 2),
        ("fDsrSensitivity", wintypes.DWORD, 1),
        ("fTXContinueOnXoff", wintypes.DWORD, 1),
        ("fOutX", wintypes.DWORD, 1),
        ("fInX", wintypes.DWORD, 1),
        ("fErrorChar", wintypes.DWORD, 1),
        ("fNull", wintypes.DWORD, 1),
        ("fRtsControl", wintypes.DWORD, 2),
        ("fAbortOnError", wintypes.DWORD, 1),
        ("fDummy2", wintypes.DWORD, 17),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", wintypes.BYTE),
        ("Parity", wintypes.BYTE),
        ("StopBits", wintypes.BYTE),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class _COMSTAT(ctypes.Structure):
    _fields_ = [("flags", wintypes.DWORD), ("cbInQue", wintypes.DWORD),
                ("cbOutQue", wintypes.DWORD)]


class _COMMTIMEOUTS(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_void_p),
        ("InternalHigh", ctypes.c_void_p),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _RealKernel:
    """Thin ctypes binding over kernel32. Only instantiated on Windows."""

    def __init__(self):
        self._comm_error_lock = threading.Lock()
        self._pending_breaks = {}
        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.HANDLE]
        dll.CreateFileW.restype = wintypes.HANDLE
        dll.CloseHandle.argtypes = [wintypes.HANDLE]
        dll.CloseHandle.restype = wintypes.BOOL
        dll.GetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(_DCB)]
        dll.GetCommState.restype = wintypes.BOOL
        dll.SetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(_DCB)]
        dll.SetCommState.restype = wintypes.BOOL
        dll.SetCommTimeouts.argtypes = [wintypes.HANDLE, ctypes.POINTER(_COMMTIMEOUTS)]
        dll.SetCommTimeouts.restype = wintypes.BOOL
        dll.SetupComm.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        dll.SetupComm.restype = wintypes.BOOL
        dll.PurgeComm.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        dll.PurgeComm.restype = wintypes.BOOL
        dll.SetCommMask.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        dll.SetCommMask.restype = wintypes.BOOL
        dll.EscapeCommFunction.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        dll.EscapeCommFunction.restype = wintypes.BOOL
        dll.GetCommModemStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        dll.GetCommModemStatus.restype = wintypes.BOOL
        dll.SetCommBreak.argtypes = [wintypes.HANDLE]
        dll.SetCommBreak.restype = wintypes.BOOL
        dll.ClearCommBreak.argtypes = [wintypes.HANDLE]
        dll.ClearCommBreak.restype = wintypes.BOOL
        dll.ClearCommError.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        dll.ClearCommError.restype = wintypes.BOOL
        dll.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
        dll.ReadFile.restype = wintypes.BOOL
        dll.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)]
        dll.WriteFile.restype = wintypes.BOOL
        dll.GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED),
                                            ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        dll.GetOverlappedResult.restype = wintypes.BOOL
        dll.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        dll.WaitForSingleObject.restype = wintypes.DWORD
        dll.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        dll.CreateEventW.restype = wintypes.HANDLE
        dll.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(_OVERLAPPED)]
        dll.CancelIoEx.restype = wintypes.BOOL
        self._dll = dll
        self._invalid = wintypes.HANDLE(-1).value

    def _check(self, ok, what):
        if ok:
            return
        raise ctypes.WinError(ctypes.get_last_error(), what)

    def create_file(self, path):
        handle = self._dll.CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, None,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                                       None)
        if handle == self._invalid or not handle:
            err = ctypes.get_last_error()
            if err == 2:
                raise FileNotFoundError("no such COM port: %s" % path)
            if err == 5:
                raise OSError("cannot open %s: access denied (port owned by another program,"
                              " or com0com requires elevation)" % path)
            raise OSError("cannot open %s: WinError %d" % (path, err))
        return handle

    def close_handle(self, handle):
        self._check(self._dll.CloseHandle(handle), "CloseHandle")

    def configure(self, handle, baud):
        self._check(self._dll.SetupComm(handle, 65536, 65536), "SetupComm")
        dcb = _DCB()
        dcb.DCBlength = ctypes.sizeof(_DCB)
        self._check(self._dll.GetCommState(handle, ctypes.byref(dcb)), "GetCommState")
        dcb.BaudRate = baud
        dcb.fBinary = 1
        dcb.fParity = 0
        dcb.fOutxCtsFlow = 0  # DCE enforces flow itself; never let the host swallow XON/XOFF
        dcb.fOutxDsrFlow = 0
        dcb.fDtrControl = DTR_CONTROL_ENABLE
        dcb.fDsrSensitivity = 0
        dcb.fTXContinueOnXoff = 0
        dcb.fOutX = 0
        dcb.fInX = 0
        dcb.fErrorChar = 0
        dcb.fNull = 0
        dcb.fRtsControl = RTS_CONTROL_ENABLE
        dcb.fAbortOnError = 0
        dcb.ByteSize = 8
        dcb.Parity = 0
        dcb.StopBits = 0
        self._check(self._dll.SetCommState(handle, ctypes.byref(dcb)), "SetCommState")
        timeouts = _COMMTIMEOUTS(0, 0, 100, 0, 2000)
        self._check(self._dll.SetCommTimeouts(handle, ctypes.byref(timeouts)), "SetCommTimeouts")
        self._check(self._dll.PurgeComm(handle, PURGE_TXCLEAR | PURGE_RXCLEAR), "PurgeComm")
        self._check(self._dll.SetCommMask(handle, 0), "SetCommMask")

    def set_dtr(self, handle, on):
        self._check(self._dll.EscapeCommFunction(handle, SET_DTR if on else CLR_DTR),
                    "EscapeCommFunction(DTR)")

    def set_rts(self, handle, on):
        self._check(self._dll.EscapeCommFunction(handle, SET_RTS if on else CLR_RTS),
                    "EscapeCommFunction(RTS)")

    def modem_status(self, handle):
        status = wintypes.DWORD()
        self._check(self._dll.GetCommModemStatus(handle, ctypes.byref(status)), "GetCommModemStatus")
        return status.value

    def take_break(self, handle):
        with self._comm_error_lock:
            queued = self._pending_breaks.get(handle, 0)
            if queued:
                self._pending_breaks[handle] = queued - 1
                return True
            errors = wintypes.DWORD()
            self._check(self._dll.ClearCommError(handle, ctypes.byref(errors), None), "ClearCommError")
            return bool(errors.value & CE_BREAK)

    def output_pending(self, handle):
        with self._comm_error_lock:
            errors = wintypes.DWORD()
            status = _COMSTAT()
            self._check(self._dll.ClearCommError(handle, ctypes.byref(errors), ctypes.byref(status)),
                        "ClearCommError")
            if errors.value & CE_BREAK:
                self._pending_breaks[handle] = self._pending_breaks.get(handle, 0) + 1
            return int(status.cbOutQue)

    def discard_output(self, handle):
        self._check(self._dll.PurgeComm(handle, PURGE_TXABORT | PURGE_TXCLEAR), "PurgeComm(TX)")

    def send_break(self, handle):
        self._check(self._dll.SetCommBreak(handle), "SetCommBreak")

    def clear_break(self, handle):
        self._check(self._dll.ClearCommBreak(handle), "ClearCommBreak")

    def _event(self):
        event = self._dll.CreateEventW(None, True, False, None)
        if event == self._invalid or not event:
            raise ctypes.WinError(ctypes.get_last_error(), "CreateEventW")
        return event

    def read(self, handle, size, timeout_s):
        buf = ctypes.create_string_buffer(size)
        ov = _OVERLAPPED()
        ov.hEvent = self._event()
        try:
            done = wintypes.DWORD()
            if not self._dll.ReadFile(handle, buf, size, None, ctypes.byref(ov)):
                err = ctypes.get_last_error()
                if err != ERROR_IO_PENDING:
                    raise ctypes.WinError(err, "ReadFile")
                waited = self._dll.WaitForSingleObject(ov.hEvent, max(0, int(timeout_s * 1000)))
                if waited == WAIT_TIMEOUT:
                    self._dll.CancelIoEx(handle, ctypes.byref(ov))
                    self._dll.WaitForSingleObject(ov.hEvent, 2000)
                    return b""
                if waited != WAIT_OBJECT_0:
                    raise ctypes.WinError(ctypes.get_last_error(), "WaitForSingleObject")
            if not self._dll.GetOverlappedResult(handle, ctypes.byref(ov), ctypes.byref(done), False):
                raise ctypes.WinError(ctypes.get_last_error(), "GetOverlappedResult")
            return bytes(buf.raw[:done.value])
        finally:
            self._dll.CloseHandle(ov.hEvent)

    def write(self, handle, data, timeout_s):
        ov = _OVERLAPPED()
        ov.hEvent = self._event()
        try:
            done = wintypes.DWORD()
            if not self._dll.WriteFile(handle, data, len(data), None, ctypes.byref(ov)):
                err = ctypes.get_last_error()
                if err != ERROR_IO_PENDING:
                    raise ctypes.WinError(err, "WriteFile")
                waited = self._dll.WaitForSingleObject(ov.hEvent, max(0, int(timeout_s * 1000)))
                if waited == WAIT_TIMEOUT:
                    self._dll.CancelIoEx(handle, ctypes.byref(ov))
                    self._dll.WaitForSingleObject(ov.hEvent, 2000)
                    if not self._dll.GetOverlappedResult(handle, ctypes.byref(ov),
                                                         ctypes.byref(done), False):
                        return done.value
                    return done.value
                if waited != WAIT_OBJECT_0:
                    raise ctypes.WinError(ctypes.get_last_error(), "WaitForSingleObject")
            if not self._dll.GetOverlappedResult(handle, ctypes.byref(ov), ctypes.byref(done), False):
                raise ctypes.WinError(ctypes.get_last_error(), "GetOverlappedResult")
            return done.value
        finally:
            self._dll.CloseHandle(ov.hEvent)

    def cancel(self, handle):
        try:
            self._dll.CancelIoEx(handle, None)
        except OSError:
            pass


def probe_device(device):
    """Open a COM port just long enough to prove it exists and is usable."""
    short, path = normalize_com_device(device)
    kernel = _RealKernel()
    handle = kernel.create_file(path)
    try:
        kernel.modem_status(handle)
    finally:
        try:
            kernel.close_handle(handle)
        except OSError:
            pass
    return short


class ComSerial:
    """DCE side of a virtual COM pair on Windows. See module docstring."""

    supports_modem_control = True
    is_com = True

    def __init__(self, device, baud=115200, _kernel=None):
        short, path = normalize_com_device(device)
        baud = check_baud(baud)
        if _kernel is None:
            if os.name != "nt":
                raise ValueError("COM serial backend requires Windows; choose PTY/TCP")
            kernel = _RealKernel()
        else:
            kernel = _kernel
        handle = kernel.create_file(path)
        try:
            kernel.configure(handle, baud)
            status = kernel.modem_status(handle)
        except BaseException:
            try:
                kernel.close_handle(handle)
            except OSError:
                pass
            raise
        self._kernel = kernel
        self._handle = handle
        self.device = short
        self.name = short
        self.baud = baud
        self.pending = bytearray()
        self.hardware = False
        self.hardware_rts = False
        self.hardware_cts = False
        self._rts_asserted = False
        self._ring = False
        self._prev_dsr = bool(status & MS_DSR_ON)
        self._dead = False
        self._rx = queue.Queue()
        self._breaks = queue.Queue()
        self._break_lock = threading.Lock()
        self._break_timer = None
        self._break_generation = 0
        self._break_active = False
        self._output_shadow = bytearray()
        self._can_reorder_break = (callable(getattr(kernel, "output_pending", None))
                                   and callable(getattr(kernel, "discard_output", None)))
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._pump, name="comserial-%s" % short.lower(),
                                        daemon=True)
        self._reader.start()
        self.update_lines(False, False, True, False)

    # ----- modem-control lines -----

    def _poll(self):
        if self._dead:
            return 0
        try:
            return self._kernel.modem_status(self._handle)
        except OSError:
            self._dead = True
            return 0

    def client_present(self):
        """Peer DTR asserted (terminal program holds the far end open)."""
        return bool(self._poll() & MS_DSR_ON)

    def signals(self):
        """TR/RTS/CTS for the dashboard lights, matching DriverSerial shape."""
        status = self._poll()
        return dict(TR=bool(status & MS_DSR_ON),
                    RTS=bool(status & MS_CTS_ON),
                    CTS=self._rts_asserted)

    def update_lines(self, carrier, ring, ready, hardware, backlog=0, *, rts_flow=None, cts_flow=None):
        """Drive DTR=carrier (peer DCD) and RTS=flow state (peer CTS).

        ``ring`` is recorded for a future DCE-capable presenting driver;
        com0com static wiring has no independent RI pin, so there is no
        pin action today (option B). Ringing still reaches terminals as
        the Hayes RING result code.
        """
        self.hardware_rts = bool(hardware if rts_flow is None else (hardware or rts_flow))
        self.hardware_cts = bool(hardware if cts_flow is None else (hardware or cts_flow))
        self.hardware = self.hardware_rts or self.hardware_cts
        self._ring = bool(ring)
        cts_out = bool(ready) and (not self.hardware_cts or backlog < COM_CTS_BACKLOG_BITS)
        try:
            self._kernel.set_dtr(self._handle, bool(carrier))
            self._kernel.set_rts(self._handle, cts_out)
        except OSError:
            self._dead = True
            raise
        self._rts_asserted = cts_out

    # ----- byte pump -----

    def _signal_wakeup(self):
        try:
            self._wake_w.send(b"\0")
        except (BlockingIOError, OSError):
            pass

    def _pump(self):
        while not self._stop.is_set():
            try:
                data = self._kernel.read(self._handle, 4096, 0.15)
                take_break = getattr(self._kernel, "take_break", None)
                got_break = bool(take_break(self._handle)) if take_break is not None else False
            except OSError:
                self._dead = True
                self._signal_wakeup()
                return
            if data:
                self._rx.put(data)
            if got_break:
                self._breaks.put(1)
            if data or got_break:
                self._signal_wakeup()

    def _drain_wakeup(self):
        try:
            while self._wake_r.recv(65536):
                pass
        except (BlockingIOError, OSError):
            pass

    def _may_send(self):
        if self._dead:
            return False
        if not self.hardware_rts:
            return True
        return bool(self._poll() & MS_CTS_ON)

    def _flush(self):
        if not self.pending or self._dead or self._break_active:
            return
        if self.hardware_rts and not bool(self._poll() & MS_CTS_ON):
            return
        try:
            sent = self._kernel.write(self._handle, bytes(self.pending), 0.25)
        except OSError:
            self._dead = True
            raise
        del self.pending[:sent]

    # ----- DriverSerial-compatible interface -----

    def fds(self):
        return [self._wake_r]

    def write_fds(self):
        return [self._wake_r] if self.pending and not self._break_active and self._may_send() else []

    def service(self, readable, writable=()):
        if self._wake_r in writable:
            try:
                self._flush()
            except OSError:
                pass
        status = self._poll()
        dtr = bool(status & MS_DSR_ON)
        dropped = self._prev_dsr and not dtr
        self._prev_dsr = dtr
        data = b""
        if self._wake_r in readable:
            self._drain_wakeup()
            while True:
                try:
                    data += self._rx.get_nowait()
                except queue.Empty:
                    break
        return data, dropped

    def take_break_events(self):
        count = 0
        while True:
            try:
                count += self._breaks.get_nowait()
            except queue.Empty:
                return count

    def send_break(self, duration_ms=250):
        """Drive the COM line BREAK signal asynchronously for the requested duration."""
        sender = getattr(self._kernel, "send_break", None)
        clearer = getattr(self._kernel, "clear_break", None)
        if sender is None or clearer is None:
            return False
        with self._break_lock:
            if self._break_timer is not None:
                self._break_timer.cancel()
            self._break_generation += 1
            generation = self._break_generation
            self._break_active = True
            sender(self._handle)
            timer = threading.Timer(max(1, int(duration_ms)) / 1000.0,
                                    self._clear_break, args=(generation,))
            timer.daemon = True
            self._break_timer = timer
            timer.start()
        return True

    def output_pending(self):
        """Queued DTE output bytes, or None when the backend cannot report it."""
        query = getattr(self._kernel, "output_pending", None)
        if query is None:
            return None
        try:
            total = len(self.pending) + max(0, int(query(self._handle)))
            if len(self._output_shadow) >= total:
                del self._output_shadow[:len(self._output_shadow) - total]
            return total
        except OSError:
            return None

    def discard_output(self):
        """Discard output not yet delivered to the DTE (V.42 destructive break)."""
        purge = getattr(self._kernel, "discard_output", None)
        if purge is None:
            return False
        try:
            purge(self._handle)
            self.pending.clear()
            self._output_shadow.clear()
        except OSError:
            return False
        return True

    def place_break_before_pending_output(self):
        """Requeue undelivered bytes after the next BREAK signal."""
        query = getattr(self._kernel, "output_pending", None)
        purge = getattr(self._kernel, "discard_output", None)
        if not self._can_reorder_break or query is None or purge is None:
            return False
        try:
            total = len(self.pending) + max(0, int(query(self._handle)))
            if len(self._output_shadow) < total:
                return False
            del self._output_shadow[:len(self._output_shadow) - total]
            purge(self._handle)
            self.pending = bytearray(self._output_shadow)
        except OSError:
            return False
        return True

    def _clear_break(self, generation):
        with self._break_lock:
            if generation != self._break_generation:
                return
            self._break_timer = None
            if self._stop.is_set():
                return
            try:
                self._kernel.clear_break(self._handle)
            except OSError:
                self._dead = True
            self._break_active = False
            self._signal_wakeup()

    def write(self, data):
        if self._dead:
            raise OSError("COM port is no longer usable: %s" % self.device)
        if len(self.pending) + len(data) > PENDING_LIMIT:
            raise BufferError("serial receive queue exhausted")
        self.pending.extend(data)
        if self._can_reorder_break:
            self._output_shadow.extend(data)
        self._flush()
        self.output_pending()

    def open_terminal(self):
        raise ValueError("COM backend cannot host the dashboard terminal: attach a terminal"
                         " program to the peer port paired with %s" % self.device)

    def close(self):
        self._stop.set()
        with self._break_lock:
            self._break_generation += 1
            if self._break_timer is not None:
                self._break_timer.cancel()
                self._break_timer = None
            try:
                self._kernel.clear_break(self._handle)
            except (AttributeError, OSError):
                pass
        try:
            self.update_lines(False, False, False, False)
        except OSError:
            pass
        try:
            self._kernel.cancel(self._handle)
        except OSError:
            pass
        self._reader.join(timeout=3)
        try:
            self._kernel.close_handle(self._handle)
        except OSError:
            pass
        for sock in (self._wake_r, self._wake_w):
            try:
                sock.close()
            except OSError:
                pass
