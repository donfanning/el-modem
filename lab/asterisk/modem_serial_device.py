"""Native POSIX serial-device backend for Linux and macOS.

This is for a real tty device (for example /dev/ttyUSB0 or /dev/cu.usbserial-*).
The modem process owns the device. The TUI's terminal is multiplexed through the
worker control channel while it is open, so it never races another process for
the tty descriptor.
"""
from __future__ import annotations

import array
import errno
import os

if os.name != 'posix':
    raise ImportError('native serial-device backend is available on POSIX systems')

import fcntl
import termios
import tty


_BAUDS = {rate: getattr(termios, f'B{rate}') for rate in
          (300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600,
           115200, 230400, 460800, 500000, 576000, 921600)
          if hasattr(termios, f'B{rate}')}
SUPPORTED_BAUDS = frozenset(_BAUDS)


def normalize_device(path: str) -> str:
    path = os.path.expanduser(str(path or '').strip())
    if not path or not os.path.isabs(path) or '\x00' in path:
        raise ValueError('serial device must be an absolute path, such as /dev/ttyUSB0 or /dev/cu.usbserial-*')
    return os.path.normpath(path)


class DeviceSerial:
    """Own a POSIX character tty and expose the modem serial interface."""
    terminal_is_internal = True

    def __init__(self, device: str, baud: int = 115200):
        self.device = normalize_device(device)
        if isinstance(baud, bool) or baud not in _BAUDS:
            raise ValueError(f'unsupported serial baud rate {baud!r}')
        self.baud = baud
        flags = os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
        flags |= getattr(os, 'O_CLOEXEC', 0)
        self.fd = os.open(self.device, flags)
        self._pending = bytearray()
        self._terminal_input = bytearray()
        self._terminal_output = bytearray()
        self._terminal_active = False
        self._hardware_flow = False
        self._previous_dtr = self._get_line(getattr(termios, 'TIOCM_DTR', 0))
        try:
            tty.setraw(self.fd, termios.TCSANOW)
            attrs = termios.tcgetattr(self.fd)
            attrs[4] = attrs[5] = _BAUDS[baud]
            attrs[2] |= termios.CLOCAL | termios.CREAD
            if hasattr(termios, 'CRTSCTS'):
                attrs[2] &= ~termios.CRTSCTS
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            if hasattr(termios, 'TIOCEXCL'):
                fcntl.ioctl(self.fd, termios.TIOCEXCL)
        except BaseException:
            os.close(self.fd)
            raise
        self.name = self.device

    def fds(self):
        return [] if self._terminal_active else [self.fd]

    def write_fds(self):
        if self._terminal_active or not self._pending:
            return []
        if self._hardware_flow and self._get_line(getattr(termios, 'TIOCM_RTS', 0)) is False:
            return []
        return [self.fd]

    def service(self, readable, writable=()):
        if not self._terminal_active and self.fd in writable:
            self._flush()
        data = bytearray()
        if self._terminal_active:
            data.extend(self._terminal_input[:4096])
            del self._terminal_input[:4096]
        elif self.fd in readable:
            try:
                data.extend(os.read(self.fd, 4096))
            except BlockingIOError:
                pass
        dtr = self._get_line(getattr(termios, 'TIOCM_DTR', 0))
        dropped = self._previous_dtr is True and dtr is False
        if dtr is not None:
            self._previous_dtr = dtr
        return bytes(data), dropped

    def _get_line(self, bit):
        if not bit or not hasattr(termios, 'TIOCMGET'):
            return None
        bits = array.array('i', [0])
        try:
            fcntl.ioctl(self.fd, termios.TIOCMGET, bits, True)
        except OSError as exc:
            if exc.errno in (errno.ENOTTY, errno.EINVAL, errno.ENOSYS):
                return None
            raise
        return bool(bits[0] & bit)

    def _set_lines(self, mask, value):
        if not hasattr(termios, 'TIOCMBIS') or not hasattr(termios, 'TIOCMBIC'):
            return
        for bit in (1 << n for n in range(16)):
            if mask & bit:
                request = termios.TIOCMBIS if value & bit else termios.TIOCMBIC
                bits = array.array('i', [bit])
                try:
                    fcntl.ioctl(self.fd, request, bits, True)
                except OSError as exc:
                    if exc.errno not in (errno.ENOTTY, errno.EINVAL, errno.ENOSYS):
                        raise

    def signals(self):
        names = {'TR': 'TIOCM_DTR', 'RTS': 'TIOCM_RTS', 'CTS': 'TIOCM_CTS',
                 'DSR': 'TIOCM_DSR', 'CD': 'TIOCM_CAR', 'RI': 'TIOCM_RI'}
        return {key: self._get_line(getattr(termios, attr, 0)) for key, attr in names.items()}

    def update_lines(self, carrier, ring, ready, hardware, backlog=0):
        self._hardware_flow = bool(hardware)
        cts = bool(ready and (not hardware or backlog < 40960))
        outputs = (('TIOCM_DSR', bool(ready)), ('TIOCM_CAR', bool(carrier)),
                   ('TIOCM_RI', bool(ring)), ('TIOCM_CTS', cts))
        mask = 0
        value = 0
        for name, enabled in outputs:
            bit = getattr(termios, name, 0)
            mask |= bit
            if enabled:
                value |= bit
        self._set_lines(mask, value)

    def _flush(self):
        if not self._pending:
            return
        if self._hardware_flow and self._get_line(getattr(termios, 'TIOCM_RTS', 0)) is False:
            return
        try:
            count = os.write(self.fd, self._pending)
            del self._pending[:count]
        except BlockingIOError:
            pass

    def write(self, data: bytes):
        if self._terminal_active:
            if len(self._terminal_output) + len(data) > 65536:
                raise BufferError('terminal output queue exhausted')
            self._terminal_output.extend(data)
            return
        if len(self._pending) + len(data) > 65536:
            raise BufferError('serial transmit queue exhausted')
        self._pending.extend(data)
        self._flush()

    def client_present(self):
        # A real tty remains owned by this worker. DTR is the best available
        # indication that an attached DTE has asserted its side of the link.
        return bool(self._get_line(getattr(termios, 'TIOCM_DTR', 0)))

    def open_terminal(self):
        if self._terminal_active:
            raise ValueError('terminal already open')
        self._terminal_active = True
        self._terminal_input.clear()
        self._terminal_output.clear()
        return self

    def terminal_write(self, data):
        if not self._terminal_active:
            raise ValueError('terminal does not own modem')
        if len(self._terminal_input) + len(data) > 65536:
            raise BufferError('terminal input queue full')
        self._terminal_input.extend(data)

    def terminal_read(self, limit=4096):
        data = bytes(self._terminal_output[:limit])
        del self._terminal_output[:limit]
        return data

    def release_terminal(self, _handle=None):
        self._terminal_active = False
        self._terminal_input.clear()
        self._terminal_output.clear()

    def close(self):
        self.release_terminal()
        os.close(self.fd)
