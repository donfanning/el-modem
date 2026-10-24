"""Native POSIX serial-device backend for Linux and macOS.

This is for a real tty device (for example /dev/ttyUSB0 or /dev/cu.usbserial-*).
The modem process owns the device. The TUI's terminal is multiplexed through the
worker control channel while it is open, so it never races another process for
the tty descriptor. Where the platform tty supports it, PARMRK reports input
BREAK as a sideband event and TIOCSBRK/TIOCCBRK provide timed output BREAK.
"""
from __future__ import annotations

import array
import errno
import os
import struct
import sys
import threading

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
_TCGETS2 = 0x802C542A
_TCSETS2 = 0x402C542B
_CBAUD = 0x100F
_BOTHER = 0x1000


def _is_bell_custom_rate(rate: int) -> bool:
    return sys.platform.startswith("linux") and 1170 <= rate <= 1212


def _linux_termios2(fd: int) -> bytearray | None:
    if not sys.platform.startswith("linux"):
        return None
    attrs = bytearray(44)  # Linux struct termios2, NCCS=19
    try:
        fcntl.ioctl(fd, _TCGETS2, attrs, True)
    except OSError:
        return None
    return attrs


def _linux_set_custom_baud(fd: int, baud: int) -> bool:
    original = _linux_termios2(fd)
    if original is None:
        return False
    attrs = bytearray(original)
    cflag, = struct.unpack_from("=I", attrs, 8)
    struct.pack_into("=I", attrs, 8, (cflag & ~_CBAUD) | _BOTHER)
    struct.pack_into("=II", attrs, 36, baud, baud)
    try:
        fcntl.ioctl(fd, _TCSETS2, attrs, True)
        applied = _linux_termios2(fd)
        if applied is not None and struct.unpack_from("=II", applied, 36) == (baud, baud):
            return True
    except OSError:
        pass
    try:
        fcntl.ioctl(fd, _TCSETS2, original, True)
    except OSError:
        pass
    return False


def decode_parmrk(data: bytes, state: int = 0) -> tuple[bytes, int, int]:
    """Decode POSIX PARMRK input, separating BREAK from ordinary byte data.

    PARMRK represents a break as ``ff 00 00`` and a literal 0xff as
    ``ff ff``. A ``ff 00 x`` parity/framing-error marker is reduced to its
    received data octet; the native backend can configure termios framing, but
    the DCE has no parity/framing-error indication API.
    """
    output = bytearray()
    breaks = 0
    for value in data:
        if state == 0:
            if value == 0xFF:
                state = 1
            else:
                output.append(value)
        elif state == 1:
            if value == 0xFF:
                output.append(0xFF)
                state = 0
            elif value == 0:
                state = 2
            else:
                output.extend((0xFF, value))
                state = 0
        else:
            if value == 0:
                breaks += 1
            else:
                output.append(value)
            state = 0
    return bytes(output), breaks, state


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
        if isinstance(baud, bool) or (baud not in _BAUDS and not _is_bell_custom_rate(baud)):
            raise ValueError(f'unsupported serial baud rate {baud!r}')
        self.baud = baud
        flags = os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
        flags |= getattr(os, 'O_CLOEXEC', 0)
        self.fd = os.open(self.device, flags)
        self._pending = bytearray()
        # Mirror bytes submitted to the tty until TIOCOUTQ says they have
        # physically drained. This lets V.42 S=1 move undelivered bytes after
        # an out-of-band BREAK without losing their contents.
        self._output_shadow = bytearray()
        self._can_reorder_break = (hasattr(termios, 'TIOCOUTQ')
                                   and hasattr(termios, 'TCOFLUSH'))
        self._terminal_input = bytearray()
        self._terminal_output = bytearray()
        self._terminal_active = False
        self._hardware_flow = False
        self._parmrk_state = 0
        self._parmrk_enabled = hasattr(termios, 'PARMRK')
        self._break_events = 0
        self._break_lock = threading.Lock()
        self._break_timer = None
        self._break_generation = 0
        self._break_active = False
        self._closed = False
        self._previous_dtr = self._get_line(getattr(termios, 'TIOCM_DTR', 0))
        try:
            tty.setraw(self.fd, termios.TCSANOW)
            attrs = termios.tcgetattr(self.fd)
            initial_rate = baud if baud in _BAUDS else 1200
            attrs[4] = attrs[5] = _BAUDS[initial_rate]
            attrs[2] |= termios.CLOCAL | termios.CREAD
            if hasattr(termios, 'CRTSCTS'):
                attrs[2] &= ~termios.CRTSCTS
            if hasattr(termios, 'PARMRK'):
                attrs[0] &= ~(getattr(termios, 'IGNBRK', 0) | getattr(termios, 'BRKINT', 0))
                attrs[0] |= termios.PARMRK
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            if baud not in _BAUDS and not _linux_set_custom_baud(self.fd, baud):
                raise ValueError(f'this serial device cannot apply custom baud {baud}')
            if hasattr(termios, 'TIOCEXCL'):
                fcntl.ioctl(self.fd, termios.TIOCEXCL)
            # USB serial adapters vary: advertise line-based IFC only when
            # this particular device can report the required modem signals.
            signal_state = self.signals()
            self.supports_modem_control = signal_state['RTS'] is not None and signal_state['CTS'] is not None
        except BaseException:
            os.close(self.fd)
            raise
        self.name = self.device

    def supported_dte_baud_rates(self) -> tuple[int, ...]:
        rates = set(SUPPORTED_BAUDS)
        if _linux_termios2(self.fd) is not None:
            rates.update(range(1170, 1213))
        return tuple(sorted(rates))

    def get_dte_baud(self) -> int | None:
        attrs2 = _linux_termios2(self.fd)
        if attrs2 is not None:
            input_baud, output_baud = struct.unpack_from("=II", attrs2, 36)
            if input_baud == output_baud and input_baud:
                return input_baud
        try:
            attrs = termios.tcgetattr(self.fd)
        except (OSError, termios.error):
            return None
        if attrs[4] != attrs[5]:
            return None
        return next((rate for rate, token in _BAUDS.items() if token == attrs[4]), None)

    def set_dte_baud(self, baud: int) -> bool:
        baud = int(baud)
        token = _BAUDS.get(int(baud))
        if token is None:
            if not _is_bell_custom_rate(baud) or not _linux_set_custom_baud(self.fd, baud):
                return False
            self.baud = baud
            return True
        original = None
        try:
            original = termios.tcgetattr(self.fd)
            attrs = original[:]
            attrs[4] = attrs[5] = token
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            if self.get_dte_baud() != int(baud):
                termios.tcsetattr(self.fd, termios.TCSANOW, original)
                return False
            self.baud = int(baud)
            return True
        except (OSError, termios.error, ValueError):
            return False

    @staticmethod
    def _format_cflag(cflag: int, format_code: int, parity_code: int) -> int:
        if format_code not in (1, 2, 3, 4, 5, 6) or parity_code not in (0, 1, 2, 3):
            raise ValueError("unsupported V.250 character format")
        data_bits, stops, uses_parity = {
            1: (8, 2, False), 2: (8, 1, True), 3: (8, 1, False),
            4: (7, 2, False), 5: (7, 1, True), 6: (7, 1, False),
        }[format_code]
        cflag &= ~(termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                   | getattr(termios, "CMSPAR", 0))
        cflag |= termios.CS8 if data_bits == 8 else termios.CS7
        if stops == 2:
            cflag |= termios.CSTOPB
        if uses_parity:
            if parity_code in (2, 3) and not hasattr(termios, "CMSPAR"):
                raise ValueError("mark/space parity is unavailable")
            cflag |= termios.PARENB
            if parity_code == 0 or parity_code == 2:
                cflag |= termios.PARODD
            if parity_code in (2, 3):
                cflag |= termios.CMSPAR
        return cflag

    def supported_dte_character_formats(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        required = ("CS7", "CS8", "CSTOPB", "PARENB", "PARODD")
        if not all(hasattr(termios, name) for name in required):
            return (3,), (3,)
        original = termios.tcgetattr(self.fd)
        formats = []
        parities = set()
        try:
            for format_code in (1, 2, 3, 4, 5, 6):
                uses_parity = format_code in (2, 5)
                candidates = (0, 1, 2, 3) if uses_parity else (3,)
                format_supported = False
                for parity_code in candidates:
                    trial = original[:]
                    try:
                        trial[2] = self._format_cflag(original[2], format_code, parity_code)
                        termios.tcsetattr(self.fd, termios.TCSANOW, trial)
                        applied = termios.tcgetattr(self.fd)[2]
                    except (OSError, termios.error, ValueError):
                        continue
                    relevant = (termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                                | getattr(termios, "CMSPAR", 0))
                    if applied & relevant == trial[2] & relevant:
                        format_supported = True
                        if uses_parity:
                            parities.add(parity_code)
                if format_supported:
                    formats.append(format_code)
        finally:
            termios.tcsetattr(self.fd, termios.TCSANOW, original)
        return tuple(formats), tuple(sorted(parities or {0, 1, 2, 3}))

    def set_dte_character_format(self, format_code: int, parity_code: int) -> bool:
        formats, parities = self.supported_dte_character_formats()
        if format_code not in formats or parity_code not in (0, 1, 2, 3):
            return False
        uses_parity = format_code in (2, 5)
        if uses_parity and parity_code not in parities:
            return False
        try:
            original = termios.tcgetattr(self.fd)
            attrs = original[:]
            attrs[2] = self._format_cflag(original[2], format_code, parity_code)
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            applied = termios.tcgetattr(self.fd)[2]
            relevant = (termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                        | getattr(termios, "CMSPAR", 0))
            if applied & relevant != attrs[2] & relevant:
                termios.tcsetattr(self.fd, termios.TCSANOW, original)
                return False
            return True
        except (OSError, termios.error, ValueError):
            return False

    def set_bell212_dte_format(self, character_bits: int, parity: str) -> bool:
        """Apply Bell 212A asynchronous DTE framing to a native UART.

        Bell's project extension uses 9/10-bit start/data/stop characters and
        11/12-bit start/data/parity/stop characters. Unlike a PTY, a native
        serial device can generate these physical UART framing bits.
        """
        if character_bits not in (9, 10, 11, 12) or parity not in ("E", "O", "M", "S"):
            return False
        data_bits = 7 if character_bits == 9 else 8
        uses_parity = character_bits >= 11
        stops = 2 if character_bits in (12,) else 1
        parity_code = {"O": 0, "E": 1, "M": 2, "S": 3}[parity]
        original = None
        try:
            original = termios.tcgetattr(self.fd)
            attrs = original[:]
            cflag = attrs[2]
            cflag &= ~(termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                       | getattr(termios, "CMSPAR", 0))
            cflag |= termios.CS7 if data_bits == 7 else termios.CS8
            if stops == 2:
                cflag |= termios.CSTOPB
            if uses_parity:
                if parity_code in (2, 3) and not hasattr(termios, "CMSPAR"):
                    return False
                cflag |= termios.PARENB
                if parity_code in (0, 2):
                    cflag |= termios.PARODD
                if parity_code in (2, 3):
                    cflag |= termios.CMSPAR
            attrs[2] = cflag
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            applied = termios.tcgetattr(self.fd)[2]
            relevant = (termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                        | getattr(termios, "CMSPAR", 0))
            if applied & relevant != cflag & relevant:
                termios.tcsetattr(self.fd, termios.TCSANOW, original)
                return False
            return True
        except (OSError, termios.error, ValueError):
            if original is not None:
                try:
                    termios.tcsetattr(self.fd, termios.TCSANOW, original)
                except (OSError, termios.error):
                    pass
            return False

    def fds(self):
        return [] if self._terminal_active else [self.fd]

    def write_fds(self):
        if self._terminal_active or self._break_active or not self._pending:
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
                raw = os.read(self.fd, 4096)
                if self._parmrk_enabled:
                    decoded, breaks, self._parmrk_state = decode_parmrk(raw, self._parmrk_state)
                else:
                    decoded, breaks = raw, 0
                data.extend(decoded)
                self._break_events += breaks
            except BlockingIOError:
                pass
        dtr = self._get_line(getattr(termios, 'TIOCM_DTR', 0))
        dropped = self._previous_dtr is True and dtr is False
        if dtr is not None:
            self._previous_dtr = dtr
        return bytes(data), dropped

    def take_break_events(self):
        """Return and clear BREAK conditions decoded from PARMRK input."""
        count, self._break_events = self._break_events, 0
        return count

    def send_break(self, duration_ms: int = 250) -> bool:
        """Assert a real tty BREAK without blocking the modem's event loop."""
        set_break = getattr(termios, 'TIOCSBRK', None)
        clear_break = getattr(termios, 'TIOCCBRK', None)
        if set_break is None or clear_break is None:
            return False
        duration_ms = max(1, int(duration_ms))
        with self._break_lock:
            if self._break_timer is not None:
                self._break_timer.cancel()
            self._break_generation += 1
            generation = self._break_generation
            self._break_active = True
            fcntl.ioctl(self.fd, set_break)
            timer = threading.Timer(duration_ms / 1000.0, self._clear_break,
                                    args=(generation,))
            timer.daemon = True
            self._break_timer = timer
            timer.start()
        return True

    def output_pending(self) -> int | None:
        """Queued DTE output bytes, or None when the tty cannot report it."""
        ioctl = getattr(termios, 'TIOCOUTQ', None)
        if ioctl is None:
            return None
        queued = array.array('i', [0])
        try:
            fcntl.ioctl(self.fd, ioctl, queued, True)
        except OSError:
            return None
        total = len(self._pending) + max(0, int(queued[0]))
        if len(self._output_shadow) >= total:
            del self._output_shadow[:len(self._output_shadow) - total]
        return total

    def discard_output(self) -> bool:
        """Discard bytes not yet delivered to the DTE (V.42 destructive break)."""
        flush = getattr(termios, 'TCOFLUSH', None)
        if flush is None:
            return False
        try:
            termios.tcflush(self.fd, flush)
            self._pending.clear()
            self._output_shadow.clear()
        except OSError:
            return False
        return True

    def place_break_before_pending_output(self) -> bool:
        """Requeue undelivered bytes after the next BREAK signal."""
        flush = getattr(termios, 'TCOFLUSH', None)
        ioctl = getattr(termios, 'TIOCOUTQ', None)
        if not self._can_reorder_break or flush is None or ioctl is None:
            return False
        queued = array.array('i', [0])
        try:
            fcntl.ioctl(self.fd, ioctl, queued, True)
            total = len(self._pending) + max(0, int(queued[0]))
            if len(self._output_shadow) < total:
                return False
            del self._output_shadow[:len(self._output_shadow) - total]
            termios.tcflush(self.fd, flush)
            self._pending = bytearray(self._output_shadow)
        except OSError:
            return False
        return True

    def _clear_break(self, generation):
        clear_break = getattr(termios, 'TIOCCBRK', None)
        with self._break_lock:
            if generation != self._break_generation:
                return
            self._break_timer = None
            if self._closed:
                return
            if clear_break is not None:
                try:
                    fcntl.ioctl(self.fd, clear_break)
                except OSError:
                    pass
            self._break_active = False

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

    def update_lines(self, carrier, ring, ready, hardware, backlog=0, *, rts_flow=None, cts_flow=None):
        self._hardware_flow = bool(hardware if rts_flow is None else (hardware or rts_flow))
        gate_cts = bool(hardware if cts_flow is None else (hardware or cts_flow))
        cts = bool(ready and (not gate_cts or backlog < 40960))
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
        if not self._pending or self._break_active:
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
        if self._can_reorder_break:
            self._output_shadow.extend(data)
        self._flush()
        self.output_pending()

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
        with self._break_lock:
            self._closed = True
            self._break_generation += 1
            if self._break_timer is not None:
                self._break_timer.cancel()
                self._break_timer = None
            if hasattr(termios, 'TIOCCBRK'):
                try:
                    fcntl.ioctl(self.fd, termios.TIOCCBRK)
                except OSError:
                    pass
        os.close(self.fd)
