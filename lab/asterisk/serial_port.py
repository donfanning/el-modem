"""The modem's serial side, in two flavours with one interface (our own code).

``PtySerial``  POSIX: a pseudo-terminal that minicom/screen/pppd open directly. The slave is not kept open here, so
               the master reading EIO means the terminal closed the port, which is how a DTR drop is seen.
``TcpSerial``  every OS (the only one on Windows): the modem listens on a TCP port and one client at a time is "the
               terminal". A client disconnecting is a DTR drop. Terminal programs reach it with a telnet/raw-TCP
               client, or on Windows through a virtual COM-port bridge (com0com + a TCP redirector such as hub4com).

Interface: ``fds()`` and ``write_fds()`` -> selectable objects; ``service(readable, writable)`` -> (data, dropped); ``write(data)`` queues output;
``name`` (what to show the user); ``close()``.
"""
import os
import socket
import sys
import time
from serial_metadata import encode_baud, encode_character_format, take_metadata_frames


_TCGETS2 = 0x802C542A
_TCSETS2 = 0x402C542B
_CBAUD = 0x100F
_BOTHER = 0x1000


def _linux_termios2(fd: int) -> bytearray | None:
    if not sys.platform.startswith("linux"):
        return None
    import fcntl  # noqa: PLC0415 -- Linux only

    attrs = bytearray(44)  # Linux struct termios2, NCCS=19
    try:
        fcntl.ioctl(fd, _TCGETS2, attrs, True)
    except OSError:
        return None
    return attrs


def _linux_set_custom_baud(fd: int, baud: int) -> bool:
    import fcntl  # noqa: PLC0415 -- Linux only
    import struct  # noqa: PLC0415 -- Linux only

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


def get_tty_baud(fd: int) -> int | None:
    """Read a tty's nominal input/output baud, including Linux BOTHER rates."""
    import termios  # noqa: PLC0415 -- POSIX only
    import struct  # noqa: PLC0415

    attrs2 = _linux_termios2(fd)
    if attrs2 is not None:
        input_baud, output_baud = struct.unpack_from("=II", attrs2, 36)
        if input_baud == output_baud and input_baud:
            return input_baud
        return None
    try:
        attrs = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return None
    if attrs[4] != attrs[5]:
        return None
    speeds = {
        value: int(name[1:])
        for name, value in vars(termios).items()
        if name.startswith("B") and name[1:].isdigit() and isinstance(value, int)
    }
    return speeds.get(attrs[4])


def set_tty_baud(fd: int, baud: int) -> bool:
    """Set equal input/output baud on a tty, including Linux Bell custom rates."""
    import termios  # noqa: PLC0415 -- POSIX only

    rate = int(baud)
    token = getattr(termios, f"B{rate}", None)
    if token is None and 1170 <= rate <= 1212:
        return _linux_set_custom_baud(fd, rate)
    if token is None:
        return False
    try:
        attrs = termios.tcgetattr(fd)
        attrs[4] = attrs[5] = token
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        return get_tty_baud(fd) == rate
    except (OSError, termios.error, ValueError):
        return False


def get_tty_character_format(fd: int) -> tuple[int, int] | None:
    """Return (V.250 format, parity) from POSIX termios, if representable."""
    import termios  # noqa: PLC0415 -- POSIX only
    try:
        cflag = termios.tcgetattr(fd)[2]
    except (OSError, termios.error):
        return None
    bits = 8 if cflag & termios.CSIZE == termios.CS8 else 7 if cflag & termios.CSIZE == termios.CS7 else 0
    stops = 2 if cflag & termios.CSTOPB else 1
    enabled = bool(cflag & termios.PARENB)
    fmt = {(8, 2, False): 1, (8, 1, True): 2, (8, 1, False): 3,
           (7, 2, False): 4, (7, 1, True): 5, (7, 1, False): 6}.get((bits, stops, enabled))
    if fmt is None:
        return None
    parity = 0 if cflag & termios.PARODD else 1
    if cflag & getattr(termios, "CMSPAR", 0):
        parity = 2 if cflag & termios.PARODD else 3
    return fmt, parity


def supported_tty_character_formats(fd: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Probe formats accepted by this tty without leaving its settings changed."""
    import termios  # noqa: PLC0415 -- POSIX only
    if not all(hasattr(termios, name) for name in ("CS7", "CS8", "CSTOPB", "PARENB", "PARODD")):
        return (3,), (0, 1)
    formats = []
    original = termios.tcgetattr(fd)
    try:
        for code, data_bits, stops in ((1, 8, 2), (2, 8, 1), (3, 8, 1),
                                       (4, 7, 2), (5, 7, 1), (6, 7, 1)):
            trial = original[:]
            cflag = original[2] & ~termios.CSIZE
            cflag |= termios.CS8 if data_bits == 8 else termios.CS7
            cflag &= ~(termios.PARENB | termios.PARODD | getattr(termios, "CMSPAR", 0) | termios.CSTOPB)
            if stops == 2:
                cflag |= termios.CSTOPB
            if code in (2, 5):
                cflag |= termios.PARENB
            trial[2] = cflag
            try:
                termios.tcsetattr(fd, termios.TCSANOW, trial)
                relevant = termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD | getattr(termios, "CMSPAR", 0)
                if termios.tcgetattr(fd)[2] & relevant == cflag & relevant:
                    formats.append(code)
            except termios.error:
                pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)
    parities = (0, 1, 2, 3) if hasattr(termios, "CMSPAR") else (0, 1)
    return tuple(formats), parities


def set_tty_character_format(fd: int, format_code: int, parity_code: int) -> bool:
    """Apply one V.250 +ICF format/parity pair to a POSIX tty."""
    import termios  # noqa: PLC0415 -- POSIX only
    formats, parities = supported_tty_character_formats(fd)
    if format_code not in formats or parity_code not in parities:
        return False
    bits, stops, uses_parity = {1: (8, 2, False), 2: (8, 1, True), 3: (8, 1, False),
                                4: (7, 2, False), 5: (7, 1, True), 6: (7, 1, False)}[format_code]
    if uses_parity and parity_code in (2, 3) and not hasattr(termios, "CMSPAR"):
        return False
    try:
        original = termios.tcgetattr(fd)
        attrs = original[:]
        cflag = attrs[2] & ~termios.CSIZE
        cflag |= termios.CS8 if bits == 8 else termios.CS7
        cflag &= ~(termios.PARENB | termios.PARODD | getattr(termios, "CMSPAR", 0) | termios.CSTOPB)
        if stops == 2:
            cflag |= termios.CSTOPB
        if uses_parity:
            cflag |= termios.PARENB
            if parity_code == 0:
                cflag |= termios.PARODD
            elif parity_code == 2:
                cflag |= termios.CMSPAR | termios.PARODD
            elif parity_code == 3:
                cflag |= termios.CMSPAR
        attrs[2] = cflag
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        return get_tty_character_format(fd) == (format_code, parity_code if uses_parity else 1)
    except (OSError, termios.error, ValueError):
        return False


class PtySerial:
    supports_modem_control = False
    supports_dte_baud = True

    def __init__(self, link: str | None = None):
        import pty  # noqa: PLC0415 -- POSIX only
        import tty  # noqa: PLC0415

        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self.name = os.ttyname(slave)
        os.close(slave)
        os.set_blocking(self.master, False)
        self._open = False
        self._pending = bytearray()
        self._next_probe = 0.0
        self._link = link
        if link:
            os.symlink(self.name, link)

    def fds(self):
        if self._open or time.monotonic() >= self._next_probe:
            return [self.master]
        return []

    def service(self, readable, writable=()):
        if self.master in writable:
            self._flush()
        probing = not self._open and time.monotonic() >= self._next_probe
        if probing:
            self._next_probe = time.monotonic() + 0.1
            if self.master not in readable:
                self._open = True               # not EIO any more: a terminal has opened the port
        dropped = False
        data = b""
        if self.master in readable:
            peer_closed = False
            try:
                data = os.read(self.master, 4096)
                peer_closed = not data
            except BlockingIOError:
                data = b""
            except OSError:
                data = b""
                peer_closed = True               # Linux reports a closed PTY as EIO
            if peer_closed:
                dropped = self._open
                self._open = False
                self._pending.clear()
                self._next_probe = time.monotonic() + 0.1
                # Darwin reports a closed PTY slave as a readable master whose
                # read returns b""; Linux reports EIO. Both mean the terminal
                # dropped DTR.
            if data:
                self._open = True
        return data, dropped

    def write_fds(self):
        return [self.master] if self._pending else []

    def _flush(self):
        try:
            n = os.write(self.master, self._pending)
            del self._pending[:n]
        except BlockingIOError:
            pass
        except OSError:
            self._pending.clear()

    def write(self, data: bytes) -> None:
        self._pending.extend(data)
        self._flush()

    def get_dte_baud(self) -> int | None:
        """Return the PTY's configured baud when input and output agree.

        This reports the nominal termios setting, not measured oscillator
        error. Many terminal applications use a high DTE speed even for a
        low-speed modem connection; callers must decide whether it applies to
        the active modem profile.
        """
        return get_tty_baud(self.master)

    def supported_dte_baud_rates(self) -> tuple[int, ...]:
        """Return rates this PTY can apply, including Linux Bell 212A rates."""
        import termios  # noqa: PLC0415 -- POSIX only

        rates = {int(name[1:]) for name, value in vars(termios).items()
                 if name.startswith("B") and name[1:].isdigit() and isinstance(value, int)}
        rates = {rate for rate in rates if rate > 0}
        # Bell 212A's asynchronous DTE adaptation uses nominal rates in this
        # narrow range. Linux PTYs support arbitrary termios2/BOTHER rates;
        # advertise them only when the running kernel accepts TCGETS2 here.
        if _linux_termios2(self.master) is not None:
            rates.update(range(1170, 1213))
        return tuple(sorted(rates))

    def set_dte_baud(self, baud: int) -> bool:
        """Set both PTY directions to a supported standard or Bell custom baud."""
        return set_tty_baud(self.master, baud)

    def supported_dte_character_formats(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Probe V.250 +ICF formats the active PTY actually accepts."""
        import termios  # noqa: PLC0415 -- POSIX only

        if not all(hasattr(termios, name) for name in ("CS7", "CS8", "CSTOPB", "PARENB", "PARODD")):
            return (3,), (3,)
        formats = []
        original = termios.tcgetattr(self.master)
        try:
            for code, data_bits, stops in ((1, 8, 2), (2, 8, 1), (3, 8, 1),
                                           (4, 7, 2), (5, 7, 1), (6, 7, 1)):
                trial = original[:]
                cflag = original[2] & ~termios.CSIZE
                cflag |= termios.CS8 if data_bits == 8 else termios.CS7
                cflag &= ~(termios.PARENB | termios.PARODD | getattr(termios, "CMSPAR", 0)
                           | termios.CSTOPB)
                if stops == 2:
                    cflag |= termios.CSTOPB
                if code in (2, 5):
                    cflag |= termios.PARENB
                trial[2] = cflag
                try:
                    termios.tcsetattr(self.master, termios.TCSANOW, trial)
                    applied = termios.tcgetattr(self.master)[2]
                    relevant = (termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                                | getattr(termios, "CMSPAR", 0))
                    if applied & relevant == cflag & relevant:
                        formats.append(code)
                except termios.error:
                    pass
        finally:
            termios.tcsetattr(self.master, termios.TCSANOW, original)
        if any(code in formats for code in (2, 5)):
            parities = (0, 1, 2, 3) if hasattr(termios, "CMSPAR") else (0, 1)
        else:
            parities = (0, 1, 2, 3)  # ignored by formats without a parity element
        return tuple(formats), parities

    def get_dte_character_format(self) -> tuple[int, int] | None:
        return get_tty_character_format(self.master)

    def set_dte_character_format(self, format_code: int, parity_code: int) -> bool:
        """Mirror V.250 command/result framing in the PTY's termios state."""
        import termios  # noqa: PLC0415 -- POSIX only

        formats, parities = self.supported_dte_character_formats()
        if format_code not in formats or parity_code not in (0, 1, 2, 3):
            return False
        data_bits, stop_bits, uses_parity = {
            1: (8, 2, False), 2: (8, 1, True), 3: (8, 1, False),
            4: (7, 2, False), 5: (7, 1, True), 6: (7, 1, False),
        }[format_code]
        if uses_parity and parity_code not in parities:
            return False
        if uses_parity and parity_code in (2, 3) and not hasattr(termios, "CMSPAR"):
            return False
        try:
            original = termios.tcgetattr(self.master)
            attrs = original[:]
            cflag = attrs[2] & ~termios.CSIZE
            cflag |= termios.CS8 if data_bits == 8 else termios.CS7
            cflag &= ~(termios.PARENB | termios.PARODD | getattr(termios, "CMSPAR", 0)
                       | termios.CSTOPB)
            if stop_bits == 2:
                cflag |= termios.CSTOPB
            if uses_parity:
                cflag |= termios.PARENB
                if parity_code == 0:  # odd
                    cflag |= termios.PARODD
                elif parity_code == 2:  # mark
                    cflag |= termios.CMSPAR | termios.PARODD
                elif parity_code == 3:  # space
                    cflag |= termios.CMSPAR
            attrs[2] = cflag
            termios.tcsetattr(self.master, termios.TCSANOW, attrs)
            applied = termios.tcgetattr(self.master)[2]
            relevant = (termios.CSIZE | termios.CSTOPB | termios.PARENB | termios.PARODD
                        | getattr(termios, "CMSPAR", 0))
            if applied & relevant != cflag & relevant:
                termios.tcsetattr(self.master, termios.TCSANOW, original)
                return False
            return True
        except (OSError, termios.error, ValueError):
            return False

    def close(self) -> None:
        os.close(self.master)
        if self._link:
            os.unlink(self._link)


class TcpSerial:
    supports_modem_control = False

    _DTE_BAUD_RATES = tuple(sorted(
        {300, 600, 1200, 1800, 2400, 4800, 7200, 9600, 14400, 19200, 28800,
         38400, 57600, 115200, 230400, 460800, 921600} | set(range(1170, 1213))))

    def __init__(self, host: str = "127.0.0.1", port: int = 9100, exclusive: bool = False,
                 baud_control_port: int | None = None):
        self.exclusive = exclusive
        self.supports_dte_baud = baud_control_port is not None
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._srv.setblocking(False)
        self._conn: socket.socket | None = None
        self._pending = bytearray()
        self.name = f"tcp://{host}:{port}"
        self._baud_srv = None
        self._baud_conn = None
        self._baud_input = bytearray()
        self._baud_pending = bytearray()
        self._dte_baud = 115200
        self._dte_format = (3, 1)
        if baud_control_port is not None:
            self._baud_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._baud_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self._baud_srv.bind((host, baud_control_port))
                self._baud_srv.listen(1)
                self._baud_srv.setblocking(False)
            except BaseException:
                self._baud_srv.close()
                self._srv.close()
                raise

    def fds(self):
        result = [self._srv]
        if self._conn:
            result.append(self._conn)
        if self._baud_srv:
            result.append(self._baud_srv)
        if self._baud_conn:
            result.append(self._baud_conn)
        return result

    def service(self, readable, writable=()):
        if self._conn is not None and self._conn in writable:
            self._flush()
        if self._baud_conn is not None and self._baud_conn in writable:
            self._flush_baud()
        dropped = False
        data = b""
        if self._srv in readable:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                conn = None
            if conn is not None:
                if self._conn is not None and self.exclusive:
                    conn.close()
                    conn = None
                if conn is None:
                    return b"", False
                if self._conn is not None:      # a newer client replaces the old one
                    self._close_conn()
                    dropped = True
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setblocking(False)
                self._conn = conn
        if self._baud_srv is not None and self._baud_srv in readable:
            try:
                ctrl, _ = self._baud_srv.accept()
            except OSError:
                ctrl = None
            if ctrl is not None:
                if self._conn is None:
                    ctrl.close()
                else:
                    self._close_baud_conn()
                    ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    ctrl.setblocking(False)
                    self._baud_conn = ctrl
        if self._conn is not None and self._conn in readable:
            try:
                data = self._conn.recv(4096)
                if not data:
                    raise ConnectionResetError
            except BlockingIOError:
                data = b""
            except OSError:
                data = b""
                self._close_conn()
                dropped = True
        if self._baud_conn is not None and self._baud_conn in readable:
            try:
                chunk = self._baud_conn.recv(128)
                if not chunk:
                    self._close_baud_conn()
                else:
                    self._baud_input.extend(chunk)
                    for kind, value, aux in take_metadata_frames(self._baud_input):
                        if kind == "baud" and value in self._DTE_BAUD_RATES:
                            self._dte_baud = value
                        elif kind == "format" and value in range(1, 7) and aux in range(4):
                            self._dte_format = (value, aux)
            except BlockingIOError:
                pass
            except OSError:
                self._close_baud_conn()
        return data, dropped

    def _close_conn(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass
        self._conn = None
        self._pending.clear()
        self._close_baud_conn()

    def _close_baud_conn(self) -> None:
        if self._baud_conn is not None:
            try:
                self._baud_conn.close()
            except OSError:
                pass
            self._baud_conn = None
        self._baud_input.clear()
        self._baud_pending.clear()

    def write_fds(self):
        result = [self._conn] if self._conn is not None and self._pending else []
        if self._baud_conn is not None and self._baud_pending:
            result.append(self._baud_conn)
        return result

    def _flush(self):
        try:
            n = self._conn.send(self._pending)
            del self._pending[:n]
        except (BlockingIOError, OSError):
            pass

    def _flush_baud(self):
        try:
            sent = self._baud_conn.send(self._baud_pending)
            del self._baud_pending[:sent]
        except (BlockingIOError, OSError):
            pass

    def supported_dte_baud_rates(self) -> tuple[int, ...]:
        return self._DTE_BAUD_RATES if self.supports_dte_baud else ()

    def get_dte_baud(self) -> int | None:
        return self._dte_baud if self.supports_dte_baud else None

    def set_dte_baud(self, baud: int) -> bool:
        if not self.supports_dte_baud or isinstance(baud, bool):
            return False
        try:
            rate = int(baud)
        except (TypeError, ValueError):
            return False
        if rate not in self._DTE_BAUD_RATES:
            return False
        self._dte_baud = rate
        if self._baud_conn is not None:
            self._baud_pending.extend(encode_baud(rate))
            self._flush_baud()
        return True

    def supported_dte_character_formats(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # Linux PTYs commonly coerce CS7 and parity modes. Until the control
        # channel negotiates peer capabilities, advertise only the tested 8N1/8N2.
        return ((1, 3), (0, 1, 2, 3)) if self.supports_dte_baud else ((3,), (0, 1, 2, 3))

    def get_dte_character_format(self) -> tuple[int, int] | None:
        return self._dte_format if self.supports_dte_baud else None

    def set_dte_character_format(self, format_code: int, parity_code: int) -> bool:
        formats, parities = self.supported_dte_character_formats()
        if format_code not in formats or parity_code not in parities:
            return False
        self._dte_format = (format_code, parity_code)
        if self._baud_conn is not None:
            self._baud_pending.extend(encode_character_format(format_code, parity_code))
            self._flush_baud()
        return True

    def write(self, data: bytes) -> None:
        if self._conn is None:
            return
        self._pending.extend(data)
        self._flush()

    def close(self) -> None:
        if self._conn is not None:
            self._close_conn()
        self._srv.close()
        if self._baud_srv is not None:
            self._baud_srv.close()
