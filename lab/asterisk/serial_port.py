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
import time


class PtySerial:
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

    def close(self) -> None:
        os.close(self.master)
        if self._link:
            os.unlink(self._link)


class TcpSerial:
    def __init__(self, host: str = "127.0.0.1", port: int = 9100, exclusive: bool = False):
        self.exclusive = exclusive
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._srv.setblocking(False)
        self._conn: socket.socket | None = None
        self._pending = bytearray()
        self.name = f"tcp://{host}:{port}"

    def fds(self):
        return [self._srv] + ([self._conn] if self._conn else [])

    def service(self, readable, writable=()):
        if self._conn is not None and self._conn in writable:
            self._flush()
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
        return data, dropped

    def _close_conn(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass
        self._conn = None
        self._pending.clear()

    def write_fds(self):
        return [self._conn] if self._conn is not None and self._pending else []

    def _flush(self):
        try:
            n = self._conn.send(self._pending)
            del self._pending[:n]
        except (BlockingIOError, OSError):
            pass

    def write(self, data: bytes) -> None:
        if self._conn is None:
            return
        self._pending.extend(data)
        self._flush()

    def close(self) -> None:
        if self._conn is not None:
            self._close_conn()
        self._srv.close()
