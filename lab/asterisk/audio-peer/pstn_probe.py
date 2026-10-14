#!/usr/bin/env python3
"""Place one test call through a lab modem PTY and ALWAYS hang up properly.

Hangs up the way a terminal program would, so the modem (and the PSTN call)
is never left connected: guard time of silence, ``+++`` (no CR), guard time,
wait for ``OK``, then ``ATH0``, wait for ``OK``. This runs in a ``finally``
block, so errors and Ctrl-C still hang up. Never pipe this script through
``head``/``tail -n`` (a closed pipe kills the script before it can hang up);
write its output to a file or let it run to completion.

    python3 pstn_probe.py /tmp/modem_probe 13034944774 --secs 30
"""

import argparse
import os
import select
import sys
import time
import tty

GUARD = 1.3  # seconds; > S12 (the default 1 s escape guard time)


def say(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


class Line:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)

    def read(self, seconds: float) -> bytes:
        out, end = b"", time.time() + seconds
        while time.time() < end:
            if select.select([self.fd], [], [], 0.2)[0]:
                try:
                    out += os.read(self.fd, 4096)
                except OSError:
                    break
        return out

    def wait_for(self, tokens, seconds: float) -> bytes:
        out, end = b"", time.time() + seconds
        while time.time() < end:
            out += self.read(0.3)
            if any(t in out for t in tokens):
                break
        return out

    def send(self, data: bytes) -> None:
        os.write(self.fd, data)


def hang_up(line: Line, online: bool) -> bool:
    """Standard Hayes: guard, +++, guard, OK, ATH0, OK/NO CARRIER.

    When the call never connected the modem is already in command mode, so
    the escape sequence is skipped and only ATH0 is sent."""
    if not online:
        line.send(b"ATH0\r")
        reply = line.wait_for((b"OK", b"NO CARRIER"), 6)
        return b"OK" in reply or b"NO CARRIER" in reply
    for attempt in (1, 2):
        line.read(GUARD)  # silence before +++ (also drains pending data)
        line.send(b"+++")
        reply = line.wait_for((b"OK", b"NO CARRIER"), GUARD + 3)
        if b"OK" in reply or b"NO CARRIER" in reply:
            line.send(b"ATH0\r")
            reply = line.wait_for((b"OK", b"NO CARRIER"), 6)
            if b"OK" in reply or b"NO CARRIER" in reply:
                return True
        say(f"hangup attempt {attempt} not confirmed: {reply!r}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pty")
    ap.add_argument("number")
    ap.add_argument("--secs", type=float, default=30.0, help="seconds to listen after CONNECT")
    ap.add_argument("--init", default="ATE0X3", help="init string sent before dialing")
    ap.add_argument("--dial-timeout", type=float, default=60.0)
    ap.add_argument("--raw-out", default=None, help="write the raw bytes received after CONNECT to this file")
    args = ap.parse_args()

    line = Line(args.pty)
    connected = False
    try:
        line.read(0.5)
        line.send(args.init.encode() + b"\r")
        say(f"init: {line.wait_for((b'OK', b'ERROR'), 3)!r}")
        line.send(f"ATDT{args.number}\r".encode())
        t0 = time.time()
        resp = line.wait_for((b"CONNECT", b"NO CARRIER", b"BUSY", b"NO ANSWER"), args.dial_timeout)
        say(f"[{time.time() - t0:5.1f}s] {resp!r}")
        connected = b"CONNECT" in resp
        if connected:
            end = time.time() + args.secs
            got = b""
            while time.time() < end:
                chunk = line.read(1.0)
                if chunk:
                    got += chunk
                    say(f"[{time.time() - t0:5.1f}s] {chunk!r}")
                    if b"NO CARRIER" in chunk:
                        # The far end dropped: the modem is back in command
                        # mode, so hang up with ATH0 only (no escape).
                        connected = False
                        break
            say(f"TOTAL BYTES after CONNECT: {len(got)}")
            if args.raw_out:
                os.makedirs(os.path.dirname(args.raw_out) or ".", exist_ok=True)
                with open(args.raw_out, "wb") as f:
                    f.write(got)
    finally:
        ok = hang_up(line, online=connected)
        say(f"hangup confirmed: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
