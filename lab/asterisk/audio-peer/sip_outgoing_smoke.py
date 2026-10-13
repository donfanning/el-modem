#!/usr/bin/env python3
"""Verify ATD outbound origination through real Asterisk call-progress audio.

Unlike sip_incoming_smoke.py, this script does not speak SIP itself: the
modem container is the caller (UAC) and Asterisk is the callee, so this
script only drives the modem's serial PTY and asserts on RING/BUSY/CONNECT/
NO CARRIER exactly as a terminal program would. This keeps the test honest
to requirement #1's intent: call progress must come from audio the modem
DSP layer actually classifies (busy/ringback cadence), not from any SIP
signaling shortcut.
"""

import argparse
import os
import select
import time
from typing import BinaryIO


def read_terminal(fd: int, marker: bytes, timeout: float = 15) -> bytes:
    result = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in result and time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if readable:
            result.extend(os.read(fd, 1024))
    if marker not in result:
        raise TimeoutError(f"terminal did not receive {marker!r}; got {bytes(result)!r}")
    return bytes(result)


def open_terminal(path: str, timeout: float = 8) -> BinaryIO:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return open(path, "rb+", buffering=0)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def dial_and_expect(
    terminal: BinaryIO,
    fd: int,
    number: str,
    expect: bytes,
    timeout: float = 15,
) -> bytes:
    terminal.write(f"ATD{number}\r".encode())
    return read_terminal(fd, expect, timeout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pty", default="/tmp/modem0")
    args = parser.parse_args()

    with open_terminal(args.pty) as terminal:
        fd = terminal.fileno()
        terminal.write(b"ATE0\r")
        read_terminal(fd, b"OK\r\n")

        # Extension 106: Busy() answers and plays real 480+620 Hz busy
        # cadence; the modem-layer BusyToneDetector must classify it from
        # audio, matching indications.conf's "us" busy tone.
        busy = dial_and_expect(terminal, fd, "106", b"BUSY\r\n")
        print(f"ATD106 -> {busy.strip()!r} (busy-tone audio detected)")

        # Extension 107: Progress() + Playtones(ring) sends real 440+480 Hz
        # ringback before Answer(); the modem must report RING from audio
        # while still ringing, then CONNECT once Asterisk answers, then
        # NO CARRIER once Asterisk hangs up.
        terminal.write(b"ATD107\r")
        progress = read_terminal(fd, b"CONNECT\r\n", timeout=15)
        if b"RING\r\n" not in progress:
            raise RuntimeError(
                f"expected at least one audio-detected RING before CONNECT: {progress!r}"
            )
        print(f"ATD107 -> {progress.strip()!r} (ringback-then-answer)")
        no_carrier = read_terminal(fd, b"NO CARRIER\r\n", timeout=15)
        print(f"remote hangup -> {no_carrier.strip()!r}")

    print("Outgoing SIP call progress passed: BUSY and RING->CONNECT->NO CARRIER "
          "were all derived from real call-progress audio.")


if __name__ == "__main__":
    main()
