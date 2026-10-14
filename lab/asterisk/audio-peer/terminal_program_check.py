#!/usr/bin/env python3
"""Exercise the modem PTY the way a real serial terminal program (Telix,
ProComm, minicom, etc.) would: open it as a termios serial device at a host
baud rate that is deliberately different from the negotiated Bell 103 line
rate, send AT commands, dial/answer, exchange data in CONNECT mode, and hang
up with the guard-time escape. Uses pyserial, which is what most portable
Python serial tools use and which enforces real termios baud-rate framing
on the host side of the PTY, rather than talking to the raw fd directly."""

import argparse
import sys
import time

import serial


def expect(ser: serial.Serial, substr: bytes, timeout: float = 10.0) -> bytes:
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf.extend(chunk)
            if substr in buf:
                return bytes(buf)
        else:
            time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {substr!r}; got {bytes(buf)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pty")
    parser.add_argument("--role", choices=["originate", "answer"], required=True)
    parser.add_argument("--dial", default="110")
    # Deliberately NOT 300 -- proves the host serial baud rate is
    # independent of the negotiated Bell 103 line rate.
    parser.add_argument("--host-baud", type=int, default=115200)
    args = parser.parse_args()

    ser = serial.Serial(args.pty, baudrate=args.host_baud, timeout=0.1)
    try:
        ser.write(b"ATE0\r")
        expect(ser, b"OK")

        if args.role == "originate":
            ser.write(f"ATD{args.dial}\r".encode())
            expect(ser, b"CONNECT", timeout=20.0)
            ser.write(b"HELLO FROM TERMINAL PROGRAM TEST\r")
            expect(ser, b"ACK", timeout=20.0)
            ser.write(b"+++")
            time.sleep(1.2)
            expect(ser, b"OK", timeout=5.0)
            ser.write(b"ATH\r")
            expect(ser, b"OK", timeout=5.0)
            print("originate: PASS")
        else:
            ser.write(b"ATS0=1\r")
            expect(ser, b"OK")
            expect(ser, b"CONNECT", timeout=30.0)
            expect(ser, b"HELLO FROM TERMINAL PROGRAM TEST\r", timeout=20.0)
            ser.write(b"ACK\r")
            expect(ser, b"NO CARRIER", timeout=20.0)
            print("answer: PASS")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
