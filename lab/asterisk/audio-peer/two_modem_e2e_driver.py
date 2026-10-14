#!/usr/bin/env python3
"""Drive one side of a real two-modem end-to-end test over a modem_dce.py PTY.

Two independent modem_dce.py processes (each with its own SIP/RTP media leg
into Asterisk) call each other, exchange complete payloads through modem
audio over SIP/RTP/Asterisk, and hang up cleanly. Run one instance with
--role originate against the dialing modem's PTY and one with --role answer
against the auto-answering modem's PTY (see the lab README).
"""

import argparse
import os
import select
import sys
import time


def open_pty(path: str) -> int:
    return os.open(path, os.O_RDWR | os.O_NOCTTY)


def send(fd: int, data: str | bytes) -> None:
    if isinstance(data, str):
        data = data.encode("ascii")
    os.write(fd, data)


def drain(fd: int, duration: float) -> bytes:
    deadline = time.monotonic() + duration
    buf = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if fd in ready:
            buf.extend(os.read(fd, 4096))
    return bytes(buf)


def expect(fd: int, substr: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([fd], [], [], max(0, remaining))
        if fd in ready:
            buf.extend(os.read(fd, 4096))
            if substr in buf:
                return bytes(buf)
    raise TimeoutError(f"timed out waiting for {substr!r}; got {bytes(buf)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pty", help="path to the modem PTY (the --link symlink)")
    parser.add_argument("--role", choices=["originate", "answer"], required=True)
    parser.add_argument("--dial", default="110", help="extension the originator dials")
    parser.add_argument("--expected-rate", type=int, help="require this CONNECT speed")
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--init", action="append", default=[], help="extra AT command(s) sent before dialling")
    parser.add_argument("--error-control", choices=["raw", "lapm", "mnp"])
    args = parser.parse_args()

    fd = open_pty(args.pty)
    drain(fd, 0.3)  # discard any boot banner / stray bytes
    send(fd, "ATE0\r")
    expect(fd, b"OK", 5)
    for cmd in args.init:
        send(fd, cmd + "\r")
        expect(fd, b"OK", 5)
    if args.error_control:
        code = {"raw": 0, "lapm": 3, "mnp": 2}[args.error_control]
        send(fd, f"AT\\N{code}\r")
        expect(fd, b"OK", 5)

    my_payload = f"HELLO FROM {args.role.upper()} 0123456789\r\n".encode()
    peer_role = "ANSWER" if args.role == "originate" else "ORIGINATE"
    their_payload = f"HELLO FROM {peer_role} 0123456789\r\n".encode()

    if args.role == "answer":
        send(fd, "ATS0=1\r")
        expect(fd, b"OK", 5)
        connect = expect(fd, b"CONNECT", args.connect_timeout)
    else:
        send(fd, f"ATDT{args.dial}\r".encode())
        connect = expect(fd, b"CONNECT", args.connect_timeout)

    if args.expected_rate and f"CONNECT {args.expected_rate}".encode() not in connect:
        raise RuntimeError(f"unexpected CONNECT result: {connect!r}")
    protocol_suffix = {"lapm": b"/LAPM", "mnp": b"/MNP"}.get(args.error_control)
    if protocol_suffix and protocol_suffix not in connect:
        raise RuntimeError(f"error control did not negotiate: {connect!r}")
    print(f"{args.role}: connected: {connect!r}", flush=True)

    time.sleep(0.5)
    send(fd, my_payload)

    deadline = time.monotonic() + 15
    received = bytearray()
    while time.monotonic() < deadline and their_payload not in received:
        received.extend(drain(fd, 0.5))
    if their_payload not in received:
        raise RuntimeError(f"never received peer payload; got {bytes(received)!r}")
    print(f"{args.role}: received peer payload: {bytes(received)!r}", flush=True)

    if args.role == "originate":
        time.sleep(1.5)  # let the answer side finish draining our bytes
        send(fd, "+++")
        time.sleep(1.2)
        expect(fd, b"OK", 5)
        send(fd, "ATH\r")
        expect(fd, b"OK", 10)
        print("originate: hung up (ATH OK)", flush=True)
    else:
        expect(fd, b"NO CARRIER", 20)
        print("answer: remote hangup produced NO CARRIER", flush=True)

    print(f"{args.role}: PASS", flush=True)


if __name__ == "__main__":
    main()
