#!/usr/bin/env python3
"""Measure real Bell 103 online-data-mode throughput across two modem_dce.py
PTYs dialing each other through Asterisk (serial -> audio -> Asterisk ->
audio -> serial, same path as two_modem_e2e_driver.py, but timed).

Run one instance with --role answer against the auto-answering modem's PTY
and one with --role originate against the dialing modem's PTY. The
originate side sends a payload of --bytes random bytes after CONNECT; the
answer side times from first received byte to the full payload arriving,
verifies it is byte-exact, and reports elapsed time, bytes/sec, bits/sec,
and the percentage of the raw 300 bps (30 cps, 8N1) Bell 103 line rate
achieved.
"""

import argparse
import os
import random
import select
import sys
import time

BAUD = 300
BITS_PER_CHAR = 10  # 1 start + 8 data + 1 stop (8N1, no parity)
CHARS_PER_SEC = BAUD / BITS_PER_CHAR  # 30 cps


def open_pty(path: str) -> int:
    return os.open(path, os.O_RDWR | os.O_NOCTTY)


def send(fd: int, data: bytes) -> None:
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
            buf.extend(os.read(fd, 65536))
    return bytes(buf)


def expect(fd: int, substr: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([fd], [], [], max(0, remaining))
        if fd in ready:
            buf.extend(os.read(fd, 65536))
            if substr in buf:
                return bytes(buf)
    raise TimeoutError(f"timed out waiting for {substr!r}; got {bytes(buf)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pty", help="path to the modem PTY (the --link symlink)")
    parser.add_argument("--role", choices=["originate", "answer"], required=True)
    parser.add_argument("--dial", default="110", help="extension the originator dials")
    parser.add_argument(
        "--bytes", type=int, default=300, help="payload size in bytes to transfer"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-error-control",
        action="store_true",
        help="send AT\\N0 before dialing/answering to disable the lab's "
        "simplified ARQ layer and measure the raw Bell 103 baseline",
    )
    args = parser.parse_args()

    fd = open_pty(args.pty)
    drain(fd, 0.3)  # discard any boot banner / stray bytes
    send(fd, b"ATE0\r")
    expect(fd, b"OK", 5)
    if args.no_error_control:
        send(fd, b"AT\\N0\r")
        expect(fd, b"OK", 5)

    if args.role == "answer":
        send(fd, b"ATS0=1\r")
        expect(fd, b"OK", 5)
        expect(fd, b"CONNECT", 30)
    else:
        send(fd, f"ATDT{args.dial}\r".encode())
        expect(fd, b"CONNECT", 30)

    time.sleep(0.5)

    rng = random.Random(args.seed)
    payload = bytes(rng.randrange(256) for _ in range(args.bytes))
    # Use a fixed sentinel so the receiver can tell real payload data from
    # whatever idle/no-op bytes it may have already buffered.
    sentinel = b"\x02START\x03"
    terminator = b"\x02END\x03"

    if args.role == "originate":
        start = time.monotonic()
        send(fd, sentinel + payload + terminator)
        # Wait for the far end's own payload + its ACK text so we know the
        # whole exchange (not just our send-side buffering) completed.
        expect(fd, b"ACK-DONE", max(30.0, args.bytes / CHARS_PER_SEC * 3))
        elapsed = time.monotonic() - start
        print(
            f"originate: sent {len(payload)} bytes, round trip incl. peer ack "
            f"confirmed in {elapsed:.2f}s",
            flush=True,
        )
        send(fd, b"+++")
        time.sleep(1.2)
        expect(fd, b"OK", 5)
        send(fd, b"ATH\r")
        expect(fd, b"OK", 10)
        print("originate: PASS", flush=True)
    else:
        buf = bytearray()
        deadline = time.monotonic() + max(60.0, args.bytes / CHARS_PER_SEC * 3)
        first_byte_time = None
        while time.monotonic() < deadline and terminator not in buf:
            chunk = drain(fd, 0.2)
            if chunk and first_byte_time is None:
                first_byte_time = time.monotonic()
            buf.extend(chunk)
        if terminator not in buf:
            raise RuntimeError(f"never received full payload; got {len(buf)} bytes")
        last_byte_time = time.monotonic()

        start_index = buf.index(sentinel) + len(sentinel)
        end_index = buf.index(terminator, start_index)
        received_payload = bytes(buf[start_index:end_index])
        if received_payload != payload:
            mismatches = sum(
                1 for a, b in zip(received_payload, payload) if a != b
            )
            raise RuntimeError(
                f"payload mismatch: {len(received_payload)} of {len(payload)} bytes, "
                f"{mismatches} byte(s) differ, "
                f"length delta {len(received_payload) - len(payload)}"
            )

        elapsed = last_byte_time - first_byte_time
        cps = len(payload) / elapsed if elapsed > 0 else float("inf")
        bps = cps * BITS_PER_CHAR
        pct_of_300baud = (bps / BAUD) * 100
        print(
            f"answer: received {len(payload)} byte-exact payload in {elapsed:.3f}s "
            f"-> {cps:.2f} chars/sec, {bps:.1f} bits/sec "
            f"({pct_of_300baud:.1f}% of nominal 300 bps / {CHARS_PER_SEC:.0f} cps)",
            flush=True,
        )
        send(fd, b"ACK-DONE\r\n")
        expect(fd, b"NO CARRIER", 20)
        print("answer: PASS", flush=True)


if __name__ == "__main__":
    main()
