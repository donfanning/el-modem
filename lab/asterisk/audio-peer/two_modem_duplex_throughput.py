#!/usr/bin/env python3
"""Measure full-duplex online-data-mode throughput: both modems
send a payload to each other *simultaneously* (not sequentially), over the
same live SIP/RTP/Asterisk call used by two_modem_e2e_driver.py and
two_modem_throughput.py. Each side times its own receive from first byte to
full payload, verifies it byte-exact, and reports its measured rate. This
covers the two gaps the single-direction throughput test left open:
reverse-direction (answer -> originate) measurement and simultaneous
(loaded, both directions at once) traffic -- a harsher test of the shared
20ms RTP send-thread cadence and jitter tolerance than either side sending
alone.
"""

import argparse
import os
import random
import re
import select
import sys
import threading
import time

def open_pty(path: str) -> int:
    return os.open(path, os.O_RDWR | os.O_NOCTTY)


def send(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise RuntimeError("PTY write made no progress")
        view = view[written:]


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
        "--bytes", type=int, default=300, help="payload size in bytes each side sends"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--expected-rate", type=int)
    parser.add_argument("--error-control", choices=["raw", "lapm", "mnp"])
    parser.add_argument(
        "--v23-rate", type=int, choices=[600, 1200],
        help="select V.23 at this forward-channel rate before dialing/auto-answer",
    )
    parser.add_argument("--connect-timeout", type=float, default=60)
    parser.add_argument("--transfer-timeout", type=float)
    args = parser.parse_args()

    fd = open_pty(args.pty)
    drain(fd, 0.3)  # discard any boot banner / stray bytes
    send(fd, b"ATE0\r")
    expect(fd, b"OK", 5)
    if args.v23_rate is not None:
        if args.error_control not in (None, "raw"):
            raise ValueError("V.23 uses its 75-baud return channel; LAPM/MNP is not supported")
        send(fd, f"AT+MS=V23,0,300,{args.v23_rate}\r".encode("ascii"))
        expect(fd, b"OK", 5)
    if args.error_control:
        code = {"raw": 0, "lapm": 3, "mnp": 2}[args.error_control]
        send(fd, f"AT\\N{code}\r".encode())
        expect(fd, b"OK", 5)

    if args.role == "answer":
        send(fd, b"ATS0=1\r")
        expect(fd, b"OK", 5)
        connect = expect(fd, b"CONNECT", args.connect_timeout)
    else:
        send(fd, f"ATDT{args.dial}\r".encode())
        connect = expect(fd, b"CONNECT", args.connect_timeout)

    match = re.search(rb"CONNECT (\d+)", connect)
    if match is None:
        raise RuntimeError(f"missing CONNECT speed: {connect!r}")
    rate = int(match.group(1))
    if args.expected_rate and rate != args.expected_rate:
        raise RuntimeError(f"unexpected CONNECT speed: {connect!r}")
    suffix = {"lapm": b"/LAPM", "mnp": b"/MNP"}.get(args.error_control)
    if suffix and suffix not in connect:
        raise RuntimeError(f"error control did not negotiate: {connect!r}")
    # LAPM/MNP carry synchronous octets, without async start/stop bits.
    # Their framing, acknowledgements and stuffing lower the actual rate
    # below this upper bound; raw 8N1 uses ten line bits per byte.
    bits_per_char = 8 if b"/LAPM" in connect or b"/MNP" in connect else 10
    chars_per_sec = rate / bits_per_char
    transfer_timeout = args.transfer_timeout or max(60.0, args.bytes / chars_per_sec * 4)
    print(f"{args.role}: {connect!r}; sending {args.bytes} bytes", flush=True)

    time.sleep(0.5)

    # Each role uses a distinct seed so the two payloads differ and can't be
    # confused with each other (useful since both directions share the PTY).
    own_seed = args.seed if args.role == "originate" else args.seed + 1
    peer_seed = args.seed + 1 if args.role == "originate" else args.seed
    own_rng = random.Random(own_seed)
    peer_rng = random.Random(peer_seed)
    own_payload = bytes(own_rng.randrange(256) for _ in range(args.bytes))
    expected_peer_payload = bytes(peer_rng.randrange(256) for _ in range(args.bytes))

    own_tag = b"\x02O\x03" if args.role == "originate" else b"\x02A\x03"
    peer_tag = b"\x02A\x03" if args.role == "originate" else b"\x02O\x03"
    terminator = b"\x02END\x03"

    receive_result: dict[str, object] = {}
    verified = threading.Event()

    def receiver() -> None:
        buf = bytearray()
        deadline = time.monotonic() + transfer_timeout
        first_byte_time = None
        while time.monotonic() < deadline:
            if peer_tag in buf:
                start = buf.index(peer_tag) + len(peer_tag)
                if len(buf) >= start + args.bytes + len(terminator):
                    break
            chunk = drain(fd, 0.1)
            if chunk and first_byte_time is None:
                first_byte_time = time.monotonic()
            buf.extend(chunk)
        if peer_tag not in buf:
            receive_result["error"] = f"never saw peer tag; got {len(buf)} bytes"
            verified.set()
            return
        start_index = buf.index(peer_tag) + len(peer_tag)
        end_index = start_index + args.bytes
        if bytes(buf[end_index:end_index + len(terminator)]) != terminator:
            receive_result["error"] = f"peer payload never terminated; got {len(buf)} bytes"
            verified.set()
            return
        last_byte_time = time.monotonic()
        received_payload = bytes(buf[start_index:end_index])
        if received_payload != expected_peer_payload:
            mismatches = sum(
                1 for a, b in zip(received_payload, expected_peer_payload) if a != b
            )
            receive_result["error"] = (
                f"payload mismatch: {len(received_payload)} of {len(expected_peer_payload)} "
                f"bytes, {mismatches} differ"
            )
            verified.set()
            return
        elapsed = (last_byte_time - first_byte_time) if first_byte_time else 0.0
        receive_result["elapsed"] = elapsed
        receive_result["bytes"] = len(received_payload)
        verified.set()
        # Keep reading on this same fd (avoiding a race against a separate
        # main-thread read) until the peer's post-verification rendezvous
        # marker also arrives, past whatever trailing bytes are already in
        # ``buf`` from this last chunk.
        rendezvous_deadline = time.monotonic() + transfer_timeout
        trailing = bytearray(buf[end_index + len(terminator):])
        while rendezvous_tag not in trailing and time.monotonic() < rendezvous_deadline:
            trailing.extend(drain(fd, 0.1))
        receive_result["rendezvous_seen"] = rendezvous_tag in trailing

    rendezvous_tag = b"\x02DONE\x03"
    recv_thread = threading.Thread(target=receiver, daemon=True)
    recv_thread.start()
    # The receiver signals verification before waiting for the peer's DONE.
    # Both main threads can therefore send DONE without waiting for each
    # other's receiver thread to finish.
    send(fd, own_tag + own_payload + terminator)
    if not verified.wait(transfer_timeout):
        raise RuntimeError(f"{args.role}: payload verification timed out")
    if "error" not in receive_result:
        send(fd, rendezvous_tag)
    recv_thread.join(timeout=transfer_timeout + 1)

    if "error" in receive_result:
        raise RuntimeError(f"{args.role}: {receive_result['error']}")
    if "elapsed" not in receive_result:
        raise RuntimeError(f"{args.role}: receiver thread did not complete")
    if not receive_result.get("rendezvous_seen"):
        raise RuntimeError(f"{args.role}: never saw peer's post-verification rendezvous marker")

    elapsed = receive_result["elapsed"]
    n = receive_result["bytes"]
    cps = n / elapsed if elapsed > 0 else float("inf")
    bps = cps * bits_per_char
    pct_of_nominal = (bps / rate) * 100
    print(
        f"{args.role}: full-duplex received {n} byte-exact payload in {elapsed:.3f}s "
        f"-> {cps:.2f} chars/sec, {bps:.1f} bits/sec "
        f"({pct_of_nominal:.1f}% of nominal {rate} bps / {chars_per_sec:.0f} cps) "
        f"while simultaneously sending {len(own_payload)} bytes",
        flush=True,
    )

    if args.role == "originate":
        time.sleep(1.2)  # the transmit-side guard interval before +++
        send(fd, b"+++")
        time.sleep(1.2)
        expect(fd, b"OK", 5)
        send(fd, b"ATH\r")
        expect(fd, b"OK", 10)
    else:
        expect(fd, b"NO CARRIER", 20)
    print(f"{args.role}: PASS", flush=True)


if __name__ == "__main__":
    main()
