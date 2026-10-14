#!/usr/bin/env python3
"""Verify manual ATA and ATS0 answer timing through Asterisk."""

import argparse
import os
import select
import socket
import time
import tty
import uuid
from typing import BinaryIO


def parse_sip(data: bytes) -> tuple[str, dict[str, str]]:
    text = data.decode("utf-8", "replace")
    lines = text.split("\r\n")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(":")
        if separator:
            headers[key.lower()] = value.strip()
    return lines[0], headers


def read_terminal(fd: int, marker: bytes, timeout: float = 8) -> bytes:
    result = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in result and time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if readable:
            result.extend(os.read(fd, 1024))
    if marker not in result:
        raise TimeoutError(f"terminal did not receive {marker!r}; got {bytes(result)!r}")
    return bytes(result)


def read_terminal_count(fd: int, marker: bytes, count: int, timeout: float = 12) -> bytes:
    result = bytearray()
    deadline = time.monotonic() + timeout
    while result.count(marker) < count and time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if readable:
            result.extend(os.read(fd, 1024))
    if result.count(marker) < count:
        raise TimeoutError(
            f"terminal received {result.count(marker)} of {count} {marker!r} events: "
            f"{bytes(result)!r}"
        )
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


def read_sip(
    sock: socket.socket,
    call_id: str,
    cseq: str,
    timeout: float = 8,
) -> tuple[bytes, str, dict[str, str]]:
    sock.settimeout(timeout)
    while True:
        packet, _ = sock.recvfrom(65535)
        start, headers = parse_sip(packet)
        if headers.get("call-id") != call_id or headers.get("cseq", "").upper() != cseq:
            continue
        if start.startswith("SIP/2.0 100"):
            continue
        return packet, start, headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asterisk", default="asterisk")
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--extension", default="105")
    parser.add_argument("--pty", default="/tmp/modem0")
    parser.add_argument("--auto-answer-rings", type=int, default=0)
    args = parser.parse_args()

    sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip.bind(("0.0.0.0", 15060))
    sip.connect((args.asterisk, args.port))
    local_ip, local_port = sip.getsockname()
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("0.0.0.0", 17000))
    with open_terminal(args.pty) as terminal:
        fd = terminal.fileno()
        tty.setraw(fd)
        terminal.write(b"ATE0\r")
        read_terminal(fd, b"OK\r\n")
        if args.auto_answer_rings:
            terminal.write(f"ATS0={args.auto_answer_rings}\r".encode())
            read_terminal(fd, b"OK\r\n")

        call_id = f"{uuid.uuid4()}@modem-smoke"
        from_tag = uuid.uuid4().hex[:12]
        branch = "z9hG4bK" + uuid.uuid4().hex
        sdp = (
            "v=0\r\n"
            f"o=modem-smoke 1 1 IN IP4 {local_ip}\r\n"
            "s=IncomingModemCall\r\n"
            f"c=IN IP4 {local_ip}\r\n"
            "t=0 0\r\n"
            "m=audio 17000 RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
            "a=sendrecv\r\n"
        )
        invite = (
            f"INVITE sip:{args.extension}@{args.asterisk} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{local_port};branch={branch};rport\r\n"
            f"From: <sip:modem-smoke@{args.asterisk}>;tag={from_tag}\r\n"
            f"To: <sip:{args.extension}@{args.asterisk}>\r\n"
            f"Call-ID: {call_id}\r\n"
            "CSeq: 1 INVITE\r\n"
            f"Contact: <sip:modem-smoke@{local_ip}:{local_port}>\r\n"
            "Max-Forwards: 70\r\n"
            "Content-Type: application/sdp\r\n"
            f"Content-Length: {len(sdp.encode())}\r\n\r\n{sdp}"
        )
        sip.send(invite.encode())

        ringing_seen = False
        final_headers: dict[str, str] | None = None
        while not ringing_seen:
            _, start, headers = read_sip(sip, call_id, "1 INVITE")
            if start.startswith("SIP/2.0 180"):
                ringing_seen = True
            elif start.startswith("SIP/2.0 200"):
                final_headers = headers
                break
            elif start.startswith("SIP/2.0 4"):
                raise RuntimeError(f"Asterisk rejected modem call: {start}")
        if not ringing_seen:
            raise RuntimeError(f"call answered before the modem rang: {final_headers}")

        if args.auto_answer_rings:
            ring = read_terminal_count(fd, b"RING\r\n", args.auto_answer_rings)
            connect_offset = ring.find(b"CONNECT\r\n")
            last_ring_offset = ring.rfind(b"RING\r\n")
            if connect_offset >= 0 and connect_offset < last_ring_offset:
                raise RuntimeError("modem auto-answered before configured ring count")
            connect = (
                ring
                if connect_offset >= 0
                else read_terminal(fd, b"CONNECT\r\n")
            )
        else:
            ring = read_terminal(fd, b"RING\r\n")
            if b"CONNECT" in ring:
                raise RuntimeError("modem connected before ATA")
            terminal.write(b"ATA\r")
            connect = read_terminal(fd, b"CONNECT\r\n")

        _, start, final_headers = read_sip(sip, call_id, "1 INVITE")
        if not start.startswith("SIP/2.0 200"):
            raise RuntimeError(f"ATA did not answer SIP call: {start}")
        to_header = final_headers["to"]
        ack_branch = "z9hG4bK" + uuid.uuid4().hex
        ack = (
            f"ACK sip:{args.extension}@{args.asterisk} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{local_port};branch={ack_branch};rport\r\n"
            f"From: <sip:modem-smoke@{args.asterisk}>;tag={from_tag}\r\n"
            f"To: {to_header}\r\n"
            f"Call-ID: {call_id}\r\n"
            "CSeq: 1 ACK\r\n"
            f"Contact: <sip:modem-smoke@{local_ip}:{local_port}>\r\n"
            "Max-Forwards: 70\r\nContent-Length: 0\r\n\r\n"
        )
        sip.send(ack.encode())

        bye_branch = "z9hG4bK" + uuid.uuid4().hex
        bye = (
            f"BYE sip:{args.extension}@{args.asterisk} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{local_port};branch={bye_branch};rport\r\n"
            f"From: <sip:modem-smoke@{args.asterisk}>;tag={from_tag}\r\n"
            f"To: {to_header}\r\n"
            f"Call-ID: {call_id}\r\n"
            "CSeq: 2 BYE\r\nContent-Length: 0\r\n\r\n"
        )
        sip.send(bye.encode())
        _, bye_status, _ = read_sip(sip, call_id, "2 BYE")
        if not bye_status.startswith("SIP/2.0 200"):
            raise RuntimeError(f"BYE failed: {bye_status}")
        no_carrier = read_terminal(fd, b"NO CARRIER\r\n")

    rtp.close()
    sip.close()
    print(
        f"Incoming SIP call passed: {'S0 auto-answer' if args.auto_answer_rings else 'delayed ATA'} "
        "produced CONNECT; remote hangup "
        f"produced NO CARRIER ({connect.strip()!r}, {no_carrier.strip()!r})"
    )


if __name__ == "__main__":
    main()
