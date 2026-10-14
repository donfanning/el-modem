#!/usr/bin/env python3
"""Verify the AudioSocket fixture returns one exact signed-linear PCM frame."""

import argparse
import socket
import struct
import sys
import uuid

TYPE_UUID = 0x01
TYPE_AUDIO = 0x10


def recv_exact(conn: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        data = conn.recv(size - len(result))
        if not data:
            raise RuntimeError("AudioSocket closed before returning the frame")
        result.extend(data)
    return bytes(result)


def packet(conn: socket.socket, kind: int, payload: bytes) -> None:
    conn.sendall(struct.pack("!BH", kind, len(payload)) + payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    args = parser.parse_args()

    pcm = b"".join(struct.pack("<h", ((i * 193) % 60001) - 30000) for i in range(160))
    with socket.create_connection((args.host, args.port), timeout=5) as conn:
        conn.settimeout(5)
        packet(conn, TYPE_UUID, uuid.uuid4().bytes)
        packet(conn, TYPE_AUDIO, pcm)
        kind, size = struct.unpack("!BH", recv_exact(conn, 3))
        echoed = recv_exact(conn, size)

    if kind != TYPE_AUDIO:
        print(f"expected audio packet 0x10; got 0x{kind:02x}", file=sys.stderr)
        return 1
    if echoed != pcm:
        print("PCM frame differs from the transmitted payload", file=sys.stderr)
        return 1
    print(f"AudioSocket loopback passed: {len(pcm)} PCM bytes returned byte-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
