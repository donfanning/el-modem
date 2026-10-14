#!/usr/bin/env python3
"""AudioSocket PCM loopback fixture for the containerized Asterisk lab."""

import logging
import socket
import struct
import uuid
import wave
from threading import Thread

from bell103 import Bell103Receiver, Bell103Transmitter

HOST = "0.0.0.0"
PORT = 9019
TYPE_UUID = 0x01
TYPE_DTMF = 0x03
TYPE_AUDIO = 0x10
TYPE_HANGUP = 0x00
MAX_PACKET = 65535
BELL103_UUID = uuid.UUID("00000000-0000-0000-0000-000000000104").bytes
BELL103_REPLY = b"ANSWER abcdefghij\r"
BELL103_RECEIVED = "/tmp/bell103-received.bin"
BELL103_INPUT_WAV = "/tmp/bell103-caller.wav"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def recv_exact(conn: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        data = conn.recv(size - len(chunks))
        if not data:
            return None
        chunks.extend(data)
    return bytes(chunks)


def recv_packet(conn: socket.socket) -> tuple[int, bytes] | None:
    header = recv_exact(conn, 3)
    if header is None:
        return None
    packet_type, length = struct.unpack("!BH", header)
    if length > MAX_PACKET:
        raise ValueError(f"AudioSocket packet too large: {length}")
    payload = recv_exact(conn, length)
    if payload is None:
        raise ConnectionError("truncated AudioSocket packet")
    return packet_type, payload


def send_packet(conn: socket.socket, packet_type: int, payload: bytes) -> None:
    conn.sendall(struct.pack("!BH", packet_type, len(payload)) + payload)


def serve_client(conn: socket.socket, address: tuple[str, int]) -> None:
    logging.info("AudioSocket connected from %s:%d", *address)
    try:
        first = recv_packet(conn)
        if first is None or first[0] != TYPE_UUID or len(first[1]) != 16:
            logging.warning("connection did not start with a 16-byte UUID")
            return
        logging.info("AudioSocket UUID %s", first[1].hex())
        bell103 = first[1] == BELL103_UUID
        if bell103:
            receiver = Bell103Receiver("originate")
            transmitter = Bell103Transmitter("answer", BELL103_REPLY)
            received = bytearray()
            caller_audio = bytearray()
            try:
                import os

                os.unlink(BELL103_RECEIVED)
                os.unlink(BELL103_INPUT_WAV)
            except FileNotFoundError:
                pass
            logging.info("Bell 103 answer endpoint ready")

        while True:
            packet = recv_packet(conn)
            if packet is None:
                break
            packet_type, payload = packet
            if packet_type == TYPE_HANGUP:
                break
            if packet_type == TYPE_AUDIO:
                if len(payload) % 2:
                    logging.warning("discarding odd-sized linear PCM frame (%d bytes)", len(payload))
                    continue
                if bell103:
                    received.extend(receiver.feed_pcm(payload))
                    caller_audio.extend(payload)
                    samples = transmitter.samples(len(payload) // 2)
                    response = struct.pack(f"<{len(samples)}h", *samples)
                    send_packet(conn, TYPE_AUDIO, response)
                else:
                    send_packet(conn, TYPE_AUDIO, payload)
            elif packet_type == TYPE_DTMF:
                logging.info("DTMF event payload=%s", payload.hex())
            else:
                logging.warning("ignoring AudioSocket packet type=0x%02x length=%d", packet_type, len(payload))
    except (ConnectionError, OSError, ValueError) as exc:
        logging.info("AudioSocket ended: %s", exc)
    finally:
        if first is not None and first[0] == TYPE_UUID and first[1] == BELL103_UUID:
            with open(BELL103_RECEIVED, "wb") as result:
                result.write(received)
            with wave.open(BELL103_INPUT_WAV, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(caller_audio)
            logging.info("Bell 103 answer endpoint decoded %d bytes: %s", len(received), received.hex())
        conn.close()
        logging.info("AudioSocket disconnected from %s:%d", *address)


def main() -> None:
    with socket.create_server((HOST, PORT), reuse_port=True) as server:
        logging.info("AudioSocket PCM loopback listening on %s:%d", HOST, PORT)
        while True:
            conn, address = server.accept()
            Thread(target=serve_client, args=(conn, address), daemon=True).start()


if __name__ == "__main__":
    main()
