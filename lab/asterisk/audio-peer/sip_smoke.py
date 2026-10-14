#!/usr/bin/env python3
"""Call Asterisk extension 103 and verify RTP reaches the AudioSocket loopback."""

import argparse
import math
import re
import select
import socket
import struct
import time
import uuid
import wave

from bell103 import Bell103Receiver, Bell103Transmitter

SIP_PORT = 5060
RTP_PORT = 14000
CALL_ID = f"{uuid.uuid4()}@audio-peer"
FROM_TAG = uuid.uuid4().hex[:12]
CSEQ = 1


def linear_to_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), 32635) + 0x84
    exponent = 0
    for threshold in (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF):
        if magnitude <= threshold:
            break
        exponent += 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def ulaw_to_linear(sample: int) -> int:
    value = (~sample) & 0xFF
    magnitude = ((value & 0x0F) << 3) + 0x84
    magnitude <<= (value & 0x70) >> 4
    return 0x84 - magnitude if value & 0x80 else magnitude - 0x84


def read_wav_ulaw(path: str) -> bytes:
    with wave.open(path, "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 8000
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("input WAV must be mono, 16-bit PCM, 8000 Hz")
        pcm = audio.readframes(audio.getnframes())
    return bytes(
        linear_to_ulaw(struct.unpack_from("<h", pcm, offset)[0])
        for offset in range(0, len(pcm), 2)
    )


def write_wav_ulaw(path: str, payload: bytes) -> None:
    pcm = b"".join(struct.pack("<h", ulaw_to_linear(sample)) for sample in payload)
    with wave.open(path, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(pcm)


def recv_sip(sock: socket.socket, timeout: float = 8) -> tuple[str, tuple[str, int]]:
    sock.settimeout(timeout)
    data, addr = sock.recvfrom(65535)
    return data.decode("utf-8", "replace"), addr


def headers(message: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in message.split("\r\n")[1:]:
        if not line:
            break
        key, sep, value = line.partition(":")
        if sep:
            result[key.lower()] = value.strip()
    return result


def response_code(message: str) -> int:
    match = re.match(r"SIP/2.0\s+(\d+)", message)
    if not match:
        raise RuntimeError(f"not a SIP response: {message[:120]!r}")
    return int(match.group(1))


def send_request(sock: socket.socket, request: str, target: tuple[str, int]) -> None:
    sock.sendto(request.encode(), target)


def drain_rtp(rtp: socket.socket, remote_rtp: tuple[str, int]) -> tuple[int, bytes]:
    received = 0
    audio = bytearray()
    while True:
        try:
            packet, source = rtp.recvfrom(2048)
        except BlockingIOError:
            break
        if (
            source == remote_rtp
            and len(packet) >= 12
            and packet[0] >> 6 == 2
            and packet[1] & 0x7F == 0
        ):
            received += 1
            audio.extend(packet[12:])
    return received, bytes(audio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asterisk", nargs="?", default="asterisk")
    parser.add_argument("--wav-in", help="send an 8 kHz mono PCM WAV file")
    parser.add_argument("--wav-out", help="save returned PCMU audio as a PCM WAV file")
    parser.add_argument("--bell103", action="store_true", help="run a Bell 103 caller/answerer pair")
    args = parser.parse_args()

    if args.bell103 and args.wav_in:
        parser.error("--bell103 cannot be combined with --wav-in")
    target_extension = "104" if args.bell103 else "103"
    tx_audio = read_wav_ulaw(args.wav_in) if args.wav_in else b""
    bell103_tx = (
        Bell103Transmitter("originate", b"ORIGINATE 0123456789\r")
        if args.bell103
        else None
    )
    bell103_rx = Bell103Receiver("answer") if args.bell103 else None
    bell103_reply = b"ANSWER abcdefghij\r"
    sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip.bind(("0.0.0.0", SIP_PORT))
    sip.connect((args.asterisk, SIP_PORT))
    local_ip, _ = sip.getsockname()

    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("0.0.0.0", RTP_PORT))
    rtp.settimeout(0.1)
    call_server = (args.asterisk, SIP_PORT)

    branch = "z9hG4bK" + uuid.uuid4().hex
    sdp = (
        "v=0\r\n"
        f"o=audio-peer 1 1 IN IP4 {local_ip}\r\n"
        "s=Bell103LabAudioPath\r\n"
        f"c=IN IP4 {local_ip}\r\n"
        "t=0 0\r\n"
        f"m=audio {RTP_PORT} RTP/AVP 0\r\n"
        "a=rtpmap:0 PCMU/8000\r\n"
        "a=ptime:20\r\n"
        "a=sendrecv\r\n"
    )
    invite = (
        f"INVITE sip:{target_extension}@{args.asterisk} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:audio-peer@{args.asterisk}>;tag={FROM_TAG}\r\n"
        f"To: <sip:{target_extension}@{args.asterisk}>\r\n"
        f"Call-ID: {CALL_ID}\r\n"
        f"CSeq: {CSEQ} INVITE\r\n"
        f"Contact: <sip:audio-peer@{local_ip}:{SIP_PORT}>\r\n"
        "Max-Forwards: 70\r\n"
        "Content-Type: application/sdp\r\n"
        f"Content-Length: {len(sdp.encode())}\r\n"
        "\r\n"
        f"{sdp}"
    )
    send_request(sip, invite, call_server)

    final_response = ""
    while True:
        response, source = recv_sip(sip)
        code = response_code(response)
        if code == 401 or code == 403:
            raise RuntimeError(f"Asterisk rejected internal SIP test endpoint: {response.splitlines()[0]}")
        if code >= 200:
            if code >= 300:
                raise RuntimeError(f"INVITE failed: {response.splitlines()[0]}")
            final_response = response
            break
    response_headers = headers(final_response)
    to_header = response_headers["to"]
    contact_match = re.search(r"c=IN IP4 ([^\r\n]+).*?m=audio (\d+)", final_response, re.S)
    if not contact_match:
        raise RuntimeError("Asterisk 200 response did not contain an audio SDP address")
    remote_rtp = (contact_match.group(1).strip(), int(contact_match.group(2)))

    ack_branch = "z9hG4bK" + uuid.uuid4().hex
    ack = (
        f"ACK sip:{target_extension}@{args.asterisk} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{SIP_PORT};branch={ack_branch};rport\r\n"
        f"From: <sip:audio-peer@{args.asterisk}>;tag={FROM_TAG}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {CALL_ID}\r\n"
        f"CSeq: {CSEQ} ACK\r\n"
        f"Contact: <sip:audio-peer@{local_ip}:{SIP_PORT}>\r\n"
        "Max-Forwards: 70\r\n"
        "Content-Length: 0\r\n\r\n"
    )
    send_request(sip, ack, call_server)

    sequence = 0
    timestamp = 0
    lead_frames = 50 if tx_audio or bell103_tx else 0
    frame_count = max(
        200 if bell103_tx else 150,
        lead_frames + math.ceil(len(tx_audio) / 160),
    )
    received = 0
    returned_audio = bytearray()
    next_frame = time.monotonic()
    rtp.setblocking(False)
    for frame in range(frame_count):
        delay = next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        audio_frame = frame - lead_frames
        start = audio_frame * 160
        if bell103_tx and audio_frame >= 0:
            payload = bytes(linear_to_ulaw(sample) for sample in bell103_tx.samples(160))
        else:
            payload = (
                tx_audio[start : start + 160].ljust(160, b"\xff")
                if audio_frame >= 0
                else b"\xff" * 160
            )
        rtp.sendto(struct.pack("!BBHII", 0x80, 0, sequence, timestamp, 0x103) + payload, remote_rtp)
        sequence = (sequence + 1) & 0xFFFF
        timestamp = (timestamp + 160) & 0xFFFFFFFF
        next_frame += 0.02
        count, audio = drain_rtp(rtp, remote_rtp)
        received += count
        returned_audio.extend(audio)

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        select.select([rtp], [], [], min(0.05, deadline - time.monotonic()))
        count, audio = drain_rtp(rtp, remote_rtp)
        received += count
        returned_audio.extend(audio)

    bye_branch = "z9hG4bK" + uuid.uuid4().hex
    bye = (
        f"BYE sip:103@{args.asterisk} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{SIP_PORT};branch={bye_branch};rport\r\n"
        f"From: <sip:audio-peer@{args.asterisk}>;tag={FROM_TAG}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {CALL_ID}\r\n"
        f"CSeq: 2 BYE\r\n"
        "Max-Forwards: 70\r\n"
        "Content-Length: 0\r\n\r\n"
    )
    send_request(sip, bye, call_server)
    try:
        bye_response, _ = recv_sip(sip, 3)
        if response_code(bye_response) != 200:
            raise RuntimeError(f"BYE was not accepted: {bye_response.splitlines()[0]}")
    finally:
        sip.close()
        rtp.close()

    if received < 10:
        raise RuntimeError(f"only received {received} RTP packets from the PBX")
    if args.wav_out:
        write_wav_ulaw(args.wav_out, returned_audio)
    if args.wav_in and len(returned_audio) < len(tx_audio):
        raise RuntimeError(
            f"returned audio is short: {len(returned_audio)} of {len(tx_audio)} samples"
        )
    if bell103_rx:
        returned_pcm = b"".join(
            struct.pack("<h", ulaw_to_linear(sample)) for sample in returned_audio
        )
        answer_data = bell103_rx.feed_pcm(returned_pcm)
        if answer_data != bell103_reply:
            raise RuntimeError(
                f"Bell 103 answer payload mismatch: got {answer_data!r}, "
                f"expected {bell103_reply!r}"
            )
        print(f"Bell 103 caller decoded answer bytes: {answer_data!r}")
    print(f"SIP AudioSocket call passed: {received} PCMU RTP packets received from {remote_rtp[0]}:{remote_rtp[1]}")


if __name__ == "__main__":
    main()
