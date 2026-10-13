"""A small, self-contained ARQ/framing layer for the online Bell 103 data
path -- NOT an implementation of ITU-T V.42/LAPM.

Why this exists: the lab's two-modem throughput testing found that raw RTP
audio carrying the Bell 103 waveform has no forward error correction or
retransmission, so even 1% random packet loss (via ``tc netem``) reliably
corrupted transferred data (a lost packet drops a slice of FSK waveform,
desyncing async-bit framing). Real modems solve this with LAPM
(ITU-T V.42 Annex A), an HDLC-derived ARQ protocol negotiated during a
V.8bis-style detection phase, plus V.42bis compression.

What this module provides instead: a simple stop-and-wait ARQ with
byte-stuffed HDLC-style framing (flag/escape bytes) and a CRC-16-CCITT
(X.25/HDLC FCS polynomial) frame check, so corrupted or lost frames are
detected and retransmitted automatically rather than silently delivered
as corrupted application bytes. This is "good enough" for this project's
own two softmodem endpoints to interoperate reliably over a lossy audio
path, and is deliberately simple to reason about and test.

What this module does NOT do, and should not be assumed to do:
 - It does not implement the ITU-T V.42 bit-exact protocol, so it will
   NOT let this softmodem negotiate real error control with an actual
   hardware modem or another LAPM implementation. For interoperability
   with real hardware modems, genuine V.42/LAPM (and the V.8bis/V.42
   detection-phase negotiation that precedes it) is still required --
   this is tracked as separate, larger future work.
 - It uses a single-frame (stop-and-wait) window, not LAPM's sliding
   window, trading some throughput for a far simpler, easier-to-verify
   implementation given the already-slow 300 bps Bell 103 link.
 - It does not implement data compression (the V.42bis counterpart).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

FLAG = 0x7E
ESC = 0x7D
ESC_XOR = 0x20

FRAME_I = 0x49  # "I" -- carries a chunk of application payload
FRAME_RR = 0x52  # "R" -- acknowledges a specific sequence number

MAX_PAYLOAD = 32
BELL103_BAUD = 300
BELL103_BITS_PER_CHAR = 10  # async 8N1: start + 8 data + stop


def _frame_transmit_seconds(payload_len: int, baud: float = BELL103_BAUD) -> float:
    """Worst-case time to transmit an encode_frame() result over an async
    serial link at ``baud``: every body byte (frame type + seq + payload +
    2-byte CRC) could need HDLC byte-stuffing (doubling it), plus the two
    unescaped flag bytes."""
    worst_case_bytes = 2 * (payload_len + 4) + 2
    return worst_case_bytes * BELL103_BITS_PER_CHAR / baud


def default_retransmit_timeout(
    max_payload: int = MAX_PAYLOAD, baud: float = BELL103_BAUD
) -> float:
    """A retransmit timeout must safely exceed the round trip of sending
    an I-frame and getting its RR acknowledgement back. Testing this
    module against the real Bell 103 audio path found that a fixed,
    payload-size-independent timeout (the original 0.75s guess) retransmits
    before the first attempt has even finished going out over the slow
    serial link -- interleaving duplicate frame bytes into the transmit
    queue and corrupting the stream even with zero real packet loss. This
    derives a safe timeout from the configured channel rate and frame size
    instead."""
    i_frame_time = _frame_transmit_seconds(max_payload, baud)
    rr_frame_time = _frame_transmit_seconds(0, baud)
    return 2.5 * (i_frame_time + rr_frame_time)


DEFAULT_RETRANSMIT_TIMEOUT = default_retransmit_timeout()


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """The same CRC-16/CCITT-FALSE (X.25/HDLC FCS) polynomial real LAPM
    frames use for their frame check sequence."""
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _stuff(data: bytes) -> bytes:
    output = bytearray()
    for byte in data:
        if byte in (FLAG, ESC):
            output.append(ESC)
            output.append(byte ^ ESC_XOR)
        else:
            output.append(byte)
    return bytes(output)


def _unstuff(data: bytes) -> bytes:
    output = bytearray()
    escaped = False
    for byte in data:
        if escaped:
            output.append(byte ^ ESC_XOR)
            escaped = False
        elif byte == ESC:
            escaped = True
        else:
            output.append(byte)
    return bytes(output)


def encode_frame(frame_type: int, seq: int, payload: bytes = b"") -> bytes:
    body = bytes((frame_type, seq)) + payload
    crc = crc16_ccitt(body)
    body_with_crc = body + bytes((crc >> 8, crc & 0xFF))
    return bytes((FLAG,)) + _stuff(body_with_crc) + bytes((FLAG,))


@dataclass
class _ParsedFrame:
    frame_type: int
    seq: int
    payload: bytes


class FrameParser:
    """Accumulates a raw byte stream (as decoded off the Bell 103 audio
    channel) and extracts flag-delimited frames, discarding anything that
    fails its CRC (treated the same as a lost frame -- the ARQ layer's
    timeout/retransmit logic recovers either way)."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._in_frame = False

    def feed(self, data: bytes) -> list[_ParsedFrame]:
        frames: list[_ParsedFrame] = []
        for byte in data:
            if byte == FLAG:
                if self._in_frame and self._buffer:
                    frame = self._try_parse(bytes(self._buffer))
                    if frame is not None:
                        frames.append(frame)
                self._buffer.clear()
                self._in_frame = True
            elif self._in_frame:
                self._buffer.append(byte)
        return frames

    @staticmethod
    def _try_parse(stuffed: bytes) -> _ParsedFrame | None:
        raw = _unstuff(stuffed)
        if len(raw) < 4:
            return None
        body, received_crc = raw[:-2], (raw[-2] << 8) | raw[-1]
        if crc16_ccitt(body) != received_crc:
            return None
        return _ParsedFrame(frame_type=body[0], seq=body[1], payload=body[2:])


@dataclass
class _PendingFrame:
    seq: int
    encoded: bytes
    sent_at: float


class ArqSession:
    """Stop-and-wait ARQ: one outstanding unacknowledged I-frame at a time.
    Call ``feed_terminal_bytes`` with new application bytes to queue them,
    call ``pump_outgoing`` periodically (and after feeding bytes) to get
    bytes that should be written to the Bell 103 transmitter, and call
    ``feed_received`` with bytes decoded off the Bell 103 receiver to get
    back any newly-delivered, in-order application payload."""

    def __init__(
        self,
        max_payload: int = MAX_PAYLOAD,
        retransmit_timeout: float | None = None,
    ):
        self._max_payload = max_payload
        self._retransmit_timeout = (
            retransmit_timeout
            if retransmit_timeout is not None
            else default_retransmit_timeout(max_payload)
        )
        self._send_queue: bytearray = bytearray()
        self._pending: _PendingFrame | None = None
        self._tx_seq = 0
        self._rx_expected_seq = 0
        self._parser = FrameParser()
        self._pending_rr: int | None = None

    def feed_terminal_bytes(self, data: bytes) -> None:
        self._send_queue.extend(data)

    def pump_outgoing(self, now: float | None = None) -> bytes:
        now = time.monotonic() if now is None else now
        output = bytearray()
        if self._pending_rr is not None:
            output.extend(encode_frame(FRAME_RR, self._pending_rr))
            self._pending_rr = None
        if self._pending is not None:
            if now - self._pending.sent_at >= self._retransmit_timeout:
                self._pending = _PendingFrame(
                    self._pending.seq, self._pending.encoded, now
                )
                output.extend(self._pending.encoded)
            return bytes(output)
        if self._send_queue:
            chunk = bytes(self._send_queue[: self._max_payload])
            del self._send_queue[: self._max_payload]
            encoded = encode_frame(FRAME_I, self._tx_seq, chunk)
            self._pending = _PendingFrame(self._tx_seq, encoded, now)
            output.extend(encoded)
        return bytes(output)

    def feed_received(self, data: bytes) -> bytes:
        delivered = bytearray()
        for frame in self._parser.feed(data):
            if frame.frame_type == FRAME_RR:
                if self._pending is not None and frame.seq == self._pending.seq:
                    self._pending = None
                    self._tx_seq = (self._tx_seq + 1) & 0xFF
            elif frame.frame_type == FRAME_I:
                if frame.seq == self._rx_expected_seq:
                    delivered.extend(frame.payload)
                    self._pending_rr = frame.seq
                    self._rx_expected_seq = (self._rx_expected_seq + 1) & 0xFF
                elif frame.seq == (self._rx_expected_seq - 1) & 0xFF:
                    # Duplicate of the last frame we already accepted --
                    # our RR must have been lost. Re-send it without
                    # re-delivering the payload.
                    self._pending_rr = frame.seq
        return bytes(delivered)

    def idle(self) -> bool:
        """True once every queued byte has been sent and acknowledged --
        useful for tests/drivers that want to know when it is safe to
        tear down a session without losing buffered data."""
        return not self._send_queue and self._pending is None

    def pending_bytes(self) -> int:
        """Application bytes not yet fully acknowledged by the peer --
        both still queued and in the single outstanding (possibly
        in-flight/retransmitted) frame."""
        pending_frame_bytes = (
            len(self._pending.encoded) if self._pending is not None else 0
        )
        return len(self._send_queue) + pending_frame_bytes
