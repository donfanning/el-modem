"""Simplified, bidirectional V.8bis-style modulation-mode negotiation.

NOTE: this is NOT ITU-T V.8 or V.8bis. Those standards define their own
tone/FSK signalling and capability octets so unlike modems can find a common
mode; this module is a small lab protocol that lets this project's own
softmodems do the same job over the one modulation every endpoint always
speaks: Bell 103 (300 bps FSK). It will not negotiate with a real hardware
modem.

Negotiation is symmetric -- neither side is "the master":

1. Both ends open a connection speaking Bell 103 and each sends an OFFER
   listing the modes it supports.
2. When an end receives the peer's OFFER it picks the highest mode both
   support (a pure function of the two offers, so both ends agree without
   a coordinator) and sends a CONFIRM naming it.
3. An end is finished once it has sent its own CONFIRM *and* received the
   peer's. The caller then switches its transmitter (after its CONFIRM has
   drained) and receiver (as soon as the peer's CONFIRM arrives) to the
   agreed modulation.

Robustness: frames carry a checksum and are resent while the peer has not
been heard from. If nothing at all is heard before the timeout, the session
falls back to Bell 103 and never switches -- the universal fallback. There is
no protection once an end has switched modulation, so a CONFIRM lost *after*
the peer already acted on it cannot be recovered; this is acceptable only
because the lab channel is clean at negotiation time.

Frame layout (all over the Bell 103 async byte stream):

    SYN SYN kind length payload... checksum

``checksum`` is the low byte of the sum of kind, length and payload bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

MODE_BELL103 = 1
MODE_BELL212 = 2

# mode id -> (name, line speed in bps)
MODES = {
    MODE_BELL103: ("Bell 103", 300),
    MODE_BELL212: ("Lab 1200", 1200),
}

SYN = 0x16
KIND_OFFER = ord("O")
KIND_CONFIRM = ord("C")
MAX_PAYLOAD = 16

DEFAULT_RETRY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 4.0


def line_speed(mode: int) -> int:
    return MODES[mode][1]


def encode_frame(kind: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("negotiation payload too large")
    body = bytes((kind, len(payload))) + payload
    return bytes((SYN, SYN)) + body + bytes((sum(body) & 0xFF,))


class FrameParser:
    """Incrementally extracts valid frames from a byte stream, silently
    discarding noise, partial garbage, and frames whose checksum fails."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self._buffer.extend(data)
        frames: list[tuple[int, bytes]] = []
        while True:
            start = self._find_sync()
            if start < 0:
                # Keep a possible lone trailing SYN; drop everything else.
                keep = 1 if self._buffer and self._buffer[-1] == SYN else 0
                del self._buffer[: len(self._buffer) - keep]
                return frames
            del self._buffer[:start]
            if len(self._buffer) < 4:
                return frames
            length = self._buffer[3]
            if length > MAX_PAYLOAD:
                del self._buffer[:2]
                continue
            total = 2 + 2 + length + 1
            if len(self._buffer) < total:
                return frames
            body = bytes(self._buffer[2 : total - 1])
            if sum(body) & 0xFF == self._buffer[total - 1]:
                frames.append((body[0], body[2:]))
                del self._buffer[:total]
            else:
                del self._buffer[:2]

    def _find_sync(self) -> int:
        for index in range(len(self._buffer) - 1):
            if self._buffer[index] == SYN and self._buffer[index + 1] == SYN:
                return index
        return -1


def choose_mode(local: tuple[int, ...], remote: tuple[int, ...]) -> int:
    common = set(local) & set(remote)
    return max(common) if common else MODE_BELL103


@dataclass
class _State:
    peer_modes: tuple[int, ...] | None = None
    chosen: int | None = None
    sent_confirm: bool = False
    got_confirm: bool = False
    fell_back: bool = False
    last_heard: float = 0.0
    last_offer_sent: float = 0.0


class ModeNegotiator:
    """Pure protocol logic: no audio, sockets, or clocks of its own. Every
    method that can produce bytes to transmit returns them; the caller
    queues them on its (Bell 103) transmitter."""

    def __init__(
        self,
        supported: tuple[int, ...] = (MODE_BELL103, MODE_BELL212),
        retry_seconds: float = DEFAULT_RETRY_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if MODE_BELL103 not in supported:
            raise ValueError("Bell 103 is the universal fallback and must be supported")
        unknown = [mode for mode in supported if mode not in MODES]
        if unknown:
            raise ValueError(f"unknown modes: {unknown}")
        self.supported = tuple(sorted(set(supported)))
        self._retry = retry_seconds
        self._timeout = timeout_seconds
        self._parser = FrameParser()
        self._state = _State()
        self._started = False

    # -- results ---------------------------------------------------------
    @property
    def complete(self) -> bool:
        s = self._state
        return s.fell_back or (s.sent_confirm and s.got_confirm)

    @property
    def mode(self) -> int | None:
        """The agreed mode once ``complete``; ``None`` before."""
        if not self.complete:
            return None
        return MODE_BELL103 if self._state.fell_back else self._state.chosen

    @property
    def fell_back(self) -> bool:
        return self._state.fell_back

    @property
    def peer_confirmed(self) -> bool:
        return self._state.got_confirm

    @property
    def confirm_sent(self) -> bool:
        return self._state.sent_confirm

    # -- protocol --------------------------------------------------------
    def _offer(self) -> bytes:
        return encode_frame(KIND_OFFER, bytes(self.supported))

    def _confirm(self) -> bytes:
        return encode_frame(KIND_CONFIRM, bytes((self._state.chosen,)))

    def start(self, now: float) -> bytes:
        self._started = True
        self._state.last_heard = now
        self._state.last_offer_sent = now
        return self._offer()

    def feed(self, data: bytes, now: float) -> bytes:
        """Consume decoded Bell 103 bytes; returns bytes to transmit."""
        out = bytearray()
        s = self._state
        if s.fell_back:
            return b""
        for kind, payload in self._parser.feed(data):
            s.last_heard = now
            if kind == KIND_OFFER and payload:
                peer = tuple(m for m in payload if m in MODES)
                s.peer_modes = peer
                s.chosen = choose_mode(self.supported, peer)
                # The peer (re)sending its OFFER means it may not have
                # seen our CONFIRM/OFFER yet: answer every time.
                out.extend(self._confirm())
                s.sent_confirm = True
            elif kind == KIND_CONFIRM and len(payload) == 1:
                theirs = payload[0]
                if theirs not in MODES:
                    continue
                # Both ends normally computed the same mode. If they ever
                # differ, taking the lower is symmetric, so both agree.
                s.chosen = theirs if s.chosen is None else min(s.chosen, theirs)
                s.got_confirm = True
        return bytes(out)

    def poll(self, now: float) -> bytes:
        """Drive retries and the give-up timeout; returns bytes to transmit."""
        s = self._state
        if not self._started or self.complete:
            return b""
        if now - s.last_heard >= self._timeout:
            s.fell_back = True
            return b""
        # Keep offering until the peer's CONFIRM proves it received our
        # OFFER -- hearing the peer's OFFER alone does not.
        if not s.got_confirm and now - s.last_offer_sent >= self._retry:
            s.last_offer_sent = now
            return self._offer()
        return b""
