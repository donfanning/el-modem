"""IAX2 signaling/media adapter backed by the small sans-I/O Rust protocol core."""
from __future__ import annotations

import socket
import time


class IaxUnavailable(RuntimeError):
    pass


class IaxTransport:
    """Own one IAX2 UDP association and expose call events plus G.711 µ-law.

    The modem DSP remains in OnlineAudioSession. This class only handles IAX2
    registration, call control and the transport's µ-law audio frames.
    """

    def __init__(self, host: str, port: int, username: str, secret: str, bind_host: str = "0.0.0.0"):
        try:
            from el_modem_iax import Client, MediaJitter
        except ImportError as exc:
            raise IaxUnavailable("native IAX2 module is missing; build/install native/iax") from exc
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
        if not infos:
            raise OSError(f"cannot resolve IAX server {host!r}")
        family, socktype, proto, _, address = infos[0]
        self.address = address
        self.socket = socket.socket(family, socktype, proto)
        # Match the wildcard address to the selected address family. DNS may
        # return an IPv6 result first; binding that socket to 0.0.0.0 raises
        # EAFNOSUPPORT before the modem can register or place a call.
        local_host = "::" if family == socket.AF_INET6 and bind_host == "0.0.0.0" else bind_host
        self.socket.bind((local_host, 0))
        self.socket.setblocking(False)
        self.client = Client("EL-Modem", username, secret)
        self.jitter = MediaJitter()
        self._play_at: float | None = None
        self._last_ulaw = b""
        self.call_id: int | None = None
        self.incoming_call_id: int | None = None
        self.call_direction: str | None = None
        self._events: list[dict] = []
        self.voice_rx_frames = 0
        self.voice_tx_frames = 0
        self._flush()

    def _flush(self) -> None:
        for datagram in self.client.drain_tx():
            self.socket.sendto(datagram, self.address)

    def poll(self) -> list[dict]:
        self.client.tick()
        while True:
            try:
                data, _ = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            self.client.input(data)
        self._flush()
        events = []
        for event in self.client.drain_events():
            if event["type"] in ("incoming", "dialing"):
                self.call_id = event["call"]
                if event["type"] == "incoming":
                    self.incoming_call_id = event["call"]
                    self.call_direction = "incoming"
                    self.jitter.reset()
                    self.voice_rx_frames = self.voice_tx_frames = 0
                    self._play_at = None
                else:
                    self.call_direction = "outgoing"
            elif event["type"] == "ended" and event.get("call") == self.call_id:
                self.call_id = None
                self.incoming_call_id = None
                self.call_direction = None
                self._play_at = None
            if event["type"] == "voice":
                self.jitter.push(event["ts"], event.get("ulaw", b""))
                self.voice_rx_frames += 1
                continue
            events.append(event)
        return events

    def audio_delay(self) -> float | None:
        return None if self._play_at is None else max(0.0, self._play_at - time.monotonic())

    def drain_audio(self) -> list[bytes]:
        now = time.monotonic()
        if self.call_id is None:
            self._play_at = None
            return []
        if self._play_at is None:
            self._play_at = now + 0.040
            return []
        frames = []
        while now >= self._play_at:
            kind, payload = self.jitter.pull()
            if kind == "play":
                self._last_ulaw = payload
                frames.append(payload)
            elif kind == "conceal" and self._last_ulaw:
                frames.append(self._last_ulaw)
            self._play_at += 0.020
        return frames

    def dial(self, number: str) -> None:
        self.call_direction = "outgoing"
        self.voice_rx_frames = self.voice_tx_frames = 0
        self.jitter.reset()
        self._play_at = None
        self.client.command("dial", number)
        self._flush()

    def answer(self, call_id: int | None = None) -> bool:
        call_id = self.incoming_call_id if call_id is None else call_id
        if call_id is None:
            return False
        self.client.command("answer", str(call_id))
        self._flush()
        self.incoming_call_id = None
        return True

    def hangup(self) -> None:
        if self.call_id is not None:
            self.client.command("hangup", str(self.call_id))
            self._flush()
            self.call_id = None

    def send_voice(self, payload: bytes) -> None:
        if self.call_id is not None and payload:
            self.client.voice(self.call_id, payload)
            self.voice_tx_frames += 1
            self._flush()

    def audio_stats(self) -> tuple[int, int, tuple[int, int, int, int, int, int]]:
        return self.voice_rx_frames, self.voice_tx_frames, self.jitter.stats()

    def fileno(self) -> int:
        """Expose the UDP socket so the DCE can wake on media arrival."""
        return self.socket.fileno()

    def close(self) -> None:
        self.hangup()
        self.socket.close()


class IaxMediaSocket:
    """RTP-shaped send sink used by OnlineAudioSession over IAX2."""

    def __init__(self, transport: IaxTransport):
        self.transport = transport

    def sendto(self, packet: bytes, _address) -> int:
        if len(packet) < 12 or packet[0] != 0x80 or packet[1] & 0x7F != 0:
            raise ValueError("IAX media adapter accepts only RTP PCMU packets")
        self.transport.send_voice(packet[12:])
        return len(packet)
