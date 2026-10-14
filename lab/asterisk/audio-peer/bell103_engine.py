"""Bell 103 (300 bit/s) as a line engine with the same surface as the V.22/V.32/V.34 engines, so ``V8Chain`` can fall back to it.

Wraps the stream transmitter and the receiver of ``bell103.py`` and adds a remote-carrier detector: ``connected`` goes True once the
far end's mark/space tones have dominated the line for ~100 ms (and not the 2100 Hz answer tone, which sits between the answerer's
2025/2225 Hz pair), and False after 0.5 s without them. Async 8N1 only; no error control (V.42 stays off).
"""
import math
import struct

from bell103 import FREQUENCIES, SAMPLE_RATE, Bell103Receiver, Bell103StreamTransmitter
from v42_layer import V42Layer

BLOCK = 160                                   # 20 ms detector blocks


def _energy(block, freq):
    w = 2.0 * math.pi * freq / SAMPLE_RATE
    re = sum(s * math.cos(w * i) for i, s in enumerate(block))
    im = sum(s * math.sin(w * i) for i, s in enumerate(block))
    return (re * re + im * im) / len(block)


class Bell103Engine(V42Layer):
    is_v32 = False
    is_v22 = False
    capture_detect_bits = False
    bit_rate = 300

    def __init__(self, calling_party: bool):
        self.calling = calling_party
        self.events: list[tuple[str, int]] = []
        self.connected = False
        self._tx = Bell103StreamTransmitter("originate" if calling_party else "answer")
        self._rx = Bell103Receiver("answer" if calling_party else "originate")
        self._peer = FREQUENCIES["answer" if calling_party else "originate"]
        self._buf: list[int] = []
        self._on = self._off = 0
        self._power = 0.0
        self._v42_setup()

    @property
    def rx_signal_power_dbm0(self) -> float:
        return 10 * math.log10(self._power / 6.76e6 + 1e-12) if self._power > 0 else -99.0

    def is_settled(self) -> bool:
        return self.connected

    def feed_bytes(self, data: bytes) -> None:
        self._tx.feed_bytes(data)

    def pending_bits(self) -> int:
        return self._tx.pending()

    def tx_samples(self, count: int) -> list[int]:
        return self._tx.samples(count)

    def feed_pcm(self, pcm: bytes) -> bytes:
        self._buf.extend(s for (s,) in struct.iter_unpack("<h", pcm))
        while len(self._buf) >= BLOCK:
            block, self._buf = self._buf[:BLOCK], self._buf[BLOCK:]
            self._carrier(block)
        return self._rx.feed_pcm(pcm)

    def _carrier(self, block) -> None:
        self._power = sum(s * s for s in block) / len(block)
        mark, space = (_energy(block, f) for f in self._peer)
        between = _energy(block, 2100.0) if self.calling else 0.0     # the V.25 answer tone is not a Bell 103 carrier
        present = self._power > 80.0 ** 2 and max(mark, space) > 0.25 * self._power and max(mark, space) > 3.0 * between
        if present:
            self._on, self._off = self._on + 1, 0
            if not self.connected and self._on >= 5:
                self.connected = True
                self.events.append(("CARRIER_UP", -2))
        else:
            self._off, self._on = self._off + 1, 0
            if self.connected and self._off >= 25:
                self.connected = False
                self.events.append(("CARRIER_DOWN", -1))
