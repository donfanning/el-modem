"""Bridge an ordered V.90 bit path to the existing V.42/LAPM layer.

The V.90 digital downstream mapper supplies the transmit-bit callback; the
analogue-upstream V.34 receiver supplies decoded receive bits. This module
owns LAPM/V.42 and DTE byte queues, but deliberately does not pretend to
demodulate the analogue upstream waveform or manage RTP timing.
"""
from __future__ import annotations

from collections import deque

from v42_layer import V42Layer, _Framer


class V90LapmLink(V42Layer):
    """V.42/LAPM endpoint for a V.90 asymmetric modem data path.

    The caller supplies received raw line bits to :meth:`feed_line_bits` and
    requests downstream bits with :meth:`read_tx_bits`. Rate is explicit
    because V.90 has separate upstream and downstream rate negotiation; this
    class does not infer a single LAPM clock from either direction.
    """

    supports_v42 = True

    def __init__(self, *, rate: int, calling_party: bool,
                 frame_size: int = 0, window: int = 0,
                 xid_fallback: bool | None = None,
                 compression: tuple[int, int, int] | None = None,
                 compression_recovery: bool = False,
                 detect: bool = True):
        self._plain_bits: deque[int] = deque()
        self._plain_rx_framer = _Framer()
        self._rx_application = bytearray()
        self._v42_setup()
        self.start_v42(
            calling_party=bool(calling_party), rate=int(rate),
            frame_size=int(frame_size), window=int(window),
            xid_fallback=xid_fallback, protocol="lapm",
            compression=compression,
            compression_recovery=compression_recovery,
            detect=bool(detect))

    def feed_bytes(self, data: bytes) -> None:
        """Queue DTE bytes for LAPM (or async fallback if negotiation fails)."""
        self.v42_feed_bytes(bytes(data))

    def read_tx_bits(self, count: int) -> list[int]:
        """Generate bits for the downstream V.90 encoder."""
        return self.v42_tx_bits(count)

    def feed_line_bits(self, bits) -> bytes:
        """Feed upstream demapper bits and return newly delivered DTE bytes."""
        received = bytearray(self.v42_rx_bits(bits))
        received.extend(self.v42_read())
        self._rx_application.extend(received)
        return bytes(received)

    def take_received(self) -> bytes:
        """Drain bytes delivered by LAPM or raw async fallback."""
        data = bytes(self._rx_application)
        self._rx_application.clear()
        return data

    def pending_bits(self) -> int:
        return self.v42_pending_bits() + len(self._plain_bits)

    def diagnostics(self) -> dict:
        """Return bounded V.42 startup and LAPM counters for call reports."""
        link = self._v42
        if link is None:
            return {"mode": self.v42_mode, "state": None}
        detector = self._v42_adp_detector
        return {
            "mode": self.v42_mode,
            "state": getattr(link, "state", None),
            "calling": getattr(self, "_v42_calling", None),
            "direct_start": bool(getattr(self, "_v42_direct", False)),
            "bits_emitted": int(getattr(self, "_v42_bits", 0)),
            "detect_deadline_bits": int(
                getattr(self, "_v42_detect_deadline", 0)),
            "establish_deadline_bits": int(
                getattr(self, "_v42_establish_deadline", 0)),
            "adp_armed": bool(getattr(self, "_v42_adp_armed", False)),
            "adp_seen": getattr(link, "adp_seen", None),
            "adp_detector_result": getattr(detector, "result", None),
            "answerer_adp_count": getattr(link, "_adp_count", None),
            "answerer_adp_started": getattr(link, "_answer_adp", None),
            "link_phase": getattr(link, "_phase", None),
            "link_bits_sent": getattr(link, "bits_sent", None),
            "link_detect_bits": getattr(link, "_detect_bit_count", None),
            "link_stats": dict(getattr(link, "stats", {})),
            "link_events": list(getattr(link, "events", ())[-32:]),
        }

    def _plain_tx_bit(self) -> int:
        return self._plain_bits.popleft() if self._plain_bits else 1

    def _plain_feed_bytes(self, data: bytes) -> None:
        for value in data:
            self._plain_bits.extend(
                [0] + [(value >> bit) & 1 for bit in range(8)] + [1])

    def _plain_feed_marks(self, count: int) -> None:
        self._plain_bits.extend([1] * int(count))

    def _plain_rx_bits(self, bits) -> bytes:
        return self._plain_rx_framer.push(list(bits))
