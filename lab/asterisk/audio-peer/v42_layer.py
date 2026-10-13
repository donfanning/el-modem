"""V.42 (LAPM) error control over any engine's raw line-bit stream.

The V.22 engine (spandsp_v22.py) has this logic built in; this is the same behaviour as a mixin
for engines that expose bits one at a time, such as the V.32 caller. The behaviours that were
learned against real peers are kept:

* detection (ODP out, ADP in) runs on the bits; user data is held until the outcome is known;
* a peer whose ADP characters come back to back (not 8-16 marks apart, like NIST's modem) is
  accepted by a lenient async decode of the same bits, then we go straight to LAPM;
* fallback to raw async data happens inside a gap of our ODP so the peer gets no stray byte;
* an established link that fails is "lost" (the call should drop), not silently raw.

The engine supplies (override in the subclass):
    _plain_tx_bit()            next framed bit when V.42 is not active (idle = 1)
    _plain_feed_bytes(data)    queue application bytes for async framing
    _plain_rx_bits(bits)       turn received line bits into application bytes (bytes)
and calls ``v42_tx_bit()`` once per transmitted bit and ``v42_rx_bits(bits)`` per received group.
"""

from __future__ import annotations

import threading
import math


class _Framer:
    """Incremental async framing (start 0, 8 data bits LSB first, stop 1)."""

    def __init__(self) -> None:
        self.buf: list[int] = []

    def push(self, bits) -> bytes:
        b = self.buf
        b += bits
        out = bytearray()
        i = 0
        while i + 10 <= len(b):
            if b[i] == 0 and b[i + 9] == 1:
                out.append(sum(b[i + 1 + j] << j for j in range(8)))
                i += 10
            else:
                i += 1
        del b[:i]
        return bytes(out)


class V42Layer:
    v42_mode = "off"                    # off | detect | lapm | failed | lost
    # Seconds added to the detection phase timer T400 (V.42 9.1.1, 750 ms by default, sized for a satellite hop). An engine whose
    # line has a longer round trip, or whose receive path runs behind real time, sets this so the ADP can still arrive inside the
    # window (dialup.world, 2026-10-07: ADP arrived 0.8-1.1 s after connect, the 0.73 s window had already closed on 2 of 4 calls).
    v42_detect_extra_seconds = 0.0

    def _v42_setup(self) -> None:
        self._v42 = None
        self._v42_lock = threading.RLock()
        self._v42_bits = 0
        self._v42_detect_deadline = 0
        self._v42_establish_deadline = 0
        self._v42_params = (0, 0, 0, None)
        self._v42_calling = False
        self._v42_direct = False
        self._v42_hold = bytearray()
        self._v42_adp_framer = _Framer()
        self._v42_adp_detector = None
        self._v42_rx = bytearray()      # bytes delivered by LAPM, waiting to be collected

    # -- control -------------------------------------------------------------------
    def set_v42_rate(self, bit_rate: int) -> None:
        """Update the retained link after a physical rate change, under its lock."""
        with self._v42_lock:
            link = self._v42
            if link is None:
                return
            rate = max(int(bit_rate), 300)
            previous = link.rate
            if rate == previous:
                return
            link.set_bit_rate(rate)
            for name in ('_v42_detect_deadline', '_v42_establish_deadline'):
                deadline = getattr(self, name)
                setattr(self, name, self._v42_bits + math.ceil(
                    max(0, deadline-self._v42_bits) * rate / previous))
            self._v42_params = (rate, *self._v42_params[1:])

    def start_v42(self, calling_party: bool, detect_seconds: float = 0.73, establish_seconds: float = 10.0,
                  frame_size: int = 0, window: int = 0, rate: int | None = None, detect: bool = True,
                  compression: tuple[int, int, int] | None = None, protocol: str = "lapm",
                  mnp_level: int = 4) -> None:
        """``protocol`` "lapm" is V.42 proper (ODP/ADP detection, then LAPM); "mnp" is the Annex A alternative
        procedure (MNP 2-4 class, see mnp_link.py): the originator sends its link request at once, no detection."""
        from v42_lapm import V42Link
        from v42_detect import AdpDetector

        with self._v42_lock:
            rate = max(rate or getattr(self, "bit_rate", 0), 300)
            extra = max(0.0, float(self.v42_detect_extra_seconds)) if detect else 0.0
            detect_seconds += extra
            self._v42_params = (rate, frame_size, window, compression)
            self._v42_calling = calling_party
            self._v42_direct = False
            self._v42_adp_framer = _Framer()
            self._v42_adp_detector = AdpDetector() if calling_party else None
            if protocol == "mnp":
                from mnp_link import MNP_LEVELS, MnpLink

                detect = False
                self._v42_adp_detector = None
                self._v42 = MnpLink(calling_party, bit_rate=rate, **MNP_LEVELS[mnp_level])
            else:
                self._v42 = V42Link(calling_party, detect=detect, bit_rate=rate, frame_size=frame_size,
                                    window=window, compression=compression, t400=0.75 + extra)
            self._v42_bits = 0
            self._v42_detect_deadline = int(detect_seconds * rate) if detect else 10 ** 9
            self._v42_establish_deadline = int((detect_seconds + establish_seconds) * rate)
            self._v42_direct = not detect
            self.v42_mode = "detect"

    @property
    def v42_compression(self):
        """(p0, p1, p2) when V.42bis was agreed on the established link, else None."""
        link = self._v42
        return link.compression if link is not None and self.v42_mode == "lapm" else None

    @property
    def v42_resolved(self) -> bool:
        return self.v42_mode in ("off", "lapm", "failed", "lost")

    @property
    def v42_lost(self) -> bool:
        return self.v42_mode == "lost"

    # -- bit I/O (called by the engine) -------------------------------------------------
    def v42_tx_bit(self) -> int:
        with self._v42_lock:
            if self.v42_mode == "lost":
                return 1
            if self.v42_mode in ("detect", "lapm"):
                bit = self._v42.tx_bit()
                self._v42_bits += 1
                self._v42_check()
                return bit
        return self._plain_tx_bit()

    def v42_tx_bits(self, count: int) -> list[int]:
        """Generate a mapping frame while preserving per-bit protocol clocks.

        Only drain already-generated bits on an established ordinary LAPM
        link. Detection, fallback, link loss, MNP and every empty-queue
        scheduling boundary use the existing single-bit path. Holding the
        layer lock prevents RX or application writes changing a queued run.
        """
        from v42_lapm import V42Link
        if count<0:raise ValueError('bit count must be nonnegative')
        single=self.v42_tx_bit
        if getattr(single,'__func__',None) is not V42Layer.v42_tx_bit:
            return [single() for _ in range(count)]
        bits=[]
        with self._v42_lock:
            while len(bits)<count:
                link=self._v42
                if (self.v42_mode=='lapm' and type(link) is V42Link
                        and link.state=='lapm' and not self._v42_hold):
                    queued=link.take_queued_tx(count-len(bits))
                    if queued:
                        self._v42_bits+=len(queued)
                        bits.extend(queued)
                        continue
                bits.append(single())
        return bits

    def v42_rx_bits(self, bits) -> bytes:
        """Feed received line bits; returns application bytes for the non-LAPM case."""
        with self._v42_lock:
            if self.v42_mode == "lost":
                return b""
            if self.v42_mode in ("detect", "lapm"):
                from v42_lapm import V42Link
                link = self._v42
                if (self.v42_mode == 'lapm' and type(link) is V42Link
                        and link.state == 'lapm'
                        and getattr(link.rx_bit, '__func__', None) is V42Link.rx_bit):
                    link.rx_bits(bits)
                    return b""
                for bit in bits:
                    if self.v42_mode == "detect" and self._v42_adp_detector is not None:
                        got = self._v42_adp_framer.push([bit])
                        if got:
                            self._v42_adp_detector.feed(got)
                    self._v42.rx_bit(bit)
                return b""
        return self._plain_rx_bits(bits)

    # -- application data ---------------------------------------------------------------
    def v42_feed_bytes(self, data: bytes) -> None:
        with self._v42_lock:
            if self.v42_mode == "lost":
                return
            if self.v42_mode == "detect":
                self._v42_hold += data
                return
            if self.v42_mode == "lapm":
                self._v42.write(data)
                return
        self._plain_feed_bytes(data)

    def v42_read(self) -> bytes:
        with self._v42_lock:
            if self._v42 is not None and self.v42_mode == "lapm":
                return self._v42.read()
        return b""

    def v42_pending_bits(self) -> int:
        with self._v42_lock:
            if self.v42_mode == "lapm":
                return self._v42.pending() * 10
            if self.v42_mode == "detect":
                return len(self._v42_hold) * 10
        return 0

    # -- state machine (as in spandsp_v22.py) -------------------------------------------------
    def _v42_check(self) -> None:
        link = self._v42
        state = link.state
        if self.v42_mode == "lapm" and state in ("down", "unsupported"):
            self.v42_mode = "lost"
            return
        if state == "lapm":
            self.v42_mode = "lapm"
            if self._v42_hold:
                link.write(bytes(self._v42_hold))
                self._v42_hold.clear()
        elif self._v42_lenient_adp():
            return
        elif state in ("unsupported", "down") or (
            state in ("detect", "idle")
            and ((state == 'detect' and not link.adp_seen
                  and self._v42_bits > self._v42_detect_deadline and self._odp_in_gap())
                 or self._v42_bits > self._v42_establish_deadline)
        ):
            self._v42_fallback()

    def _v42_lenient_adp(self) -> bool:
        link = self._v42
        det = self._v42_adp_detector
        if det is None or self.v42_mode != "detect" or link.adp_seen or self._v42_direct:
            return False
        verdict = det.result
        if verdict == "no_v42":
            if self._odp_in_gap():
                self._v42_fallback()
                return True
            return False
        if verdict != "v42" or not self._odp_in_gap():
            return False
        from v42_lapm import V42Link

        rate, frame_size, window, compression = self._v42_params
        link.close()
        self._v42 = V42Link(self._v42_calling, detect=False, bit_rate=rate, frame_size=frame_size, window=window,
                            compression=compression)
        self._v42_direct = True
        self._v42_bits = 0
        self._v42_detect_deadline = 10 ** 9
        self._v42_establish_deadline = int(10.0 * rate)
        return True

    def _odp_in_gap(self) -> bool:
        if self._v42_direct:
            return True
        p = self._v42_bits % 36
        return 10 <= p < 18 or 28 <= p < 36

    def _v42_fallback(self) -> None:
        self.v42_mode = "failed"
        self._plain_feed_marks(24)               # let the peer's async receiver resync
        held, self._v42_hold = bytes(self._v42_hold), bytearray()
        if held:
            self._plain_feed_bytes(held)

    # -- defaults an engine overrides ---------------------------------------------------------
    def _plain_tx_bit(self) -> int:
        return 1

    def _plain_feed_bytes(self, data: bytes) -> None:
        raise NotImplementedError

    def _plain_feed_marks(self, n: int) -> None:
        raise NotImplementedError

    def _plain_rx_bits(self, bits) -> bytes:
        raise NotImplementedError
