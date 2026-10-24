"""V.42 (LAPM) error control over any engine's raw line-bit stream.

The project-owned V.22 engine (`v22.py`) and engines that expose synchronous
bits one at a time, such as the V.32 caller, use this layer. The behaviors that were
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
import os
import time


class _Framer:
    """Incremental async framing (start 0, data bits LSB first, stop 1)."""

    def __init__(self, data_bits: int = 8) -> None:
        if data_bits not in (7, 8):
            raise ValueError("async framing supports 7 or 8 data bits")
        self.data_bits = data_bits
        self.frame_bits = data_bits + 2
        self.buf: list[int] = []
        self.total_bits = 0
        self.last_char_end: int | None = None

    def push(self, bits) -> bytes:
        return bytes(char for char, _gap in self.push_with_gaps(bits))

    def push_with_gaps(self, bits) -> list[tuple[int, int | None]]:
        b = self.buf
        first_bit = self.total_bits - len(b)
        b += bits
        self.total_bits += len(bits)
        out: list[tuple[int, int | None]] = []
        i = 0
        while i + self.frame_bits <= len(b):
            if b[i] == 0 and b[i + self.frame_bits - 1] == 1:
                start = first_bit + i
                gap = (None if self.last_char_end is None else
                       start - self.last_char_end)
                value = sum(b[i + 1 + j] << j for j in range(self.data_bits))
                out.append((value, gap))
                self.last_char_end = start + self.frame_bits
                i += self.frame_bits
            else:
                i += 1
        del b[:i]
        return out


class V42Layer:
    ADP_TRACE_LIMIT = 8192
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
        self._v42_params = (0, 0, 0, None, None)
        self._v42_calling = False
        self._v42_direct = False
        self._v42_hold = bytearray()
        self._v42_adp_framer = _Framer()
        self._v42_adp_detector = None
        self._v42_adp_armed = False
        self._v42_adp_trace_enabled = os.environ.get("V42_CAPTURE_DETECTOR") == "1"
        self._v42_adp_trace_bytes = bytearray()
        self._v42_adp_trace_gaps: list[int | None] = []
        self._v42_adp_trace_truncated = False
        self._v42_rx = bytearray()      # bytes delivered by LAPM, waiting to be collected
        self._v42_compression_recovery = False
        self._v42_recovery_stage: str | None = None
        self._v42_recovery_deadline = 0
        self._v42_recovery_peer_busy_at: float | None = None
        self._v42_local_release = False
        self._v42_recovery_timeout_seconds = 8.0
        self._v42_recovery_hold_limit = 1024 * 1024
        self._v42_recovery_stats = dict(
            attempts=0, completed=0, failures=0,
            requeued_dte_bytes=0, c_error_input_octets=0,
            discarded_encoded_octets=0,
            last_failure=None,
        )

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

    def start_v42(self, calling_party: bool, detect_seconds: float = 0.75, establish_seconds: float = 10.0,
                  frame_size: int = 0, window: int = 0, rate: int | None = None, detect: bool = True,
                  compression: tuple[int, int, int] | None = None, protocol: str = "lapm",
                  mnp_level: int = 4, xid_fallback: bool | None = None,
                  compression_recovery: bool = False) -> None:
        """``protocol`` "lapm" is V.42 proper (ODP/ADP detection, then LAPM); "mnp" is the Annex A alternative
        procedure (MNP 2-4 class, see mnp_link.py): the originator sends its link request at once, no detection."""
        from v42_lapm import V42Link
        from v42_detect import AdpDetector

        with self._v42_lock:
            rate = max(rate or getattr(self, "bit_rate", 0), 300)
            extra = max(0.0, float(self.v42_detect_extra_seconds)) if detect else 0.0
            detect_seconds += extra
            self._v42_params = (rate, frame_size, window, compression, xid_fallback)
            self._v42_calling = calling_party
            self._v42_compression_recovery = bool(compression_recovery and protocol == "lapm")
            self._v42_recovery_stage = None
            self._v42_recovery_deadline = 0
            self._v42_recovery_peer_busy_at = None
            self._v42_local_release = False
            self._v42_direct = False
            self._v42_adp_framer = _Framer()
            self._v42_adp_detector = AdpDetector() if calling_party else None
            self._v42_adp_armed = False
            self._v42_adp_trace_bytes.clear()
            self._v42_adp_trace_gaps.clear()
            self._v42_adp_trace_truncated = False
            if protocol == "mnp":
                from mnp_link import MNP_LEVELS, MnpLink

                detect = False
                self._v42_adp_detector = None
                self._v42 = MnpLink(calling_party, bit_rate=rate, **MNP_LEVELS[mnp_level])
            else:
                self._v42 = V42Link(calling_party, detect=detect, bit_rate=rate, frame_size=frame_size,
                                    window=window, compression=compression, t400=0.75 + extra,
                                    xid_fallback=xid_fallback)
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
            if self.v42_mode in ("detect", "lapm", "recovering"):
                bit = self._v42.tx_bit()
                self._v42_bits += 1
                self._v42_arm_adp_detector()
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
            if self.v42_mode in ("detect", "lapm", "recovering"):
                from v42_lapm import V42Link
                link = self._v42
                if (self.v42_mode == 'lapm' and type(link) is V42Link
                        and link.state == 'lapm'
                        and getattr(link.rx_bit, '__func__', None) is V42Link.rx_bit):
                    link.rx_bits(bits)
                    self._v42_check()
                    return b""
                for bit in bits:
                    if (self.v42_mode == "detect" and self._v42_adp_detector is not None
                            and self._v42_adp_armed):
                        got = self._v42_adp_framer.push_with_gaps([bit])
                        if got:
                            if self._v42_adp_trace_enabled:
                                room = self.ADP_TRACE_LIMIT - len(self._v42_adp_trace_bytes)
                                if room > 0:
                                    kept = got[:room]
                                    self._v42_adp_trace_bytes.extend(char for char, _gap in kept)
                                    self._v42_adp_trace_gaps.extend(gap for _char, gap in kept)
                                if len(got) > room:
                                    self._v42_adp_trace_truncated = True
                            self._v42_adp_detector.feed(
                                bytes(char for char, _gap in got),
                                [gap for _char, gap in got],
                            )
                    self._v42.rx_bit(bit)
                self._v42_check()
                return b""
        return self._plain_rx_bits(bits)

    def v42_adp_trace(self) -> dict | None:
        """Return the opt-in, bounded byte/framing trace used by ADP detection."""
        with self._v42_lock:
            if not self._v42_adp_trace_enabled:
                return None
            return {
                "bytes_hex": bytes(self._v42_adp_trace_bytes).hex(),
                "mark_gaps": list(self._v42_adp_trace_gaps),
                "truncated": self._v42_adp_trace_truncated,
                "byte_limit": self.ADP_TRACE_LIMIT,
            }

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
            if self.v42_mode == "recovering":
                if len(self._v42_hold) + len(data) > self._v42_recovery_hold_limit:
                    self._v42_recovery_fail("DTE data exceeded recovery hold limit")
                else:
                    self._v42_hold += data
                return
        self._plain_feed_bytes(data)

    def v42_read(self) -> bytes:
        with self._v42_lock:
            if self._v42_rx:
                out = bytes(self._v42_rx)
                self._v42_rx.clear()
                return out
            if self._v42 is not None and self.v42_mode == "lapm":
                return self._v42.read()
        return b""

    def v42_pending_bits(self) -> int:
        with self._v42_lock:
            if self.v42_mode == "lapm":
                return self._v42.pending() * 10
            if self.v42_mode == "detect":
                return len(self._v42_hold) * 10
            if self.v42_mode == "recovering":
                pending = self._v42.pending() if self._v42 is not None else 0
                return (pending + len(self._v42_hold)) * 10
        return 0

    def v42_set_local_busy(self, busy: bool) -> None:
        """Throttle an established LAPM peer while DTE receive flow is paused."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                self._v42.set_local_busy(busy)

    def v42_renegotiate_parameters(self, *, tx_n401: int, rx_n401: int,
                                   tx_k: int, rx_k: int) -> bool:
        """Request connected V.42 N401/window renegotiation when the link supports it."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                return self._v42.renegotiate_parameters(
                    tx_n401=tx_n401, rx_n401=rx_n401, tx_k=tx_k, rx_k=rx_k
                )
        return False

    def v42_request_compression_reset(self) -> bool:
        """Queue a V.42bis encoder RESET at the current DTE transmit boundary."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                return self._v42.request_v42bis_reset()
        return False

    def v42_request_test(self, information: bytes) -> bool:
        """Request the negotiated V.42 TEST procedure (L-TEST primitive)."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                return self._v42.request_test(information)
        return False

    def v42_read_test_indication(self) -> bytes | None:
        """Read an L-TEST indication for the control function to handle."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                return self._v42.take_test_indication()
        return None

    def v42_release(self) -> bool:
        """Initiate LAPM orderly release; physical-call policy stays with the DCE."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                started = self._v42.release()
                self._v42_local_release = started
                return started
        return False

    def v42_request_break(self, *, length_ms: int | None = None,
                          destructive: bool = False, expedited: bool = False) -> bool:
        """Request a V.42 L-SIGNAL break; the control function owns DTE action."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                return self._v42.request_break(length_ms=length_ms,
                                               destructive=destructive,
                                               expedited=expedited)
        return False

    def v42_read_break_events(self) -> list[dict]:
        """Collect LAPM break indications/confirmations for the DCE control function."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                return self._v42.read_break_events()
        return []

    def v42_acknowledge_break(self) -> bool:
        """Return the L-SIGNAL response after the DCE has applied a received BREAK."""
        from v42_lapm import V42Link
        with self._v42_lock:
            if self.v42_mode == "lapm" and type(self._v42) is V42Link:
                acknowledged = self._v42.acknowledge_break()
                if acknowledged and self._v42._break_rx_reset_applied:
                    # These octets were decoded but had not yet crossed the
                    # L-DATA indication to the DCE when destructive BREAK hit.
                    self._v42_rx.clear()
                    self._v42._break_rx_reset_applied = False
                return acknowledged
        return False

    # -- state machine (as in spandsp_v22.py) -------------------------------------------------
    def _v42_check(self) -> None:
        link = self._v42
        state = link.state
        if self.v42_mode == "lapm" and getattr(link, "_compression_error", None):
            if not self._v42_compression_recovery:
                self.v42_mode = "lost"
                return
            # C-ERROR is local to the decoder. Stop new peer data with RNR,
            # let our reverse direction drain, and only then release LAPM.
            # The application data accepted during this interval is held
            # locally until a fresh L-ESTABLISH confirms new dictionaries.
            self.v42_mode = "recovering"
            self._v42_recovery_stage = "drain"
            self._v42_recovery_deadline = time.monotonic() + self._v42_recovery_timeout_seconds
            self._v42_recovery_peer_busy_at = None
            self._v42_recovery_stats["attempts"] += 1
            self._v42_recovery_stats["c_error_input_octets"] += int(
                getattr(link, "_compression_error_octets", 0)
            )
            link.set_local_busy(True)
            link._event("V.42bis C-ERROR: bounded LAPM recovery started; RNR sent")
            return
        if self.v42_mode == "recovering":
            if time.monotonic() >= self._v42_recovery_deadline:
                self._v42_recovery_fail("bounded LAPM re-establishment timed out")
                return
            if state == "released":
                self._v42_rx += link.read()
                if not self._v42_recovery_preserve_unsent(link):
                    return
                # A peer may initiate DISC while this endpoint is still
                # draining its reverse direction. Treat that as release
                # progress, not as a failed LAPM state transition.
                self._v42_restart_link(calling=self._v42_calling)
                return
            if state in ("down", "unsupported"):
                self._v42_recovery_fail(f"LAPM re-establishment failed ({state})")
                return
            if self._v42_recovery_stage == "drain":
                if state in ("releasing", "release_pending"):
                    self._v42_recovery_stage = "release"
                    return
                if state != "lapm":
                    self._v42_recovery_fail(f"LAPM left connected state while draining ({state})")
                    return
                now = time.monotonic()
                if link._peer_busy:
                    if self._v42_recovery_peer_busy_at is None:
                        self._v42_recovery_peer_busy_at = now
                    elif now - self._v42_recovery_peer_busy_at >= max(
                            0.1, link._t401_bits / max(link.rate, 1)):
                        # A persistent RNR during recovery can mean that both
                        # endpoints have C-ERROR and are waiting for the
                        # opposite data direction to drain. Wait one T401 for
                        # transient busy to clear, then let DISC reset the
                        # ambiguous encoded window within the overall bound.
                        link._event(
                            "V.42bis recovery: peer remained busy for T401; ending drain"
                        )
                        if not link.release():
                            self._v42_recovery_fail("could not begin LAPM release after busy drain")
                        else:
                            self._v42_recovery_stage = "release"
                            self._v42_recovery_peer_busy_at = None
                        return
                else:
                    self._v42_recovery_peer_busy_at = None
                if (link.pending() == 0 and not link._unacked and not link._retx
                        and not link._ack_pending and not link._tx):
                    if not link.release():
                        self._v42_recovery_fail("could not begin orderly LAPM release")
                    else:
                        self._v42_recovery_stage = "release"
                        self._v42_recovery_peer_busy_at = None
                        link._event("V.42bis recovery: outbound data drained; DISC sent")
                return
            if state == "lapm":
                self.v42_mode = "lapm"
                self._v42_recovery_stage = None
                self._v42_recovery_deadline = 0
                self._v42_recovery_peer_busy_at = None
                self._v42_local_release = False
                self._v42_recovery_stats["completed"] += 1
                if self._v42_hold:
                    link.write(bytes(self._v42_hold))
                    self._v42_hold.clear()
                link._event("V.42bis recovery complete; LAPM and compression re-established")
                return
        if state == "released":
            # An orderly DISC/release indication is not a damaged line.
            # Preserve already-delivered DTE bytes if LAPM had been active,
            # then return this auto-reliable control function to raw mode.
            if self.v42_mode == "lapm":
                self._v42_rx += link.read()
            if self._v42_compression_recovery and not self._v42_local_release:
                # A peer DISC may be the other endpoint's C-ERROR recovery
                # request. Re-arm LAPM reception during the bounded window;
                # the caller will initiate, while the answerer waits.
                if not self._v42_recovery_preserve_unsent(link):
                    return
                self.v42_mode = "recovering"
                self._v42_recovery_stage = "await_establish"
                self._v42_recovery_deadline = time.monotonic() + self._v42_recovery_timeout_seconds
                self._v42_restart_link(calling=self._v42_calling)
                return
            self._v42_local_release = False
            self._v42_fallback()
            return
        if state == "down" and getattr(link, "_fatal_protocol_error", False):
            # Some negotiation errors require releasing the call, not the
            # normal no-V.42/raw-data fallback. The DCE observes v42_lost and
            # drops the carrier after the control function rejects the XID.
            self.v42_mode = "lost"
            return
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
        else:
            # Once the originator has actually sent XID, a strict policy
            # owns establishment timing: keep the link in protocol mode until
            # V42Link exhausts N400 and enters ``down``. Falling back merely
            # because the layer's broader establishment deadline elapsed
            # would turn an incomplete parameter exchange into raw CONNECT.
            strict_xid_pending = (
                self._v42_calling and state == "idle"
                and getattr(link, "_xid_sent", 0) > 0
                and not getattr(link, "xid_fallback", True)
            )
            if state in ("unsupported", "down") or (
                not strict_xid_pending and state in ("detect", "idle")
                and ((state == 'detect' and not link.adp_seen
                      and self._v42_bits > self._v42_detect_deadline and self._odp_in_gap())
                     or self._v42_bits > self._v42_establish_deadline)
            ):
                self._v42_fallback()

    def _v42_restart_link(self, *, calling: bool) -> None:
        """Re-arm LAPM after orderly release, retaining the physical carrier."""
        from v42_lapm import V42Link

        rate, frame_size, window, compression, xid_fallback = self._v42_params
        old = self._v42
        old.close()
        self._v42 = V42Link(
            calling, detect=False, bit_rate=rate, frame_size=frame_size,
            window=window, compression=compression,
            xid_fallback=xid_fallback,
        )
        self._v42_calling = calling
        self._v42_direct = True
        self._v42_bits = 0
        self._v42_detect_deadline = 10 ** 9
        self._v42_establish_deadline = int(self._v42_recovery_timeout_seconds * rate)
        self.v42_mode = "recovering"
        self._v42_recovery_stage = "establish"

    def _v42_recovery_preserve_unsent(self, link) -> bool:
        """Carry only definitely-unsent DTE bytes across peer-initiated DISC."""
        take = getattr(link, "take_release_tx_snapshot", None)
        snapshot = take() if take is not None else None
        if snapshot is None:
            return True
        if snapshot["encoded_pending"]:
            count = int(snapshot["encoded_octets"])
            self._v42_recovery_stats["discarded_encoded_octets"] += count
            link._event(
                f"V.42bis recovery: {count} encoded/unacknowledged octets discarded at DISC; "
                "application retransmission may be required"
            )
        raw = snapshot["raw"]
        if len(self._v42_hold) + len(raw) > self._v42_recovery_hold_limit:
            self._v42_recovery_fail("unsent DTE data exceeded recovery hold limit")
            return False
        if raw:
            # This queue predates bytes accepted while recovery was already
            # active, so prepend it to retain DTE stream order.
            self._v42_hold[:0] = raw
            self._v42_recovery_stats["requeued_dte_bytes"] += len(raw)
            link._event(f"V.42bis recovery: preserved {len(raw)} unsent DTE bytes")
        return True

    def _v42_recovery_fail(self, reason: str) -> None:
        link = self._v42
        if link is not None:
            link._event(f"V.42bis recovery failed: {reason}")
        self._v42_recovery_stats["failures"] += 1
        self._v42_recovery_stats["last_failure"] = reason
        self._v42_hold.clear()
        self._v42_recovery_peer_busy_at = None
        self.v42_mode = "lost"
        self._v42_recovery_stage = "failed"

    def _v42_lenient_adp(self) -> bool:
        link = self._v42
        det = self._v42_adp_detector
        if (det is None or not self._v42_adp_armed or self.v42_mode != "detect"
                or link.adp_seen or self._v42_direct
                or self._v42_bits > self._v42_detect_deadline):
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

        rate, frame_size, window, compression, xid_fallback = self._v42_params
        link.close()
        self._v42 = V42Link(self._v42_calling, detect=False, bit_rate=rate, frame_size=frame_size, window=window,
                            compression=compression, xid_fallback=xid_fallback)
        self._v42_direct = True
        self._v42_bits = 0
        self._v42_recovery_peer_busy_at = None
        self._v42_detect_deadline = 10 ** 9
        self._v42_establish_deadline = int(10.0 * rate)
        return True

    def _v42_arm_adp_detector(self) -> None:
        """Start the byte-level compatibility detector only after a full ODP prefix."""
        if self._v42_adp_detector is None or self._v42_adp_armed:
            return
        from v42_detect import ADP_ARM_BITS, AdpDetector

        if self._v42_bits < ADP_ARM_BITS:
            return

        # Clear any partial character or stale bytes accumulated before the
        # peer could legally have observed the required four ODP characters.
        self._v42_adp_framer = _Framer()
        self._v42_adp_detector = AdpDetector()
        self._v42_adp_armed = True

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
