"""ctypes binding for SpanDSP's V.22/V.22bis/Bell 212A modem.

This is a test/reference binding to SpanDSP (LGPL-2.1), not the active
product modem engine. ``bell212.py`` remains a separate lab fixture.

SpanDSP's modem moves a *synchronous* bit stream and performs the V.22 /
Bell 212A handshake (calling/answering sequences, scrambler, training)
internally. The DTE side is asynchronous 8N1 bytes, so this wrapper does the
async<->sync conversion the way V.22 specifies for basic-rate async data:
characters travel as start/8 data/stop bit groups in the sync stream, and the
sender may delete an occasional stop bit to absorb a fast DTE clock (the
receiver here accepts a start bit directly after the data bits).

Requires ``libspandsp.so.4`` (set SPANDSP_LIB to override the path).
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections import deque

SIG_STATUS_NAMES = {
    -1: "CARRIER_DOWN",
    -2: "CARRIER_UP",
    -3: "TRAINING_IN_PROGRESS",
    -4: "TRAINING_SUCCEEDED",
    -5: "TRAINING_FAILED",
    -6: "FRAMING_OK",
    -7: "END_OF_DATA",
    -8: "ABORT",
    -9: "BREAK",
    -10: "SHUTDOWN_COMPLETE",
    -11: "OCTET_REPORT",
    -12: "POOR_SIGNAL_QUALITY",
    -13: "MODEM_RETRAIN_OCCURRED",
    -14: "LINK_CONNECTED",
    -15: "LINK_DISCONNECTED",
    -16: "LINK_ERROR",
    -17: "LINK_IDLE",
}
TRAINING_SUCCEEDED = -4
CARRIER_UP = -2
CARRIER_DOWN = -1

V22BIS_GUARD_TONE_NONE = 0
V22BIS_GUARD_TONE_550HZ = 1
V22BIS_GUARD_TONE_1800HZ = 2
# This option is declared by the pinned SpanDSP headers but is not consumed by
# its v22bis_tx.c/v22bis_rx.c implementation. Passing it does not create an
# independent Bell 212A handshake/reference in this build.
V22BIS_BELL212A_COMPATIBILITY_MODE = 0x100
V22BIS_USE_U00 = 0x200

_GET_BIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_PUT_BIT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
_STATUS = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)

_lib = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("SPANDSP_LIB", "libspandsp.so.4")
    lib = ctypes.CDLL(path)
    lib.v22bis_init.restype = ctypes.c_void_p
    lib.v22bis_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
        _GET_BIT, ctypes.c_void_p, _PUT_BIT, ctypes.c_void_p,
    ]
    lib.v22bis_rx.restype = ctypes.c_int
    lib.v22bis_rx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.v22bis_rx_fillin.restype = ctypes.c_int
    lib.v22bis_rx_fillin.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.v22bis_tx.restype = ctypes.c_int
    lib.v22bis_tx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.v22bis_tx_power.restype = None
    lib.v22bis_tx_power.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.v22bis_get_current_bit_rate.restype = ctypes.c_int
    lib.v22bis_get_current_bit_rate.argtypes = [ctypes.c_void_p]
    lib.v22bis_rx_signal_power.restype = ctypes.c_float
    lib.v22bis_rx_signal_power.argtypes = [ctypes.c_void_p]
    lib.v22bis_restart.restype = ctypes.c_int
    lib.v22bis_restart.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.v22bis_set_modem_status_handler.restype = None
    lib.v22bis_set_modem_status_handler.argtypes = [
        ctypes.c_void_p, _STATUS, ctypes.c_void_p,
    ]
    lib.v22bis_release.restype = ctypes.c_int
    lib.v22bis_release.argtypes = [ctypes.c_void_p]
    lib.v22bis_free.restype = ctypes.c_int
    lib.v22bis_free.argtypes = [ctypes.c_void_p]
    _lib = lib
    return lib


def available() -> bool:
    try:
        _load()
    except OSError:
        return False
    return True


class _AsyncFramer:
    """Sync bit stream -> async 8N1 bytes, tolerating deleted stop bits."""

    def __init__(self) -> None:
        self._state = "idle"
        self._count = 0
        self._value = 0

    def push(self, bit: int) -> list[int]:
        bit = 1 if bit else 0
        out: list[int] = []
        if self._state == "idle":
            if bit == 0:  # start bit; the idle line is all ones
                self._state, self._count, self._value = "data", 0, 0
        elif self._state == "data":
            self._value |= bit << self._count
            self._count += 1
            if self._count == 8:
                self._state = "stop"
        else:  # stop
            out.append(self._value)
            if bit == 1:
                self._state = "idle"
            else:
                # Stop bit deleted by the sender (allowed by V.22 to absorb a
                # fast DTE clock): this 0 is already the next start bit.
                self._state, self._count, self._value = "data", 0, 0
        return out


class V22bisModem:
    """One end of a SpanDSP V.22/V.22bis connection.

    ``tx_samples``/``feed_pcm`` work on 8 kHz signed 16-bit linear audio.
    ``events`` collects the modem's status changes as (name, status) pairs.
    """

    def __init__(
        self,
        calling_party: bool,
        bit_rate: int = 1200,
        bell212a: bool = False,
        guard_tone: int = V22BIS_GUARD_TONE_NONE,
        tx_power_dbm0: float = -10.0,
    ):
        lib = _load()
        self._lib = lib
        # One lock for every call into the C engines. SpanDSP's V.42 keeps a
        # single state shared by its transmit and receive sides (acks received
        # rewrite the send window the transmitter is reading, a received frame
        # can pull queued data to piggyback an ack), while the session drives
        # transmit from the RTP send thread and receive from the main thread --
        # and ctypes drops the GIL during C calls, so they truly run at once.
        # Unlocked, that corrupted memory (segfaults in a stress test, and in
        # the lab a whole LAPM frame silently vanishing). Reentrant, because
        # the engines call back into Python (get_bit/put_bit) on the same thread.
        self._lock = threading.RLock()
        self._tx_bits: deque[int] = deque()
        self._framer = _AsyncFramer()
        self._rx_bytes = bytearray()
        self.events: list[tuple[str, int]] = []
        self.connected = False
        self._rx_samples = 0
        self._connected_at = 0
        self.target_rate = bit_rate
        # V.42 (LAPM) error control over the same bit stream; see start_v42().
        self._v42 = None
        self.v42_mode = "off"  # off | detect | lapm | failed | lost
        self._v42_bits = 0
        self._v42_detect_deadline = 0
        self._v42_establish_deadline = 0
        self._app_hold = bytearray()
        # Lenient ADP detection (see _adp_lenient_check): the bits received
        # during V.42 detection are also decoded as plain async characters.
        self._adp_framer = _AsyncFramer()
        self._adp_detector = None
        self._adp_armed = False
        self._v42_direct = False  # peer's ADP accepted by us, not by SpanDSP
        self._v42_calling = False
        self.capture_detect_bits = bool(os.environ.get("MODEM_V42_CAPTURE"))
        self.detect_rx_bits: list[int] = []

        def get_bit(_user):
            if self.v42_mode == "lost":
                return 1  # error-control link is dead: send only idle marks
            if self.v42_mode in ("detect", "lapm"):
                bit = self._v42.tx_bit()
                self._v42_bits += 1
                self._arm_adp_detector()
                self._v42_check()
                return bit
            return self._tx_bits.popleft() if self._tx_bits else 1

        def put_bit(_user, bit):
            if bit >= 0:
                if self.v42_mode == "lost":
                    pass
                elif self.v42_mode in ("detect", "lapm"):
                    if (self.capture_detect_bits and self.v42_mode == "detect"
                            and len(self.detect_rx_bits) < 6000):
                        self.detect_rx_bits.append(bit)
                    if (self.v42_mode == "detect" and self._adp_detector is not None
                            and self._adp_armed):
                        got = bytes(self._adp_framer.push(bit))
                        if got:
                            self._adp_detector.feed(got)
                    self._v42.rx_bit(bit)
                else:
                    for value in self._framer.push(bit):
                        self._rx_bytes.append(value)
            else:
                self._on_status(bit)

        def status(_user, code):
            self._on_status(code)

        # Keep references: ctypes callbacks are freed if garbage collected.
        self._cb = (_GET_BIT(get_bit), _PUT_BIT(put_bit), _STATUS(status))
        options = guard_tone | (V22BIS_BELL212A_COMPATIBILITY_MODE if bell212a else 0)
        self._state = lib.v22bis_init(
            None, bit_rate, options, calling_party,
            self._cb[0], None, self._cb[1], None,
        )
        if not self._state:
            raise RuntimeError("v22bis_init failed")
        lib.v22bis_set_modem_status_handler(self._state, self._cb[2], None)
        lib.v22bis_tx_power(self._state, ctypes.c_float(tx_power_dbm0))

    def _on_status(self, code: int) -> None:
        self.events.append((SIG_STATUS_NAMES.get(code, str(code)), code))
        if code == TRAINING_SUCCEEDED:
            self.connected = True
            self._connected_at = self._rx_samples
        elif code in (CARRIER_DOWN,):
            self.connected = False

    @property
    def connected_for_seconds(self) -> float:
        """Audio time since training last succeeded (0 if not connected).

        Counted in received samples, so it is deterministic in tests and
        real time alike. Used to debounce a training "success" that a
        tone or noise can produce before the real handshake."""
        if not self.connected:
            return 0.0
        return (self._rx_samples - self._connected_at) / 8000.0

    # A V.22bis link that wants 2400 first trains at 1200, drops carrier and
    # retrains at 2400 (seen live: ~2.4 s apart). A connection still at 1200
    # is therefore provisional until this much time passes with no retrain.
    PROVISIONAL_SECONDS = 4.0
    STABLE_SECONDS = 0.5

    def is_settled(self) -> bool:
        """True once the line is trained and the final rate is known."""
        wait = (
            self.STABLE_SECONDS
            if self.bit_rate >= self.target_rate
            else self.PROVISIONAL_SECONDS
        )
        return self.connected and self.connected_for_seconds >= wait

    @property
    def bit_rate(self) -> int:
        return self._lib.v22bis_get_current_bit_rate(self._state)

    @property
    def rx_signal_power_dbm0(self) -> float:
        return float(self._lib.v22bis_rx_signal_power(self._state))

    # -- V.42 / LAPM error control ----------------------------------------
    def start_v42(self, *args, **kwargs) -> None:
        with self._lock:
            self._start_v42(*args, **kwargs)

    def _start_v42(self, calling_party: bool, detect_seconds: float = 0.75,
                   establish_seconds: float = 10.0, frame_size: int = 0,
                   window: int = 0, compression: tuple[int, int, int] | None = None,
                   protocol: str = "lapm", mnp_level: int = 4) -> None:
        """Run V.42 detection (and LAPM if the peer supports it) on the bit
        stream. User data is held until the outcome is known; on failure the
        modem falls back to raw async data. Deadlines are counted in bits at
        the current line rate. ``detect_seconds`` defaults to V.42's 750 ms
        T400. Fallback is deferred until the next ODP mark gap, so expiry does
        not switch to HDLC flags in the middle of an ODP character and leave
        the peer a stray byte."""
        from v42_lapm import V42Link  # noqa: PLC0415

        from v42_detect import AdpDetector  # noqa: PLC0415

        rate = max(self.bit_rate, 300)
        self._v42_params = (rate, frame_size, window, compression)
        self._v42_calling = calling_party
        self._v42_direct = False
        self._adp_framer = _AsyncFramer()
        # Only a caller looks for the peer's ADP; an answerer sends it.
        self._adp_detector = AdpDetector() if calling_party else None
        self._adp_armed = False
        if protocol == "mnp":
            # V.42 Annex A (MNP 2-4 class, mnp_link.py): no detection phase, the originator just sends its LR
            from mnp_link import MNP_LEVELS, MnpLink  # noqa: PLC0415

            self._adp_detector = None
            self._v42 = MnpLink(calling_party, bit_rate=rate, **MNP_LEVELS[mnp_level])
            self._v42_direct = True
            self._v42_bits = 0
            self._v42_detect_deadline = 10 ** 9
            self._v42_establish_deadline = int((detect_seconds + establish_seconds) * rate)
            self.v42_mode = "detect"
            return
        self._v42 = V42Link(calling_party, detect=True, bit_rate=rate,
                            frame_size=frame_size, window=window, compression=compression)
        self._v42_bits = 0
        self._v42_detect_deadline = int(detect_seconds * rate)
        self._v42_establish_deadline = int((detect_seconds + establish_seconds) * rate)
        self.v42_mode = "detect"

    def _v42_check(self) -> None:
        link = self._v42
        state = link.state
        if self.v42_mode == "lapm" and state in ("down", "unsupported"):
            # An established error-controlled link has failed (retries
            # exhausted, e.g. during a long retrain). The peer is still in
            # LAPM, so falling back to raw data would corrupt the stream
            # silently; report the loss so the call is dropped, as a real
            # modem does.
            self.v42_mode = "lost"
            return
        if state == "lapm":
            self.v42_mode = "lapm"
            if self._app_hold:
                link.write(bytes(self._app_hold))
                self._app_hold.clear()
        elif self._adp_lenient_check():
            return
        elif state in ("unsupported", "down") or (
            state in ("detect", "idle")
            and (
                (not link.adp_seen and self._v42_bits > self._v42_detect_deadline
                 and self._odp_in_gap())
                or self._v42_bits > self._v42_establish_deadline
            )
        ):
            self._v42_fallback()

    def _adp_lenient_check(self) -> bool:
        """Accept a peer's ADP that SpanDSP's strict detector rejects.

        SpanDSP requires 8-16 mark bits between the ADP characters. A real
        peer (NIST's modem) sends E and C back to back, so SpanDSP never saw an
        ADP and we gave up, whereupon the peer -- already waiting for LAPM --
        hung up. A plain async decode of the same bits does see ``E C E C``.
        On that, abandon SpanDSP's detection (inside a gap in our ODP, so the
        pattern is not cut mid-character) and go straight to LAPM, which is what
        a V.42 originator does after the ADP anyway. Returns True if the mode
        changed this call.
        """
        link = self._v42
        det = self._adp_detector
        if (det is None or not self._adp_armed or self.v42_mode != "detect"
                or link.adp_seen or self._v42_direct
                or self._v42_bits > self._v42_detect_deadline):
            return False
        verdict = det.result
        if verdict == "no_v42":
            # The peer says it does not support V.42: stop sending ODP.
            if self._odp_in_gap():
                self._v42_fallback()
                return True
            return False
        if verdict != "v42" or not self._odp_in_gap():
            return False
        from v42_lapm import V42Link  # noqa: PLC0415

        rate, frame_size, window, compression = self._v42_params
        link.close()
        self._v42 = V42Link(self._v42_calling, detect=False, bit_rate=rate,
                            frame_size=frame_size, window=window, compression=compression)
        self._v42_direct = True
        self._v42_bits = 0
        # The ADP is already accepted, so there is no detection deadline any
        # more; LAPM establishment gets the full allowance from here.
        self._v42_detect_deadline = 10 ** 9
        self._v42_establish_deadline = int(10.0 * rate)
        return True

    def _arm_adp_detector(self) -> None:
        """Arm the lenient detector only after the minimum valid ODP prefix."""
        if self._adp_detector is None or self._adp_armed:
            return
        from v42_detect import ADP_ARM_BITS, AdpDetector

        if self._v42_bits < ADP_ARM_BITS:
            return
        self._adp_framer = _AsyncFramer()
        self._adp_detector = AdpDetector()
        self._adp_armed = True

    def _odp_in_gap(self) -> bool:
        """True when the ODP being sent is between characters.

        The ODP repeats in 36-bit blocks (char, 8 ones, char, 8 ones; each char
        is 10 bits), starting at the first bit sent. Abandoning V.42 inside a
        run of ones lets the raw data start cleanly; cutting mid-character
        would hand the peer a stray byte."""
        if self._v42_direct:
            return True
        p = self._v42_bits % 36
        return 10 <= p < 18 or 28 <= p < 36

    def _v42_fallback(self) -> None:
        """No error control: carry on with raw async data."""
        self.v42_mode = "failed"
        # The ODP may have been cut off mid-character; a run of idle marks
        # lets the peer's async receiver resynchronise before any raw data.
        self._tx_bits.extend([1] * 24)
        held, self._app_hold = bytes(self._app_hold), bytearray()
        if held:
            self.feed_bytes(held)

    def detect_bits_rle(self, limit: int = 120) -> str:
        """Run-length view of the bits received during V.42 detection, e.g.
        ``0x1 1x1 0x1 ...`` -- for diagnosing a peer's ODP/ADP spacing."""
        out, bits = [], self.detect_rx_bits
        i = 0
        while i < len(bits) and len(out) < limit:
            j = i
            while j < len(bits) and bits[j] == bits[i]:
                j += 1
            out.append(f"{bits[i]}x{j - i}")
            i = j
        return " ".join(out)

    @property
    def v42_compression(self):
        """(p0, p1, p2) when V.42bis was agreed on the established link, else None."""
        link = getattr(self, "_v42", None)
        return link.compression if link is not None and self.v42_mode == "lapm" else None

    @property
    def v42_resolved(self) -> bool:
        return self.v42_mode in ("off", "lapm", "failed", "lost")

    @property
    def v42_lost(self) -> bool:
        """True if an established LAPM link failed (the call should be dropped)."""
        return self.v42_mode == "lost"

    def feed_bytes(self, data: bytes) -> None:
        with self._lock:
            self._feed_bytes(data)

    def _feed_bytes(self, data: bytes) -> None:
        if self.v42_mode == "lost":
            return
        if self.v42_mode == "detect":
            self._app_hold += data
            return
        if self.v42_mode == "lapm":
            self._v42.write(data)
            return
        for value in data:
            self._tx_bits.append(0)  # start
            self._tx_bits.extend((value >> i) & 1 for i in range(8))
            self._tx_bits.append(1)  # stop

    def feed_bits(self, bits) -> None:
        """Queue raw line bits (already framed) for transmission.

        Used for patterns whose exact inter-character spacing matters, such as
        the V.42 detection ODP/ADP, which a byte stream cannot express."""
        with self._lock:
            self._tx_bits.extend(1 if b else 0 for b in bits)

    def pending_bits(self) -> int:
        with self._lock:
            return self._pending_bits()

    def _pending_bits(self) -> int:
        extra = 0
        if self.v42_mode == "lapm":
            extra = self._v42.pending() * 10
        elif self.v42_mode == "detect":
            extra = len(self._app_hold) * 10
        return len(self._tx_bits) + extra

    def tx_samples(self, count: int) -> list[int]:
        with self._lock:
            return self._tx_samples(count)

    def _tx_samples(self, count: int) -> list[int]:
        buf = (ctypes.c_int16 * count)()
        got = self._lib.v22bis_tx(self._state, buf, count)
        samples = list(buf[:got])
        if got < count:
            samples.extend([0] * (count - got))
        return samples

    def feed_pcm(self, pcm: bytes) -> bytes:
        with self._lock:
            return self._feed_pcm(pcm)

    def _feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        count = len(pcm) // 2
        self._rx_samples += count
        buf = (ctypes.c_int16 * count).from_buffer_copy(pcm)
        self._lib.v22bis_rx(self._state, buf, count)
        data = bytes(self._rx_bytes)
        self._rx_bytes.clear()
        if self._v42 is not None and self.v42_mode == "lapm":
            data += self._v42.read()
        return data

    def fill_missing(self, count: int) -> None:
        """Tell the receiver ``count`` samples of audio were lost.

        A lost packet delivered as silence looks like the carrier vanishing,
        so the modem drops carrier and retrains (seconds of no data). The
        library's fill-in keeps carrier and symbol timing running across the
        gap, so the damage is a few bit errors that error control repairs."""
        with self._lock:
            self._rx_samples += count
            self._lib.v22bis_rx_fillin(self._state, count)

    def close(self) -> None:
        with self._lock:
            if self._state:
                self._lib.v22bis_release(self._state)
                self._lib.v22bis_free(self._state)
                self._state = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
