"""ctypes binding for SpanDSP's V.42 (detection + LAPM error control).

V.42 runs directly on the modem's synchronous bit stream: it generates the
ODP/ADP detection patterns, then HDLC-framed LAPM frames (SABME, I-frames,
RR/REJ, ...). Every timer in it counts *bits*, so ``tx_bit`` must be called
exactly once per transmitted bit and ``rx_bit`` once per received bit.

Application data goes in with ``write()`` and comes out of ``read()``;
V.42 carries it as the octets of I-frames (the async start/stop bits are not
sent on the wire, as the standard specifies for async-to-sync conversion).

``state`` is one of ``detect`` (sending/looking for ODP/ADP), ``lapm``
(error-corrected link up), ``unsupported`` (peer did not support V.42) or
``down`` (link released/failed).

Timers: SpanDSP hard-codes ``tx_bit_rate = 28800`` to convert its timers (T400
750 ms, T401, T403) into bit counts, so on a 2400 bps link they run ~12x slow.
``spandsp-v42-bitrate.patch`` adds ``v42_set_bit_rate()``; pass ``bit_rate`` to
``V42Link`` to use it (``timers_scaled`` says whether the loaded library has it).
"""

from __future__ import annotations

import ctypes
import os
from collections import deque

# span_get_msg_func_t / span_put_msg_func_t / span_modem_status_func_t
_GET_MSG = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
_PUT_MSG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
_STATUS = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)

# lapm states reported through the status callback (v42.c)
LAPM_DETECT = 0
LAPM_IDLE = 1
LAPM_V42_UNSUPPORTED = 8
# SIG_STATUS_* (async.h)
SIG_STATUS_LINK_CONNECTED = -14
SIG_STATUS_LINK_DISCONNECTED = -15
SIG_STATUS_LINK_ERROR = -16

_lib = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(os.environ.get("SPANDSP_LIB", "libspandsp.so.4"))
    lib.v42_init.restype = ctypes.c_void_p
    lib.v42_init.argtypes = [
        ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
        _GET_MSG, _PUT_MSG, ctypes.c_void_p,
    ]
    lib.v42_set_status_callback.restype = None
    lib.v42_set_status_callback.argtypes = [ctypes.c_void_p, _STATUS, ctypes.c_void_p]
    lib.v42_rx_bit.restype = None
    lib.v42_rx_bit.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.v42_tx_bit.restype = ctypes.c_int
    lib.v42_tx_bit.argtypes = [ctypes.c_void_p]
    lib.v42_restart.restype = None
    lib.v42_restart.argtypes = [ctypes.c_void_p]
    # Added by spandsp-v42-bitrate.patch; absent from an unpatched library.
    if hasattr(lib, "v42_set_frame_params"):
        lib.v42_set_frame_params.restype = None
        lib.v42_set_frame_params.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    if hasattr(lib, "v42_set_bit_rate"):
        lib.v42_set_bit_rate.restype = None
        lib.v42_set_bit_rate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    # Added by spandsp-v42bis.patch
    if hasattr(lib, "v42_set_compression"):
        lib.v42_set_compression.restype = None
        lib.v42_set_compression.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.v42_get_compression.restype = ctypes.c_int
        lib.v42_get_compression.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 3
    lib.v42bis_init.restype = ctypes.c_void_p
    lib.v42bis_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                _PUT_MSG, ctypes.c_void_p, ctypes.c_int, _PUT_MSG, ctypes.c_void_p, ctypes.c_int]
    lib.v42bis_compress.restype = ctypes.c_int
    lib.v42bis_compress.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.v42bis_compress_flush.restype = ctypes.c_int
    lib.v42bis_compress_flush.argtypes = [ctypes.c_void_p]
    lib.v42bis_decompress.restype = ctypes.c_int
    lib.v42bis_decompress.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.v42bis_decompress_flush.restype = ctypes.c_int
    lib.v42bis_decompress_flush.argtypes = [ctypes.c_void_p]
    lib.v42bis_free.restype = ctypes.c_int
    lib.v42bis_free.argtypes = [ctypes.c_void_p]
    lib.v42_release.restype = ctypes.c_int
    lib.v42_release.argtypes = [ctypes.c_void_p]
    lib.v42_free.restype = ctypes.c_int
    lib.v42_free.argtypes = [ctypes.c_void_p]
    _lib = lib
    return lib


def available() -> bool:
    try:
        _load()
    except OSError:
        return False
    return True


class V42Link:
    """One end of a V.42 (LAPM) error-controlled link."""

    def __init__(self, calling_party: bool, detect: bool = True, bit_rate: int = 0,
                 frame_size: int = 0, window: int = 0, compression: tuple[int, int, int] | None = None):
        lib = _load()
        self._lib = lib
        self._tx = deque()  # application bytes waiting to be framed
        self._rx = bytearray()  # application bytes received
        self._calling = calling_party
        self._bis = None  # v42bis context once LAPM is up and compression was agreed
        self.compression = None  # (p0, p1, p2) as agreed, or None
        self.compressed_in = self.compressed_out = 0  # bytes of line traffic, for the ratio
        self.state = "detect" if detect else "idle"
        self.events: list[str] = []
        self.bits_sent = 0
        self.adp_seen = False  # peer answered our ODP (or sent ADP to it)

        def get_msg(_user, buf, max_len):
            if not self._tx:
                return 0
            chunk = bytearray()
            while self._tx and len(chunk) < max_len:
                chunk.append(self._tx.popleft())
            ctypes.memmove(buf, bytes(chunk), len(chunk))
            return len(chunk)

        def put_msg(_user, buf, length):
            if buf and length > 0:
                chunk = ctypes.string_at(buf, length)
                if self._bis is not None:
                    self.compressed_in += len(chunk)
                    self._lib.v42bis_decompress(self._bis, chunk, len(chunk))     # -> _bis_rx
                    self._lib.v42bis_decompress_flush(self._bis)
                else:
                    self._rx += chunk
            else:
                self._on_status(length)  # NULL msg: length carries a status

        def status(_user, code):
            self._on_status(code)

        def bis_rx(_user, buf, length):             # decompressed application data
            if buf and length > 0:
                self._rx += ctypes.string_at(buf, length)

        def bis_tx(_user, buf, length):             # compressed bytes for the LAPM I-frames
            if buf and length > 0:
                chunk = ctypes.string_at(buf, length)
                self.compressed_out += len(chunk)
                self._tx.extend(chunk)

        self._bis_cb = (_PUT_MSG(bis_tx), _PUT_MSG(bis_rx))
        self._cb = (_GET_MSG(get_msg), _PUT_MSG(put_msg), _STATUS(status))
        self._state_ptr = lib.v42_init(None, calling_party, detect, self._cb[0], self._cb[1], None)
        if not self._state_ptr:
            raise RuntimeError("v42_init failed")
        lib.v42_set_status_callback(self._state_ptr, self._cb[2], None)
        if compression is not None and hasattr(lib, "v42_set_compression"):
            lib.v42_set_compression(self._state_ptr, *compression)
        # SpanDSP's timers count bits at an assumed 28800 bps; give it the real
        # rate (patched libraries only) so retransmissions are not ~12x late.
        self.timers_scaled = bool(bit_rate) and hasattr(lib, "v42_set_bit_rate")
        if self.timers_scaled:
            lib.v42_set_bit_rate(self._state_ptr, bit_rate)
        # LAPM frame size (N401, octets) and window (k): smaller frames lose
        # less to each damaged frame on a lossy path. Patched libraries only.
        self.frame_params_set = False
        if (frame_size or window) and hasattr(lib, "v42_set_frame_params"):
            lib.v42_set_frame_params(self._state_ptr, frame_size or 128, window or 15)
            self.frame_params_set = True
        # v42_init only sets up the context; v42_restart starts the detection
        # sequence and its timers (this is what SpanDSP's own harness does).
        lib.v42_restart(self._state_ptr)

    def _on_status(self, code: int) -> None:
        if code == SIG_STATUS_LINK_CONNECTED:
            self.state, name = "lapm", "LINK_CONNECTED"
            self._start_compression()
        elif code == LAPM_V42_UNSUPPORTED:
            self.state, name = "unsupported", "V42_UNSUPPORTED"
        elif code in (SIG_STATUS_LINK_DISCONNECTED, SIG_STATUS_LINK_ERROR):
            self.state, name = "down", "LINK_DISCONNECTED" if code == -15 else "LINK_ERROR"
        elif code == LAPM_IDLE:
            # Detection succeeded; LAPM link establishment follows.
            self.adp_seen, name = True, "DETECTED"
        else:
            name = f"status_{code}"
        self.events.append(name)

    def _start_compression(self) -> None:
        lib = self._lib
        if not hasattr(lib, "v42_get_compression") or self._bis is not None:
            return
        p0, p1, p2 = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
        if not lib.v42_get_compression(self._state_ptr, ctypes.byref(p0), ctypes.byref(p1), ctypes.byref(p2)):
            return
        if p0.value == 0 or p1.value < 512 or p2.value < 6:
            return
        # P0: bit 0 = initiator->responder, bit 1 = responder->initiator. SpanDSP's v42bis_init wants it
        # from this end's point of view: bit 1 = we compress, bit 0 = we decompress.
        local = p0.value if not self._calling else ((p0.value & 1) << 1) | ((p0.value >> 1) & 1)
        self._bis = lib.v42bis_init(None, local, p1.value, p2.value, self._bis_cb[0], None, 256,
                                    self._bis_cb[1], None, 1024)
        self.compression = (p0.value, p1.value, p2.value) if self._bis else None

    # -- line side (one call per bit) --------------------------------------
    def tx_bit(self) -> int:
        self.bits_sent += 1
        return self._lib.v42_tx_bit(self._state_ptr)

    def rx_bit(self, bit: int) -> None:
        self._lib.v42_rx_bit(self._state_ptr, 1 if bit else 0)

    # -- application side --------------------------------------------------
    def write(self, data: bytes) -> None:
        if self._bis is not None and data:
            self._lib.v42bis_compress(self._bis, bytes(data), len(data))        # -> bis_tx
            self._lib.v42bis_compress_flush(self._bis)
        else:
            self._tx.extend(data)

    def read(self) -> bytes:
        data = bytes(self._rx)
        self._rx.clear()
        return data

    def pending(self) -> int:
        """Application bytes accepted by write() but not yet framed."""
        return len(self._tx)

    def close(self) -> None:
        if self._state_ptr:
            self._lib.v42_release(self._state_ptr)
            self._lib.v42_free(self._state_ptr)
            self._state_ptr = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
