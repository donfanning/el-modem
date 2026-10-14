"""ctypes binding for SpanDSP's V.8 (modem start-up negotiation, ITU-T V.8).

Used by the V.32 caller: a modern answerer opens with an ANSam tone and waits for the caller's CM
(call menu); we answer it listing the modulations we can do (V.32/V.32bis), it replies JM, we send CJ,
and the negotiated modulation's own start-up follows. ``V8Negotiator`` is one end of that exchange:
feed received samples with ``rx()``, take samples to transmit from ``tx()``, and watch ``status``.
"""

from __future__ import annotations

import ctypes
import os

V8_CALL_V_SERIES = 6
V8_MOD_V21, V8_MOD_V22, V8_MOD_V23, V8_MOD_V32 = 1 << 1, 1 << 2, 1 << 4, 1 << 9
V8_MOD_V34, V8_MOD_V90 = 1 << 11, 1 << 12
V8_PROTOCOL_NONE, V8_PROTOCOL_LAPM_V42 = 0, 1
MODEM_CONNECT_TONES_NONE, MODEM_CONNECT_TONES_ANSAM_PR = 0, 5

STATUS_NAMES = {0: "in-progress", 1: "v8-offered", 2: "v8-call", 3: "non-v8-call", 4: "failed",
                5: "call-function-received", 6: "calling-tone", 7: "fax-cng"}


class _CmJm(ctypes.Structure):
    _fields_ = [("call_function", ctypes.c_int32), ("modulations", ctypes.c_uint32),
                ("protocols", ctypes.c_int32), ("pstn_access", ctypes.c_int32),
                ("nsf", ctypes.c_int32), ("pcm_modem_availability", ctypes.c_int32),
                ("t66", ctypes.c_int32)]


class _Parms(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("gateway_mode", ctypes.c_bool),
                ("modem_connect_tone", ctypes.c_int32), ("send_ci", ctypes.c_int32),
                ("v92", ctypes.c_int32), ("jm_cm", _CmJm)]


_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(_Parms))
_lib = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get("SPANDSP_LIB", "libspandsp.so.4"))
        lib.v8_init.restype = ctypes.c_void_p
        lib.v8_init.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.POINTER(_Parms), _HANDLER, ctypes.c_void_p]
        lib.v8_tx.restype = ctypes.c_int
        lib.v8_tx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
        lib.v8_rx.restype = ctypes.c_int
        lib.v8_rx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
        lib.v8_release.restype = ctypes.c_int
        lib.v8_release.argtypes = [ctypes.c_void_p]
        lib.v8_free.restype = ctypes.c_int
        lib.v8_free.argtypes = [ctypes.c_void_p]
        _lib = lib
    return _lib


def available() -> bool:
    try:
        _load()
    except OSError:
        return False
    return True


class V8Negotiator:
    def __init__(self, calling_party: bool, modulations: int = V8_MOD_V32,
                 protocols: int = V8_PROTOCOL_NONE, send_ci: bool = False):
        lib = _load()
        self._lib = lib
        parms = _Parms()
        parms.gateway_mode = False
        parms.modem_connect_tone = MODEM_CONNECT_TONES_NONE if calling_party else MODEM_CONNECT_TONES_ANSAM_PR
        parms.send_ci = 1 if send_ci else 0
        parms.v92 = -1
        parms.jm_cm.call_function = V8_CALL_V_SERIES
        parms.jm_cm.modulations = modulations
        parms.jm_cm.protocols = protocols
        parms.jm_cm.pcm_modem_availability = 0
        parms.jm_cm.pstn_access = 0
        parms.jm_cm.nsf = -1
        parms.jm_cm.t66 = -1
        self._parms = parms
        self.status = 0
        self.result: dict | None = None

        def handler(_user, result):
            r = result.contents
            self.status = r.status
            self.result = {"status": r.status, "call_function": r.jm_cm.call_function,
                           "modulations": r.jm_cm.modulations, "protocols": r.jm_cm.protocols,
                           "tone": r.modem_connect_tone}

        self._cb = _HANDLER(handler)
        self._ptr = lib.v8_init(None, calling_party, ctypes.byref(parms), self._cb, None)
        if not self._ptr:
            raise RuntimeError("v8_init failed")

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, str(self.status))

    def tx(self, count: int) -> list[int]:
        buf = (ctypes.c_int16 * count)()
        got = self._lib.v8_tx(self._ptr, buf, count)
        out = list(buf[:got])
        return out + [0] * (count - got)

    def rx(self, samples) -> int:
        n = len(samples)
        buf = (ctypes.c_int16 * n)(*samples)
        return self._lib.v8_rx(self._ptr, buf, n)

    def close(self) -> None:
        if self._ptr:
            self._lib.v8_free(self._ptr)         # v8_free also releases the tx queue
            self._ptr = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
