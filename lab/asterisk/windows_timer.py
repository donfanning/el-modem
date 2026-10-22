"""Scoped, opt-in Windows timer-resolution request for RTP pacing tests."""
from __future__ import annotations

import ctypes
import os


class TimerResolutionRequest:
    """Pair WinMM timeBeginPeriod/timeEndPeriod within one active call.

    The request is process-scoped on current Windows versions. Keep it off by
    default and hold it only while a real-time audio session is active.
    """

    def __init__(self, milliseconds: int, *, platform: str | None = None,
                 library_loader=None):
        if isinstance(milliseconds, bool) or not isinstance(milliseconds, int):
            raise ValueError("timer resolution must be an integer number of milliseconds")
        if not 1 <= milliseconds <= 15:
            raise ValueError("timer resolution must be between 1 and 15 milliseconds")
        self.milliseconds = milliseconds
        self.platform = os.name if platform is None else platform
        self._library_loader = library_loader
        self._winmm = None
        self.status = "not_started"
        self.begin_result = None
        self.end_result = None

    @classmethod
    def from_environment(cls, environ=None, **kwargs):
        environ = os.environ if environ is None else environ
        raw = environ.get("EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS")
        if raw is None or raw == "":
            return None
        try:
            milliseconds = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS must be an integer") from exc
        return cls(milliseconds, **kwargs)

    def start(self) -> None:
        if self.status != "not_started":
            raise RuntimeError("timer-resolution request already started")
        if self.platform != "nt":
            self.status = "unsupported_platform"
            return
        loader = self._library_loader or ctypes.WinDLL
        winmm = loader("winmm")
        begin = winmm.timeBeginPeriod
        end = winmm.timeEndPeriod
        begin.argtypes = [ctypes.c_uint]
        begin.restype = ctypes.c_uint
        end.argtypes = [ctypes.c_uint]
        end.restype = ctypes.c_uint
        result = int(begin(self.milliseconds))
        self.begin_result = result
        if result != 0:
            self.status = "request_failed"
            raise OSError(result, "WinMM timeBeginPeriod failed")
        self._winmm = winmm
        self.status = "active"

    def close(self) -> None:
        if self.status != "active":
            return
        result = int(self._winmm.timeEndPeriod(self.milliseconds))
        self.end_result = result
        self.status = "released" if result == 0 else "release_failed"
        self._winmm = None

    def report(self) -> dict:
        return {
            "period_ms": self.milliseconds,
            "platform": self.platform,
            "status": self.status,
            "begin_result": self.begin_result,
            "end_result": self.end_result,
        }
