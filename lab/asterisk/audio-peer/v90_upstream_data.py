"""Streaming V.90 analogue-upstream B1 qualification and V.34 data decoding.

The media/training receiver supplies ordered phase-rotated equalizer outputs
with absolute symbol coordinates. E/B1 boundary detection remains the
session's responsibility; this stage refuses discontinuous coordinates and
does not emit payload until production V.34 B1 qualification succeeds.
"""
from __future__ import annotations

import os
import numpy as np
import v34_datarx
import v34_receive


def warmup_v90_receiver_power() -> tuple[tuple[int, int], ...]:
    """Precompute V.34 zero-precoder power estimates before call media starts.

    The answerer's selected analogue upstream rate is not known until CP, but
    its baud and rate candidates come from the fixed V.34 geometry. The
    estimates are deterministic and cached by DataTx; warming the supported
    3200/3429 baud combinations here keeps their first-use simulation off the
    RTP receive callback. This is called only for the opt-in V.90 profile.
    """
    from v34_data import DataTx, frame_params, mapping_params
    from v34_negotiation import supported
    from v34_b1_reference import standard_sequences
    from v34_tx import SYMBOL_RATES
    try:
        from v34_viterbi import _trellis_geometry
    except ImportError:
        _trellis_geometry = None
    try:
        from v34_viterbi import _native_trellis_layouts
    except ImportError:
        _native_trellis_layouts = None

    warmed = []
    for baud in (SYMBOL_RATES[4], SYMBOL_RATES[5]):
        for rate in range(4800, 33601, 2400):
            if supported(rate, baud, shaping=False):
                DataTx._estimate_power(rate, baud, False, (0j, 0j, 0j))
                # CP selects the analogue-upstream rate during the media
                # callback. Prebuild the deterministic known B1 sequence now
                # so Receiver construction does not execute the independent
                # shell mapper while handling a 20 ms RTP frame.
                standard_sequences(rate, baud, "answer", False,
                                   (0, 0, 0, 0, 0, 0), (0, 0, 0, 0))
                framing = frame_params(rate, baud)
                mapping = mapping_params(framing["b"], False)
                if _trellis_geometry is not None:
                    _trellis_geometry(mapping["L"])
                if (_native_trellis_layouts is not None
                        and os.environ.get("V34_NATIVE_TRELLIS") == "1"):
                    _native_trellis_layouts(mapping["L"])
                warmed.append((baud, rate))
    return tuple(warmed)


class V90UpstreamDataReceiver:
    WAIT_B1 = "wait-b1"
    QUALIFYING = "qualifying-b1"
    DATA = "data"
    FAILED = "failed"
    def __init__(self, profile, *, remote_role: str = "answer"):
        if remote_role not in ("call", "answer"):
            raise ValueError("remote V.34 role must be call or answer")
        self.profile = profile
        self.remote_role = remote_role
        self.receiver = v34_receive.Receiver(profile, remote_role)
        self.state = self.WAIT_B1
        self.failure_reason: str | None = None
        self.b1_start_symbol: int | None = None
        self.next_symbol: int | None = None
        self._b1: list[complex] = []
        self._data_symbols: list[complex] = []
        self._mapper_points: list[complex] = []
        self._descrambler_history: list[int] = []
        self._gain = 1 + 0j
        # The B1 fit is qualification-only in the curated default path. Keep
        # this explicit None so the phase-3 session can preserve the fitted
        # upstream equalizer while applying the measured B1 gain.
        self.equalizer_taps: np.ndarray | None = None
        self.events: list[tuple[str, object]] = []

    @property
    def b1_symbols_required(self) -> int:
        return len(self.receiver.linear)

    def start_b1(self, symbol_index: int) -> None:
        if self.state != self.WAIT_B1:
            raise RuntimeError("V.90 upstream B1 boundary was already set")
        if isinstance(symbol_index, bool) or not isinstance(symbol_index, int) or symbol_index < 0:
            raise ValueError("B1 symbol coordinate must be a nonnegative integer")
        self.b1_start_symbol = symbol_index
        self.next_symbol = symbol_index
        self.state = self.QUALIFYING
        self.events.append(("b1-start", symbol_index))

    def feed_symbol(self, symbol_index: int, observed: complex) -> list[int]:
        """Consume one equalizer output; return newly decoded, descrambled bits."""
        if isinstance(symbol_index, bool) or not isinstance(symbol_index, int):
            raise ValueError("symbol coordinate must be an integer")
        value = complex(observed)
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            return self._fail("non-finite equalizer observation")
        if self.state == self.FAILED:
            return []
        if self.state == self.WAIT_B1:
            return []
        assert self.next_symbol is not None
        if symbol_index < self.next_symbol:
            return []
        if symbol_index != self.next_symbol:
            return self._fail(
                f"upstream symbol gap: expected {self.next_symbol}, got {symbol_index}")
        self.next_symbol += 1

        if self.state == self.QUALIFYING:
            self._b1.append(value)
            if len(self._b1) < self.b1_symbols_required:
                return []
            try:
                observed_b1 = np.asarray(self._b1, dtype=np.complex128)
                self._gain, _ = self.receiver.qualify(observed_b1)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                return self._fail(f"B1 qualification raised {type(exc).__name__}: {exc}")
            if not self.receiver.diagnostics.get("qualified"):
                return self._fail("V.34 B1 receive qualification failed")
            self._b1.clear()
            self._descrambler_history = list(self.receiver.bits[-23:])
            self.state = self.DATA
            self.events.append(("b1-qualified", dict(self.receiver.diagnostics)))
            return []

        return self._feed_data_symbol(value / self._gain)

    def _feed_data_symbol(self, value: complex) -> list[int]:
        self._data_symbols.append(value)
        if len(self._data_symbols) < 2:
            return []
        self.receiver.vit.feed(self._data_symbols[0], self._data_symbols[1])
        self._data_symbols.clear()
        while self.receiver.vit.decided:
            pair = self.receiver.vit.decided.pop(0)
            self._mapper_points.extend(pair)
        output: list[int] = []
        while len(self._mapper_points) >= 8:
            points = self._mapper_points[:8]
            del self._mapper_points[:8]
            scrambled, _ = self.receiver.drx.decode_frame(points)
            output.extend(v34_datarx.descramble_stream(
                scrambled, self._descrambler_history, self.remote_role))
            self._descrambler_history = (self._descrambler_history + scrambled)[-23:]
        return output

    def _fail(self, reason: str) -> list[int]:
        self.state = self.FAILED
        self.failure_reason = reason
        self.events.append(("failed", reason))
        return []
