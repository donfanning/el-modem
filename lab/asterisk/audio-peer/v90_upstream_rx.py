"""V.90 analogue-modem upstream Phase 3 symbol and Ja receive path.

The caller must acquire S/S-bar/PP and fit the equalizer first, using
``Phase3Rx.acquire`` and ``train_trn_auto``. This tracker then follows the
known-one TRN, detects its transition to Ja without looking for V.34's J
marker, descrambles the subsequent 4-point symbols, and incrementally parses
Ja descriptors. It is a DSP component; RTP buffering and media-loss recovery
remain the call controller's responsibility.
"""

from __future__ import annotations

from collections import deque
import math
import os
import time

import numpy as np

import v34_rx
import v34_tx
from v90_ja import V90JaScanner


class V90UpstreamPhase3Receiver:
    """Track an already-acquired/trained analogue-side Phase 3 waveform."""

    PREAMBLE_SYMBOLS = 128 + 16 + 288
    MIN_TRN_SYMBOLS = 512
    JA_BOUNDARY_SEARCH_INTERVAL = 32

    def __init__(self, frontend: v34_rx.Phase3Rx, trn_scrambler_role: str,
                 *, equalizer_mu: float | None = None):
        if trn_scrambler_role not in ("answer", "call"):
            raise ValueError("TRN scrambler role must be 'answer' or 'call'")
        if frontend.taps is None:
            raise ValueError("Phase3Rx must be trained before starting the V.90 tracker")
        self.frontend = frontend
        self.role = trn_scrambler_role
        # ``equalizer_mu`` is the LMS step of the T/2 equalizer. None keeps the
        # V.34 default (environment ``V34_MU`` or 0.2); a V.90 profile may set it
        # without touching the V.34 receivers, which read the same variable.
        self.eq = v34_rx.SymbolEq(frontend, self.PREAMBLE_SYMBOLS, mu=equalizer_mu)
        # Collect a signal-power reference during analogue training, but keep
        # this behavior scoped to V.90 rather than enabling the V.34 option.
        self.eq._input_guard = v34_rx.InputPowerGuard("silence")
        self.scrambler = v34_tx.Scrambler(self.role)
        self.indices: list[int] = []
        self._index_storage = np.empty(0, dtype=np.int64)
        self.quarters: list[int] = []
        self._equalized_pending: deque[tuple[int, complex, tuple[int, int], float | None]] = deque()
        self.input_powers: list[float | None] = []
        self.mismatch: list[int] = []
        self.trn_end: int | None = None
        self.scanner = V90JaScanner()
        self._symbol_cursor = 0
        self._decision_cursor = 0
        self._phase4_active = False
        self._phase4_constellation = 4
        self._data_decider = None
        self._symbol_callback = None
        self._sq: float | None = None
        self._last: complex = 0j
        self._ja_search_active = False
        self._ja_last_search_at = 0
        self._ja_candidate_boundary: int | None = None
        self._ja_symbol_cursor = 0
        self._stage_timing_enabled = os.environ.get("V90_RX_STAGE_TIMING") == "1"
        self._stage_timing_rows: deque[dict[str, int]] = deque(maxlen=2048)

    def _append_rotation(self, rotation: int) -> None:
        """Append a symbol decision to the public list and geometric array."""
        required = len(self.indices) + 1
        storage = self._index_storage
        if required > storage.size:
            expanded = np.empty(max(required, 1024, 2 * storage.size),
                                dtype=np.int64)
            if self.indices:
                expanded[:len(self.indices)] = storage[:len(self.indices)]
            self._index_storage = storage = expanded
        storage[required - 1] = int(rotation)
        self.indices.append(int(rotation))

    def _indices_view(self) -> np.ndarray:
        """View all absolute symbol decisions without rebuilding them per packet."""
        return self._index_storage[:len(self.indices)]

    def hold_transition_feedback(self) -> None:
        """Freeze decision feedback while V.90 sends TRN1d, Jd and DIL.

        Those intervals contain silence and S markers, not QAM training
        symbols. The symbol clock still advances and decisions remain
        available, but the equalizer, carrier and timing loops stay at their
        trained state until a CRC-valid CPt confirms that upstream QAM has
        returned.
        """
        self.eq._input_guard.arm(hold=True)
        self.eq.adapt = False

    def begin_phase4(self, constellation: int) -> None:
        """Switch future decisions to the Jd-selected Phase 4 constellation."""
        if constellation not in (4, 16):
            raise ValueError("V.90 Phase 4 constellation must be 4 or 16")
        if self.trn_end is None:
            raise RuntimeError("cannot enter V.90 Phase 4 before TRN tracking")
        self._phase4_constellation = int(constellation)
        self._phase4_active = True
        self._sq = None
        # Skip symbols already tracked through the Phase 3/DIL boundary.
        self._symbol_cursor = len(self.indices)
        self._decision_cursor = len(self.indices)
        self._equalized_pending.clear()

    def resume_phase4_feedback(self) -> None:
        """Resume carrier, timing, and tap adaptation after CRC-valid CPt."""
        if self.eq._input_guard is not None:
            self.eq._input_guard.enable_release()
        self.eq.adapt = True
        self.eq.adapt_nlms = True

    def begin_phase4_carrier_timing_acquisition(self) -> None:
        """Track returning CPt carrier/clock while keeping trained taps fixed.

        CPt starts with a known sync pattern. Once its measured symbol onset is
        reached, carrier and timing loops can reacquire while the 16-point
        slicer is still ambiguous. NLMS stays frozen until the full CPt CRC
        validates the received constellation/profile.
        """
        if not self._phase4_active:
            raise RuntimeError("cannot acquire Phase 4 before its Jd transition")
        if self.eq._input_guard is not None:
            self.eq._input_guard.enable_release()
        self.eq.adapt = True
        self.eq.adapt_nlms = False

    def _decide16(self, y: complex) -> complex:
        # Match the production V.34 Phase4Receiver's gain-invariant 16-point
        # slicer so its ring decision and equalizer adaptation use one value.
        power = float(y.real * y.real + y.imag * y.imag)
        guard = getattr(self.eq, "_input_guard", None)
        reference = None if guard is None else guard.reference
        input_power = getattr(self.eq, "last_input_power", None)
        signal_active = (reference is None or input_power is None
                         or input_power >= reference * guard.EXIT_RATIO)
        if signal_active:
            self._sq = (power if self._sq is None else
                        (1.0 - v34_rx.Phase4Receiver.AGC_ALPHA) * self._sq
                        + v34_rx.Phase4Receiver.AGC_ALPHA * power)
        if self._sq is None or self._sq <= 1e-9:
            self._last = y
        else:
            self._last = y * np.sqrt(v34_rx._C16_MEAN_SQ / self._sq)
        return v34_rx.decide16(self._last)

    def _decide4_with_rotation(self, y: complex) -> complex:
        """Slice a four-point symbol and retain its rotation in one angle pass."""
        point = y / v34_tx.P0
        rotation = int(round(-math.atan2(point.imag, point.real)
                             / (math.pi / 2))) % 4
        self._last_rotation = rotation
        return v34_tx.P0 * v34_tx.ROT_CW[rotation]

    def feed(self, max_symbols: int | None = None) -> list[dict]:
        """Consume newly available baseband and return newly qualified Ja frames."""
        if not getattr(self, "_stage_timing_enabled", False):
            return self._feed(max_symbols)
        started = time.thread_time_ns()
        start_k = self.eq.k
        self._stage_cpu_ns = {"equalizer": 0, "symbol_routing": 0,
                              "ja_scan": 0}
        result = self._feed(max_symbols)
        self._stage_timing_rows.append({
            "symbols": int(self.eq.k - start_k),
            "total": time.thread_time_ns() - started,
            **self._stage_cpu_ns,
        })
        return result

    def timing_report(self) -> dict | None:
        """Return bounded per-feed CPU timing, when explicitly enabled."""
        if not getattr(self, "_stage_timing_enabled", False):
            return None
        rows = list(self._stage_timing_rows)
        if not rows:
            return {"enabled": True, "count": 0, "stages": {}}
        stages = {}
        for name in ("total", "equalizer", "symbol_routing", "ja_scan"):
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            stages[name] = {
                "mean_ms": float(np.mean(values) / 1e6),
                "p95_ms": float(np.percentile(values, 95) / 1e6),
                "p99_ms": float(np.percentile(values, 99) / 1e6),
                "max_ms": float(np.max(values) / 1e6),
            }
        return {"enabled": True, "count": len(rows), "capacity": 2048,
                "stages": stages}

    def _feed(self, max_symbols: int | None = None) -> list[dict]:
        """Uninstrumented receiver loop; timing wrapper is opt-in."""
        produced = 0
        timed = getattr(self, "_stage_timing_enabled", False)
        while self.eq.k < self.eq.available() and (max_symbols is None or produced < max_symbols):
            stage_started = time.thread_time_ns() if timed else 0
            window = self.eq.window(self.eq.k)
            phase = float(self.eq.phase)
            if self._phase4_active and self._phase4_constellation == 16:
                y = self.eq.step(decide=self._data_decider or self._decide16,
                                 _window=window)
                quarter, rotation = v34_rx.slice16(self._last)
            else:
                if self._data_decider is None:
                    y = self.eq.step(decide=self._decide4_with_rotation,
                                     _window=window)
                    rotation = self._last_rotation
                else:
                    y = self.eq.step(decide=self._data_decider,
                                     _window=window)
                    rotation = int(round(-np.arctan2((y / v34_tx.P0).imag,
                                                      (y / v34_tx.P0).real)
                                           / (np.pi / 2))) % 4
                quarter = 0
            if timed:
                self._stage_cpu_ns["equalizer"] += time.thread_time_ns() - stage_started
                stage_started = time.thread_time_ns()
            self._append_rotation(rotation)
            self.quarters.append(int(quarter))
            if not hasattr(self, "input_powers"):
                self.input_powers = []
            input_power = getattr(self.eq, "last_input_power", None)
            self.input_powers.append(input_power)
            observation = (len(self.indices) - 1, complex(y),
                           (rotation, int(quarter)), input_power)
            if self._symbol_callback is None:
                self._equalized_pending.append(observation)
            else:
                self._symbol_callback(observation)
            if timed:
                self._stage_cpu_ns["symbol_routing"] += time.thread_time_ns() - stage_started
            bit1, bit2 = self.scrambler.step(1), self.scrambler.step(1)
            self.mismatch.append(0 if rotation == 2 * bit2 + bit1 else 1)
            produced += 1

            count = len(self.indices)
            if (self.trn_end is None and count >= 16
                    and sum(self.mismatch[-16:]) >= 8
                    and count - 16 > 300):
                self._ja_search_active = True

        if self.trn_end is None:
            if (self._ja_search_active
                    and len(self.indices) - self._ja_last_search_at
                    >= self.JA_BOUNDARY_SEARCH_INTERVAL):
                stage_started = time.thread_time_ns() if timed else 0
                result = self._search_ja_boundary()
                if timed:
                    self._stage_cpu_ns["ja_scan"] += time.thread_time_ns() - stage_started
                return result
            return []

        if self._ja_symbol_cursor >= len(self.indices):
            return []
        idx = self._indices_view()
        new_bits = v34_rx._descramble_at(
            idx, self.role, self._ja_symbol_cursor,
            2 * (len(idx) - self._ja_symbol_cursor), abs_end=self.trn_end)
        self._ja_symbol_cursor = len(idx)
        return self.scanner.feed(new_bits)

    def _search_ja_boundary(self) -> list[dict]:
        """Validate the estimated TRN end by incrementally scanning for Ja."""
        self._ja_last_search_at = len(self.indices)
        if self._ja_candidate_boundary is None:
            estimate = v34_rx.trn_end_from_mismatch(self.mismatch, last=True)
            if estimate is None or estimate < self.MIN_TRN_SYMBOLS:
                return []
            self._ja_candidate_boundary = estimate
            # One full differential symbol plus the 23-bit self-synchronizing
            # descrambler transient are unavailable at the absolute/differential
            # transition. Start on the next symbol boundary; V90JaScanner then
            # locates and CRC-checks a complete repeated Ja descriptor.
            self._ja_symbol_cursor = estimate + 12

        boundary = self._ja_candidate_boundary
        if self._ja_symbol_cursor < len(self.indices):
            idx = self._indices_view()
            bits = v34_rx._descramble_at(
                idx, self.role, self._ja_symbol_cursor,
                2 * (len(idx) - self._ja_symbol_cursor), abs_end=boundary)
            self._ja_symbol_cursor = len(idx)
            frames = self.scanner.feed(bits)
            valid = [item for item in frames if item.get("valid")
                     and item.get("crc_valid") and item.get("reserved_bits_zero")]
            if valid:
                self.trn_end = boundary
                return valid
        return []

    def read_new_symbol_rotations(self) -> list[int]:
        """Return each newly tracked four-point rotation exactly once.

        The V.90 Phase 4 CP path uses the same trained upstream timing and
        equalizer, but has its own differential/GPA decoder. Keeping this
        cursor separate avoids re-decoding old symbols while allowing that
        decoder to run continuously before CPt arrives.
        """
        end = len(self.indices)
        start = self._symbol_cursor
        self._symbol_cursor = end
        return self.indices[start:end]

    def read_new_symbol_decisions(self) -> list[tuple[int, int]]:
        """Drain (rotation, quarter) decisions for the V.90 CP/E decoder."""
        return [decision for decision, _ in self.read_new_symbol_observations()]

    def read_new_symbol_observations(self) -> list[tuple[tuple[int, int], float | None]]:
        """Drain phase-four decisions with their aligned input-window power."""
        end = len(self.indices)
        start = self._decision_cursor
        self._decision_cursor = end
        decisions = list(zip(self.indices[start:end], self.quarters[start:end]))
        return list(zip(decisions, self.input_powers[start:end]))

    def read_new_equalized_observations(self):
        """Drain aligned equalizer outputs, decisions, and input powers."""
        output = list(self._equalized_pending)
        self._equalized_pending.clear()
        return output

    def begin_data_tracking(self, decider) -> None:
        """Use the qualified V.34 slicer as equalizer feedback after B1."""
        if not callable(decider):
            raise TypeError("V.90 data decision feedback must be callable")
        self._data_decider = decider

    def set_symbol_callback(self, callback) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("V.90 symbol callback must be callable")
        self._symbol_callback = callback
