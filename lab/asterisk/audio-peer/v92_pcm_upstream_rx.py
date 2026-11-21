"""Streaming digital-modem receiver for V.92 PCM-upstream data mode (clause 8.7.1, 6.4, 9.6).

Takes the G.711 octets of the upstream direction as they arrive from RTP, in any chunking, and:

* skips the units before B1u (``start_skip``; the caller knows where data frame interval 0 begins);
* cuts the rest into 12-interval data frames and decodes each with ``PcmUpstreamReceiver``
  (nearest constellation point per interval, trellis parity check, inverse modulus encoder);
* descrambles with the self-synchronising scrambler (analogue GPA, tap 5; the scrambler is
  initialised to zero before B1u, which a self-synchronising descrambler does not need);
* qualifies B1u: 48 data frames of scrambled ones, no wrong bit and no trellis violation allowed
  (the limits are arguments), then passes data bits on.

It decodes the *constellation domain*: the precoder is designed so the line output equals the
constellation level u(n), so the received level is used as it is, divided by ``gain`` if the analogue
transmit gain G at the prefilter output has to be undone. A line that does not deliver u(n)
(an unequalised, noisy or non-flat one) simply produces wrong decisions here: there is no
equaliser, no timing recovery and no gain control in this class.
"""

from __future__ import annotations

import os

import numpy as np

import v34_data
import v34_g711
import v92_pcm_upstream as pu

B1_FRAMES = 48
TAP = 5


class _SoftTrellis32:
    """Fixed-lag soft decoder for the four-interval V.92 32-state trellis."""

    def __init__(self, profile: pu.PcmUpstreamProfile, *, candidates: int = 2,
                 traceback_groups: int = 64):
        if profile.trellis != 1:
            raise ValueError("soft PCM upstream decoder currently requires the 32-state trellis")
        if candidates < 1 or traceback_groups < 1:
            raise ValueError("candidate and traceback limits must be positive")
        self.profile = profile
        self.candidates = int(candidates)
        self.traceback_groups = int(traceback_groups)
        self.metrics = np.full(32, np.inf)
        self.metrics[0] = 0.0
        self.history: list[list[tuple[int, tuple[int, ...], tuple[int, ...]] | None]] = []
        self.symbols_seen = 0
        # The convolutional transition depends only on the two 3-bit subset
        # labels. Precompute it so the hot path avoids complex arithmetic and
        # repeated encoder object construction for every candidate branch.
        # eta runs over -L..L-1 for the largest set (L <= 64 in a CPd).
        self._offset = max(len(levels) for levels in profile.constellations)
        span = range(-self._offset, self._offset)
        self._subset = np.empty((len(span), len(span)), dtype=np.uint8)
        for ia, a in enumerate(span):
            for ib, b in enumerate(span):
                self._subset[ia, ib] = v34_data.subset_label(
                    complex(2 * a + 1, 2 * b + 1))
        self._grids = {}
        self._points = []
        for levels in profile.constellations:
            n = len(levels)
            etas = np.arange(-n, n, dtype=np.int64)
            self._points.append((np.array([-levels[-eta - 1] if eta < 0 else levels[eta] for eta in etas],
                                          dtype=float), etas))
        self._next = np.empty((32, 8, 8), dtype=np.uint8)
        for state in range(32):
            for s0 in range(8):
                for s1 in range(8):
                    conv = v34_data.ConvEncoder32()
                    conv.d = [(state >> i) & 1 for i in range(5)]
                    conv.step(v34_data.TABLE13[s0][s1])
                    self._next[state, s0, s1] = sum(
                        bit << i for i, bit in enumerate(conv.d))

    def _candidate_sets(self, values: np.ndarray):
        """Per slot the ``candidates`` nearest points as (etas, costs) arrays, nearest first (ties: lower eta);
        slot 3 per parity. The constellation of each interval is precomputed in ``_points``."""
        out = []
        for slot, value in enumerate(values):
            interval = (self.symbols_seen + slot) % pu.INTERVALS
            signed, etas = self._points[self.profile.interval_set[interval]]
            costs = (float(value) - signed) ** 2
            order = np.lexsort((etas, costs))
            if slot < 3:
                pick = order[:self.candidates]
                out.append((etas[pick], costs[pick]))
            else:
                per_parity = []
                for parity in (0, 1):
                    pick = order[(etas[order] & 1) == parity][:self.candidates]
                    per_parity.append((etas[pick], costs[pick]))
                out.append(per_parity)
        return out

    def _traceback(self, full: bool = False):
        """The survivor path's (etas, hard) per group, oldest first; only the oldest unless ``full``."""
        state = int(np.argmin(self.metrics))
        states = []
        for previous, _, _ in reversed(self.history):
            states.append(state)
            state = previous[state]
            if state < 0:
                raise RuntimeError("broken soft-trellis survivor chain")
        states.reverse()
        groups = range(len(self.history)) if full else range(1)
        return [(tuple(self.history[i][1][states[i]]), self.history[i][2]) for i in groups]

    def _grid(self, sizes):
        """Flat (a, b, c, d-candidate, state) indices of every branch for these candidate counts, in the order
        (a, b, c, state parity, d, state) of the earlier nested loops. State parity is y0 = bit 4."""
        grid = self._grids.get(sizes)
        if grid is None:
            ia, ib, ic, sp, ik, low = (axis.ravel() for axis in np.indices(sizes[:3] + (2, sizes[3], 16)))
            grid = self._grids[sizes] = (ia, ib, ic, sp, ik, sp * 16 + low)
        return grid

    def push(self, values) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        samples = np.asarray(values, dtype=float).reshape(-1)
        if samples.size != 4:
            raise ValueError("soft trellis consumes four intervals at a time")
        (ea, ca), (eb, cb), (ec, cc), last = self._candidate_sets(samples)
        hard = (int(ea[0]), int(eb[0]), int(ec[0]))
        (e0, c0), (e1, c1) = last
        # the nearest of either parity; on a tie the parity-0 point (it is listed first)
        hard4 = int(e0[0]) if len(e0) and (not len(e1) or c0[0] <= c1[0]) else int(e1[0])
        hard = hard + (hard4,)
        # Slot 3 per required parity, padded to a common width; a missing candidate has an infinite cost.
        width = max(len(e0), len(e1), 1)
        ed = np.zeros((2, width), dtype=np.int64)
        cd = np.full((2, width), np.inf)
        ed[0, :len(e0)], cd[0, :len(e0)] = e0, c0
        ed[1, :len(e1)], cd[1, :len(e1)] = e1, c1
        o = self._offset
        ia, ib, ic, sp, ik, states = self._grid((len(ea), len(eb), len(ec), width))
        a, b, c = ea[ia], eb[ib], ec[ic]
        parity = ((a + b + c) & 1) ^ sp
        d = ed[parity, ik]
        metrics = ca[ia] + cb[ib] + cc[ic] + cd[parity, ik] + self.metrics[states]
        nexts = self._next[states, self._subset[a + o, b + o], self._subset[c + o, d + o]]
        next_metrics = np.full(32, np.inf)
        np.minimum.at(next_metrics, nexts, metrics)
        if not np.isfinite(next_metrics).any():
            raise ValueError("no valid 32-state PCM upstream trellis path")
        winners = np.flatnonzero(np.isfinite(metrics)
                                 & np.isclose(metrics, next_metrics[nexts], rtol=1e-12, atol=1e-12))
        targets, first = np.unique(nexts[winners], return_index=True)
        chosen = winners[first]
        previous = np.full(32, -1, dtype=np.int64)
        branch_etas = np.zeros((32, 4), dtype=np.int64)
        previous[targets] = states[chosen]
        branch_etas[targets] = np.stack((a[chosen], b[chosen], c[chosen], d[chosen]), axis=1)
        self.metrics = next_metrics
        self.history.append((previous.tolist(), branch_etas.tolist(), hard))
        self.symbols_seen += 4
        if len(self.history) > self.traceback_groups:
            path = self._traceback()
            self.history.pop(0)
            return path
        return []

    def flush(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        if not self.history:
            return []
        path = self._traceback(full=True)
        state = int(np.argmin(self.metrics))
        self.history.clear()
        self.metrics[:] = np.inf
        self.metrics[state] = 0.0
        return path


class PcmUpstreamStreamReceiver:
    """Stream decoder with a bounded consecutive out-of-constellation run.

    ``lost_frames`` is cumulative for diagnostics. ``max_lost_frames`` bounds
    only uninterrupted invalid frames; a successfully decoded frame resets the
    run counter so a later independent RTP impairment gets its own budget.
    """

    IDLE, B1, DATA, FAILED = "idle", "b1", "data", "failed"

    def __init__(self, profile: pu.PcmUpstreamProfile, law: str, *, start_skip: int = 0,
                 gain: float = 1.0, b1_error_limit: int = 0, b1_violation_limit: int = 0,
                 max_lost_frames: int = 3, max_data_lost_frames: int = 32,
                 decoder_mode: str = "hard",
                 viterbi_candidates: int = 2, viterbi_traceback_groups: int = 64):
        if law not in ("pcmu", "pcma"):
            raise ValueError("law must be pcmu or pcma")
        if decoder_mode not in ("hard", "viterbi32"):
            raise ValueError("decoder_mode must be hard or viterbi32")
        if start_skip < 0 or gain <= 0:
            raise ValueError("start_skip must not be negative and gain must be positive")
        if (isinstance(max_lost_frames, bool) or not isinstance(max_lost_frames, int)
                or max_lost_frames < 0):
            raise ValueError("max_lost_frames must be a nonnegative integer")
        if (isinstance(max_data_lost_frames, bool) or not isinstance(max_data_lost_frames, int)
                or max_data_lost_frames < 0):
            raise ValueError("max_data_lost_frames must be a nonnegative integer")
        self.profile, self.law, self.gain = profile, law, float(gain)
        self._rx = pu.PcmUpstreamReceiver(profile)
        self.decoder_mode = decoder_mode
        self._soft = (_SoftTrellis32(profile, candidates=viterbi_candidates,
                                     traceback_groups=viterbi_traceback_groups)
                      if decoder_mode == "viterbi32" else None)
        self._soft_input_frames = 0
        self._soft_ks: list[int] = []
        self._soft_group_etas: list[int] = []
        self.soft_representative_corrections = 0
        self._skip = int(start_skip)
        self._pending: list[float] = []
        self._reg = [0] * 23                    # last 23 received (scrambled) bits, newest first
        self.state = self.B1 if self._skip == 0 else self.IDLE
        self.frames = 0
        self.lost_frames = 0
        self.consecutive_lost_frames = 0
        self.max_consecutive_lost_frames = 0
        self.b1_errors = 0
        # B1u bits whose errors are a descrambler transient, not line errors: set after an FPE, where the peer may
        # zero its scrambler at FB1u rather than at B1u (see _FpeUpstreamDataRx.SCRAMBLER_SPAN).
        self.b1_scrambler_transient_bits = 0
        self.b1_scrambler_transient_errors = 0
        # When a list: (frames, sign, descrambler register) after each frame, the last FRAME_STATES of them.
        self.frame_states: list | None = None
        self._b1_error_limit, self._b1_violation_limit = b1_error_limit, b1_violation_limit
        self._b1_frame_trace_enabled = os.environ.get("EL_MODEM_V92_B1_CUT_DIAGNOSTICS") == "1"
        self._b1_frame_violation_start: int | None = None
        self._active_b1_frame_trace: dict | None = None
        self.b1_frame_trace: list[dict] = []
        self._max_lost = max_lost_frames
        self._max_data_lost = max_data_lost_frames
        self._loss_trace_enabled = os.environ.get("EL_MODEM_V92_UPSTREAM_LOSS_TRACE") == "1"
        self.loss_trace: list[dict] = []
        self._data_bits: list[int] = []
        self.failure: str | None = None

    @property
    def trellis_violations(self) -> int:
        return self._rx.trellis_violations

    def feed(self, codewords: bytes) -> None:
        if self.state == self.FAILED:
            return
        data = bytes(codewords)
        if self._skip:
            drop = min(self._skip, len(data))
            self._skip -= drop
            data = data[drop:]
            if self._skip == 0:
                self.state = self.B1
        if not data:
            return
        linear = v34_g711.decode(np.frombuffer(data, dtype=np.uint8), self.law).astype(float)
        self.feed_samples(linear)

    def feed_samples(self, samples) -> None:
        """Consume linear PCM samples directly, without a redundant G.711 encode/decode."""
        if self.state == self.FAILED:
            return
        linear = np.asarray(samples, dtype=float).reshape(-1)
        self._pending.extend(linear / self.gain)
        while len(self._pending) >= pu.INTERVALS and self.state != self.FAILED:
            frame, self._pending = self._pending[:pu.INTERVALS], self._pending[pu.INTERVALS:]
            self._frame(frame)

    def _descramble(self, bits) -> list[int]:
        out = []
        reg = self._reg
        for y in bits:
            out.append(y ^ reg[TAP - 1] ^ reg[22])
            reg = [y] + reg[:-1]
        self._reg = reg
        return out

    def _frame(self, samples) -> None:
        trace_b1 = self._b1_frame_trace_enabled and self.state == self.B1
        record = None
        if trace_b1:
            input_frame = self._soft_input_frames + 1 if self._soft is not None else self.frames + 1
            record = {
                "input_frame": input_frame,
                "bit_errors": 0,
                "bit_errors_total": self.b1_errors,
                "parity_events": 0,
                "parity_events_total": self.trellis_violations,
                "lost_frames": 0,
                "lost_frames_total": self.lost_frames,
                "decoded_frames": [],
                "state_after": self.state,
            }
            self._active_b1_frame_trace = record
        errors_before = self.b1_errors
        violations_before = self.trellis_violations
        lost_before = self.lost_frames
        try:
            self._process_frame(samples)
        finally:
            if record is not None:
                record["bit_errors"] = self.b1_errors - errors_before
                record["bit_errors_total"] = self.b1_errors
                record["parity_events"] = self.trellis_violations - violations_before
                record["parity_events_total"] = self.trellis_violations
                record["lost_frames"] = self.lost_frames - lost_before
                record["lost_frames_total"] = self.lost_frames
                record["state_after"] = self.state
                self.b1_frame_trace.append(record)
                self._active_b1_frame_trace = None

    def _process_frame(self, samples) -> None:
        if self._soft is not None:
            values = np.asarray(samples, dtype=float)
            try:
                for offset in range(0, pu.INTERVALS, 4):
                    for etas, hard in self._soft.push(values[offset:offset + 4]):
                        self._soft_group(etas, hard)
            except ValueError:
                self._lose_frame()
                return
            self._soft_input_frames += 1
            if self.state == self.B1 and self._soft_input_frames == B1_FRAMES:
                for etas, hard in self._soft.flush():
                    self._soft_group(etas, hard)
                if self._soft_ks:
                    self._fail("soft trellis ended with an incomplete data frame")
            return
        try:
            ks = self._rx.decode_intervals(samples)
        except ValueError:
            self._lose_frame()
            return
        try:
            scrambled, next_sign = pu.modulus_decode(
                ks, self.profile.moduli, self.profile.total_modulus,
                self._rx._sign, self.profile.bits_per_frame)
        except ValueError:
            if self.state == self.DATA:
                scrambled, next_sign = self._best_effort_data_decode(ks)
                self._rx._sign = next_sign
                self._lose_frame(data_guess=scrambled)
            else:
                self._lose_frame()
            return
        self._rx._sign = next_sign
        self.consecutive_lost_frames = 0
        self._accept_scrambled(scrambled)

    def _best_effort_data_decode(self, ks: list[int]) -> tuple[list[int], int]:
        """Recover a K-bit estimate and sign from a DATA K-vector outside 2**K."""
        total = self.profile.total_modulus
        raw = 0
        scale = 1
        for k, modulus in zip(ks, self.profile.moduli):
            raw += int(k) * scale
            scale *= modulus
        raw = min(max(raw, 0), total - 1)
        old_sign = self._rx._sign
        value = raw if old_sign == 0 else total - 1 - raw
        sign = 1 if 2 * value > total - 1 else 0
        bits_value = min(value, (1 << self.profile.bits_per_frame) - 1)
        bits = [(bits_value >> i) & 1 for i in range(self.profile.bits_per_frame)]
        return bits, sign ^ old_sign

    def _lose_frame(self, *, data_guess: list[int] | None = None) -> None:
        if self._loss_trace_enabled:
            frame = self.frames + 1
            self.loss_trace.append({
                "frame_counter": frame,
                "data_frame_index": (frame - B1_FRAMES - 1
                                     if self.state == self.DATA else None),
                "state": self.state,
                "consecutive_lost_frames": self.consecutive_lost_frames + 1,
            })
        self.lost_frames += 1
        self.consecutive_lost_frames += 1
        self.max_consecutive_lost_frames = max(
            self.max_consecutive_lost_frames, self.consecutive_lost_frames)
        if self.state == self.DATA:
            # Keep the fixed K-bit cadence; LAPM can discard/retransmit a damaged frame.
            bits = data_guess if data_guess is not None else [0] * self.profile.bits_per_frame
            self._accept_scrambled(bits)
            if self.consecutive_lost_frames > self._max_data_lost:
                self._fail("too many consecutive data frames outside the constellation")
            return
        if self.consecutive_lost_frames > self._max_lost:
            self._fail("too many data frames outside the constellation")
            return
        self._accept_scrambled([0] * self.profile.bits_per_frame)

    def _accept_scrambled(self, scrambled) -> None:
        trace_b1 = self._b1_frame_trace_enabled and self.state == self.B1
        frame_number = self.frames + 1
        errors_before = self.b1_errors
        bits = self._descramble(scrambled)
        self.frames += 1
        if self.frame_states is not None:
            self.frame_states.append((self.frames, self._rx._sign, list(self._reg)))
            del self.frame_states[:-self.FRAME_STATES]
        if self.state == self.B1:
            first = (self.frames - 1) * len(bits)
            errors = [first + i for i, b in enumerate(bits) if b != 1]
            transient = sum(index < self.b1_scrambler_transient_bits for index in errors)
            self.b1_scrambler_transient_errors += transient
            self.b1_errors += len(errors) - transient
            if self.frames == B1_FRAMES:
                if (self.b1_errors > self._b1_error_limit
                        or self.trellis_violations > self._b1_violation_limit):
                    self._fail(f"B1u failed: {self.b1_errors} bit errors, "
                               f"{self.trellis_violations} trellis violations")
                else:
                    self.state = self.DATA
        elif self.state == self.DATA:
            self._data_bits.extend(bits)
        if trace_b1:
            decoded = {
                "frame": frame_number,
                "bit_errors": self.b1_errors - errors_before,
                "bit_errors_total": self.b1_errors,
                "state_after": self.state,
            }
            if self._active_b1_frame_trace is not None:
                self._active_b1_frame_trace["decoded_frames"].append(decoded)
            else:
                # Defensive fallback for direct test/tool calls outside _frame().
                self.b1_frame_trace.append({
                    "input_frame": None, "bit_errors": decoded["bit_errors"],
                    "bit_errors_total": decoded["bit_errors_total"],
                    "parity_events": 0, "parity_events_total": self.trellis_violations,
                    "lost_frames": 0, "lost_frames_total": self.lost_frames,
                    "decoded_frames": [decoded], "state_after": self.state,
                })

    def _soft_group(self, etas: tuple[int, ...], hard: tuple[int, ...]) -> None:
        self.soft_representative_corrections += sum(a != b for a, b in zip(etas, hard))
        for offset, eta in enumerate(etas):
            interval = len(self._soft_ks) % pu.INTERVALS
            modulus = self.profile.moduli[interval]
            if interval % pu.TRELLIS_FRAME < 3:
                self._soft_ks.append(eta % modulus)
            else:
                parity = eta & 1
                expected = (sum(self._soft_group_etas[-3:]) + self._rx.conv.y0) & 1
                if parity != expected:
                    self._rx.trellis_violations += 1
                self._soft_ks.append(((eta - parity) // 2) % modulus)
                pu._trellis_step(self._rx.conv, self._soft_group_etas[-3:] + [eta])
            self._soft_group_etas.append(eta)
            if len(self._soft_group_etas) == 4:
                self._soft_group_etas.clear()
            if len(self._soft_ks) == pu.INTERVALS:
                try:
                    scrambled, next_sign = pu.modulus_decode(
                        self._soft_ks, self.profile.moduli, self.profile.total_modulus,
                        self._rx._sign, self.profile.bits_per_frame)
                except ValueError:
                    if self.state == self.DATA:
                        scrambled, next_sign = self._best_effort_data_decode(self._soft_ks)
                        self._rx._sign = next_sign
                        self._soft_ks.clear()
                        self._lose_frame(data_guess=scrambled)
                        continue
                    self._soft_ks.clear()
                    self._lose_frame()
                    continue
                self._soft_ks.clear()
                self._rx._sign = next_sign
                self.consecutive_lost_frames = 0
                self._accept_scrambled(scrambled)


    def _fail(self, reason: str) -> None:
        self.state, self.failure = self.FAILED, reason

    def take_bits(self) -> list[int]:
        out, self._data_bits = self._data_bits, []
        return out

    FRAME_STATES = 64

    def flush_soft(self) -> None:
        """Decide what the soft decoder still holds (its traceback lag); the stream must end on a whole group."""
        if self._soft is not None and self.state != self.FAILED:
            for etas, hard in self._soft.flush():
                self._soft_group(etas, hard)

    def take_bytes(self) -> bytes:
        """Whole bytes (least-significant bit first); a partial byte stays for the next call."""
        n = len(self._data_bits) // 8 * 8
        bits, self._data_bits = self._data_bits[:n], self._data_bits[n:]
        return bytes(sum(bits[8 * j + i] << i for i in range(8)) for j in range(n // 8))
