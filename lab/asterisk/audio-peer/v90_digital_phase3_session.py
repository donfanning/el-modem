"""Join analogue upstream Ja/CP reception to the digital Phase 3 FSM."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict
import os
import time

import numpy as np

from v90_digital_phase3 import V90DigitalAnswererPhase3
from v90_cp import V90CpSymbolDecoder
from v90_digital_phase4_session import V90DigitalAnswererPhase4Session
from v90_mp_policy import (answerer_mp_profile,
                           select_analog_upstream_profile,
                           validate_answerer_mp_profile)
from v90_s_transition import V90STransitionDetector
from v90_upstream_data import V90UpstreamDataReceiver
from v90_upstream_session import V90UpstreamPhase3Session


class V90DigitalAnswererPhase3Session:
    """Start digital Phase 3 from Ja and track subsequent Phase 4 CP messages.

    PCM receive and exact G.711 transmit are separate clocked APIs. A media
    adapter must supply ordered 8 kHz linear PCM to ``feed_pcm`` and call
    ``read_g711`` on the transmit clock. A continuous four-point decoder
    records CRC-valid CPt/CP/CP-prime profiles and routes them into the
    clocked Phase 4 transmitter after DIL. The optional LAPM adapter joins
    decoded analogue-upstream bits and the digital-downstream mapper; it
    remains opt-in until paired waveform qualification is complete. Phase 2
    handoff and gap recovery remain enclosing-engine work.
    """

    # In an ideal continuous waveform the +/-6T equalizer window and 8T pulse
    # shaper make the input-power threshold rise 13 symbols before CPt's first
    # symbol center. Search a small tolerance around that measured boundary.
    CPT_START_HISTORY = 5
    CPT_START_OFFSET = 13
    CPT_START_TOLERANCE = 16
    # Keep CP frame scanning incremental while amortizing the Python scanner
    # call across a few symbols per timing candidate.
    CPT_CANDIDATE_BATCH_SYMBOLS = 16
    STAGE_TIMING_SLOW_CALLBACK_NS = 8_000_000
    STAGE_TIMING_SLOW_CALLBACK_LIMIT = 32
    CPT_CANDIDATE_MAX_SYMBOLS = 960
    # Phase 4 CPt onset needs a stronger transition than the very sensitive
    # input-power guard uses to keep carrier tracking alive. The latter's
    # 0.04% threshold treats low-level training tails as continuously active.
    CPT_ACTIVITY_RATIO = 0.01
    # Avoid releasing on the short activity burst seen before call-6 CPt;
    # this exceeds that burst, while the CPt candidate-start window remains
    # anchored to CPT_START_OFFSET and its tolerance.
    CPT_FEEDBACK_RELEASE_SYMBOLS = 20
    # The paired caller begins CPt after receiving 320 exact-law octets from
    # the initial R-i burst. If the preceding training tail never produces a
    # power-guard quiet edge, use that protocol clock to re-arm CPt acquisition.
    CPT_EXPECTED_DELAY_SAMPLES = 320

    def __init__(self, *, baud: float, carrier: float, law: str, uinfo: int,
                 rate_mask: int, phase3_16point: bool = False,
                 renegotiation_16point: bool = False,
                 max_lookahead: int = 1, trn1d_symbols: int = 2040,
                 round_trip_delay_samples: int = 0,
                 trn_scrambler_role: str = "answer", span: int = 12,
                 enable_lapm: bool = False,
                 lapm_detect: bool = True,
                 upstream_probe_frequency_offset_units: int = 0,
                 upstream_equalizer_mu: float | None = None,
                 phase4_mp_profile: dict | None = None):
        if law.lower() not in ("pcmu", "pcma"):
            raise ValueError("V.90 Phase 3 requires negotiated PCMU or PCMA")
        self.law = law.lower()
        self.baud = float(baud)
        self.carrier = float(carrier)
        if (isinstance(upstream_probe_frequency_offset_units, bool)
                or not isinstance(upstream_probe_frequency_offset_units, int)
                or not -512 <= upstream_probe_frequency_offset_units <= 511):
            raise ValueError("upstream probe frequency offset must be -512 or in -511..511 units")
        self.upstream_probe_frequency_offset_units = int(
            upstream_probe_frequency_offset_units)
        self.upstream = V90UpstreamPhase3Session(
            baud, carrier, trn_scrambler_role=trn_scrambler_role, span=span,
            probe_frequency_offset_units=upstream_probe_frequency_offset_units,
            equalizer_mu=upstream_equalizer_mu)
        self.transmitter = V90DigitalAnswererPhase3(
            uinfo, self.law, None, rate_mask,
            phase3_16point=phase3_16point,
            renegotiation_16point=renegotiation_16point,
            max_lookahead=max_lookahead, trn1d_symbols=trn1d_symbols,
            round_trip_delay_samples=round_trip_delay_samples)
        self.transitions: list[int] = []
        self.transition_events: list[tuple[str, int]] = []
        self._ignore_initial_phase3_flip = False
        self._phase4_receive_armed = False
        self.events: list[tuple[str, object]] = []
        self.phase4_mp_profile = (
            answerer_mp_profile(self.baud) if phase4_mp_profile is None
            else validate_answerer_mp_profile(phase4_mp_profile))
        self.analog_upstream_rate: int | None = None
        self.analog_upstream_profile = None
        self.phase4_constellation = 16 if phase3_16point else 4
        self.phase4: V90DigitalAnswererPhase4Session | None = None
        self.upstream_data: V90UpstreamDataReceiver | None = None
        self._upstream_data_tracking = False
        self._upstream_data_bits: deque[int] = deque()
        self.enable_lapm = bool(enable_lapm)
        self.lapm_detect = bool(lapm_detect)
        self.lapm = None
        self._queued_dte = bytearray()
        self._received_dte = bytearray()
        self._phase4_profile_cursor = 0
        self.phase4_error: str | None = None
        # CPt/CP is decoded continuously from the same tracked upstream stream.
        # It must start before CPt because the differential and GPA memories
        # reset at that protocol boundary, not at a timestamp guessed by us.
        self.phase4_receiver = V90CpSymbolDecoder(self.phase4_constellation)
        self.phase4_profiles: list[dict] = []
        self._phase4_cpt_acquired = False
        self._phase4_cpt_candidates: list[dict] = []
        self._phase4_cpt_quiet_count = 0
        self._phase4_cpt_active_run: int | None = None
        self._phase4_feedback_released_early = False
        self._phase4_cpt_recent: list[tuple[tuple[int, int], float | None, int]] = []
        self._phase4_observation_trace: deque[dict] = deque(maxlen=48)
        self._phase4_decision_trace: deque[tuple[int, int, int, int]] = deque(maxlen=6000)
        self._phase4_equalized_recent: deque[tuple[int, float, float]] = deque(maxlen=256)
        self._upstream_b1_trace: list[tuple[int, float, float]] = []
        # Per-symbol traces are useful for offline diagnosis, but allocating
        # several Python objects for every received symbol competes with the
        # real-time receive path. Keep the production default lean; enable the
        # detailed trace explicitly for a diagnostic capture.
        self._capture_symbol_trace = (
            os.environ.get("V90_CAPTURE_SYMBOL_TRACE") == "1")
        self._phase4_decoder_origin_symbol: int | None = None
        self._phase4_speculative_decoder: V90CpSymbolDecoder | None = None
        self._phase4_speculative_origin_symbol: int | None = None
        self._phase4_speculative_candidate: dict | None = None
        self._phase4_start_symbol_index: int | None = None
        self._phase4_arm_symbol_index: int | None = None
        self._phase4_scheduled_rearm = False
        self._phase4_symbol_origin: int | None = None
        self._cp_end_bit: int | None = None
        self._e_tail: tuple[int, ...] = ()
        self._e_tail_start = 0
        # E ends Phase 4 once. After it, a run of 20 ones in the descrambled
        # stream is ordinary data or line idle, not a second E; acting on it
        # restarted the B1 bookkeeping and killed the upstream session.
        self._phase4_e_seen = False
        self._detector: V90STransitionDetector | None = None
        self._rx_next: int | None = None
        self._stage_timing_enabled = os.environ.get("V90_RX_STAGE_TIMING") == "1"
        self._stage_timing_rows: deque[dict[str, int]] = deque(maxlen=2048)
        self._cpt_trace_enabled = os.environ.get("V90_CPT_TRACE") == "1"
        self._cpt_candidate_trace: deque[dict[str, int | str]] = deque(maxlen=2048)
        self._cpt_trace_total = 0

    @property
    def state(self) -> str:
        if self.upstream.state == self.upstream.FAILED:
            return self.upstream.FAILED
        if self._detector is None:
            return self.upstream.state
        return self.transmitter.state

    @property
    def failure_reason(self) -> str | None:
        return self.transmitter.failure_reason

    @property
    def descriptor(self) -> dict | None:
        return self.transmitter.descriptor

    @property
    def transmit_ready(self) -> bool:
        """Whether Ja has arrived and digital Phase 3 has begun emitting."""
        return self._detector is not None

    def feed_pcm(self, samples, *, sample_at: int | None = None) -> list[dict]:
        if not self._stage_timing_enabled and not self._cpt_trace_enabled:
            return self._feed_pcm(samples, sample_at=sample_at)
        trace_started = (time.thread_time_ns()
                         if self._cpt_trace_enabled else 0)
        trace_before = self._cpt_trace_total
        receiver_before = self.upstream.receiver
        start_symbol = (None if receiver_before is None
                        else int(receiver_before.eq.k))
        if not self._stage_timing_enabled:
            try:
                return self._feed_pcm(samples, sample_at=sample_at)
            finally:
                self._record_cpt_feed_batch(
                    trace_started, trace_before, start_symbol)
        started = time.thread_time_ns()
        start_symbol = (0 if receiver_before is None else
                        receiver_before.eq.k)
        self._stage_cpu_ns = {"observation": 0, "phase4_control": 0,
                              "phase4_scanner": 0, "cpt_candidate_search": 0,
                              "cpt_candidate_replay": 0,
                              "cpt_candidate_advance": 0,
                              "phase4_profile_e": 0,
                              "profile_prepare": 0, "profile_select": 0,
                              "receiver_construct": 0, "phase4_fsm": 0,
                              "lapm_setup": 0,
                              "upstream_data_lapm": 0}
        try:
            return self._feed_pcm(samples, sample_at=sample_at)
        finally:
            receiver = self.upstream.receiver
            symbols = (0 if receiver is None else receiver.eq.k - start_symbol)
            self._stage_timing_rows.append({
                "total": time.thread_time_ns() - started,
                "symbols": int(symbols),
                "rx_sample": int(self._rx_next or 0),
                **self._stage_cpu_ns,
            })
            self._record_cpt_feed_batch(
                trace_started, trace_before,
                None if start_symbol is None else int(start_symbol))

    def stage_timing_report(self) -> dict | None:
        """Return bounded per-media-chunk callback CPU timings."""
        if not self._stage_timing_enabled:
            return None
        rows = list(self._stage_timing_rows)
        if not rows:
            return {"enabled": True, "count": 0, "capacity": 2048,
                    "stages": {}}
        stages = {}
        for name in ("total", "observation", "phase4_control", "phase4_scanner",
                     "cpt_candidate_search", "cpt_candidate_replay",
                     "cpt_candidate_advance", "phase4_profile_e",
                     "profile_prepare", "profile_select",
                     "receiver_construct", "phase4_fsm", "lapm_setup",
                     "upstream_data_lapm"):
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            worst = max(rows, key=lambda row: row[name])
            stages[name] = {
                "mean_ms": float(np.mean(values) / 1e6),
                "p95_ms": float(np.percentile(values, 95) / 1e6),
                "p99_ms": float(np.percentile(values, 99) / 1e6),
                "max_ms": float(worst[name] / 1e6),
                "max_symbols": int(worst["symbols"]),
                "max_rx_sample": int(worst["rx_sample"]),
            }
        slow_rows = sorted(
            (row for row in rows
             if row["total"] >= self.STAGE_TIMING_SLOW_CALLBACK_NS),
            key=lambda row: row["total"], reverse=True
        )[:self.STAGE_TIMING_SLOW_CALLBACK_LIMIT]
        slow_callbacks = []
        for row in slow_rows:
            slow_callbacks.append({
                "total_ms": float(row["total"] / 1e6),
                "symbols": int(row["symbols"]),
                "rx_sample": int(row["rx_sample"]),
                **{f"{name}_ms": float(row[name] / 1e6)
                   for name in row
                   if name not in ("total", "symbols", "rx_sample")},
            })
        return {"enabled": True, "count": len(rows), "capacity": 2048,
                "slow_callback_threshold_ms": (
                    self.STAGE_TIMING_SLOW_CALLBACK_NS / 1e6),
                "slow_callback_count": len(slow_callbacks),
                "slow_callback_capacity": self.STAGE_TIMING_SLOW_CALLBACK_LIMIT,
                "slow_callbacks": slow_callbacks,
                "stages": stages}

    def cpt_candidate_trace_report(self) -> dict | None:
        """Return the opt-in, low-overhead CPt candidate-bank timing trace."""
        if not self._cpt_trace_enabled:
            return None
        rows = list(self._cpt_candidate_trace)
        if not rows:
            return {"enabled": True, "count": 0, "capacity": 2048,
                    "rows": []}
        worst = max(rows, key=lambda row: int(row["cpu_ns"]))
        return {"enabled": True, "count": len(rows), "capacity": 2048,
                "max_cpu_ns": int(worst["cpu_ns"]),
                "max_kind": str(worst["kind"]),
                "max_candidate_count": int(worst["candidate_count"]),
                "max_work_symbols": int(worst["work_symbols"]),
                "max_receive_symbol": int(worst["receive_symbol"]),
                "rows": rows}

    def _record_cpt_candidate_timing(self, kind: str, symbol_index: int,
                                     candidate_count: int, work_symbols: int,
                                     started_ns: int) -> None:
        if self._cpt_trace_enabled:
            self._cpt_candidate_trace.append({
                "kind": kind,
                "receive_symbol": int(symbol_index),
                "candidate_count": int(candidate_count),
                "work_symbols": int(work_symbols),
                "cpu_ns": int(time.thread_time_ns() - started_ns),
            })
            self._cpt_trace_total += 1

    def _record_cpt_feed_batch(self, started_ns: int, trace_before: int,
                               start_symbol: int | None) -> None:
        if not self._cpt_trace_enabled:
            return
        trace_count = self._cpt_trace_total - trace_before
        if trace_count <= 0:
            return
        rows = list(self._cpt_candidate_trace)[-trace_count:]
        receiver = self.upstream.receiver
        end_symbol = (start_symbol if receiver is None else
                      int(receiver.eq.k))
        self._cpt_candidate_trace.append({
            "kind": "feed_batch",
            "receive_symbol": int(end_symbol or 0),
            "symbol_start": int(start_symbol or 0),
            "batch_symbols": (0 if start_symbol is None or receiver is None
                              else max(0, int(receiver.eq.k) - start_symbol)),
            "candidate_calls": int(trace_count),
            "candidate_count": max(
                int(row["candidate_count"]) for row in rows),
            "work_symbols": sum(int(row["work_symbols"]) for row in rows),
            "cpu_ns": int(time.thread_time_ns() - started_ns),
        })
        self._cpt_trace_total += 1

    def _feed_pcm(self, samples, *, sample_at: int | None = None) -> list[dict]:
        """Feed contiguous caller-side analogue PCM and route protocol events."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("V.90 Phase 3 receive PCM must be finite")
        start = self._rx_next if sample_at is None else int(sample_at)
        if start is None:
            start = 0 if sample_at is None else int(sample_at)
        if self._rx_next is not None and start != self._rx_next:
            raise ValueError("V.90 Phase 3 PCM discontinuity; restart after gap recovery")
        self._rx_next = start + len(x)
        if not len(x):
            return []

        if self._detector is None:
            descriptors = self.upstream.feed(x)
            self._feed_phase4_symbols()
            if descriptors:
                descriptor = descriptors[0]
                self.upstream.receiver.hold_transition_feedback()
                self.transmitter.set_dil_descriptor(descriptor)
                self.transmitter.start()
                self.events.append(("ja-accepted", descriptor))
                # Ja was completed within this chunk. The next protocol S
                # event is later (after digital TRN1d/Jd), so arm detection at
                # the following media sample instead of rescanning old S.
                self._detector = V90STransitionDetector(
                    self.upstream.receive_baud, self.upstream.receive_carrier)
                self._detector.reset(sample_at=start + len(x))
            return descriptors

        # The S transitions belong to Phase 3/DIL. Once the Phase 4 receiver
        # has been armed, the remaining V.34 upstream path needs the equalizer
        # and decoder but no further S-boundary search. Avoid repeatedly
        # re-running the whole-window FFT/convolution on every RTP packet.
        found = ([] if self._phase4_receive_armed else
                 self._detector.feed(x, sample_at=start))
        # Keep the equalizer/timing tracker alive after Ja. The S detector is
        # only for Phase 3 event boundaries; it does not replace Phase 4 data
        # symbol decisions.
        self.upstream.feed(x)
        if self.phase4 is None:
            self._feed_phase4_symbols()
        for signal, at in found:
            state = self.transmitter.state
            # V.90 §9.3.1.5: S onset while repeating Jd triggers J′d.
            # §9.3.1.6: the later S-to-S-bar transition ends DIL.
            if signal == "S" and state == self.transmitter.JD:
                self.transmitter.notify_s_transition()
                self._ignore_initial_phase3_flip = True
                self.transitions.append(at)
                self.transition_events.append((signal, at))
                self.events.append(("s-detected", at))
            elif signal == "flip" and self._ignore_initial_phase3_flip:
                # The first S-to-S-bar edge belongs to the caller's required
                # 128T S response and following 16T S-bar, after J′d. It is
                # not the later DIL-completion edge.
                self._ignore_initial_phase3_flip = False
                self.events.append(("initial-s-to-sbar", at))
                if (self.phase4 is not None
                        and self.transmitter.state == self.transmitter.PHASE4_READY):
                    self._arm_phase4_receiver()
            elif signal == "flip" and state == self.transmitter.DIL:
                self.transmitter.notify_s_transition()
                self.transitions.append(at)
                self.transition_events.append((signal, at))
                self.events.append(("s-transition", at))
        return []

    def _feed_phase4_symbols(self) -> None:
        # CPt resets its GPA/differential state. Do not feed silence, S markers,
        # or Phase 3 symbols into that decoder; locate the CPt start from input
        # power and accept a candidate only after its CRC validates.
        if self.phase4 is None or not self._phase4_receive_armed:
            # These observations belong to Phase 3/DIL or the caller's
            # post-J′d S-bar marker. Drain them until CPt receive is armed.
            self.upstream.read_new_equalized_observations()
            return
        observations = self.upstream.read_new_equalized_observations()
        for observation in observations:
            self._process_phase4_observation(observation)

    def _process_phase4_observation(self, observation) -> None:
        timed = getattr(self, "_stage_timing_enabled", False)
        stage_started = time.thread_time_ns() if timed else 0
        symbol_index, equalized, decision, power = observation
        if self._capture_symbol_trace:
            self._phase4_equalized_recent.append((
                int(symbol_index), float(complex(equalized).real),
                float(complex(equalized).imag)))
        if (self.upstream_data is not None
                and self.upstream_data.state == self.upstream_data.QUALIFYING
                and self.upstream_data.b1_start_symbol is not None
                and int(symbol_index) >= self.upstream_data.b1_start_symbol
                and self._capture_symbol_trace
                and len(self._upstream_b1_trace) < 512):
            self._upstream_b1_trace.append((
                int(symbol_index), float(complex(equalized).real),
                float(complex(equalized).imag)))
        if self._capture_symbol_trace:
            self._phase4_decision_trace.append((
                int(self._rx_next or 0), int(symbol_index),
                int(decision[0]), int(decision[1])))
        guard = self.upstream.receiver.eq._input_guard
        reference = None if guard is None else guard.reference
        threshold = (None if reference is None else
                     float(reference) * max(float(guard.EXIT_RATIO),
                                             self.CPT_ACTIVITY_RATIO))
        active = (threshold is not None and power is not None
                  and np.isfinite(power) and power >= threshold)
        if self._capture_symbol_trace:
            self._phase4_observation_trace.append({
                "symbol": int(symbol_index),
                "decision": [int(decision[0]), int(decision[1])],
                "power": None if power is None else float(power),
                "power_reference": reference,
                "active_threshold": threshold,
                "active": bool(active),
            })
        if timed:
            self._stage_cpu_ns["observation"] += time.thread_time_ns() - stage_started
            stage_started = time.thread_time_ns()
        if self._phase4_cpt_acquired:
            bit_start = self.phase4_receiver.bits_seen
            scanner_started = time.thread_time_ns() if timed else 0
            # Decisions came from the receiver's own validated slicer; avoid
            # rebuilding a singleton list and repeating the public API's
            # input-type/range checks for every symbol in the media callback.
            frames = self.phase4_receiver._feed_symbol_unchecked(
                *decision, empty_none=True) or ()
            if timed:
                self._stage_cpu_ns["phase4_scanner"] += time.thread_time_ns() - scanner_started
                scanner_started = time.thread_time_ns()
            for profile in frames:
                self.receive_phase4_profile(profile)
            self._detect_phase4_e(self.phase4_receiver.last_descrambled_bits,
                                  bit_start, symbol_index)
            if timed:
                self._stage_cpu_ns["phase4_profile_e"] += time.thread_time_ns() - scanner_started
        else:
            # CPt is repeated until accepted, so search the continuous
            # descrambled symbol stream. A power-onset-only bank can miss it
            # when the preceding S-bar/training tail never falls below the
            # sensitive carrier guard threshold.
            speculative = self._phase4_speculative_decoder
            if speculative is not None:
                scanner_started = time.thread_time_ns() if timed else 0
                frames = speculative.feed([decision])
                if timed:
                    self._stage_cpu_ns["phase4_scanner"] += time.thread_time_ns() - scanner_started
                bps = 4 if self.phase4_constellation == 16 else 2
                origin = self._phase4_speculative_origin_symbol
                for profile in frames:
                    if profile.get("kind") != "cpt":
                        continue
                    start_bit = int(profile.get("start_bit", 0))
                    if (start_bit % bps or origin is None
                            or origin + start_bit // bps
                            < int(self._phase4_arm_symbol_index or 0) + 64):
                        continue
                    candidate = self._phase4_speculative_candidate
                    if candidate is None:
                        candidate = {"decoder": speculative,
                                     "symbol_origin": origin,
                                     "wire_cpt_profile": None}
                        self._phase4_speculative_candidate = candidate
                    profile_started = time.thread_time_ns() if timed else 0
                    self._accept_phase4_cpt_candidate(candidate, [profile])
                    if timed:
                        self._stage_cpu_ns["phase4_profile_e"] += time.thread_time_ns() - profile_started
                    if self._phase4_cpt_acquired:
                        break
            if not self._phase4_cpt_acquired:
                candidate_started = time.thread_time_ns() if timed else 0
                self._acquire_phase4_cpt(decision, power, symbol_index)
                if timed:
                    self._stage_cpu_ns["cpt_candidate_search"] += time.thread_time_ns() - candidate_started
        if timed:
            self._stage_cpu_ns["phase4_control"] += time.thread_time_ns() - stage_started
            stage_started = time.thread_time_ns()
        if self.upstream_data is not None:
            decoded = self.upstream_data.feed_symbol(
                symbol_index, equalized)
            if self.upstream_data.state == self.upstream_data.DATA:
                self._activate_upstream_data_tracking()
            if self.lapm is not None and decoded:
                self.lapm.feed_line_bits(decoded)
                self._received_dte.extend(self.lapm.take_received())
            else:
                self._upstream_data_bits.extend(decoded)
        if timed:
            self._stage_cpu_ns["upstream_data_lapm"] += time.thread_time_ns() - stage_started

    def _activate_upstream_data_tracking(self) -> None:
        if getattr(self, "_upstream_data_tracking", False):
            return
        receiver = self.upstream_data.receiver
        taps = self.upstream_data.equalizer_taps
        # B1 qualification estimates the complex receive gain. Payload
        # decisions and decision-directed equalizer feedback must both see
        # mapper-domain amplitudes. Previously only the optional refined-tap
        # branch absorbed that gain into h; without refinement the decoder
        # normalized its input, but Receiver.decide() fed unnormalized targets
        # back to the equalizer and drove it away from the trained solution.
        eq = self.upstream.receiver.eq
        eq.h = np.asarray(eq.h if taps is None else taps,
                          dtype=np.complex128) / self.upstream_data._gain
        self.upstream_data._gain = 1 + 0j
        self.upstream.begin_data_tracking(receiver)
        self._upstream_data_tracking = True

    def _acquire_phase4_cpt(self, decision: tuple[int, int],
                            power: float | None,
                            symbol_index: int | None = None) -> None:
        batch_enabled = symbol_index is not None
        if symbol_index is None:
            receiver = self.upstream.receiver
            symbol_index = 0 if receiver is None else len(receiver.indices)
        if (not self._phase4_scheduled_rearm
                and self._phase4_start_symbol_index is not None):
            target = (self._phase4_start_symbol_index
                      + int(round(self.CPT_EXPECTED_DELAY_SAMPLES
                                   * self.baud / 8000.0))
                      - self.CPT_START_OFFSET
                      - self.CPT_START_TOLERANCE)
            if symbol_index >= target:
                self._phase4_cpt_candidates.clear()
                self._phase4_cpt_recent.clear()
                self._phase4_cpt_active_run = None
                self._phase4_cpt_quiet_count = 12
                self._phase4_scheduled_rearm = True
        self._phase4_cpt_recent.append((decision, power, symbol_index))
        if len(self._phase4_cpt_recent) > self.CPT_START_HISTORY:
            del self._phase4_cpt_recent[:-self.CPT_START_HISTORY]
        guard = self.upstream.receiver.eq._input_guard
        reference = None if guard is None else guard.reference
        active = (reference is not None and power is not None
                  and np.isfinite(power)
                  and power >= reference * max(
                      guard.EXIT_RATIO, self.CPT_ACTIVITY_RATIO))
        if not active:
            self._phase4_cpt_quiet_count += 1
            if self._phase4_cpt_quiet_count >= 12:
                self._phase4_cpt_candidates.clear()
                self._phase4_cpt_active_run = None
            if self._phase4_cpt_candidates:
                candidate_count = len(self._phase4_cpt_candidates)
                work_symbols = 0
                advance_started = (time.thread_time_ns()
                                   if self._cpt_trace_enabled else 0)
                for candidate in self._phase4_cpt_candidates:
                    candidate["age"] += 1
                    pending = candidate["pending"]
                    pending.append(decision)
                    if ((not batch_enabled
                         or len(pending) >= self.CPT_CANDIDATE_BATCH_SYMBOLS)
                            or candidate["age"] > self.CPT_CANDIDATE_MAX_SYMBOLS):
                        work_symbols += len(pending)
                        candidate["decoder"]._feed_symbols_unchecked(
                            pending, capture_last_bits=False, empty_none=True)
                        pending.clear()
                if self._cpt_trace_enabled:
                    self._record_cpt_candidate_timing(
                        "quiet_advance", symbol_index, candidate_count,
                        work_symbols, advance_started)
                self._phase4_cpt_candidates = [
                    item for item in self._phase4_cpt_candidates
                    if item["age"] <= self.CPT_CANDIDATE_MAX_SYMBOLS]
            if self._phase4_cpt_active_run is not None:
                self._phase4_cpt_active_run += 1
            return

        # The threshold leads CPt's first symbol center by about 13 symbols
        # with the 8T pulse shape and +/-6T receive window. Try a narrow set
        # around that boundary, plus a few late-threshold fallbacks.
        if self._phase4_cpt_active_run is None:
            self._phase4_cpt_candidates.clear()
            recent = list(self._phase4_cpt_recent)
            replay_started = (time.thread_time_ns()
                              if self._stage_timing_enabled
                              or self._cpt_trace_enabled else 0)
            replay_work_symbols = 0
            for start in range(len(recent)):
                decoder = V90CpSymbolDecoder(self.phase4_constellation)
                replay = recent[start:]
                replay_work_symbols += len(replay)
                frames = decoder._feed_symbols_unchecked(
                    [item[0] for item in replay], capture_last_bits=False,
                    empty_none=True)
                candidate = {"decoder": decoder, "age": len(recent) - start,
                             "symbol_origin": recent[start][2],
                             "pending": []}
                if self._accept_phase4_cpt_candidate(candidate, frames):
                    if self._stage_timing_enabled:
                        self._stage_cpu_ns["cpt_candidate_replay"] += (
                            time.thread_time_ns() - replay_started)
                    if self._cpt_trace_enabled:
                        self._record_cpt_candidate_timing(
                            "replay", symbol_index, start + 1,
                            replay_work_symbols, replay_started)
                    return
                self._phase4_cpt_candidates.append(candidate)
            if self._stage_timing_enabled:
                self._stage_cpu_ns["cpt_candidate_replay"] += (
                    time.thread_time_ns() - replay_started)
            if self._cpt_trace_enabled:
                self._record_cpt_candidate_timing(
                    "replay", symbol_index, len(recent), replay_work_symbols,
                    replay_started)
            self._phase4_cpt_active_run = 0
            self._phase4_cpt_quiet_count = 0
            return

        self._phase4_cpt_active_run += 1
        self._phase4_cpt_quiet_count = 0
        advance_started = (time.thread_time_ns()
                           if self._stage_timing_enabled
                           or self._cpt_trace_enabled else 0)
        candidate_count = len(self._phase4_cpt_candidates)
        work_symbols = 0
        candidates_scanned = 0
        for candidate in self._phase4_cpt_candidates:
            candidates_scanned += 1
            candidate["age"] += 1
            pending = candidate["pending"]
            pending.append(decision)
            frames = None
            if ((not batch_enabled
                 or len(pending) >= self.CPT_CANDIDATE_BATCH_SYMBOLS)
                    or candidate["age"] > self.CPT_CANDIDATE_MAX_SYMBOLS):
                work_symbols += len(pending)
                frames = candidate["decoder"]._feed_symbols_unchecked(
                    pending, capture_last_bits=False, empty_none=True)
                pending.clear()
            if self._accept_phase4_cpt_candidate(candidate, frames):
                if self._stage_timing_enabled:
                    self._stage_cpu_ns["cpt_candidate_advance"] += (
                        time.thread_time_ns() - advance_started)
                if self._cpt_trace_enabled:
                    self._record_cpt_candidate_timing(
                        "advance", symbol_index, candidates_scanned,
                        work_symbols,
                        advance_started)
                return
        low_start = self.CPT_START_OFFSET - self.CPT_START_TOLERANCE
        high_start = self.CPT_START_OFFSET + self.CPT_START_TOLERANCE
        if low_start <= self._phase4_cpt_active_run <= high_start:
            decoder = V90CpSymbolDecoder(self.phase4_constellation)
            frames = decoder._feed_symbol_unchecked(
                *decision, capture_last_bits=False, empty_none=True)
            candidate = {"decoder": decoder, "age": 1,
                         "symbol_origin": symbol_index, "pending": []}
            if self._accept_phase4_cpt_candidate(candidate, frames):
                return
            self._phase4_cpt_candidates.append(candidate)
        self._phase4_cpt_candidates = [
            item for item in self._phase4_cpt_candidates
            if item["age"] <= self.CPT_CANDIDATE_MAX_SYMBOLS]
        if self._stage_timing_enabled:
            self._stage_cpu_ns["cpt_candidate_advance"] += (
                time.thread_time_ns() - advance_started)
        if self._cpt_trace_enabled:
            self._record_cpt_candidate_timing(
                "advance", symbol_index, candidate_count, work_symbols,
                advance_started)
        # The FIR window sees the pulse ramp before the first CP symbol center,
        # so do not release feedback at the raw power threshold. At the
        # measured sync boundary, 4-point decisions support full reacquisition;
        # for 16-point CPt enable only carrier/timing tracking until CRC
        # validates the profile, keeping NLMS taps frozen during ambiguous
        # decisions.
        if (not self._phase4_feedback_released_early
                and self._phase4_cpt_active_run
                == self.CPT_FEEDBACK_RELEASE_SYMBOLS):
            if self.phase4_constellation == 4:
                self.upstream.receiver.resume_phase4_feedback()
            else:
                self.upstream.receiver.begin_phase4_carrier_timing_acquisition()
            self._phase4_feedback_released_early = True

    def _accept_phase4_cpt_candidate(self, candidate: dict,
                                     frames: list[dict] | None) -> bool:
        if not frames:
            return False
        decoder = candidate["decoder"]
        for profile in frames:
            if profile.get("kind") != "cpt":
                continue
            # A single CRC-valid CPt can be an early speculative alignment.
            # The wire repeats CPt until the response is selected, so require
            # two matching, non-overlapping frames from this same decoder
            # before the production path changes Phase 4 state. This keeps
            # confirmations from neighboring candidate offsets independent.
            frame_start = profile.get("start_bit")
            frame_bits = profile.get("frame_bits")
            if (not isinstance(frame_start, int)
                    or not isinstance(frame_bits, int) or frame_bits <= 0):
                continue
            previous = candidate.get("wire_cpt_profile")
            if previous is None:
                candidate["wire_cpt_profile"] = dict(profile)
                self.events.append(("phase4-cpt-candidate", {
                    "start_bit": frame_start, "frame_bits": frame_bits}))
                continue
            previous_end = (int(previous["start_bit"])
                            + int(previous["frame_bits"]))
            same_profile = (
                V90DigitalAnswererPhase4Session._profile_key(profile)
                == V90DigitalAnswererPhase4Session._profile_key(previous))
            if not same_profile or frame_start < previous_end:
                candidate["wire_cpt_profile"] = dict(profile)
                continue
            self.phase4_receiver = decoder
            self._phase4_symbol_origin = int(candidate["symbol_origin"])
            self._phase4_cpt_candidates.clear()
            self._phase4_cpt_acquired = True
            self.events.append(("phase4-cpt-confirmed", {
                "first_start_bit": int(previous["start_bit"]),
                "second_start_bit": frame_start,
                "frame_bits": frame_bits}))
            self.receive_phase4_profile(profile)
            return True
        return False

    def _detect_phase4_e(self, bits, bit_start: int,
                         symbol_index: int | None = None) -> bool:
        """Accept E only after selected CP and a transmitted MP′.

        V.90 delegates the 20-bit E sequence to V.34. Its binary-one check
        belongs in the descrambled bit domain, not in PCM or symbol amplitude.
        The CP boundary and local MP′ output gate prevent ordinary
        training/control runs of ones from being mistaken for completion.
        """
        phase4 = self.phase4
        if self._phase4_e_seen:
            self._e_tail = ()
            return False
        if (phase4 is None or self._cp_end_bit is None
                or phase4.state not in (
                    phase4.transmitter.WAIT_CP_PRIME,
                    phase4.transmitter.DATA_READY)
                or phase4.transmitter.mp_prime_frames_sent == 0):
            self._e_tail = ()
            return False

        current = tuple(int(bit) for bit in bits)
        if any(bit not in (0, 1) for bit in current):
            raise ValueError("V.90 E detector input must contain bits")
        if self._e_tail:
            combined = self._e_tail + current
            combined_start = self._e_tail_start
        else:
            combined = current
            combined_start = int(bit_start)
        for offset in range(max(0, len(combined) - 19)):
            if (combined_start + offset >= self._cp_end_bit
                    and combined[offset:offset + 20] == (1,) * 20):
                self._phase4_e_seen = True
                phase4.receive_e()
                bps = 2 if self.phase4_constellation == 4 else 4
                e_end_bit = combined_start + offset + 20
                if e_end_bit % bps:
                    self.phase4_error = "V.90 E boundary is not aligned to a symbol"
                    self.events.append(("upstream-b1-not-started", self.phase4_error))
                elif self._phase4_symbol_origin is None or self.upstream_data is None:
                    self.phase4_error = "V.90 E boundary lacks a selected upstream profile"
                    self.events.append(("upstream-b1-not-started", self.phase4_error))
                else:
                    b1_start = self._phase4_symbol_origin + e_end_bit // bps
                    self.upstream_data.start_b1(b1_start)
                    if self._capture_symbol_trace:
                        self._upstream_b1_trace = list(
                            self._phase4_equalized_recent)
                    self.upstream.hold_b1_feedback()
                    self.events.append(("upstream-b1-start", b1_start))
                self._e_tail = ()
                self.events.append(("phase4-e-detected", combined_start + offset))
                return True

        keep = min(19, len(combined))
        self._e_tail = combined[-keep:] if keep else ()
        self._e_tail_start = combined_start + len(combined) - keep
        return False

    def receive_phase4_profile(self, profile: dict) -> bool:
        """Queue a parsed CPt/CP/CP-prime event and apply it when active.

        The normal path supplies profiles from ``phase4_receiver``. This seam
        also lets a media adapter provide an independently qualified parser
        result without replacing PCM timing/equalizer tracking.
        """
        if not isinstance(profile, dict) or profile.get("kind") not in ("cpt", "cp"):
            raise ValueError("expected a parsed V.90 CPt/CP profile")
        self.phase4_profiles.append(dict(profile))
        self.events.append(("phase4-profile", dict(profile)))
        if self.phase4 is None:
            return False
        self._phase4_profile_cursor += 1
        return self._apply_phase4_profile(profile)

    def _start_phase4(self) -> None:
        if self.phase4 is not None:
            return
        self._phase4_cpt_acquired = False
        self._phase4_cpt_candidates.clear()
        self._phase4_cpt_quiet_count = 0
        self._phase4_cpt_active_run = None
        self._phase4_feedback_released_early = False
        self._phase4_cpt_recent.clear()
        self._phase4_speculative_decoder = None
        self._phase4_speculative_origin_symbol = None
        self._phase4_speculative_candidate = None
        if self.phase4_constellation == 16:
            # Four-point decisions before the DIL boundary cannot be promoted
            # into 16-point symbols. Reset before the analogue R transition
            # and CPt response; the retained waveform still needs to qualify
            # this boundary assumption for 16-point callers.
            self.phase4_receiver = V90CpSymbolDecoder(16)
            self._e_tail = ()
        self.phase4 = V90DigitalAnswererPhase4Session(
            uinfo=self.transmitter.uinfo, law=self.law, constellation=4,
            mp=self.phase4_mp_profile)
        self.phase4.start()
        receiver = self.upstream.receiver
        # Phase 4 observations and CP candidates use len(receiver.indices).
        # SymbolEq.k includes its 432-symbol preamble origin, while this list
        # counts only tracked decisions; both advance together after startup,
        # but their absolute values differ by that fixed origin. Keep scheduled
        # CPt references in the observation domain used below.
        self._phase4_start_symbol_index = (
            None if receiver is None else len(receiver.indices))
        self.events.append(("phase4-started", dict(self.phase4_mp_profile)))
        # Ri transmission begins at the J′d/DIL boundary. On zero-DIL startup,
        # receive conditioning waits for the caller's following 16T S-bar
        # marker; CPt follows the required 192T Ri interval.
        if not self._ignore_initial_phase3_flip:
            self._arm_phase4_receiver()
        while self._phase4_profile_cursor < len(self.phase4_profiles):
            profile = self.phase4_profiles[self._phase4_profile_cursor]
            self._phase4_profile_cursor += 1
            self._apply_phase4_profile(profile)

    def _arm_phase4_receiver(self) -> None:
        if self._phase4_receive_armed:
            return
        self.upstream.begin_phase4(self.phase4_constellation)
        self.upstream.set_symbol_callback(self._process_phase4_observation)
        receiver = self.upstream.receiver
        self._phase4_decoder_origin_symbol = (
            None if receiver is None else len(receiver.indices))
        self._phase4_speculative_decoder = V90CpSymbolDecoder(
            self.phase4_constellation)
        self._phase4_speculative_origin_symbol = self._phase4_decoder_origin_symbol
        self._phase4_speculative_candidate = {
            "decoder": self._phase4_speculative_decoder,
            "symbol_origin": self._phase4_speculative_origin_symbol,
            "wire_cpt_profile": None,
        }
        self._phase4_arm_symbol_index = (
            None if receiver is None else len(receiver.indices))
        self._phase4_receive_armed = True

    def _apply_phase4_profile(self, profile: dict) -> bool:
        if self.phase4 is None:
            return False
        timed = getattr(self, "_stage_timing_enabled", False)
        profile_started = time.thread_time_ns() if timed else 0
        selected_upstream_rate = None
        selected_upstream_profile = None
        selected_upstream_receiver = None
        duplicate_cp = (
            profile.get("kind") == "cp" and profile.get("ack") == 0
            and self.phase4.state == self.phase4.transmitter.WAIT_CP_PRIME
            and self.phase4._cp_profile is not None
            and self.phase4._profile_key(profile)
            == self.phase4._profile_key(self.phase4._cp_profile))
        if (profile.get("kind") == "cp" and profile.get("ack") == 0
                and not duplicate_cp):
            try:
                stage_started = time.thread_time_ns() if timed else 0
                selected_upstream_profile = select_analog_upstream_profile(
                    profile, self.phase4_mp_profile,
                    baud=self.baud, carrier=self.carrier)
                if timed:
                    self._stage_cpu_ns["profile_select"] += time.thread_time_ns() - stage_started
                selected_upstream_rate = selected_upstream_profile.rate
                # V.90 assigns GPA (the V.34 answerer polynomial) to the
                # analogue modem even though it is normally the caller. Do
                # not use the generic V.34 caller-role default here: B1 and
                # data descrambling follow the modem type's polynomial.
                stage_started = time.thread_time_ns() if timed else 0
                selected_upstream_receiver = V90UpstreamDataReceiver(
                    selected_upstream_profile, remote_role="answer")
                if timed:
                    self._stage_cpu_ns["receiver_construct"] += time.thread_time_ns() - stage_started
                if timed:
                    self._stage_cpu_ns["profile_prepare"] += time.thread_time_ns() - profile_started
            except ValueError as exc:
                if timed:
                    self._stage_cpu_ns["profile_prepare"] += time.thread_time_ns() - profile_started
                self.phase4_error = str(exc)
                self.events.append(("phase4-profile-rejected", self.phase4_error))
                return False
        elif timed:
            self._stage_cpu_ns["profile_prepare"] += time.thread_time_ns() - profile_started
        stage_started = time.thread_time_ns() if timed else 0
        try:
            accepted = self.phase4.receive_profile(profile)
            if timed:
                self._stage_cpu_ns["phase4_fsm"] += time.thread_time_ns() - stage_started
        except (ValueError, RuntimeError) as exc:
            if timed:
                self._stage_cpu_ns["phase4_fsm"] += time.thread_time_ns() - stage_started
            self.phase4_error = str(exc)
            self.events.append(("phase4-profile-rejected", self.phase4_error))
            return False
        if accepted:
            self.events.append(("phase4-profile-accepted", profile))
            if selected_upstream_rate is not None:
                self.analog_upstream_rate = selected_upstream_rate
                self.analog_upstream_profile = selected_upstream_profile
                self.upstream_data = selected_upstream_receiver
                self._upstream_data_tracking = False
                self.events.append(("analog-upstream-rate-selected",
                                    asdict(selected_upstream_profile)))
                if self.enable_lapm:
                    # CP selects the digital downstream data clock. The
                    # analogue upstream has its separately negotiated rate
                    # above; V42Layer's bit timer advances on local TX bits.
                    stage_started = time.thread_time_ns() if timed else 0
                    from v90_lapm import V90LapmLink
                    self.lapm = V90LapmLink(
                        rate=int(round(float(profile["rate"]))),
                        calling_party=False, detect=self.lapm_detect)
                    if self._queued_dte:
                        self.lapm.feed_bytes(bytes(self._queued_dte))
                        self._queued_dte.clear()
                    self.events.append(("lapm-started", {
                        "downstream_rate": int(round(float(profile["rate"]))),
                        "upstream_rate": selected_upstream_rate,
                    }))
                    if timed:
                        self._stage_cpu_ns["lapm_setup"] += time.thread_time_ns() - stage_started
            if profile.get("kind") == "cpt" and self.upstream.receiver is not None:
                # The answerer's Ri and the caller's turnaround can leave a
                # silent gap before CPt. Do not reopen decision feedback until
                # the receive decoder has CRC-validated the first QAM frame.
                self.upstream.receiver.resume_phase4_feedback()
                self._phase4_cpt_acquired = True
            if profile.get("kind") == "cp" and profile.get("ack") == 0:
                start = profile.get("start_bit")
                length = profile.get("frame_bits")
                if isinstance(start, int) and isinstance(length, int):
                    self._cp_end_bit = start + length
                    self._e_tail = ()
        return accepted

    def read_upstream_bits(self, count: int | None = None) -> list[int]:
        """Drain decoded, descrambled analogue-upstream bits for LAPM."""
        if count is None:
            count = len(self._upstream_data_bits)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("upstream bit count must be a nonnegative integer")
        take = min(count, len(self._upstream_data_bits))
        return [self._upstream_data_bits.popleft() for _ in range(take)]

    def feed_bytes(self, data: bytes) -> None:
        """Queue DTE data for negotiated V.90 downstream transmission."""
        payload = bytes(data)
        if self.lapm is not None:
            self.lapm.feed_bytes(payload)
            return
        if len(self._queued_dte) + len(payload) > 65536:
            raise OverflowError("V.90 pre-LAPM DTE queue exceeds 64 KiB")
        self._queued_dte.extend(payload)

    def read_received_data(self) -> bytes:
        """Drain DTE bytes delivered by the V.90 upstream LAPM link."""
        data = bytes(self._received_dte)
        self._received_dte.clear()
        return data

    @property
    def pending_bits(self) -> int:
        if self.lapm is not None:
            return self.lapm.pending_bits()
        return 8 * len(self._queued_dte)

    def read_g711(self, count: int, law: str | None = None, *,
                  data_bit_source: Callable[[int], Sequence[int]] | None = None) -> bytes:
        """Return exact-law codewords and optionally pull Phase 4 data bits.

        ``data_bit_source`` is called only after B1d, when the Phase 4
        transmitter needs another complete mapper-frame run. This lets the
        enclosing LAPM layer remain the owner of framing and sequence state.
        """
        if law is not None and law.lower() != self.law:
            raise ValueError("G.711 law changed after V.90 negotiation")
        if self._detector is None:
            raise RuntimeError("V.90 digital Phase 3 has not started from Ja")
        payload = self.transmitter.read(count)
        if self.transmitter.state == self.transmitter.FAILED:
            reason = self.transmitter.failure_reason or "V.90 Phase 3 failed"
            if not any(name == "phase3-s-timeout" for name, _ in self.events):
                self.events.append(("phase3-s-timeout", reason))
            if not payload:
                raise RuntimeError(reason)
        if self.transmitter.phase4_ready:
            self._start_phase4()
        if self.phase4 is not None and len(payload) < count:
            if data_bit_source is None and self.lapm is not None:
                data_bit_source = self.lapm.read_tx_bits
            payload += self.phase4.read_g711(
                count - len(payload), self.law,
                data_bit_source=data_bit_source)
        return payload
