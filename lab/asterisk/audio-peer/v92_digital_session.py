"""A V.92 digital-answerer session for the live engine's ``phase3_factory`` hook: Phase 3, Phase 4, data.

It has the surface ``V90DigitalAnswererPhase2Engine`` uses from its Phase 3 session (``feed_pcm``, ``read_g711``,
``transmit_ready``, ``state``, ``phase4`` with a ``state`` and ``transmitter.cp_profile``, ``lapm``, ``feed_bytes``,
``read_received_data``, ``pending_bits``) and runs the offline pieces as streams:

* ``v92_phase3_exchange.DigitalSide`` for Phase 3 (Ru .. Ja, Sd .. Jd, Su / Jp / Jp', DIL or SCR, Ri, CPt, R-bar-i, E1u);
* ``v92_phase4_controller.digital_side`` for Phase 4 (TRN2d, SUVd, CPd, E, B1d / B1u), reading the analogue modem's
  Phase 4 from the sample where its E1u ended;
* then data: downstream through the V.90 mapper with the parameters of the analogue modem's CPu, upstream through the
  PCM-upstream receiver with the parameters of our CPd, both carrying a V.42 LAPM link (``V90LapmLink``) so the DTE sees a
  reliable byte stream.

Optional training-based equalisation supplies a receive filter and quantised CPd precoder.
Rate changes can reuse this model within its constellation capacity. A changed channel
requires a new training/design cycle; independent-peer and real-line limits are in the gap map.
"""

from __future__ import annotations

import os
from functools import partial
from dataclasses import replace
from math import prod
import sys
import threading
import traceback
from types import SimpleNamespace

import numpy as np

import v92_data_mode as dm
import v92_phase3_exchange as p3x
import v92_phase4_controller as p4c
import v92_fpe_signals
from v92_phase3_rx import PatternTracker
from v90_lapm import V90LapmLink

DATA_READY = "data-ready"
ESTIMATE_BIAS_ISI = 0.004



def _cpd_gain4(cpd_constellation_mode: str) -> int | None:
    """CPd 4G: EL_MODEM_V92_CPD_GAIN4 if set; else 0x0300, validated with the SmartLink peer, for the adaptive and
    expanded10 constellations; else None, which makes the digital side send G = 1/LU."""
    raw = os.environ.get("EL_MODEM_V92_CPD_GAIN4")
    if raw is not None:
        return int(raw, 0)
    return 0x0300 if cpd_constellation_mode in ("adaptive", "expanded10") else None

class V92DigitalSession:
    def __init__(self, *, law: str, uinfo: int = 67, rate_mask: int = 0x3FFFFF, enable_lapm: bool = True,
                 ack_wait: int = 1600, drn_up: int = 9, equalise: bool = False, async_design: bool = False, phase3_equalise: bool = False,
                 fractional_rx_timing: bool = False,
                 fractional_rx_reacquisition: bool | None = None,
                 phase3_lu: float = 1000.0,
                 max_tx_rms: float = 6000.0,
                 pcm_upstream_max_lost_frames: int = 3,
                 digital_trn_symbols: int = 2040, b1u_rx_mode: str = "off",
                 cpd_constellation_mode: str = "default",
                 enable_rate_renegotiation_response: bool = False,
                 enable_fast_parameter_exchange: bool = False,
                 enable_fast_parameter_exchange_response: bool = False,
                 lapm_detect: bool = True,
                 v42bis_compression: tuple[int, int, int] | None = None,
                 v42_compression_recovery: bool = False,
                 empty_precoder_policy: str = "omit",
                 filter_limits: tuple[int, int] | None = None):
        # (Lmax, Ltot) from the peer's INFO1a; None = the minimum every analogue modem supports.
        self._filter_limits = filter_limits
        if (isinstance(pcm_upstream_max_lost_frames, bool)
                or not isinstance(pcm_upstream_max_lost_frames, int)
                or not 0 <= pcm_upstream_max_lost_frames <= 1024):
            raise ValueError("pcm_upstream_max_lost_frames must be an integer in 0..1024")
        if (isinstance(digital_trn_symbols, bool)
                or not isinstance(digital_trn_symbols, int)
                or digital_trn_symbols < 2040):
            raise ValueError("digital_trn_symbols must be an integer >= 2040")
        self.law = law
        self._pcm_upstream_max_lost_frames = pcm_upstream_max_lost_frames
        self._digital_trn_symbols = digital_trn_symbols
        if b1u_rx_mode not in ("off", "trn2u-fit", "trn2u-fit-viterbi"):
            raise ValueError("b1u_rx_mode must be off, trn2u-fit or trn2u-fit-viterbi")
        self._b1u_rx_mode = b1u_rx_mode
        if cpd_constellation_mode not in ("default", "sparse", "expanded10", "adaptive"):
            raise ValueError("cpd_constellation_mode must be default, sparse, expanded10 or adaptive")
        self._cpd_constellation_mode = cpd_constellation_mode
        if type(enable_rate_renegotiation_response) is not bool:
            raise TypeError("enable_rate_renegotiation_response must be bool")
        if type(enable_fast_parameter_exchange) is not bool:
            raise TypeError("enable_fast_parameter_exchange must be bool")
        if type(enable_fast_parameter_exchange_response) is not bool:
            raise TypeError("enable_fast_parameter_exchange_response must be bool")
        self._enable_rate_renegotiation_response = enable_rate_renegotiation_response
        self._enable_fpe = enable_fast_parameter_exchange or enable_fast_parameter_exchange_response
        self._fpe_detector: v92_fpe_signals.RmSequenceDetector | None = None
        self._fpe_detector_filter = None
        self._fpe_detector_front = None             # TRN2u-fit data mode: its FIR, following the live NLMS weights
        self._peer_reneg_tracker = (PatternTracker(families=("ru",))
                                    if enable_rate_renegotiation_response else None)
        self._peer_reneg_probe = np.empty(0, dtype=float)
        self._peer_reneg_probe_base = 0
        self._peer_reneg_total = 0
        self._peer_reneg_candidate: int | None = None
        self._peer_fpe_cutover: tuple[int, int, int, int] | None = None
        self.enable_lapm = enable_lapm
        self._lapm_detect = bool(lapm_detect)
        self._v42bis_compression = v42bis_compression
        self._v42_compression_recovery = bool(v42_compression_recovery)
        if empty_precoder_policy not in ("include", "omit", "identity"):
            raise ValueError("empty_precoder_policy must be include, omit, or identity")
        self._empty_precoder_policy = empty_precoder_policy
        self._ack_wait, self._drn_up = ack_wait, drn_up
        self._equalise, self._phase3_lu, self._max_tx_rms = equalise, phase3_lu, max_tx_rms
        self._phase3_equalise = bool(phase3_equalise)
        if type(fractional_rx_timing) is not bool:
            raise TypeError("fractional_rx_timing must be bool")
        self._fractional_rx_timing = fractional_rx_timing
        if fractional_rx_reacquisition is None:
            fractional_rx_reacquisition = fractional_rx_timing
        if type(fractional_rx_reacquisition) is not bool:
            raise TypeError("fractional_rx_reacquisition must be bool")
        if fractional_rx_reacquisition and not fractional_rx_timing:
            raise ValueError("fractional RX reacquisition requires fractional RX timing")
        self._fractional_rx_reacquisition = fractional_rx_reacquisition
        self._async_design = bool(async_design)       # live media: design off the media thread; tests stay deterministic
        self.line_design: dict | None = None
        self._p3 = p3x.DigitalSide(
            law, uinfo=uinfo, rate_mask=rate_mask,
            equalise=self._phase3_equalise,
            fractional_rx_timing=self._fractional_rx_timing,
            fractional_rx_reacquisition=self._fractional_rx_reacquisition)
        self._p3_event_index = 0
        self._p4: p4c.Side | None = None
        self._tx_sample_count = 0
        self._backlog = np.empty(0)
        self._frames_fed = 0
        self._phase4_diagnostics_failed = False
        # Debug aid: V92_RX_DUMP=<path> saves the line audio the session received (np.save at hang-up/exit) for offline replay.
        self._tx_dump = [] if os.environ.get("V92_TXD_DUMP") else None
        if self._tx_dump is not None:
            import atexit
            atexit.register(lambda: open(os.environ["V92_TXD_DUMP"], "wb").write(b"".join(self._tx_dump)))
        self._dump = [] if os.environ.get("V92_RX_DUMP") else None
        if self._dump is not None:
            import atexit
            atexit.register(lambda: np.save(os.environ["V92_RX_DUMP"], np.concatenate(self._dump)) if self._dump else None)
        self._design_job: tuple | None = None            # (thread, result list) for the line design
        self._p4_fed_from: int | None = None
        self.stage = "phase3"
        self._training_profile: dict | None = None
        self._cpd_builder = None
        self._align_search = 0
        self._reneg: dict | None = None
        self.lapm: V90LapmLink | None = None
        self.events: list[tuple[str, object]] = []
        self.phase4_error: str | None = None
        self.phase4 = SimpleNamespace(state="not-started", events=self.events,
                                      transmitter=SimpleNamespace(DATA_READY=DATA_READY, cp_profile=None))
        self._queued_dte = bytearray()
        self._received_dte = bytearray()
        self.transmit_ready = True
        self.failure: str | None = None
        # The DCE's metrics read these from a V.90 session; a V.92 session has no such objects.
        self.upstream = None
        self.upstream_data = None

    # -- state ---------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self.stage

    @property
    def pending_bits(self) -> int:
        if self.lapm is not None:
            return self.lapm.pending_bits()
        return 8 * len(self._queued_dte)

    def _fail(self, reason: str) -> None:
        self.stage, self.failure = "failed", reason
        self.events.append(("v92-failed", reason))

    def phase4_diagnostics(self) -> dict | None:
        """Return a best-effort Phase-4 snapshot without affecting media processing."""
        if self._p4 is None:
            return None
        try:
            diagnostics = self._p4.diagnostics()
            decoder = diagnostics.get("decoder")
            if isinstance(decoder, dict) and self._p4_fed_from is not None:
                origin = int(self._p4_fed_from)
                decoder["receive_sample_origin"] = origin
                relative = decoder.get("trn2u_at")
                if relative is not None:
                    decoder["trn2u_absolute_sample"] = origin + int(relative)
            return diagnostics
        except Exception as exc:
            # This is an observability path. A broken snapshot must not interrupt
            # the receive stream, nor flood the event list once per media packet.
            if not self._phase4_diagnostics_failed:
                self._phase4_diagnostics_failed = True
                self.events.append(("diagnostics-error", {
                    "source": "phase4-diagnostics",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "stage": self.stage,
                    "frames_fed": self._frames_fed,
                }))
            return None

    # -- receive -------------------------------------------------------------------------------
    def feed_pcm(self, samples, *, sample_at: int | None = None) -> list:
        x = np.asarray(samples, dtype=float).reshape(-1)
        if self._dump is not None:
            self._dump.append(x.copy())
        if self.stage == "phase3":
            self._p3.feed(x)
            self._capture_phase3_events()
            self._start_line_design()
            if self._p3.state == "failed":
                self._fail("Phase 3 failed: " + str(self._p3.events[-1]))
            elif self._p3.trn2u_start is not None and self._p4 is not None and self._p4_fed_from is None:
                initial_sign = self._p3.e1u_last_sign
                if initial_sign is None:
                    self._fail("Phase 4 receive started without the E1u differential sign")
                    return []
                set_initial_sign = getattr(self._p4._decoder, "set_initial_sign", None)
                if set_initial_sign is None:
                    self._fail("Phase 4 upstream decoder cannot accept the E1u differential sign")
                    return []
                # The Phase-4 Side may have been constructed before E1u
                # arrived, but no Phase-4 sample is fed until this point.
                # Seed the stateful upstream decoder before replaying the
                # saved Phase-4 receive backlog.
                set_initial_sign(initial_sign)
                self._e1u_sign = int(initial_sign)        # a renegotiation's TRN2u starts from it too
                self._p4_fed_from = self._p3.trn2u_start
                self.events.append(("phase4-e1u-sign", {
                    "sign": int(initial_sign),
                    "e1u_last_sample": int(self._p4_fed_from) - 1,
                }))
                # Phase 3 decisions may use a receive equaliser. Phase 4 installs its own filter
                # and must be seeded from the raw line samples, at the corresponding raw index.
                self._backlog = np.array(self._p3.raw_from(self._p4_fed_from))
                self._drain_backlog(0)
                self._p3 = None                             # Phase 3 is over; stop keeping its audio
                self.stage = "phase4"
                self.events.append(("phase4-rx-start", int(self._p4_fed_from)))
        elif self._reneg is not None and self._reneg.get("rx_active"):
            self._feed_renegotiation_rx(x)
        elif (self.stage == "data" and self._reneg is None
              and (self._peer_reneg_tracker is not None or self._fpe_detector is not None)):
            self._feed_peer_rate_renegotiation_probe(x)
        elif self.stage in ("phase4", "data"):
            self._frames_fed += 1
            if self._frames_fed % 25 == 1 and self._p4 is not None:
                diagnostics = self.phase4_diagnostics()
                if diagnostics is not None:
                    self.events[:] = [e for e in self.events if e[0] != "phase4-diag"] + [("phase4-diag", diagnostics)]
            self._backlog = np.concatenate((self._backlog, x))
            self._drain_backlog(len(x))
            self._pump_data()
        return []

    def _feed_peer_rate_renegotiation_probe(self, x: np.ndarray) -> None:
        """Detect peer-initiated Ru or RM while preserving the old data stream."""
        tracker = self._peer_reneg_tracker
        old_side = self._p4
        if (tracker is None and self._fpe_detector is None) or old_side is None:
            return
        base = self._peer_reneg_probe_base
        self._peer_reneg_probe = np.concatenate((self._peer_reneg_probe, x))
        self._peer_reneg_total += len(x)
        if self._fpe_detector is not None and self._peer_fpe_cutover is None:
            fpe_events = self._fpe_detector.feed(self._fpe_detector_input(x, old_side))
            if fpe_events:
                fpe_positions = dict(fpe_events)
                rm_at = fpe_positions["rm"] - self._fpe_detector_shift()
                rm_prime = fpe_positions["rm-prime"] - self._fpe_detector_shift()
                rm_end = rm_prime + 2 * 12
                end_tail = 0
                if self._equalise:
                    z2 = np.asarray(self._fpe_detector.profile.prefilter_z2, dtype=float)
                    support = np.flatnonzero(np.abs(z2) > 1e-4)
                    flush_samples = int(support[-1]) if len(support) else 0
                    end_tail = ((flush_samples + 11) // 12) * 12
                self._peer_fpe_cutover = (rm_at, rm_prime, rm_end, end_tail)
        if self._peer_fpe_cutover is not None:
            rm_at, rm_prime, rm_end, end_tail = self._peer_fpe_cutover
            rm_offset, end_offset = rm_at - base, rm_end + end_tail - base
            if end_offset > len(self._peer_reneg_probe):
                # Preserve the mapper's full prefilter tail before switching
                # the receiver to the new FPE PCM control stream.
                return
            if rm_offset < 0:
                self._fail("V.92 FPE RM/RM-prime receive history expired")
                return
            if rm_offset:
                old_side.receive(self._peer_reneg_probe[:rm_offset])
                self._pump_data()
            if old_side.rx_filter is not None:
                old_side.rx_filter.apply(self._peer_reneg_probe[rm_offset:end_offset])
            if old_side.peer_cp is None:
                self._fail("V.92 FPE RM arrived without an active CPu data profile")
                return
            old_down = dm.downstream_profile_from_cpu(old_side.peer_cp)
            old_up = old_side.data_rx.rx.profile
            kwargs = {}
            if self._equalise:
                if old_side.rx_filter is None or self._cpd_builder is None:
                    self._fail("V.92 equalised FPE lacks its trained receive filter or CPd builder")
                    return
                kwargs = {"cpd_builder": self._cpd_builder,
                          "align_search": self._align_search,
                          "cpd_gain4": self._call_lu_gain4(old_side)}
            else:
                kwargs = self._fpe_options(old_side, self._peer_reneg_probe[rm_offset:end_offset])
                if kwargs is None:
                    return
            new_side = p4c.digital_side(
                old_down, self.law, digital_trn=0, ack_wait=self._ack_wait,
                drn=self._drn_up, keep_payload=False,
                lu=None,
                max_lost_frames=self._pcm_upstream_max_lost_frames,
                empty_precoder_policy=self._empty_precoder_policy,
                async_upstream_equalizer=self._async_design,
                fast_upstream_profile=old_up, **kwargs)
            if self._equalise:
                # FPE's upstream control is carried by the existing CPd PCM
                # mapper, which already puts the waveform in constellation
                # domain. Retain the trained model for later control stages.
                new_side.rx_filter = old_side.rx_filter
                new_side.rx_filter_delay = old_side.rx_filter_delay
                new_side.isi = old_side.isi
                new_side._decoder.receiver.gain = old_side.data_rx.gain
            new_side.bit_source = self._tx_bits
            import v92_fast_parameter_exchange as fpe
            signal = fpe.rf_signals(old_down, self.law)[0]
            new_side.receive(self._peer_reneg_probe[end_offset:])
            self._reneg = {
                "phase": "drain", "signal": signal, "signal_at": 0,
                "new_side": new_side, "old_side": old_side,
                "drn_up": self._drn_up, "rx_active": True,
                "rx_started": True, "round_started": False,
                "responder": True, "fpe": True,
                "rx_history": np.empty(0), "rx_total": 0,
            }
            self.stage = "renegotiating"
            self._peer_reneg_probe = np.empty(0, dtype=float)
            self._peer_reneg_probe_base = self._peer_reneg_total
            self._peer_reneg_candidate = None
            self._peer_fpe_cutover = None
            self._fpe_detector = None
            self._fpe_detector_filter = self._fpe_detector_front = None
            self.events.append(("v92-fpe-rm-detected", {"sample": rm_at}))
            self.events.append(("v92-fpe-rm-prime-detected", {"sample": rm_prime}))
            self.events.append(("v92-fpe-rf-response-start", None))
            return
        for name, at in (() if tracker is None else tracker.feed(x)):
            if name == "ru" and self._peer_reneg_candidate is None:
                self._peer_reneg_candidate = at
                continue
            if name != "ru-bar" or self._peer_reneg_candidate is None:
                continue
            candidate = self._peer_reneg_candidate
            trn_start = at + 24
            candidate_offset = candidate - base
            trn_offset = trn_start - base
            if candidate_offset < 0 or trn_offset < 0 or trn_offset > len(self._peer_reneg_probe):
                self._fail("V.92 peer Ru receive history expired before R-bar")
                return
            if self._training_profile is None:
                self._fail("V.92 peer rate renegotiation requires an available Phase 3 training profile")
                return
            kwargs = {}
            if self._equalise:
                if old_side.rx_filter is None or self._cpd_builder is None:
                    self._fail("V.92 equalised rate renegotiation lacks its trained receive filter or CPd builder")
                    return
                kwargs = {"cpd_builder": self._cpd_builder,
                          "align_search": self._align_search}
            if candidate_offset:
                old_side.receive(self._peer_reneg_probe[:candidate_offset])
                self._pump_data()
            if old_side.peer_cp is None:
                self._fail("V.92 peer rate renegotiation arrived without a negotiated CPu")
                return
            import v92_renegotiation as reneg
            new_side = p4c.digital_side(
                self._training_profile, self.law, ack_wait=self._ack_wait,
                drn=self._drn_up, keep_payload=False,
                lu=None,
                **self._rate_renegotiation_options(old_side),
                max_lost_frames=self._pcm_upstream_max_lost_frames,
                empty_precoder_policy=self._empty_precoder_policy,
                async_upstream_equalizer=self._async_design, **kwargs)
            self._seed_renegotiation_sign(new_side)
            if self._equalise:
                new_side.rx_filter = old_side.rx_filter
                new_side.rx_filter_delay = old_side.rx_filter_delay
                new_side.isi = old_side.isi
            new_side.bit_source = self._tx_bits
            new_side.receive(self._peer_reneg_probe[trn_offset:])
            self._reneg = {"phase": "drain", "signal": reneg.digital_r_pair(
                               dm.downstream_profile_from_cpu(old_side.peer_cp), self.law),
                           "signal_at": 0, "new_side": new_side, "old_side": old_side,
                           "drn_up": self._drn_up, "rx_active": True, "rx_started": True,
                           "round_started": False, "responder": True,
                           "silent_period": False, "silent_early_rt_after": None,
                           "rx_history": np.empty(0), "rx_total": 0}
            self._peer_reneg_probe = np.empty(0, dtype=float)
            self._peer_reneg_probe_base = self._peer_reneg_total
            self._peer_reneg_candidate = None
            self.events.append(("v92-rate-Ru-detected", candidate))
            self.events.append(("v92-rate-Rd-response-start", None))
            return

        safe_end = max(base, self._peer_reneg_total - 512)
        if self._peer_reneg_candidate is not None:
            if self._peer_reneg_total <= self._peer_reneg_candidate + 600:
                safe_end = min(safe_end, self._peer_reneg_candidate)
            else:
                self._peer_reneg_candidate = None
        release = min(len(self._peer_reneg_probe), max(0, safe_end - base))
        if release:
            old_side.receive(self._peer_reneg_probe[:release])
            self._peer_reneg_probe = self._peer_reneg_probe[release:]
            self._peer_reneg_probe_base += release
            self._pump_data()

    def _start_data_listeners(self, side: p4c.Side) -> None:
        """Rearm peer Ru/RM listeners at the new PCM-upstream data origin."""
        # A new data mode has a new receive origin. Never replay the passive
        # listener's retained old-mode samples into its freshly qualified B1.
        self._peer_reneg_probe = np.empty(0, dtype=float)
        self._peer_reneg_probe_base = self._peer_reneg_total = 0
        self._peer_reneg_candidate = None
        self._peer_fpe_cutover = None
        self._peer_reneg_tracker = (PatternTracker(families=("ru",))
                                    if self._enable_rate_renegotiation_response else None)
        if not self._enable_fpe:
            self._fpe_detector = None
            return
        # A peer-initiated RM arrives on its transmit clock, which is not
        # phase-locked to this receiver's 12-sample frame clock across RTP.
        # Search all offsets instead of trusting the local decoder phase.
        sample_phase = None
        rx = side.data_rx.rx
        self._fpe_detector = v92_fpe_signals.RmSequenceDetector(
            rx.profile, sample_phase=sample_phase)
        self._fpe_detector_filter = None
        self._fpe_detector_front = p4c.trn2u_front(side.data_rx, drop=0, follow=True)
        if self._equalise and side.rx_filter is not None:
            # Keep the passive RM detector in the same calibrated signal
            # domain as the explicit FPE initiator path. Use a separate filter
            # state: probing must not advance the production receiver twice.
            from v92_equaliser import StreamFilter
            self._fpe_detector_filter = StreamFilter(side.rx_filter.w)
            self._fpe_detector_filter._history = side.rx_filter._history.copy()
        self.events.append(("v92-fpe-listener-ready", {"sample_phase": sample_phase}))

    def _drain_backlog(self, fresh: int) -> None:
        """Phase 4 starts on audio that arrived while the line design ran (up to a second of it). Feed it a few frames at a
        time, ahead of the live audio, so one media callback does not take a quarter of a second."""
        take = min(len(self._backlog), max(fresh, 160) * 3 if len(self._backlog) > fresh else fresh)
        if take:
            self._p4.receive(self._backlog[:take])
            self._backlog = self._backlog[take:]

    def _pump_data(self) -> None:
        p4 = self._p4
        if self.stage == "phase4" and p4.connected and p4.data_rx.qualified:
            self.stage = "data"
            self.phase4.state = DATA_READY
            downstream_rate = (dm.downstream_profile_from_cpu(p4.peer_cp)["rate"]
                               if p4.peer_cp is not None else None)
            upstream_rate = (p4.data_rx.rx.profile.bits_per_frame * 8000 / 12
                             if hasattr(p4.data_rx, "rx") else None)
            if p4.peer_cp is not None:
                # The Phase 3 CPt is only the training profile. Report the live CPu
                # data rate through the compatibility surface consumed by the DCE.
                self.phase4.transmitter.cp_profile = dm.downstream_profile_from_cpu(p4.peer_cp)
            if getattr(p4.data_rx, "gain_calibration", None) is not None:
                self.events.append(("v92-b1-gain", {"gain": p4.data_rx.gain,
                                                    **p4.data_rx.gain_calibration}))
            self.events.append(("v92-data", {"downstream_rate": downstream_rate,
                                              "upstream_rate": upstream_rate}))
            if self._reneg is None:
                self._start_data_listeners(p4)
            if self._reneg is not None and self._reneg.get("round_started"):
                self._drn_up = self._reneg["drn_up"]
                if self.lapm is not None and downstream_rate is not None:
                    # LAPM's timer rate at this endpoint follows its transmitted direction
                    # (digital-to-analogue CPu); the changed CPd rate is the reverse stream.
                    self.lapm.set_v42_rate(int(round(downstream_rate)))
                if self._reneg.get("fpe"):
                    self.events.append(("v92-fpe-complete", {"downstream_rate": downstream_rate,
                                                               "upstream_rate": upstream_rate}))
                else:
                    self.events.append(("v92-rate-renegotiated", {"drn_up": self._drn_up,
                                                                   "downstream_rate": downstream_rate,
                                                                   "upstream_rate": upstream_rate}))
                self._reneg = None
                self._start_data_listeners(p4)
        if self.lapm is not None:
            bits = p4.data_rx.take_bits()
            if bits:
                self.lapm.feed_line_bits(bits)
                self._received_dte.extend(self.lapm.take_received())

    # -- transmit ------------------------------------------------------------------------------
    def read_g711(self, count: int, law: str | None = None, *, data_bit_source=None) -> bytes:
        if law is not None and law.lower() != self.law:
            raise ValueError("G.711 law changed after negotiation")
        out = self._read_g711(count)
        self._tx_sample_count += len(out)
        if self._tx_dump is not None:
            self._tx_dump.append(out)
        return out

    def _read_g711(self, count: int) -> bytes:
        if self._reneg is not None:
            return self._read_renegotiation(count)
        if self.stage == "failed":
            return bytes([self._silence()]) * count
        if self._p4 is None and self._p3 is not None:
            # Stop Phase 3's renderer at the exact end of R-bar-i, even when
            # it changes to phase4-ready while filling this block. The normal
            # fixed-size tx() API pads the remainder with silence, which made
            # the phase transition depend on whether readiness was observed
            # at the start or during a media callback.
            prefix = self._p3.tx(count, stop_at_phase4=True)
            self._capture_phase3_events()
            if self._p3.state != "phase4-ready":
                return prefix

            rbar_start = self._p3.rbar_i_start_sample
            if rbar_start is None:
                raise RuntimeError("Phase 4 handoff has no R-bar-i transmit origin")
            trn2d_start = self._tx_sample_count + len(prefix)
            rbar_relative_offset = trn2d_start - rbar_start
            if rbar_relative_offset != 24:
                raise RuntimeError(
                    "R-bar-i/Phase-4 handoff is not contiguous: "
                    f"rbar_start={rbar_start}, trn2d_start={trn2d_start}, "
                    f"offset={rbar_relative_offset}")
            frame_origin = self._p3.sd_start_sample
            if frame_origin is None:
                raise RuntimeError("Phase 4 handoff has no Sd frame origin")
            frame_phase = (trn2d_start - frame_origin) % 6
            self.events.append(("phase4-tx-handoff", {
                "frame_origin_name": "Sd",
                "frame_origin_sample": int(frame_origin),
                "rbar_i_start_sample": int(rbar_start),
                "rbar_i_end_sample": int(rbar_start + 24),
                "trn2d_start_sample": int(trn2d_start),
                "rbar_relative_offset": int(rbar_relative_offset),
                "frame_phase_from_sd": int(frame_phase),
                "frame_phase_verified": frame_phase == 0,
                "same_media_block_tail": int(count - len(prefix)),
            }))
            if self._equalise:
                self._start_line_design()
                if self._design_job is not None and self._design_job[0] is not None and self._design_job[0].is_alive():
                    # The existing asynchronous equaliser owns receive-side
                    # CPd setup. Do not ask Phase 3 to pad past its terminal
                    # state; preserve the established silence fallback until
                    # that receive model is available.
                    return prefix + bytes([self._silence()]) * (count - len(prefix))
            self._begin_phase4()
            if self.stage == "failed":
                return prefix + bytes([self._silence()]) * (count - len(prefix))
            if self._p4 is not None:
                return prefix + bytes(self._p4.transmit(count - len(prefix)))
        if self._p4 is None:
            out = self._p3.tx(count, stop_at_phase4=True)
            self._capture_phase3_events()
            return out
        return bytes(self._p4.transmit(count))

    def _capture_phase3_events(self) -> None:
        """Expose Ja/DIL milestones in the DCE's ordinary phase-3 metrics."""
        while self._p3_event_index < len(self._p3.events):
            name, value = self._p3.events[self._p3_event_index]
            self._p3_event_index += 1
            if name == "ja":
                ja = self._p3.ja or {}
                details = {key: ja.get(key) for key in ("N", "Ucodes", "bits")}
                details["start_sample"] = value
            elif name in ("dil-start", "dil-stop-request", "dil-stopped"):
                details = value
            elif name in ("su-phase-anchor", "su-phase", "trn1u-2"):
                details = value
            elif name in ("sd-tx-start", "jd-tx-start", "dil-tx-start",
                          "ri-tx-start", "r-bar-i-tx-start"):
                details = value
            elif name in ("cpt-preamble", "cpt", "ri-sent", "e1u-window",
                          "e1u", "e1u-qualified", "e1u-timeout",
                          "bad-cpt-repeat"):
                # Keep the raw CPt/E1u bit-window trace visible in the
                # ordinary DCE metrics. These diagnostics are needed to tell
                # a classifier miss from a Phase-4 handoff failure on a live
                # V.92 exchange.
                details = value
            else:
                continue
            self.events.append((f"phase3-{name}", details))

    def _silence(self) -> int:
        import v34_g711
        return int(v34_g711.encode(np.zeros(1), self.law)[0]) & 0xFF

    def _start_line_design(self) -> None:
        """The design costs about 0.7 s of CPU; start it as soon as the second TRN1u has ended so it does not stall the media
        thread when Phase 4 begins."""
        p3 = self._p3
        if not self._equalise or self._design_job is not None or p3 is None or p3.cpt_at is None:
            return
        n = p3.cpt_at - p3.trn1u2_start
        if n < 2000:
            return
        received_trn = np.array(p3.raw_from(p3.trn1u2_start)[:n])
        result: list = []
        cpt = dict(p3.cpt)

        def job():
            # A busy worker makes the media thread wait a full switch interval (5 ms) for the GIL on every call, which turned a
            # 15 ms callback into 150-200 ms. A short interval keeps the media thread on time.
            previous = sys.getswitchinterval()
            sys.setswitchinterval(0.0002)
            try:
                result.append(build())
            finally:
                sys.setswitchinterval(previous)

        def build():
            designed = design_for_line(received_trn, self.law, lu=self._phase3_lu, max_tx_rms=self._max_tx_rms,
                                       filter_limits=self._filter_limits)
            side = None
            if designed is not None:
                info = designed[3]
                # Building the Side searches CPd candidates (about 0.1 s): keep that off the media thread too.
                side = p4c.digital_side(dm.downstream_profile_from_cpu(cpt), self.law, ack_wait=self._ack_wait,
                                        drn=self._drn_up, keep_payload=False,
                                        lu=None, cpd_builder=designed[2],
                                        align_search=max(24, info["shift"] + 16),
                                        max_lost_frames=self._pcm_upstream_max_lost_frames,
                                        empty_precoder_policy=self._empty_precoder_policy,
                                        async_upstream_equalizer=self._async_design)
            return (designed, side)

        if not self._async_design:
            result.append(build())
            self._design_job = (None, result)
            return
        thread = threading.Thread(target=job, daemon=True)
        thread.start()
        self._design_job = (thread, result)

    def _begin_phase4(self) -> None:
        self.phase4.state = "starting"
        try:
            self._initialize_phase4()
        except Exception as exc:
            self.phase4.state = "failed"
            self.phase4_error = str(exc)
            self.events.append(("phase4-init-failed", self.phase4_error))
            raise
        self.phase4.state = "running"

    def _initialize_phase4(self) -> None:
        profile = dm.downstream_profile_from_cpu(self._p3.cpt)
        self._training_profile = dict(profile)
        self.phase4.transmitter.cp_profile = profile
        kwargs = {}
        filt = None
        if self._equalise:
            self._start_line_design()
            designed = prebuilt = None
            if self._design_job is not None:
                if self._design_job[0] is not None:
                    self._design_job[0].join()
                designed, prebuilt = self._design_job[1][0] if self._design_job[1] else (None, None)
            if designed is None:
                self._fail("no usable design for this line")
                return
            w, delay, cpd_builder, info = designed
            self.line_design = {"rx_delay": delay, "taps": len(w), "tx_gain": info["tx_gain"], "shift": info["shift"],
                                "rate": info["plan"]["drn"], "residual_rms": info["fit"]["residual_rms"],
                                "z2_scale": info["z2_scale"]}
            self.events.append(("line-designed", dict(self.line_design)))
            self._cpd_builder = cpd_builder
            self._align_search = max(24, info["shift"] + 16)
            kwargs = {"cpd_builder": cpd_builder, "align_search": self._align_search}
            from v92_equaliser import StreamFilter
            filt = (StreamFilter(w), delay)
        raw_empty_z2 = os.environ.get("EL_MODEM_V92_CPD_EMPTY_PRECODER_Z2")
        empty_z2 = (None if raw_empty_z2 is None else
                    [int(tap.strip(), 0) for tap in raw_empty_z2.split(",")])
        self._p4 = prebuilt if filt is not None else p4c.digital_side(
            profile, self.law, digital_trn=self._digital_trn_symbols,
            ack_wait=self._ack_wait, drn=self._drn_up, keep_payload=False,
            max_lost_frames=self._pcm_upstream_max_lost_frames,
            empty_precoder_policy=self._empty_precoder_policy,
            async_upstream_equalizer=self._async_design,
            b1u_rx_mode=self._b1u_rx_mode,
            cpd_constellation_mode=self._cpd_constellation_mode,
            cpd_gain4=_cpd_gain4(self._cpd_constellation_mode),
            empty_precoder_z2=empty_z2,
            fitted_fractional_delay_samples=self._fitted_fractional_delay(),
            lu=None, **kwargs)
        if filt is not None:
            self._p4.rx_filter, self._p4.rx_filter_delay = filt
            h = np.asarray(info["fit"]["h"], dtype=float)
            self._p4.isi = (h, int(np.argmax(np.abs(h))))
        self._p4.bit_source = self._tx_bits
        self.events.append(("phase4-start", {
            "cpt_rate": profile["rate"],
            "digital_trn_symbols": self._digital_trn_symbols,
        }))

    def _flush_peer_probe(self) -> None:
        """Deliver held old-mode data before initiating a local control exchange."""
        if len(self._peer_reneg_probe):
            self._p4.receive(self._peer_reneg_probe)
            self._pump_data()
            self._peer_reneg_probe = np.empty(0, dtype=float)
            self._peer_reneg_probe_base = self._peer_reneg_total
            self._peer_reneg_candidate = None

    def request_rate_renegotiation(self, drn_up: int, *, silent_period: bool = False,
                                   silent_early_rt_after: int | None = None) -> None:
        """Start opt-in V.92 §9.8 digital-initiated rate renegotiation.

        Reuse the Phase 3 training profile and, on an equalised line, its trained
        filter and quantised precoder. The requested rate must fit the trained
        constellation's modulus capacity. ``silent_early_rt_after`` selects the
        Figure 17 early-R-t timing probe.
        """
        if self.stage != "data" or self._p4 is None or self.lapm is None:
            raise RuntimeError("rate renegotiation requires an established V.92 LAPM data session")
        if isinstance(drn_up, bool) or not isinstance(drn_up, int) or not 1 <= drn_up <= 19:
            raise ValueError("drn_up must be an integer in 1..19; zero is the CPd cleardown value")
        if type(silent_period) is not bool:
            raise TypeError("silent_period must be bool")
        if (silent_early_rt_after is not None
                and (isinstance(silent_early_rt_after, bool)
                     or not isinstance(silent_early_rt_after, int)
                     or not 2400 <= silent_early_rt_after < 8004)):
            raise ValueError("silent_early_rt_after must be an integer in 2400..8003 or None")
        if silent_early_rt_after is not None and not silent_period:
            raise ValueError("silent_early_rt_after requires silent_period=True")
        if self._training_profile is None or self._p4.peer_cp is None:
            raise RuntimeError("rate renegotiation lacks the original Phase 3/CPu profile")
        kwargs = {}
        if self._equalise:
            if self._cpd_builder is None or self._p4.rx_filter is None:
                raise RuntimeError("equalised rate renegotiation lacks its trained filter or CPd")
            if 1 << (2 * (drn_up + 17)) > self._p4.data_rx.profile.total_modulus:
                raise ValueError("requested data rate exceeds the trained constellation capacity")
            data_profile = replace(self._p4.data_rx.profile, bits=2 * (drn_up + 17))
            kwargs = {"cpd_builder": partial(self._cpd_builder, drn=drn_up),
                      "align_search": self._align_search,
                      "upstream_data_profile": data_profile}
        import v92_renegotiation as reneg
        signal = reneg.digital_r_pair(dm.downstream_profile_from_cpu(self._p4.peer_cp), self.law)
        next_side = p4c.digital_side(self._training_profile, self.law, ack_wait=self._ack_wait,
                                     drn=drn_up, keep_payload=False,
                                     lu=None,
                                     **self._rate_renegotiation_options(self._p4, drn_up),
                                     max_lost_frames=self._pcm_upstream_max_lost_frames,
                                     empty_precoder_policy=self._empty_precoder_policy,
                                     async_upstream_equalizer=self._async_design,
                                     silent_early_rt_after=silent_early_rt_after, **kwargs)
        self._seed_renegotiation_sign(next_side)
        # Construction validates the requested K against the trained moduli
        # before consuming held data or replacing any active receive state.
        self._flush_peer_probe()
        if self._equalise:
            next_side.rx_filter = self._p4.rx_filter
            next_side.rx_filter_delay = self._p4.rx_filter_delay
            next_side.isi = self._p4.isi
            self._cpd_builder = kwargs["cpd_builder"]
        if silent_period:
            next_side.set_silent_period_request(True)
        next_side.bit_source = self._tx_bits
        self._reneg = {"phase": "drain", "signal": signal, "signal_at": 0,
                       "new_side": next_side, "drn_up": drn_up, "rx_active": False,
                       "silent_period": silent_period,
                       "silent_early_rt_after": silent_early_rt_after,
                       "rx_tracker": PatternTracker(families=("ru",)),
                       "rx_history": np.empty(0), "rx_started": False, "round_started": False}
        self.events.append(("v92-rate-renegotiation-start", {
            "drn_up": drn_up, "silent_period": silent_period,
            "silent_early_rt_after": silent_early_rt_after}))

    def _fitted_fractional_delay(self) -> float | None:
        return next((details.get("delay_samples") for event, details in reversed(self.events)
                     if event == "phase3-su-phase" and isinstance(details, dict)), None)

    # TRN2d before SUVd in a rate renegotiation with the TRN2u-fit receiver (9.8.1.1.2 allows up to 16008T). The
    # analogue modem may end TRN2u when SUVd arrives (9.8.2.1.2), and it starts after Ru/Ru-bar and a round trip;
    # the fit needs 2016 symbols of it.
    RENEGOTIATION_TRN2D_SYMBOLS = 8004

    def _seed_renegotiation_sign(self, side) -> None:
        """A renegotiation's TRN2u starts its differential sign from E1u's last sign, like Phase 4's (8.7.6)."""
        set_initial_sign = getattr(side._decoder, "set_initial_sign", None)
        sign = getattr(self, "_e1u_sign", None)
        if set_initial_sign is not None and sign is not None:
            set_initial_sign(sign)

    def _rate_renegotiation_options(self, old_side, max_drn: int | None = None) -> dict:
        """digital_side options of a rate-renegotiation side. With the TRN2u-fit receiver the new side sends TRN2d,
        fits the TRN2u that it holds open and designs its CPd again (DRN at most ``max_drn``, a digital-initiated
        request); the unit-gain receiver goes straight to SUVd and keeps the call's G = 1/LU."""
        if self._b1u_rx_mode == "trn2u-fit-viterbi" and not self._equalise:
            return {"b1u_rx_mode": self._b1u_rx_mode, "acquire_trn2u": True,
                    "digital_trn": self.RENEGOTIATION_TRN2D_SYMBOLS,
                    "cpd_constellation_mode": self._cpd_constellation_mode,
                    "cpd_gain4": _cpd_gain4(self._cpd_constellation_mode),
                    "adaptive_max_drn": max_drn if self._cpd_constellation_mode == "adaptive" else None,
                    "fitted_fractional_delay_samples": self._fitted_fractional_delay()}
        return {"digital_trn": 0, "cpd_gain4": self._call_lu_gain4(old_side)}

    def _fpe_detector_input(self, x, old_side) -> np.ndarray:
        """Old-mode samples in the domain the RM detector expects: the CPd points."""
        if self._fpe_detector_front is not None:
            return self._fpe_detector_front.points(x)
        detector_samples = self._fpe_detector_filter.apply(x) if self._fpe_detector_filter is not None else x
        return detector_samples / old_side.data_rx.gain if self._equalise else detector_samples

    def _fpe_detector_shift(self) -> int:
        """Raw sample = detector sample - this (the TRN2u FIR's delay)."""
        return self._fpe_detector_front.delay if self._fpe_detector_front is not None else 0

    def _fpe_options(self, old_side, after_rm) -> dict | None:
        """``digital_side`` options for the new mode of an FPE that is not equalised. ``after_rm``: the raw samples from
        RM to the new mode's first, which the old receiver never sees. A TRN2u-fit data mode keeps that receiver: the
        new side decodes the old mode's control through the old mode's FIR and starts the new mode from that fit (FPE
        sends no TRN2u); its CPd keeps this call's G, so the line stays in the old fit's domain."""
        if self._b1u_rx_mode == "off":
            return {"cpd_gain4": self._call_lu_gain4(old_side)}
        front = self._fpe_front(old_side, after_rm)
        if front is None:
            self._fail("V.92 FPE needs the TRN2u-fit receiver's filter")
            return None
        gain4 = _cpd_gain4(self._cpd_constellation_mode)
        return {"b1u_rx_mode": self._b1u_rx_mode, "cpd_constellation_mode": self._cpd_constellation_mode,
                "cpd_gain4": self._call_lu_gain4(old_side) if gain4 is None else gain4,
                "fitted_fractional_delay_samples": self._fitted_fractional_delay(),
                "fast_upstream_front": front}

    def _fpe_front(self, old_side, after_rm):
        """The old data mode's TRN2u FIR as it stands, continued over ``after_rm`` (None if it has none)."""
        live = getattr(getattr(old_side.data_rx, "new_receiver", old_side.data_rx), "_trn2u_rx_filter", None)
        if live is None:
            return None
        return p4c.trn2u_front(old_side.data_rx, np.concatenate((live._history, after_rm)))

    def _call_lu_gain4(self, side) -> int | None:
        """The CPd 4G of G = 1/LU that this call's first CPd fixed; renegotiated CPds keep it."""
        if getattr(self, "_lu_gain4", None) is None:
            self._lu_gain4 = getattr(side, "lu_gain4", None)
        return self._lu_gain4

    def request_fast_parameter_exchange(self) -> None:
        """Initiate opt-in V.92 FPE by sending Rf/Rf-bar in the current data mode."""
        if not self._enable_fpe:
            raise RuntimeError("enable_fast_parameter_exchange must be set when constructing the session")
        if self.stage != "data" or self._p4 is None or self.lapm is None or self._reneg is not None:
            raise RuntimeError("fast parameter exchange requires an established idle LAPM data session")
        if self._training_profile is None or self._p4.peer_cp is None:
            raise RuntimeError("fast parameter exchange requires negotiated CPu and CPd profiles")
        if self._equalise and (self._p4.rx_filter is None or self._cpd_builder is None):
            raise RuntimeError("equalised fast parameter exchange requires the trained receive filter and CPd builder")
        self._flush_peer_probe()
        import v92_fast_parameter_exchange as fpe
        old_side = self._p4
        old_down = dm.downstream_profile_from_cpu(old_side.peer_cp)
        old_up = old_side.data_rx.rx.profile
        # A TRN2u-fit side's FIR is reseeded from the live one when RM arrives.
        kwargs = ({"cpd_builder": self._cpd_builder, "align_search": self._align_search,
                   "cpd_gain4": self._call_lu_gain4(old_side)}
                  if self._equalise else self._fpe_options(old_side, np.empty(0)))
        if kwargs is None:
            raise RuntimeError("fast parameter exchange needs the TRN2u-fit receiver's filter")
        new_side = p4c.digital_side(
            old_down, self.law, digital_trn=0, ack_wait=self._ack_wait,
            drn=self._drn_up, keep_payload=False,
            lu=None,
            max_lost_frames=self._pcm_upstream_max_lost_frames,
            empty_precoder_policy=self._empty_precoder_policy,
            async_upstream_equalizer=self._async_design,
            fast_upstream_profile=old_up, **kwargs)
        if self._equalise:
            new_side.rx_filter = old_side.rx_filter
            new_side.rx_filter_delay = old_side.rx_filter_delay
            new_side.isi = old_side.isi
            new_side._decoder.receiver.gain = old_side.data_rx.gain
        new_side.bit_source = self._tx_bits
        # The caller's response is delayed by the media round trip. Its frame
        # phase need not equal the local receive decoder's phase, so this rare
        # exchange searches all 12 offsets while the passive listener keeps
        # the frame-phase fast path.
        self._fpe_detector = v92_fpe_signals.RmSequenceDetector(
            old_up, sample_phase=None)
        self._fpe_detector_filter = None
        self._fpe_detector_front = p4c.trn2u_front(old_side.data_rx, drop=0, follow=True)
        if self._equalise:
            from v92_equaliser import StreamFilter
            self._fpe_detector_filter = StreamFilter(old_side.rx_filter.w)
            self._fpe_detector_filter._history = old_side.rx_filter._history.copy()
        self._reneg = {
            "phase": "drain", "signal": fpe.rf_signals(old_down, self.law)[0],
            "signal_at": 0, "old_side": old_side, "new_side": new_side,
            "drn_up": self._drn_up, "rx_active": True, "rx_started": False,
            "round_started": False, "initiator": True, "fpe": True,
            "rx_history": np.empty(0), "rx_total": 0,
        }
        self._peer_reneg_probe = np.empty(0, dtype=float)
        self._peer_reneg_probe_base = 0
        self._peer_reneg_total = 0
        self.stage = "renegotiating"
        self.events.append(("v92-fpe-rf-start", None))

    def _read_renegotiation(self, count: int) -> bytes:
        reneg = self._reneg
        out = bytearray()
        remaining = count
        if reneg["phase"] == "drain":
            old_side = reneg.get("old_side", self._p4)
            frame_units = 12 if reneg.get("fpe") else 6
            prefix = old_side.transmit_to_frame_boundary(remaining, frame_units)
            out.extend(prefix)
            remaining -= len(prefix)
            if old_side.units_sent % frame_units == 0:
                if old_side._units_out:
                    self._fail("old Phase 4 mapper crossed the rate-renegotiation frame boundary")
                    return bytes(out) + bytes([self._silence()]) * (count - len(out))
                reneg["phase"] = "R"
                # The initiator now listens for Ru; a responder's new side has been receiving the peer's TRN2u
                # since Ru-bar and must keep it whole for the TRN2u fit.
                reneg["rx_active"] = True
                self.events.append(("v92-rate-Rd-start", old_side.units_sent))
            else:
                return bytes(out)
        if reneg["phase"] == "R" and remaining:
            signal = reneg["signal"]
            take = min(remaining, len(signal) - reneg["signal_at"])
            start = reneg["signal_at"]
            out.extend(signal[start:start + take])
            reneg["signal_at"] += take
            remaining -= take
            if reneg["signal_at"] == len(signal):
                self._p4 = reneg["new_side"]
                self.stage = "phase4"
                self.phase4.state = "running"
                reneg["phase"] = "TRN2"
                reneg["round_started"] = True
                reneg["rx_active"] = True
                self.events.append(("v92-rate-trn2d-start", None))
        if remaining and reneg["phase"] == "TRN2":
            out.extend(self._p4.transmit(remaining))
        return bytes(out)

    def _feed_fpe_rm_response(self, samples: np.ndarray) -> None:
        """Keep old-mode PCM samples until an initiated FPE's RM/RM′ response is found."""
        reneg = self._reneg
        detector = self._fpe_detector
        if reneg is None or detector is None:
            return
        x = np.asarray(samples, dtype=float).reshape(-1)
        base = self._peer_reneg_probe_base
        self._peer_reneg_probe = np.concatenate((self._peer_reneg_probe, x))
        self._peer_reneg_total += len(x)
        events = detector.feed(self._fpe_detector_input(x, reneg["old_side"]))
        if events:
            positions = dict(events)
            shift = self._fpe_detector_shift()
            start, rm_prime = positions["rm"] - shift, positions["rm-prime"] - shift
            end = rm_prime + 2 * 12
            end_tail = 0
            if self._equalise:
                profile = detector.profile
                z2 = np.asarray(profile.prefilter_z2, dtype=float)
                support = np.flatnonzero(np.abs(z2) > 1e-4)
                flush_samples = int(support[-1]) if len(support) else 0
                flush_frames = (flush_samples + 11) // 12
                # Wait through the old mapper's frame-rounded flush and the
                # fresh mapper's leading FIR delay.  RM's detected start is
                # already delayed by z2's first tap; the fresh mapper's first
                # guard output is delayed by that same amount, so the delays
                # cancel in this relative interval.  Subtracting first_delay
                # here cuts over before the new transmitter's first valid
                # guard frame and leaves its trellis decoder out of state.
                end_tail = flush_frames * 12
            reneg["fpe_cutover"] = (start, rm_prime, end, end_tail)
        cutover = reneg.get("fpe_cutover")
        if cutover is not None:
            start, rm_prime, end, end_tail = cutover
            start_offset, end_offset = start - base, end - base + end_tail
            # Do not switch receive modes until the CPd transmit prefilter's
            # post-RM tail has actually arrived. The detector sees RM′ first;
            # the sender still has to flush its causal filter before TRN2u.
            if end_offset > len(self._peer_reneg_probe):
                return
            if start_offset < 0 or end_offset > len(self._peer_reneg_probe):
                self._fail("V.92 FPE RM/RM-prime receive history expired")
                return
            old_side = reneg["old_side"]
            if start_offset:
                old_side.receive(self._peer_reneg_probe[:start_offset])
                self._pump_data()
            if old_side.rx_filter is not None:
                old_side.rx_filter.apply(self._peer_reneg_probe[start_offset:end_offset])
            if getattr(reneg["new_side"]._decoder, "front", None) is not None:
                # The old receiver has now had everything before RM: continue its FIR as it stands.
                reneg["new_side"]._decoder.reseed(
                    self._fpe_front(old_side, self._peer_reneg_probe[start_offset:end_offset]))
            reneg["new_side"].receive(self._peer_reneg_probe[end_offset:])
            reneg["rx_started"] = True
            self._peer_reneg_probe = np.empty(0, dtype=float)
            self._peer_reneg_probe_base = self._peer_reneg_total
            self._fpe_detector = None
            self._fpe_detector_filter = self._fpe_detector_front = None
            reneg.pop("fpe_cutover", None)
            reneg["rx_history"] = np.empty(0)
            reneg["rx_total"] = 0
            self.events.append(("v92-fpe-rm-detected", {"sample": start}))
            self.events.append(("v92-fpe-rm-prime-detected", {"sample": rm_prime}))
            self._pump_data()
            return
        safe_end = max(base, self._peer_reneg_total - 512)
        release = min(len(self._peer_reneg_probe), max(0, safe_end - base))
        if release:
            reneg["old_side"].receive(self._peer_reneg_probe[:release])
            self._peer_reneg_probe = self._peer_reneg_probe[release:]
            self._peer_reneg_probe_base += release
            self._pump_data()

    def _feed_renegotiation_rx(self, samples) -> None:
        reneg = self._reneg
        x = np.asarray(samples, dtype=float).reshape(-1)
        history = reneg["rx_history"]
        base = reneg.get("rx_total", 0) - len(history)
        history = np.concatenate((history, x))
        reneg["rx_total"] = reneg.get("rx_total", 0) + len(x)
        if reneg.get("fpe"):
            if reneg.get("initiator") and not reneg.get("rx_started"):
                self._feed_fpe_rm_response(x)
                return
            reneg["new_side"].receive(x)
            reneg["rx_history"] = history[-4096:]
            reneg["diag_feeds"] = reneg.get("diag_feeds", 0) + 1
            if reneg["diag_feeds"] % 25 == 1:
                # The new side's control exchange, kept current like phase4-diag (a retrain snapshot keeps it).
                try:
                    diagnostics = reneg["new_side"].diagnostics()
                except Exception as exc:                      # observability only
                    diagnostics = {"error": f"{type(exc).__name__}: {exc}"}
                self.events[:] = [e for e in self.events if e[0] != "v92-fpe-diag"] + [("v92-fpe-diag", diagnostics)]
            self._pump_data()
            return
        if not reneg["rx_started"]:
            for name, at in reneg["rx_tracker"].feed(x):
                if name.endswith("bar"):
                    start = at + 24
                    offset = start - base
                    if offset < 0 or reneg["new_side"] is None:
                        self._fail("V.92 Ru-bar receive history was unavailable")
                        return
                    self._p4 = reneg["new_side"]
                    self._p4.receive(history[offset:])
                    reneg["rx_started"] = True
                    self.events.append(("v92-rate-trn2u-detected", start))
                    break
        elif reneg.get("responder"):
            reneg["new_side"].receive(x)
        elif self.stage == "phase4":
            self._p4.receive(x)
        reneg["rx_history"] = history[-4096:]
        self._pump_data()

    def _tx_bits(self, n: int) -> list[int]:
        """Downstream data bits: LAPM's once the link exists, idle ones before."""
        if self.lapm is None and self._p4 is not None and self._p4.sent_e and self.enable_lapm:
            cpu = self._p4.peer_cp
            rate = int(round(float(dm.downstream_profile_from_cpu(cpu)["rate"]))) if cpu else 0
            self.lapm = V90LapmLink(
                rate=rate, calling_party=False,
                detect=self._lapm_detect,
                compression=self._v42bis_compression,
                compression_recovery=self._v42_compression_recovery)
            if self._queued_dte:
                self.lapm.feed_bytes(bytes(self._queued_dte))
                self._queued_dte.clear()
            self.events.append(("lapm-started", {"downstream_rate": rate}))
        if self.lapm is not None:
            return self.lapm.read_tx_bits(n)
        return [1] * n

    # -- DTE -----------------------------------------------------------------------------------
    def feed_bytes(self, data: bytes) -> None:
        payload = bytes(data)
        if self.lapm is not None:
            self.lapm.feed_bytes(payload)
        else:
            self._queued_dte.extend(payload)

    def read_received_data(self) -> bytes:
        data = bytes(self._received_dte)
        self._received_dte.clear()
        return data


def factory(*, law: str, uinfo: int | None = None, rate_mask: int = 0x3FFFFF, enable_lapm: bool = True,
            equalise: bool = False, async_design: bool = False, phase3_equalise: bool = False,
            fractional_rx_timing: bool = False,
            fractional_rx_reacquisition: bool | None = None,
            enable_rate_renegotiation_response: bool = False,
            enable_fast_parameter_exchange: bool = False,
            enable_fast_parameter_exchange_response: bool = False,
            lapm_detect: bool = True,
            pcm_upstream_max_lost_frames: int = 3,
            digital_trn_symbols: int | None = None, b1u_rx_mode: str | None = None,
            cpd_constellation_mode: str | None = None,
            v42bis_compression: tuple[int, int, int] | None = None,
            v42_compression_recovery: bool = False):
    """A ``phase3_factory`` for the engine: the engine passes the decoded INFO1a, the law and clocks."""
    if digital_trn_symbols is None:
        # Lab-only discriminator for how long SmartLink needs to observe
        # downstream TRN2. Normal operation retains the 2040T minimum.
        digital_trn_symbols = int(os.environ.get("EL_MODEM_V92_DIGITAL_TRN_SYMBOLS", "2040"))
    if b1u_rx_mode is None:
        # "trn2u-fit-viterbi" with cpd "adaptive" is the pair validated against the SmartLink peer
        # (lab/V92_SMARTLINK_DATA_2026-11-17.md). "off"/"default" is the unit-gain pair of the paired unit tests; the
        # profile keys v92_b1u_rx_mode / v92_cpd_constellation or these variables select it.
        b1u_rx_mode = os.environ.get("EL_MODEM_V92_B1U_RX", "trn2u-fit-viterbi")
    if cpd_constellation_mode is None:
        cpd_constellation_mode = ("default" if b1u_rx_mode == "off"
                                  else os.environ.get("EL_MODEM_V92_CPD_CONSTELLATION", "adaptive"))

    def make(*, info1a: dict, law: str, handoff_at: int, media_origin: float,
             media_clock=None, lapm_detect: bool = lapm_detect):
        return V92DigitalSession(law=law, uinfo=int(info1a.get("v90_uinfo", uinfo or 67)),
                                 rate_mask=rate_mask, enable_lapm=enable_lapm, equalise=equalise,
                                 async_design=async_design, phase3_equalise=phase3_equalise,
                                 fractional_rx_timing=fractional_rx_timing,
                                 fractional_rx_reacquisition=fractional_rx_reacquisition,
                                 lapm_detect=lapm_detect,
                                 enable_rate_renegotiation_response=enable_rate_renegotiation_response,
                                 enable_fast_parameter_exchange=enable_fast_parameter_exchange,
                                 enable_fast_parameter_exchange_response=enable_fast_parameter_exchange_response,
                                 pcm_upstream_max_lost_frames=pcm_upstream_max_lost_frames,
                                 digital_trn_symbols=digital_trn_symbols,
                                 b1u_rx_mode=b1u_rx_mode,
                                 cpd_constellation_mode=cpd_constellation_mode,
                                 v42bis_compression=v42bis_compression,
                                 v42_compression_recovery=v42_compression_recovery,
                                 empty_precoder_policy=(os.environ.get(
                                     "EL_MODEM_V92_CPD_EMPTY_PRECODER_POLICY", "omit")),
                                 filter_limits=filter_limits_from_info1a(info1a))
    return make


# -- design on a dispersive line ------------------------------------------------------------------------------
def filter_limits_from_info1a(info1a: dict) -> tuple[int, int]:
    """(Lmax, Ltot) from the analogue modem's INFO1a (V.92 Table 18 bits 16:17 and 14:15), carried in the V.90
    parser's ``v90_reserved_12_17``. Absent or not a PCM-upstream INFO1a: the minimum every modem supports."""
    import v92_pair_session as pair
    raw = info1a.get("v90_reserved_12_17") if info1a else None
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < 64:
        return pair.MIN_SECTION_COEFFS, pair.MIN_TOTAL_COEFFS
    return 128 + 64 * ((raw >> 4) & 3), 192 + 64 * ((raw >> 2) & 3)


def design_for_line(received, law: str, *, lu: float, max_tx_rms: float = 6000.0, trellis: int = 1,
                    safety: float = 1.6, filter_limits: tuple[int, int] | None = None):
    """From the second TRN1u as it was received (``received``, starting at the TRN1u's first symbol, TRN1u of
    amplitude ``lu``): the receive equaliser, and a CPd builder carrying the precoder, prefilter, constellation and
    moduli that suit the line. Returns ``(w, delay, cpd_builder, plan)`` or None when the line is hopeless."""
    import v92_channel_estimate as estimate
    import v92_cp
    import v92_equaliser as equaliser
    import v92_pair_session as pair
    bits = p3x._trn1u_bits(len(received))
    reference = lu * np.where(np.array(bits) == 1, -1.0, 1.0)
    w, delay, _ = equaliser.train(received, reference)
    # h with its main cursor some samples in: take the reference a few symbols earlier so the causal fit can place it.
    ahead = 8
    sent = np.concatenate((reference[ahead:], np.zeros(ahead)))
    fit = estimate.estimate(received, sent, pair.ESTIMATE_TAPS, skip=64)
    prefilter_taps, postcursors = pair.filter_tap_budget(*(filter_limits or ()))
    designed = pair.design_from_fit(fit, law=law, lu=lu, trellis=trellis, max_tx_rms=max_tx_rms,
                                    training_symbols=len(received), prefilter_taps=prefilter_taps,
                                    postcursors=postcursors)
    if designed is None:
        return None
    taps, tx_gain, design_plan = designed
    plan = design_plan(safety)
    if plan is None:
        return None
    gap = 9.0
    q = lambda values, scale: [int(max(-32768, min(32767, round(v * scale)))) for v in values]
    # The prefilter puts the channel's main cursor at 1, which can need a tap above 1; Q0.15 stops just short of it
    # (a channel with a 0.25 precursor gave 1.067, clipped to 6 % ISI). Scale the prefilter into range instead: the
    # line output is then ``z2_scale`` times the constellation, which the data receiver's B1 gain calibration removes.
    z2_scale = min(1.0, (32767.0 / 32768.0) / max(abs(v) for v in taps["z2"]))
    precoder = {"z1": [], "p1": q(taps["p1"], 16384.0), "z2": q(taps["z2"], 32768.0 * z2_scale), "p2": []}
    # The CPd carries the filter taps as 16-bit integers (Q1.14 and Q0.15), which leaves a little ISI the float design
    # does not have. Measure it by running the quantised taps through the estimated channel and size the rate for it.
    isi_q, shift = _quantisation_isi(fit["h"], precoder, plan, law, scale=z2_scale)
    # The estimate itself is biased: TRN1u is measured at one level through a companding codec and the data run at many
    # (about 0.04% rms on a 12-tap test line, but about 1% on a 3-tap line, whose quantisation error does not average out).
    # There is no retry ladder here, so the first design has to carry a margin for it.
    taps["isi_ratio"] = float(np.hypot(np.hypot(taps["isi_ratio"], isi_q), ESTIMATE_BIAS_ISI))
    plan = None
    for gap in (9.0, 6.0, 4.0):                   # relax the constellation spacing until some rate fits
        designed = pair.design_from_fit(fit, law=law, lu=lu, trellis=trellis, max_tx_rms=max_tx_rms,
                                        gap_sigmas=gap, training_symbols=len(received),
                                        prefilter_taps=prefilter_taps, postcursors=postcursors)
        designed[0]["isi_ratio"] = taps["isi_ratio"]
        plan = designed[2](safety)
        if plan is not None:
            break
    if plan is None:
        return None

    def cpd_builder(ack: bool, *, drn: int | None = None):
        selected = plan["drn"] if drn is None else drn
        if (isinstance(selected, bool) or not isinstance(selected, int)
                or not 1 <= selected <= 19):
            raise ValueError("CPd data rate must be an integer DRN in 1..19")
        if 1 << (2 * (selected + 17)) > prod(plan["moduli"]):
            raise ValueError("requested data rate exceeds the trained constellation capacity")
        return v92_cp.build_cpd(drn=selected, trellis=plan["trellis"], gain4=0x4000, ack=ack,
                                moduli=plan["moduli"], interval_sets=(0,) * 6,
                                constellations=[[int(v) for v in plan["constellation"]]], precoder=precoder)
    return w, delay, cpd_builder, {"plan": plan, "taps": taps, "tx_gain": tx_gain, "fit": fit,
                                   "shift": shift, "isi_quantisation": isi_q, "z2_scale": z2_scale}


def _quantisation_isi(h, precoder: dict, plan: dict, law: str, frames: int = 40,
                      scale: float = 1.0) -> tuple[float, int]:
    """RMS of the line-output error against the constellation targets (times ``scale``, the prefilter's scaling),
    relative to their RMS, when the analogue modem uses the integer taps of ``precoder`` over the channel ``h``, and the
    shift (samples) at which it is smallest: the line output lags its target by the channel cursor plus the precoder's
    own delay, which the data receiver must allow for."""
    import random

    import v92_pcm_upstream as pu
    profile = pu.PcmUpstreamProfile(
        constellations=(tuple(float(v) for v in plan["constellation"]),), interval_set=(0,) * 12,
        moduli=tuple(plan["moduli"]), trellis=plan["trellis"], bits=2 * (plan["drn"] + 17),
        precoder_p1=tuple(v / 16384.0 for v in precoder["p1"]), prefilter_z2=tuple(v / 32768.0 for v in precoder["z2"]))
    tx = pu.PcmUpstreamTransmitter(profile, record=True)
    rng = random.Random(5)
    k = profile.bits_per_frame
    skip = 300                                    # past the start-up transient of the long prefilter
    stop = frames * pu.INTERVALS
    if stop <= skip:
        raise ValueError("need more than 300 samples after constructing the quantisation probe")
    # A 256-tap prefilter and the fitted channel can place the main cursor
    # beyond 47 samples. Cover their complete feedforward support, and extend
    # the generated signal so every delay is scored on the same target window.
    max_shift = max(0, len(h) + len(precoder["z2"]) - 2)
    extra_frames = -(-max_shift // pu.INTERVALS)
    wire = []
    for _ in range(frames + extra_frames):
        wire.extend(tx.encode_frame([rng.randint(0, 1) for _ in range(k)]))
    line = np.convolve(np.array(wire), np.asarray(h, dtype=float))[:len(wire)]
    target = scale * np.array(tx.levels[skip:stop])
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    best = None
    for shift in range(max_shift + 1):
        err = line[shift + skip:shift + stop] - target
        ratio = float(np.sqrt(np.mean(err ** 2)) / target_rms)
        if best is None or ratio < best[0]:
            best = (ratio, shift)
    return best
