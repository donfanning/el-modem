"""Production-DCE adapter for opt-in V.90 analogue-caller startup and data.

The explicit caller profile drives Phase 2/3/4 through the production media
surface. It reports CONNECT only after B1d and LAPM qualify; a paired full-DCE
path is still under qualification.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import replace
import threading

import numpy as np

from g711 import (alaw_to_linear, linear_to_alaw, linear_to_ulaw,
                  ulaw_to_linear)
from v8 import V8_PCM_ANALOG, V8_PROTOCOL_LAPM_V42
from v34_call import probing_signal
from v90_analogue_caller_phase2 import V90AnalogueCallerPhase2
from v90_analogue_phase2_rx import V90AnaloguePhase2Adapter
from v90_analogue_phase2_tx import V90AnaloguePhase2Transmitter
from v90_analogue_caller_phase3 import V90AnalogueCallerPhase3
from v90_analogue_downstream_rx import V90AnalogueDownstreamPhase3Receiver
from v90_analogue_phase3_session import V90AnaloguePhase3Session
from v90_analogue_phase3_tx import V90AnaloguePhase3Transmitter
from v90_analogue_phase4 import (V90AnaloguePhase4Controller,
                                 V90AnaloguePhase4MediaTx)
from v90_media_clock import RtpMediaClock
import v90_info_reference as v90
import v34_tx
import v34_info
from v90_dil import dil_codewords_match, dil_g711_codewords
from v90_mp_policy import select_downstream_drn_from_jd


FS = 8000
_PHASE4_RATE_FALLBACK_MAX_RETRIES = 3
_PHASE4_RATE_FALLBACK_BUDGET_SAMPLES = 90 * FS
_DEFAULT_CP_COMMON = {
    "drn": 7,
    "selectors": (0, 1, 2, 0, 1, 2),
    "tx_masks": (tuple(range(128)),) * 3,
    "spectral_sign_bits": 0,
}
_DEFAULT_CP_PROFILE = {
    "analog_rate_mask": 0x1081, "codec_masks": None,
    "lookahead": 1, "gain_q13": 0x4000,
    "shape_q6": (0x10, -16, 0x08, -24), **_DEFAULT_CP_COMMON,
}
_DECODERS = {"pcmu": ulaw_to_linear, "pcma": alaw_to_linear}
_ENCODERS = {"pcmu": linear_to_ulaw, "pcma": linear_to_alaw}
_SILENCE = {"pcmu": 0xff, "pcma": 0xd5}


def _omit_ucode_zero(masks, *, field_name: str):
    """Return CP/CPt masks with Ucode 0 removed from every constellation."""
    if masks is None:
        return None
    filtered = tuple(tuple(value for value in mask if value != 0)
                     for mask in masks)
    if any(not mask for mask in filtered):
        raise ValueError(
            f"phase4_omit_ucode0 leaves an empty {field_name} constellation")
    return filtered


def _write_test_phase3_marker(role: str, media_sample: int) -> None:
    """Emit an opt-in filesystem marker for phase-targeted lab impairment."""
    path = os.environ.get("EL_MODEM_TEST_PHASE3_MARKER")
    if not path:
        return
    try:
        from pathlib import Path
        import time
        Path(path).write_text(f"{role} {time.monotonic():.9f} {media_sample}\n")
    except OSError:
        # Test diagnostics must not change modem behavior.
        pass


def _phase_failure_detail(exc: Exception) -> str:
    detail = str(exc)
    if os.environ.get("V90_DEBUG_TRACEBACK") == "1":
        detail += "\n" + traceback.format_exc()
    return detail


def select_v90_analogue_symbol_rate(info0a: dict, info0d: dict, info1d: dict,
                                    preferred_index: int) -> int | None:
    """Return the highest V.90 upstream rate no higher than the preference.

    INFO1d's projected-rate block must be nonzero and the selected carrier
    must be advertised in the local INFO0a. The 3429-symbol/s rate additionally
    requires INFO0a bit 19 set (raw ``disallow3429`` key) and INFO0d bit 40.
    Indices 3, 4 and 5 represent 3000, 3200 and 3429 symbols/s respectively.
    """
    if (isinstance(preferred_index, bool)
            or not isinstance(preferred_index, int)
            or not 3 <= preferred_index <= 5):
        return None
    if (not isinstance(info0a, dict) or not isinstance(info0d, dict)
            or not isinstance(info1d, dict)):
        return None
    blocks = info1d.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 6:
        return None
    v90_3429 = info0d.get("v90_3429_upstream")
    if isinstance(v90_3429, bool) or v90_3429 not in (0, 1):
        return None
    for index in range(preferred_index, 2, -1):
        block = blocks[index]
        if not isinstance(block, dict):
            continue
        maximum = block.get("max_rate")
        if (isinstance(maximum, bool) or not isinstance(maximum, int)
                or maximum <= 0):
            continue
        carrier = block.get("hi_carrier")
        if isinstance(carrier, bool) or carrier not in (0, 1):
            continue
        if index == 3 and info0a.get("hi3000" if carrier else "lo3000") != 1:
            continue
        if index == 4 and info0a.get("hi3200" if carrier else "lo3200") != 1:
            continue
        if index == 5 and (v90_3429 != 1
                           or info0a.get("rate3429") != 1
                           or info0a.get("disallow3429") != 1):
            continue
        return index
    return None


def select_v34_fallback_symbol_rate(info0a: dict, info1d: dict,
                                    preferred_index: int) -> int | None:
    """Select a V.34-compatible upstream profile when V.90 cannot be used.

    Unlike V.90, V.34 3429 baud does not depend on INFO0d bit 40. It still
    requires the analogue modem's INFO0a permission (raw bit 19) and a
    nonzero peer INFO1d offer for the selected carrier/rate block.
    """
    if (isinstance(preferred_index, bool) or not isinstance(preferred_index, int)
            or not 0 <= preferred_index <= 5 or not isinstance(info0a, dict)
            or not isinstance(info1d, dict)):
        return None
    blocks = info1d.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 6:
        return None
    for index in range(preferred_index, -1, -1):
        block = blocks[index]
        if not isinstance(block, dict):
            continue
        maximum = block.get("max_rate")
        carrier = block.get("hi_carrier", 0)
        if (isinstance(maximum, bool) or not isinstance(maximum, int)
                or maximum <= 0 or isinstance(carrier, bool)
                or carrier not in (0, 1)):
            continue
        supported = (
            index == 0
            or (index == 1 and info0a.get("rate2743") == 1)
            or (index == 2 and info0a.get("rate2800") == 1)
            or (index == 3 and info0a.get("hi3000" if carrier else "lo3000") == 1)
            or (index == 4 and info0a.get("hi3200" if carrier else "lo3200") == 1)
            or (index == 5 and info0a.get("rate3429") == 1
                and info0a.get("allow3429", info0a.get("disallow3429", 0)) == 1)
        )
        if supported:
            return index
    return None


class V90AnalogueCallerPhase2Engine:
    """Run V.90 analogue-caller startup and data under the production DCE API.

    The profile makes the transmitted INFO0a capabilities and signal levels
    explicit. It joins the caller Phase 3 state machine to exact downstream
    G.711 codewords and sample-timed V.34 waveform responses. It does not claim
    CONNECT or pass DTE payload until caller Phase 4 has qualified B1d and LAPM.
    """

    kind = "v90"

    @property
    def supports_v42(self) -> bool:
        """Expose the V.34 fallback's V.42 capability to the DCE.

        Native V.90 Phase 4 owns its LAPM link. Once Phase 2 falls back to
        V.34, however, the ordinary DCE V.42 manager must start and qualify
        the fallback modem's link using the user's configured policy.
        """
        fallback = self._v34_fallback_engine
        return bool(fallback is not None
                    and getattr(fallback, "supports_v42", True))

    @property
    def v42_managed_by_engine(self) -> bool:
        # Pure V.90 Phase 4 owns LAPM. V.34 fallback delegates it to the DCE.
        return self._v34_fallback_engine is None

    @property
    def v42_calling_party(self) -> bool:
        # V.90 §9.2.2.1.9 makes the analogue modem the V.34 answer modem on
        # fallback. V.42 §7.2.1.1 derives LAPM roles from the carrier role
        # assigned by the modulation Recommendation, so this side is the
        # LAPM answerer even though it originated the PSTN call.
        return False

    def start_v42(self, *args, **kwargs) -> None:
        fallback = self._v34_fallback_engine
        if fallback is None:
            raise RuntimeError("V.90 Phase 4 manages its own LAPM link")
        fallback.start_v42(*args, **kwargs)

    @property
    def _v42(self):
        fallback = self._v34_fallback_engine
        if fallback is not None:
            return getattr(fallback, "_v42", None)
        phase4 = self._phase4_controller
        link = None if phase4 is None else phase4.lapm
        return None if link is None else getattr(link, "_v42", None)

    def __init__(self, *, role: str, negotiation: dict, media_at: int,
                 media_origin: float, law: str, profile: dict,
                 phase1_silence_samples: int = 0):
        if role != "analogue-caller":
            raise ValueError("caller engine requires the analogue-caller role")
        if int(negotiation.get("pcm_negotiated_bits", 0)) != V8_PCM_ANALOG:
            raise ValueError("V.8 did not negotiate the analogue PCM role")
        law = str(law).lower()
        if law not in _DECODERS:
            raise ValueError("law must be pcmu or pcma")
        if not isinstance(profile, dict):
            raise TypeError("caller Phase 2 profile must be a mapping")
        omit_ucode0 = profile.get("phase4_omit_ucode0", False)
        if not isinstance(omit_ucode0, bool):
            raise ValueError("phase4_omit_ucode0 must be a boolean")
        cp_omit_ucode0 = profile.get("phase4_cp_omit_ucode0", False)
        if not isinstance(cp_omit_ucode0, bool):
            raise ValueError("phase4_cp_omit_ucode0 must be a boolean")
        if (isinstance(phase1_silence_samples, bool)
                or not isinstance(phase1_silence_samples, int)
                or not 0 <= phase1_silence_samples <= FS):
            raise ValueError("Phase 1 silence must be an integer sample count in 0..8000")
        required = {"info0a_fields", "info1a_fields", "nominal_peak", "probe_rms"}
        missing = required - set(profile)
        if missing:
            raise ValueError("caller Phase 2 profile missing: "
                             + ", ".join(sorted(missing)))
        if not isinstance(profile["info0a_fields"], dict):
            raise TypeError("info0a_fields must be a mapping")
        if not isinstance(profile["info1a_fields"], dict):
            raise TypeError("info1a_fields must be a mapping")
        if not bool(profile.get("prefer_v90", True)):
            raise ValueError("V.90 analogue-caller profile requires prefer_v90=true")
        dil_codewords = profile.get("dil_codewords", 768)
        if (isinstance(dil_codewords, bool) or not isinstance(dil_codewords, int)
                or dil_codewords < 0 or dil_codewords > 768
                or (dil_codewords and dil_codewords % 6)):
            raise ValueError("dil_codewords must be 0 or a multiple of 6 in 6..768")
        cp_override = profile.get("phase4_cp_fields", {})
        if not isinstance(cp_override, dict):
            raise TypeError("phase4_cp_fields must be a mapping")
        downstream_cap = profile.get(
            "phase4_downstream_drn_cap", cp_override.get("drn", 22))
        if (isinstance(downstream_cap, bool)
                or not isinstance(downstream_cap, int)
                or not 1 <= downstream_cap <= 22):
            raise ValueError("phase4_downstream_drn_cap must be an integer in 1..22")
        downstream_pin = profile.get("phase4_downstream_drn_pin")
        if (downstream_pin is not None
                and (isinstance(downstream_pin, bool)
                     or not isinstance(downstream_pin, int)
                     or not 1 <= downstream_pin <= 22)):
            raise ValueError("phase4_downstream_drn_pin must be an integer in 1..22")
        if downstream_pin is not None and downstream_pin > downstream_cap:
            raise ValueError("phase4_downstream_drn_pin cannot exceed its DRN cap")
        phase3_rms = float(profile.get("phase3_rms", 2000.0))
        if not np.isfinite(phase3_rms) or phase3_rms <= 0:
            raise ValueError("phase3_rms must be finite and positive")
        phase3_sd_timeout_ms = profile.get("phase3_sd_timeout_ms", 1500)
        if (isinstance(phase3_sd_timeout_ms, bool)
                or not isinstance(phase3_sd_timeout_ms, int)
                or not 500 <= phase3_sd_timeout_ms <= 10000):
            raise ValueError("phase3_sd_timeout_ms must be an integer in 500..10000")
        phase2_gap_policy = profile.get("phase2_rtp_gap_policy", "fail")
        if phase2_gap_policy not in ("fail", "fill_silence"):
            raise ValueError("phase2_rtp_gap_policy must be fail or fill_silence")
        phase2_gap_max_samples = profile.get("phase2_rtp_gap_max_samples", 0)
        if (isinstance(phase2_gap_max_samples, bool)
                or not isinstance(phase2_gap_max_samples, int)
                or not 0 <= phase2_gap_max_samples <= 1600):
            raise ValueError("phase2_rtp_gap_max_samples must be an integer in 0..1600")
        if phase2_gap_policy == "fill_silence" and phase2_gap_max_samples <= 0:
            raise ValueError("fill_silence requires a positive Phase 2 gap limit")
        clock_low_delay_percentile = profile.get(
            "phase2_clock_low_delay_percentile", 15.0)
        if (isinstance(clock_low_delay_percentile, bool)
                or not isinstance(clock_low_delay_percentile, (int, float))
                or not np.isfinite(clock_low_delay_percentile)
                or not 5.0 <= float(clock_low_delay_percentile) <= 50.0):
            raise ValueError("phase2_clock_low_delay_percentile must be in 5..50")
        clock_max_rate_error_ppm = profile.get(
            "phase2_clock_max_rate_error_ppm", 2000.0)
        if (isinstance(clock_max_rate_error_ppm, bool)
                or not isinstance(clock_max_rate_error_ppm, (int, float))
                or not np.isfinite(clock_max_rate_error_ppm)
                or not 1.0 <= float(clock_max_rate_error_ppm) <= 50000.0):
            raise ValueError(
                "phase2_clock_max_rate_error_ppm must be in 1..50000")
        clock_anchor_limit = profile.get("phase2_clock_anchor_limit", 64)
        if (isinstance(clock_anchor_limit, bool)
                or not isinstance(clock_anchor_limit, int)
                or not 16 <= clock_anchor_limit <= 512):
            raise ValueError("phase2_clock_anchor_limit must be an integer in 16..512")
        clock_min_fit_span = profile.get(
            "phase2_clock_min_fit_span_samples", 6400)
        if (isinstance(clock_min_fit_span, bool)
                or not isinstance(clock_min_fit_span, int)
                or not 6400 <= clock_min_fit_span <= 64000):
            raise ValueError(
                "phase2_clock_min_fit_span_samples must be an integer in 6400..64000")
        for key in ("nominal_peak", "probe_rms"):
            value = float(profile[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")

        self.law = law
        self._decoder = _DECODERS[law]
        self._encoder = _ENCODERS[law]
        self._clock = RtpMediaClock(
            origin=media_origin, media_at=media_at,
            low_delay_percentile=float(clock_low_delay_percentile),
            max_rate_error_ppm=float(clock_max_rate_error_ppm),
            max_anchors=clock_anchor_limit,
            min_fit_span_samples=clock_min_fit_span)
        self._media_at = int(media_at)
        self._rx_at = 0
        self._tx_media = int(media_at)
        self._phase1_silence_samples = phase1_silence_samples
        self._phase2_start_at = int(media_at) + phase1_silence_samples
        self._phase2_gap_policy = phase2_gap_policy
        self._phase2_gap_max_samples = phase2_gap_max_samples
        self._closed = False
        self._failed = False
        self._lock = threading.RLock()
        self._queued_dte = bytearray()
        self._queue_limit = 65536
        self.events: list[tuple[str, object]] = []
        self._controller = V90AnalogueCallerPhase2(
            l2_base_samples=int(profile.get("l2_base_samples", 4400)),
            prefer_v90=bool(profile.get("prefer_v90", True)),
            v34_mode=int(profile.get("v34_mode", 5)),
            late_l2_echo_policy=profile.get(
                "phase2_late_l2_echo_policy", "fail"),
            late_l2_echo_max_samples=int(profile.get(
                "phase2_late_l2_echo_max_samples", 0)),
            accept_early_peer_recovery=bool(profile.get(
                "phase2_accept_early_peer_recovery", True)),
            info1d_timeout_retrain=profile.get(
                "phase2_info1d_timeout_retrain", False))
        info1a_fields = dict(profile["info1a_fields"])
        self._profile_info1a_fields = info1a_fields
        self._profile = dict(profile)
        # As in the V.34 path (modem_dce), a V.8 LAPM agreement starts V.42
        # LAPM directly; V42_FORCE_DETECT_AFTER_V8=1 restores ODP/ADP
        # detection for comparison captures. captured peer answered V.34 XID
        # directly but cleared a V.90 call during ODP detection (2026-11-10).
        v8_lapm = bool(int(negotiation.get("protocols", 0)) & V8_PROTOCOL_LAPM_V42)
        self._phase4_lapm_detect = (
            not v8_lapm
            or os.environ.get("V42_FORCE_DETECT_AFTER_V8", "0") == "1")
        hold_ms = self._profile.get("phase4_data_fault_hold_ms", 0)
        if (isinstance(hold_ms, bool) or not isinstance(hold_ms, int)
                or not 0 <= hold_ms <= 10000):
            raise ValueError("phase4_data_fault_hold_ms must be an integer in 0..10000")
        # V.90 mapper frames are six PCM octets: 8000 / 6 frames per second.
        self._phase4_data_fault_hold_frames = hold_ms * 8000 // 6000
        self._v34_fallback_info1a_fields: dict | None = None
        self._v34_fallback_profile: tuple[dict, int] | None = None
        self._v34_fallback_engine = None
        self._renderer = V90AnaloguePhase2Transmitter(
            info0a_fields=dict(profile["info0a_fields"]),
            info1a_builder=self._build_info1a,
            probe_builder=probing_signal,
            nominal_peak=float(profile["nominal_peak"]),
            probe_rms=float(profile["probe_rms"]),
            # The renderer's cursor is the DCE/RTP transmit sample clock. The
            # controller schedules INFO0a at the end of the Phase 1 silence;
            # starting the cursor there would emit that signal immediately
            # while labeling it with the earlier V.8 handoff timestamp.
            start_at=int(media_at))
        self._adapter = V90AnaloguePhase2Adapter(
            self._controller, rx_to_media=self._clock.rx_to_media)
        self._renderer.submit(self._adapter.start(self._phase2_start_at))
        self._reported_completion: int | None = None
        self._reported_handoffs = 0
        self._pending_v34_fallback_handoff = None
        self._last_state = self._controller.state
        self._dil_codewords = dil_codewords
        self._phase4_downstream_drn_cap = downstream_cap
        self._phase4_downstream_drn_pin = downstream_pin
        self._phase3_rms = phase3_rms
        self._phase3_sd_timeout_ms = phase3_sd_timeout_ms
        self._phase3_controller: V90AnalogueCallerPhase3 | None = None
        self._phase3_receiver: V90AnalogueDownstreamPhase3Receiver | None = None
        self._phase3_session: V90AnaloguePhase3Session | None = None
        self._phase3_tx: V90AnaloguePhase3Transmitter | None = None
        self._phase3_retrain_attempts = 0
        self._phase2_retrain_attempts = 0
        self._phase3_start_at: int | None = None
        self._phase3_start_rx_at: int | None = None
        self._phase3_expected_dil = b""
        self._phase3_received_dil = bytearray()
        self._phase3_ja_descriptor: dict | None = None
        self._phase3_jd_descriptor: dict | None = None
        self._phase4_downstream_selection: dict | None = None
        self._phase4_rate_fallback_attempts = 0
        self._phase4_rate_fallback_started_at: int | None = None
        self._phase3_baud: float | None = None
        self._phase3_carrier: float | None = None
        self._phase4_controller: V90AnaloguePhase4Controller | None = None
        self._phase4_media: V90AnaloguePhase4MediaTx | None = None
        self._phase4_cpt_decisions = None
        self._phase4_pending_decisions = []
        self._phase4_initial_octets = 0
        self._phase4_timeout_at: int | None = None
        self._phase4_start_at: int | None = None
        self._reported_phase4_events = 0

    def _build_info1a(self, mode: int) -> list[int]:
        """Build the selected protocol's INFO1a frame on the shared 2400-Hz carrier."""
        if self._v34_fallback_info1a_fields is not None:
            fields = dict(self._v34_fallback_info1a_fields)
            fields["sym_ans"] = fields["sym_call"] = int(mode)
            return v34_info.build_info1a(**fields)
        return v90.build_info1a_v90(mode=mode, **self._profile_info1a_fields)

    @property
    def connected(self) -> bool:
        if self._v34_fallback_engine is not None:
            return bool(self._v34_fallback_engine.connected)
        phase4 = self._phase4_controller
        if phase4 is None or phase4.state != phase4.DATA:
            return False
        link = phase4.lapm
        return bool(link is not None and link.v42_resolved and not link.v42_lost)

    @property
    def bit_rate(self) -> int:
        # The caller's transmit (analogue-to-digital, V.34 upstream) rate.
        if self._v34_fallback_engine is not None:
            return int(self._v34_fallback_engine.bit_rate)
        profile = (None if self._phase4_controller is None
                   else self._phase4_controller.upstream_profile)
        return 0 if profile is None else int(round(float(profile.rate)))

    @property
    def rx_bit_rate(self) -> int:
        """Receive (digital-to-analogue, V.90 downstream) rate.

        V.90 is asymmetric: the downstream PCM rate is the headline 56k-class
        figure and is what CONNECT should report, not the V.34 upstream.
        Available once Phase 4 reaches DATA; the caller's CP carries it.
        """
        if self._v34_fallback_engine is not None:
            return int(getattr(self._v34_fallback_engine, "rx_bit_rate", 0))
        phase4 = self._phase4_controller
        if (phase4 is None or phase4.state != phase4.DATA
                or not isinstance(phase4.cp_profile, dict)):
            return 0
        rate = phase4.cp_profile.get("rate")
        return 0 if rate is None else int(round(float(rate)))

    @property
    def v42_mode(self) -> str:
        if self._v34_fallback_engine is not None:
            return self._v34_fallback_engine.v42_mode
        link = None if self._phase4_controller is None else self._phase4_controller.lapm
        return "off" if link is None else link.v42_mode

    @property
    def v42_ever_established(self) -> bool:
        """Preserve LAPM history through fallback and post-link failure metrics."""
        if self._v34_fallback_engine is not None:
            return bool(getattr(self._v34_fallback_engine,
                                "_v42_ever_established", False))
        phase4 = self._phase4_controller
        lapm = None if phase4 is None else getattr(phase4, "lapm", None)
        return bool(getattr(lapm, "_v42_ever_established", False))

    @property
    def v42_lost(self) -> bool:
        if self._v34_fallback_engine is not None:
            return bool(self._v34_fallback_engine.v42_lost)
        link = None if self._phase4_controller is None else self._phase4_controller.lapm
        return False if link is None else bool(link.v42_lost)

    @property
    def v42_resolved(self) -> bool:
        if self._v34_fallback_engine is not None:
            return bool(self._v34_fallback_engine.v42_resolved)
        link = None if self._phase4_controller is None else self._phase4_controller.lapm
        return bool(link is not None and link.v42_resolved)

    @property
    def v42_compression(self):
        if self._v34_fallback_engine is not None:
            return self._v34_fallback_engine.v42_compression
        link = None if self._phase4_controller is None else self._phase4_controller.lapm
        return None if link is None else link.v42_compression

    @property
    def failed_reason(self) -> str | None:
        if self._v34_fallback_engine is not None:
            reason = self._v34_fallback_engine.failed_reason
            if reason:
                return str(reason)
        for name, value in reversed(self.events):
            if name in ("phase2-failed", "phase2-retrain-failed",
                        "phase3-failed", "phase3-timeout",
                        "phase4-failed", "phase4-timeout"):
                return str(value)
        return None

    @property
    def phase3_state(self) -> str | None:
        if self._v34_fallback_engine is not None:
            return str(self._v34_fallback_engine.answerer.stage)
        return None if self._phase3_controller is None else self._phase3_controller.state

    @property
    def phase4_state(self) -> str | None:
        return None if self._phase4_controller is None else self._phase4_controller.state

    @property
    def v34_fallback_diagnostics(self) -> dict | None:
        """Return bounded state for the V.34 leg used after V.90 selection.

        The wrapper's ordinary Phase 3/4 controllers are not used on this
        path: a V34AnswerModem owns the fallback handshake and LAPM state.
        Keeping that state visible prevents a successful Phase-2 handoff from
        looking like an unexplained null Phase-3/4 result in call metrics.
        """
        fallback = self._v34_fallback_engine
        if fallback is None:
            return None
        answerer = getattr(fallback, "answerer", None)
        link = getattr(fallback, "_v42", None)
        receiver = getattr(answerer, "_p4rcv", None)
        receiver_eq = getattr(receiver, "eq", None)
        gap_feedback_guard = getattr(receiver_eq, "_transport_guard", None)
        qualification = getattr(answerer, "rx_qualification", None)
        # The PSTN fallback investigations need to distinguish three cases
        # that the former mode/state pair could not: no LAPM symbols reached
        # the V.42 layer, detection saw ODP but rejected/missed ADP, or LAPM
        # decoded a frame and then failed to establish.  Keep this snapshot
        # bounded and collect it only when call metrics are requested.
        detector = getattr(fallback, "_v42_adp_detector", None)
        v42_diagnostics = None
        if link is not None:
            stats = getattr(link, "stats", None)
            v42_diagnostics = {
                "calling": getattr(fallback, "_v42_calling", None),
                "mode": getattr(fallback, "v42_mode", None),
                "state": getattr(link, "state", None),
                "phase": getattr(link, "_phase", None),
                "bits_emitted": getattr(fallback, "_v42_bits", None),
                "detect_deadline_bits": getattr(fallback, "_v42_detect_deadline", None),
                "establish_deadline_bits": getattr(fallback, "_v42_establish_deadline", None),
                "adp_armed": getattr(fallback, "_v42_adp_armed", None),
                "adp_seen": getattr(link, "adp_seen", None),
                "link_bits_sent": getattr(link, "bits_sent", None),
                "link_detect_bits": getattr(link, "_detect_bit_count", None),
                "link_t400_at": getattr(link, "_t400_at", None),
                "answer_detect_at": getattr(link, "_answer_detect_at", None),
                "decoded_chars_hex": bytes(getattr(link, "_chars", ()))[-64:].hex(),
                "decoded_char_gaps": list(getattr(link, "_char_gaps", ())[-64:]),
                "partial_async_bits": list(getattr(link, "_async", ())[-32:]),
                "answerer_adp_count": getattr(link, "_adp_count", None),
                "answerer_adp_started": getattr(link, "_answer_adp", None),
                "hdlc_flags": getattr(getattr(link, "_hdlc", None), "flags", None),
                "hdlc_continuous_flags": getattr(
                    getattr(link, "_hdlc", None), "continuous_flags", None),
                "adp_detector_result": getattr(detector, "result", None),
                "adp_marker_tail_hex": bytes(getattr(detector, "_buf", b""))[-16:].hex(),
                "adp_gap_tail_bits": list(getattr(detector, "_gaps", ()))[-15:],
                "link_stats": dict(stats) if isinstance(stats, dict) else None,
                "link_events": list(getattr(link, "events", ())[-64:]),
                "detector_trace": fallback.v42_adp_trace(),
                "bit_trace": fallback.v42_bit_trace(),
            }
        b1_qualification = None
        if isinstance(qualification, dict):
            keys = (
                "profile", "qualified", "reason", "bit_errors", "bit_count",
                "held_out_nmse", "calibration_nmse", "tap_calibration_nmse",
                "energy", "gain_real", "gain_imag", "length_mismatch",
                "mapper_symbol_errors", "differential_convention",
                "convention_policy", "boundary_offset", "rate_candidates",
                "state_candidates", "boundary_candidates",
                "standard_qualification",
            )
            b1_qualification = {key: qualification[key] for key in keys
                                if key in qualification}
            error_indices = qualification.get("error_bit_indices")
            if isinstance(error_indices, (list, tuple)):
                b1_qualification["error_bit_index_count"] = len(error_indices)
                b1_qualification["first_error_bit_indices"] = list(error_indices[:32])
            divergence = qualification.get("first_divergence")
            if isinstance(divergence, dict):
                b1_qualification["first_divergence"] = {
                    key: divergence[key] for key in ("symbol", "interval")
                    if key in divergence
                }
        return {
            "modem_state": getattr(fallback, "state", None),
            "answerer_stage": getattr(answerer, "stage", None),
            "answerer_phase4_state": getattr(answerer, "_p4_state", None),
            "connected": bool(getattr(fallback, "connected", False)),
            "tx_rate": int(getattr(fallback, "bit_rate", 0)),
            "rx_rate": int(getattr(fallback, "rx_bit_rate", 0)),
            "failed_reason": getattr(fallback, "failed_reason", None),
            "b1_qualification": b1_qualification,
            "answerer_data_rx_stage_profile": (
                None if getattr(answerer, "_data_rx_stage_profile", None) is None
                else list(answerer._data_rx_stage_profile)),
            "answerer_p4_poll_profile": (
                None if getattr(answerer, "_p4_poll_profile", None) is None
                else list(answerer._p4_poll_profile)),
            "answerer_tx_samples": int(getattr(answerer, "t", 0)),
            "answerer_rx_samples": int(getattr(answerer, "rx_t", 0)),
            "phase4_rx": (None if receiver is None else {
                "state": getattr(answerer, "_p4_state", None),
                "symbols_seen": len(getattr(receiver, "idx", ())),
                "trn_end": getattr(receiver, "trn_end", None),
                "end_estimate": getattr(receiver, "_end_est", None),
                "frames": list(getattr(receiver, "frames", ())[-8:]),
                "b1_symbol": getattr(receiver, "b1_symbol", None),
                "timing_slips": int(getattr(receiver_eq, "slips", 0)),
                "gap_feedback": (None if gap_feedback_guard is None
                                 else gap_feedback_guard.report()),
                "equalizer_timing": (None if receiver_eq is None else
                                     receiver_eq.timing_snapshot()),
            }),
            "mp_state": {
                "sent": getattr(answerer, "mp_ans", None),
                "sent_ack": getattr(answerer, "_mp_ack", None),
                "ack_frame_sent": getattr(answerer, "_mp_ack_sent", None),
                "e_sent": getattr(answerer, "e_sent", None),
                "peer_mp": getattr(answerer, "mp_call", None),
            },
            "v42_mode": getattr(fallback, "v42_mode", None),
            "v42_state": (None if link is None else
                          str(getattr(link, "state", "unknown"))),
            "v42_diagnostics": v42_diagnostics,
            "events": list(getattr(fallback, "events", ())[-128:]),
            "answerer_events": list(getattr(answerer, "events", ())[-128:]),
        }

    def is_settled(self) -> bool:
        return self.connected

    @property
    def pending_bits(self) -> int:
        if self._v34_fallback_engine is not None:
            return int(self._v34_fallback_engine.pending_bits())
        phase4 = self._phase4_controller
        if phase4 is not None and phase4.lapm is not None:
            return int(phase4.lapm.pending_bits())
        return 8 * len(self._queued_dte)

    def tx_samples(self, count: int) -> list[int]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        with self._lock:
            if self._closed or self._failed:
                self._tx_media += count
                return [0] * count
            if self._v34_fallback_engine is not None:
                samples = self._v34_fallback_engine.tx_samples(count)
                self._tx_media += count
                return samples
            start = self._tx_media
            pcm = self._renderer.render(count)
            if self._phase3_tx is not None:
                cursor = self._phase3_tx.cursor
                if cursor is None:
                    raise RuntimeError("caller Phase 3 transmitter has no sample cursor")
                offset = 0
                if cursor < start:
                    # The RX thread may recognize the INFO1a boundary after
                    # the TX thread has crossed it. Only backfill within the
                    # protocol's 70 ms silent lead; beyond that, training was
                    # missed and the exchange cannot be claimed valid.
                    late = start - cursor
                    if late > 70 * FS // 1000:
                        raise RuntimeError("caller Phase 3 transmitter missed its scheduled start")
                    self._phase3_tx.read_pcm(late)
                elif cursor > start:
                    offset = min(count, cursor - start)
                active_count = count - offset
                # INFO1a completion is the Phase 2/3 handoff. The Phase 2
                # renderer may still hold Tone A as its last mode; it must
                # not be mixed underneath the mandated Phase 3 silence or
                # the caller's shaped QAM waveform.
                pcm[offset:] = 0.0
                if active_count:
                    pcm[offset:] += self._phase3_tx.read_pcm(active_count)
            if self._phase4_media is not None:
                pcm += self._phase4_media.read_pcm(count)
            self._tx_media += count
            self._pump_timer()
            self._pump_tx_events()
            self._maybe_start_phase3()
            self._start_phase4_media()
            rendered = np.clip(np.rint(pcm), -32768, 32767).astype(np.int16)
            return rendered.tolist()

    def tx_g711(self, count: int, codec: str) -> bytes:
        if codec != self.law:
            raise ValueError("RTP codec changed after V.8 PCM negotiation")
        return bytes(self._encoder(int(sample)) for sample in self.tx_samples(count))

    def feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("linear PCM must contain complete 16-bit samples")
        samples = np.frombuffer(pcm, dtype="<i2")
        payload = bytes(self._encoder(int(sample)) for sample in samples)
        deadline = self._clock.origin + self._rx_at / FS
        return self.feed_g711(payload, self.law, {
            "packet_playout_deadline_monotonic": deadline,
            "payload_samples": len(payload),
        })

    def feed_g711(self, payload: bytes, codec: str, context: dict) -> bytes:
        if codec != self.law:
            raise ValueError("RTP codec changed after V.8 PCM negotiation")
        data = bytes(payload)
        if not data:
            return b""
        with self._lock:
            if self._closed or self._failed:
                return b""
            try:
                self._clock.observe(self._rx_at, context)
                samples = np.fromiter((self._decoder(value) for value in data),
                                      dtype=float, count=len(data))
                if self._v34_fallback_engine is not None:
                    pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
                    deadline = context.get("packet_playout_deadline_monotonic")
                    set_deadline = getattr(
                        self._v34_fallback_engine, "set_rx_media_deadline", None)
                    if callable(set_deadline) and deadline is not None:
                        set_deadline(float(deadline))
                    result = self._v34_fallback_engine.feed_pcm(pcm)
                    self._rx_at += len(data)
                    return result
                tx_now = max(self._clock.tx_now(), self._renderer.cursor)
                actions = self._adapter.feed(
                    samples, rx_at=self._rx_at, tx_now=tx_now)
                actions = self._prepare_v90_info1a(actions)
                self._renderer.submit(actions)
                self._rx_at += len(data)
                self._pump_timer()
                self._pump_tx_events()
                self._maybe_start_phase3()
                self._feed_phase3(data)
                if self._controller.state != self._last_state:
                    self.events.append(("phase2-state", self._controller.state))
                    self._last_state = self._controller.state
                if self._phase4_controller is None:
                    return b""
                return self._phase4_controller.take_received()
            except (TypeError, ValueError, RuntimeError, KeyError, IndexError) as exc:
                self._failed = True
                detail = _phase_failure_detail(exc)
                if self._phase4_controller is not None:
                    self._phase4_controller.state = self._phase4_controller.FAILED
                    self.events.append(("phase4-failed", detail))
                elif self._phase3_controller is not None:
                    self._phase3_controller.state = self._phase3_controller.FAILED
                    self.events.append(("phase3-failed", detail))
                else:
                    self._controller.state = self._controller.FAILED
                    self.events.append(("phase2-failed", detail))
                return b""

    def _prepare_v90_info1a(self, actions: list) -> list:
        """Apply the peer's measured V.90 upstream-rate constraints.

        Do this after INFO1d is decoded but before its INFO1a action reaches
        the transmitter. INFO1d's V.34 3429 profile alone is not sufficient:
        V.90 3429 upstream additionally requires INFO0d bit 40.
        """
        request = next((action for action in actions
                        if getattr(action, "signal", None) == "info1a"
                        and getattr(action, "value", None) == 6), None)
        if request is None:
            return actions
        info0d = self._controller.info0d
        info1d = self._controller.info1d
        preferred = int(self._profile_info1a_fields.get(
            "analog_symbol_rate", 5))
        selected = select_v90_analogue_symbol_rate(
            self._profile.get("info0a_fields"), info0d, info1d, preferred)
        if selected is None:
            v34_preferred = int(self._profile.get("v34_fallback_symbol_rate",
                                                 preferred))
            v34_selected = select_v34_fallback_symbol_rate(
                self._profile.get("info0a_fields"), info1d, v34_preferred)
            if v34_selected is not None:
                block = info1d["blocks"][v34_selected]
                self._v34_fallback_profile = (dict(info1d), v34_selected)
                self._v34_fallback_info1a_fields = {
                    # INFO1d's power fields command our transmitter; they are
                    # not values to mirror back as INFO1a requests to the peer.
                    "power_min": int(self._profile.get(
                        "v34_fallback_info1a_fields", {}).get("power_min", 0)),
                    "power_extra": int(self._profile.get(
                        "v34_fallback_info1a_fields", {}).get("power_extra", 0)),
                    "md_len": int(self._profile_info1a_fields.get("md_len", 0)),
                    "hi_carrier": int(block.get("hi_carrier", 0)),
                    "preemph": int(block.get("preemph", 0)),
                    "max_rate": int(block["max_rate"]),
                    "freq_offset": int(self._profile_info1a_fields.get(
                        "freq_offset", 0)),
                }
                self._controller.mode = v34_selected
                self.events.append(("phase2-v34-fallback-selected", {
                    "symbol_rate_index": v34_selected,
                    "baud": v34_tx.SYMBOL_RATES[v34_selected],
                    "peer_info0d_v90_3429": (info0d.get("v90_3429_upstream")
                                               if isinstance(info0d, dict) else None),
                    "peer_info1d_rate": int(block["max_rate"]),
                }))
                rewritten = []
                for action in actions:
                    if (getattr(action, "signal", None) == "info1a"
                            and getattr(action, "value", None) == 6):
                        action = replace(action, value=v34_selected)
                    elif (getattr(action, "signal", None) == "handoff-v90-phase3"):
                        action = replace(action,
                                         signal="handoff-v34-answer-phase3",
                                         value=v34_selected)
                    rewritten.append(action)
                return rewritten
            blocks = info1d.get("blocks") if isinstance(info1d, dict) else None
            rates = ([block.get("max_rate") if isinstance(block, dict) else None
                      for block in blocks] if isinstance(blocks, list) else None)
            self._failed = True
            self._controller.state = self._controller.FAILED
            self.events.append(("phase2-failed", {
                "reason": "peer INFO0d/INFO1d offers no usable V.90 analogue-to-digital symbol rate",
                "preferred_index": preferred,
                "v90_3429_upstream": (info0d.get("v90_3429_upstream")
                                        if isinstance(info0d, dict) else None),
                "info1d_projected_rates": rates,
                "local_info0a": dict(self._profile.get("info0a_fields") or {}),
                "peer_info0d": dict(info0d) if isinstance(info0d, dict) else None,
                "peer_info1d": dict(info1d) if isinstance(info1d, dict) else None,
                "v34_fallback_available": False,
            }))
            return []
        self._profile_info1a_fields["analog_symbol_rate"] = selected
        self.events.append(("phase2-rate-selected", {
            "analog_symbol_rate_index": selected,
            "baud": v34_tx.SYMBOL_RATES[selected],
            "preferred_index": preferred,
            "info1d_projected_rate": info1d["blocks"][selected]["max_rate"],
        }))
        return actions

    def feed_gap(self, samples: int, codec: str, context: dict) -> bytes:
        if codec != self.law:
            raise ValueError("RTP codec changed after V.8 PCM negotiation")
        if samples < 0:
            raise ValueError("gap length must be nonnegative")
        with self._lock:
            if samples:
                if self._v34_fallback_engine is not None:
                    fallback = self._v34_fallback_engine
                    fallback.note_concealed_rx(int(samples))
                    gap_context = dict(context)
                    gap_context["payload_samples"] = int(samples)
                    sample = _DECODERS[self.law](_SILENCE[self.law])
                    concealed = np.full(int(samples), sample, dtype="<i2")
                    deadline = gap_context.get("packet_playout_deadline_monotonic")
                    set_deadline = getattr(fallback, "set_rx_media_deadline", None)
                    if callable(set_deadline) and deadline is not None:
                        set_deadline(float(deadline))
                    result = fallback.feed_pcm(concealed.tobytes())
                    self._rx_at += int(samples)
                    return result
                if (self._phase3_session is None
                        and self._phase2_gap_policy == "fill_silence"
                        and samples <= self._phase2_gap_max_samples
                        and not self._closed and not self._failed):
                    # This is an explicit, bounded playout-concealment policy,
                    # not recovered signal: preserve the receive clock by
                    # advancing the Phase 2 detectors through G.711 silence.
                    # The default remains fail-closed because probe/tone timing
                    # can be corrupted by missing media.
                    gap_context = dict(context)
                    gap_context["payload_samples"] = int(samples)
                    result = self.feed_g711(
                        bytes((_SILENCE[self.law],)) * int(samples),
                        codec, gap_context)
                    if not self._failed:
                        self.events.append(("phase2-rx-gap-filled", {
                            "samples": int(samples), "context": dict(context),
                        }))
                    return result
                self._rx_at += int(samples)
                stage = ("phase3" if self._phase3_session is not None
                         else "phase2")
                self._failed = True
                if stage == "phase3" and self._phase3_controller is not None:
                    self._phase3_controller.state = self._phase3_controller.FAILED
                else:
                    self._controller.state = self._controller.FAILED
                self.events.append((f"{stage}-failed", {
                    "reason": f"RTP gap during V.90 analogue-caller {stage.title()}",
                    "samples": int(samples), "context": dict(context),
                }))
        return b""

    def feed_bytes(self, data: bytes) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("V.90 caller engine is closed")
            payload = bytes(data)
            if self._v34_fallback_engine is not None:
                self._v34_fallback_engine.feed_bytes(payload)
                return
            if (self._phase4_controller is not None
                    and self._phase4_controller.lapm is not None):
                self._phase4_controller.feed_bytes(payload)
                return
            if len(self._queued_dte) + len(payload) > self._queue_limit:
                raise OverflowError("DTE queue exceeds 64 KiB before Phase 4")
            self._queued_dte.extend(payload)

    def _pump_tx_events(self) -> None:
        completed = self._renderer._completion.get("info0a-complete")
        if completed is not None and completed <= self._renderer.cursor \
                and completed != self._reported_completion:
            self._reported_completion = completed
            actions = self._adapter.feed_tx_complete(
                "info0a_complete", media_at=completed,
                tx_now=max(self._clock.tx_now(), self._renderer.cursor))
            self._renderer.submit(actions)
        # INFO1a's handoff is scheduled when the controller selects a mode,
        # which can be ahead of the media cursor. Do not replace the renderer
        # with the V.34 engine until the complete INFO1a has actually gone on
        # the wire; otherwise its tail is truncated and the peer never sees a
        # CRC-valid frame.
        pending = self._pending_v34_fallback_handoff
        if (pending is not None and pending.at <= self._renderer.cursor
                and self._v34_fallback_engine is None):
            self._start_v34_fallback(pending.mode)
            self._pending_v34_fallback_handoff = None

        while self._reported_handoffs < len(self._renderer.handoffs):
            handoff = self._renderer.handoffs[self._reported_handoffs]
            self.events.append(("phase2-handoff", {
                "signal": handoff.signal, "at": handoff.at,
                "mode": handoff.mode,
            }))
            self._reported_handoffs += 1
            if handoff.signal.startswith("handoff-retrain-"):
                if handoff.signal == "handoff-retrain-v90-phase2":
                    self._start_phase2_retrain(handoff)
                else:
                    self._failed = True
                    self.events.append((
                        "phase2-failed",
                        f"unsupported V.90 retrain handoff ({handoff.signal})"))
            elif handoff.signal == "handoff-v34-answer-phase3":
                self._pending_v34_fallback_handoff = handoff
                if handoff.at <= self._renderer.cursor:
                    self._start_v34_fallback(handoff.mode)
                    self._pending_v34_fallback_handoff = None

    def _start_phase2_retrain(self, handoff) -> None:
        """Restart V.90 Phase 2 once after its bounded recovery handoff."""
        info0d = self._controller.info0d
        if (self._phase2_retrain_attempts >= 1
                or self._phase3_retrain_attempts >= 1):
            self._controller.state = self._controller.FAILED
            self._failed = True
            self.events.append(("phase2-retrain-failed", {
                "reason": ("V.90 Phase-2 retrain limit reached"
                           if self._phase2_retrain_attempts >= 1 else
                           "V.90 Phase-3 recovery already used"),
                "handoff": handoff.signal,
                "attempts": self._phase2_retrain_attempts,
                "safe_disconnect": True,
            }))
            self.events.append(("phase2-failed",
                                "bounded V.90 retrain recovery exhausted"))
            return
        if not isinstance(info0d, dict) or not info0d:
            self._controller.state = self._controller.FAILED
            self._failed = True
            self.events.append(("phase2-retrain-failed", {
                "reason": "V.90 Phase-2 retrain lacks retained INFO0d",
                "handoff": handoff.signal,
                "safe_disconnect": True,
            }))
            return

        old = self._controller
        media_at = max(int(handoff.at or 0), self._tx_media,
                       self._renderer.cursor)
        controller = V90AnalogueCallerPhase2(
            l2_base_samples=int(self._profile.get("l2_base_samples", 4400)),
            prefer_v90=bool(self._profile.get("prefer_v90", True)),
            v34_mode=int(self._profile.get("v34_mode", 5)),
            late_l2_echo_policy=self._profile.get(
                "phase2_late_l2_echo_policy", "fail"),
            late_l2_echo_max_samples=int(self._profile.get(
                "phase2_late_l2_echo_max_samples", 0)),
            accept_early_peer_recovery=bool(self._profile.get(
                "phase2_accept_early_peer_recovery", True)),
            info1d_timeout_retrain=False)
        adapter = V90AnaloguePhase2Adapter(
            controller, rx_to_media=self._clock.rx_to_media)
        actions = adapter.start_v90_retrain_initiation(
            media_at, info0d=info0d)
        renderer = V90AnaloguePhase2Transmitter(
            info0a_fields=dict(self._profile.get("info0a_fields", {})),
            info1a_builder=self._build_info1a,
            probe_builder=probing_signal,
            nominal_peak=float(self._profile["nominal_peak"]),
            probe_rms=float(self._profile["probe_rms"]),
            start_at=media_at)
        renderer.submit(actions)
        self._controller = controller
        self._adapter = adapter
        self._renderer = renderer
        self._phase2_retrain_attempts += 1
        self._reported_completion = None
        self._reported_handoffs = 0
        self._pending_v34_fallback_handoff = None
        self._phase4_timeout_at = None
        self._last_state = controller.state
        self.events.append(("phase2-retrain-started", {
            "media_at": media_at,
            "attempt": self._phase2_retrain_attempts,
            "previous_state": old.state,
            "retained_info0d": True,
            "same_engine": True,
            "initial_silence_ms": 70,
            "tone_b_watchdog_ms": 5000,
        }))

    def _start_v34_fallback(self, rate_index: int | None) -> None:
        if self._v34_fallback_profile is None:
            self._failed = True
            self.events.append(("phase2-failed",
                                "V.34 fallback handoff has no selected profile"))
            return
        info1d, selected_index = self._v34_fallback_profile
        if rate_index is not None and int(rate_index) != selected_index:
            raise ValueError("V.34 fallback handoff rate differs from selected profile")
        rate = max(2400, int(info1d["blocks"][selected_index]["max_rate"]) * 2400)
        from v34_answer_modem import V34AnswerModem
        if self._v34_fallback_engine is not None:
            return
        fallback = V34AnswerModem(
            fs=FS, data_rate=rate, min_data_rate=2400,
            rms=self._phase3_rms, with_v8=False)
        fallback.begin_v90_v34_fallback(info1d, selected_index)
        if self._queued_dte:
            fallback.feed_bytes(bytes(self._queued_dte))
            self._queued_dte.clear()
        self._v34_fallback_engine = fallback
        capture_callback = getattr(
            self, '_dce_v34_fallback_capture_callback', None)
        if callable(capture_callback):
            capture_callback(fallback, kind='v90-v34-fallback')
        self.events.append(("phase2-v34-fallback-started", {
            "symbol_rate_index": selected_index,
            "baud": v34_tx.SYMBOL_RATES[selected_index],
            "data_rate_ceiling": rate,
        }))
        _write_test_phase3_marker("caller", self._rx_at)

    def _start_phase3_retrain(self, *, media_at: int, reason: str,
                              rate_fallback: bool = False) -> None:
        """Start an analogue-initiated V.90 retrain from a failed Phase 3.

        V.90 §9.3 permits the analogue modem to initiate retrain, and §9.5.2.1
        defines the Tone-A/Tone-B preamble. Keep the retry count bounded and
        fail closed if there is no retained V.90 negotiation state.
        """
        info0d = self._controller.info0d
        retry_limit = (_PHASE4_RATE_FALLBACK_MAX_RETRIES
                       if rate_fallback else 1)
        if self._phase3_retrain_attempts >= retry_limit:
            self._failed = True
            self.events.append(("phase3-retrain-failed", {
                "reason": "V.90 Phase-3 retrain limit reached",
                "rate_fallback": bool(rate_fallback),
                "attempts": self._phase3_retrain_attempts,
                "limit": retry_limit,
                "safe_disconnect": True,
            }))
            return
        if (rate_fallback
                and self._phase3_retrain_attempts
                > self._phase4_rate_fallback_attempts):
            self._failed = True
            self.events.append(("phase3-retrain-failed", {
                "reason": "non-rate Phase-3 recovery already used",
                "rate_fallback": True,
                "safe_disconnect": True,
            }))
            return
        if not isinstance(info0d, dict) or not info0d:
            self._failed = True
            self.events.append(("phase3-retrain-failed", {
                "reason": "V.90 Phase-3 retrain lacks retained INFO0d",
                "safe_disconnect": True,
            }))
            return

        previous_phase3 = (None if self._phase3_controller is None else
                            self._phase3_controller.state)
        controller = V90AnalogueCallerPhase2(
            l2_base_samples=int(self._profile.get("l2_base_samples", 4400)),
            prefer_v90=True,
            v34_mode=int(self._profile.get("v34_mode", 5)),
            late_l2_echo_policy=self._profile.get(
                "phase2_late_l2_echo_policy", "fail"),
            late_l2_echo_max_samples=int(self._profile.get(
                "phase2_late_l2_echo_max_samples", 0)),
            accept_early_peer_recovery=bool(self._profile.get(
                "phase2_accept_early_peer_recovery", True)),
            info1d_timeout_retrain=False)
        adapter = V90AnaloguePhase2Adapter(
            controller, rx_to_media=self._clock.rx_to_media)
        actions = adapter.start_v90_retrain_initiation(
            int(media_at), info0d=info0d)
        renderer = V90AnaloguePhase2Transmitter(
            info0a_fields=dict(self._profile.get("info0a_fields", {})),
            info1a_builder=self._build_info1a,
            probe_builder=probing_signal,
            nominal_peak=float(self._profile["nominal_peak"]),
            probe_rms=float(self._profile["probe_rms"]),
            start_at=int(media_at))
        self._controller = controller
        self._adapter = adapter
        self._renderer = renderer
        self._renderer.submit(actions)
        self._reported_completion = None
        self._reported_handoffs = 0
        self._pending_v34_fallback_handoff = None
        self._phase3_controller = None
        self._phase3_receiver = None
        self._phase3_session = None
        self._phase3_tx = None
        self._phase3_start_at = None
        self._phase3_start_rx_at = None
        self._phase3_expected_dil = b""
        self._phase3_received_dil.clear()
        # Every new Phase 3 must negotiate a fresh Jd and deliver a fresh DIL.
        # Carrying either observation across retrain can make Phase 4 select
        # a rate from stale channel data.
        self._phase3_ja_descriptor = None
        self._phase3_jd_descriptor = None
        self._phase4_downstream_selection = None
        self._phase3_baud = None
        self._phase3_carrier = None
        self._phase4_controller = None
        self._phase4_media = None
        self._phase4_cpt_decisions = None
        self._phase4_pending_decisions.clear()
        self._phase4_initial_octets = 0
        self._phase4_timeout_at = None
        self._phase4_start_at = None
        self._reported_phase4_events = 0
        self._phase3_retrain_attempts += 1
        self._last_state = controller.state
        self.events.append(("phase3-retrain-started", {
            "at": int(media_at),
            "reason": reason,
            "attempt": self._phase3_retrain_attempts,
            "previous_phase3_state": previous_phase3,
            "retained_info0d": True,
            "analog_tone_a_after_silence_ms": 70,
            "tone_b_watchdog_ms": 5000,
        }))

    def _pump_timer(self) -> None:
        now = max(self._clock.tx_now(), self._renderer.cursor)
        if (not self._failed and not self.connected
                and self._phase4_rate_fallback_attempts > 0
                and self._phase4_rate_fallback_started_at is not None
                and now >= (self._phase4_rate_fallback_started_at
                            + _PHASE4_RATE_FALLBACK_BUDGET_SAMPLES)):
            self._failed = True
            self._phase4_timeout_at = None
            self.events.append(("phase4-rate-walkdown-exhausted", {
                "reason": "connect_time_budget_expired",
                "attempts": self._phase4_rate_fallback_attempts,
                "budget_samples": _PHASE4_RATE_FALLBACK_BUDGET_SAMPLES,
                "elapsed_samples": max(
                    0, now - self._phase4_rate_fallback_started_at),
            }))
            return
        actions = self._adapter.tick(media_now=now)
        if actions:
            self._renderer.submit(actions)
        if self._phase3_controller is not None:
            actions = self._phase3_controller.tick(now)
            if actions:
                details = [
                    {"signal": action.signal, "at": action.at}
                    for action in actions]
                self.events.append(("phase3-timeout", details))
                if any(action.signal == "handoff-retrain"
                       for action in actions):
                    self._start_phase3_retrain(
                        media_at=now,
                        reason="V.90 Phase-3 recovery timer expired")
                else:
                    self._failed = True
        if (self._phase4_timeout_at is not None
                and now >= self._phase4_timeout_at and not self.connected):
            self.events.append(("phase4-timeout", {
                "reason": "caller Phase 4 did not establish LAPM",
                "state": self.phase4_state,
            }))
            self._phase4_timeout_at = None
            # A timeout can mean missing CP/Ed/LAPM traffic after B1d passed.
            # Only an explicit B1d qualification failure can justify a DRN
            # step-down; timeouts fail closed without pretending B1d failed.
            self._failed = True

    def _maybe_start_phase3(self) -> None:
        if self._phase3_session is not None:
            return
        if (self._controller.state != self._controller.COMPLETE
                or self._controller.mode != 6
                or not self._controller.info1d):
            return
        start_at = self._renderer._completion.get("info1a-complete")
        if start_at is None:
            return
        # The signal builder owns INFO1a fields; the selected baud is the
        # profile's analogue symbol-rate index, and the far-end INFO1d block
        # supplies its permitted carrier choice.
        rate_index = int(self._profile_info1a_fields.get(
            "analog_symbol_rate", 5))
        if not 0 <= rate_index < len(v34_tx.SYMBOL_RATES):
            raise ValueError("negotiated analogue symbol-rate index is invalid")
        baud = float(v34_tx.SYMBOL_RATES[rate_index])
        blocks = self._controller.info1d.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != 6:
            raise ValueError("negotiated INFO1d must contain six rate blocks")
        carrier_index = int(blocks[rate_index].get("hi_carrier", 0))
        carriers = v34_tx.CARRIERS.get(round(baud))
        if carriers is None or not 0 <= carrier_index < len(carriers):
            raise ValueError("negotiated INFO1d has an invalid caller carrier")

        if self._dil_codewords:
            # Keep the established one-level DIL on the production path. The
            # experimental multilevel descriptor is useful for margin studies,
            # but changes the upstream waveform and is not required to select
            # a downstream rate from the peer's Jd capability mask.
            h0 = self._dil_codewords // 6 - 1
            ja_fields = dict(
                N=1, LSP=1, LTP=1, SP=[0], TP=[0],
                H=[h0] + [0] * 7, REF=[0] * 8, Ucodes=[0])
        else:
            ja_fields = dict(
                N=0, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
                REF=[0] * 8, Ucodes=[])
        ja_bits = v90.build_ja_descriptor(**ja_fields)
        descriptor = v90.parse_ja_descriptor(ja_bits)
        if descriptor is None or descriptor.get("valid") is not True:
            raise ValueError("local caller Ja descriptor did not validate")
        self._phase3_ja_descriptor = descriptor
        self._phase3_expected_dil = dil_g711_codewords(descriptor, self.law)
        self._phase3_controller = V90AnalogueCallerPhase3(
            baud=baud, ja_bits=ja_bits, descriptor=descriptor,
            sd_timeout_ms=self._phase3_sd_timeout_ms)
        self._phase3_receiver = V90AnalogueDownstreamPhase3Receiver(
            int(self._profile_info1a_fields.get("uinfo", 90)), self.law)
        self._phase3_tx = V90AnaloguePhase3Transmitter(
            self._phase3_controller, carrier=carriers[carrier_index],
            rms=self._phase3_rms)
        self._phase3_baud = baud
        self._phase3_carrier = float(carriers[carrier_index])
        self._phase3_session = V90AnaloguePhase3Session(
            self._phase3_controller, self._phase3_receiver, self._phase3_tx)
        self._phase3_start_at = int(start_at)
        self._phase3_start_rx_at = self._clock.media_to_rx(self._phase3_start_at)
        actions = self._phase3_session.start(self._phase3_start_at)
        self.events.append(("phase3-start", {
            "at": self._phase3_start_at, "baud": baud,
            "carrier": carriers[carrier_index],
            "dil_codewords": len(self._phase3_expected_dil),
            "initial_actions": [action.signal for action in actions],
        }))

    def _feed_phase3(self, raw_g711: bytes) -> None:
        if (self._phase3_session is None or self._phase3_start_at is None
                or self._phase3_controller is None):
            return
        rx_start = self._rx_at - len(raw_g711)
        phase3_start_rx = self._phase3_start_rx_at
        if phase3_start_rx is None or rx_start + len(raw_g711) <= phase3_start_rx:
            return
        offset = max(0, phase3_start_rx - rx_start)
        codewords = raw_g711[offset:]
        if not codewords:
            return
        sequence_at = rx_start + offset
        start_at = self._clock.rx_to_media(sequence_at)
        records = self._phase3_session.feed_downstream_g711(
            codewords, at=start_at, sequence_at=sequence_at)
        phase4_tail = bytearray()
        phase4_ready_at = None
        for record in records:
            details = dict(record["details"])
            if record["event"] == "jd":
                self._phase3_jd_descriptor = dict(details.get("descriptor") or {})
            elif record["event"] == "dil_codewords":
                chunk = bytes(details.pop("codewords"))
                self._phase3_received_dil.extend(chunk)
                details["codeword_count"] = len(chunk)
            elif record["event"] == "phase4_codewords":
                chunk = bytes(details.pop("codewords"))
                phase4_tail.extend(chunk)
                details["codeword_count"] = len(chunk)
            for action in record.get("actions", ()):
                if action.signal == "phase4-ready":
                    phase4_ready_at = int(action.at)
            self.events.append(("phase3-event", {
                "event": record["event"], "at": int(record["at"]),
                "details": details,
            }))
        if self._phase3_controller.phase4_ready:
            received_dil = bytes(self._phase3_received_dil)
            if len(received_dil) != len(self._phase3_expected_dil):
                self._failed = True
                self._phase3_controller.state = self._phase3_controller.FAILED
                self.events.append(("phase3-failed", {
                    "reason": "caller received an incomplete DIL sequence",
                    "received": len(self._phase3_received_dil),
                    "expected": len(self._phase3_expected_dil),
                }))
                return
            matches = dil_codewords_match(
                self._phase3_expected_dil, received_dil, self.law)
            if not any(name == "phase3-ready" for name, _ in self.events):
                self.events.append(("phase3-ready", {
                    "dil_codewords": len(self._phase3_received_dil),
                    "exact": received_dil == self._phase3_expected_dil,
                    "matches_descriptor": matches,
                }))
            if self._phase4_controller is None:
                self._start_phase4(phase4_ready_at)
        if phase4_tail:
            self._feed_phase4(bytes(phase4_tail))

    def _start_phase4(self, ready_at: int | None) -> None:
        if self._phase4_controller is not None:
            return
        if ready_at is None:
            raise RuntimeError("Phase 3 became ready without a timed Phase 4 boundary")
        if self._phase3_baud is None or self._phase3_carrier is None:
            raise RuntimeError("Phase 4 lacks negotiated carrier parameters")
        constellation = int(self._profile.get("phase4_constellation", 4))
        if constellation not in (4, 16):
            raise ValueError("phase4_constellation must be 4 or 16")
        cpt_fields = dict(_DEFAULT_CP_PROFILE)
        cp_fields = dict(_DEFAULT_CP_PROFILE)
        cpt_fields.update(dict(self._profile.get("phase4_cpt_fields", {})))
        cp_fields.update(dict(self._profile.get("phase4_cp_fields", {})))
        omit_fields = []
        if self._profile.get("phase4_omit_ucode0", False):
            omit_fields = [cpt_fields, cp_fields]
        elif self._profile.get("phase4_cp_omit_ucode0", False):
            # captured peer rejected a Ucode-0-free CPt (call 4, 2026-11-10) but
            # flattens PCMU -0 to +0, erasing the sign of every DATA Ucode 0
            # (call 5). Keep the CPt training mask and drop Ucode 0 from CP.
            omit_fields = [cp_fields]
        if omit_fields:
            for fields in omit_fields:
                fields["tx_masks"] = _omit_ucode_zero(
                    fields.get("tx_masks"), field_name="tx_masks")
                fields["codec_masks"] = _omit_ucode_zero(
                    fields.get("codec_masks"), field_name="codec_masks")
        jd = self._phase3_jd_descriptor
        if not isinstance(jd, dict):
            self._failed = True
            self.events.append(("phase4-rate-rejected", {
                "reason": "missing CRC-valid Jd downstream rate mask",
                "jd_rate_mask": None,
                "profile_cap": self._phase4_downstream_drn_cap,
            }))
            return
        jd_rate_mask = jd.get("rate_mask")
        selection_cap = min(
            self._phase4_downstream_drn_cap,
            self._phase4_downstream_drn_pin or self._phase4_downstream_drn_cap)
        try:
            selection = select_downstream_drn_from_jd(
                jd_rate_mask, profile_cap=selection_cap)
        except (TypeError, ValueError) as exc:
            self._failed = True
            self.events.append(("phase4-rate-rejected", {
                "reason": f"no valid Jd/profile downstream rate: {exc}",
                "jd_rate_mask": jd_rate_mask,
                "profile_cap": selection_cap,
            }))
            return
        # phase4_cp_fields.drn remains a legacy ceiling for compatibility;
        # use phase4_downstream_drn_pin when an exact DRN is required.
        selection["configured_cp_drn_cap"] = (
            self._profile.get("phase4_cp_fields", {}).get("drn"))
        selection["pinned_drn"] = self._phase4_downstream_drn_pin
        if self._phase4_downstream_drn_pin is not None:
            pin = self._phase4_downstream_drn_pin
            if not isinstance(jd_rate_mask, int) or not jd_rate_mask & (1 << (pin - 1)):
                self._failed = True
                self.events.append(("phase4-rate-rejected", {
                    "reason": "configured pinned DRN is absent from Jd",
                    "pinned_drn": pin, "jd_rate_mask": jd_rate_mask,
                }))
                return
            if selection["drn"] != pin:
                self._failed = True
                self.events.append(("phase4-rate-rejected", {
                    "reason": "configured pinned DRN is not the selected DRN",
                    **selection,
                    "requested_pin_met": False,
                }))
                return
        selection["requested_pin_met"] = (
            self._phase4_downstream_drn_pin is None
            or selection["drn"] == self._phase4_downstream_drn_pin)
        cp_fields["drn"] = selection["drn"]
        self._phase4_downstream_selection = selection
        self._phase4_controller = V90AnaloguePhase4Controller(
            cpt_fields=cpt_fields, cp_fields=cp_fields, law=self.law,
            constellation=constellation, baud=self._phase3_baud,
            carrier=self._phase3_carrier, lapm_detect=self._phase4_lapm_detect,
            data_fault_hold_frames=self._phase4_data_fault_hold_frames)
        self._phase4_cpt_decisions = self._phase4_controller.start()
        self._phase4_start_at = int(ready_at)
        if self._phase4_rate_fallback_started_at is None:
            self._phase4_rate_fallback_started_at = int(ready_at)
        self._phase4_timeout_at = int(ready_at) + 20 * FS
        self.events.append(("phase4-start", {
            "at": int(ready_at), "constellation": constellation,
            "state": self._phase4_controller.state,
            "downstream_rate": dict(selection),
        }))

    def _start_phase4_media(self) -> None:
        if (self._phase4_media is not None or self._phase4_controller is None
                or self._phase4_start_at is None
                or self._tx_media < self._phase4_start_at):
            return
        if self._phase3_baud is None or self._phase3_carrier is None:
            raise RuntimeError("caller Phase 4 waveform lacks negotiated carrier")
        self._phase4_media = V90AnaloguePhase4MediaTx(
            self._phase4_controller, baud=self._phase3_baud,
            carrier=self._phase3_carrier, rms=self._phase3_rms)
        self._phase4_media.start(decisions=(
            list(self._phase4_cpt_decisions) + self._phase4_pending_decisions))
        self._phase4_pending_decisions = []
        self.events.append(("phase4-cpt-transmit", {
            "at": self._tx_media,
            "initial_ri_codewords": self._phase4_initial_octets,
        }))

    def _feed_phase4(self, codewords: bytes) -> None:
        phase4 = self._phase4_controller
        if phase4 is None:
            return
        self._phase4_initial_octets += len(codewords)
        if self._phase4_media is None:
            self._phase4_pending_decisions.extend(
                phase4.feed_downstream(codewords))
            self._start_phase4_media()
        else:
            self._phase4_media.feed_downstream(codewords)
        while self._reported_phase4_events < len(phase4.events):
            self.events.append(phase4.events[self._reported_phase4_events])
            self._reported_phase4_events += 1
        if phase4.state == phase4.FAILED:
            reason = str(getattr(phase4, "_failure_reason", ""))
            failure_code = getattr(phase4, "_failure_code", None)
            if self._try_phase4_rate_stepdown(failure_code, reason):
                return
            self._failed = True
        if phase4.lapm is not None and self._queued_dte:
            phase4.feed_bytes(bytes(self._queued_dte))
            self._queued_dte.clear()

    def _try_phase4_rate_stepdown(self, failure_code: str | None,
                                  reason: str = "") -> bool:
        """Walk down Jd-advertised DRNs after bounded B1d-only failures.

        V.90 does not renegotiate CP in place during startup. The bounded
        recovery uses analogue-initiated retrains (Tone A followed by fresh
        Phase 2/3/4) and lowers the Phase-4 cap before each exchange. Only a
        typed B1d qualification failure is eligible. The retry count and total
        elapsed media time are bounded; unrelated errors fail closed.
        """
        selection = self._phase4_downstream_selection
        if (self.connected or selection is None
                or failure_code != "b1d_not_all_ones"
                or self._phase4_downstream_drn_pin is not None):
            return False
        media_at = max(self._tx_media, self._renderer.cursor, self._rx_at)
        if self._phase4_rate_fallback_started_at is None:
            self._phase4_rate_fallback_started_at = media_at
        elapsed = max(0, media_at - self._phase4_rate_fallback_started_at)
        if (self._phase4_rate_fallback_attempts
                >= _PHASE4_RATE_FALLBACK_MAX_RETRIES):
            self.events.append(("phase4-rate-walkdown-exhausted", {
                "reason": "retry_limit",
                "attempts": self._phase4_rate_fallback_attempts,
                "budget_samples": _PHASE4_RATE_FALLBACK_BUDGET_SAMPLES,
                "elapsed_samples": elapsed,
                "last_failed_drn": selection.get("drn"),
            }))
            return False
        if elapsed >= _PHASE4_RATE_FALLBACK_BUDGET_SAMPLES:
            self.events.append(("phase4-rate-walkdown-exhausted", {
                "reason": "connect_time_budget",
                "attempts": self._phase4_rate_fallback_attempts,
                "budget_samples": _PHASE4_RATE_FALLBACK_BUDGET_SAMPLES,
                "elapsed_samples": elapsed,
                "last_failed_drn": selection.get("drn"),
            }))
            return False
        selected = int(selection.get("drn", 0))
        rate_mask = selection.get("jd_rate_mask")
        if not isinstance(rate_mask, int) or isinstance(rate_mask, bool):
            return False
        lower_candidates = [bit + 1 for bit in range(22)
                            if rate_mask & (1 << bit)
                            and bit + 1 < selected
                            and bit + 1 <= self._phase4_downstream_drn_cap]
        if (not lower_candidates
                or self._phase3_retrain_attempts
                > self._phase4_rate_fallback_attempts
                or not isinstance(self._controller.info0d, dict)
                or not self._controller.info0d):
            return False
        retry_cap = max(lower_candidates)
        self._phase4_rate_fallback_attempts += 1
        self._phase4_downstream_drn_cap = min(
            self._phase4_downstream_drn_cap, retry_cap)
        media_at = max(self._tx_media, self._renderer.cursor, self._rx_at)
        self.events.append(("phase4-rate-stepdown", {
            "attempt": self._phase4_rate_fallback_attempts,
            "retry_limit": _PHASE4_RATE_FALLBACK_MAX_RETRIES,
            "elapsed_samples": elapsed,
            "budget_samples": _PHASE4_RATE_FALLBACK_BUDGET_SAMPLES,
            "failed_drn": selected,
            "retry_cap_drn": self._phase4_downstream_drn_cap,
            "reason": reason,
            "recovery": "analogue-initiated V.90 retrain",
        }))
        self._start_phase3_retrain(
            media_at=media_at,
            reason=f"Phase-4 B1d failed at DRN {selected}; retry below it",
            rate_fallback=True)
        return not self._failed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._v34_fallback_engine is not None:
                close = getattr(self._v34_fallback_engine, "close", None)
                if callable(close):
                    close()


def build_v90_analogue_caller_engine(*, law: str, profile: dict,
                                     phase1_silence_samples: int = 0):
    """Create a V8Chain factory for an explicitly configured caller profile."""
    configured = dict(profile)

    def factory(*, role: str, negotiation: dict, media_at: int,
                media_origin: float):
        return V90AnalogueCallerPhase2Engine(
            role=role, negotiation=negotiation, media_at=media_at,
            media_origin=media_origin, law=law, profile=configured,
            phase1_silence_samples=phase1_silence_samples)

    return factory
