"""Explicit V.90 role factories for the production DCE.

The answerer profile advertises digital V.8 PCM and requires explicit INFO
fields, signal levels, and a downstream rate mask. The caller profile drives
the analogue V.90 path through Phase 4. Both roles remain explicitly
configured and disabled by default; neither setting changes generic V.34
automode behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

from v8 import V8_PCM_DIGITAL


_REQUIRED = {
    "info0d_fields", "info1d_fields", "tone_b_peak", "probe_rms",
    "phase3_rate_mask",
}

# The DCE creates the V.90 engine when V.8 negotiation resolves, at the start
# of Phase 1's terminal silence. The renderer preserves sample positions within
# each 20 ms RTP frame, so this is exactly 75 ms, within V.90's 75 +/- 5 ms.
# A completed V.92 Quick Connect exchange has already emitted its 75 ms
# post-TONEq silence before handing off to Phase 2; do not repeat it here.
V90_PHASE1_SILENCE_SAMPLES = 600

_CALL_ROLES = ("answer", "originate")


def v90_profile_call_roles(profile, default: str) -> tuple[str, ...]:
    """Return the call directions a V.90 profile serves.

    V.90 fixes Phase 2 onward by digital/analogue role, not by who dialled
    (§9.1.1, §9.2), so either profile may serve either direction when its
    ``call_roles`` lists it. Without the key each keeps its usual direction.
    """
    if not isinstance(profile, Mapping):
        return (default,)
    roles = profile.get("call_roles", [default])
    if (isinstance(roles, (str, bytes)) or not isinstance(roles, (list, tuple))
            or not roles or len(set(roles)) != len(roles)
            or any(role not in _CALL_ROLES for role in roles)):
        raise ValueError("V.90 profile call_roles must list answer and/or originate")
    return tuple(roles)


def install_v90_digital_answerer(chain_options: dict, *, tx_role: str,
                                 law: str, v42bis: bool = False,
                                 compression_recovery: bool = False) -> bool:
    """Consume an explicit ``v90_digital_answerer`` chain profile if present.

    Returns False without changing the options when no profile is configured.
    A configured profile sets V.8 digital PCM availability and supplies the
    V.90 engine factory. Existing PCM configuration cannot be silently
    overwritten.
    """
    profile = chain_options.pop("v90_digital_answerer", None)
    if profile is None:
        return False
    if tx_role not in v90_profile_call_roles(profile, "answer"):
        raise ValueError(
            f"V.90 digital-answerer profile call_roles do not include the {tx_role} role")
    if law not in ("pcmu", "pcma"):
        raise ValueError("V.90 digital-answerer profile requires negotiated G.711")
    if "pcm_availability" in chain_options and chain_options["pcm_availability"] not in (0, False):
        raise ValueError("V.90 profile cannot override existing V.8 PCM availability")
    if not isinstance(profile, Mapping):
        raise TypeError("v90_digital_answerer must be a profile mapping")
    missing = _REQUIRED - set(profile)
    if missing:
        raise ValueError("V.90 digital-answerer profile missing: "
                         + ", ".join(sorted(missing)))
    for name in ("info0d_fields", "info1d_fields"):
        if not isinstance(profile[name], Mapping):
            raise TypeError(f"V.90 profile {name} must be a mapping")
    for name in ("tone_b_peak", "probe_rms"):
        value = float(profile[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"V.90 profile {name} must be finite and positive")

    configured = deepcopy(dict(profile))
    configured.pop("call_roles", None)
    phase4_mp_profile = configured.get("phase4_mp_profile")
    if phase4_mp_profile is not None:
        if configured.get("v92_pcm_upstream", False):
            raise ValueError("phase4_mp_profile is unavailable with V.92 PCM upstream")
        from v90_mp_policy import validate_answerer_mp_profile
        phase4_mp_profile = validate_answerer_mp_profile(phase4_mp_profile)
    rate_mask = configured["phase3_rate_mask"]
    if isinstance(rate_mask, bool) or not isinstance(rate_mask, int):
        raise ValueError("V.90 profile phase3_rate_mask must be an integer")
    from v90_digital_tx import build_jd_bits
    build_jd_bits(rate_mask,
                  phase3_16point=bool(configured.get("phase3_16point", False)))
    span = configured.get("phase3_span", 12)
    if span not in (12, 24):
        raise ValueError("V.90 profile phase3_span must be 12 or 24")
    enable_lapm = configured.get("phase3_enable_lapm", False)
    if type(enable_lapm) is not bool:
        raise ValueError("V.90 profile phase3_enable_lapm must be boolean")
    v92 = configured.get("v92", False)
    if type(v92) is not bool:
        raise ValueError("V.90 profile v92 must be boolean")
    v92_pcm = configured.get("v92_pcm_upstream", False)
    if type(v92_pcm) is not bool:
        raise ValueError("V.90 profile v92_pcm_upstream must be boolean")
    if v92_pcm and not v92:
        raise ValueError("v92_pcm_upstream needs v92 to be true")
    v92_quick_connect = configured.get("v92_quick_connect", False)
    if type(v92_quick_connect) is not bool:
        raise ValueError("V.90 profile v92_quick_connect must be boolean")
    if v92_quick_connect and not (v92 and v92_pcm):
        raise ValueError("v92_quick_connect needs V.92 PCM upstream")
    quick_connect_level_code = configured.get(
        "v92_quick_connect_level_code", 0b01)
    if (isinstance(quick_connect_level_code, bool)
            or not isinstance(quick_connect_level_code, int)
            or not 0 <= quick_connect_level_code <= 3):
        raise ValueError(
            "V.90 profile v92_quick_connect_level_code must be an integer in 0..3")
    # Quick Connect is an analogue caller's request; a digital caller skips it.
    chain_options["v92_quick_connect"] = v92_quick_connect and tx_role == "answer"
    chain_options["v92_quick_connect_law"] = law
    chain_options["v92_quick_connect_level_code"] = quick_connect_level_code
    fast_parameter_exchange = configured.get("v92_fast_parameter_exchange", False)
    if type(fast_parameter_exchange) is not bool:
        raise ValueError("V.90 profile v92_fast_parameter_exchange must be boolean")
    if fast_parameter_exchange and (not v92_pcm or not enable_lapm):
        raise ValueError("v92_fast_parameter_exchange needs V.92 PCM upstream and LAPM")
    max_lost_frames = configured.get("v92_pcm_upstream_max_lost_frames", 3)
    if (isinstance(max_lost_frames, bool) or not isinstance(max_lost_frames, int)
            or not 0 <= max_lost_frames <= 1024):
        raise ValueError("V.90 profile v92_pcm_upstream_max_lost_frames must be an integer in 0..1024")
    if max_lost_frames != 3 and not v92_pcm:
        raise ValueError("v92_pcm_upstream_max_lost_frames needs v92_pcm_upstream to be true")
    v92_eq = {}
    for key in ("v92_equalise", "v92_phase3_equalise", "v92_fractional_rx_timing",
                "v92_fractional_rx_reacquisition"):
        default = (configured.get("v92_fractional_rx_timing", False)
                   if key == "v92_fractional_rx_reacquisition" else False)
        value = configured.get(key, default)
        if type(value) is not bool:
            raise ValueError(f"V.90 profile {key} must be boolean")
        if value and not v92_pcm:
            raise ValueError(f"{key} needs v92_pcm_upstream to be true")
        v92_eq[key[4:]] = value
    if (v92_eq["fractional_rx_reacquisition"]
            and not v92_eq["fractional_rx_timing"]):
        raise ValueError(
            "v92_fractional_rx_reacquisition needs v92_fractional_rx_timing")
    if type(v42bis) is not bool or type(compression_recovery) is not bool:
        raise ValueError("V.42bis and compression-recovery settings must be boolean")
    if compression_recovery and not v42bis:
        raise ValueError("V.42bis compression recovery requires V.42bis to be enabled")
    if v42bis and v92_pcm and not enable_lapm:
        raise ValueError("V.42bis requires phase3_enable_lapm")
    if v92:
        # V.92 capability (INFO0d bit 27) and PCM-upstream permission
        # (INFO1d bit 70) are separate offers. Default to the V.90 path.
        info0 = dict(configured["info0d_fields"])
        info0["reserved"] = int(info0.get("reserved", 0)) | 2
        # For the V.92 digital answerer, limit the analog-side modem to the
        # measured codec-output level. Explicit profile values remain the
        # operator's choice. The V.90-only path keeps its prior profile values.
        info0.setdefault("max_power", 23)
        info0.setdefault("power_at_codec", 1)
        configured["info0d_fields"] = info0
        blocks = [dict(block) for block in
                  configured["info1d_fields"].get("blocks", [{}] * 6)]
        if len(blocks) != 6:
            raise ValueError("V.90 profile info1d_fields needs six rate blocks")
        if int(blocks[5].get("hi_carrier", 0)) and not v92_pcm:
            raise ValueError("a V.92 profile cannot set INFO1d bit 70: it would offer PCM upstream")
        if v92_pcm:
            blocks[5]["hi_carrier"] = 1
        info1 = dict(configured["info1d_fields"])
        info1["blocks"] = blocks
        configured["info1d_fields"] = info1
    upstream_mu = configured.get("phase3_upstream_equalizer_mu")
    if upstream_mu is not None and (
            isinstance(upstream_mu, bool) or not isinstance(upstream_mu, (int, float))
            or not 0.0 < float(upstream_mu) <= 1.0):
        raise ValueError(
            "V.90 profile phase3_upstream_equalizer_mu must be in (0, 1]")
    l2_tolerance = configured.get("phase2_l2_echo_tolerance_samples", 0)
    if (isinstance(l2_tolerance, bool) or not isinstance(l2_tolerance, int)
            or not 0 <= l2_tolerance <= 80):
        raise ValueError(
            "V.90 profile phase2_l2_echo_tolerance_samples must be an integer in 0..80")
    l2_echo_policy = configured.get(
        "phase2_l2_echo_window_policy", "require")
    if l2_echo_policy not in ("require", "advance-after-window", "on-tone-a"):
        raise ValueError(
            "V.90 profile phase2_l2_echo_window_policy must be "
            "'require', 'advance-after-window' or 'on-tone-a'")
    info1a_timeout_retrain = configured.get(
        "phase2_info1a_timeout_retrain", False)
    if not isinstance(info1a_timeout_retrain, bool):
        raise ValueError(
            "V.90 profile phase2_info1a_timeout_retrain must be boolean")
    project_info1d_rates = configured.get(
        "phase2_project_info1d_rates_from_probe", True)
    if type(project_info1d_rates) is not bool:
        raise ValueError(
            "V.90 profile phase2_project_info1d_rates_from_probe must be boolean")
    # V.92 PCM-upstream calls run tens of seconds of dense data; ~40-sample arrival-delay steps bent single clock fits
    # past the envelope and killed them (arm 12/12d), so they default to the finer step detector and 3 epochs.
    clock_offset_step_samples = configured.get(
        "phase2_clock_offset_step_samples", 40 if v92_pcm else 80)
    if (isinstance(clock_offset_step_samples, bool)
            or not isinstance(clock_offset_step_samples, int)
            or not 1 <= clock_offset_step_samples <= 8000):
        raise ValueError(
            "V.90 profile phase2_clock_offset_step_samples must be an integer in 1..8000")
    clock_violation_epochs = configured.get("phase2_clock_violation_epochs", 3 if v92_pcm else 1)
    # PCM-upstream receiver and CPd constellation; unset keeps the session defaults (and their env overrides),
    # {"v92_b1u_rx_mode": "trn2u-fit-viterbi", "v92_cpd_constellation": "adaptive"}: the pair validated
    # against a real analogue modem (SmartLink) in lab/V92_SMARTLINK_DATA_2026-11-17.md.
    b1u_rx_mode = configured.get("v92_b1u_rx_mode")
    if b1u_rx_mode is not None and b1u_rx_mode not in ("off", "trn2u-fit-viterbi"):
        raise ValueError("V.90 profile v92_b1u_rx_mode must be off or trn2u-fit-viterbi")
    cpd_constellation = configured.get("v92_cpd_constellation")
    if cpd_constellation is not None and cpd_constellation not in ("default", "sparse", "expanded10", "adaptive"):
        raise ValueError("V.90 profile v92_cpd_constellation must be default, sparse, expanded10 or adaptive")
    if (b1u_rx_mode is not None or cpd_constellation is not None) and not v92_pcm:
        raise ValueError("V.90 profile v92_b1u_rx_mode/v92_cpd_constellation require v92_pcm_upstream")
    if (isinstance(clock_violation_epochs, bool)
            or not isinstance(clock_violation_epochs, int)
            or not 1 <= clock_violation_epochs <= 16):
        raise ValueError(
            "V.90 profile phase2_clock_violation_epochs must be an integer in 1..16")
    retrain_on_failure = configured.get("retrain_on_failure", False)
    if type(retrain_on_failure) is not bool:
        raise ValueError("V.90 profile retrain_on_failure must be boolean")
    max_retrains = configured.get("max_retrains", 2)
    if isinstance(max_retrains, bool) or not isinstance(max_retrains, int) \
            or not 1 <= max_retrains <= 5:
        raise ValueError("V.90 profile max_retrains must be an integer in 1..5")
    from v90_digital_phase2 import V90DigitalPhase2Signals
    # V8Chain calls the returned engine factory from its RTP receive path.
    # Load the comparatively large adapter graph now, while the opted-in
    # OnlineAudioSession is being constructed and before media threads start.
    from v90_digital_phase2_engine import V90DigitalAnswererPhase2Engine
    from v90_digital_phase2_rx import warm_probe_reference_bank
    info0 = dict(configured["info0d_fields"])
    expected_alaw = int(law == "pcma")
    if "alaw" in info0 and int(info0["alaw"]) != expected_alaw:
        raise ValueError("V.90 profile INFO0d law does not match the RTP law")
    info0["alaw"] = expected_alaw
    V90DigitalPhase2Signals(info0d=info0,
                            info1d=dict(configured["info1d_fields"]), v92=v92)
    # The first L2 frequency/quality analysis needs references for 29 tones.
    # Build that bounded cache during profile loading rather than an RTP
    # receive callback, where its cold cost can consume the 20 ms media budget.
    warm_probe_reference_bank()

    def factory(*, role: str, negotiation: dict, media_at: int,
                media_origin: float):
        from v8 import V8_PROTOCOL_LAPM_V42

        negotiated_lapm = bool(int(negotiation.get("protocols", 0)) & V8_PROTOCOL_LAPM_V42)
        enable_negotiated_lapm = enable_lapm and negotiated_lapm
        phase1_silence_samples = (
            0 if negotiation.get("quick_connect") is True
            else V90_PHASE1_SILENCE_SAMPLES)

        return V90DigitalAnswererPhase2Engine(
            role=role, negotiation=negotiation,
            media_at=media_at, media_origin=media_origin, law=law,
            info0d_fields=dict(configured["info0d_fields"]),
            info1d_fields=dict(configured["info1d_fields"]),
            tone_b_peak=float(configured["tone_b_peak"]),
            probe_rms=float(configured["probe_rms"]),
            phase3_rate_mask=configured["phase3_rate_mask"],    # with V.92 PCM: the V.90-only peer's Phase 3
            phase3_16point=bool(configured.get("phase3_16point", False)),
            phase3_span=int(configured.get("phase3_span", 12)),
            phase4_mp_profile=phase4_mp_profile,
            phase3_upstream_equalizer_mu=(
                None if upstream_mu is None else float(upstream_mu)),
            phase2_l2_echo_tolerance_samples=l2_tolerance,
            phase2_l2_echo_window_policy=l2_echo_policy,
            phase2_info1a_timeout_retrain=info1a_timeout_retrain,
            phase2_project_info1d_rates_from_probe=project_info1d_rates,
            phase2_clock_offset_step_samples=clock_offset_step_samples,
            phase2_clock_violation_epochs=clock_violation_epochs,
            retrain_on_failure=retrain_on_failure, max_retrains=max_retrains,
            phase1_silence_samples=phase1_silence_samples,
            phase3_enable_lapm=enable_lapm, v92=v92,
            phase3_factory=(None if not v92_pcm else _v92_session_factory(
                law, enable_negotiated_lapm, v92_eq, v42bis,
                compression_recovery, max_lost_frames,
                fast_parameter_exchange, b1u_rx_mode, cpd_constellation)))

    def _v92_session_factory(law_name, lapm, equalisers, v42bis_enabled,
                             recover_compression, max_lost_frames,
                             fast_parameter_exchange, b1u_rx_mode=None, cpd_constellation=None):
        from v92_digital_session import factory as v92_factory
        return v92_factory(
            law=law_name, enable_lapm=lapm, async_design=True, **equalisers,
            v42bis_compression=(3, 1024, 8) if v42bis_enabled else None,
            v42_compression_recovery=recover_compression,
            enable_rate_renegotiation_response=True,
            enable_fast_parameter_exchange_response=fast_parameter_exchange,
            pcm_upstream_max_lost_frames=max_lost_frames,
            b1u_rx_mode=b1u_rx_mode, cpd_constellation_mode=cpd_constellation)

    chain_options["pcm_availability"] = V8_PCM_DIGITAL
    chain_options["v90_engine_factory"] = factory
    return True


def install_v90_analogue_caller(chain_options: dict, *, tx_role: str,
                                law: str) -> bool:
    """Install the explicitly configured local analogue-caller profile.

    It is kept separate from the digital-answerer profile and disabled unless
    the caller supplies the profile explicitly.
    """
    from v8 import V8_PCM_ANALOG

    profile = chain_options.pop("v90_analogue_caller", None)
    if profile is None:
        return False
    if tx_role not in v90_profile_call_roles(profile, "originate"):
        # A process may accept inbound calls as well as originate. Unless the
        # profile lists the answer role, leave the answer-role V.8
        # advertisement untouched for those sessions.
        return False
    if law not in ("pcmu", "pcma"):
        raise ValueError("V.90 analogue-caller profile requires negotiated G.711")
    if "pcm_availability" in chain_options and chain_options["pcm_availability"] not in (0, False):
        raise ValueError("V.90 caller profile cannot override existing V.8 PCM availability")
    if not isinstance(profile, Mapping):
        raise TypeError("v90_analogue_caller must be a profile mapping")
    from v90_analogue_caller_engine import build_v90_analogue_caller_engine
    from v90_info_reference import build_info1a_v90

    required = {"info0a_fields", "info1a_fields", "nominal_peak", "probe_rms"}
    missing = required - set(profile)
    if missing:
        raise ValueError("V.90 analogue-caller profile missing: "
                         + ", ".join(sorted(missing)))
    if not isinstance(profile["info1a_fields"], Mapping):
        raise TypeError("V.90 analogue-caller info1a_fields must be a mapping")
    configured = deepcopy(dict(profile))
    configured.pop("call_roles", None)
    v92_pcm_upstream = configured.get("v92_pcm_upstream", False)
    if type(v92_pcm_upstream) is not bool:
        raise ValueError("V.90 analogue-caller v92_pcm_upstream must be boolean")
    if v92_pcm_upstream:
        from v92_info import build_info1a_pcm

        fields = configured.get("v92_info1a_fields", {})
        if not isinstance(fields, Mapping):
            raise TypeError("v92_info1a_fields must be a mapping")
        # Validate all Table 18 fields now, before any call is placed.
        frame = build_info1a_pcm(**dict(fields))
        if len(frame) != 70:
            raise ValueError("V.92 PCM INFO1a profile did not build a 70-bit frame")
        info0a = dict(configured["info0a_fields"])
        info0a["v92"] = True
        configured["info0a_fields"] = info0a
        configured["v92_info1a_fields"] = dict(fields)
        fpe = configured.get("v92_fast_parameter_exchange", False)
        if type(fpe) is not bool:
            raise ValueError("v92_fast_parameter_exchange must be boolean")
        ucodes = configured.get("v92_dil_ucodes", ())
        if (not isinstance(ucodes, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or not 0 <= value <= 127 for value in ucodes)):
            raise ValueError("v92_dil_ucodes must be a sequence of integers in 0..127")
        recovery = configured.get("v92_v42_compression_recovery", False)
        if type(recovery) is not bool:
            raise ValueError("v92_v42_compression_recovery must be boolean")
    late_l2_echo_policy = profile.get("phase2_late_l2_echo_policy", "fail")
    if late_l2_echo_policy not in (
            "fail", "bounded", "continue", "advance-after-window"):
        raise ValueError(
            "V.90 analogue-caller phase2_late_l2_echo_policy must be "
            "'fail', 'bounded', 'continue' or 'advance-after-window'")
    # The digital peer's V.90 §9.2.1.2.4 recovery is recognizable as a
    # Tone-B reversal followed by the receive-side L1/L2 sequence. Captured
    # traces and paired payload tests validate this route. Keep an explicit
    # profile override for diagnostics and deployments that need to disable it.
    early_recovery = profile.get("phase2_accept_early_peer_recovery", True)
    if not isinstance(early_recovery, bool):
        raise ValueError(
            "V.90 analogue-caller phase2_accept_early_peer_recovery must be boolean")
    timeout_retrain = profile.get("phase2_info1d_timeout_retrain", False)
    if not isinstance(timeout_retrain, bool):
        raise ValueError(
            "V.90 analogue-caller phase2_info1d_timeout_retrain must be boolean")
    # Fail at DCE startup for malformed frame settings, not after a call has
    # already negotiated the analogue PCM role.
    frame = build_info1a_v90(mode=6, **dict(configured["info1a_fields"]))
    if len(frame) != 70:
        raise ValueError("V.90 analogue-caller INFO1a profile did not build a 70-bit frame")

    chain_options["pcm_availability"] = V8_PCM_ANALOG
    chain_options["v90_engine_factory"] = build_v90_analogue_caller_engine(
        law=law, profile=configured,
        phase1_silence_samples=V90_PHASE1_SILENCE_SAMPLES)
    return True
