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
V90_PHASE1_SILENCE_SAMPLES = 600


def install_v90_digital_answerer(chain_options: dict, *, tx_role: str,
                                 law: str) -> bool:
    """Consume an explicit ``v90_digital_answerer`` chain profile if present.

    Returns False without changing the options when no profile is configured.
    A configured profile sets V.8 digital PCM availability and supplies the
    V.90 engine factory. Existing PCM configuration cannot be silently
    overwritten.
    """
    profile = chain_options.pop("v90_digital_answerer", None)
    if profile is None:
        return False
    if tx_role != "answer":
        raise ValueError("V.90 digital-answerer profile requires answer role")
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
    v92_options = {key for key in configured if key.startswith("v92")}
    if v92_options:
        raise ValueError("V.92 options are outside this V.90 profile")
    phase4_mp_profile = configured.get("phase4_mp_profile")
    if phase4_mp_profile is not None:
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
    clock_offset_step_samples = configured.get(
        "phase2_clock_offset_step_samples", 80)
    if (isinstance(clock_offset_step_samples, bool)
            or not isinstance(clock_offset_step_samples, int)
            or not 1 <= clock_offset_step_samples <= 8000):
        raise ValueError(
            "V.90 profile phase2_clock_offset_step_samples must be an integer in 1..8000")
    retrain_on_failure = configured.get("retrain_on_failure", False)
    if type(retrain_on_failure) is not bool:
        raise ValueError("V.90 profile retrain_on_failure must be boolean")
    max_retrains = configured.get("max_retrains", 2)
    if isinstance(max_retrains, bool) or not isinstance(max_retrains, int) \
            or not 1 <= max_retrains <= 5:
        raise ValueError("V.90 profile max_retrains must be an integer in 1..5")
    from v90_digital_phase2 import V90DigitalPhase2Signals
    from v90_digital_phase2_rx import warm_probe_reference_bank
    info0 = dict(configured["info0d_fields"])
    expected_alaw = int(law == "pcma")
    if "alaw" in info0 and int(info0["alaw"]) != expected_alaw:
        raise ValueError("V.90 profile INFO0d law does not match the RTP law")
    info0["alaw"] = expected_alaw
    V90DigitalPhase2Signals(info0d=info0,
                            info1d=dict(configured["info1d_fields"]))
    # The first L2 frequency/quality analysis needs references for 29 tones.
    # Build that bounded cache during profile loading rather than an RTP
    # receive callback, where its cold cost can consume the 20 ms media budget.
    warm_probe_reference_bank()

    def factory(*, role: str, negotiation: dict, media_at: int,
                media_origin: float):
        from v90_digital_phase2_engine import V90DigitalAnswererPhase2Engine

        return V90DigitalAnswererPhase2Engine(
            role=role, negotiation=negotiation,
            media_at=media_at, media_origin=media_origin, law=law,
            info0d_fields=dict(configured["info0d_fields"]),
            info1d_fields=dict(configured["info1d_fields"]),
            tone_b_peak=float(configured["tone_b_peak"]),
            probe_rms=float(configured["probe_rms"]),
            phase3_rate_mask=configured["phase3_rate_mask"],
            phase3_16point=bool(configured.get("phase3_16point", False)),
            phase3_span=int(configured.get("phase3_span", 12)),
            phase4_mp_profile=phase4_mp_profile,
            phase3_upstream_equalizer_mu=(
                None if upstream_mu is None else float(upstream_mu)),
            phase2_l2_echo_tolerance_samples=l2_tolerance,
            phase2_info1a_timeout_retrain=info1a_timeout_retrain,
            phase2_project_info1d_rates_from_probe=project_info1d_rates,
            phase2_clock_offset_step_samples=clock_offset_step_samples,
            retrain_on_failure=retrain_on_failure, max_retrains=max_retrains,
            phase1_silence_samples=V90_PHASE1_SILENCE_SAMPLES,
            phase3_enable_lapm=enable_lapm)

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
    if tx_role != "originate":
        # A process may accept inbound calls as well as originate. This
        # capability is exclusively a caller offer, so leave the answer-role
        # V.8 advertisement untouched for those sessions.
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
    frame = build_info1a_v90(mode=6, **dict(profile["info1a_fields"]))
    if len(frame) != 70:
        raise ValueError("V.90 analogue-caller INFO1a profile did not build a 70-bit frame")

    chain_options["pcm_availability"] = V8_PCM_ANALOG
    chain_options["v90_engine_factory"] = build_v90_analogue_caller_engine(
        law=law, profile=deepcopy(dict(profile)),
        phase1_silence_samples=V90_PHASE1_SILENCE_SAMPLES)
    return True
