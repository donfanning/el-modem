"""Build a conservative V.90 MP from El-Modem's V.34 receive support."""
from __future__ import annotations

from collections.abc import Mapping
import math

from v34_negotiation import DirectionParameters, supported
from v90_mp import build_mp, parse_mp


_V90_DOWNSTREAM_STEPS = 22
_V90_DOWNSTREAM_STEP_BPS = 8000.0 / 6.0


def v90_downstream_rate(drn: int) -> float:
    """Return the CP downstream signalling rate for a legal DRN.

    In CP, DRN 1 is 28 kbit/s and DRN 22 is 56 kbit/s, in 8/6 kbit/s
    increments. DRN zero is cleardown and is intentionally rejected here.
    """
    if isinstance(drn, bool) or not isinstance(drn, int) or not 1 <= drn <= 22:
        raise ValueError("V.90 CP DRN must be in 1..22 (zero is cleardown)")
    return (drn + 20) * _V90_DOWNSTREAM_STEP_BPS


def select_downstream_drn_from_jd(jd_rate_mask: int, *,
                                  profile_cap: int = 22) -> dict:
    """Select the highest CP DRN allowed by the peer's CRC-valid Jd and cap.

    This is the production one-level-DIL policy. Jd is the peer's explicit
    downstream rate capability; the profile value limits our local offer.
    The existing one-level DIL does not provide a calibrated rate-margin
    estimate, so selection must not pretend that it does. An explicit B1d
    qualification failure may trigger the separately bounded one-shot retry.
    """
    if (isinstance(jd_rate_mask, bool) or not isinstance(jd_rate_mask, int)
            or not 0 < jd_rate_mask < (1 << _V90_DOWNSTREAM_STEPS)):
        raise ValueError("Jd downstream rate mask must enable one of 22 rates")
    if (isinstance(profile_cap, bool) or not isinstance(profile_cap, int)
            or not 1 <= profile_cap <= 22):
        raise ValueError("V.90 downstream profile cap must be in 1..22")

    advertised = [bit + 1 for bit in range(_V90_DOWNSTREAM_STEPS)
                  if jd_rate_mask & (1 << bit)]
    candidates = [drn for drn in advertised if drn <= profile_cap]
    if not candidates:
        raise ValueError("Jd and local profile have no common downstream DRN")
    selected = max(candidates)
    return {
        "drn": selected,
        "rate_bps": int(round(v90_downstream_rate(selected))),
        "advertised_max_drn": max(advertised),
        "profile_cap": profile_cap,
        "jd_rate_mask": jd_rate_mask,
        "selection": "jd-mask-and-profile-cap",
    }


def answerer_mp_profile(baud: float) -> dict:
    """Return the Type 0 MP capability currently implemented by our receiver.

    V.90 MP rates use bit 0 for 4800 bit/s through bit 12 for 33600 bit/s.
    The MP trellis selector describes the analogue transmitter which we will
    receive, so this profile advertises only the 16-state path currently
    accepted by ``v34_receive.initialize``. Nonlinear and expanded shaping
    are kept off, matching the production V.34 receive defaults; Type 0 makes
    the zero precoder explicit by omitting Type 1 coefficients.

    This is a capability-derived offer, not a per-line qualification result.
    Rates are limited by the same production geometry check used for V.34
    receive profile selection at the negotiated upstream baud.
    """
    baud = float(baud)
    if baud <= 0:
        raise ValueError("V.90 upstream baud must be positive")
    rates = [rate for rate in range(4800, 33601, 2400)
             if supported(rate, baud, shaping=False)]
    if not rates:
        raise ValueError("V.34 receiver supports no V.90 upstream rate at this baud")
    mask = sum(1 << (rate // 2400 - 2) for rate in rates)
    return {
        "mp_type": 0,
        "max_a2d_drn": max(rates) // 2400,
        "trellis": 0,
        "nonlinear": 0,
        "shaping": 0,
        "rate_mask": mask,
    }


def validate_answerer_mp_profile(profile: Mapping) -> dict:
    """Validate an explicit digital-answerer MP profile by build/parse roundtrip.

    This is intentionally separate from :func:`answerer_mp_profile`: callers
    must opt in to non-default Type 1 offers, while the conservative Type 0
    capability remains unchanged.
    """
    if not isinstance(profile, Mapping):
        raise TypeError("V.90 phase4_mp_profile must be a mapping")
    allowed = {"mp_type", "max_a2d_drn", "trellis", "nonlinear",
               "shaping", "rate_mask", "precoder"}
    unknown = set(profile) - allowed
    if unknown:
        raise ValueError("unknown V.90 phase4_mp_profile fields: "
                         + ", ".join(sorted(unknown)))
    required = {"mp_type", "max_a2d_drn", "trellis", "nonlinear",
                "shaping", "rate_mask"}
    missing = required - set(profile)
    if missing:
        raise ValueError("V.90 phase4_mp_profile missing: "
                         + ", ".join(sorted(missing)))
    mp_type = profile["mp_type"]
    if type(mp_type) is not int or mp_type not in (0, 1):
        raise ValueError("V.90 phase4_mp_profile mp_type must be 0 or 1")
    if type(profile["trellis"]) is not int or profile["trellis"] != 0:
        raise ValueError(
            "V.90 digital answerer currently supports only the 16-state trellis")
    if mp_type == 1 and "precoder" not in profile:
        raise ValueError("V.90 Type 1 phase4_mp_profile requires precoder")
    if mp_type == 0 and "precoder" in profile:
        raise ValueError("V.90 Type 0 phase4_mp_profile cannot set precoder")
    values = dict(profile)
    if "precoder" in values:
        values["precoder"] = tuple(values["precoder"])
    try:
        parsed = parse_mp(build_mp(**values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid V.90 phase4_mp_profile: {exc}") from exc
    if parsed is None:
        raise ValueError("V.90 phase4_mp_profile failed MP roundtrip validation")
    result = {
        "mp_type": parsed["type"],
        "max_a2d_drn": parsed["max_a2d_drn"],
        "trellis": parsed["trellis"],
        "nonlinear": parsed["nonlinear"],
        "shaping": parsed["shaping"],
        "rate_mask": parsed["rate_mask"],
    }
    if parsed["type"] == 1:
        result["precoder"] = tuple(parsed["precoder"])
    return result


def select_analog_upstream_rate(cp_profile: dict, mp_profile: dict) -> int:
    """Select the normative analogue-to-digital V.90 data rate.

    CP advertises rates enabled in the analogue transmitter. MP advertises
    digital-receiver capabilities and gives the maximum A-to-D DRN. The
    selected rate is the highest rate in both masks no greater than that MP
    ceiling, as specified by V.90 §9.4.2.4. Both masks encode bit 0 as
    4800 bit/s and bit 12 as 33600 bit/s.
    """
    if not isinstance(cp_profile, dict) or cp_profile.get("kind") != "cp":
        raise ValueError("analogue upstream rate requires a selected CP profile")
    if cp_profile.get("cleardown") or cp_profile.get("rate") is None:
        raise ValueError("CP profile requests cleardown")
    value = cp_profile.get("analog_rate_mask")
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 0 <= value < (1 << 13)):
        raise ValueError("CP analog_rate_mask must be a 13-bit rate mask")
    if not isinstance(mp_profile, dict):
        raise ValueError("MP profile must be a parsed capability dictionary")
    mask = mp_profile.get("rate_mask")
    maximum_drn = mp_profile.get("max_a2d_drn")
    if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < (1 << 13):
        raise ValueError("MP rate_mask must be a 13-bit rate mask")
    if (isinstance(maximum_drn, bool) or not isinstance(maximum_drn, int)
            or maximum_drn not in range(2, 15)):
        raise ValueError("MP max_a2d_drn must be in 2..14")

    common = value & mask
    ceiling_drn = maximum_drn
    for bit in range(12, -1, -1):
        drn = bit + 2
        if common & (1 << bit) and drn <= ceiling_drn:
            return drn * 2400
    raise ValueError("CP and MP have no common analogue-upstream data rate")


def select_analog_upstream_profile(cp_profile: dict, mp_profile: dict,
                                   *, baud: float, carrier: float
                                   ) -> DirectionParameters:
    """Resolve the selected V.90 analogue upstream into a V.34 RX profile.

    The MP parameters describe the digital answerer's receiver request for
    the remote analogue transmitter. The V.90 CP/MP rate-mask intersection
    chooses the rate; the Phase 3 symbol rate and carrier remain those
    acquired from INFO. As in V.34, the transmitter takes the requested
    shaping, nonlinear encoding and (MP Type 1) precoder coefficients. Real
    digital modems send Type 1: captured peer requested precoding, nonlinear
    encoding and shaping on its first native V.90 call (2026-11-10).
    """
    if not isinstance(mp_profile, dict):
        raise ValueError("MP profile must be a parsed capability dictionary")
    mp_type = mp_profile.get("mp_type")
    if mp_type not in (0, 1):
        raise ValueError("V.90 MP type must be 0 or 1")
    if mp_profile.get("trellis") != 0:
        raise ValueError("only the 16-state V.90 upstream trellis is supported")
    flags = {}
    for name in ("nonlinear", "shaping"):
        value = mp_profile.get(name, 0)
        if value not in (0, 1, False, True):
            raise ValueError(f"V.90 MP {name} must be 0 or 1")
        flags[name] = bool(value)
    precoder = (0, 0, 0, 0, 0, 0)
    if mp_type == 1:
        precoder = tuple(mp_profile.get("precoder") or ())
        if (len(precoder) != 6 or any(
                isinstance(value, bool) or not isinstance(value, int)
                or not -32768 <= value <= 32767 for value in precoder)):
            raise ValueError("V.90 MP Type 1 requires six signed 16-bit precoder coefficients")
    if (not isinstance(baud, (int, float)) or isinstance(baud, bool)
            or not math.isfinite(baud) or baud <= 0):
        raise ValueError("V.90 upstream baud must be positive")
    if (not isinstance(carrier, (int, float)) or isinstance(carrier, bool)
            or not math.isfinite(carrier) or carrier <= 0):
        raise ValueError("V.90 upstream carrier must be positive")
    rate = select_analog_upstream_rate(cp_profile, mp_profile)
    if not supported(rate, float(baud), shaping=flags["shaping"]):
        raise ValueError("selected V.90 upstream rate is unsupported at this baud")
    return DirectionParameters(
        rate=rate, baud=float(baud), carrier=float(carrier), trellis=0,
        shaping=flags["shaping"], nonlinear=flags["nonlinear"],
        precoder=precoder)
