"""Symbol-level paired V.92 PCM-upstream session (an idealised model, not a modem).

Runs the content of the start-up between an analogue and a digital modem through the
modules built from the V.92 text, with the detectors, timing and RTP left out:

  Phase 2   INFO0d/INFO0a V.92 bits; INFO1d advertising PCM upstream; INFO1a (Table 18)
  Phase 3   Ru/MD/TRN1u from the analogue modem -> the digital modem estimates the channel;
            Jd, Jp, Jp' (Tables 21-22) and the CPt/E1u hand-over (Table 23)
  Phase 4   rate and constellation selection -> CPd (Table 30), precoder/prefilter
            design, B1u (48 data frames of scrambled ones), then payload frames

The "channel" is a causal symbol-spaced impulse response followed by a G.711 A/D at the
digital modem. Nothing here proves interoperability with a real modem or detector
robustness; it proves that the pieces compose and that the numbers each side derives
from the other's messages are consistent. Every message crosses as bits and is parsed.
"""

from __future__ import annotations

import numpy as np

import v34_datarx
import v34_g711 as g711
import v34_tx
import v92_channel_estimate as estimate
import v92_cp as cp
import v92_info as info
import v92_modulation_design as modulation
import v92_pcm_upstream as pcm
import v92_phase3 as phase3
import v92_precoder_design as precoder
import v90_info_reference as v90

TRN1U_SYMBOLS = 16320               # >= 2040; a longer TRN1u averages the estimate down
B1U_FRAMES = 48
DEFAULT_LU = 3000.0
ESTIMATE_TAPS = 64
PREFILTER_TAPS = 256                 # within INFO1a: Lmax 256, Ltot 384
POSTCURSORS = 48
SAFETY_LADDER = (1.0, 1.6, 2.5, 4.0, 6.0)      # noise/ISI multiplier per CPd attempt
NOISE_FLOOR = 1.0e-4


MIN_SECTION_COEFFS, MIN_TOTAL_COEFFS = 128, 192        # INFO1a codes 0: what every analogue modem supports


def filter_tap_budget(max_section_coeffs: int = MIN_SECTION_COEFFS,
                      max_total_coeffs: int = MIN_TOTAL_COEFFS) -> tuple[int, int]:
    """(LZ2, LP1) for the z2 + p1 design within INFO1a Lmax (bits 16:17) and Ltot (bits 14:15).
    The feedback section keeps POSTCURSORS; the prefilter takes what remains, up to PREFILTER_TAPS."""
    lmax, ltot = int(max_section_coeffs), int(max_total_coeffs)
    if lmax < 1 or ltot < 2:
        raise ValueError("INFO1a filter limits must allow at least one tap per section")
    post = min(POSTCURSORS, lmax, ltot - 1)
    return min(PREFILTER_TAPS, lmax, ltot - post), post


def positive_levels(law: str) -> list[float]:
    values = g711.decode(np.arange(256), law)
    return sorted({float(v) for v in values if v > 0})


def _codec(samples, law: str):
    return g711.decode(g711.encode(np.rint(samples).astype(np.int32), law), law).astype(float)


def design_from_fit(fit: dict, *, law: str, lu: float, trellis: int = 0, max_tx_rms: float | None = None,
                    gap_sigmas: float = 9.0, training_symbols: int = TRN1U_SYMBOLS,
                    prefilter_taps: int = PREFILTER_TAPS, postcursors: int = POSTCURSORS):
    """Phase 4 parameter design from a TRN1u channel estimate: the precoder / prefilter, the transmit gain, and a
    ``design_plan(safety, measured_noise)`` giving the constellation, moduli and rate. None if the precoder is unstable.
    Shared by the paired session and the streaming digital session. ``prefilter_taps`` (LZ2) and ``postcursors`` (LP1)
    must respect the analogue modem's INFO1a limits (see ``filter_tap_budget``)."""
    taps = precoder.design(fit["h"], prefilter_taps=prefilter_taps, postcursors=postcursors,
                           noise_floor=NOISE_FLOOR)
    if not taps["stable"]:
        return None
    tx_gain = precoder.transmit_gain(taps["z2"], taps["p1"])
    # The precoder gain turns the channel-estimate error into slicer noise proportional to
    # the signal: |dh| ~ sqrt(taps/N) * sigma / LU per the least-squares variance.
    estimate_isi = (np.sqrt(ESTIMATE_TAPS / training_symbols) * fit["residual_rms"] / lu
                    * np.sqrt(precoder.feedback_power_gain(taps["p1"])))
    taps["isi_ratio"] = float(np.hypot(taps["isi_ratio"], estimate_isi))
    base_noise = max(fit["residual_rms"], 1.0) * 1.5
    levels = positive_levels(law)

    def design_plan(safety: float, measured_noise: float = 0.0):
        isi = taps["isi_ratio"] * safety
        noise = max(base_noise, measured_noise) * safety
        if max_tx_rms is None:           # no transmit-power budget modelled (see the gap map)
            return modulation.design_with_isi(levels, noise, isi, trellis=trellis,
                                              min_gap_sigmas=gap_sigmas)
        return modulation.design_with_power(levels, noise, isi,
                                            max_tx_rms / tx_gain, trellis=trellis,
                                            min_gap_sigmas=gap_sigmas)

    return taps, tx_gain, design_plan


def run(channel, payload: bytes, *, law: str = "pcmu", noise_rms: float = 0.0,
        lu: float = DEFAULT_LU, seed: int = 1, trellis: int = 0,
        max_tx_rms: float | None = None, gap_sigmas: float = 9.0,
        later_segments=()) -> dict:
    """Run one session; returns the events, the negotiated plan and the received payload."""
    rng = np.random.default_rng(seed)
    channel = np.asarray(channel, dtype=float)
    events: list[tuple[str, str, object]] = []

    def say(who, what, detail=None):
        events.append((who, what, detail))

    noise_state = {"rms": noise_rms}

    def through_line(samples):
        out = precoder.apply_channel(samples, channel)
        if noise_state["rms"]:
            out = out + rng.normal(0.0, noise_state["rms"], out.shape)
        return _codec(out, law)

    # -- Phase 2: capability exchange (bits cross and are parsed) ----------------------------
    info0d = v90.parse_info0d(info.build_info0d(v92=True, rate3429=1))
    info0a = info.parse_info0a(info.build_info0a(v92=True, rate3429=1))
    negotiated = info.negotiate(info0d, info0a)
    say("both", "negotiated", negotiated)
    if not negotiated["v92"]:
        return {"events": events, "connected": False, "reason": "not V.92"}
    blocks = [{}] * 5 + [{"hi_carrier": 1}]
    info1d = v90.parse_info1d(__import__("v34_info").build_info1c(blocks=blocks))
    if not info.pcm_upstream_supported(info1d):
        return {"events": events, "connected": False, "reason": "no PCM upstream"}
    info1a = info.parse_info1a_pcm(info.build_info1a_pcm(sections=0, max_coeffs_code=3,
                                                         max_section_code=2, md_len=0))
    say("analogue", "INFO1a", {"sections": info1a["filter_sections"],
                               "max_coeffs": info1a["max_total_coeffs"]})

    # -- Phase 3: training -------------------------------------------------------------------
    trn = lu * np.array(phase3.trn1u(TRN1U_SYMBOLS))
    heard = through_line(trn)
    fit = estimate.estimate(heard, trn, ESTIMATE_TAPS)
    say("digital", "channel estimated", {"residual_rms": round(fit["residual_rms"], 2)})
    jd = phase3.parse_jd_jp(phase3.build_jd(phase3.jd_rate_mask([28000, 48000, 56000])))
    jp = phase3.parse_jd_jp(phase3.build_jp(0))
    say("digital", "Jd", jd["rates"])
    say("digital", "Jp", {"phase_symbols": jp["phase_symbols"]})
    cpt_bits = cp.build_cpu(kind=cp.CPT, drn=9, constellations=[(10, 20, 30)])
    say("analogue", "CPt", cp.parse_cpu(cpt_bits)["rate"])

    # -- Phase 4: parameter design (all bits cross as a CPd) ---------------------------------
    designed = design_from_fit(fit, law=law, lu=lu, trellis=trellis, max_tx_rms=max_tx_rms,
                               gap_sigmas=gap_sigmas, training_symbols=TRN1U_SYMBOLS)
    if designed is None:
        say("digital", "precoder unstable", None)
        return {"events": events, "connected": False, "reason": "precoder unstable"}
    taps, tx_gain, design_plan = designed

    # B1u qualifies the parameters. A failure sends the digital modem back to a more
    # conservative CPd, which stands in for rate renegotiation (clause 9.8).
    for attempt, safety in enumerate(SAFETY_LADDER, 1):
        plan = design_plan(safety)
        if plan is None:
            say("digital", "no PCM upstream rate", {"attempt": attempt})
            return {"events": events, "connected": False, "reason": "rate too low",
                    "attempts": attempt}
        outcome = _phase4(plan, taps, law, payload, through_line, say)
        outcome.update(attempts=attempt, safety=safety, tx_gain=tx_gain, plan=plan)
        if outcome["qualified"]:
            result = {"events": events, **outcome}
            if later_segments:
                result["segments"] = _later_segments(
                    later_segments, noise_state, plan, attempt - 1, taps, law, through_line,
                    say, design_plan, [dict(rate=outcome["rate"],
                                            payload_exact=outcome["payload_exact"],
                                            renegotiations=0)])
                result["payload_exact"] = all(s["payload_exact"] for s in result["segments"])
            return result
        say("digital", "B1u failed, renegotiating", {"attempt": attempt, **{
            k: outcome[k] for k in ("b1_errors", "trellis_violations")}})
    return {"events": events, "connected": False, "reason": "B1u never qualified",
            "attempts": len(SAFETY_LADDER)}


def _later_segments(segments, noise_state, plan, ladder_index, taps, law, through_line, say,
                    design_plan, results) -> list[dict]:
    """Carry further payload segments after a change in line noise.

    The CPd in force is tried first (the call continues if B1u and the whole segment are clean).
    If not, the digital modem measures the slicer noise it saw, redesigns with that noise and a
    more conservative safety factor and sends a new CPd: rate renegotiation on the existing
    channel estimate (clause 9.8), with no new TRN1u."""
    for payload, noise in segments:
        noise_state["rms"] = noise
        say("line", "noise changed", noise)
        outcome = _phase4(plan, taps, law, payload, through_line, say)
        renegotiations = 0
        rung = -1
        while not outcome["qualified"]:
            # A renegotiation designs from the noise just measured, with its own safety ladder:
            # the startup ladder's multiplier was sized for the unknown noise of the first CPd.
            rung += 1
            if rung >= len(SAFETY_LADDER):
                results.append(dict(rate=None, payload_exact=False, renegotiations=renegotiations))
                return results
            measured = outcome.get("slicer_error_rms") or 0.0
            plan = design_plan(SAFETY_LADDER[rung], measured * 1.5)
            if plan is None:
                say("digital", "no PCM upstream rate after renegotiation", None)
                results.append(dict(rate=None, payload_exact=False, renegotiations=renegotiations))
                return results
            renegotiations += 1
            say("digital", "rate renegotiation", {"measured_noise": round(measured, 1),
                                                   "attempt": renegotiations})
            outcome = _phase4(plan, taps, law, payload, through_line, say)
        results.append(dict(rate=outcome["rate"], payload_exact=outcome["payload_exact"],
                            renegotiations=renegotiations))
    return results


def _phase4(plan, taps, law, payload, through_line, say) -> dict:
    """One CPd -> B1u -> data pass. ``qualified`` means B1u arrived without a bit error and
    the trellis parity held for the whole transfer (the parity check stands in for the
    error monitoring that would trigger a retrain or rate renegotiation)."""
    cpd_bits = cp.build_cpd(
        drn=plan["drn"], trellis=plan["trellis"], gain4=0x4000, moduli=plan["moduli"],
        interval_sets=(0,) * 6, constellations=[[int(v) for v in plan["constellation"]]],
        precoder={"z1": [], "p1": [], "z2": [], "p2": []})
    parsed = cp.parse_cpd(cpd_bits)
    say("digital", "CPd", {"rate": parsed["rate"], "bits": plan["bits"]})
    # The analogue modem takes the constellation and moduli from the CPd bits; the filter
    # taps travel at higher resolution here than Q0.15/Q1.14 would allow (see the gap map).
    profile = pcm.PcmUpstreamProfile(
        constellations=tuple(tuple(float(v) for v in s) for s in parsed["constellations"] if s),
        interval_set=tuple(parsed["interval_sets"]) * 2, moduli=parsed["moduli"],
        trellis=parsed["trellis"], bits=2 * (parsed["drn"] + 17),
        precoder_p1=taps["p1"], prefilter_z2=taps["z2"])
    delay = taps["delay"]
    tx = pcm.PcmUpstreamTransmitter(profile, record=True)
    rx = pcm.PcmUpstreamReceiver(profile)
    scrambler = v34_tx.Scrambler("answer")
    k = profile.bits_per_frame
    data_bits = [(byte >> i) & 1 for byte in payload for i in range(8)]
    data_bits += [0] * (-len(data_bits) % k)
    padding = delay // 12 + 3
    stream = [1] * (B1U_FRAMES * k) + data_bits
    scrambled = [scrambler.step(b) for b in stream]
    frames = [scrambled[i:i + k] for i in range(0, len(scrambled), k)]
    wire: list[float] = []
    for frame in frames + [[0] * k] * padding:           # flush the pre-filter delay
        wire.extend(tx.encode_frame(frame))
    line = through_line(wire)
    received = line[delay:delay + 12 * len(frames)]
    reference = np.array(tx.levels[:12 * len(frames)])
    result = {"connected": False, "qualified": False, "rate": parsed["rate"],
              "delay": delay, "peak_transmit": float(max(abs(v) for v in wire)),
              "tx_rms": float(np.sqrt(np.mean(np.square(wire)))),
              "slicer_error_rms": float(np.sqrt(np.mean((received - reference) ** 2))),
              "planned_noise_rms": plan.get("noise_rms"),
              "profile": profile, "line_samples": line, "line_delay": delay,
              "b1_errors": B1U_FRAMES * k, "trellis_violations": 0}
    decoded: list[int] = []
    violations_at_b1 = 0
    for i in range(len(frames)):
        try:
            decoded.extend(rx.decode_frame([float(v) for v in received[12 * i:12 * i + 12]]))
        except ValueError as error:
            say("digital", "data frame lost", {"frame": i, "error": str(error)})
            result.update(reason="data frame lost", failed_frame=i,
                          trellis_violations=rx.trellis_violations)
            return result
        if i == B1U_FRAMES - 1:
            violations_at_b1 = rx.trellis_violations
    plain = v34_datarx.descramble_stream(decoded, [], "answer")
    b1_errors = sum(1 for b in plain[:B1U_FRAMES * k] if b != 1)
    payload_bits = plain[B1U_FRAMES * k:B1U_FRAMES * k + 8 * len(payload)]
    got = bytes(sum(payload_bits[8 * j + i] << i for i in range(8)) for j in range(len(payload)))
    say("digital", "B1u", {"errors": int(b1_errors)})
    result.update(b1_errors=int(b1_errors), trellis_violations=rx.trellis_violations,
                  qualified=(b1_errors == 0 and rx.trellis_violations == 0),
                  connected=(b1_errors == 0 and rx.trellis_violations == 0),
                  payload=got, payload_exact=got == payload)
    return result
