"""V.92 data modes, driven by the Phase 4 CP frames.

* Downstream (digital to analogue) is the V.90 data mode: V.92 says B1d is as in V.90 and the
  digital modem's mapper is the V.90 one. Its parameters come from the analogue modem's CPu, which has
  the layout of the V.90 CP, so ``downstream_profile_from_cpu`` turns a parsed CPu into the profile that
  ``V90DigitalPhase4Encoder`` / ``V90DigitalPhase4Receiver`` take. B1d is 48 six-symbol frames of
  scrambled ones with mapper memories fresh at its start (the encoder's ``reset``).
* Upstream (analogue to digital) is the PCM-upstream mode of ``v92_pcm_upstream`` with the
  profile from the digital modem's CPd (``profile_from_cpd``), received by
  ``PcmUpstreamStreamReceiver``.

The receiver here supports S_r = 0 (no spectral shaping) only, as the V.90 receiver does.
"""

from __future__ import annotations

import v90_digital_phase4 as dp4
import v90_digital_phase4_rx as rx4

B1D_FRAMES = 48


def downstream_profile_from_cpu(parsed: dict) -> dict:
    """V.90-style CP (or, for a CPt, training) profile from a parsed V.92 CPu / CPt (``v92_cp.parse_cpu``)."""
    if parsed.get("kind") not in ("cpu", "cpt"):
        raise ValueError("need a parsed CPu or CPt")
    if not parsed.get("rate"):
        raise ValueError("a CPu with DRN 0 is a cleardown request, not a data mode")
    sets = tuple(tuple(c) for c in parsed["constellations"])
    return {
        "kind": "cpt" if parsed["kind"] == "cpt" else "cp", "codec": parsed["codec"], "ack": int(parsed["ack"]),
        "rate": parsed["rate"],
        "spectral_sign_bits": int(parsed["sr"]), "lookahead": int(parsed["lookahead"]),
        "shape": tuple(parsed["shaping"]) if parsed["sr"] else (),
        "interval_constellations": tuple(parsed["interval_sets"]),
        "constellations": sets,
        "codec_constellations": (tuple(tuple(c) for c in parsed["codec_constellations"])
                                 if parsed.get("codec_constellations") else sets),
    }


def encode_downstream(profile: dict, payload: bytes, *, b1_frames: int = B1D_FRAMES) -> bytes:
    """B1d (``b1_frames`` frames of ones, fresh mapper memories) then ``payload`` as data bits,
    least-significant bit first, padded with zeros to a whole mapper frame."""
    law = profile["codec"]
    encoder = dp4.V90DigitalPhase4Encoder(profile, law)
    encoder.reset()
    frame_bits = encoder.s + encoder.k
    wire = encoder.encode_bits([1] * (b1_frames * frame_bits))
    bits = [(byte >> i) & 1 for byte in payload for i in range(8)]
    bits += [0] * (-len(bits) % frame_bits)
    return wire + encoder.encode_bits(bits) + encoder.flush_lookahead()


def decode_downstream(profile: dict, wire: bytes, *, b1_frames: int = B1D_FRAMES) -> dict:
    receiver = rx4.V90DigitalPhase4Receiver(profile, profile["codec"])
    bits = receiver.decode_codewords(wire)
    frame_bits = len(bits) // (len(wire) // 6)
    b1 = bits[:b1_frames * frame_bits]
    data = bits[b1_frames * frame_bits:]
    return {"b1_errors": sum(1 for b in b1 if b != 1),
            "payload": bytes(sum(data[8 * j + i] << i for i in range(8))
                             for j in range(len(data) // 8))}
