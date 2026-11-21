"""V.92 Phase 4, digital modem to analogue modem: TRN2d, then SUVd / CPd frames (clause 8.8).

V.92 says TRN2d is "as defined in 8.6.5/V.90" and that SUVd and CPd are scrambled and
"transmitted using the corresponding TRN2d modulation", so this reuses the V.90 digital Phase 4
mapper (``V90DigitalPhase4Encoder``: mixed-radix modulus mapping, GPC-scrambled ones, sign
differential) and its receiver for the bits, and the V.92 frame builders / parsers in ``v92_cp``.

The mapper takes S + K input bits per six-symbol frame, so each SUVd/CPd frame is followed by zero
fill bits to the next mapper-frame boundary (SUVd's own text says to extend to the next multiple of
6 symbols). The V.90 receiver supports S_r = 0 only, so the training profile must have
``spectral_sign_bits`` 0.

Not checked against hardware or a printed figure: only that the V.90 mapper, which is validated
against the Gough V.90 data in this repo, carries V.92 frames and the V.92 parsers read them back.
"""

from __future__ import annotations

import v90_digital_phase4 as dp4
import v90_digital_phase4_rx as rx4
import v92_cp


def build_downstream(frames, cpt_profile: dict, law: str, *, trn_symbols: int = 2040) -> bytes:
    encoder = dp4.V90DigitalPhase4Encoder(cpt_profile, law)
    out = encoder.trn2d(trn_symbols)
    bits = []
    frame_bits = encoder.s + encoder.k
    for frame in frames:
        bits.extend(int(b) & 1 for b in frame)
        bits.extend([0] * (-len(frame) % frame_bits))
    return out + encoder.encode_bits(bits) + encoder.flush_lookahead()


def recover_downstream(codewords: bytes, cpt_profile: dict, law: str) -> list[dict]:
    receiver = rx4.V90DigitalPhase4Receiver(cpt_profile, law)
    usable = len(codewords) - len(codewords) % 6
    bits = []
    for i in range(0, usable, 6):
        bits.extend(receiver.decode_codewords(bytes(codewords[i:i + 6])))
    try:
        first_zero = bits.index(0)
    except ValueError:
        return []
    pos = max(0, first_zero - 17)
    frames = []
    while pos < len(bits):
        chunk = bits[pos:]
        parsed = v92_cp.parse_suvd(chunk) or v92_cp.parse_cpd(chunk)
        if parsed is None:
            break
        frames.append(parsed)
        end = pos + int(parsed["frame_bits"])
        pos = end + (-end % (receiver.s + receiver.k))
    return frames
