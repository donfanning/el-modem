"""A whole V.92 Phase 4 session carried by the two stream coders, checked against the sequencing model.

``v92_final_training.run`` decides, per slot, what each modem sends (TRN, SUV, CP, E, B1 and their
acknowledge bits). This turns each side's trace into the bits of the real frames, codes them with
``v92_phase4_upstream`` (analogue to digital, 4-point) and ``v92_phase4_downstream`` (digital to
analogue, the V.90 mapper), decodes them again, and returns what each receiver read: the frame kinds
and acknowledge bits in order, then the E and B1 tail.

It is still offline and still one direction at a time: there is no detection in the loop, no delay
between the streams, and E / B1 use the TRN2 modulation rather than the data mode they would really use
(B1u and B1d are in the data-mode modulation of the preceding CP; that needs the PCM-upstream receiver
and the downstream data mapper). What it checks is that the framing, the acknowledge bits and the
E-then-B1 tail survive the coders in the order the sequencing rules produce.
"""

from __future__ import annotations

import numpy as np

import v34_g711
import v90_digital_phase4 as dp4
import v90_digital_phase4_rx as rx4
import v92_cp
import v92_final_training as ft
import v92_phase4_downstream as down
import v92_phase4_upstream as up

UP_SLOT_SYMBOLS = 1008          # one analogue TRN slot; 12 slots = 12096 >= 12000T, whole 12-symbol groups
DOWN_SLOT_SYMBOLS = 510         # four digital TRN slots = 2040T, whole six-symbol frames
B1_FRAMES = 48


def _cpu_bits(ack: bool) -> list[int]:
    return v92_cp.build_cpu(drn=14, ack=ack, constellations=[(0, 3, 6, 9, 12, 15)],
                            interval_sets=(0,) * 6)


def _cpd_bits(ack: bool) -> list[int]:
    return v92_cp.build_cpd(drn=12, trellis=0, gain4=40000, ack=ack)


def trace_bits(trace, side: str, *, nb: int, mapper_frame_bits: int = 0) -> list[int]:
    """Bits for one side's trace. ``nb`` is bits per symbol upstream (2 for 4-point); downstream
    messages are counted in mapper frames of ``mapper_frame_bits``."""
    bits: list[int] = []
    for m in trace:
        if m.kind == "TRN":
            n = UP_SLOT_SYMBOLS * nb if side == "up" else DOWN_SLOT_SYMBOLS // 6 * mapper_frame_bits
            bits += [1] * n
        elif m.kind == "SUV":
            bits += v92_cp.build_suvu(ack=m.ack) if side == "up" else v92_cp.build_suvd(ack=m.ack)
        elif m.kind == "CP":
            bits += _cpu_bits(m.ack) if side == "up" else _cpd_bits(m.ack)
        elif m.kind == "E":
            bits += [0] * (12 * nb if side == "up" else 2 * mapper_frame_bits)
        elif m.kind == "B1":
            bits += [1] * (B1_FRAMES * (12 * nb if side == "up" else mapper_frame_bits))
            break                                   # B1 runs to the end of the session
    return bits


def _read_stream(bits, *, parsers, e_bits: int) -> dict:
    first_zero = bits.index(0)
    pos = max(0, first_zero - 17)
    frames = []
    while pos < len(bits):
        parsed = next((p(bits[pos:]) for p in parsers if p(bits[pos:]) is not None), None)
        if parsed is None:
            break
        frames.append((parsed["kind"], bool(parsed.get("ack", False))))
        pos += int(parsed["frame_bits"])
    rest = bits[pos:]
    zeros = 0
    while zeros < len(rest) and rest[zeros] == 0:
        zeros += 1
    return {"frames": frames, "e_zero_bits": zeros, "b1_all_ones": all(rest[zeros:]),
            "b1_bits": len(rest) - zeros, "e_expected": e_bits}


def run_session(law: str = "pcmu", *, latency: int = 1, cpt_profile: dict | None = None,
                drop=None) -> dict:
    model = ft.run(latency, digital_trn=4, analogue_trn=12, drop=drop)
    if not model["connected"]:
        return {"model": model, "connected": False}
    nb = 2
    profile = cpt_profile or _default_profile(law)
    encoder = dp4.V90DigitalPhase4Encoder(profile, law)
    frame_bits = encoder.s + encoder.k

    up_bits = trace_bits(model["traces"]["analogue"], "up", nb=nb)
    # The coder takes whole symbols: pad a multiple of 12 symbols with ones (B1 is ones).
    up_bits += [1] * (-len(up_bits) % (12 * nb))
    up_samples = np.array(up.SymbolCoder(4).encode(up_bits)) * 2000.0
    up_line = v34_g711.roundtrip(np.rint(up_samples), law).astype(float)

    down_bits = trace_bits(model["traces"]["digital"], "down", nb=nb, mapper_frame_bits=frame_bits)
    down_bits += [1] * (-len(down_bits) % frame_bits)
    down_wire = encoder.encode_bits(down_bits)

    up_read = _read_stream(up.SymbolCoder(4).decode(up_line / 2000.0),
                           parsers=(v92_cp.parse_suvu, v92_cp.parse_cpu), e_bits=12 * nb)
    receiver = rx4.V90DigitalPhase4Receiver(profile, law)
    rx_bits = []
    for i in range(0, len(down_wire) - len(down_wire) % 6, 6):
        rx_bits.extend(receiver.decode_codewords(bytes(down_wire[i:i + 6])))
    down_read = _read_stream(rx_bits, parsers=(v92_cp.parse_suvd, v92_cp.parse_cpd),
                             e_bits=2 * frame_bits)
    return {"connected": True, "model": model, "up": up_read, "down": down_read,
            "frame_bits": frame_bits}


def expected_frames(trace, side: str) -> list[tuple[str, bool]]:
    kinds = {"up": {"SUV": "suvu", "CP": "cpu"}, "down": {"SUV": "suvd", "CP": "cpd"}}[side]
    out = []
    for m in trace:
        if m.kind in kinds:
            out.append((kinds[m.kind], bool(m.ack)))
    return out


def _default_profile(law: str) -> dict:
    masks = tuple(tuple(range(16 + 16 * index, 32 + 16 * index)) for index in range(6))
    return {"kind": "cpt", "codec": law, "ack": 0, "rate": 10 * 8000 / 6,
            "spectral_sign_bits": 0, "lookahead": 0, "shape": (),
            "interval_constellations": (0, 1, 2, 3, 4, 5),
            "constellations": masks, "codec_constellations": masks}
