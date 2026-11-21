"""A whole V.92 start-up between a digital and an analogue modem, offline: Phase 3 to Phase 4 to data.

Phase 3 is ``v92_phase3_exchange`` (Ru..Ja, Sd..Jd, Su/Jp/Jp'/SCR, Ri, CPt, R-bar-i, E1u). Its CPt, parsed by
the digital modem, is the training profile for Phase 4: ``v92_phase4_controller`` then runs TRN2 through E
into data mode with that profile (the training mapper both sides use for TRN2d, SUV and CP) and carries a
payload each way: downstream in the V.90 data mode set by the analogue modem's CPu, upstream in PCM-upstream
mode set by the digital modem's CPd.

Both phases run over a delay line each way (the upstream also crosses G.711) with the lines drained between
them. Everything the limits of the parts say still applies: unit-gain lines, sign decisions, a transmit
gain G of 1, the Phase 4 assumptions (TRN2u label bit order), DIL of length zero, no timers. This is the
shape of the start-up, with every message crossing as bits or symbols and detected, not a modem.
"""

from __future__ import annotations

import numpy as np

import v34_g711
import v92_cp
import v92_data_mode as dm
import v92_ja
import v92_phase3_exchange as p3
import v92_phase4_controller as p4

BLOCK = 160


def _masks():
    return [tuple(range(16 + 16 * i, 32 + 16 * i)) for i in range(6)]


def run(law: str = "pcmu", *, delay: int = 240, up_payload: bytes = b"", down_payload: bytes = b"",
        max_blocks: int = 1600) -> dict:
    descriptor = v92_ja.build_ja_descriptor(N=0, upstream_rates=(28000.0, 32000.0, 36000.0))
    cpt_bits = v92_cp.build_cpu(kind=v92_cp.CPT, drn=2, constellations=_masks(),
                                interval_sets=(0, 1, 2, 3, 4, 5), alaw=(law == "pcma"))
    analogue = p3.AnalogueSide(descriptor, law=law, cpt_bits=cpt_bits)
    digital = p3.DigitalSide(law)
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    to_digital = [0.0] * delay
    to_analogue = [silence] * delay

    def exchange(a_samples, d_wire):
        a = v34_g711.roundtrip(np.rint(a_samples), law).astype(float)
        d = v34_g711.decode(np.frombuffer(d_wire, dtype=np.uint8), law).astype(float)
        to_digital.extend(a)
        to_analogue.extend(d)
        digital.feed(np.array(to_digital[:BLOCK]))
        del to_digital[:BLOCK]
        analogue.feed(np.array(to_analogue[:BLOCK]))
        del to_analogue[:BLOCK]

    blocks = 0
    while blocks < max_blocks and not (analogue.e1u_sent and digital.state == "phase4-ready"):
        exchange(analogue.tx(BLOCK), digital.tx(BLOCK))
        blocks += 1
    if not (analogue.e1u_sent and digital.state == "phase4-ready"):
        return {"ok": False, "reason": "phase 3 incomplete", "analogue": analogue, "digital": digital}
    phase3_blocks = blocks
    # Drain the lines: both sides idle while what was in flight arrives and is ignored.
    for _ in range(delay // BLOCK + 4):
        exchange(analogue.tx(BLOCK), digital.tx(BLOCK))

    profile = dm.downstream_profile_from_cpu(digital.cpt)
    a4 = p4.analogue_side(profile, law, ack_wait=2 * delay + 400, payload=up_payload)
    d4 = p4.digital_side(profile, law, ack_wait=2 * delay + 400, payload=down_payload)
    to_digital = [0.0] * delay
    to_analogue = [silence] * delay
    for t in range(0, 90000, BLOCK):
        a = np.array(a4.transmit(BLOCK))
        a = v34_g711.roundtrip(np.rint(a), law).astype(float)
        d = bytes(d4.transmit(BLOCK))
        to_digital.extend(a)
        to_analogue.extend(d)
        d4.receive(np.array(to_digital[:BLOCK]))
        del to_digital[:BLOCK]
        a4.receive(bytes(to_analogue[:BLOCK]))
        del to_analogue[:BLOCK]
        if (len(d4.data_rx.payload()) >= len(up_payload)
                and len(a4.data_rx.payload()) >= len(down_payload) and a4.connected and d4.connected):
            break
    return {"ok": True, "phase3_blocks": phase3_blocks, "analogue3": analogue, "digital3": digital,
            "analogue4": a4, "digital4": d4, "profile": profile,
            "up_received": d4.data_rx.payload(), "down_received": a4.data_rx.payload()}
