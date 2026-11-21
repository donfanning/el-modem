"""V.92 fast parameter exchange (ITU-T V.92 clause 9.9, Figure 19), paired/offline.

From data mode the analogue modem sends RM for 384T and RM' for 24T (Tables 25 and 26: K values set directly, run
through the data-mode precoder, prefilter and trellis: ``PcmUpstreamTransmitter.encode_intervals``) and goes
straight into SUVu. The digital modem detects RM and the RM-to-RM' transition, answers with Rf for 384T and Rf-bar for
24T (clause 8.8.4: the 12-symbol pattern of each interval's highest-power codeword with the + + - - sign pattern) and then
sends SUVd. Each side reads the other's SUV / CP / E / B1 from where the peer's R signal ends; from there it is the
exchange of ``v92_phase4_controller`` without TRN2 (``analogue_trn`` and ``digital_trn`` are zero), with new
parameters, then data again.

The paired model runs SUV/CP/E in the preceding data-mode encoders/decoders,
starts fresh FPE control memories after the R signals, transmits the required
48-frame old-mode FB1u, then starts B1u in fresh new-CPd state. The functions
model both initiation directions with known frame alignment, an ideal delay,
and pattern-window detection for R signals. They are not live DCE integration
or real-line qualification.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import v34_g711
import v90_digital_phase4 as dp4
import v92_pcm_upstream as pu
import v92_phase3
import v92_phase4_controller as p4
from v92_renegotiation import _Tx

BLOCK = 160
R_FRAMES = 32                  # 384T = 32 frames of 12
RBAR_FRAMES = 2                # 24T


def rf_masks(data_profile: dict) -> list[tuple[int, ...]]:
    sets = data_profile.get("codec_constellations") or data_profile["constellations"]
    return [tuple(sets[i]) for i in data_profile["interval_constellations"]]


def rf_signals(data_profile: dict, law: str) -> tuple[bytes, bytes]:
    """(Rf for 384T then Rf-bar for 24T as one bytes object, the 12-symbol Rf and Rf-bar blocks)."""
    masks = rf_masks(data_profile)
    rf = v92_phase3.rf_codewords(masks, law)
    rbar = v92_phase3.rf_codewords(masks, law, inverted=True)
    return rf * R_FRAMES + rbar * RBAR_FRAMES, rf, rbar


def exchange(round1: dict, law: str, *, delay: int = 240, up_payload: bytes = b"", down_payload: bytes = b"",
             drn_down: int = 1, drn_up: int = 8, lu: float = 2000.0, max_blocks: int = 700) -> dict:
    data_profile = dm_profile(round1)
    up_profile = round1["digital4"].data_rx.rx.profile             # the CPd-derived upstream profile (gain 1)
    moduli = up_profile.moduli
    both, rf, rbar = rf_signals(data_profile, law)
    tx = pu.PcmUpstreamTransmitter(up_profile)
    rm = np.concatenate([np.array(tx.encode_intervals(pu.rm_intervals(moduli))) for _ in range(R_FRAMES)])
    rmp = np.concatenate([np.array(tx.encode_intervals(pu.rm_intervals(moduli, prime=True)))
                          for _ in range(RBAR_FRAMES)])
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    a_tx, d_tx = _Tx(0.0, as_bytes=False), _Tx(silence, as_bytes=True)
    # The analogue modem starts at once: RM, RM', then its SUV exchange.
    a2 = p4.analogue_side(data_profile, law, lu=lu, analogue_trn=0,
                          ack_wait=2 * delay + 400, drn=drn_down,
                          payload=up_payload, fast_upstream_profile=up_profile)
    a_tx.pre = [float(v) for v in np.concatenate((rm, rmp))]
    a_tx.side = a2
    d2 = p4.digital_side(data_profile, law, digital_trn=0, ack_wait=2 * delay + 400,
                         drn=drn_up, payload=down_payload,
                         fast_upstream_profile=up_profile)
    to_digital = [0.0] * delay
    to_analogue = [silence] * delay
    up_hist: list[float] = []
    down_hist: list[int] = []
    rx = pu.PcmUpstreamReceiver(up_profile)
    rm_ks, rmp_ks = pu.rm_intervals(moduli), pu.rm_intervals(moduli, prime=True)
    align = delay % 12                                             # data-frame alignment, known from data mode
    next_frame, rm_run = align, 0
    d_start = a_start = None
    d_fed = a_fed = None
    scan_at, rf_at = delay % 12, None
    events: list[tuple[str, int]] = []
    tx_cursor = 0
    for blk in range(max_blocks):
        a_out = a_tx.next(BLOCK)
        d_out = d_tx.next(BLOCK)
        tx_cursor += BLOCK
        to_digital.extend(v34_g711.roundtrip(np.rint(a_out), law).astype(float))
        to_analogue.extend(d_out)
        up_chunk = np.array(to_digital[:BLOCK])
        del to_digital[:BLOCK]
        down_chunk = bytes(to_analogue[:BLOCK])
        del to_analogue[:BLOCK]
        up_hist.extend(up_chunk)
        down_hist.extend(down_chunk)
        # digital: RM frames, then the RM-to-RM' transition
        while d_start is None and next_frame + 12 <= len(up_hist):
            ks = rx.decode_intervals([float(v) for v in up_hist[next_frame:next_frame + 12]])
            if ks == rm_ks:
                rm_run += 1
                if rm_run == 8:
                    events.append(("rm", next_frame))
            elif rm_run >= 8 and ks == rmp_ks:
                events.append(("rm-prime", next_frame))
                d_start = next_frame + 12 * RBAR_FRAMES          # SUVu begins when RM' ends
                d_fed = d_start
                # RM-prime is detected after the current 160-sample callback
                # has already been emitted. Schedule Rf at the next data-frame
                # boundary rather than at that arbitrary callback edge.
                wait = (-tx_cursor) % 12
                d_tx.pre = [silence] * wait + list(both)         # answer with Rf then Rf-bar
                d_tx.side = d2
            else:
                rm_run = 0
            next_frame += 12
        # analogue: find Rf (eight whole periods in a row at one phase), then watch that phase for the half-period
        # jump to Rf-bar (Rf-bar is Rf shifted by six symbols, so only the established phase tells them apart)
        while a_start is None:
            if rf_at is None:
                if scan_at + 12 * 8 > len(down_hist):
                    break
                if all(bytes(down_hist[scan_at + 12 * k:scan_at + 12 * k + 12]) == rf for k in range(8)):
                    rf_at = scan_at + 12 * 8
                    events.append(("rf", scan_at))
                else:
                    scan_at += 12
            else:
                if rf_at + 12 > len(down_hist):
                    break
                window = bytes(down_hist[rf_at:rf_at + 12])
                if window == rf:
                    rf_at += 12
                elif window == rbar:
                    events.append(("rf-bar", rf_at))
                    a_start = rf_at + 12 * RBAR_FRAMES
                    a_fed = a_start
                else:
                    rf_at, scan_at = None, rf_at          # not Rf after all: look again from here
        if d_fed is not None and len(up_hist) > d_fed:
            d2.receive(np.array(up_hist[d_fed:]))
            d_fed = len(up_hist)
        if a_fed is not None and len(down_hist) > a_fed:
            a2.receive(bytes(down_hist[a_fed:]))
            a_fed = len(down_hist)
        if (a_start is not None and d_start is not None and a2.connected and d2.connected
                and len(d2.data_rx.payload()) >= len(up_payload)
                and len(a2.data_rx.payload()) >= len(down_payload)):
            break
    return {"a2": a2, "d2": d2, "events": events, "up_received": d2.data_rx.payload(),
            "down_received": a2.data_rx.payload()}


def exchange_digital_initiated(round1: dict, law: str, *, delay: int = 240,
                               up_payload: bytes = b"", down_payload: bytes = b"",
                               drn_down: int = 1, drn_up: int = 8, lu: float = 2000.0,
                               max_blocks: int = 700) -> dict:
    """Paired mirror: digital sends Rf/Rf-bar first; analogue responds with RM/RM-prime.

    As in :func:`exchange`, this is an idealized sample-synchronous loop. Both
    starts are placed on data-frame boundaries and the frame phase is carried
    through the transport delay. The caller supplies a prior connected round
    so the new parameter exchange reuses its negotiated training profile.
    """
    data_profile = dm_profile(round1)
    up_profile = round1["digital4"].data_rx.rx.profile
    moduli = up_profile.moduli
    both, rf, rbar = rf_signals(data_profile, law)
    tx = pu.PcmUpstreamTransmitter(up_profile)
    rm = np.concatenate([np.array(tx.encode_intervals(pu.rm_intervals(moduli)))
                         for _ in range(R_FRAMES)])
    rmp = np.concatenate([np.array(tx.encode_intervals(pu.rm_intervals(moduli, prime=True)))
                          for _ in range(RBAR_FRAMES)])
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    a_tx, d_tx = _Tx(0.0, as_bytes=False), _Tx(silence, as_bytes=True)
    a2 = p4.analogue_side(data_profile, law, lu=lu, analogue_trn=0,
                          ack_wait=2 * delay + 400, drn=drn_down,
                          payload=up_payload, fast_upstream_profile=up_profile)
    d2 = p4.digital_side(data_profile, law, digital_trn=0,
                         ack_wait=2 * delay + 400, drn=drn_up,
                         payload=down_payload, fast_upstream_profile=up_profile)
    # The digital initiator starts Rf at a frame boundary and continues into
    # SUVd. The analogue responder waits for Rf-bar before starting RM.
    d_tx.pre = list(both)
    d_tx.side = d2
    to_digital = [0.0] * delay
    to_analogue = [silence] * delay
    up_hist: list[float] = []
    down_hist: list[int] = []
    rx = pu.PcmUpstreamReceiver(up_profile)
    rm_ks, rmp_ks = pu.rm_intervals(moduli), pu.rm_intervals(moduli, prime=True)
    # Rf crosses one-way delay before the analogue responds; RM crosses the
    # return delay. Since the two R sequences are each frame-length multiples,
    # RM's observed phase at the digital receiver is the round-trip phase.
    upstream_align = (2 * delay) % 12
    next_frame, rm_run = upstream_align, 0
    d_start = a_start = None
    d_fed = a_fed = None
    scan_at, rf_at = delay % 12, None
    events: list[tuple[str, int]] = []
    tx_cursor = 0
    for _ in range(max_blocks):
        a_out = a_tx.next(BLOCK)
        d_out = d_tx.next(BLOCK)
        tx_cursor += BLOCK
        to_digital.extend(v34_g711.roundtrip(np.rint(a_out), law).astype(float))
        to_analogue.extend(d_out)
        up_chunk = np.array(to_digital[:BLOCK])
        del to_digital[:BLOCK]
        down_chunk = bytes(to_analogue[:BLOCK])
        del to_analogue[:BLOCK]
        up_hist.extend(up_chunk)
        down_hist.extend(down_chunk)

        # Digital side: find the RM run and its transition, then begin receiving
        # the new CPu/SUVu exchange only after the complete RM-prime interval.
        while d_start is None and next_frame + 12 <= len(up_hist):
            ks = rx.decode_intervals(up_hist[next_frame:next_frame + 12])
            if ks == rm_ks:
                rm_run += 1
                if rm_run == 8:
                    events.append(("rm", next_frame))
            elif rm_run >= 8 and ks == rmp_ks:
                events.append(("rm-prime", next_frame))
                d_start = next_frame + 12 * RBAR_FRAMES
                d_fed = d_start
            else:
                rm_run = 0
            next_frame += 12

        # Analogue side: detect Rf at the known data-frame phase, then the
        # Rf-to-Rf-bar transition. RM begins at the end of Rf-bar.
        while a_start is None:
            if rf_at is None:
                if scan_at + 12 * 8 > len(down_hist):
                    break
                if all(bytes(down_hist[scan_at + 12 * k:scan_at + 12 * k + 12]) == rf
                       for k in range(8)):
                    rf_at = scan_at + 12 * 8
                    events.append(("rf", scan_at))
                else:
                    scan_at += 12
            else:
                if rf_at + 12 > len(down_hist):
                    break
                window = bytes(down_hist[rf_at:rf_at + 12])
                if window == rf:
                    rf_at += 12
                elif window == rbar:
                    events.append(("rf-bar", rf_at))
                    a_start = rf_at + 12 * RBAR_FRAMES
                    a_fed = a_start
                    # The trigger was observed in a 160-sample callback. Wait
                    # until the next response-side data-frame boundary rather
                    # than starting RM at an arbitrary RTP block edge.
                    response_phase = delay % 12
                    wait = (response_phase - tx_cursor) % 12
                    a_tx.pre = [0.0] * wait + list(np.concatenate((rm, rmp)))
                    a_tx.side = a2
                else:
                    rf_at, scan_at = None, rf_at

        if d_fed is not None and len(up_hist) > d_fed:
            d2.receive(np.asarray(up_hist[d_fed:], dtype=float))
            d_fed = len(up_hist)
        if a_fed is not None and len(down_hist) > a_fed:
            a2.receive(bytes(down_hist[a_fed:]))
            a_fed = len(down_hist)
        if (a_start is not None and d_start is not None and a2.connected and d2.connected
                and len(d2.data_rx.payload()) >= len(up_payload)
                and len(a2.data_rx.payload()) >= len(down_payload)):
            break
    return {"a2": a2, "d2": d2, "events": events,
            "up_received": d2.data_rx.payload(),
            "down_received": a2.data_rx.payload()}


def dm_profile(round1: dict) -> dict:
    import v92_data_mode as dm
    return dm.downstream_profile_from_cpu(round1["digital4"].peer_cp)
