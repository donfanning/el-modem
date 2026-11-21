"""V.92 rate renegotiation initiated by the digital modem with no silence (clause 9.8, Figure 15), offline.

From data mode the digital modem sends Rd for 384T and Rd-bar for 24T (Rd: the highest-power Ucode of each
data-mode interval with the +++--- sign pattern, V.90's ``rd_codewords`` shape) and goes straight into TRN2d.
The analogue modem detects Rd and its flip, answers with Ru 384T and Ru-bar 24T (the Ru of Phase 3) and goes into
TRN2u; each side starts reading the other's TRN2 at the end of the peer's R-bar. From there it is the Phase 4
exchange of ``v92_phase4_controller`` (SUV, one CP, SUV', E, B1) with new parameters, then data again.

Assumptions beyond Phase 4's: TRN2/SUV/CP in the second round use the Phase 3 training profile with fresh mapper
memories (the text says "corresponding TRN2 modulation"); the old data link is quiet when renegotiation starts
(the figures show DATA continuing up to R, and the requirement that R begins on a data frame boundary is not
modelled); detection of R is by the pattern tracker, which needs the R signals' interval magnitudes to be close.
The opt-in ``silent_period`` path in this offline paired runner models the digital-requested, maximum-wait
sequence in Figure 16, including Ucode-0 silence, the analogue second TRN2u, R-t/R-bar-t, and a fresh Phase 4.
The optional ``silent_early_rt_after`` timing probe selects Figure 17's early R-t termination branch; paired
offline tests and the explicit test-only DCE SIP/RTP route exercise it under PCMU and PCMA. Figure 18's
analogue-requested silent exchange now has offline and live paired-session coverage under both laws, including
exact payload delivery. The runtime digital-answerer responder is enabled by the V.92 PCM DCE factory. Figure 18
has also been exercised through the local DCE SIP/RTP route, including a synthetic equalized line; it has not
been tested against an independent/real modem. Fast parameter exchange is also not implemented here.
"""

from __future__ import annotations

import numpy as np

import v34_g711
import v90_digital_phase4 as dp4
import v92_data_mode as dm
import v92_phase3
import v92_phase4_controller as p4
from v92_phase3_rx import PatternTracker

BLOCK = 160
R_SYMBOLS = 384
RBAR_SYMBOLS = 24


def digital_r_pair(profile: dict, law: str) -> bytes:
    ucodes = dp4._highest_power_interval_ucodes(profile, law)
    return dp4._r_signal(ucodes, law) * (R_SYMBOLS // 24) + dp4._r_signal(ucodes, law, r_bar=True) * (RBAR_SYMBOLS // 24)


def digital_rt_pair(profile: dict, law: str) -> bytes:
    """Return the Figure 16–18 R-t 384T followed by R-bar-t 24T pair."""
    ucodes = dp4._highest_power_interval_ucodes(profile, law)
    return (dp4._r_signal(ucodes, law) * (R_SYMBOLS // 24)
            + dp4._r_signal(ucodes, law, r_bar=True) * (RBAR_SYMBOLS // 24))


def analogue_r_pair(lu: float = 2000.0) -> np.ndarray:
    return np.concatenate((np.asarray(v92_phase3.ru(R_SYMBOLS // 6), dtype=float),
                           np.asarray(v92_phase3.ru(RBAR_SYMBOLS // 6, bar=True), dtype=float))) * float(lu)


# Kept for the existing offline runner/tests.
_r_pair = digital_r_pair


class _Tx:
    """The R signals first, then whatever the side transmits; idle before a side exists."""

    def __init__(self, idle, as_bytes: bool):
        self.pre: list = []
        self.side = None
        self.idle = idle
        self.as_bytes = as_bytes

    def next(self, n: int):
        out = self.pre[:n]
        self.pre = self.pre[n:]
        if len(out) < n:
            rest = n - len(out)
            out = out + (list(self.side.transmit(rest)) if self.side is not None else [self.idle] * rest)
        return bytes(out) if self.as_bytes else np.array(out, dtype=float)


def renegotiate(round1: dict, law: str, *, delay: int = 240, up_payload: bytes = b"",
                down_payload: bytes = b"", drn_down: int = 1, drn_up: int = 8,
                lu: float = 2000.0, max_blocks: int = 700,
                silent_period: bool = False, silent_early_rt_after: int | None = None) -> dict:
    """``round1`` is the result of ``v92_session.run``. Returns the second round's sides and what they got."""
    profile = round1["profile"]                              # Phase 3 training profile
    data_profile = dm.downstream_profile_from_cpu(round1["digital4"].peer_cp)
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    ru_tx = analogue_r_pair(lu).tolist()
    d_tx = _Tx(silence, as_bytes=True)
    a_tx = _Tx(0.0, as_bytes=False)
    d_tx.pre = list(digital_r_pair(data_profile, law))       # Rd 384T, Rd-bar 24T, then TRN2d...
    d2 = p4.digital_side(profile, law, ack_wait=2 * delay + 400, drn=drn_up, payload=down_payload,
                         silent_early_rt_after=silent_early_rt_after)
    if silent_period:
        d2.set_silent_period_request(True)
    d_tx.side = d2
    a2 = None
    to_digital = [0.0] * delay
    to_analogue = [silence] * delay
    down_hist: list[int] = []
    up_hist: list[float] = []
    a_track = PatternTracker(families=("ru",))
    d_track = PatternTracker(families=("ru",))
    a_seen = d_seen = False
    a_start = d_start = None                                 # where each side starts reading the peer's TRN2
    a_fed = d_fed = None
    events: list[tuple[str, int]] = []
    for blk in range(max_blocks):
        a_out = a_tx.next(BLOCK)
        d_out = d_tx.next(BLOCK)
        to_digital.extend(v34_g711.roundtrip(np.rint(a_out), law).astype(float))
        to_analogue.extend(d_out)
        up_chunk = np.array(to_digital[:BLOCK])
        del to_digital[:BLOCK]
        down_chunk = bytes(to_analogue[:BLOCK])
        del to_analogue[:BLOCK]
        up_hist.extend(up_chunk)
        down_hist.extend(down_chunk)
        decoded = v34_g711.decode(np.frombuffer(down_chunk, dtype=np.uint8), law).astype(float)
        for name, at in a_track.feed(decoded):
            if not a_seen:
                a_seen = True
                events.append(("rd", at))
            elif name.endswith("bar") and a_start is None:
                events.append(("rd-bar", at))
                a_start = at + RBAR_SYMBOLS                  # TRN2d begins when Rd-bar ends
                a2 = p4.analogue_side(profile, law, lu=lu, ack_wait=2 * delay + 400, drn=drn_down,
                                      payload=up_payload)
                a_tx.pre = list(ru_tx)                       # answer with Ru 384T, Ru-bar 24T, then TRN2u
                a_tx.side = a2
                a_fed = a_start
        for name, at in d_track.feed(up_chunk):
            if not d_seen:
                d_seen = True
                events.append(("ru", at))
            elif name.endswith("bar") and d_start is None:
                events.append(("ru-bar", at))
                d_start = at + RBAR_SYMBOLS
                d_fed = d_start
        if a2 is not None and a_fed is not None and len(down_hist) > a_fed:
            a2.receive(bytes(down_hist[a_fed:]))
            a_fed = len(down_hist)
        if d_fed is not None and len(up_hist) > d_fed:
            d2.receive(np.array(up_hist[d_fed:]))
            d_fed = len(up_hist)
        if (a2 is not None and a2.connected and d2.connected
                and len(d2.data_rx.payload()) >= len(up_payload)
                and len(a2.data_rx.payload()) >= len(down_payload)):
            break
    return {"a2": a2, "d2": d2, "events": events,
            "up_received": d2.data_rx.payload(),
            "down_received": None if a2 is None else a2.data_rx.payload()}


def renegotiate_analogue_initiated(round1: dict, law: str, *, up_payload: bytes = b"",
                                   down_payload: bytes = b"", drn_down: int = 1,
                                   drn_up: int = 8, lu: float = 2000.0,
                                   max_blocks: int = 700) -> dict:
    """Paired Figure 18 model: the analogue side initiates and requests silence."""
    profile = round1["profile"]
    data_profile = dm.downstream_profile_from_cpu(round1["digital4"].peer_cp)
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    a_tx = _Tx(0.0, as_bytes=False)
    d_tx = _Tx(silence, as_bytes=True)
    a_tx.pre = analogue_r_pair(lu).tolist()
    a2 = p4.analogue_side(profile, law, lu=lu, ack_wait=800, drn=drn_down,
                          payload=up_payload)
    a2.set_silent_period_request(True)
    a_tx.side = a2
    d2 = None
    to_digital = [0.0] * 240
    to_analogue = [silence] * 240
    down_hist: list[int] = []
    up_hist: list[float] = []
    a_track = PatternTracker(families=("ru",))
    d_track = PatternTracker(families=("ru",))
    a_seen = d_seen = False
    a_start = d_start = None
    a_fed = d_fed = None
    events: list[tuple[str, int]] = [("analogue-ru-start", 0)]

    for _ in range(max_blocks):
        a_out = a_tx.next(BLOCK)
        d_out = d_tx.next(BLOCK)
        to_digital.extend(v34_g711.roundtrip(np.rint(a_out), law).astype(float))
        to_analogue.extend(d_out)
        up_chunk = np.array(to_digital[:BLOCK])
        del to_digital[:BLOCK]
        down_chunk = bytes(to_analogue[:BLOCK])
        del to_analogue[:BLOCK]
        up_hist.extend(up_chunk)
        down_hist.extend(down_chunk)
        decoded = v34_g711.decode(np.frombuffer(down_chunk, dtype=np.uint8), law).astype(float)

        for name, at in a_track.feed(decoded):
            if not a_seen:
                a_seen = True
                events.append(("rd", at))
            elif name.endswith("bar") and a_start is None:
                events.append(("rd-bar", at))
                a_start = at + RBAR_SYMBOLS
                a_fed = a_start

        for name, at in d_track.feed(up_chunk):
            if not d_seen:
                d_seen = True
                events.append(("ru", at))
            elif name.endswith("bar") and d_start is None:
                events.append(("ru-bar", at))
                d_start = at + RBAR_SYMBOLS
                d2 = p4.digital_side(profile, law, ack_wait=800, drn=drn_up,
                                     payload=down_payload)
                d_tx.pre = list(digital_r_pair(data_profile, law))
                d_tx.side = d2
                d_fed = d_start

        if a_fed is not None and len(down_hist) > a_fed:
            a2.receive(bytes(down_hist[a_fed:]))
            a_fed = len(down_hist)
        if d2 is not None and d_fed is not None and len(up_hist) > d_fed:
            d2.receive(np.array(up_hist[d_fed:]))
            d_fed = len(up_hist)
        if (d2 is not None and a2.connected and d2.connected
                and len(d2.data_rx.payload()) >= len(up_payload)
                and len(a2.data_rx.payload()) >= len(down_payload)):
            break

    return {"a2": a2, "d2": d2, "events": events,
            "up_received": b"" if d2 is None else d2.data_rx.payload(),
            "down_received": a2.data_rx.payload()}
