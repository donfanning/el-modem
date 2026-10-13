#!/usr/bin/env python3
"""Replay a real V.34 capture through our answerer and print its event log.

The V.32 equivalent is ``replay_v32_capture.py``. Same idea: feed a recording of a real peer's
signal back in so the handshake can be studied offline, without dialling.

The recording has to contain the *caller's* side of the call, because that is what an answerer
listens to. ``fixtures/reference/`` holds third-party recordings of two commercial modems on a
loop-current interface; there is no hybrid, so no echo and both ends transmit at once, which our
answerer copes with but which is worth knowing when reading the log.

    python3 replay_v34_capture.py <capture.raw> [seconds] [--modem]
"""
from __future__ import annotations

import struct
import sys

import numpy as np

from v34_ans import V34Answerer
from v34_modem import V34Modem

FS = 8000


def replay(path, secs=25.0, caller=False, log=print):
    """Drive an answerer (default) or a caller with a capture, and return the events.

    ``caller=False`` feeds the capture to a ``V34Answerer``: our transmit timeline still advances,
    so the state machine proceeds, but the audio it would have sent is discarded and the far end
    is whatever the recording contains. ``caller=True`` does the same through ``V34Modem``.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    rx = struct.unpack(f"<{len(raw) // 2}h", raw)

    events: list[tuple[float, str]] = []
    t = [0.0]
    blk = FS // 50
    got = bytearray()

    if caller:
        modem = V34Modem(debug=True, with_v8=False)
        modem._sync_events = lambda: None            # keep our own event log, not the DCE's
        modem._ev = lambda text: events.append((t[0], "C " + text))
        for i in range(0, min(len(rx), int(FS * secs)) - blk, blk):
            modem.tx_samples(blk)                    # audio discarded
            got += modem.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))
            t[0] += blk / FS
    else:
        modem = V34Answerer(log=lambda m: events.append((t[0], m)))
        for i in range(0, min(len(rx), int(FS * secs)) - blk, blk):
            modem.tx(blk)                            # audio discarded
            modem.rx(np.array(rx[i:i + blk], dtype=float))
            t[0] += blk / FS

    for at, msg in events:
        log(f"  {at:6.2f}s {msg}")
    return modem, bytes(got)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/reference/usr_v34.raw"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 25.0
    m, got = replay(path, secs, caller="--modem" in sys.argv)
    print(f"connected={getattr(m, 'connected', '?')} got={len(got)}B")