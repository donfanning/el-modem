#!/usr/bin/env python3
"""Drive a V.34 modem from a capture with per-event timestamps and exception reporting.

``replay_v34_capture.py`` pipes the same loop through a plain event list. This one wraps each
``feed_pcm`` so an exception is reported with the time it happened rather than escaping as a bare
traceback -- useful when replaying a third-party recording whose modem may do something we have
never seen.

    python3 replay_v34_probe.py <capture.raw> [seconds] [--modem|--answer]
"""
from __future__ import annotations

import struct
import sys

from v34_ans import V34Answerer
from v34_modem import V34Modem

FS = 8000


def probe(path: str, secs: float = 27.0, mode: str = "modem", show: int = 60, with_v8: bool = False):
    with open(path, "rb") as fh:
        raw = fh.read()
    rx = struct.unpack(f"<{len(raw) // 2}h", raw)

    events: list[tuple[float, str]] = []
    t = [0.0]
    blk = FS // 50
    modem = V34Modem(debug=True, with_v8=with_v8) if mode == "modem" else V34Answerer()
    if mode == "modem":
        modem._sync_events = lambda: None
        modem._ev = lambda s: events.append((t[0], "C " + s))
    else:
        modem._log = lambda s: events.append((t[0], "A " + s))

    crashed = None
    for i in range(0, min(len(rx), int(FS * secs)) - blk, blk):
        try:
            if mode == "modem":
                modem.tx_samples(blk)
                modem.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))
            else:
                modem.tx(blk)
                modem.rx(list(rx[i:i + blk]))
        except Exception as exc:                       # noqa: BLE001 - report, do not hide
            crashed = f"{type(exc).__name__}: {exc}"
            break
        t[0] += blk / FS

    for at, msg in events[:show]:
        print(f"  {at:6.2f}s {msg}")
    if len(events) > show:
        print(f"  ... {len(events) - show} more events")
    if crashed:
        print(f"  CRASH at {t[0]:.2f}s: {crashed}")
    if mode == "modem":
        print(f"stage={modem.caller.stage!r} failed={modem.caller.failed_reason!r} "
              f"state={modem.state!r}")
    return modem, crashed


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode = "answer" if "--answer" in sys.argv else "modem"
    probe(args[0], float(args[1]) if len(args) > 1 else 27.0, mode, with_v8="--v8" in sys.argv)