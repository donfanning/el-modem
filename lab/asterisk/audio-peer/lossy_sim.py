#!/usr/bin/env python3
"""In-memory simulation of two V.22bis/V.42 modems over an RTP-like path that
loses whole 20 ms audio frames (a lost frame reaches the receiver as silence,
exactly as the modem session's gap-fill delivers it).

Fast enough to run many trials per loss level, so negotiation changes can be
judged on statistics instead of single live runs.

    python3 lossy_sim.py --loss 0 1 2 3 4 --trials 20
"""

import argparse
import random
import struct
import sys

from spandsp_v22 import V22bisModem

FRAME = 160
ZERO = struct.pack("<160h", *([0] * FRAME))


def trial(loss: float, seed: int, nbytes: int = 800, want_v42: bool = True,
          max_seconds: float = 90.0, rate: int = 2400, v42_kwargs=None,
          conceal: str = "zero") -> dict:
    rng = random.Random(seed)
    caller = V22bisModem(True, bit_rate=rate, bell212a=False)
    answerer = V22bisModem(False, bit_rate=rate, bell212a=False)
    payload = random.Random(seed ^ 0x5A5A).randbytes(nbytes)
    started = {"c": False, "a": False}
    got = bytearray()
    sent = False
    t_ready = None
    out = {"line": False, "rate": None, "c_v42": None, "a_v42": None, "ok": False,
           "seconds": None}

    def lose():
        return rng.random() < loss

    last = {"a": ZERO, "c": ZERO}

    def deliver(modem, who, frame_bytes, lost):
        """Feed one 20 ms frame, or conceal it if it was lost."""
        if not lost:
            last[who] = frame_bytes
            return modem.feed_pcm(frame_bytes)
        if conceal == "fillin":
            modem.fill_missing(FRAME)
            return b""
        if conceal == "repeat":
            return modem.feed_pcm(last[who])
        return modem.feed_pcm(ZERO)

    for tick in range(int(max_seconds * 50)):
        fc, fa = caller.tx_samples(FRAME), answerer.tx_samples(FRAME)
        decoded = deliver(answerer, "a", struct.pack("<160h", *fc), lose())
        deliver(caller, "c", struct.pack("<160h", *fa), lose())
        if sent:
            # Like the modem session, drop everything decoded before CONNECT
            # (line training/idle bits, ODP/ADP characters): it is not user data.
            got += decoded

        if want_v42:
            for who, m, calling in (("c", caller, True), ("a", answerer, False)):
                if not started[who] and m.is_settled():
                    m.start_v42(calling, **(v42_kwargs or {}))
                    started[who] = True
        if not sent and caller.is_settled() and answerer.is_settled() and (
                not want_v42 or (caller.v42_resolved and answerer.v42_resolved
                                 and started["c"] and started["a"])):
            out["line"] = True
            out["rate"] = caller.bit_rate
            out["c_v42"], out["a_v42"] = caller.v42_mode, answerer.v42_mode
            caller.feed_bytes(payload)
            sent, t_ready = True, tick
        if sent and len(got) >= nbytes:
            out["ok"] = bytes(got[:nbytes]) == payload
            out["seconds"] = (tick - t_ready) / 50
            break
    else:
        out["c_v42"] = out["c_v42"] or caller.v42_mode
        out["a_v42"] = out["a_v42"] or answerer.v42_mode
    return out


def classify(r: dict) -> str:
    if r["ok"]:
        return "ok-lapm" if r["c_v42"] == "lapm" and r["a_v42"] == "lapm" else "ok-raw"
    if not r["line"]:
        return "no-line"
    if r["c_v42"] != r["a_v42"]:
        return "v42-mismatch"
    return "data-bad"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", type=float, nargs="+", default=[0, 1, 2, 3, 4],
                    help="percent of 20 ms audio frames lost, per direction")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--raw", action="store_true", help="no V.42 (baseline)")
    ap.add_argument("--frame-size", type=int, nargs="+", default=[0],
                    help="LAPM N401 values to compare (0 = library default 128)")
    ap.add_argument("--conceal", choices=["zero", "repeat", "fillin"], default="zero",
                    help="how a lost frame is presented to the modem")
    args = ap.parse_args()
    cats = ["ok-lapm", "ok-raw", "v42-mismatch", "data-bad", "no-line"]
    print(f"{'loss%':>6} | " + " ".join(f"{c:>12}" for c in cats) + " | median s (ok)")
    for fs in args.frame_size:
        if len(args.frame_size) > 1:
            print(f"-- LAPM frame size {fs or 128}")
        for pct in args.loss:
            counts = {c: 0 for c in cats}
            times = []
            for seed in range(args.trials):
                r = trial(pct / 100.0, seed, want_v42=not args.raw, conceal=args.conceal,
                          v42_kwargs={"frame_size": fs} if fs else None)
                c = classify(r)
                counts[c] += 1
                if r["ok"]:
                    times.append(r["seconds"])
            med = sorted(times)[len(times) // 2] if times else float("nan")
            print(f"{pct:>6} | " + " ".join(f"{counts[c]:>12}" for c in cats)
                  + f" | {med:8.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
