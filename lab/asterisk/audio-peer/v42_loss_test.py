#!/usr/bin/env python3
"""One side of a two-modem transfer test, for measuring V.42 vs raw under network
impairment. Run the answerer first, then the originator, e.g. inside the two
modem containers (both modems started with --line-mode v22bis):

    python3 v42_loss_test.py /tmp/modem1 --role answer   --bytes 1500 --error-control 3
    python3 v42_loss_test.py /tmp/modem0 --role originate --bytes 1500 --error-control 3

The originator sends a random payload; the answerer checks it byte for byte and
replies with a short marker; the originator reports how long the whole exchange
took. Both sides hang up with the standard guard-time +++ / ATH0 sequence.
"""

import argparse
import os
import random
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pstn_probe import Line, hang_up  # noqa: E402

START, END = b"\x02START\x03", b"\x02END\x03"
DONE_OK, DONE_FAIL = b"XFER-OK", b"XFER-FAIL"


def collect_until(line, token, seconds):
    buf, end = bytearray(), time.time() + seconds
    while time.time() < end and token not in buf:
        buf += line.read(0.5)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pty")
    ap.add_argument("--role", choices=["originate", "answer"], required=True)
    ap.add_argument("--bytes", type=int, default=1500)
    ap.add_argument("--error-control", type=int, choices=[0, 1, 3], default=3)
    ap.add_argument("--dial", default="110")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", help="answerer: write the raw bytes received to this file")
    args = ap.parse_args()

    line = Line(args.pty)
    online = False
    ok = False
    try:
        line.read(0.5)
        line.send(f"ATE0\\N{args.error_control}\r".encode())
        line.wait_for((b"OK",), 5)
        if args.role == "answer":
            line.send(b"ATS0=1\r")
            line.wait_for((b"OK",), 5)
        else:
            line.send(f"ATDT{args.dial}\r".encode())
        connect = line.wait_for((b"CONNECT", b"NO CARRIER", b"BUSY"), 90)
        text = connect.decode("latin1").strip().replace("\r\n", " ")
        print(f"{args.role}: {text}", flush=True)
        online = b"CONNECT" in connect
        if not online:
            return 2
        payload = random.Random(args.seed).randbytes(args.bytes)
        if args.role == "originate":
            time.sleep(0.5)
            t0 = time.monotonic()
            line.send(START + payload + END)
            reply = collect_until(line, b"XFER-", args.timeout)
            took = time.monotonic() - t0
            ok = DONE_OK in reply  # the answerer says whether it got every byte
            print(f"originate: transfer {'OK' if ok else 'FAILED'}: {args.bytes} bytes, "
                  f"{took:.1f}s -> {args.bytes * 8 / took:.0f} bit/s of user data", flush=True)
        else:
            data = collect_until(line, END, args.timeout)
            if args.dump:
                with open(args.dump, "wb") as fh:
                    fh.write(data)
            got = data[data.find(START) + len(START): data.rfind(END)] if START in data and END in data else b""
            ok = got == payload
            print(f"answer: received {len(got)}/{args.bytes} bytes, "
                  f"{'byte-exact' if ok else 'MISMATCH or incomplete'}", flush=True)
            line.send(DONE_OK if ok else DONE_FAIL)
            time.sleep(1.0)
    finally:
        confirmed = hang_up(line, online)
        print(f"{args.role}: hangup confirmed: {confirmed}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
