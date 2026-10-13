#!/usr/bin/env python3
"""Call-level retry ladder for a handshake-only real call: try the best rate, and if the call produced no readable data, hang up
(real_call.sh always does) and redial one rung lower. Needed because an answerer such as NIST ACTS that has been offered 2400
(S1) runs a timer-driven V.22bis sequence nothing we transmit can change, so only a new call at 1200 or 300 works
(lab/V34_NIST_ACTS_CALLS_2026_10_07.md).

    python3 real_call_ladder.py NAME NUMBER [--start N] [--max-calls N] [--secs N]

Every rung is ONE real call through ``real_call.sh NAME-rungK NUMBER`` (standard hang-up, captures kept). The ladder stops at the
first rung whose post-CONNECT bytes are readable text; ``--max-calls`` caps the number of calls (default 3 = every rung once).
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, AT init string, extra environment). Rung 1 is the automode chain up to V.22bis with the CONNECT hold (V22_BIS_SETTLE_S=3.5: a handshake restart inside the hold ends the call with NO CARRIER before CONNECT); rungs 2 and 3 never offer S1.
RUNGS = (
    ("auto-2400", "ATE0X4", {"V22_BIS_SETTLE_S": "3.5"}),
    ("bell212a-1200", "ATE0X4+MS=B212,0,1200,1200", {}),
    ("bell103-300", "ATE0X4+MS=B103,0,300,300", {}),
)
LINE = re.compile(r"^\[\s*[\d.]+s\] (b'.*'|b\".*\")\s*$")
MIN_BYTES, MIN_PRINTABLE = 60, 0.90


def post_connect_bytes(output: str) -> tuple[bool, bytes]:
    """The bytes the probe printed after ``CONNECT`` (and whether a CONNECT was seen)."""
    connected, data = False, bytearray()
    for line in output.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        try:
            chunk = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            continue
        if connected:
            data += chunk
        elif b"CONNECT" in chunk:
            connected = True
            data += chunk.split(b"CONNECT", 1)[1].split(b"\n", 1)[-1] if b"\n" in chunk.split(b"CONNECT", 1)[1] else b""
    return connected, bytes(data)


def judge(output: str) -> dict:
    """Success means a CONNECT followed by enough readable text; anything else (no CONNECT, NO CARRIER, binary noise) is a failure."""
    connected, data = post_connect_bytes(output)
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in data) / len(data) if data else 0.0
    ok = connected and len(data) >= MIN_BYTES and printable >= MIN_PRINTABLE
    return dict(connected=connected, bytes=len(data), printable=round(printable, 3), ok=ok)


def run_ladder(name: str, number: str, runner, start: int = 1, max_calls: int = 3, secs: int = 40) -> list[dict]:
    """``runner(rung_name, number, init, env, secs) -> output text``. Returns one result per call placed."""
    results = []
    for index, (label, init, env) in enumerate(RUNGS[start - 1:], start=start):
        if len(results) >= max_calls:
            break
        output = runner(f"{name}-rung{index}", number, init, env, secs)
        verdict = judge(output)
        results.append(dict(rung=index, label=label, init=init, **verdict))
        if verdict["ok"]:
            break
    return results


def shell_runner(rung_name, number, init, env, secs) -> str:
    out_dir = os.environ.get("MODEM_ARTIFACT_DIR", "/tmp/modem-artifacts")
    environ = dict(os.environ, LINE_MODE="ms", PROBE_SECS=str(secs), PROBE_INIT=init)
    proc = subprocess.run(["sh", str(HERE / "real_call.sh"), rung_name, number, *[f"{k}={v}" for k, v in env.items()]],
                          capture_output=True, text=True, env=environ, cwd=HERE)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{rung_name}.out").write_text(proc.stdout + proc.stderr)
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("number")
    ap.add_argument("--start", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--max-calls", type=int, default=3)
    ap.add_argument("--secs", type=int, default=40)
    args = ap.parse_args()
    results = run_ladder(args.name, args.number, shell_runner, args.start, args.max_calls, args.secs)
    print(json.dumps(results, indent=1))
    return 0 if results and results[-1]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
