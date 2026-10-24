#!/usr/bin/env python3
"""Local Asterisk two-DCE V.23 gates at 600/75 and 1200/75.

Each call uses the existing modem instances and Asterisk extension 110. The
answerer arms ATS0, the caller dials, and both exchange distinct exact payloads
simultaneously. This is a local lab test; it does not dial a provider.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


LAB = Path(__file__).resolve().parent
DRIVER = "/app/two_modem_duplex_throughput.py"


def run(cmd: list[str], *, timeout: float, log_path: Path | None = None) -> int:
    if log_path is None:
        return subprocess.run(cmd, cwd=LAB, timeout=timeout, check=False).returncode
    with log_path.open("w") as log:
        return subprocess.run(
            cmd, cwd=LAB, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        ).returncode


def wait_modems_healthy(timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    names = ("asterisk-modem-1", "asterisk-modem2-1")
    states = ["unknown", "unknown"]
    while time.monotonic() < deadline:
        states = []
        for name in names:
            probe = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", name],
                text=True, capture_output=True, check=False,
            )
            states.append(probe.stdout.strip() if probe.returncode == 0 else "missing")
        if states == ["healthy", "healthy"]:
            return
        time.sleep(1)
    raise TimeoutError(f"modem containers did not become healthy: {states}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", choices=(600, 1200),
                        default=(600, 1200))
    parser.add_argument("--bytes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=6200)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    try:
        # Keep the modem daemon in its supported automatic startup mode. The
        # driver selects V.23 through AT+MS separately on each call.
        subprocess.run(
            ["docker", "compose", "up", "-d", "--no-deps", "--force-recreate",
             "modem", "modem2"],
            cwd=LAB, env={**os.environ, "LINE_MODE": "ms"}, check=True,
        )
        wait_modems_healthy()
        for rate in args.rates:
            name = f"v23-{rate}"
            common = ["python3", DRIVER, "--bytes", str(args.bytes),
                      "--seed", str(args.seed + rate), "--expected-rate", str(rate),
                      "--v23-rate", str(rate), "--error-control", "raw",
                      "--connect-timeout", "40", "--transfer-timeout", "180"]
            answer = ["docker", "compose", "exec", "-T", "modem2", *common,
                      "/tmp/modem1", "--role", "answer"]
            caller = ["docker", "compose", "exec", "-T", "modem", *common,
                      "/tmp/modem0", "--role", "originate", "--dial", "110"]
            answer_log = output / f"{name}-answer.log"
            caller_log = output / f"{name}-caller.log"
            with answer_log.open("w") as answer_stream:
                answer_proc = subprocess.Popen(
                    answer, cwd=LAB, stdout=answer_stream,
                    stderr=subprocess.STDOUT,
                )
                time.sleep(2)
                try:
                    caller_rc = run(caller, timeout=240, log_path=caller_log)
                except subprocess.TimeoutExpired:
                    caller_rc = 124
                try:
                    answer_rc = answer_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    answer_proc.kill()
                    answer_rc = answer_proc.wait()
            result = {
                "rate": rate,
                "bytes_each_way": args.bytes,
                "caller_exit": caller_rc,
                "answer_exit": answer_rc,
                "status": "PASS" if caller_rc == 0 and answer_rc == 0 else "FAIL",
            }
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(name, result["status"], flush=True)
            if result["status"] != "PASS":
                return 1
        return 0
    finally:
        # Restore the lab modems to their normal automode configuration even
        # if a test times out or fails.
        subprocess.run(
            ["docker", "compose", "up", "-d", "--no-deps", "--force-recreate",
             "modem", "modem2"],
            cwd=LAB, env={**os.environ, "LINE_MODE": "ms"},
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
