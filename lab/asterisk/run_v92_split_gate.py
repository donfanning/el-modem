#!/usr/bin/env python3
"""Run the paired V.92 cold-start gate with two independent modem_dce processes.

This uses direct SIP/RTP on the host network, not Asterisk or a PSTN route. It
requires the local `el-modem-v92-split-gate:20261113` image built from the
current checkout. Artifacts are written outside the repository.
"""
import argparse
import hashlib
import json
import pathlib
import random
import select
import socket
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "lab/asterisk/profiles"
IMAGE = "el-modem-v92-split-gate-native:20261113"
ANSWER_NAME = "v92-split-answer-20261113"
CALL_NAME = "v92-split-caller-20261113"


class GateError(RuntimeError):
    pass


def docker_start(out, name, role, cpu, serial, sip, media, metrics, profile,
                 profile_option, host="127.0.0.1", sip_args=(), jitter_ms=40):
    args = [
        "docker", "run", "-d", "--name", name, "--network", "host",
        "--cpuset-cpus", str(cpu),
        "-e", "PYTHONPATH=/src:/src/audio-peer",
        "-e", "OPENBLAS_NUM_THREADS=1",
        "-e", "EL_MODEM_FACTORY_NATIVE=1",
        "-e", f"EL_MODEM_RTP_JITTER_MS={jitter_ms}",
        "-e", "EL_MODEM_DSP_TRAINING_STRIDE=8",
        "-e", "V34_PHASE4_EQ_SPAN=24",
        "-e", "V34_RATE=33600",
        "-e", "V34_SYMS=3429",
        "-e", f"NUMBA_CACHE_DIR=/captures/numba-cache-{role}",
        # V.92 lab runs always retain the TX-stall stack and the first two
        # exact outbound CP bit strings. These are bounded diagnostics and
        # are needed to distinguish a frozen media path from a wire-frame
        # defect; neither changes modem signalling.
        "-e", "EL_MODEM_TIMING_TRACE=1",
        "-e", "EL_MODEM_TX_WATCHDOG=1",
        "-e", "EL_MODEM_TX_WATCHDOG_THRESHOLD_MS=250",
        "-e", "EL_MODEM_V92_CAPTURE_CP_BITS=1",
        "-e", "MODEM_DEBUG=1", "-e", "V42_DEBUG=1",
        # Opt-in digital Phase 2 event trace, passed through when set.
        "-e", "EL_MODEM_V90_PHASE2_TRACE",
        # Opt-in digital-initiated V.92 renegotiation test trigger.
        "-e", "EL_MODEM_TEST_V92_RENEGOTIATE_AFTER_MS",
        "-e", "EL_MODEM_TEST_V92_RENEGOTIATE_DRN_UP",
        "-e", "EL_MODEM_TEST_V92_RENEGOTIATE_KIND",
        "-e", "EL_MODEM_TEST_V92_RENEGOTIATE_SILENT_PERIOD",
        "-e", "EL_MODEM_TEST_V92_RENEGOTIATE_EARLY_RT_AFTER",
        "-v", f"{out}:/captures",
        "-v", f"{out / 'runtime-source'}:/src:ro",
        "-v", f"{profile}:/profiles/profile.json:ro",
        "--entrypoint", "python3", IMAGE, "/src/modem_dce.py",
        "--serial-tcp", f"127.0.0.1:{serial}", "--line-mode", "chain",
        "--v34-native-trellis",
        "--sip-host", host, "--sip-port", str(sip),
        "--media-host", host, "--media-port", str(media),
        "--timing-trace", "--metrics-file", f"/captures/{metrics}",
        "--capture", f"/captures/{role}.pcm", "--capture-seconds", "180",
    ]
    args += [profile_option, "/profiles/profile.json"]
    if sip_args:
        # Relay mode: the password reaches the container from this process's
        # environment, never the command line.
        args[args.index("--entrypoint"):args.index("--entrypoint")] = [
            "-e", "EL_MODEM_SIP_PASSWORD"]
        args += list(sip_args)
    elif role == "caller":
        args += ["--dial-host", "127.0.0.1", "--dial-port", "15060"]
    cp = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=30)
    if cp.returncode:
        raise GateError(f"{role} container start failed: {cp.stdout}")
    print(f"{role} container {cp.stdout.strip()[:12]} on CPU {cpu}", flush=True)


def tcp_ready(port, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = subprocess.run(["ss", "-ltn"], text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        if f":{port} " in result.stdout or f":{port}\n" in result.stdout:
            return
        time.sleep(.1)
    raise GateError(f"serial TCP port {port} did not listen")


def connect_dte(port):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            sock.settimeout(.5)
            return sock
        except OSError:
            time.sleep(.1)
    raise GateError(f"cannot connect serial TCP port {port}")


def read_until(sock, tokens, timeout=15, initial=b""):
    if isinstance(tokens, bytes):
        tokens = [tokens]
    data = bytearray(initial)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(token in data for token in tokens):
            return bytes(data)
        try:
            chunk = sock.recv(4096)
            if not chunk:
                raise GateError("DTE connection closed")
            data.extend(chunk)
        except socket.timeout:
            pass
    raise GateError(f"timed out waiting for {tokens}; received {bytes(data)[-300:]!r}")


def command(sock, text):
    sock.sendall(text.encode("ascii") + b"\r")
    return read_until(sock, b"OK", timeout=8)


def payload(seed, length=65536):
    rng = random.Random(seed)
    data = bytearray()
    while len(data) < length:
        data.extend(rng.randbytes(1024))
    return bytes(data[:length]).replace(b"+++", b"+\x00+")


def paced_send(sock, data, rate=6000, chunk=1000):
    started = time.monotonic()
    for offset in range(0, len(data), chunk):
        delay = started + offset / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        sock.sendall(data[offset:offset + chunk])
    return len(data)


def summarize(path):
    if not path.exists():
        return {"missing": str(path)}
    obj = json.loads(path.read_text())
    trace = obj.get("timing_trace", {})
    events = trace.get("events", {})
    event_summary = {}
    for name, item in events.items():
        worst = item.get("worst", {})
        event_summary[name] = {
            "count": item.get("count"),
            "worst_ms": round(1000 * worst.get("score", 0), 3),
            "row": worst.get("row", {}),
        }
    measurements = obj.get("realtime", {}).get("measurements", {})
    rx_max = max((measurements.get(key, {}).get("maximum", 0.0)
                  for key in ("startup_rx_seconds", "data_rx_seconds")),
                 default=0.0)
    tx_gap_max = max((measurements.get(key, {}).get("maximum", 0.0)
                      for key in ("startup_tx_packet_gap_seconds",
                                  "data_tx_packet_gap_seconds")),
                     default=0.0)
    rt_gate = {
        "rx_max_ms": round(rx_max * 1000, 3),
        "tx_packet_gap_max_ms": round(tx_gap_max * 1000, 3),
        "limit_ms": 40.0,
        "pass": (rx_max > 0 and tx_gap_max > 0
                 and rx_max <= .040 and tx_gap_max <= .040),
    }
    return {
        "realtime_budget": rt_gate,
        "realtime": obj.get("realtime", {}),
        "timing_trace_counts": trace.get("counts"),
        "timing_trace_thresholds_seconds": trace.get("thresholds_seconds"),
        "events": event_summary,
        "rtp": obj.get("rtp"),
        "v90_analogue_caller": obj.get("v90_analogue_caller"),
        "line_state": obj.get("line_state"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path.home() / "modem-artifacts" /
                        "v92-split-dce-gate-20261113")
    parser.add_argument("--reverse", action="store_true",
                        help="the dialling DCE is the digital modem and the answering DCE the analogue one")
    parser.add_argument("--base-image", default="el-modem-native-v90-slice-e8d",
                        help="Numba runtime image the gate layer is built on")
    parser.add_argument("--cpus", default="2,3", help="answerer,caller CPU pins")
    parser.add_argument("--relay", metavar="HOST",
                        help="route the call through this Asterisk (SIP 5060): the answerer registers as "
                             "EL_MODEM_SIP_USER and the caller dials --relay-number; EL_MODEM_SIP_PASSWORD from the environment")
    parser.add_argument("--relay-number", default="225",
                        help="relay extension that rings the registered answerer (never a PSTN number)")
    parser.add_argument("--jitter-ms", type=int, default=40,
                        help="receive jitter buffer of both DCEs (EL_MODEM_RTP_JITTER_MS)")
    parser.add_argument("--plain-v90", action="store_true",
                        help="offer V.90 only, without V.92 or PCM upstream")
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Rebuild a small, credential-free image layer on the already qualified
    # Numba runtime. Only Python source is copied into the layer; profiles and
    # all logs remain mounted separately below.
    runtime_source = out / "runtime-source"
    audio_source = runtime_source / "audio-peer"
    audio_source.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "lab/asterisk").glob("*.py"):
        (runtime_source / source.name).write_bytes(source.read_bytes())
    for source in (ROOT / "lab/asterisk/audio-peer").glob("*.py"):
        (audio_source / source.name).write_bytes(source.read_bytes())
    (runtime_source / "runtime-source-sha256.json").write_text(json.dumps({
        str(path.relative_to(runtime_source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime_source.rglob("*.py"))
    }, indent=2) + "\n")
    build_context = out / "build-context"
    build_context.mkdir(exist_ok=True)
    (build_context / "Dockerfile").write_text(
        f"FROM {args.base_image}\n"
        "COPY runtime-source /src\n"
        "ENV PYTHONPATH=/src:/src/audio-peer\n")
    # A separate build context contains no credentials, PBX configuration, or
    # recordings. Dockerfile COPY only sees the filtered runtime-source tree.
    import shutil
    staged_source = build_context / "runtime-source"
    if staged_source.exists():
        shutil.rmtree(staged_source)
    shutil.copytree(runtime_source, staged_source)
    build = subprocess.run(
        ["docker", "build", "--tag", IMAGE, "--file",
         str(build_context / "Dockerfile"), str(build_context)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "image-build.log").write_text(build.stdout)
    if build.returncode:
        raise GateError(f"gate image build failed; see {out / 'image-build.log'}")
    image_id = subprocess.run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    (out / "image-id.txt").write_text(image_id + "\n")
    print(f"built {IMAGE} ({image_id}) from filtered current-branch Python source",
          flush=True)

    # The live-like V.92 answerer fixture used in the previous split gate.
    answer_profile = {
        "info0d_fields": {"rate3429": 1, "v90_3429_upstream": 1},
        "info1d_fields": {"blocks": [{}, {}, {}, {}, {}, {"hi_carrier": 0}]},
        "tone_b_peak": 900, "probe_rms": 450,
        "phase3_rate_mask": 65535, "phase3_enable_lapm": True,
        "v92": True, "v92_pcm_upstream": True,
        "v92_fast_parameter_exchange": False,
        "phase2_l2_echo_window_policy": "advance-after-window",
    }
    caller_profile = json.loads(
        (PROFILE_DIR / "v92-pcm-upstream-analogue-caller.json").read_text())
    caller_profile["phase2_late_l2_echo_policy"] = "advance-after-window"
    if args.plain_v90:
        answer_profile.update(v92=False, v92_pcm_upstream=False)
        answer_profile.pop("v92_fast_parameter_exchange")
        caller_profile["v92_pcm_upstream"] = False
    digital_option = "--v90-digital-answerer-profile"
    analogue_option = "--v90-analogue-caller-profile"
    if args.reverse:
        answer_profile["call_roles"] = ["originate"]
        caller_profile["call_roles"] = ["answer"]
        answer_profile, caller_profile = caller_profile, answer_profile
        answer_option, caller_option = analogue_option, digital_option
    else:
        answer_option, caller_option = digital_option, analogue_option
    answer_profile_path = out / "answerer-profile.json"
    answer_profile_path.write_text(json.dumps(answer_profile, indent=2) + "\n")
    caller_profile_path = out / "caller-profile.json"
    caller_profile_path.write_text(json.dumps(caller_profile, indent=2) + "\n")

    # Fresh containers plus per-role empty cache directories make Numba cold.
    for role in ("answerer", "caller"):
        cache = out / f"numba-cache-{role}"
        if cache.exists():
            import shutil
            shutil.rmtree(cache)
        cache.mkdir()
    for name in (ANSWER_NAME, CALL_NAME):
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    answer_cpu, caller_cpu = (int(cpu) for cpu in args.cpus.split(","))
    host, answer_sip, caller_sip, number = "127.0.0.1", (), (), "110"
    if args.relay:
        import os
        import re
        if not re.fullmatch(r"\d{3}", args.relay_number):
            raise GateError("relay number must be a 3-digit lab extension")
        user = os.environ.get("EL_MODEM_SIP_USER")
        if not user or not os.environ.get("EL_MODEM_SIP_PASSWORD"):
            raise GateError("relay mode needs EL_MODEM_SIP_USER and EL_MODEM_SIP_PASSWORD")
        host = subprocess.run(["hostname", "-I"], text=True, stdout=subprocess.PIPE,
                              check=True).stdout.split()[0]
        answer_sip = ("--sip-register", f"{args.relay}:5060", "--sip-user", user)
        caller_sip = ("--dial-host", args.relay, "--dial-port", "5060", "--sip-user", user)
        number = args.relay_number
    answer = caller = None
    call_error = None
    try:
        docker_start(out, ANSWER_NAME, "answerer", answer_cpu, 19101, 15060, 16100,
                     "answerer-metrics.json", answer_profile_path, answer_option,
                     host, answer_sip, args.jitter_ms)
        tcp_ready(19101)
        docker_start(out, CALL_NAME, "caller", caller_cpu, 19100, 15062, 16102,
                     "caller-metrics.json", caller_profile_path, caller_option,
                     host, caller_sip, args.jitter_ms)
        tcp_ready(19100)
        time.sleep(1)
        answer, caller = connect_dte(19101), connect_dte(19100)
        print("answerer ATE0:", command(answer, "ATE0")[-100:], flush=True)
        print("answerer AT\\N3:", command(answer, "AT\\N3")[-100:], flush=True)
        print("answerer ATS0=1:", command(answer, "ATS0=1")[-100:], flush=True)
        print("caller ATE0:", command(caller, "ATE0")[-100:], flush=True)
        print("caller AT\\N3:", command(caller, "AT\\N3")[-100:], flush=True)
        if args.relay:
            time.sleep(3)          # let the answerer's REGISTER land first
        caller.sendall(f"ATD{number}\r".encode())
        started = time.monotonic()
        caller_result = read_until(caller, b"CONNECT", timeout=150)
        answer_result = read_until(
            answer, b"CONNECT", timeout=max(1, 150 - (time.monotonic() - started)))
        print("caller result:", caller_result[-300:], flush=True)
        print("answerer result:", answer_result[-300:], flush=True)

        caller_data, answer_data = payload(9113), payload(9114)
        caller.setblocking(False)
        answer.setblocking(False)
        received_by_answer, received_by_caller = bytearray(), bytearray()
        sent = {}
        caller_tx = threading.Thread(
            target=lambda: sent.setdefault("caller", paced_send(caller, caller_data)),
            daemon=True)
        answer_tx = threading.Thread(
            target=lambda: sent.setdefault("answerer", paced_send(answer, answer_data)),
            daemon=True)
        caller_tx.start()
        answer_tx.start()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and (
                len(received_by_answer) < len(caller_data)
                or len(received_by_caller) < len(answer_data)):
            ready, _, _ = select.select([caller, answer], [], [], .1)
            for sock in ready:
                try:
                    chunk = sock.recv(16384)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise GateError("DTE TCP closed during payload")
                (received_by_caller if sock is caller else received_by_answer).extend(chunk)
        caller_tx.join(3)
        answer_tx.join(3)
        payload_report = {
            "sent": sent,
            "caller_to_answerer_bytes": len(received_by_answer),
            "answerer_to_caller_bytes": len(received_by_caller),
            "caller_to_answerer_exact": bytes(received_by_answer) == caller_data,
            "answerer_to_caller_exact": bytes(received_by_caller) == answer_data,
            "caller_sha256_expected": hashlib.sha256(caller_data).hexdigest(),
            "caller_sha256_received": hashlib.sha256(received_by_answer).hexdigest(),
            "answerer_sha256_expected": hashlib.sha256(answer_data).hexdigest(),
            "answerer_sha256_received": hashlib.sha256(received_by_caller).hexdigest(),
        }
        (out / "payload-result.json").write_text(
            json.dumps(payload_report, indent=2) + "\n")
        print("payload:", json.dumps(payload_report), flush=True)
        if not payload_report["caller_to_answerer_exact"] or not payload_report[
                "answerer_to_caller_exact"]:
            raise GateError("payload mismatch or short transfer")

        time.sleep(1.2)
        caller.setblocking(True)
        caller.settimeout(1)
        caller.sendall(b"+++")
        escaped = read_until(caller, b"OK", timeout=8)
        caller.sendall(b"ATH0\r")
        hung = read_until(caller, b"OK", timeout=15)
        print("escape/hangup:", escaped[-120:], hung[-120:], flush=True)
        time.sleep(1)
    except Exception as exc:
        call_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sock in (caller, answer):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
        for name in (ANSWER_NAME, CALL_NAME):
            subprocess.run(["docker", "stop", "-t", "3", name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for name, role in ((ANSWER_NAME, "answerer"), (CALL_NAME, "caller")):
            logs = subprocess.run(["docker", "logs", name], text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
            (out / f"{role}-docker.log").write_text(logs.stdout)
            subprocess.run(["docker", "rm", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        report = {"error": call_error}
        for role in ("answerer", "caller"):
            report[role] = summarize(out / f"{role}-metrics.json")
        # The gate covers the analogue modem, whichever side dialled.
        analogue = "answerer" if args.reverse else "caller"
        analogue_gate = report[analogue].get("realtime_budget", {})
        report["analogue_realtime_gate_pass"] = bool(analogue_gate.get("pass"))
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("summary:", json.dumps(report, default=str)[:12000], flush=True)
        if call_error is None and not report["analogue_realtime_gate_pass"]:
            raise GateError(f"analogue {analogue} receive/TX-gap real-time gate exceeded 40 ms")


if __name__ == "__main__":
    main()
