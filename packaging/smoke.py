"""Built-artifact engine/payload gates, then an AT-only TCP serial smoke check.

The artifact gates never dial or start SIP/RTP. The serial smoke opens local
listeners only; it sends no ATD/ATA. Failures retain the executable's output.
"""
import argparse
import json
import os
import hashlib
from pathlib import Path
import socket
import subprocess
import time


def offline(exe, payload=True):
    option = '--self-test-payload' if payload else '--self-test'
    result = subprocess.run([exe, option], capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError('artifact self-test failed:\n' + result.stdout + result.stderr)
    report = json.loads(result.stdout)
    if report['status'] != 'PASS' or (payload and not report['payload_requested']):
        raise RuntimeError('incomplete artifact self-test: ' + result.stdout)
    return report


def serial_smoke(exe):
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0)); port = reserve.getsockname()[1]
    environment = {k: v for k, v in os.environ.items() if not k.startswith('MODEM_CAPTURE')
                   and k not in ('EL_MODEM_DSP_CAPTURE', 'EL_MODEM_METRICS_FILE', 'V32_CAPTURE')}
    proc = subprocess.Popen([exe, '--serial-tcp', f'127.0.0.1:{port}', '--line-mode', 'ms', '--profile-file', ''],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=environment)
    connection = None
    try:
        for _ in range(100):
            if proc.poll() is not None:
                raise RuntimeError('modem exited before listening: ' + proc.stderr.read().decode(errors='replace'))
            try:
                connection = socket.create_connection(('127.0.0.1', port), timeout=1)
                break
            except OSError:
                time.sleep(.2)
        if connection is None:
            raise RuntimeError('modem never listened')
        connection.settimeout(3)
        def at(command):
            connection.sendall(command)
            data = bytearray()
            while not data.endswith(b'OK\r\n') and b'ERROR' not in data:
                block = connection.recv(4096)
                if not block:
                    raise RuntimeError('serial connection closed')
                data.extend(block)
            return bytes(data)
        assert at(b'ATZ\r').endswith(b'OK\r\n')
        assert b'https://github.com/donfanning/el-modem' in at(b'ATI3\r')
        assert at(b'AT+MS=V34,1,300,33600\r').endswith(b'OK\r\n')
        assert b'+MS: V34,1,300,33600' in at(b'AT+MS?\r')
        assert at(b'ATZ\r').endswith(b'OK\r\n')
        assert b'+MS: V32B,1,300,14400' in at(b'AT+MS?\r')
    finally:
        if connection is not None:
            connection.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)
        proc.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable')
    parser.add_argument('--factories-only', action='store_true', help='skip local PCM payload gates')
    parser.add_argument('--report', help='retain artifact gate report as JSON')
    args = parser.parse_args()
    report = offline(args.executable, not args.factories_only)
    if not report.get('build_manifest'):
        raise RuntimeError('built executable did not retain its source manifest')
    report['artifact_sha256'] = hashlib.sha256(Path(args.executable).read_bytes()).hexdigest()
    serial_smoke(args.executable)
    report['serial_at_smoke'] = 'PASS'
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(f"artifact smoke passed: {len(report['gates'])} offline gates and serial AT profile/reset")


if __name__ == '__main__':
    main()
