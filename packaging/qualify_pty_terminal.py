"""Qualify a frozen modem through actual minicom and its OS PTY; no calls.

Python types into minicom's UI terminal only. It never opens or writes the
modem's serial endpoint. Each published executable must match its pinned hash.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import pty
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import termios
import time
import tty
import uuid

HASHES = {
    'macos-x86_64': '9cf4a1d863c6c780ab770d7c90e814c8cbe254909476b9040957ac3e217de18f',
    'macos-arm64': 'e85818e067caba2983a07b911bdbad66e77ad1c488a5c9f064d8b8286ddf4096',
    'linux-development': '836a8bf8a9959aefec5a227a31185f7ecfe43578188271cd7d36aa830a158afc',
}


def port():
    with socket.socket(type=socket.SOCK_DGRAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def terminal_session(endpoint, root, name):
    app = shutil.which('minicom')
    if not app:
        raise RuntimeError('Install actual minicom first')
    config_name = 'elmodem-qualification-' + uuid.uuid4().hex
    config = Path.home() / ('.minirc.' + config_name)
    capture = root / (name + '-capture.txt')
    config.write_text('pu rtscts No\npu xonxoff No\n')
    config.chmod(0o600)
    master, slave = pty.openpty()
    tty.setraw(slave)
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 100, 0, 0))
    def session():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    proc = None
    data = bytearray()
    def read_until(needle, begin=0, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if needle in data[begin:]:
                return
            if proc.poll() is not None:
                raise RuntimeError('minicom exited before expected response: ' + needle.decode())
            ready, _, _ = select.select([master], [], [], .1)
            if ready:
                data.extend(os.read(master, 16384))
                if len(data) > 1024 * 1024:
                    raise RuntimeError('Unbounded terminal UI output')
        raise RuntimeError('minicom response timeout: ' + needle.decode())
    try:
        env = dict(os.environ, TERM='xterm-256color', LC_ALL='C')
        env.pop('MINICOM', None)
        argv = [app, '-o', '-D', str(endpoint), '-b', '115200', '-8', '-C', str(capture), config_name]
        proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, preexec_fn=session, env=env)
        os.close(slave); slave = None
        read_until(b'Press CTRL-A')
        commands = [('ATZ', b'OK'), ('ATE0', b'OK'), ('ATI3', b'donfanning/el-modem'), ('ATS0=9', b'OK'), ('ATS0?', b'9'), ('ATZ', b'OK'), ('ATE0', b'OK'), ('ATS0?', b'0')]
        for command, expected in commands:
            start = len(data)
            os.write(master, command.encode() + b'\r')
            read_until(expected, start)
            if expected != b'OK':
                read_until(b'OK', start)
        quit_start = len(data)
        os.write(master, b'\x01q')
        read_until(b'Leave without reset', quit_start, seconds=5)
        os.write(master, b'\r')
        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], .1)
            if ready:
                try:
                    data.extend(os.read(master, 16384))
                except OSError:
                    time.sleep(.05)
        proc.wait(timeout=1)
        if proc.returncode != 0:
            raise RuntimeError('minicom exit status ' + str(proc.returncode))
        text = capture.read_text(errors='replace')
        if 'donfanning/el-modem' not in text or not re.search(r'(?m)^9\nOK$', text) or not re.search(r'(?m)^0\nOK$', text):
            raise RuntimeError('Independent serial transcript identity/readback check failed')
        return {'application': app, 'argv': argv, 'exit_code': proc.returncode, 'capture_sha256': hashlib.sha256(capture.read_bytes()).hexdigest(), 'commands': [x[0] for x in commands]}
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        os.close(master)
        if slave is not None:
            os.close(slave)
        config.unlink()
        (root / (name + '-ui.txt')).write_bytes(data)


def qualify(executable, target, root):
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    if digest != HASHES[target]:
        raise RuntimeError('Executable differs from published pinned artifact')
    if target.startswith('macos') and platform.system() != 'Darwin':
        raise RuntimeError('Actual macOS required for macOS qualification')
    expected_cpu = {'macos-x86_64': 'x86_64', 'macos-arm64': 'arm64'}.get(target)
    if expected_cpu and platform.machine() != expected_cpu:
        raise RuntimeError('Wrong native architecture')
    root.mkdir(parents=True, exist_ok=True)
    result = {'status': 'FAIL', 'target': target, 'executable_sha256': digest, 'platform': platform.platform(), 'machine': platform.machine(), 'scope': 'Actual minicom → OS PTY → frozen supervisor/worker, Hayes commands, terminal close/reopen and clean teardown. No DSP/PSTN call or hardware RS232 signals.'}
    proc = None
    try:
        with tempfile.TemporaryDirectory(prefix='el-modem-pty-') as folder:
            work = Path(folder)
            sip, media = port(), port()
            while media == sip:
                media = port()
            config = work / 'config.json'
            config.write_text(json.dumps({'schema': 1, 'modems': [{'id': 'modem-1', 'name': 'PTY qualification', 'sip_host': '127.0.0.1', 'sip_port': sip, 'media_host': '127.0.0.1', 'media_port': media, 'serial': 'pty', 'serial_port': 9100, 'enabled': True}]}))
            endpoint = work / 'instances/modem-1/serial'
            with (root / 'supervisor.log').open('wb') as log:
                proc = subprocess.Popen([str(executable.resolve()), '--headless', '--config', str(config)], stdout=log, stderr=log, start_new_session=True)
                try:
                    deadline = time.monotonic() + 120
                    while not endpoint.exists():
                        if proc.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('Frozen worker did not publish PTY endpoint')
                        time.sleep(.1)
                    real_endpoint = os.readlink(endpoint)
                    result['pty_endpoint'] = real_endpoint
                    result['minicom_version'] = subprocess.check_output(['minicom', '--version'], text=True).strip()
                    result['sessions'] = [terminal_session(endpoint, root, 'first')]
                    if proc.poll() is not None or os.readlink(endpoint) != real_endpoint:
                        raise RuntimeError('Worker/PTY did not survive terminal close')
                    result['sessions'].append(terminal_session(endpoint, root, 'reopened'))
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=20)
                    if proc.returncode != 0 or endpoint.exists() or endpoint.is_symlink():
                        raise RuntimeError('Unclean worker shutdown or remaining serial link')
                    result['cleanup'] = {'supervisor_exit': proc.returncode, 'serial_link_removed': True}
                finally:
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=20)
                    worker_log = work / 'instances/modem-1/worker.log'
                    if worker_log.exists():
                        shutil.copyfile(worker_log, root / 'worker.log')
                        if 'Traceback' in worker_log.read_text(errors='replace'):
                            raise RuntimeError('Worker traceback detected')
        result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--target', choices=HASHES, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.executable, args.target, args.reports.resolve())
