"""No-call native-serial/control checks against an explicitly selected executable.

The harness uses source supervisor IPC and the actual frozen modem workers.
It does not qualify TUI rendering or connected-line performance.
"""
import argparse
import array
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'lab' / 'asterisk'))
from modem_supervisor import Supervisor


def run(executable, output):
    executable = executable.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(output / 'config.json')
    supervisor.acquire()
    transcript = {}
    checks = []
    processes = []

    def poll():
        supervisor.poll()
        for ident, data in supervisor.events:
            transcript.setdefault(ident, bytearray()).extend(bytes.fromhex(data))
        supervisor.events.clear()
        for ident, state in supervisor.states.items():
            if state.get('state') == 'failed':
                raise RuntimeError(f'{ident}: {state}')

    def wait(predicate, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            poll()
            if predicate():
                return
            time.sleep(.02)
        raise TimeoutError('bounded control wait expired')

    def request(ident, action, ok=True, **fields):
        key = supervisor.request(ident, action, **fields)
        wait(lambda: key in supervisor.responses)
        response = supervisor.responses.pop(key)
        if response['ok'] != ok:
            raise AssertionError(response)
        return response.get('value')

    def start(ident):
        with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', str(executable)):
            supervisor.start(ident)
        processes.append(supervisor.workers[ident][0])
        wait(lambda: supervisor.states[ident].get('state') == 'idle', 120)

    def settings(ident):
        return {entry['name']: entry['value'] for entry in request(ident, 'settings')}

    report = dict(status='FAIL', external_calls=0,
                  artifact_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  scope='frozen workers with source supervisor; no TUI or calls', checks=checks)
    try:
        first = supervisor.add(serial='driver')
        second = supervisor.add(serial='driver')
        first.sip_port, first.media_port = 27202, 27400
        second.sip_port, second.media_port = 27204, 27404
        supervisor.save()
        for spec in (first, second):
            start(spec.id)
        assert processes[0].pid != processes[1].pid
        endpoints = [supervisor.states[spec.id]['serial'] for spec in (first, second)]
        assert endpoints[0] != endpoints[1]
        report['serial_endpoints'] = endpoints
        checks.append('independent processes and native serial devices')
        original_second = settings(second.id)
        request(first.id, 'set', name='sip.user', value='control-test')
        request(first.id, 'set', name='iax.user', value='stored-iax-test')
        request(first.id, 'save')
        assert settings(second.id) == original_second
        checks.append('shared settings API and per-worker SIP/IAX profile isolation')
        request(first.id, 'terminal.open')
        try:
            fd = os.open(endpoints[0], os.O_RDWR | os.O_NONBLOCK | os.O_NOCTTY)
        except OSError as exc:
            assert exc.errno == errno.EBUSY
        else:
            os.close(fd)
            raise AssertionError('native terminal did not enforce exclusive ownership')
        def command(body, filename):
            transcript[first.id] = bytearray()
            request(first.id, 'terminal.write', hex=body.hex())
            wait(lambda: b'OK' in transcript[first.id] or b'ERROR' in transcript[first.id])
            data = bytes(transcript[first.id])
            (output / filename).write_bytes(data)
            assert b'OK' in data and b'ERROR' not in data, data
            return data

        assert b'V34,1,300,33600' in command(b'AT+MS?\r', 'hayes-ms.raw')
        command(b'ATS0=2\r', 'hayes-s0.raw')
        assert settings(first.id)['s0'] == 2
        request(first.id, 'set', name='s0', value=0)
        assert b'\r\n0\r\n' in command(b'ATS0?\r', 'hayes-s0-query.raw')
        command(b'AT+CFG="iax.user","HayesStoredIax"\r', 'hayes-cfg.raw')
        assert settings(first.id)['iax.user'] == 'HayesStoredIax'
        request(first.id, 'save')
        checks.append('exclusive ASCII terminal and native factory Hayes profile')
        checks.append('ATS and AT+CFG share control settings in both directions')
        request(first.id, 'terminal.close')
        wait(lambda: supervisor.states[first.id].get('owner') == 'none')
        fd = os.open(endpoints[0], os.O_RDWR | os.O_NONBLOCK | os.O_NOCTTY)
        try:
            wait(lambda: supervisor.states[first.id].get('owner') == 'external')
            request(first.id, 'terminal.open', ok=False)
            bits = array.array('I', [0])
            fcntl.ioctl(fd, termios.TIOCMGET, bits, True)
            assert bits[0] & termios.TIOCM_DSR
            assert not bits[0] & (termios.TIOCM_CAR | termios.TIOCM_RNG)
            fcntl.ioctl(fd, termios.TIOCMBIC, array.array('I', [termios.TIOCM_DTR]))
            wait(lambda: supervisor.states[first.id]['lights'].get('TR') is False)
            fcntl.ioctl(fd, termios.TIOCMBIS, array.array('I', [termios.TIOCM_DTR]))
            wait(lambda: supervisor.states[first.id]['lights'].get('TR') is True)
            assert settings(first.id)['sip.user'] == 'control-test'
            tty.setraw(fd)
            attrs = termios.tcgetattr(fd)
            attrs[2] |= termios.CLOCAL | termios.CRTSCTS
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            request(first.id, 'set', name='modem.flow_control', value=3)
            termios.tcflush(fd, termios.TCIFLUSH)
            fcntl.ioctl(fd, termios.TIOCMBIC, array.array('I', [termios.TIOCM_RTS]))
            os.write(fd, b'AT\r')
            deadline = time.monotonic() + .3
            while time.monotonic() < deadline:
                poll()
                time.sleep(.01)
            assert not select.select([fd], [], [], 0)[0], 'modem output ignored low RTS'
            fcntl.ioctl(fd, termios.TIOCMBIS, array.array('I', [termios.TIOCM_RTS]))
            response = bytearray()
            deadline = time.monotonic() + 3
            while b'OK' not in response and time.monotonic() < deadline:
                poll()
                if select.select([fd], [], [], .02)[0]:
                    response.extend(os.read(fd, 4096))
            (output / 'native-rts-response.raw').write_bytes(response)
            assert b'OK' in response and b'ERROR' not in response
        finally:
            os.close(fd)
        checks.append('external app ownership, idle hardware lines and live DTR lamp')
        checks.append('external native RTS pauses modem output and resumes Hayes response')
        survivor = supervisor.workers[second.id][0]
        supervisor.stop(first.id)
        assert survivor.poll() is None
        start(first.id)
        restored = settings(first.id)
        assert restored['sip.user'] == 'control-test'
        assert restored['iax.user'] == 'HayesStoredIax'
        checks.append('independent stop and save/restart persistence')
        report['status'] = 'PASS'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['states'] = supervisor.states
        raise
    finally:
        supervisor.close()
        report['cleanup'] = not supervisor.workers and all(proc.poll() is not None for proc in processes)
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['cleanup']:
        raise AssertionError('worker cleanup failed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.executable, arguments.output)
