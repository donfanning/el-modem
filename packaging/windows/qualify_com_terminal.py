"""Windows-only real COM/Tera Term check of the exact rc1 modem, no calls.

Run after probe_com_driver.ps1 on a disposable runner. Downloads third-party
programs to temporary storage; publishes only transcripts and identity reports.
"""
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
import zipfile


def download(path, urls, digest):
    errors = []
    for url in urls:
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('digest mismatch')
            path.write_bytes(data)
            return url
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError('Pinned download failed: ' + '; '.join(errors))


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(type=kind) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run():
    if os.name != 'nt':
        raise RuntimeError('Actual Windows serial driver required')
    report = Path('com-reports').resolve()
    report.mkdir(exist_ok=True)
    identities = {}
    processes = []
    handles = []
    outcome = {'passed': False}
    folder = None
    try:
        folder = tempfile.mkdtemp(prefix='el-modem-com-')
        root = Path(folder)
        packages = [
            ('modem', ['https://github.com/donfanning/el-modem/releases/download/v0.1.0-rc1/el-modem-0.1.0-rc1-windows-x86_64.zip'], '5766d971843b03dec3a665d4809c5072731829ad193da7fcc15365e0d1ec84b3'),
            ('terminal', ['https://github.com/TeraTermProject/teraterm/releases/download/v5.7.1/teraterm-5.7.1-x64.zip'], '68521f36a07754396eb925aabb0cd1f2092b15fffac74d330b428952552adafc'),
            ('bridge', [f'https://{host}/project/com0com/com2tcp/1.3.0.0/com2tcp-1.3.0.0-386.zip' for host in ('downloads.sourceforge.net', 'master.dl.sourceforge.net', 'phoenixnap.dl.sourceforge.net')], 'c13cd062187b84cdc25b5e774c8a45be83dad14721755d44c9bf9494ca460b8b'),
        ]
        for name, urls, digest in packages:
            path = root / (name + '.zip')
            url = download(path, urls, digest)
            identities[name] = {'url': url, 'archive_sha256': digest}
            with zipfile.ZipFile(path) as z:
                z.extractall(root / name)
        exe = next((root / 'modem').rglob('el-modem.exe'))
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        if digest != 'f98372c76cb648e1faaf92c925659ab8e6ba065b7e5fd249230ccf3fc82cd7a7':
            raise RuntimeError('Wrong modem executable')
        identities['modem']['executable_sha256'] = digest
        macro = next((root / 'terminal').rglob('ttpmacro.exe'))
        bridge = next((root / 'bridge').rglob('com2tcp.exe'))
        port = free_port()
        config = root / 'config.json'
        config.write_text(json.dumps({'schema': 1, 'modems': [{'id': 'modem-1', 'name': 'COM qualification', 'sip_host': '127.0.0.1', 'sip_port': free_port(socket.SOCK_DGRAM), 'media_host': '127.0.0.1', 'media_port': free_port(socket.SOCK_DGRAM), 'serial': 'tcp', 'serial_port': port, 'enabled': True}]}))
        def start(argv, name):
            log = (report / (name + '.log')).open('wb')
            handles.append(log)
            proc = subprocess.Popen(argv, stdout=log, stderr=log, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            processes.append(proc)
            return proc
        modem = start([str(exe), '--headless', '--config', str(config)], 'modem')
        deadline = time.monotonic() + 120
        while True:
            if modem.poll() is not None:
                raise RuntimeError('Modem exited before DTE ready')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('DTE startup timeout')
                time.sleep(.2)
        start([str(bridge), r'\\.\COM42', '127.0.0.1', str(port)], 'bridge')
        time.sleep(2)
        transcript = report / 'terminal.log'
        marker = report / 'terminal-success.txt'
        ttl = root / 'check.ttl'
        lines = ["connect '/C=41 /BAUD=115200'", 'timeout = 15', f"logopen '{transcript}' 0 0"]
        # Real application waits for responses from the modem through COM41.
        for command, expected in [('ATZ', 'OK'), ('ATE0', 'OK'), ('ATI3', 'donfanning/el-modem'), ('ATS0=9', 'OK'), ('ATS0?', '9'), ('ATZ', 'OK'), ('ATE0', 'OK'), ('ATS0?', '0')]:
            lines += [f"sendln '{command}'", f"wait '{expected}'", 'if result = 0 goto failed']
            if expected != 'OK':
                lines += ["wait 'OK'", 'if result = 0 goto failed']
        lines += [f"fileopen h '{marker}' 0", "filewriteln h 'PASS: Hayes identity, S-register readback and reset through actual COM/Tera Term'", 'fileclose h', 'logclose', 'closett', 'end', ':failed', 'logclose', 'closett', 'end']
        ttl.write_text('\n'.join(lines) + '\n')
        terminal = start([str(macro), '/V', str(ttl)], 'macro')
        terminal.wait(timeout=150)
        if not marker.exists():
            raise RuntimeError('Tera Term macro did not complete command/readback gates')
        text = transcript.read_text(errors='replace')
        if 'donfanning/el-modem' not in text or not re.search(r'(?m)^9\nOK$', text) or not re.search(r'(?m)^0\nOK$', text):
            raise RuntimeError('Terminal transcript failed independent identity/readback checks')
        modem.send_signal(signal.CTRL_BREAK_EVENT)
        modem.wait(timeout=20)
        if modem.returncode != 0:
            raise RuntimeError('Modem shutdown was not clean')
        outcome = {'passed': True, 'scope': 'Exact published rc1 TCP DTE through com0com/com2tcp and real Tera Term serial command path; no DSP call, modem-control signal or oldest-OS claim'}
    except Exception as exc:
        outcome = {'passed': False, 'error': str(exc)}
        raise
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
                proc.wait(timeout=20)
        for handle in handles:
            handle.close()
        if folder is not None:
            shutil.rmtree(folder)
        outcome['cleanup'] = [{'pid': proc.pid, 'exit_code': proc.poll()} for proc in processes]
        if any(proc.poll() is None for proc in processes):
            raise RuntimeError('Owned process remains alive')
        (report / 'terminal-identities.json').write_text(json.dumps(identities, indent=2) + '\n')
        (report / 'terminal-result.json').write_text(json.dumps(outcome, indent=2) + '\n')


if __name__ == '__main__':
    run()
