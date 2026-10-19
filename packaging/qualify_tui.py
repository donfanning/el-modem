"""No-call smoke of the actual packaged TUI and its live modem worker.

Requires pyte in the harness interpreter only. Run in the matched Linux runtime
with --network none; the modem executable and its dependencies are unchanged.
"""
import argparse
import codecs
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import termios
import time

import pyte


def run(executable, output):
    executable = executable.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    config = output / 'config.json'
    config.write_text(json.dumps(dict(schema=1, modems=[dict(
        id='modem-1', name='UI qualification', serial='pty', enabled=True,
        sip_host='127.0.0.1', sip_port=27232, media_host='127.0.0.1',
        media_port=27432, serial_port=27234)])))
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 60, 180, 0, 0))
    process = subprocess.Popen([str(executable), '--tui', '--config', str(config)],
                               stdin=slave, stdout=slave, stderr=slave,
                               env=dict(os.environ, TERM='xterm-256color'))
    os.close(slave)
    screen = pyte.Screen(180, 60)
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    checks = []
    report = dict(status='FAIL', external_calls=0,
                  artifact_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  scope='actual frozen TUI, actual frozen worker, no calls', checks=checks)
    raw = (output / 'screen.raw').open('wb')

    def read(seconds=.3):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if select.select([master], [], [], min(.1, max(0, end-time.monotonic())))[0]:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    return
                raw.write(data)
                stream.feed(decoder.decode(data))

    def view():
        return '\n'.join(screen.display)

    def wait(predicate, seconds=15):
        deadline = time.monotonic()+seconds
        while time.monotonic()<deadline:
            read()
            if predicate():
                return
            if process.poll() is not None:
                raise RuntimeError(f'TUI exited {process.returncode}')
        raise TimeoutError('UI condition not observed')

    def save(name):
        (output / (name+'.txt')).write_text(view())

    def send(data):
        os.write(master, data)
        read(.4)

    def click_at(x, y):
        send(f'\x1b[<0;{x+1};{y+1}M\x1b[<0;{x+1};{y+1}m'.encode())

    def type_text(text):
        for ch in text:
            os.write(master, ch.encode())
            read(.04)
        read(.3)

    def select_modem_action(index):
        """Use the dashboard's visible pane/menu navigation, not old hotkeys."""
        send(b'\t')       # system menu -> modem stack
        send(b'\r')       # selected modem -> modem actions
        for _ in range(index):
            send(b'\x1b[B')  # Down through the modem action list
        send(b'\r')       # activate the marked action

    def click(text):
        for y,line in reversed(list(enumerate(screen.display))):
            x=line.find(text)
            if x>=0:
                x+=max(1,len(text)//2)
                send(f'\x1b[<0;{x+1};{y+1}M\x1b[<0;{x+1};{y+1}m'.encode())
                return
        raise AssertionError(f'UI text not visible: {text}')

    try:
        # The dashboard summarizes worker availability as ONLINE/OFFLINE;
        # internal states such as ``idle`` are intentionally not shown.
        wait(lambda:'ONLINE' in view() and 'UI qualification' in view(), 120)
        save('dashboard')
        assert '(.)' in view() and 'HS' in view() and 'ARQ' in view()
        checks.append('dashboard displays actual idle worker and modem lamp columns')
        send(b's')
        wait(lambda:'modem.echo' in view() and 'sip.user' in view())
        save('settings')
        assert 'IAX2 share the modem DSP' in view()
        click('All')
        wait(lambda:'Iax' in view())
        click('Iax')
        wait(lambda:'iax.user' in view())
        assert 'iax.user' in view()
        click('Iax')
        wait(lambda:'Routing' in view())
        click('Routing')
        wait(lambda:'network.trunk' in view())
        assert 'network.trunk' in view()
        click('Routing')
        wait(lambda:'All' in view())
        click('All')
        wait(lambda:'modem.echo' in view())
        checks.append('settings screen loads live modem, SIP and stored IAX values')
        click('modem.echo')
        # Selecting the row fills the editor. Tab from the table focuses it.
        click('modem.echo')
        save('settings-selected')
        # The modal puts the value editor at a stable fixed row below the
        # scrollable table. Focus it directly; Tab order varies by Textual.
        click_at(42,32)
        send(b'\x1b[H\x1b[3~')  # Home, Delete: replace the current Input value.
        type_text('0')
        save('settings-editor')
        click('Apply')
        wait(lambda:re.search(r'modem\.echo\s+0\s+0\s+immediate',view()) is not None)
        save('settings-applied')
        click('Save profile')
        profile=output/'instances/modem-1/profiles.json'
        wait(lambda:profile.exists() and json.loads(profile.read_text()).get('profiles',{}).get('0',{}).get('echo') == 0)
        checks.append('setting edited and saved through actual UI callbacks into worker profile')
        send(b'\x1b')
        wait(lambda:'UI qualification' in view() and 'ONLINE' in view())
        send(b't')
        wait(lambda:'ASCII TERMINAL' in view().upper())
        type_text('ATS0?')
        send(b'\r')
        wait(lambda:'OK' in view())
        save('terminal-hayes')
        assert '0' in view()
        checks.append('actual ASCII terminal sends Hayes query and renders worker response')
        type_text('AT+CFG="modem.echo",1')
        send(b'\r')
        wait(lambda:'APPLY: immediate' in view())
        type_text('AT&W0')
        send(b'\r')
        wait(lambda:json.loads(profile.read_text()).get('profiles',{}).get('0',{}).get('echo') == 1)
        save('terminal-shared-setting')
        checks.append('Hayes terminal edits shared setting and persists it in live worker')
        send(b'\x1b[21~')  # F10 releases terminal rather than transmitting it.
        wait(lambda:'UI qualification' in view() and 'ASCII TERMINAL' not in view().upper())
        send(b's')
        wait(lambda:'modem.echo' in view())
        save('settings-after-terminal')
        click('modem.echo')
        wait(lambda:re.search(r'modem\.echo\s+1\s+1\s+immediate',view()) is not None)
        save('settings-echo-restored')
        click('Close [Esc]')
        select_modem_action(4)  # fifth modem action: start/stop
        wait(lambda:'OFFLINE' in view())
        select_modem_action(3)  # fourth modem action: configure instance
        wait(lambda:'INSTANCE / CONNECTION SETUP' in view())
        save('instance-dialog')
        assert 'sip_port' in view() and 'media_port' in view() and '27232' in view()
        click('Cancel')
        wait(lambda:'UI qualification' in view() and 'OFFLINE' in view())
        select_modem_action(4)
        wait(lambda:'ONLINE' in view(),120)
        checks.append('instance dialog displays actual ports; worker stops and restarts through dashboard')
        send(b'q')
        process.wait(timeout=20)
        assert process.returncode==0
        checks.append('terminal release returns to dashboard and ordinary quit cleans up')
        report.update(status='PASS', exit_code=process.returncode)
    except Exception as exc:
        save('failure')
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        read(.1)
        raw.close()
        os.close(master)
        report['exit_code']=process.returncode
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def qualify_posix_device(executable, output):
    """Exercise the frozen worker's real-device backend using a PTY pair."""
    if os.name!='posix':
        return dict(status='SKIP',reason='POSIX serial-device backend only')
    master,slave=pty.openpty()
    device_dir=output/'device';device_dir.mkdir()
    config=device_dir/'config.json'
    config.write_text(json.dumps(dict(schema=1,modems=[dict(
        id='modem-1',name='POSIX device qualification',serial='device',
        serial_device=os.ttyname(slave),serial_baud=115200,enabled=True,
        sip_host='127.0.0.1',sip_port=28232,media_host='127.0.0.1',
        media_port=28432,serial_port=28234)])))
    process=subprocess.Popen([str(executable),'--headless','--config',str(config)],
                             stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
    try:
        os.set_blocking(master,False)
        start=time.monotonic()
        deadline=time.monotonic()+15
        response=bytearray()
        sent=False
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError(f'packaged device worker exited {process.returncode}')
            if not sent and time.monotonic()>=start+1:
                os.write(master,b'ATI\r');sent=True
            ready,_,_=select.select([master],[],[],.1)
            if ready:
                try:response.extend(os.read(master,4096))
                except BlockingIOError:pass
            if b'EL-MODEM' in response and b'OK' in response:break
        assert b'ATI' in response and b'OK' in response, response.decode('ascii','replace')
        return dict(status='PASS',check='frozen worker returned a Hayes ATI response over POSIX device backend',
                    response=response.decode('ascii','replace'))
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        os.close(master);os.close(slave)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=run(args.executable,args.output)
    device=qualify_posix_device(args.executable,args.output)
    report['posix_device_backend']=device
    if report['status']=='PASS' and device['status'] not in ('PASS','SKIP'):
        report['status']='FAIL'
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
