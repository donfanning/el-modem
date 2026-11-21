#!/usr/bin/env python3
"""Telnet-reachable load server for modem data-segment tests (protocol: audio-peer/telnet_load.py).

A dial-in pool's telnet gateway connects here on behalf of our modem. Nothing is sent before the
token, one session runs at a time and every session is capped, so a stray connection gets only
the banner. Each session is logged as one JSON line.

    telnet_load_server.py --listen 192.0.2.1:2323 --token-file /path/to/token \\
        --log ~/modem-artifacts/telnet-load/sessions.jsonl
"""
import argparse
import json
from pathlib import Path
import select
import socket
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).parent / 'audio-peer'))
import telnet_load as tl  # noqa: E402


def serve_session(conn, peer, token, max_seconds, log):
    session = dict(peer=f'{peer[0]}:{peer[1]}', opened=time.time())
    telnet = tl.TelnetFilter()
    buffer = bytearray()
    conn.settimeout(0.05)

    def receive():
        try:
            data = conn.recv(65536)
        except socket.timeout:
            return b''
        if not data:
            raise ConnectionError('peer closed')
        clean = telnet.feed(data)
        if telnet.replies:
            conn.sendall(bytes(telnet.replies)); telnet.replies.clear()
        return clean

    def read_line(deadline):
        while time.monotonic() < deadline:
            for line in tl._lines(buffer):
                if line:
                    return line
            buffer.extend(receive())
        return None

    try:
        conn.sendall(tl.BANNER)
        given = read_line(time.monotonic() + 60)
        if given is None or given.decode('latin1') != token:
            session['outcome'] = 'bad_token' if given is not None else 'no_token'
            conn.sendall(b'DENIED\r\n')
            return
        command = (read_line(time.monotonic() + 30) or b'').decode('latin1').split()
        try:
            _, mode, seconds, seed, *options = command
            seconds, seed = min(float(seconds), max_seconds), int(seed)
            down_window = None
            for option in options:
                if not option.startswith('W'):
                    raise ValueError
                # At least two K intervals, or the window could wait on an acknowledgement never due.
                down_window = max(int(option[1:]), 2 * tl.DOWN_ACK_EVERY * tl.RECORD_BYTES)
            if command[0] != 'RUN' or mode not in tl.MODES or seconds <= 0 or len(options) > 1:
                raise ValueError
        except ValueError:
            session['outcome'] = 'bad_command'; session['command'] = command[:6]
            conn.sendall(b'ERROR RUN <down|up|both|echo> <seconds> <seed> [W<bytes>]\r\n')
            return
        session.update(mode=mode, seconds=seconds, seed=seed, down_window=down_window)
        conn.sendall(b'GO\r\n')
        start = time.monotonic()
        up = tl.RecordChecker(seed, 'U')
        down_rate, up_rate = tl.Throughput(start), tl.Throughput(start)
        down_seq = 0
        down_acked = 0                       # client's next expected down record (K lines)
        last_ack = -1
        client_sent = None
        outgoing = bytearray()
        end_at = start + seconds
        drain_until = None
        while True:
            now = time.monotonic()
            if drain_until is None and (now >= end_at or client_sent is not None):
                drain_until = now + 15
            if drain_until is not None and (now >= drain_until or (
                    client_sent is not None and up.expected >= client_sent)):
                break
            if drain_until is None and mode in ('down', 'both') and len(outgoing) < 4096:
                if down_window is None:
                    chunk = b''.join(tl.record(seed, 'D', down_seq + i) for i in range(8))
                    outgoing.extend(chunk); down_seq += 8
                else:
                    while (down_seq - down_acked + 1) * tl.RECORD_BYTES <= down_window:
                        outgoing.extend(tl.record(seed, 'D', down_seq)); down_seq += 1
            if outgoing and select.select([], [conn], [], 0)[1]:
                # Link-limited: TCP backpressure from the gateway paces this.
                sent = conn.send(outgoing); down_rate.add(sent, now); del outgoing[:sent]
            data = receive()
            buffer.extend(data)
            for line in tl._lines(buffer):
                if not line:
                    continue
                if up.line(line):
                    up_rate.add(tl.RECORD_BYTES, now); continue
                down_ack = tl._DOWN_ACK.fullmatch(line)
                if down_ack:
                    down_acked = max(down_acked, int(down_ack[1])); continue
                if line.startswith(b'END'):
                    parts = line.split()
                    client_sent = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else up.expected
                elif line.startswith(b'E') and line[1:].isdigit() and mode == 'echo':
                    outgoing.extend(line + b'\r\n')
            if mode in ('up', 'both') and up.expected - 1 > last_ack:
                last_ack = up.expected - 1
                outgoing.extend(b'A%08d\r\n' % up.expected)
        up_report = dict(up.report(), **up_rate.report(), client_sent=client_sent)
        up_report['clean'] = bool(client_sent) and up.expected == client_sent and not (
            up.bad_crc or up.bad_body or up.missing or up.duplicates)
        session.update(outcome='complete', run_seconds=round(time.monotonic() - start, 2),
                       down=dict(sent_records=down_seq, acked_records=down_acked, window=down_window,
                                 **down_rate.report()),
                       up=up_report if mode in ('up', 'both') else None,
                       telnet_commands=telnet.commands)
        summary = {k: session[k] for k in ('mode', 'run_seconds', 'down', 'up')}
        conn.settimeout(30)
        conn.sendall(bytes(outgoing) + b'RESULT ' + json.dumps(summary, separators=(',', ':')).encode() + b'\r\n')
    except (OSError, ConnectionError) as error:
        session.setdefault('outcome', 'connection_error'); session['error'] = str(error)
    finally:
        session['closed'] = time.time()
        with log_lock:
            with open(log, 'a') as file:
                file.write(json.dumps(session) + '\n')
        try:
            conn.close()
        except OSError:
            pass


log_lock = threading.Lock()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--listen', default='127.0.0.1:2323', help='HOST:PORT')
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if len(token) < 12:
        parser.error('token must be at least 12 characters')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    host, port = args.listen.rsplit(':', 1)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(4)
    busy = threading.Lock()
    print(f'listening on {host}:{port}', flush=True)
    while True:
        conn, peer = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if not busy.acquire(blocking=False):
            try:
                conn.sendall(b'BUSY\r\n'); conn.close()
            except OSError:
                pass
            continue

        def run(conn=conn, peer=peer):
            try:
                serve_session(conn, peer, token, args.max_seconds, args.log)
            finally:
                busy.release()
        threading.Thread(target=run, daemon=True).start()


if __name__ == '__main__':
    main()
