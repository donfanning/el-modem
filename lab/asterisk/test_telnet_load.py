import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'audio-peer'))
import telnet_load as tl  # noqa: E402
import telnet_load_server as server  # noqa: E402
import run_pstn_capture  # noqa: E402

TOKEN = 'test-token-0123456789'


def listener():
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(4)
    return sock


def start_server(log, max_seconds=30):
    sock = listener()

    def loop():
        while True:
            try:
                conn, peer = sock.accept()
            except OSError:
                return
            threading.Thread(target=server.serve_session,
                             args=(conn, peer, TOKEN, max_seconds, log), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    return sock


def relay(target, telnet=False, bytes_per_second=None, deep_buffer=False):
    """Gateway stand-in: optional telnet noise (option requests, LF -> CR LF) and a rate limit.

    ``deep_buffer`` reads the server side without limit and only drains it to the client at
    ``bytes_per_second``, like a Patton dial-in pool (262 KB accepted in 14 s at 2400 bit/s)."""
    sock = listener()
    if deep_buffer:
        return deep_relay(sock, target, bytes_per_second)

    def pump(src, dst, to_client):
        sent_at = time.monotonic(); budget = 0.0
        try:
            if telnet and to_client:
                dst.sendall(bytes((tl.IAC, tl.DO, 24, tl.IAC, tl.WILL, 1, tl.IAC, tl.SB, 24, 1,
                                   tl.IAC, tl.SE)))
            while True:
                data = src.recv(512 if bytes_per_second else 65536)
                if not data:
                    break
                if telnet and to_client:
                    data = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                if bytes_per_second:
                    now = time.monotonic(); budget = min(budget + (now - sent_at) * bytes_per_second, 512); sent_at = now
                    if budget < len(data):
                        time.sleep((len(data) - budget) / bytes_per_second); budget = len(data); sent_at = time.monotonic()
                    budget -= len(data)
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def loop():
        client, _ = sock.accept()
        upstream = socket.create_connection(target.getsockname())
        threading.Thread(target=pump, args=(upstream, client, True), daemon=True).start()
        threading.Thread(target=pump, args=(client, upstream, False), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    return sock


def deep_relay(sock, target, bytes_per_second):
    def loop():
        client, _ = sock.accept()
        upstream = socket.create_connection(target.getsockname())
        backlog, lock, done = bytearray(), threading.Lock(), threading.Event()

        def fill():                                    # the gateway's TCP side: accept everything
            try:
                while (data := upstream.recv(65536)):
                    with lock:
                        backlog.extend(data)
            except OSError:
                pass
            done.set()

        def drain():                                   # its modem side: line rate only
            try:
                while not (done.is_set() and not backlog):
                    with lock:
                        chunk = bytes(backlog[:64]); del backlog[:len(chunk)]
                    if chunk:
                        client.sendall(chunk); time.sleep(len(chunk) / bytes_per_second)
                    else:
                        time.sleep(0.01)
            except OSError:
                pass
            client.close()

        def up():
            try:
                while (data := client.recv(4096)):
                    upstream.sendall(data)
            except OSError:
                pass
        for target_fn in (fill, drain, up):
            threading.Thread(target=target_fn, daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    return sock


class RecordTests(unittest.TestCase):
    def test_record_shape_and_check(self):
        line = tl.record(7, 'D', 42)
        self.assertEqual(len(line), tl.RECORD_BYTES)
        self.assertNotIn(b'\xff', line)
        checker = tl.RecordChecker(7, 'D')
        self.assertTrue(checker.line(line.strip()))
        self.assertEqual(checker.missing, 42)
        corrupt = bytearray(tl.record(7, 'D', 43).strip()); corrupt[-1] ^= 1
        checker.line(bytes(corrupt))
        self.assertEqual(checker.bad_crc, 1)

    def test_telnet_filter_strips_and_refuses(self):
        f = tl.TelnetFilter()
        out = f.feed(bytes((tl.IAC, tl.DO, 1)) + b'ab\r\n' + bytes((tl.IAC, tl.SB, 24, 1, tl.IAC, tl.SE)) + b'c\r\0')
        self.assertEqual(out, b'ab\nc')
        self.assertEqual(bytes(f.replies), bytes((tl.IAC, tl.WONT, 1)))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / 'sessions.jsonl'
        self.server = start_server(self.log)

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def run_mode(self, mode, seconds=1.5, down_window_bytes=None, **relay_args):
        gateway = relay(self.server, **relay_args)
        conn = socket.create_connection(gateway.getsockname())
        try:
            return tl.run_client(conn, TOKEN, mode, seconds, seed=11, banner_timeout=5, drain_seconds=10,
                                 down_window_bytes=down_window_bytes)
        finally:
            conn.close()

    def test_both_directions_clean_through_telnet_relay(self):
        result = self.run_mode('both', telnet=True)
        self.assertTrue(result['payload_verified'], result)
        self.assertGreater(result['down']['good'], 10)
        self.assertTrue(result['server']['up']['clean'])
        self.assertGreater(result['telnet_commands'], 0)

    def test_rate_limited_link_measures_the_link(self):
        result = self.run_mode('down', seconds=2, bytes_per_second=4000)
        self.assertTrue(result['payload_verified'], result)
        rate = result['down']['good_bytes'] / result['run_seconds']
        self.assertLess(rate, 6000)
        self.assertGreater(rate, 2000)

    def test_up_is_windowed_and_verified(self):
        result = self.run_mode('up', seconds=2, bytes_per_second=3000)
        self.assertTrue(result['payload_verified'], result)
        self.assertLessEqual(result['up']['sent_records'] - result['up']['acked_records'],
                             4096 // tl.RECORD_BYTES + 1)

    def test_record_zero_in_the_same_read_as_go_is_counted(self):
        # A gateway that delivers GO and the first records together (cs1771-load6, 2026-11-21).
        a, b = socket.socketpair()
        seed = 5

        def fake_server():
            time.sleep(0.2); b.recv(4096)
            b.sendall(b'GO\r\n' + tl.record(seed, 'D', 0) + tl.record(seed, 'D', 1))
            time.sleep(0.3)
            b.sendall(b'RESULT {}\r\n')
        b.sendall(tl.BANNER)
        threading.Thread(target=fake_server, daemon=True).start()
        result = tl.run_client(a, TOKEN, 'down', 0.2, seed=seed, banner_timeout=5, drain_seconds=2)
        a.close(); b.close()
        self.assertEqual((result['down']['good'], result['down']['missing']), (2, 0), result)

    def test_down_window_keeps_a_deep_gateway_buffer_empty(self):
        # Unwindowed, the server fills the gateway's buffer and its acks and RESULT queue behind
        # minutes of records; windowed by the client's K acknowledgements, both arrive promptly.
        result = self.run_mode('both', seconds=3, bytes_per_second=1500, deep_buffer=True,
                               down_window_bytes=1024)
        self.assertTrue(result['payload_verified'], result)
        self.assertEqual(result['stop_reason'], 'server_result', result)
        self.assertLessEqual(result['server']['down']['sent_records'] - result['down']['good'],
                             1024 // tl.RECORD_BYTES + 1, result)
        self.assertGreater(result['up']['acked_records'], 0, result)

    def test_xoff_from_the_dce_pauses_upstream(self):
        a, b = socket.socketpair()
        b.sendall(tl.BANNER)

        def fake_server():
            b.recv(4096)
            b.sendall(b'GO\r\n' + bytes((tl.XOFF,)))
            time.sleep(0.5)
            b.setblocking(False)
            try:
                held = b.recv(65536)
            except BlockingIOError:
                held = b''
            b.setblocking(True)
            fake_server.held = held
            b.sendall(bytes((tl.XON,)))
            time.sleep(0.3)
            b.sendall(b'RESULT {}\r\n')
        fake_server.held = None
        threading.Thread(target=fake_server, daemon=True).start()
        result = tl.run_client(a, TOKEN, 'up', 0.6, seed=3, banner_timeout=5, drain_seconds=2)
        a.close(); b.close()
        self.assertEqual(fake_server.held.count(b'\nU'), 0, fake_server.held[:200])
        self.assertGreater(result['up']['sent_records'], 0, result)
        self.assertGreater(result['xoff_count'], 0, result)

    def test_echo(self):
        result = self.run_mode('echo', seconds=1)
        self.assertTrue(result['payload_verified'], result)
        self.assertGreater(len(result['echo_rtt_seconds']), 3)

    def test_wrong_token_gets_nothing(self):
        conn = socket.create_connection(self.server.getsockname())
        result = tl.run_client(conn, 'wrong-token-xxxxx', 'down', 1, seed=1, banner_timeout=5)
        conn.close()
        self.assertEqual(result['stop_reason'], 'refused')
        time.sleep(0.2)
        self.assertEqual(json.loads(self.log.read_text().splitlines()[-1])['outcome'], 'bad_token')


def fake_modem(server_sock, bytes_per_second=6000):
    """A DCE plus dial-in gateway: AT -> OK, ATDT -> CONNECT and a host prompt, then relay to the
    host it is given until +++; ATH0 -> OK."""
    dte, modem = socket.socketpair()

    def run():
        buf = b''
        upstream = None
        while True:
            try:
                data = modem.recv(4096)
            except OSError:
                return
            if not data:
                return
            if upstream is None:
                buf += data
                while b'\r' in buf:
                    line, buf = buf.split(b'\r', 1)
                    if line.startswith(b'ATDT'):
                        modem.sendall(b'\r\nCONNECT 33600/LAPM\r\nTelnet gateway\r\nHost: ')
                    elif line.startswith(b'AT'):
                        modem.sendall(b'\r\nOK\r\n')
                    elif line.strip():
                        host, port = line.decode().split()
                        assert (host, int(port)) == server_sock.getsockname()
                        upstream = socket.create_connection((host, int(port)))
                        gateway = relay_pair(upstream, modem, bytes_per_second)
            elif data == b'+++':
                upstream.close(); upstream = 'online-command'
                modem.sendall(b'\r\nOK\r\n')
            elif upstream == 'online-command':
                modem.sendall(b'\r\nOK\r\n')
            else:
                upstream.sendall(data)
    threading.Thread(target=run, daemon=True).start()
    return dte


def relay_pair(upstream, modem, bytes_per_second):
    def pump():
        try:
            while True:
                data = upstream.recv(256)
                if not data:
                    return
                modem.sendall(data)
                time.sleep(len(data) / bytes_per_second)
        except OSError:
            return
    threading.Thread(target=pump, daemon=True).start()


class ProbeIntegrationTests(unittest.TestCase):
    def test_probe_runs_load_through_gateway_and_hangs_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            srv = start_server(Path(tmp) / 'sessions.jsonl')
            host, port = srv.getsockname()
            dte = fake_modem(srv)
            result = run_pstn_capture.probe(
                dte, '5551234', observe_seconds=30, whole_seconds=60,
                replies=[(r'(?i)host\s*:', f'{host} {port}\r'.encode())],
                load=('both', 2.0, TOKEN, 99))
            srv.close()
        self.assertEqual(result['stop_reason'], 'load_server_result', result)
        self.assertTrue(result['payload_verified'], result['load'])
        self.assertTrue(result['hangup_confirmed'])
        self.assertGreater(result['load']['down']['good'], 20)


if __name__ == '__main__':
    unittest.main()
