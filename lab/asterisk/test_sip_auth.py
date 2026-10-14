import socket
import threading
import time
import unittest

import sip_auth
from sip_uas import parse_message


class DigestTests(unittest.TestCase):
    def test_rfc2617_vector(self):
        ch = {"realm": "testrealm@host.com", "nonce": "dcd98b7102dd2f0e8b11d0f600bfb0c093", "qop": "auth",
              "opaque": "5ccc069c403ebaf9f0171e9517f40e41"}
        out = sip_auth.digest_response("Mufasa", "Circle Of Life", "GET", "/dir/index.html", ch, nc=1,
                                       cnonce="0a4f113b")
        self.assertIn('response="6629fae49393a05397450978507c4ef1"', out)

    def test_parse_challenge(self):
        ch = sip_auth.parse_challenge('Digest realm="asterisk", nonce="abc", qop="auth,auth-int", algorithm=MD5')
        self.assertEqual((ch["realm"], ch["nonce"], ch["algorithm"]), ("asterisk", "abc", "MD5"))

    def test_register_with_challenge(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.bind(("127.0.0.1", 0))
        srv.settimeout(3)
        got = []

        def registrar():
            for step in range(2):
                data, addr = srv.recvfrom(65535)
                _, h, _ = parse_message(data)
                got.append(h)
                cid, via, frm, to, cseq = h["call-id"], h["via"], h["from"], h["to"], h["cseq"]
                if step == 0:
                    resp = ('SIP/2.0 401 Unauthorized\r\nVia: %s\r\nFrom: %s\r\nTo: %s;tag=x\r\nCall-ID: %s\r\nCSeq: %s\r\n'
                            'WWW-Authenticate: Digest realm="lab", nonce="n0nce"\r\nContent-Length: 0\r\n\r\n'
                            % (via, frm, to, cid, cseq))
                else:
                    resp = ('SIP/2.0 200 OK\r\nVia: %s\r\nFrom: %s\r\nTo: %s;tag=x\r\nCall-ID: %s\r\nCSeq: %s\r\n'
                            'Content-Length: 0\r\n\r\n' % (via, frm, to, cid, cseq))
                srv.sendto(resp.encode(), addr)

        t = threading.Thread(target=registrar)
        t.start()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(0.5)
        reg = sip_auth.SipRegistration("127.0.0.1", srv.getsockname()[1], "alice", "secret", "127.0.0.1",
                                       sock.getsockname()[1])
        self.assertTrue(reg.register_once(sock))
        t.join()
        self.assertTrue(reg.registered)
        self.assertNotIn("authorization", got[0])
        self.assertIn('username="alice"', got[1]["authorization"])
        sock.close()
        srv.close()


class InviteAuthTest(unittest.TestCase):
    def test_invite_answers_401(self):
        import threading as th
        from sip_uac import SipOriginateCall

        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.bind(("127.0.0.1", 0))
        srv.settimeout(5)
        seen = []

        def uas():
            for step in range(2):
                data, addr = srv.recvfrom(65535)
                start, h, _ = parse_message(data)
                seen.append((start, h))
                base = ("Via: %s\r\nFrom: %s\r\nTo: %s;tag=t\r\nCall-ID: %s\r\nCSeq: %s\r\n"
                        % (h["via"], h["from"], h["to"], h["call-id"], h["cseq"]))
                if step == 0:
                    srv.sendto(("SIP/2.0 401 Unauthorized\r\n" + base +
                                'WWW-Authenticate: Digest realm="r", nonce="n"\r\nContent-Length: 0\r\n\r\n').encode(), addr)
                    srv.recvfrom(65535)                       # the ACK for the 401
                else:
                    sdp = "v=0\r\nc=IN IP4 127.0.0.1\r\nm=audio 4000 RTP/AVP 0\r\n"
                    srv.sendto(("SIP/2.0 200 OK\r\n" + base + "Contact: <sip:x@127.0.0.1>\r\n"
                                f"Content-Type: application/sdp\r\nContent-Length: {len(sdp)}\r\n\r\n" + sdp).encode(), addr)

        t = th.Thread(target=uas)
        t.start()
        result = []
        call = SipOriginateCall("127.0.0.1", 0, "127.0.0.1", srv.getsockname()[1], "127.0.0.1", 16100,
                                on_connected=lambda a: result.append(("up", a)),
                                on_failed=lambda r: result.append(("fail", r)), on_hangup=lambda: None,
                                username="alice", password="pw", caller_number="5551234")
        th2 = th.Thread(target=call.dial, args=("5559999", 5))
        th2.start()
        t.join()
        for _ in range(50):
            if result:
                break
            time.sleep(0.05)
        call.hangup()
        th2.join(3)
        srv.close()
        self.assertEqual(result[0], ("up", ("127.0.0.1", 4000)))
        self.assertNotIn("authorization", seen[0][1])
        self.assertIn('username="alice"', seen[1][1]["authorization"])
        self.assertEqual(seen[1][1]["cseq"], "2 INVITE")
        self.assertIn("5551234", seen[1][1]["from"])


class SharedRegistrationTest(unittest.TestCase):
    def test_register_through_uas_socket(self):
        from sip_uas import SipIncomingCallServer

        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.bind(("127.0.0.1", 0))
        srv.settimeout(5)

        def registrar():
            for step in range(2):
                data, addr = srv.recvfrom(65535)
                _, h, _ = parse_message(data)
                common = ("Via: %s\r\nFrom: %s\r\nTo: %s;tag=x\r\nCall-ID: %s\r\nCSeq: %s\r\n"
                          % (h["via"], h["from"], h["to"], h["call-id"], h["cseq"]))
                if step == 0:
                    srv.sendto(("SIP/2.0 401 Unauthorized\r\n" + common +
                                'WWW-Authenticate: Digest realm="r", nonce="n"\r\nContent-Length: 0\r\n\r\n').encode(), addr)
                else:
                    srv.sendto(("SIP/2.0 200 OK\r\n" + common + "Content-Length: 0\r\n\r\n").encode(), addr)

        t = threading.Thread(target=registrar)
        t.start()
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        uas = SipIncomingCallServer(host="127.0.0.1", port=port, media_host="127.0.0.1", media_port=16200,
                                    on_incoming=lambda *a: None, on_ring=lambda *a: None, on_hangup=lambda: None)
        uas.start()
        reg = sip_auth.SipRegistration("127.0.0.1", srv.getsockname()[1], "alice", "pw", "127.0.0.1", port)
        reg.start(shared_with=uas)
        for _ in range(60):
            if reg.registered:
                break
            time.sleep(0.05)
        self.assertTrue(reg.registered, reg.last_status)
        t.join()
        srv.close()
        reg.stop()
        uas.close()


class NoLineTest(unittest.TestCase):
    def test_closed_port_reports_no_line(self):
        # An unconnected UDP socket gets no ICMP error on Linux, so this ends as "no response" after the first-reply
        # wait; both map to NO DIALTONE.
        from sip_uac import SipOriginateCall

        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()                                      # nothing listens here: ICMP port unreachable
        result = []
        call = SipOriginateCall("127.0.0.1", 0, "127.0.0.1", port, "127.0.0.1", 16300,
                                on_connected=lambda a: result.append("up"), on_failed=result.append,
                                on_hangup=lambda: None)
        call.dial("5550000", 5)
        self.assertIn(result, (["no line"], ["no response"]))

    def test_silent_proxy_reports_no_response(self):
        from sip_uac import SipOriginateCall

        silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        silent.bind(("127.0.0.1", 0))
        result = []
        call = SipOriginateCall("127.0.0.1", 0, "127.0.0.1", silent.getsockname()[1], "127.0.0.1", 16301,
                                on_connected=lambda a: result.append("up"), on_failed=result.append,
                                on_hangup=lambda: None)
        t0 = time.monotonic()
        call.dial("5550000", 30)
        silent.close()
        self.assertEqual(result, ["no response"])
        self.assertLess(time.monotonic() - t0, 12)


if __name__ == "__main__":
    unittest.main()
