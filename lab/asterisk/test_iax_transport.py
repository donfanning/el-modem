import socket
import sys
import types
import unittest
from unittest.mock import patch

from iax_transport import IaxMediaSocket, IaxTransport


class FakeIax:
    def __init__(self):
        self.payloads = []

    def send_voice(self, payload):
        self.payloads.append(payload)


class IaxMediaSocketTests(unittest.TestCase):
    def test_transport_exposes_udp_socket_for_select(self):
        class FakeSocket:
            def fileno(self):
                return 42

        transport = object.__new__(IaxTransport)
        transport.socket = FakeSocket()
        self.assertEqual(transport.fileno(), 42)

    def test_rtp_pcmu_packet_is_forwarded_as_iax_audio(self):
        transport = FakeIax()
        adapter = IaxMediaSocket(transport)
        header = bytes((0x80, 0)) + b"\0" * 10
        payload = bytes(range(160))
        self.assertEqual(adapter.sendto(header + payload, ("unused", 0)), 172)
        self.assertEqual(transport.payloads, [payload])

    def test_rejects_non_pcmu_rtp(self):
        adapter = IaxMediaSocket(FakeIax())
        with self.assertRaises(ValueError):
            adapter.sendto(bytes((0x80, 8)) + b"\0" * 12, ("unused", 0))

    def test_ipv6_server_resolution_uses_ipv6_wildcard_bind(self):
        class FakeSocket:
            def __init__(self, *_args):
                self.bound = None

            def bind(self, address):
                self.bound = address

            def setblocking(self, _enabled):
                pass

            def sendto(self, _data, _address):
                pass

            def close(self):
                pass

            def fileno(self):
                return 42

        class FakeClient:
            def __init__(self, *_args):
                pass

            def drain_tx(self):
                return []

        class FakeJitter:
            def __init__(self):
                pass

        fake_extension = types.SimpleNamespace(Client=FakeClient, MediaJitter=FakeJitter)
        fake_socket = FakeSocket()
        with patch.dict(sys.modules, {"el_modem_iax": fake_extension}), \
             patch("iax_transport.socket.getaddrinfo", return_value=[
                 (socket.AF_INET6, socket.SOCK_DGRAM, 17, "", ("::1", 4569, 0, 0))
             ]), patch("iax_transport.socket.socket", return_value=fake_socket):
            transport = IaxTransport("iax.example", 4569, "user", "secret")
            self.assertEqual(fake_socket.bound, ("::", 0))
            transport.close()


if __name__ == "__main__":
    unittest.main()
