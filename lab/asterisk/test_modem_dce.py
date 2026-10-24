#!/usr/bin/env python3
"""Unit and Linux PTY smoke tests for the lab Hayes DCE."""

import json
import math
import os
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from contextlib import nullcontext
try:
    import tty
except ImportError:                 # Windows: no PTYs
    tty = None
import types
import unittest
from unittest import mock
from pathlib import Path

from modem_dce import (
    AtDce,
    _env_enabled,
    DeferredV42Release,
    V42BreakRequestQueue,
    default_error_control_for,
    dial_abort_applies,
    EscapeHold,
    EscapeSequenceDetector,
    OnlineAudioSession,
    bell212_dte_baud_for_serial,
    prepare_v42_break_dte_order,
    v42_break_options_for_k,
    v42_manual_break_options_for_k,
    linear_to_ulaw,
    normalize_dial_number,
    v42_break_can_be_delivered,
    ulaw_to_linear,
    v34_rx_timing_slips,
    run_pty,
)
from call_progress import (
    BusyToneDetector,
    OutgoingCallProgressMonitor,
    RingbackToneDetector,
    VoiceActivityDetector,
)
from lapm_lite import ArqSession

SCRIPT = Path(__file__).with_name("modem_dce.py")


class EnvironmentFlagTests(unittest.TestCase):
    def test_explicit_false_values_disable_modem_debug(self):
        for value in ("", "0", "false", "False", "no", "off"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"MODEM_DEBUG": value}):
                self.assertFalse(_env_enabled("MODEM_DEBUG"))

    def test_true_values_and_capture_paths_remain_enabled(self):
        for value in ("1", "true", "yes", "on", "/tmp/capture"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"MODEM_DEBUG": value}):
                self.assertTrue(_env_enabled("MODEM_DEBUG"))


class DeferredV42ReleaseTests(unittest.TestCase):
    def test_call_teardown_waits_for_lapm_release_completion(self) -> None:
        session = types.SimpleNamespace(
            request_v42_release=mock.Mock(return_value=True),
            v42_release_timeout=mock.Mock(return_value=4.0),
            v42_release_complete=mock.Mock(return_value=False),
        )
        release = DeferredV42Release()

        self.assertTrue(release.request(session, now=10.0))
        session.request_v42_release.assert_called_once_with()
        self.assertTrue(release.pending)
        self.assertFalse(release.ready(session, now=13.999))
        session.v42_release_complete.return_value = True
        self.assertTrue(release.ready(session, now=11.0))
        release.clear()
        self.assertFalse(release.pending)
        self.assertFalse(release.ready(session, now=20.0))

    def test_call_teardown_falls_back_at_the_bounded_release_deadline(self) -> None:
        session = types.SimpleNamespace(
            request_v42_release=mock.Mock(return_value=True),
            v42_release_timeout=mock.Mock(return_value=4.0),
            v42_release_complete=mock.Mock(return_value=False),
        )
        release = DeferredV42Release()

        self.assertTrue(release.request(session, now=10.0))
        self.assertFalse(release.ready(session, now=13.999))
        self.assertTrue(release.ready(session, now=14.0))

    def test_call_teardown_is_immediate_without_an_active_lapm_link(self) -> None:
        release = DeferredV42Release()
        self.assertFalse(release.request(None, now=10.0))
        self.assertFalse(release.pending)

        unavailable = types.SimpleNamespace(
            request_v42_release=mock.Mock(return_value=False),
        )
        self.assertFalse(release.request(unavailable, now=10.0))
        self.assertFalse(release.pending)


class V34RtpMediaClockBridgeTests(unittest.TestCase):
    def _session(self, context, *, skipped=0):
        session = OnlineAudioSession.__new__(OnlineAudioSession)
        session._capture = None
        session._rx_deaf_samples = skipped
        session._rx_context = dict(context)
        session.receiver = types.SimpleNamespace(
            set_rx_media_deadline=mock.Mock(), feed_pcm=mock.Mock(return_value=b""),
        )
        session._v25_caller = None
        session._rx_arrival = None
        session._rtp_jitter_samples = 320
        session._v22 = types.SimpleNamespace(connected=False)
        session.realtime = mock.Mock()
        session._dsp_capture_pending = False
        session._dsp_capture = None
        session.timing_trace = None
        session._v42_probe = False
        session._negotiator = None
        session._rx_switched = True
        session._bell212_char_bits = 10
        return session

    def test_receive_media_anchor_uses_first_sample_after_deaf_skip(self):
        session = self._session({
            "packet_playout_deadline_monotonic": 100.0,
            "payload_samples": 160,
            "pcm_source": "packet",
        }, skipped=20)
        session._receive_pcm(b"\0\0" * 160)
        session.receiver.set_rx_media_deadline.assert_called_once_with(100.0025)

    def test_transmit_deadline_is_attached_to_generated_rtp_frame(self):
        class Tx:
            def __init__(self):
                self.deadline = None

            def pending(self):
                return 0

            def set_tx_media_deadline(self, deadline):
                self.deadline = deadline

            def samples(self, count):
                return [0] * count

        session = OnlineAudioSession.__new__(OnlineAudioSession)
        session._pending_tx_mode = None
        session._guard_frames = 0
        session._online_control = None
        session._rtp_frame_samples = 160
        session.transmitter = Tx()
        session._flush_held_terminal_bytes = lambda: None
        session._next_frame_samples(123.5)
        self.assertEqual(session.transmitter.deadline, 123.5)


class RuntimeClockTests(unittest.TestCase):
    def test_windows_uses_higher_resolution_monotonic_clock(self):
        import modem_dce

        coarse = types.SimpleNamespace(
            implementation="GetTickCount64()", resolution=0.015625,
        )
        fine = types.SimpleNamespace(
            implementation="QueryPerformanceCounter()", resolution=1e-7,
        )
        perf_counter = modem_dce.time.perf_counter
        monotonic = modem_dce.time.monotonic
        with mock.patch.object(modem_dce.time, "monotonic", monotonic), \
                mock.patch.object(modem_dce.time, "perf_counter", perf_counter), \
                mock.patch.object(modem_dce.threading, "_time", monotonic):
            info = modem_dce._configure_runtime_clock(
                platform="nt", monotonic_info=coarse, perf_counter_info=fine,
            )
            self.assertIs(modem_dce.time.monotonic, perf_counter)
            self.assertIs(modem_dce.threading._time, perf_counter)
            self.assertEqual(info["source"], "time.perf_counter")
            self.assertEqual(info["replaced_resolution_seconds"], 0.015625)
            self.assertTrue(info["threading_clock_updated"])

    def test_keeps_monotonic_when_it_is_already_as_fine(self):
        import modem_dce

        fine = types.SimpleNamespace(
            implementation="QueryPerformanceCounter()", resolution=1e-7,
        )
        monotonic = modem_dce.time.monotonic
        with mock.patch.object(modem_dce.time, "monotonic", monotonic):
            info = modem_dce._configure_runtime_clock(
                platform="nt", monotonic_info=fine, perf_counter_info=fine,
            )
            self.assertIs(modem_dce.time.monotonic, monotonic)
            self.assertEqual(info["source"], "time.monotonic")


class DtrLapmReleaseIntegrationTests(unittest.TestCase):
    def test_dtr_drop_keeps_sip_call_until_lapm_release_completes(self) -> None:
        shared = {"events": [], "sip_hungup": False, "dtr1_ok_before_hangup": False}
        from modem_dce import AtDce as RealAtDce
        created_dces = []

        class FakeSerial:
            def __init__(self, hangup_mode):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "test-socketpair"
                self.supports_modem_control = False
                self.hangup_mode = hangup_mode
                self.drop_dtr = False
                self.connected = False
                self.ath_sent = False
                self.escape_started = False
                self.escape_at = None
                self.closed = False

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                if b"CONNECT" in data:
                    self.connected = True
                    if self.hangup_mode in ("dtr", "dtr1", "dtr3"):
                        self.drop_dtr = True
                    if self.hangup_mode in ("ath", "dtr0"):
                        self.escape_at = time.monotonic() + 1.05
                if (self.hangup_mode == "dtr1" and self.connected and not self.ath_sent
                        and b"\r\nOK\r\n" in data):
                    shared["dtr1_ok_before_hangup"] = not shared["sip_hungup"]
                    self.ath_sent = True
                    self.dte.sendall(b"ATH\r")
                if (self.hangup_mode in ("ath", "dtr0") and self.escape_started
                        and not self.ath_sent and b"\r\nOK\r\n" in data):
                    if self.hangup_mode == "dtr0":
                        shared["dtr0_connected_after_drop"] = not shared["sip_hungup"]
                    self.ath_sent = True
                    self.dte.sendall(b"ATH\r")

            def service(self, readable, writable=()):
                if shared["sip_hungup"]:
                    raise KeyboardInterrupt
                if self.escape_at is not None and time.monotonic() >= self.escape_at:
                    self.dte.sendall(b"+++")
                    self.escape_at = None
                    self.escape_started = True
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                dropped, self.drop_dtr = self.drop_dtr, False
                return data, dropped

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]
                self.on_hangup = kwargs["on_hangup"]
                self.hungup = False
                self.last_caller = {"number": "test", "technology": "SIP"}

            def start(self):
                self.on_incoming()
                self.on_ring()

            def answer(self):
                return True

            def remote_rtp(self):
                return ("127.0.0.1", 40000)

            def hangup(self):
                if not self.hungup:
                    self.hungup = True
                    shared["events"].append("sip_hangup")
                    shared["sip_hungup"] = True

            def close(self):
                pass

        class FakeOnlineSession:
            def __init__(self, *args, **kwargs):
                self.negotiation_complete = True
                self.failed = self.link_lost = False
                self.line_speed = 1200
                self.connect_suffix = "/LAPM"
                self.modulation_name = "B212"
                self.timing_trace = None
                self.xoff_asserted = False
                self._v22 = types.SimpleNamespace(v42_mode="lapm")
                self.release_requested = False
                self.release_polls = 0
                self.stopped = False

            def start(self):
                pass

            def poll(self, now):
                if self.release_requested:
                    self.release_polls += 1

            def poll_automatic_rate(self, **kwargs):
                return None

            def rtp_deadline_delay(self, now):
                return None

            def drain_rtp_deadline(self, now):
                return b""

            def set_dte_receive_busy(self, busy):
                pass

            def read_v42_break_events(self):
                return []

            def request_v42_break(self):
                return False

            def request_v42_release(self):
                self.release_requested = True
                shared["events"].append("lapm_disc_requested")
                return True

            def v42_release_timeout(self):
                return 5.0

            def v42_release_complete(self):
                return self.release_polls >= 2

            def link_stats(self):
                return {"v42_state": "released" if self.v42_release_complete() else "releasing"}

            def stop(self):
                self.stopped = True
                shared["events"].append("session_stopped")

        for hangup_mode in ("dtr0", "dtr", "dtr1", "dtr3", "ath"):
            with self.subTest(hangup_mode=hangup_mode):
                shared.update(events=[], sip_hungup=False, dtr1_ok_before_hangup=False,
                              dtr0_connected_after_drop=False)
                serial = FakeSerial(hangup_mode)
                dtr_value = {"dtr0": 0, "dtr1": 1, "dtr": 2, "dtr3": 3}.get(hangup_mode)
                setup = (f"ATS7=90&D{dtr_value}\r" if hangup_mode == "dtr3"
                         else f"AT&D{dtr_value}\r" if dtr_value is not None else "")
                serial.dte.sendall((setup + "ATA\r").encode())
                fake_sessions = []

                def make_dce(*args, **kwargs):
                    dce = RealAtDce(*args, **kwargs)
                    created_dces.append(dce)
                    return dce

                def make_session(*args, **kwargs):
                    session = FakeOnlineSession(*args, **kwargs)
                    fake_sessions.append(session)
                    return session

                try:
                    with mock.patch("modem_dce.AtDce", side_effect=make_dce), \
                            mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                            mock.patch("modem_dce.OnlineAudioSession", side_effect=make_session):
                        run_pty(
                            None, sip_host="127.0.0.1", media_host="127.0.0.1", media_port=0,
                            line_mode="bell103", negotiate=False, serial_backend=serial)
                finally:
                    serial.dte.close()

                self.assertTrue(serial.closed)
                self.assertEqual(len(fake_sessions), 1)
                if hangup_mode == "dtr1":
                    self.assertTrue(shared["dtr1_ok_before_hangup"],
                                    "&D1 must issue OK while the call remains connected")
                if hangup_mode == "dtr0":
                    self.assertTrue(shared["dtr0_connected_after_drop"],
                                    "&D0 must ignore DTR and preserve the call")
                if hangup_mode == "dtr3":
                    self.assertEqual(created_dces[-1].registers[7], 60,
                                     "&D3 must restore factory profile after hangup")
                    self.assertEqual(created_dces[-1].dtr_behavior, 2)
                session = fake_sessions[0]
                self.assertTrue(session.release_requested)
                self.assertGreaterEqual(session.release_polls, 2)
                self.assertTrue(session.stopped)
                events = shared["events"]
                self.assertLess(events.index("lapm_disc_requested"), events.index("session_stopped"))
                self.assertLess(events.index("session_stopped"), events.index("sip_hangup"))

    def test_run_pty_ath_waits_for_independent_spandsp_ua_before_sip_hangup(self) -> None:
        """Run the real DCE event loop and release controller over Bell RTP."""
        import spandsp_v42
        import v22

        if not spandsp_v42.available():
            self.skipTest("pinned SpanDSP reference library is unavailable")

        shared = {"events": [], "sip_hungup": False, "session": None}
        peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer_sock.bind(("127.0.0.1", 0))
        media_reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        media_reservation.bind(("127.0.0.1", 0))
        media_port = media_reservation.getsockname()[1]
        media_reservation.close()

        class SpanDSPBellCaller(v22.V22Modem):
            def __init__(self, **kwargs):
                kwargs.pop("calling_party", None)
                kwargs.pop("bit_rate", None)
                kwargs.pop("profile", None)
                super().__init__(True, 1200, profile="bell212a", **kwargs)
                self.span_link = None

            def start_v42(self, *args, **kwargs):
                self.span_link = spandsp_v42.V42Link(True, detect=True, bit_rate=1200)
                self.v42_mode = "detect"

            def _sync_span_mode(self):
                if self.span_link is not None:
                    self.v42_mode = {
                        "detect": "detect", "lapm": "lapm",
                        "unsupported": "failed", "down": "failed",
                    }[self.span_link.state]

            def v42_tx_bit(self):
                if self.span_link is None:
                    return 1
                bit = self.span_link.tx_bit()
                self._sync_span_mode()
                return bit

            def v42_rx_bits(self, bits):
                if self.span_link is not None:
                    for bit in bits:
                        self.span_link.rx_bit(bit)
                    self._sync_span_mode()
                return b""

            def v42_feed_bytes(self, data):
                if self.span_link is not None:
                    self.span_link.write(data)

            def v42_read(self):
                return self.span_link.read() if self.span_link is not None else b""

            def close(self):
                try:
                    if self.span_link is not None:
                        self.span_link.close()
                finally:
                    super().close()

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "test-socketpair"
                self.supports_modem_control = False
                self.escape_due = None
                self.escape_sent = False
                self.ath_sent = False
                self.ata_sent = False
                self.closed = False
                self.started = time.monotonic()

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                shared["events"].append(("serial_tx", bytes(data)))
                if b"CONNECT 1200" in data and self.escape_due is None:
                    self.escape_due = time.monotonic() + 1.1
                if not self.ata_sent and b"\r\nOK\r\n" in data:
                    self.ata_sent = True
                    self.dte.sendall(b"ATA\r")
                if self.escape_sent and not self.ath_sent and b"\r\nOK\r\n" in data:
                    self.ath_sent = True
                    shared["events"].append("dte_ath")
                    self.dte.sendall(b"ATH\r")

            def service(self, readable, writable=()):
                if shared["sip_hungup"]:
                    raise KeyboardInterrupt
                if time.monotonic() - self.started > 20:
                    shared["events"].append("test_timeout")
                    raise KeyboardInterrupt
                if self.escape_due is not None and not self.escape_sent \
                        and time.monotonic() >= self.escape_due:
                    self.escape_sent = True
                    self.dte.sendall(b"+++")
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                if data:
                    shared["events"].append(("serial_rx", bytes(data)))
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]
                self.on_hangup = kwargs["on_hangup"]
                self.hungup = False
                self.last_caller = {"number": "test", "technology": "SIP"}

            def start(self):
                shared["events"].append("sip_start")
                self.on_incoming()
                self.on_ring()

            def answer(self):
                shared["events"].append("sip_answer")
                return True

            def remote_rtp(self):
                shared["events"].append("remote_rtp")
                return peer_sock.getsockname()

            def hangup(self):
                if self.hungup:
                    return
                self.hungup = True
                shared["events"].append("sip_hangup")
                session = shared["session"]
                if session is not None and session._v22._v42.state == "released":
                    shared["events"].append("lapm_released")
                shared["sip_hungup"] = True

            def close(self):
                pass

        # The far end is a production Bell 212A audio/session endpoint whose
        # LAPM protocol state machine is independently implemented by SpanDSP.
        original_v22 = v22.V22Modem
        v22.V22Modem = SpanDSPBellCaller
        try:
            peer = OnlineAudioSession(
                peer_sock, ("127.0.0.1", media_port), "originate", "answer",
                line_mode="bell212a", v42=True,
            )
        finally:
            v22.V22Modem = original_v22
        peer.start()
        peer_stop = threading.Event()
        peer_error = []

        def pump_peer():
            try:
                while not peer_stop.is_set():
                    peer.poll(time.monotonic())
                    try:
                        packet, source = peer_sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    peer.handle_rtp_datagram(source, packet)
                    if peer._v22.span_link is not None \
                            and "LINK_DISCONNECTED" in peer._v22.span_link.events:
                        shared["events"].append("peer_link_disconnected")
                        return
            except Exception as exc:  # propagate worker failure to the test
                peer_error.append(exc)

        peer_sock.settimeout(0.01)
        peer_thread = threading.Thread(target=pump_peer, daemon=True)
        peer_thread.start()
        serial = FakeSerial()
        serial.dte.sendall(b"ATE0\\N3\r")
        original_stop = OnlineAudioSession.stop

        def tracked_session(*args, **kwargs):
            session = OnlineAudioSession(*args, **kwargs)
            shared["session"] = session

            def stop():
                shared["events"].append("session_stopped")
                original_stop(session)

            session.stop = stop
            return session

        try:
            with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                    mock.patch("modem_dce.OnlineAudioSession", side_effect=tracked_session):
                run_pty(None, sip_host="127.0.0.1", media_host="127.0.0.1",
                        media_port=media_port, line_mode="bell212a", negotiate=True,
                        serial_backend=serial)
        finally:
            peer_stop.set()
            peer.stop()
            peer_thread.join(timeout=2)
            peer_sock.close()
            serial.dte.close()

        self.assertFalse(peer_thread.is_alive(), "independent RTP peer did not stop")
        self.assertEqual(peer_error, [])
        self.assertTrue(serial.closed)
        self.assertIsNotNone(shared["session"], shared["events"])
        self.assertEqual(shared["session"]._v22._v42.state, "released",
                         shared["events"])
        self.assertIn("LINK_DISCONNECTED", peer._v22.span_link.events)
        events = shared["events"]
        self.assertLess(events.index("dte_ath"), events.index("lapm_released"))
        self.assertLess(events.index("peer_link_disconnected"), events.index("sip_hangup"))
        self.assertLess(events.index("session_stopped"), events.index("sip_hangup"))


class IfcDceIntegrationTests(unittest.TestCase):
    def test_run_pty_xoff_holds_received_bytes_until_xon(self) -> None:
        """Exercise +IFC and real V.42 timeout-to-NO-CARRIER through run_pty."""
        from v42_layer import V42Layer

        payload = bytes(range(1, 128)) * 3
        shared = {"busy": [], "payload_seen": False, "payload_before_xon": False,
                  "hangup": False, "server": None, "recovery_started": False}

        class RecoveryEngine(V42Layer):
            def __init__(self):
                self.bit_rate = 1200
                self._v42_setup()

            def _plain_tx_bit(self):
                return 1

            def _plain_feed_bytes(self, data):
                pass

            def _plain_feed_marks(self, count):
                pass

            def _plain_rx_bits(self, bits):
                return b""

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "ifc-test-socketpair"
                self.supports_modem_control = False
                self.ifc_set = False
                self.xoff_sent = False
                self.xon_sent = False
                self.received = bytearray()
                self.closed = False
                self.dte.sendall(b"AT+IFC=1,1\r")

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                self.received.extend(data)
                if not self.ifc_set and b"\r\nOK\r\n" in self.received:
                    self.ifc_set = True
                    self.dte.sendall(b"ATA\r")
                if self.ifc_set and not self.xoff_sent and b"CONNECT" in self.received:
                    self.xoff_sent = True
                    self.dte.sendall(b"\x13")
                if payload in data:
                    if not self.xon_sent:
                        shared["payload_before_xon"] = True
                    shared["payload_seen"] = True

            def service(self, readable, writable=()):
                if shared["hangup"]:
                    raise KeyboardInterrupt
                if shared["busy"] and shared["busy"][-1] and not self.xon_sent:
                    self.xon_sent = True
                    self.dte.sendall(b"\x11")
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]
                self.hungup = False
                self.last_caller = {"number": "test", "technology": "SIP"}
                self.peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.peer.bind(("127.0.0.1", 0))

            def start(self):
                self.on_incoming()
                self.on_ring()

            def answer(self):
                return True

            def remote_rtp(self):
                return self.peer.getsockname()

            def hangup(self):
                if not self.hungup:
                    self.hungup = True
                    shared["hangup"] = True

            def close(self):
                self.peer.close()

        class FakeOnlineSession:
            def __init__(self, media_socket, remote_rtp, *args, **kwargs):
                self.media_socket = media_socket
                self.remote_rtp = remote_rtp
                self.negotiation_complete = True
                self.failed = False
                self.line_speed = 1200
                self.connect_suffix = ""
                self.modulation_name = "BELL 103"
                self.timing_trace = None
                self.xoff_asserted = False
                self._v22 = RecoveryEngine()
                self.sent = False

            @property
            def link_lost(self):
                return self._v22.v42_lost

            def start(self):
                def send_audio():
                    time.sleep(0.15)
                    header = b"\x80\x00" + b"\x00" * 10
                    shared["server"].peer.sendto(header + b"\x00" * 160,
                                                  self.media_socket.getsockname())
                threading.Thread(target=send_audio, daemon=True).start()

            def poll(self, now):
                # Exercise the real recovery timer/state machine in the
                # production run_pty event loop, with no peer to re-establish.
                if shared["payload_seen"] and not shared["recovery_started"]:
                    self._v22.start_v42(
                        True, rate=1200, detect=False, compression=(1, 1024, 8),
                        compression_recovery=True,
                    )
                    self._v22.v42_mode = "recovering"
                    self._v22._v42_recovery_stage = "establish"
                    self._v22._v42_recovery_timeout_seconds = 0.05
                    self._v22._v42_recovery_deadline = time.monotonic() + 0.05
                    shared["recovery_started"] = True
                if shared["recovery_started"]:
                    self._v22.v42_tx_bit()

            def poll_automatic_rate(self, **kwargs):
                return None

            def rtp_deadline_delay(self, now):
                return None

            def handle_rtp_datagram(self, peer, packet, **kwargs):
                if self.sent:
                    return b""
                self.sent = True
                return payload

            def drain_rtp_deadline(self, now):
                return b""

            def set_dte_receive_busy(self, busy):
                shared["busy"].append(bool(busy))

            def read_v42_break_events(self):
                return []

            def flow_control_signal(self, extra=0):
                return b""

            def link_stats(self):
                return {"v42_state": "off"}

            def stop(self):
                if self._v22._v42 is not None:
                    self._v22._v42.close()

        serial = FakeSerial()

        def make_server(*args, **kwargs):
            server = FakeSipServer(*args, **kwargs)
            shared["server"] = server
            return server

        with tempfile.TemporaryDirectory(prefix="modem-ifc-") as directory:
            profile = str(Path(directory) / "profiles.json")
            try:
                with mock.patch("modem_dce.SipIncomingCallServer", side_effect=make_server), \
                        mock.patch("modem_dce.OnlineAudioSession", FakeOnlineSession):
                    run_pty(None, sip_host="127.0.0.1", media_host="127.0.0.1", media_port=0,
                            line_mode="bell103", negotiate=False, serial_backend=serial,
                            profile_file=profile)
            finally:
                serial.dte.close()

        self.assertTrue(serial.closed)
        self.assertTrue(serial.xoff_sent)
        self.assertTrue(serial.xon_sent)
        self.assertIn(True, shared["busy"])
        self.assertFalse(shared["payload_before_xon"])
        self.assertEqual(bytes(serial.received).count(payload), 1)
        self.assertIn(b"CONNECT", serial.received)
        self.assertIn(b"NO CARRIER", serial.received)
        self.assertTrue(shared["server"].hungup)
        self.assertTrue(shared["recovery_started"])


class StrictXidDceIntegrationTests(unittest.TestCase):
    def test_outgoing_strict_xid_loss_reports_no_carrier_and_hangs_up(self) -> None:
        """Exercise strict XID failure through run_pty and a SpanDSP audio peer."""
        import spandsp_v42
        import v22
        from collections import deque
        from v42_lapm import FLAG_BITS, N400, XID, _Hdlc

        if not spandsp_v42.available():
            self.skipTest("pinned SpanDSP reference library is unavailable")

        shared = {"dce_output": bytearray(), "sip_hungup": False, "events": [],
                  "dropped_xid_responses": 0, "peer_error": []}
        peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer_sock.bind(("127.0.0.1", 0))
        peer_sock.settimeout(0.01)
        media_reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        media_reservation.bind(("127.0.0.1", 0))
        media_port = media_reservation.getsockname()[1]
        media_reservation.close()

        class DropXidResponses:
            def __init__(self):
                self.parser = _Hdlc()
                self.pending = []

            def pass_bits(self, bit, *, intercept):
                if not intercept:
                    self.parser = _Hdlc()
                    self.pending.clear()
                    return [bit]
                out = []
                self.pending.append(bit)
                for frame, good_fcs in self.parser.push(bit):
                    if good_fcs and len(frame) >= 2 and frame[1] & 0xEF == XID:
                        shared["dropped_xid_responses"] += 1
                        out.extend(self.pending[-len(FLAG_BITS):])
                    else:
                        out.extend(self.pending)
                    self.pending.clear()
                return out

        class SpanDSPBellAnswerer(v22.V22Modem):
            def __init__(self, **kwargs):
                kwargs.pop("calling_party", None)
                kwargs.pop("bit_rate", None)
                kwargs.pop("profile", None)
                super().__init__(False, 1200, profile="bell212a", **kwargs)
                self.span_link = None
                self._xid_filter = DropXidResponses()
                self._xid_tx_queue = deque()

            def start_v42(self, *args, **kwargs):
                self.span_link = spandsp_v42.V42Link(False, detect=True, bit_rate=1200)
                self.v42_mode = "detect"

            def _sync_span_mode(self):
                if self.span_link is not None:
                    self.v42_mode = {
                        "detect": "detect", "lapm": "lapm",
                        "unsupported": "failed", "down": "failed",
                    }[self.span_link.state]

            def v42_tx_bit(self):
                if self.span_link is None:
                    return 1
                bit = self.span_link.tx_bit()
                filtered = self._xid_filter.pass_bits(bit, intercept=self.span_link.adp_seen)
                self._xid_tx_queue.extend(filtered)
                self._sync_span_mode()
                return self._xid_tx_queue.popleft() if self._xid_tx_queue else 1

            def v42_rx_bits(self, bits):
                if self.span_link is not None:
                    for bit in bits:
                        self.span_link.rx_bit(bit)
                    self._sync_span_mode()
                return b""

            def v42_feed_bytes(self, data):
                if self.span_link is not None:
                    self.span_link.write(data)

            def v42_read(self):
                return self.span_link.read() if self.span_link is not None else b""

            def close(self):
                if self.span_link is not None:
                    self.span_link.close()
                super().close()

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "strict-xid-test"
                self.supports_modem_control = False
                self.closed = False
                self.started = time.monotonic()

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                shared["dce_output"].extend(data)
                try:
                    self.dte.sendall(data)
                except OSError:
                    pass

            def service(self, readable, writable=()):
                if shared["sip_hungup"] and b"NO CARRIER" in shared["dce_output"]:
                    raise KeyboardInterrupt
                if time.monotonic() - self.started > 40:
                    session = shared.get("dce_session")
                    link = getattr(getattr(session, "_v22", None), "_v42", None)
                    shared["peer_error"].append(TimeoutError(
                        f"strict XID DCE gate timed out: state={getattr(link, 'state', None)!r}, "
                        f"xid_sent={getattr(link, '_xid_sent', None)}, tries={getattr(link, '_tries', None)}, "
                        f"bits_sent={getattr(link, 'bits_sent', None)}, events={getattr(link, 'events', [])[-8:]}"
                    ))
                    raise KeyboardInterrupt
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]

            def start(self):
                pass

            def hangup(self):
                pass

            def close(self):
                pass

        class FakeSipCall:
            def __init__(self, *args, **kwargs):
                self.on_connected = kwargs["on_connected"]
                self.hungup = False

            def dial(self, number):
                shared["events"].append(("sip_dial", number))
                shared["events"].append("sip_connected")
                self.on_connected(peer_sock.getsockname())

            def hangup(self):
                self.hungup = True
                shared["events"].append("sip_hangup")
                shared["sip_hungup"] = True

        peer = None
        peer_stop = threading.Event()
        original_v22 = v22.V22Modem
        try:
            v22.V22Modem = SpanDSPBellAnswerer
            peer = OnlineAudioSession(
                peer_sock, ("127.0.0.1", media_port), "answer", "originate",
                line_mode="bell212a", v42=True,
            )
        finally:
            v22.V22Modem = original_v22
        peer.start()

        def pump_peer():
            try:
                while not peer_stop.is_set():
                    peer.poll(time.monotonic())
                    try:
                        packet, source = peer_sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    peer.handle_rtp_datagram(source, packet)
            except Exception as exc:  # propagated to the test thread
                shared["peer_error"].append(exc)

        peer_thread = threading.Thread(target=pump_peer, daemon=True)
        peer_thread.start()
        serial = FakeSerial()
        serial.dte.sendall(b"ATD1001\r")
        original_stop = OnlineAudioSession.stop

        def tracked_session(*args, **kwargs):
            shared["events"].append(("session_created", kwargs.get("line_mode"), kwargs.get("v42"),
                                     kwargs.get("v42_xid_fallback")))
            session = OnlineAudioSession(*args, **kwargs)
            shared["dce_session"] = session

            def stop():
                shared["events"].append("session_stopped")
                original_stop(session)

            original_poll = session.poll

            def poll(now):
                original_poll(now)
                link = getattr(session._v22, "_v42", None)
                if (link is not None and link.state == "idle"
                        and shared["dropped_xid_responses"] >= 1
                        and not shared.get("injected_n400_expiry")):
                    # The bit-level SpanDSP test proves all N400 retry turns.
                    # Inject its terminal T401 expiry here after one actual
                    # response loss, so this full DCE/SIP teardown test does
                    # not depend on the host's slow wall-clock RTP loop.
                    link._tries = N400
                    link._t401_at = link.bits_sent
                    link._t401_expired()
                    shared["injected_n400_expiry"] = True
                    shared["events"].append("injected_n400_exhaustion")

            session.poll = poll
            session.stop = stop
            return session

        with tempfile.TemporaryDirectory() as profile_dir:
            profile_file = Path(profile_dir) / "strict-xid-profile.json"
            profile_file.write_text(json.dumps({"profiles": {"0": {
                "echo": False, "error_control": 3, "v42_xid_fallback": False,
                "ms": {"mod": "B212", "auto": 0, "min": 300, "max": 1200},
            }}}), encoding="utf-8")
            try:
                with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                        mock.patch("modem_dce.SipOriginateCall", FakeSipCall), \
                        mock.patch("modem_dce.OnlineAudioSession", side_effect=tracked_session):
                    run_pty(None, sip_host="127.0.0.1", media_host="127.0.0.1",
                            media_port=media_port, dial_host="127.0.0.1", line_mode="ms",
                            negotiate=True, profile_file=str(profile_file), serial_backend=serial)
            finally:
                peer_stop.set()
                peer.stop()
                peer_thread.join(timeout=2)
                peer_sock.close()
                serial.dte.close()

        output = bytes(shared["dce_output"])
        self.assertTrue(serial.closed)
        self.assertFalse(peer_thread.is_alive())
        self.assertEqual(shared["peer_error"], [],
                         (bytes(shared["dce_output"][-1200:]), shared["events"],
                          shared["dropped_xid_responses"], peer._v22.v42_mode,
                          getattr(peer._v22.span_link, "events", None),
                          getattr(getattr(shared.get("dce_session"), "_v22", None), "_v42", None)))
        self.assertGreaterEqual(shared["dropped_xid_responses"], 1,
                                (bytes(shared["dce_output"][-1200:]),
                                 shared["events"], peer._v22.v42_mode))
        self.assertIn("injected_n400_exhaustion", shared["events"])
        self.assertIn(b"NO CARRIER", output)
        self.assertNotIn(b"CONNECT", output)
        self.assertIn("sip_hangup", shared["events"])
        self.assertIn("session_stopped", shared["events"])
        self.assertLess(shared["events"].index("session_stopped"),
                        shared["events"].index("sip_hangup"))


class Bell212RawBreakDceTests(unittest.TestCase):
    def test_run_pty_bridges_dte_and_remote_raw_bell_breaks(self) -> None:
        shared = {"events": [], "ended": False, "connected": False}

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "bell-break-test"
                self.supports_modem_control = False
                self._break_count = 0
                self.closed = False
                self.started = time.monotonic()

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                shared["events"].append(("serial_tx", bytes(data)))
                if b"CONNECT" in data:
                    self._break_count = 1
                    shared["connected"] = True
                    shared["events"].append("connect")

            def take_break_events(self):
                count, self._break_count = self._break_count, 0
                return count

            def send_break(self, duration_ms=250):
                shared["events"].append(("remote_break", duration_ms))
                return True

            def service(self, readable, writable=()):
                if time.monotonic() - self.started > 8:
                    raise KeyboardInterrupt
                if "local_break" in shared["events"] and any(
                        isinstance(event, tuple) and event[0] == "remote_break"
                        for event in shared["events"]):
                    raise KeyboardInterrupt
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSession:
            def __init__(self, *args, **kwargs):
                shared["leased_line"] = kwargs.get("bell212_leased_line", False)
                self.negotiation_complete = True
                self.failed = self.link_lost = False
                self.line_speed = 1200
                self.connect_suffix = ""
                self.modulation_name = "BELL 212A"
                self.timing_trace = None
                self.xoff_asserted = False
                self._v22 = types.SimpleNamespace(
                    profile="bell212a", _bell103_fallback=False, v42_mode="off",
                )
                self.transmitter = types.SimpleNamespace(pending=lambda: 0)
                self._remote_break_pending = True

            def start(self):
                pass

            def poll(self, now):
                pass

            def poll_automatic_rate(self, **kwargs):
                return None

            def rtp_deadline_delay(self, now):
                return None

            def drain_rtp_deadline(self, now):
                return b""

            def set_dte_receive_busy(self, busy):
                pass

            def read_v42_break_events(self):
                return []

            def read_bell212_break_events(self):
                if self._remote_break_pending and shared["connected"]:
                    self._remote_break_pending = False
                    return [{"type": "break_indication", "start_bits": 23,
                             "character_bits": 10}]
                return []

            def send_bell212_break(self):
                shared["events"].append("local_break")
                return True

            def flow_control_signal(self, extra_pending_bits=0):
                return None

            def link_stats(self):
                return {"modulation": "BELL 212A"}

            def stop(self):
                pass

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]
                self.hungup = False
                self.last_caller = {"number": "test", "technology": "SIP"}

            def start(self):
                self.on_incoming()
                self.on_ring()

            def answer(self):
                return True

            def remote_rtp(self):
                return ("127.0.0.1", 40000)

            def hangup(self):
                self.hungup = True
                shared["ended"] = True

            def close(self):
                pass

        serial = FakeSerial()
        serial.dte.sendall(b"AT&L1\rATA\r")
        try:
            with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                    mock.patch("modem_dce.OnlineAudioSession", FakeSession):
                run_pty(None, sip_host="127.0.0.1", media_host="127.0.0.1",
                        media_port=0, line_mode="bell212a", negotiate=False,
                        serial_backend=serial)
        finally:
            serial.dte.close()

        self.assertTrue(serial.closed)
        self.assertTrue(shared["ended"])
        self.assertTrue(shared["leased_line"])
        self.assertIn("connect", shared["events"])
        self.assertIn("local_break", shared["events"])
        self.assertIn(("remote_break", 20), shared["events"])


class AutomaticTrafficTests(unittest.TestCase):
    def test_acknowledged_outbound_activity_without_inbound_application_bytes(self):
        session=OnlineAudioSession.__new__(OnlineAudioSession)
        link=types.SimpleNamespace(stats={'wire_acked':0,'bytes_out':9999,'i_resent':99})
        engine=types.SimpleNamespace(_v42=link)
        seen=[]
        session._online_control=types.SimpleNamespace(engine=lambda:engine)
        session._automatic_rate=types.SimpleNamespace(tick=lambda **kw:seen.append(kw))
        session._auto_traffic=(0,None,0)
        session._auto_loaded_until=session._auto_transport_until=0.
        session._auto_transport=(0,0)
        session._rtp_counts={}
        def poll(now,received=0):
            with mock.patch('modem_dce.time.monotonic',return_value=now):
                session.poll_automatic_rate(active=True,mode=2,received_chars=received)
            return seen[-1]
        self.assertFalse(poll(1.)['loaded'])  # Queuing/retransmission is not delivery.
        link.stats['wire_acked']=128
        self.assertTrue(poll(2.)['loaded'])
        self.assertFalse(poll(2.6)['loaded']) # Unchanged ACK count expires.
        link.stats['wire_acked']=256
        self.assertTrue(poll(3.)['loaded'])
        engine._v42=types.SimpleNamespace(stats={'wire_acked':0})
        self.assertFalse(poll(3.6)['loaded']) # A new link alone is not activity.
        engine._v42.stats['wire_acked']=128
        self.assertTrue(poll(4.)['loaded'])
        self.assertTrue(poll(5.,received=128)['loaded'])
        self.assertFalse(poll(5.6,received=128)['loaded'])
        session._rtp_counts['concealed_samples']=160
        self.assertFalse(poll(6.,received=128)['transport_ok'])


class Bell212FallbackSessionTests(unittest.TestCase):
    def test_bell103_fallback_reports_300_and_does_not_start_lapm(self):
        from v22 import V22Modem

        modem = V22Modem(False, 1200, profile="bell212a")
        modem._select_bell103_fallback()
        session = OnlineAudioSession.__new__(OnlineAudioSession)
        session._v22 = modem
        session._auto = None
        session._want_v42 = True

        self.assertEqual(session.line_speed, 300)
        self.assertEqual(session.modulation_name, "BELL 103 (212A FALLBACK)")
        self.assertFalse(session._v42_active())

    def test_rtp_session_falls_back_and_exchanges_data_with_independent_fsk(self):
        import math
        import socket

        from bell103 import Bell103Receiver

        vector_path = (Path(__file__).parent / "audio-peer" / "fixtures" /
                       "reference" / "independent-fsk" / "bell103a2-orig.raw")
        if not vector_path.is_file():
            self.skipTest("optional independent audio fixture is not in the public export")
        vector = vector_path.read_bytes()
        caller_payload = b"ELMOD B103 ORIGINATE 0123456789\r"
        answer_payload = b"ANSWER FROM DCE 300\r"
        caller_samples = list(struct.unpack(f"<{len(vector) // 2}h", vector))
        mark = [
            round(12000 * math.sin(2 * math.pi * 1270 * i / 8000))
            for i in range(2400)
        ]
        incoming = mark + caller_samples
        caller_starts = 2 * 8000 + round(0.611 * 8000)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "answer", "originate",
            line_mode="bell212a", v42=True,
        )
        caller_rx = Bell103Receiver("answer")
        received_at_dce = bytearray()
        received_at_caller = bytearray()
        answer_queued = False
        connected_state = None
        total_samples = caller_starts + len(incoming) + 2 * 8000
        try:
            for offset in range(0, total_samples, 160):
                # Exercise the session's production transmit framing and the
                # same PCMU quantization used for its RTP sender.
                tx = session._next_frame_samples()
                quantized_tx = [
                    ulaw_to_linear(linear_to_ulaw(sample)) for sample in tx
                ]
                received_at_caller.extend(
                    caller_rx.feed_pcm(struct.pack("<160h", *quantized_tx))
                )

                input_offset = offset - caller_starts
                block = (incoming[input_offset:input_offset + 160]
                         if input_offset >= 0 else [])
                block += [0] * (160 - len(block))
                pcmu_payload = bytes(linear_to_ulaw(sample) for sample in block)
                received_at_dce.extend(session.handle_rtp_payload(pcmu_payload))

                if connected_state is None and session.negotiation_complete:
                    connected_state = (
                        session.line_speed,
                        session.modulation_name,
                        session._v42_active(),
                        session._v22.v42_mode,
                    )
                if session.negotiation_complete and not answer_queued:
                    session.feed_terminal_bytes(answer_payload)
                    answer_queued = True
                if (received_at_dce == caller_payload
                        and received_at_caller.endswith(answer_payload)):
                    break

            self.assertEqual(connected_state, (
                300, "BELL 103 (212A FALLBACK)", False, "off",
            ))
            dce = AtDce()
            dce.finish_online_control(session.line_speed, session.connect_suffix)
            self.assertEqual(dce.take_notifications(), b"\r\nCONNECT 300\r\n")
            self.assertTrue(answer_queued)
            self.assertEqual(bytes(received_at_dce), caller_payload)
            self.assertTrue(received_at_caller.endswith(answer_payload))
        finally:
            session.stop()
            sock.close()


class AtDceTests(unittest.TestCase):
    def test_echo_results_registers_and_x_options(self) -> None:
        dce = AtDce()
        self.assertEqual(dce.feed(b"AT\r"), b"AT\r\r\nOK\r\n")
        self.assertEqual(dce.feed(b"ATE0V0X0\r"), b"ATE0V0X0\r\r\n0\r\n")
        self.assertEqual(dce.feed(b"ATS0=1S0?\r"), b"\r\n1\r\n0\r\n")
        self.assertEqual(dce.result_mode, 0)
        self.assertEqual(dce.registers[0], 1)
        self.assertEqual(dce.feed(b"ATX4\r"), b"\r\n0\r\n")
        self.assertEqual(dce.result_mode, 4)

    def test_v22_guard_tone_hayes_setting_and_profile(self) -> None:
        dce = AtDce()
        self.assertIn(b"&Gn", dce.feed(b"AT&$\r"))
        self.assertIn(b"&G0", dce.feed(b"AT&V\r"))
        self.assertTrue(dce.feed(b"AT&G2\r").endswith(b"OK\r\n"))
        self.assertIn(b"&G2", dce.feed(b"AT&V\r"))
        self.assertTrue(dce.feed(b"AT&G3\r").endswith(b"ERROR\r\n"))
        self.assertEqual(dce.guard_tone, 2)

    def test_bell212_leased_line_setting_is_explicit_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profiles.json"
            dce = AtDce()
            dce.load_profiles(str(profile_path))
            self.assertTrue(dce.feed(b"AT&L1\r").endswith(b"OK\r\n"))
            self.assertEqual(dce.bell212_line_type, 1)
            self.assertIn(b"&L1", dce.feed(b"AT&V\r"))
            dce.feed(b"AT&W\r")
            dce.feed(b"AT&L0\r")
            self.assertEqual(dce.bell212_line_type, 0)
            dce.feed(b"ATZ\r")
            self.assertEqual(dce.bell212_line_type, 1)
            other = AtDce()
            other.load_profiles(str(profile_path))
            self.assertEqual(other.bell212_line_type, 1)
            self.assertTrue(dce.feed(b"AT&L2\r").endswith(b"ERROR\r\n"))

    def test_v22_guard_tone_is_saved_and_recalled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profiles.json"
            dce = AtDce()
            dce.load_profiles(str(profile_path))
            dce.feed(b"AT&G1&W\r")
            dce.feed(b"AT&F\r")
            self.assertEqual(dce.guard_tone, 0)
            dce.feed(b"ATZ\r")
            self.assertEqual(dce.guard_tone, 1)
            other = AtDce()
            other.load_profiles(str(profile_path))
            self.assertEqual(other.guard_tone, 1)

    def test_failed_calls_are_not_reported_as_connected(self) -> None:
        dce = AtDce()
        dce.feed(b"ATE0\r")
        self.assertEqual(dce.feed(b"ATD5551212\r"), b"\r\nNO CARRIER\r\n")
        self.assertEqual(dce.feed(b"ATA\r"), b"\r\nNO CARRIER\r\n")
        self.assertFalse(dce.online)

    def test_attached_dial_receives_raw_dial_string_and_connects(self) -> None:
        dce = AtDce()
        dce.feed(b"ATE0\r")
        seen = []

        def dial(number: str) -> bool:
            seen.append(number)
            return True

        dce.attach_dial(dial)
        self.assertEqual(dce.feed(b"ATDT5551212\r"), b"\r\nCONNECT 300\r\n")
        self.assertEqual(seen, ["T5551212"])
        self.assertTrue(dce.online)

    def test_caret_modifier_requests_calling_tone_once_and_is_not_dialled(self) -> None:
        # Commercial `^`: V.25 calling tone for this dial attempt only. The
        # trunk must never see it, redial must not repeat it.
        dce = AtDce()
        dce.feed(b"ATE0\r")
        seen = []
        dce.attach_dial(lambda number: seen.append(number) or True)
        self.assertEqual(dce.feed(b"ATDT555^1234\r"), b"\r\nCONNECT 300\r\n")
        self.assertEqual(seen, ["T5551234"])
        self.assertTrue(dce.calling_tone_once)
        self.assertEqual(dce.last_dialed, "T5551234")
        dce.feed(b"ATH\r")
        dce.feed(b"ATDL\r")
        self.assertEqual(seen, ["T5551234", "T5551234"])
        self.assertFalse(dce.calling_tone_once)

    def test_pending_dial_result_resolves_asynchronously(self) -> None:
        dce = AtDce()
        dce.feed(b"ATE0\r")
        dce.attach_dial(lambda number: None)
        self.assertEqual(dce.feed(b"ATD106\r"), b"")
        self.assertTrue(dce.outgoing_pending)
        self.assertEqual(dce.notify_call_progress("RING"), None)
        self.assertTrue(dce.outgoing_pending)
        dce.notify_call_progress("BUSY")
        self.assertFalse(dce.outgoing_pending)

    def test_call_backend_and_hangup_callbacks(self) -> None:
        calls: list[tuple[str, str]] = []
        dce = AtDce(
            dial=lambda number: calls.append(("dial", number)) is None,
            answer=lambda: calls.append(("answer", "")) is None,
            hangup=lambda: calls.append(("hangup", "")),
        )
        self.assertEqual(dce.feed(b"ATE0\r"), b"ATE0\r\r\nOK\r\n")
        self.assertEqual(dce.feed(b"ATD9,5551212\r"), b"\r\nCONNECT 300\r\n")
        self.assertTrue(dce.online)
        self.assertEqual(dce.feed(b"ATH0\r"), b"\r\nOK\r\n")
        self.assertFalse(dce.online)
        self.assertEqual(
            calls,
            [("dial", "9,5551212"), ("hangup", "")],
        )

    def test_invalid_commands_return_error(self) -> None:
        dce = AtDce()
        self.assertEqual(dce.feed(b"ATZ\r"), b"ATZ\r\r\nOK\r\n")
        self.assertEqual(dce.feed(b"AT+UNKNOWN\r"), b"AT+UNKNOWN\r\r\nERROR\r\n")
        # &T is the Hayes entry point for modem loop tests. Until a loop path
        # exists, reject it rather than falsely acknowledging a no-op.
        self.assertEqual(dce.feed(b"AT&T1\r"), b"AT&T1\r\r\nERROR\r\n")

    def test_software_test_loop_commands_require_and_track_real_handler(self) -> None:
        dce = AtDce()
        transitions = []
        dce.attach_software_test_loop(lambda enabled: transitions.append(enabled) is None)
        self.assertEqual(dce.feed(b"AT&T1\r"), b"AT&T1\r\r\nCONNECT 300\r\n")
        self.assertTrue(dce.online)
        self.assertIn(b"SOFTWARE TEST LOOP: SOFTWARE-AUDIO-LOOP",
                      dce.feed(b"ATI18\r"))
        self.assertIn(b"V.22/V.22bis, Bell 212A", dce.feed(b"ATI18\r"))
        self.assertIn(b"not a physical V.54 loop", dce.feed(b"ATI18\r"))
        self.assertEqual(dce.feed(b"AT&T0\r"), b"AT&T0\r\r\nOK\r\n")
        self.assertFalse(dce.online)
        self.assertEqual(transitions, [True, False])
        self.assertIn(b"SOFTWARE TEST LOOP: OFF", dce.feed(b"ATI18\r"))

    def test_oversized_command_is_rejected(self) -> None:
        dce = AtDce()
        command = b"AT" + b"X" * 300 + b"\r"
        self.assertEqual(dce.feed(command), command + b"\r\nERROR\r\n")

    def test_incoming_rings_notify_and_ata_answers_only_pending_call(self) -> None:
        answers: list[str] = []
        dce = AtDce(answer=lambda: answers.append("answered") is None)
        dce.feed(b"ATE0\r")

        self.assertEqual(dce.feed(b"ATA\r"), b"\r\nNO CARRIER\r\n")
        dce.notify_incoming_ring()
        self.assertEqual(dce.take_notifications(), b"\r\nRING\r\n")
        self.assertEqual(dce.registers[1], 1)
        self.assertTrue(dce.incoming_ringing)
        self.assertEqual(dce.feed(b"ATA\r"), b"")
        self.assertEqual(dce.take_notifications(), b"\r\nCONNECT 300\r\n")
        self.assertEqual(answers, ["answered"])
        self.assertTrue(dce.online)

    def test_incoming_signaling_starts_a_fresh_ring_sequence(self) -> None:
        dce = AtDce()
        dce.notify_incoming_ring()
        dce.notify_incoming_ring()
        self.assertEqual(dce.registers[1], 2)
        dce.begin_incoming_call()
        self.assertEqual(dce.registers[1], 0)
        self.assertTrue(dce.incoming_ringing)
        dce.notify_incoming_ring()
        self.assertEqual(dce.registers[1], 1)
        self.assertEqual(
            dce.take_notifications(),
            b"\r\nRING\r\n\r\nRING\r\n\r\nRING\r\n",
        )

    def test_s0_autoanswer_waits_for_configured_ring_count(self) -> None:
        answers: list[str] = []
        answer_started = threading.Event()

        def answer_call() -> bool:
            answer_started.set()
            answers.append("answered")
            return True

        dce = AtDce(answer=answer_call)
        dce.feed(b"ATE0S0=2\r")

        dce.notify_incoming_ring()
        self.assertEqual(dce.take_notifications(), b"\r\nRING\r\n")
        self.assertEqual(answers, [])
        dce.notify_incoming_ring()
        self.assertTrue(answer_started.wait(1))
        deadline = time.monotonic() + 1
        while not dce.online and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(
            dce.take_notifications(),
            b"\r\nRING\r\n\r\nCONNECT 300\r\n",
        )
        self.assertEqual(answers, ["answered"])
        self.assertTrue(dce.online)

    def test_originating_call_progress_and_terminal_outcomes(self) -> None:
        dce = AtDce(dial=lambda _number: "BUSY")
        dce.feed(b"ATE0\r")
        dce.notify_call_progress("RING")
        self.assertEqual(dce.take_notifications(), b"\r\nRING\r\n")
        self.assertEqual(dce.feed(b"ATD5551212\r"), b"\r\nBUSY\r\n")
        self.assertFalse(dce.online)

        voice = AtDce(dial=lambda _number: "VOICE")
        voice.feed(b"ATE0\r")
        self.assertEqual(voice.feed(b"ATD5551212\r"), b"\r\nVOICE\r\n")
        self.assertFalse(voice.online)

        numeric_voice = AtDce(dial=lambda _number: "VOICE")
        numeric_voice.feed(b"ATE0V0\r")
        self.assertEqual(numeric_voice.feed(b"ATD5551212\r"), b"\r\n10\r\n")

    def test_atx_gates_busy_detection_and_connect_speed_reporting(self) -> None:
        # X0-X2 are "blind dial" modes: a real USR modem would not report
        # BUSY at all in these modes, and would eventually time out instead.
        blind = AtDce(dial=lambda _number: "BUSY")
        blind.feed(b"ATE0X0\r")
        self.assertEqual(blind.feed(b"ATD5551212\r"), b"\r\nNO CARRIER\r\n")

        blind2 = AtDce(dial=lambda _number: "BUSY")
        blind2.feed(b"ATE0X2\r")
        self.assertEqual(blind2.feed(b"ATD5551212\r"), b"\r\nNO CARRIER\r\n")

        # X3/X4 enable busy detection.
        aware = AtDce(dial=lambda _number: "BUSY")
        aware.feed(b"ATE0X3\r")
        self.assertEqual(aware.feed(b"ATD5551212\r"), b"\r\nBUSY\r\n")

        # X0 reports a bare CONNECT; X1-X4 append the negotiated speed.
        blind_connect = AtDce(dial=lambda _number: True)
        blind_connect.feed(b"ATE0X0\r")
        self.assertEqual(blind_connect.feed(b"ATD5551212\r"), b"\r\nCONNECT\r\n")

        full_connect = AtDce(dial=lambda _number: True)
        full_connect.feed(b"ATE0X4\r")
        self.assertEqual(full_connect.feed(b"ATD5551212\r"), b"\r\nCONNECT 300\r\n")

        # notify_call_progress (the async path used by real outbound calls)
        # applies the same gating.
        async_blind = AtDce(dial=lambda _number: None)
        async_blind.feed(b"ATE0X0\r")
        async_blind.feed(b"ATD5551212\r")
        async_blind.notify_call_progress("BUSY")
        self.assertEqual(async_blind.take_notifications(), b"\r\nNO CARRIER\r\n")

    def test_ampersand_k_accepts_full_hayes_usr_range(self) -> None:
        dce = AtDce()
        dce.feed(b"ATE0\r")
        for value in range(7):
            self.assertEqual(dce.feed(f"AT&K{value}\r".encode()), b"\r\nOK\r\n")
            self.assertEqual(dce.flow_control, value)
        self.assertEqual(dce.feed(b"AT&K7\r"), b"\r\nERROR\r\n")

    def test_software_flow_control_enabled_for_xon_xoff_modes(self) -> None:
        dce = AtDce()
        for value in (0, 1, 3):
            dce.feed(f"AT&K{value}\r".encode())
            self.assertFalse(dce.software_flow_control_enabled())
        for value in (2, 4, 5, 6):
            dce.feed(f"AT&K{value}\r".encode())
            self.assertTrue(dce.software_flow_control_enabled())

    def test_v250_icf_keeps_byte_stream_backend_at_8n1(self) -> None:
        dce = AtDce()
        self.assertIn(b"+ICF: 3,3", dce.feed(b"AT+ICF?\r"))
        self.assertIn(b"+ICF: (3),(3)", dce.feed(b"AT+ICF=?\r"))
        self.assertTrue(dce.feed(b"AT+ICF=3,3\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"AT+ICF=4\r").endswith(b"ERROR\r\n"))
        self.assertTrue(dce.feed(b"AT+ICF=6,0\r").endswith(b"ERROR\r\n"))

    def test_v250_icf_defers_supported_format_change(self) -> None:
        applied = []
        dce = AtDce(
            set_dte_character_format=lambda fmt, parity: applied.append((fmt, parity)) or True,
            dte_character_formats=(1, 2, 3, 4, 5, 6),
            dte_parity_values=(0, 1),
        )
        self.assertIn(b"+ICF: (1,2,3,4,5,6),(0,1)", dce.feed(b"AT+ICF=?\r"))
        self.assertTrue(dce.feed(b"AT+ICF=5,1\r").endswith(b"OK\r\n"))
        self.assertEqual(applied, [], "format is switched only after result output")
        pending = dce.take_pending_dte_character_format()
        self.assertEqual(pending, (5, 1))
        self.assertTrue(dce._set_dte_character_format(*pending))
        self.assertEqual(applied, [(5, 1)])
        self.assertIn(b"+ICF: 5,1", dce.feed(b"AT+ICF?\r"))
        self.assertTrue(dce.feed(b"AT+ICF=0,1\r").endswith(b"ERROR\r\n"))

    def test_v250_ifc_software_flow_controls_both_directions(self) -> None:
        dce = AtDce()
        self.assertIn(b"+IFC: 0,0", dce.feed(b"AT+IFC?\r"))
        self.assertIn(b"+IFC: (0,1,3),(0,1)", dce.feed(b"AT+IFC=?\r"))
        self.assertTrue(dce.feed(b"AT+IFC=1,1\r").endswith(b"OK\r\n"))
        self.assertEqual(dce.filter_ifc_input(b"a\x13b\x11c"), b"abc")
        self.assertFalse(dce.ifc_dte_paused)
        self.assertTrue(dce.dte_flow_control_enabled())
        self.assertTrue(dce.feed(b"AT+IFC=1,0\r").endswith(b"OK\r\n"))
        self.assertFalse(dce.dte_flow_control_enabled())
        self.assertEqual(dce.filter_ifc_input(b"a\x13b\x11c"), b"abc")
        self.assertFalse(dce.ifc_dte_paused)
        self.assertTrue(dce.feed(b"AT+IFC=0,1\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.dte_flow_control_enabled())
        self.assertEqual(dce.filter_ifc_input(b"a\x13b\x11c"), b"a\x13b\x11c")
        self.assertFalse(dce.ifc_dte_paused)
        self.assertTrue(dce.feed(b"AT+IFC=3,1\r").endswith(b"OK\r\n"))
        self.assertEqual(dce.filter_ifc_input(b"a\x13b\x11c"), b"a\x13b\x11c")
        self.assertFalse(dce.ifc_dte_paused)
        self.assertTrue(dce.feed(b"AT+IFC=1,3\r").endswith(b"ERROR\r\n"))
        self.assertTrue(dce.feed(b"AT+IFC=2,2\r").endswith(b"ERROR\r\n"))
        self.assertTrue(dce.feed(b"AT+IPR?\r").endswith(b"ERROR\r\n"))

    def test_v250_ipr_uses_backend_rates_and_defers_switch(self) -> None:
        current = {"baud": 1200}

        def set_baud(rate: int) -> bool:
            current["baud"] = rate
            return True

        dce = AtDce(set_dte_baud=set_baud, get_dte_baud=lambda: current["baud"],
                    dte_baud_rates=(1200, 2400, 9600))
        self.assertIn(b"+IPR: 1200", dce.feed(b"AT+IPR?\r"))
        self.assertIn(b"+IPR: (1200,2400,9600)", dce.feed(b"AT+IPR=?\r"))
        self.assertTrue(dce.feed(b"AT+IPR=9600\r").endswith(b"OK\r\n"))
        self.assertEqual(current["baud"], 1200, "DCE must defer the physical switch")
        self.assertEqual(dce.take_pending_dte_baud(), 9600)
        self.assertTrue(set_baud(9600))  # run_pty applies it after writing OK
        self.assertEqual(current["baud"], 9600)
        self.assertTrue(dce.feed(b"AT+IPR=14400\r").endswith(b"ERROR\r\n"))
        self.assertIsNone(dce.take_pending_dte_baud())

    def test_v250_ifc_hardware_flow_is_exposed_only_with_modem_lines(self) -> None:
        dce = AtDce(supports_hardware_flow_control=True)
        self.assertEqual(dce.ifc, (2, 2))
        self.assertIn(b"+IFC: (0,1,2,3),(0,1,2)", dce.feed(b"AT+IFC=?\r"))
        self.assertTrue(dce.feed(b"AT+IFC=2,2\r").endswith(b"OK\r\n"))

    def test_bell212_character_format_extension(self) -> None:
        dce = AtDce()
        self.assertIn(b"+BDF: 10", dce.feed(b"AT+BDF?\r"))
        self.assertIn(b"+BDF: (9,10,11,12)", dce.feed(b"AT+BDF=?\r"))
        self.assertTrue(dce.feed(b"AT+BDF=9\r").endswith(b"OK\r\n"))
        self.assertIn(b"+BDF: 9", dce.feed(b"AT+BDF?\r"))
        self.assertTrue(dce.feed(b"AT+BDF=8\r").endswith(b"ERROR\r\n"))
        self.assertIn(b"+BPAR: (E,O,M,S)", dce.feed(b"AT+BPAR=?\r"))
        self.assertTrue(dce.feed(b"AT+BPAR=O\r").endswith(b"OK\r\n"))
        self.assertIn(b"+BPAR: O", dce.feed(b"AT+BPAR?\r"))

    def test_v42_xid_fallback_extension(self) -> None:
        dce = AtDce()
        self.assertIn(b"+XID: 1", dce.feed(b"AT+XID?\r"))
        self.assertIn(b"+XID: (0,1)", dce.feed(b"AT+XID=?\r"))
        self.assertTrue(dce.feed(b"AT+XID=0\r").endswith(b"OK\r\n"))
        self.assertIn(b"+XID: 0", dce.feed(b"AT+XID?\r"))
        self.assertTrue(dce.feed(b"AT+XID=2\r").endswith(b"ERROR\r\n"))

    def test_v42_xid_environment_remains_factory_default_only(self) -> None:
        with mock.patch.dict(os.environ, {"V42_XID_FALLBACK": "0"}):
            dce = AtDce()
        self.assertFalse(dce.v42_xid_fallback)
        dce.feed(b"AT+XID=1\r")
        self.assertTrue(dce.v42_xid_fallback)


class AtProfileTests(unittest.TestCase):
    def test_store_and_recall_profiles(self) -> None:
        import tempfile
        path = tempfile.mktemp(suffix=".json")
        backend = dict(set_dte_character_format=lambda *_: True,
                       dte_character_formats=(1, 2, 3, 4, 5, 6),
                       dte_parity_values=(0, 1, 2, 3))
        dce = AtDce(**backend)
        dce.load_profiles(path)
        dce.feed(b"ATE0S0=2S7=90\\N3\\K3%C1&W\r")
        dce.feed(b"AT+ICF=5,1\r")
        dce.take_pending_dte_character_format()
        dce.feed(b"AT+IFC=1,1\r")
        dce.feed(b"AT+BDF=9\r")
        dce.feed(b"AT+BPAR=O\r")
        dce.feed(b"AT+XID=0\r")
        dce.feed(b"AT+V23BC=0\r")
        dce.feed(b"AT+V23DIR=1\r")
        dce.feed(b"AT&W\r")
        dce.feed(b"ATE1S0=0&F\r")
        self.assertEqual((dce.echo, dce.registers[0]), (True, 0))
        dce.feed(b"ATZ\r")
        self.assertEqual((dce.echo, dce.registers[0], dce.registers[7], dce.error_control, dce.compression),
                         (False, 2, 90, 3, 1))
        self.assertEqual(dce.break_control, 3)
        again = AtDce(**backend)                         # a new power-up reads the file
        again.load_profiles(path)
        self.assertEqual((again.echo, again.registers[7]), (False, 90))
        self.assertEqual(again.ifc, (1, 1))
        self.assertEqual(again.icf, (5, 1))
        self.assertEqual(again.bell212_char_bits, 9)
        self.assertEqual(again.bell212_parity, "O")
        self.assertFalse(again.v42_xid_fallback)
        self.assertFalse(again.v23_backward_channel)
        self.assertEqual(again.v23_software_loop_role, 1)
        self.assertEqual(again.break_control, 3)
        self.assertEqual(again.feed(b"ATZ1\r"), b"\r\nOK\r\n")        # profile 1 was never stored: factory defaults
        self.assertTrue(again.echo)

    def test_hardware_ifc_profile_is_sanitized_for_byte_stream_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvram.json"
            hardware = AtDce(supports_hardware_flow_control=True)
            hardware.load_profiles(path)
            hardware.feed(b"AT+IFC=2,2\r")
            hardware.feed(b"AT&W\r")
            byte_stream = AtDce()
            byte_stream.load_profiles(path)
            self.assertEqual(byte_stream.ifc, (0, 0))

    def test_view_configuration(self) -> None:
        dce = AtDce()
        out = dce.feed(b"AT&V\r")
        self.assertIn(b"El-Modem Active Settings...", out)
        self.assertIn(b"S07=060", out)
        self.assertIn(b"+XID=1", out)
        self.assertIn(b"+V23BC=1", out)
        self.assertIn(b"\\K5", out)
        self.assertIn(b"+V23DIR=0", out)
        self.assertTrue(out.endswith(b"OK\r\n"))

    def test_identification_and_ms(self) -> None:
        dce = AtDce()
        self.assertIn(b"El-Modem Software Modem", dce.feed(b"ATI3\r"))
        self.assertTrue(dce.feed(b"ATI14\r").endswith(b"ERROR\r\n"))
        self.assertTrue(dce.feed(b"AT+GCAP\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"AT+MS=V32B,1,4800,14400\r").endswith(b"OK\r\n"))
        self.assertEqual(dce.ms, {"mod": "V32B", "auto": 1, "min": 4800, "max": 14400})
        self.assertIn(b"+MS: V32B,1,4800,14400", dce.feed(b"AT+MS?\r"))
        self.assertTrue(dce.feed(b"AT+MS=V23,0,600,1200\r").endswith(b"OK\r\n"))
        self.assertIn(b"V23", dce.feed(b"AT+MS=?\r"))
        self.assertTrue(dce.feed(b"AT+MS=X99\r").endswith(b"ERROR\r\n"))
        self.assertIn(b"+V23BC: (0,1)", dce.feed(b"AT+V23BC=?\r"))
        self.assertTrue(dce.feed(b"AT+V23BC=0\r").endswith(b"OK\r\n"))
        self.assertIn(b"+V23BC: 0", dce.feed(b"AT+V23BC?\r"))
        self.assertTrue(dce.feed(b"AT+V23BC=2\r").endswith(b"ERROR\r\n"))
        self.assertIn(b"+V23DIR: (0,1)", dce.feed(b"AT+V23DIR=?\r"))
        self.assertTrue(dce.feed(b"AT+V23DIR=1\r").endswith(b"OK\r\n"))
        self.assertIn(b"+V23DIR: 1", dce.feed(b"AT+V23DIR?\r"))
        self.assertTrue(dce.feed(b"AT+V23DIR=2\r").endswith(b"ERROR\r\n"))
        self.assertIn(b"+V23BC=0", dce.feed(b"AT&V\r"))
        self.assertIn(b"+V23DIR=1", dce.feed(b"AT&V\r"))

class AtiScreenTests(unittest.TestCase):
    def ati(self, dce, n: int) -> str:
        return dce.feed(f"ATI{n}\r".encode()).decode()

    def test_v34_rx_timing_slip_snapshot_uses_equalizer_counter(self) -> None:
        engine = types.SimpleNamespace(
            caller=types.SimpleNamespace(_p4rcv=types.SimpleNamespace(
                eq=types.SimpleNamespace(slips=4))),
            answerer=types.SimpleNamespace(_p4rcv=types.SimpleNamespace(
                eq=types.SimpleNamespace(slips=3))),
        )
        self.assertEqual(v34_rx_timing_slips(engine), {
            "by_core": {"caller": 4, "answerer": 3}, "total": 7,
        })
        self.assertIsNone(v34_rx_timing_slips(types.SimpleNamespace()))

    def test_identification_screens(self) -> None:
        dce = AtDce()
        self.assertEqual(self.ati(dce, 0), "ATI0\r\r\n0030\r\n\r\nOK\r\n")        # code, blank line, OK like a Courier
        self.assertRegex(self.ati(dce, 1), r"\r\n[0-9A-F]{4}\r\n\r\nOK\r\n$")
        self.assertIn("\r\nOK\r\nOK\r\n", self.ati(dce, 2))                    # self-test OK, then the command's own OK
        out = self.ati(dce, 3)
        for text in ("El-Modem Software Modem", "Build date:", "Revision: 0.3.0-rc1", "https://github.com/donfanning/el-modem"):
            self.assertIn(text, out)
        for n in (7, 8, 9, 10, 11, 12, 15, 16, 17):
            self.assertTrue(self.ati(dce, n).endswith("OK\r\n"), n)
        out = self.ati(dce, 7)
        for text in ("El-Modem Configuration...", "Product Type:            Software", "V.34", "EPROM Checksum:          Pass",
                     "https://github.com/donfanning/el-modem"):
            self.assertIn(text, out)
        self.assertIn("ELM0001\\El-Modem Software Modem", self.ati(dce, 9))
        out = self.ati(dce, 10)
        self.assertIn("DIAL SECURITY ACCOUNT STATUS:", out)
        self.assertIn("LOCAL ACCESS:   ENABLED", out)
        self.assertTrue(self.ati(dce, 14).endswith("ERROR\r\n"))

    def test_settings_screen_layout_and_stored_profiles(self) -> None:
        dce = AtDce()
        out = self.ati(dce, 4)
        self.assertIn("El-Modem Active Settings...", out)
        self.assertIn("  S00=000  S01=000  S02=043  S03=013  S04=010  S05=008  S06=002\r\n", out)   # 7 per row
        self.assertIn("  S07=060  S08=002  S09=006  S10=014  S11=070  S12=050  S19=000\r\n", out)
        self.assertIn("  &C1  &D2  &G0  &K3", out)
        self.assertIn("  E1  Q0  V1  X4", out)
        self.assertIn("No stored profile", self.ati(dce, 5))
        dce.feed(b"ATS7=90&W\r")
        stored = self.ati(dce, 5)
        self.assertIn("El-Modem NVRAM Settings...", stored)
        self.assertIn("Stored Profile 0:", stored)
        self.assertIn("S07=090", stored)
        self.assertIn("Stored Phone Numbers:", stored)

    def test_stored_numbers_redial_and_caller_id(self) -> None:
        dce = AtDce()
        self.assertIn("(NO STORED NUMBERS)", self.ati(dce, 8))
        self.assertTrue(dce.feed(b"AT&Z0=5551234\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"AT&Z1=2025550100\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"AT&Z4=1\r").endswith(b"ERROR\r\n"))              # only slots 0-3
        out = self.ati(dce, 8)
        self.assertIn("LIST CAPACITIES: MAXIMUM 4 ENTRIES", out)
        self.assertIn("0: 5551234 DELAY: NONE", out)
        self.assertIn("1: 2025550100 DELAY: NONE", out)
        self.assertIn("BLACK LIST SCREEN: EMPTY", out)
        stored = self.ati(dce, 5)
        self.assertIn("  0= 5551234", stored)
        self.assertIn("  2= \r\n", stored)                                          # empty slots listed too
        self.assertTrue(dce.feed(b"ATDS3\r").endswith(b"ERROR\r\n"))                  # nothing stored in slot 3
        dialed = []
        dce.attach_dial(lambda number: dialed.append(number))
        dce.feed(b"ATDS0\r")
        dce.feed(b"ATDL\r")
        self.assertEqual(dialed, ["5551234", "5551234"])
        self.assertIn("REDIAL: ATDL (5551234)", self.ati(dce, 8))
        self.assertIn("EMPTY", self.ati(dce, 15))
        dce.begin_incoming_call({"number": "2025550100", "name": "Test Caller"})
        out = self.ati(dce, 15)
        self.assertIn("CALLER ID HISTORIC REGISTER:", out)
        self.assertRegex(out, r"DATE=\d{4}\r\nTIME=\d{4}\r\nNMBR=2025550100\r\nNAME=Test Caller")

    def test_numbers_persist_in_nvram(self) -> None:
        import tempfile
        path = tempfile.mktemp(suffix=".json")
        dce = AtDce()
        dce.load_profiles(path)
        dce.feed(b"AT&Z2=5559876\r")
        again = AtDce()
        again.load_profiles(path)
        self.assertEqual(again.stored_numbers, {2: "5559876"})

    def test_link_diagnostics_and_support_reports(self) -> None:
        dce = AtDce()
        self.assertIn("No call yet", self.ati(dce, 6))
        self.assertIn("NO LINK", self.ati(dce, 11))
        dce.feed(b"ATDT5551234\r")
        dce.notify_call_progress("BUSY")
        dce.take_notifications()
        self.assertIn("Not connected: Busy", self.ati(dce, 6))
        dce.connect_speed, dce.connect_suffix, dce.connect_modulation = 33600, "/LAPM", "V.34"
        dce.link_stats = lambda: {"symbol_rate": 3429, "rx_dbm0": -18.25, "error_control": "V.42 LAPM",
                                  "compression": "V.42BIS",
                                  "v34_rx_timing_slips": {"by_core": {"caller": 7}, "total": 7},
                                  "blocks": dict(i_tx=100, i_rx=90, i_resent=5, rej_tx=1, rej_rx=2, fcs_errors=3,
                                                 t401_expiries=0, disc_tx=2, disc_rx=1,
                                                 ua_tx=1, ua_rx=1, dm_tx=0, dm_rx=0,
                                                 bytes_in=4000, bytes_out=3500, wire_in=2000,
                                                 wire_out=3500)}
        dce.feed(b"ATDT5559999\r")
        dce.notify_call_progress("CONNECT")
        dce.take_notifications()
        dce.count_chars(sent=120, received=340)
        live = self.ati(dce, 11)
        self.assertIn("TECHNICAL SUPPORT REPORT - SECTION 1", live)
        self.assertIn("MODULATION: V.34", live)
        self.assertIn("SYMBOL RATE: 3429", live)
        self.assertIn("RX LEVEL: -18.2 dBm0", live)
        self.assertIn("SNR: N/A (NOT MEASURED)", live)                         # never invented
        self.assertIn("LINE QUALITY: 95.0% OF BLOCKS CLEAN", live)
        section2 = self.ati(dce, 16)
        self.assertIn("TECHNICAL SUPPORT REPORT - SECTION 2", section2)
        self.assertIn("RESENT 5", section2)
        self.assertIn("RELEASE: DISC TX/RX 2/1  UA TX/RX 1/1  DM TX/RX 0/0", section2)
        self.assertIn("RX COMPRESSION: 2.00:1", section2)
        self.assertIn("TRELLIS ERRORS: N/A   RX TIMING SLIPS: 7", section2)
        self.assertIn("NOT IMPLEMENTED", self.ati(dce, 17))
        dce.notify_call_progress("NO CARRIER")
        dce.take_notifications()
        done = self.ati(dce, 6)
        self.assertIn("El-Modem Link Diagnostics...", done)
        self.assertRegex(done, r"Chars sent\s+120\s+Chars received\s+340")
        self.assertRegex(done, r"Chars lost\s+N/A")
        self.assertRegex(done, r"Blocks sent\s+100\s+Blocks received\s+90")
        self.assertRegex(done, r"Blocks resent\s+5")
        self.assertRegex(done, r"Last Tx rate\s+33600\s+Last Rx rate\s+33600")
        self.assertRegex(done, r"Modulation\s+V\.34\s+Protocol\s+V\.42")
        self.assertRegex(done, r"Compression\s+V\.42 bis\s+Line Conditioned\s+Ok")
        self.assertRegex(done, r"Retrains Requested\s+N/A\s+Retrains Granted\s+N/A")
        self.assertRegex(done, r"Fallbacks\s+N/A")
        self.assertIn("Remote Hangup / No Carrier", done)
        self.assertIn("RESENT 5", self.ati(dce, 16))                            # frozen at the end of the call

    def test_ati16_keeps_unavailable_receiver_counters_explicit(self) -> None:
        dce = AtDce()
        dce.connect_speed, dce.connect_suffix, dce.connect_modulation = 1200, "", "V.22"
        dce.link_stats = lambda: {"error_control": "NONE", "compression": "NONE"}
        dce.feed(b"ATDT5551234\r")
        dce.notify_call_progress("CONNECT")
        dce.take_notifications()
        report = self.ati(dce, 16)
        self.assertIn("TRELLIS ERRORS: N/A   RX TIMING SLIPS: N/A", report)

    def test_help_screens(self) -> None:
        dce = AtDce()
        for cmd, needle in ((b"AT$", "HELP - BASIC COMMANDS:"), (b"AT&$", "HELP - AMPERSAND COMMANDS:"),
                            (b"ATS$", "HELP - S-REGISTERS:"), (b"AT%$", "HELP - PERCENT COMMANDS:"),
                            (b"ATD$", "HELP - DIAL MODIFIERS:"), (b"AT\\$", "HELP - BACKSLASH COMMANDS:"),
                            (b"AT+$", "HELP - EXTENDED COMMANDS:")):
            out = dce.feed(cmd + b"\r").decode()
            self.assertIn(needle, out, cmd)
            self.assertTrue(out.endswith("OK\r\n"), cmd)
        self.assertIn("S100-S106 SIP", dce.feed(b"ATS$\r").decode())
        self.assertTrue(dce.feed(b"AT#$\r").endswith(b"ERROR\r\n"))              # voice/V.92 families are not implemented

    def test_escape_registers_drive_the_detector(self) -> None:
        dce = AtDce()
        dce.feed(b"ATS2=126S12=100\r")
        self.assertEqual((dce.registers[2], dce.registers[12]), (126, 100))


class NetworkSettingsTests(unittest.TestCase):
    def test_set_read_and_mask(self) -> None:
        dce = AtDce()
        self.assertTrue(dce.feed(b"ATS100=Sip.Example.com\r").endswith(b"OK\r\n"))
        self.assertIn(b"Sip.Example.com", dce.feed(b"ATS100?\r"))                # case kept for the value
        dce.feed(b'ATS102=Alice\rATS103="Pa55 Word"\r')
        self.assertEqual(dce.net[102], "Alice")
        self.assertEqual(dce.net[103], "Pa55 Word")
        self.assertIn(b"********", dce.feed(b"ATS103?\r"))
        self.assertNotIn(b"Pa55", dce.feed(b"ATS103?\r"))
        self.assertNotIn(b"Pa55", dce.feed(b"ATI12\r") + dce.feed(b"ATI4\r") + dce.feed(b"AT&V\r"))

    def test_validation(self) -> None:
        dce = AtDce()
        for bad in (b"ATS101=0\r", b"ATS101=70000\r", b"ATS101=abc\r", b"ATS120=2\r", b"ATS115=yes\r", b"ATS150=x\r",
                    b"ATS100\r"):
            self.assertTrue(dce.feed(bad).endswith(b"ERROR\r\n"), bad)
        self.assertTrue(dce.feed(b"ATS101=5070\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"ATS101?\r").startswith(b"ATS101?\r\r\n5070"))

    def test_screens_and_defaults(self) -> None:
        dce = AtDce()
        dce.net_defaults = {100: "pbx.lan", 102: "modem1"}
        dce._factory_reset()
        out = dce.feed(b"ATI12\r").decode()
        self.assertIn("S100: SIP SERVER = pbx.lan", out)
        self.assertIn("S101: SIP PORT = 5060", out)
        self.assertIn("S111: IAX PORT = 4569", out)
        self.assertIn("RFC 5456 / G.711 U-LAW", out)
        self.assertIn("S102: SIP USER = modem1", dce.feed(b"ATI4\r").decode())

    def test_profiles_store_network_and_notify(self) -> None:
        import os
        import stat
        import tempfile
        path = tempfile.mktemp(suffix=".json")
        dce = AtDce()
        changes = []
        dce.on_network_change = lambda: changes.append(dict(dce.net))
        dce.load_profiles(path)
        dce.feed(b"ATS104=reg.example.com:5061\rATS102=bob\rATS103=secret\rAT&W\r")
        self.assertTrue(changes and changes[-1][104] == "reg.example.com:5061")
        # Windows access uses inherited ACLs; its synthetic st_mode is not a
        # POSIX permission check. ACL qualification is separate from readback.
        if os.name == 'posix':
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        again = AtDce()
        again.load_profiles(path)
        self.assertEqual((again.net[102], again.net[103], again.net[104]), ("bob", "secret", "reg.example.com:5061"))
        dce.feed(b"ATS100=other\rATZ\r")                                         # ATZ reloads the stored profile
        self.assertEqual(dce.net[100], "")
        self.assertEqual(changes[-1][102], "bob")


class AtxLevelTests(unittest.TestCase):
    def results(self, level: int, result: str) -> bytes:
        dce = AtDce()
        dce.feed(f"ATE0X{level}\r".encode())
        dce.notify_call_progress(result)
        return dce.take_notifications()

    def test_busy_and_no_dialtone_by_level(self) -> None:
        # X0/X1: neither is reported; X2: NO DIALTONE only; X3: BUSY only; X4: both.
        expect = {
            0: (b"NO CARRIER", b"NO CARRIER"), 1: (b"NO CARRIER", b"NO CARRIER"),
            2: (b"NO CARRIER", b"NO DIALTONE"), 3: (b"BUSY", b"NO CARRIER"), 4: (b"BUSY", b"NO DIALTONE"),
        }
        for level, (busy, dialtone) in expect.items():
            with self.subTest(level=level):
                self.assertIn(busy, self.results(level, "BUSY"))
                self.assertIn(dialtone, self.results(level, "NO DIALTONE"))

    def test_numeric_forms_and_connect_text(self) -> None:
        dce = AtDce()
        dce.feed(b"ATV0X4\r")
        dce.notify_call_progress("NO DIALTONE")
        self.assertEqual(dce.take_notifications(), b"\r\n6\r\n")
        dce.notify_call_progress("BUSY")
        self.assertEqual(dce.take_notifications(), b"\r\n7\r\n")
        dce = AtDce()
        dce.feed(b"ATX0\r")
        dce.connect_speed = 9600
        dce.notify_call_progress("CONNECT")
        self.assertEqual(dce.take_notifications(), b"\r\nCONNECT\r\n")        # X0: no speed
        dce = AtDce()
        dce.connect_speed = 9600
        dce.notify_call_progress("CONNECT")
        self.assertEqual(dce.take_notifications(), b"\r\nCONNECT 9600\r\n")

    def test_sip_failures_map_to_results(self) -> None:
        from modem_dce import sip_failure_result as f
        self.assertEqual([f(r) for r in ("sip 486", "sip 600", "sip 603")], ["BUSY"] * 3)
        self.assertEqual([f(r) for r in ("no line", "no response", "sip 503")], ["NO DIALTONE"] * 3)
        self.assertEqual([f(r) for r in ("timeout", "sip 480")], ["NO ANSWER"] * 2)
        self.assertEqual([f(r) for r in ("sip 404", "sip 403")], ["NO CARRIER"] * 2)


class MsSelectionTests(unittest.TestCase):
    def test_v34_rate_bounds_survive_direct_and_automode_selection(self) -> None:
        from modem_dce import line_from_ms
        from v8 import V8_MOD_V34
        for min_index in range(1, 15):
            for max_index in range(min_index, 15):
                low, high = min_index * 2400, max_index * 2400
                direct = line_from_ms({"mod": "V34", "auto": 0, "min": low, "max": high})
                self.assertEqual((direct["v34_min_rate"], direct["v34_rate"]), (low, high))
                automatic = line_from_ms({"mod": "V34", "auto": 1, "min": low, "max": high})["chain"]
                self.assertEqual((automatic["v34_min_rate"], automatic["v34_rate"]), (low, high))
                self.assertTrue(automatic["allow"] & V8_MOD_V34)

    def test_ms_maps_to_engines(self) -> None:
        from modem_dce import line_from_ms
        self.assertEqual(line_from_ms({"mod": "B212", "auto": 0, "min": 300, "max": 1200}),
                         {"line_mode": "bell212a", "v22_rate": 1200, "v22_min_rate": 300})
        self.assertEqual(line_from_ms({"mod": "B212", "auto": 0, "min": 1200, "max": 1200})["v22_min_rate"], 1200)
        with self.assertRaisesRegex(ValueError, "Bell 212A currently requires a 1200"):
            line_from_ms({"mod": "B212", "auto": 0, "min": 300, "max": 600})
        with self.assertRaisesRegex(ValueError, "Bell 212A currently requires a 1200"):
            line_from_ms({"mod": "B212", "auto": 0, "min": 2400, "max": 33600})
        self.assertEqual(line_from_ms({"mod": "V22", "auto": 0, "min": 300, "max": 1200}),
                         {"line_mode": "v22bis", "v22_rate": 1200, "v22_min_rate": 300})
        self.assertEqual(line_from_ms({"mod": "V22", "auto": 0, "min": 300, "max": 600}),
                         {"line_mode": "v22bis", "v22_rate": 600, "v22_min_rate": 300})
        with self.assertRaisesRegex(ValueError, "V.22 requested range excludes"):
            line_from_ms({"mod": "V22", "auto": 0, "min": 1300, "max": 2000})
        self.assertEqual(line_from_ms({"mod": "V22B", "auto": 0, "min": 2400, "max": 2400}),
                         {"line_mode": "v22bis", "v22_rate": 2400, "v22_min_rate": 2400})
        with self.assertRaisesRegex(ValueError, "excludes supported rates"):
            line_from_ms({"mod": "V22B", "auto": 0, "min": 1300, "max": 2000})
        chain_2400 = line_from_ms({"mod": "V22B", "auto": 1, "min": 2400, "max": 2400})["chain"]
        self.assertEqual((chain_2400["v22_rate"], chain_2400["v22_min_rate"]), (2400, 2400))
        chain_v22_only = line_from_ms({"mod": "V32", "auto": 1, "min": 1000, "max": 2000})["chain"]
        self.assertEqual(chain_v22_only["allow"], 1 << 2)
        self.assertEqual(chain_v22_only["v32_rates"], ())
        chain_v22_max = line_from_ms({"mod": "V34", "auto": 1, "min": 300, "max": 1200})["chain"]
        self.assertEqual(chain_v22_max["allow"], 1 << 2)
        with self.assertRaisesRegex(ValueError, "no supported modulation rate"):
            line_from_ms({"mod": "V32", "auto": 1, "min": 12000, "max": 14400})
        chain_32 = line_from_ms({"mod": "V32B", "auto": 1, "min": 4800, "max": 14400})["chain"]
        self.assertEqual(chain_32["allow"], 1 << 9)  # V.22 fallback is below the requested minimum.
        self.assertEqual(line_from_ms({"mod": "V23", "auto": 0, "min": 300, "max": 1200}),
                         {"line_mode": "v23", "v23_rate": 1200})
        self.assertEqual(line_from_ms({"mod": "V23", "auto": 0, "min": 300, "max": 600}),
                         {"line_mode": "v23", "v23_rate": 600})
        with self.assertRaisesRegex(ValueError, "no V.8 automode"):
            line_from_ms({"mod": "V23", "auto": 1, "min": 300, "max": 1200})
        with self.assertRaisesRegex(ValueError, "600- or 1200-baud"):
            line_from_ms({"mod": "V23", "auto": 0, "min": 300, "max": 300})
        with self.assertRaisesRegex(ValueError, "Bell 103 supports 300"):
            line_from_ms({"mod": "B103", "auto": 0, "min": 1200, "max": 1200})
        with self.assertRaisesRegex(ValueError, "V.21 supports 300"):
            line_from_ms({"mod": "V21", "auto": 0, "min": 600, "max": 1200})
        self.assertEqual(line_from_ms({"mod": "V32B", "auto": 0, "min": 4800, "max": 14400})["v32_rates"],
                         (14400, 12000, 9600, 7200, 4800))
        with self.assertRaisesRegex(ValueError, "V32 requested range excludes"):
            line_from_ms({"mod": "V32", "auto": 0, "min": 12000, "max": 14400})
        self.assertEqual(line_from_ms({"mod": "V32", "auto": 0, "min": 4800, "max": 33600})["v32_rates"],
                         (9600, 7200, 4800))
        chain = line_from_ms({"mod": "V34", "auto": 1, "min": 300, "max": 14400})
        self.assertEqual((chain["line_mode"], chain["chain"]["v34_rate"]), ("chain", 14400))
        self.assertEqual(chain["chain"]["v34_min_rate"], 2400)
        bounded = line_from_ms({"mod": "V34", "auto": 0, "min": 9600, "max": 24000})
        self.assertEqual((bounded["v34_min_rate"], bounded["v34_rate"]), (9600, 24000))
        with self.assertRaisesRegex(ValueError, "V.34 requested range"):
            line_from_ms({"mod": "V34", "auto": 0, "min": 10000, "max": 11000})
        bounded_auto = line_from_ms({"mod": "V34", "auto": 1, "min": 9600, "max": 24000})["chain"]
        self.assertEqual((bounded_auto["v34_min_rate"], bounded_auto["v34_rate"]), (9600, 24000))
        self.assertEqual(line_from_ms({"mod": "V32B", "auto": 1, "min": 4800, "max": 14400})["chain"]["allow"],
                         1 << 9)
        self.assertEqual(line_from_ms({"mod": "V21", "auto": 0, "min": 300, "max": 300}), {"line_mode": "v21"})


class DialNormalizationTests(unittest.TestCase):
    def test_strips_tone_and_pulse_dial_modifiers(self) -> None:
        self.assertEqual(normalize_dial_number("T106"), "106")
        self.assertEqual(normalize_dial_number("P5551234"), "5551234")

    def test_strips_pauses_flash_and_wait_for_dialtone(self) -> None:
        self.assertEqual(normalize_dial_number("5551234,,,9"), "55512349")
        self.assertEqual(normalize_dial_number("W2025550100"), "2025550100")
        self.assertEqual(normalize_dial_number("!5551234"), "5551234")

    def test_leaves_plain_digits_and_dtmf_symbols_untouched(self) -> None:
        self.assertEqual(normalize_dial_number("106"), "106")
        self.assertEqual(normalize_dial_number("*72#"), "*72#")


@unittest.skipIf(os.name == "nt", "PTY is POSIX only")
class PtySmokeTests(unittest.TestCase):
    def _read_until(self, fd: int, marker: bytes, timeout: float = 3) -> bytes:
        result = bytearray()
        deadline = time.monotonic() + timeout
        while marker not in result and time.monotonic() < deadline:
            readable, _, _ = select.select([fd], [], [], 0.1)
            if readable:
                result.extend(os.read(fd, 1024))
        self.assertIn(marker, result)
        return bytes(result)

    def test_daemon_exposes_terminal_usable_pty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="modem-pty-") as directory:
            link = Path(directory) / "modem0"
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--link", str(link)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            fd = -1
            try:
                deadline = time.monotonic() + 3
                while not link.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(f"PTY daemon exited with status {process.returncode}")
                    time.sleep(0.02)
                self.assertTrue(link.exists(), "PTY symlink was not created")
                fd = os.open(link, os.O_RDWR | os.O_NOCTTY)
                tty.setraw(fd)

                os.write(fd, b"AT\r")
                self.assertIn(b"OK\r\n", self._read_until(fd, b"OK\r\n"))
                os.write(fd, b"ATE0\r")
                self.assertIn(b"OK\r\n", self._read_until(fd, b"OK\r\n"))
                os.write(fd, b"ATD123\r")
                response = self._read_until(fd, b"NO CARRIER\r\n")
                self.assertNotIn(b"CONNECT", response)
            finally:
                if fd >= 0:
                    os.close(fd)
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


class TcpSerialStandaloneSmokeTests(unittest.TestCase):
    def test_standalone_tcp_serial_answers_at_without_network_configuration(self) -> None:
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--serial-tcp", f"127.0.0.1:{port}",
             "--profile-file", ""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        connection = None
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stderr = process.stderr.read().decode(errors="replace")
                    self.fail(f"standalone TCP serial modem exited: {stderr}")
                try:
                    connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                    break
                except OSError:
                    time.sleep(0.05)
            self.assertIsNotNone(connection, "modem did not publish its TCP serial endpoint")
            connection.settimeout(2)
            connection.sendall(b"ATI3\r")
            response = bytearray()
            while b"OK\r\n" not in response:
                block = connection.recv(1024)
                self.assertTrue(block, "modem closed TCP serial before replying")
                response.extend(block)
            self.assertIn(b"donfanning/el-modem", response)
        finally:
            if connection is not None:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            if process.stderr is not None:
                process.stderr.close()


class BusyToneTests(unittest.TestCase):
    def _tone(self, duration: int, enabled: bool) -> bytes:
        samples = []
        for index in range(duration):
            if enabled:
                value = round(
                    6000
                    * math.sin(2 * math.pi * 480 * index / 8000)
                    + 6000 * math.sin(2 * math.pi * 620 * index / 8000)
                )
            else:
                value = 0
            samples.append(max(-32768, min(32767, value)))
        return struct.pack(f"<{len(samples)}h", *samples)

    def test_detects_repeated_busy_cadence(self) -> None:
        detector = BusyToneDetector()
        audio = b"".join(
            self._tone(4000, state)
            for state in (True, False, True, False, True)
        )
        self.assertTrue(detector.feed_pcm(audio))

    def test_single_tone_is_not_busy(self) -> None:
        detector = BusyToneDetector()
        self.assertFalse(detector.feed_pcm(self._tone(4000, True)))

    def _ringback_tone(self, duration: int, enabled: bool) -> bytes:
        samples = []
        for index in range(duration):
            value = (
                round(
                    6000 * math.sin(2 * math.pi * 440 * index / 8000)
                    + 6000 * math.sin(2 * math.pi * 480 * index / 8000)
                )
                if enabled
                else 0
            )
            samples.append(max(-32768, min(32767, value)))
        return struct.pack(f"<{len(samples)}h", *samples)

    def test_detects_ringback_cadence_from_audio(self) -> None:
        detector = RingbackToneDetector()
        one_burst = self._ringback_tone(16_000, True) + self._ringback_tone(32_000, False)
        self.assertFalse(detector.feed_pcm(one_burst))
        second_burst = self._ringback_tone(16_000, True)
        self.assertTrue(detector.feed_pcm(second_burst))

    def test_outgoing_progress_maps_detected_ringback_to_serial_rings(self) -> None:
        dce = AtDce(dial=lambda _number: None)
        dce.feed(b"ATE0\r")
        self.assertEqual(dce.feed(b"ATD5551212\r"), b"")
        self.assertTrue(dce.outgoing_pending)
        monitor = OutgoingCallProgressMonitor(dce.notify_call_progress)
        audio = self._ringback_tone(16_000, True) + self._ringback_tone(32_000, False)
        audio += self._ringback_tone(16_000, True) + self._ringback_tone(32_000, False)
        audio += self._ringback_tone(16_000, True)
        self.assertEqual(monitor.feed_pcm(audio), "RING")
        self.assertEqual(dce.take_notifications(), b"\r\nRING\r\n\r\nRING\r\n")
        self.assertTrue(dce.outgoing_pending)

    def test_bell103_fsk_is_not_misclassified_as_voice(self) -> None:
        detector = VoiceActivityDetector()
        samples = []
        phase = 0.0
        payload = b"MODEM TEST 1234567890\r"
        symbols = [True] * 90
        for value in payload:
            symbols.extend([False])
            symbols.extend(bool(value & (1 << bit)) for bit in range(8))
            symbols.append(True)
        total_samples = round(len(symbols) * 8000 / 300)
        for sample_index in range(total_samples):
            symbol_index = min(int(sample_index * 300 / 8000), len(symbols) - 1)
            frequency = 1270 if symbols[symbol_index] else 1070
            samples.append(round(12000 * math.sin(phase)))
            phase = (phase + 2 * math.pi * frequency / 8000) % (2 * math.pi)
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        self.assertFalse(detector.feed_pcm(pcm))

    def test_voice_activity_rejects_modem_carrier_and_detects_speech_like_signal(self) -> None:
        detector = VoiceActivityDetector()
        pure_tone = self._tone(160 * 8, True)
        self.assertFalse(detector.feed_pcm(pure_tone))
        busy_dual_tone = self._tone(160 * 8, True)
        self.assertFalse(detector.feed_pcm(busy_dual_tone))

        speech = []
        for sample_index in range(160 * 8):
            frame_index, local_index = divmod(sample_index, 160)
            fundamental = 150 + (frame_index % 4) * 15
            value = sum(
                (5000 // harmonic)
                * math.sin(
                    2
                    * math.pi
                    * fundamental
                    * harmonic
                    * local_index
                    / 8000
                )
                for harmonic in range(2, 17)
            )
            speech.append(max(-32768, min(32767, round(value))))
        self.assertTrue(detector.feed_pcm(struct.pack(f"<{len(speech)}h", *speech)))

    def test_outgoing_progress_monitor_reports_voice_and_busy_once(self) -> None:
        outcomes: list[str] = []
        monitor = OutgoingCallProgressMonitor(outcomes.append)
        speech = []
        for sample_index in range(160 * 8):
            frame_index, local_index = divmod(sample_index, 160)
            fundamental = 150 + (frame_index % 4) * 15
            value = sum(
                (5000 // harmonic)
                * math.sin(
                    2 * math.pi * fundamental * harmonic * local_index / 8000
                )
                for harmonic in range(2, 17)
            )
            speech.append(max(-32768, min(32767, round(value))))
        speech_pcm = struct.pack(f"<{len(speech)}h", *speech)
        self.assertEqual(monitor.feed_pcm(speech_pcm), "VOICE")
        self.assertIsNone(monitor.feed_pcm(speech_pcm))
        self.assertEqual(outcomes, ["VOICE"])

        busy_outcomes: list[str] = []
        busy_monitor = OutgoingCallProgressMonitor(busy_outcomes.append)
        busy_pcm = b"".join(
            self._tone(4000, state)
            for state in (True, False, True, False, True)
        )
        self.assertEqual(busy_monitor.feed_pcm(busy_pcm), "BUSY")
        self.assertEqual(busy_outcomes, ["BUSY"])


class LinearUlawRoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_small_samples_within_quantization(self) -> None:
        for sample in (0, 100, -100, 2000, -2000, 16000, -16000, 32000, -32000):
            encoded = linear_to_ulaw(sample)
            decoded = ulaw_to_linear(encoded)
            self.assertLessEqual(abs(decoded - sample), max(1, abs(sample) // 32 + 4))


class EscapeSequenceDetectorTests(unittest.TestCase):
    def test_plus_plus_plus_with_guard_time_is_detected(self) -> None:
        detector = EscapeSequenceDetector(guard_time=1.0)
        now = 100.0
        passthrough = detector.feed(b"+++", now)
        self.assertEqual(passthrough, b"")
        self.assertFalse(detector.poll(now + 0.5))
        self.assertTrue(detector.poll(now + 1.1))

    def test_plus_plus_plus_without_leading_guard_time_passes_through(self) -> None:
        detector = EscapeSequenceDetector(guard_time=1.0)
        detector.feed(b"hello", 0.0)
        passthrough = detector.feed(b"+++", 0.1)
        self.assertEqual(passthrough, b"+++")
        self.assertFalse(detector.poll(2.0))

    def test_plus_plus_plus_without_trailing_guard_time_passes_through(self) -> None:
        detector = EscapeSequenceDetector(guard_time=1.0)
        detector.feed(b"+++", 10.0)
        passthrough = detector.feed(b"x", 10.2)
        self.assertEqual(passthrough, b"+++x")
        self.assertFalse(detector.poll(20.0))

    def test_fourth_plus_cancels_the_candidate(self) -> None:
        detector = EscapeSequenceDetector(guard_time=1.0)
        detector.feed(b"+++", 5.0)
        passthrough = detector.feed(b"+", 5.05)
        self.assertEqual(passthrough, b"++++")
        self.assertFalse(detector.poll(10.0))

    def test_ordinary_data_passes_through_untouched(self) -> None:
        detector = EscapeSequenceDetector(guard_time=1.0)
        self.assertEqual(detector.feed(b"ATDT5551234\r", 0.0), b"ATDT5551234\r")


class EscapeHoldTests(unittest.TestCase):
    def test_holds_in_order_and_flushes_once(self) -> None:
        hold = EscapeHold()
        self.assertEqual(len(hold), 0)
        hold.hold(b"abc")
        hold.hold(b"def")
        self.assertEqual(len(hold), 6)
        self.assertEqual(hold.flush(), b"abcdef")
        self.assertEqual(len(hold), 0)
        self.assertEqual(hold.flush(), b"")

    def test_bounded_keeps_the_most_recent(self) -> None:
        hold = EscapeHold()
        hold.hold(b"x" * (EscapeHold.LIMIT + 100))
        self.assertEqual(len(hold), EscapeHold.LIMIT)
        hold.hold(b"tail")
        self.assertTrue(bytes(hold.flush()).endswith(b"tail"))

    def test_clear_drops_without_delivering(self) -> None:
        hold = EscapeHold()
        hold.hold(b"abc")
        hold.clear()
        self.assertEqual(hold.flush(), b"")

    def test_empty_hold_is_free(self) -> None:
        hold = EscapeHold()
        hold.hold(b"")
        self.assertEqual(len(hold), 0)


class DialAbortTests(unittest.TestCase):
    def test_fires_only_before_connect(self) -> None:
        call = object()
        session = object()
        self.assertTrue(dial_abort_applies(call, None, False, False),
                        "originate attempt in flight")
        self.assertTrue(dial_abort_applies(None, None, False, False, iax_dial_pending=True),
                        "IAX originate attempt in flight")
        self.assertTrue(dial_abort_applies(None, session, False, True),
                        "session up but CONNECT held for negotiation")
        self.assertFalse(dial_abort_applies(None, None, False, False),
                         "idle line: keys are commands, not aborts")
        self.assertFalse(dial_abort_applies(None, session, False, False),
                         "negotiation done: keys are payload, not aborts")
        self.assertFalse(dial_abort_applies(call, session, True, False),
                         "online: keys are payload, not aborts")
        self.assertFalse(dial_abort_applies(call, None, True, True),
                         "online wins over everything")


class V42BreakDteOrderTests(unittest.TestCase):
    class Serial:
        def __init__(self):
            self.queued = 12
            self.bytes_after_break = b"already-sent"
            self.discarded = False

        def output_pending(self):
            return self.queued

        def discard_output(self):
            self.discarded = True
            self.queued = 0
            self.bytes_after_break = b""
            return True

        def place_break_before_pending_output(self):
            self.queued = 0
            return True

    def test_in_sequence_waits_for_data_to_reach_dte(self):
        serial = self.Serial()
        event = {"destructive": False, "expedited": False}
        held = bytearray(b"held")
        self.assertTrue(prepare_v42_break_dte_order(event, serial, held))
        self.assertFalse(v42_break_can_be_delivered(event, serial, held))
        serial.queued = 0
        held.clear()
        self.assertTrue(v42_break_can_be_delivered(event, serial, held))

    def test_destructive_discards_every_undelivered_output_queue(self):
        serial = self.Serial()
        event = {"destructive": True, "expedited": False}
        held = bytearray(b"held")
        self.assertTrue(prepare_v42_break_dte_order(event, serial, held))
        self.assertEqual(held, bytearray())
        self.assertTrue(serial.discarded)
        self.assertTrue(v42_break_can_be_delivered(event, serial, held))

    def test_expedited_requeues_undelivered_output_after_break(self):
        serial = self.Serial()
        event = {"destructive": False, "expedited": True}
        held = bytearray(b"held")
        self.assertTrue(prepare_v42_break_dte_order(event, serial, held))
        self.assertEqual(held, bytearray(b"held"))
        self.assertEqual(serial.bytes_after_break, b"already-sent")
        self.assertTrue(event["dte_action_prepared"])
        self.assertTrue(v42_break_can_be_delivered(event, serial, held))

    def test_byte_transport_cannot_ack_expedited_break(self):
        serial = types.SimpleNamespace(output_pending=lambda: 0)
        event = {"destructive": False, "expedited": True}
        self.assertFalse(prepare_v42_break_dte_order(event, serial, bytearray()))
        self.assertFalse(v42_break_can_be_delivered(event, serial, bytearray()))


class OnlineAudioSessionTests(unittest.TestCase):
    def test_c_error_recovery_requires_explicit_v42bis_opt_in(self) -> None:
        class FakeLine:
            supports_v42 = True
            bell103_fallback = False
            connected = True
            capture_detect_bits = False
            v42_mode = "off"
            v42_resolved = False
            bit_rate = 1200
            events = []

            def is_settled(self):
                return True

            def start_v42(self, *args, **kwargs):
                self.start_args = args
                self.start_kwargs = kwargs

        def new_session(v42bis):
            session = OnlineAudioSession.__new__(OnlineAudioSession)
            session._v22 = FakeLine()
            session._auto = None
            session._want_v42 = True
            session._want_mnp = False
            session._v42bis = v42bis
            session._v42_started = False
            session._v42_frame_size = 32
            session._v42_xid_fallback = True
            session._tx_role = "originate"
            session._v42_probe = False
            session._was_up = False
            session._carrier_down_at = None
            session.debug = False
            session._online_control = None
            session._negotiator = None
            return session

        with mock.patch.dict(os.environ, {}, clear=True):
            default_session = new_session(v42bis=True)
            default_session.poll(time.monotonic())
            self.assertFalse(default_session._v22.start_kwargs["compression_recovery"])

        with mock.patch.dict(os.environ, {"V42_CERROR_RECOVERY": "1"}, clear=True):
            enabled_session = new_session(v42bis=True)
            enabled_session.poll(time.monotonic())
            self.assertTrue(enabled_session._v22.start_kwargs["compression_recovery"])

            no_bis_session = new_session(v42bis=False)
            no_bis_session.poll(time.monotonic())
            self.assertFalse(no_bis_session._v22.start_kwargs["compression_recovery"])

    def test_v23_dcd_tracks_receive_carrier_and_honors_force_on(self) -> None:
        from v23 import V23Modem

        for calling_receiver, required_ms in ((False, 300), (True, 40)):
            with self.subTest(receive_channel="backward" if calling_receiver else "forward"):
                transmitter = V23Modem(not calling_receiver, forward_baud=1200)
                receiver = V23Modem(calling_receiver, forward_baud=1200)
                session = OnlineAudioSession.__new__(OnlineAudioSession)
                session._v22 = receiver
                self.assertFalse(session.dte_dcd_state(online=False))
                self.assertFalse(session.dte_dcd_state(online=True))

                for _ in range(required_ms // 10):
                    frame = struct.pack("<80h", *transmitter.tx_samples(80))
                    receiver.feed_pcm(frame)
                self.assertTrue(receiver.connected)
                # Carrier is visible before CONNECT is reported to the DTE.
                self.assertTrue(session.dte_dcd_state(online=False))
                self.assertTrue(session.dte_dcd_state(online=True))

                # A full detector window of hard silence drops circuit 109/122.
                silence_samples = receiver._carrier_block_samples
                silence = bytes(silence_samples * 2)
                receiver.feed_pcm(silence)
                self.assertFalse(receiver.connected)
                self.assertTrue(session.dte_dcd_state(online=True, output_pending=True))
                self.assertFalse(session.dte_dcd_state(online=True, output_pending=False))
                self.assertTrue(session.dte_dcd_state(online=False, force_on=True))

    def test_dcd_drop_waits_for_pending_received_data_to_drain(self) -> None:
        session = OnlineAudioSession.__new__(OnlineAudioSession)
        session._v22 = types.SimpleNamespace(connected=True)
        self.assertTrue(session.dte_dcd_state(online=True))

        session._v22.connected = False
        self.assertTrue(session.dte_dcd_state(online=False, output_pending=True))
        self.assertFalse(session.dte_dcd_state(online=False, output_pending=False))

    def test_non_v23_dcd_keeps_existing_online_behavior(self) -> None:
        session = OnlineAudioSession.__new__(OnlineAudioSession)
        session._v22 = types.SimpleNamespace(connected=False)
        self.assertTrue(session.dte_dcd_state(online=True))
        self.assertFalse(session.dte_dcd_state(online=False))
        self.assertTrue(session.dte_dcd_state(online=False, force_on=True))

    def test_link_stats_exposes_v34_rx_timing_slips(self) -> None:
        session = OnlineAudioSession.__new__(OnlineAudioSession)
        caller = types.SimpleNamespace(
            _p4rcv=types.SimpleNamespace(eq=types.SimpleNamespace(slips=5)),
            _rx=[],
        )
        session._v22 = types.SimpleNamespace(
            caller=caller, answerer=None, v42_mode="off", v42_compression=None,
            rx_signal_power_dbm0=-20.0,
        )
        session._rtp_frame_seconds = 0.02
        session._v23_detector_transitions = []
        session._timer_resolution_request = None
        session.realtime = types.SimpleNamespace(report=lambda: {})
        session.timing_trace = None
        session._want_mnp = False
        session._rtp_counts = {}
        stats = session.link_stats()
        self.assertEqual(stats["v34_rx_timing_slips"], {
            "by_core": {"caller": 5}, "total": 5,
        })

    def test_v22bis_dte_payload_waits_for_retrain_transmit_readiness(self) -> None:
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode="v22bis", v22_rate=2400,
        )
        try:
            engine = session._v22
            engine.connected = True
            engine._state = "retrain"
            engine._rx_ready = False
            engine._tx_mode = "s1"
            payload = b"queued during V.22bis retrain"
            session.feed_terminal_bytes(payload)
            self.assertEqual(bytes(session._held), payload)
            self.assertEqual(engine.pending_bits(), 0)

            # The sender frame loop observes requalification and releases the
            # held bytes only after the transmit training phase is also ready.
            engine._state = "bis"
            engine._rx_ready = True
            engine._tx_mode = "data"
            session._next_frame_samples()
            self.assertEqual(bytes(session._held), b"")
            self.assertGreater(engine.pending_bits(), 0)
        finally:
            session._v22.close()
            sock.close()
            peer.close()

    def test_v22bis_retrain_after_v42_raw_fallback_preserves_payload(self) -> None:
        near_sock, near_peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        far_sock, far_peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        near = OnlineAudioSession(
            near_sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode="v22bis", v22_rate=2400, v42=True,
        )
        far = OnlineAudioSession(
            far_sock, ("127.0.0.1", 9), "answer", "originate",
            line_mode="v22bis", v22_rate=2400,
        )
        near_rx, far_rx = bytearray(), bytearray()

        def exchange_frame():
            near_tx = near._next_frame_samples()
            far_tx = far._next_frame_samples()
            near_rx.extend(near.handle_rtp_payload(bytes(linear_to_ulaw(x) for x in far_tx)))
            far_rx.extend(far.handle_rtp_payload(bytes(linear_to_ulaw(x) for x in near_tx)))

        try:
            for _ in range(500):
                exchange_frame()
            self.assertTrue(near._v22.connected and far._v22.connected)
            near.poll(time.monotonic())
            self.assertTrue(near._v42_started)
            for _ in range(80):
                exchange_frame()
                near.poll(time.monotonic())
                if near._v22.v42_mode == "failed":
                    break
            self.assertEqual(near._v22.v42_mode, "failed")
            # Let the fallback mark sequence flush and the raw receiver settle.
            for _ in range(10):
                exchange_frame()
            near_rx.clear()
            far_rx.clear()

            # The actual V.42 no-peer timeout has fallen back to raw mode.
            # S1 lookahead must delay ordinary DTE data without changing it.
            near.feed_terminal_bytes(b"ordinary-near")
            far.feed_terminal_bytes(b"ordinary-far")
            for _ in range(20):
                exchange_frame()
            self.assertEqual(bytes(near_rx), b"ordinary-far")
            self.assertEqual(bytes(far_rx), b"ordinary-near")
            near_rx.clear()
            far_rx.clear()

            near._v22._start_retrain()
            near_payload = b"near-after-retrain-" + bytes(range(24))
            far_payload = b"far-after-retrain-" + bytes(range(24))
            near.feed_terminal_bytes(near_payload)
            self.assertEqual(bytes(near._held), near_payload)

            # The far DCE must first hear the incoming S1 and deassert its
            # transmit-ready state; writes made before that indication are
            # legitimately still eligible for transmission.
            for _ in range(20):
                exchange_frame()
                if far._v22._state == "retrain":
                    break
            self.assertEqual(far._v22._state, "retrain")
            far.feed_terminal_bytes(far_payload)
            self.assertEqual(bytes(far._held), far_payload)

            for _ in range(130):
                exchange_frame()
            self.assertIn(("RETRAIN_SUCCEEDED", 2400), near._v22.events)
            self.assertIn(("RETRAIN_SUCCEEDED", 2400), far._v22.events)
            self.assertEqual(bytes(near_rx), far_payload, repr(bytes(near_rx)))
            self.assertEqual(bytes(far_rx), near_payload, repr(bytes(far_rx)))
        finally:
            near._v22.close()
            far._v22.close()
            for sock in (near_sock, near_peer, far_sock, far_peer):
                sock.close()

    def test_v23_uses_10ms_rtp_frames_for_carrier_release_timing(self) -> None:
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer", line_mode="v23",
        )
        try:
            self.assertEqual(session._rtp_frame_samples, 80)
            self.assertAlmostEqual(session._rtp_frame_seconds, 0.010)
            self.assertEqual(len(session._next_frame_samples()), 80)
            self.assertEqual(session.link_stats()["rtp_packetization_ms"], 10.0)
        finally:
            sock.close()
            peer.close()

    def test_v23_accepts_20ms_peer_packets_without_false_gap_fill(self) -> None:
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer", line_mode="v23",
        )
        try:
            payload = bytes([0xFF]) * 160
            for sequence, timestamp in ((1, 0), (2, 160)):
                packet = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, 123)
                session.handle_rtp_packet(packet + payload)
            self.assertEqual(session._rtp_counts["concealed_samples"], 0)
            self.assertEqual(session._last_rx_payload_samples, 160)
        finally:
            sock.close()
            peer.close()

    def test_v23_detector_stats_include_sample_and_packet_timing(self) -> None:
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "answer", "originate", line_mode="v23",
        )
        try:
            for packet_index in range(30):
                first_sample = packet_index * 80
                pcm = struct.pack(
                    "<80h",
                    *(round(12000 * math.sin(2 * math.pi * 1300 *
                                             (first_sample + i) / 8000))
                      for i in range(80)),
                )
                session._rx_context = {
                    "timestamp": first_sample, "pcm_source": "packet",
                }
                session._rx_arrival = time.monotonic() - 0.002
                session._receive_pcm(pcm)
            transition = session.link_stats()["v23_detector_transitions"][-1]
            self.assertEqual(transition["event"], "CARRIER_UP")
            self.assertEqual(transition["sample_index"], 2400)
            self.assertEqual(transition["sample_offset_in_packet"], 80)
            self.assertEqual(transition["rtp_timestamp"], 2320)
            self.assertIsNotNone(transition["packet_received_monotonic"])
            self.assertGreaterEqual(transition["packet_to_state_ms"], 0)
        finally:
            sock.close()
            peer.close()

    def test_v23_known_carrier_edges_align_through_production_rtp_pcmu(self) -> None:
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "answer", "originate", line_mode="v23",
        )
        transitions = []
        try:
            for packet_index in range(31):
                first_sample = packet_index * 80
                amplitude = 12000 if packet_index < 30 else 0
                samples = [round(amplitude * math.sin(
                    2 * math.pi * 1300 * (first_sample + i) / 8000
                )) for i in range(80)]
                ulaw = bytes(linear_to_ulaw(sample) for sample in samples)
                rtp = struct.pack("!BBHII", 0x80, 0, packet_index + 1,
                                  first_sample, 0x10203040)
                session.handle_rtp_packet(rtp + ulaw, received_at=time.monotonic())
            transitions = session.link_stats()["v23_detector_transitions"]
            self.assertEqual(
                [(row["event"], row["sample_index"], row["rtp_timestamp"])
                 for row in transitions],
                [("CARRIER_UP", 2400, 2320), ("CARRIER_DOWN", 2480, 2400)],
            )
            self.assertTrue(all(row["packet_to_state_ms"] >= 0 for row in transitions))
        finally:
            sock.close()
            peer.close()

    def test_v22_guard_tone_setting_reaches_engine(self) -> None:
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "answer", "originate", line_mode="v22bis",
            v22_rate=2400, v22_min_rate=2400, guard_tone=2,
        )
        try:
            self.assertEqual(session._v22.guard_tone, 2)
            self.assertEqual(session._v22.min_rate, 2400)
        finally:
            session._v22.close()
            sock.close()
            peer.close()

    def test_v22bis_minimum_rate_reaches_v8_fallback_chain(self) -> None:
        from modem_dce import line_from_ms

        params = line_from_ms({"mod": "V22B", "auto": 1, "min": 2400, "max": 2400})
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode=params["line_mode"], chain=params["chain"],
        )
        try:
            self.assertEqual(session._v22.v22_rate, 2400)
            self.assertEqual(session._v22.v22_min_rate, 2400)
        finally:
            session._v22.close()
            sock.close()
            peer.close()

    def test_bell212_minimum_rate_reaches_engine(self) -> None:
        from modem_dce import line_from_ms

        params = line_from_ms({"mod": "B212", "auto": 0, "min": 1200, "max": 1200})
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer", **params,
        )
        try:
            self.assertEqual(session._v22.min_rate, 1200)
        finally:
            session._v22.close()
            sock.close()
            peer.close()

    def test_v22_600_selection_reaches_v22_mode_iv_engine(self) -> None:
        from modem_dce import line_from_ms

        params = line_from_ms({"mod": "V22", "auto": 0, "min": 300, "max": 600})
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer", **params,
        )
        try:
            self.assertEqual(session._v22.profile, "v22")
            self.assertEqual(session._v22.target_rate, 600)
            self.assertEqual(session.modulation_name, "V.22")
        finally:
            session._v22.close()
            sock.close()
            peer.close()

    def test_v21_originate_waits_for_ans_end_before_connect_and_dte_data(self) -> None:
        """Caller tone, answer-tone gate, post-ANS silence, then DTE payload."""
        from autodetect import tone_fraction
        from v25 import FRAME_SAMPLES

        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer", line_mode="v21",
            calling_tone=True,
        )
        payload = b"DTE DATA HELD UNTIL V.25\r"
        session.feed_terminal_bytes(payload)
        try:
            self.assertFalse(session.negotiation_complete)
            self.assertFalse(session._tx_switched)
            call_tone = session._next_frame_samples()
            fraction, rms = tone_fraction(call_tone, 1300.0)
            self.assertGreater(fraction, 0.9)
            self.assertGreater(rms, 1000)

            # Feed the answer tone through the production PCMU input path.
            # Include V.25's phase reversals: their brief low-energy frame
            # must not count as the 75 +/- 20 ms post-tone interval.
            for offset in range(0, int(3.3 * 8000), FRAME_SAMPLES):
                linear = [round(7000 * math.sin(
                    2 * math.pi * 2100 * (offset + i) / 8000
                    + math.pi * ((offset + i) // 3600)))
                    for i in range(FRAME_SAMPLES)]
                ulaw = bytes(linear_to_ulaw(value) for value in linear)
                session.handle_rtp_payload(ulaw)
            self.assertEqual(session._v25_caller.state, "answer_tone")
            self.assertEqual(session._next_frame_samples(), [0] * FRAME_SAMPLES)
            self.assertFalse(session.negotiation_complete)

            silence = bytes([0xff] * FRAME_SAMPLES)  # PCMU silence
            for _ in range(4):
                session.handle_rtp_payload(silence)
            self.assertEqual(session._v25_caller.state, "complete")
            self.assertTrue(session.negotiation_complete)
            self.assertTrue(session._tx_switched)
            self.assertEqual(session.transmitter.pending(), len(payload) * 10)
        finally:
            sock.close()
            peer.close()

    def test_v21_production_answer_prologue_precedes_pcmu_data(self) -> None:
        """V.25 answer signaling and following V.21 data traverse production media."""
        answer_sock, answer_peer = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_DGRAM)
        caller_sock, caller_peer = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_DGRAM)
        answer = OnlineAudioSession(
            answer_sock, ("127.0.0.1", 9), "answer", "originate",
            line_mode="v21",
        )
        self.assertFalse(answer.negotiation_complete,
                         "answer DCE must hold CONNECT during V.25 prologue")
        caller = OnlineAudioSession(
            caller_sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode="v21",
        )
        payload = b"V21 AFTER ANSWER TONE 0123456789\r"
        answer.feed_terminal_bytes(payload)
        received = bytearray()
        answer_pcm = []
        try:
            # Run through the V.25 billing silence, answer tone/reversals,
            # post-tone gap and enough V.21 data, all at RTP's 20-ms cadence.
            for frame_index in range(450):
                frame = answer._next_frame_samples()
                if frame_index == 267:
                    self.assertFalse(answer.negotiation_complete,
                                     "75-ms post-ANS interval has not finished")
                if frame_index == 268:
                    self.assertTrue(answer.negotiation_complete,
                                    "CONNECT should follow the complete V.25 prologue")
                pcmu = bytes(linear_to_ulaw(sample) for sample in frame)
                answer_pcm.extend(ulaw_to_linear(sample) for sample in pcmu)
                received.extend(caller.handle_rtp_payload(pcmu))

            self.assertEqual(answer_pcm[:16000], [0] * 16000)
            tone = answer_pcm[20000:24000]

            def energy(samples, frequency):
                omega = 2 * math.pi * frequency / 8000
                real = sum(value * math.cos(omega * index)
                           for index, value in enumerate(samples))
                imag = sum(value * math.sin(omega * index)
                           for index, value in enumerate(samples))
                return real * real + imag * imag

            self.assertGreater(energy(tone, 2100), energy(tone, 1650))
            self.assertEqual(bytes(received), b"\xff" * 7 + payload)
        finally:
            answer_sock.close()
            answer_peer.close()
            caller_sock.close()
            caller_peer.close()

    @unittest.skipUnless(shutil.which("minimodem"), "optional minimodem is not installed")
    def test_v21_production_pcmu_transmit_decodes_in_minimodem(self) -> None:
        """Independently decode V.21 samples emitted by the production session."""
        message = b"ELMODEM PRODUCTION V21 TX 0123456789\r"
        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer", line_mode="v21",
            v25_call_sequence=False,
        )
        session.feed_terminal_bytes(message)
        received_samples = []
        try:
            # Exercise the same 20-ms sample extraction and PCMU quantization
            # as _send_loop(), then decode the resulting audio independently.
            for _ in range(140):
                frame = session._next_frame_samples()
                payload = bytes(linear_to_ulaw(sample) for sample in frame)
                received_samples.extend(ulaw_to_linear(value) for value in payload)

            with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
                with wave.open(audio.name, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(8000)
                    wav.writeframes(struct.pack(
                        f"<{len(received_samples)}h", *received_samples))
                result = subprocess.run(
                    [shutil.which("minimodem"), "--rx", "--quiet", "--file",
                     audio.name, "--mark", "980", "--space", "1180", "300"],
                    capture_output=True, timeout=20, check=False,
                )
            self.assertEqual(result.returncode, 0,
                             result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, message,
                             result.stderr.decode(errors="replace"))
        finally:
            sock.close()
            peer.close()

    def test_v21_independent_vectors_cross_production_pcmu_session(self) -> None:
        """Decode independent V.21 audio through the DCE's PCMU media entry."""
        vectors = (
            ("v21-orig.raw", "originate", b"ELMOD V21 ORIGINATE 0123456789\r"),
            ("v21-answer.raw", "answer", b"ELMOD V21 ANSWER 9876543210\r"),
        )
        fixture_dir = (Path(__file__).parent / "audio-peer" / "fixtures" /
                       "reference" / "independent-fsk")
        if not all((fixture_dir / filename).is_file()
                   for filename, _, _ in vectors):
            self.skipTest("optional independent audio fixtures are not in the public export")

        for filename, remote_role, expected in vectors:
            with self.subTest(vector=filename):
                pcm = (fixture_dir / filename).read_bytes()
                samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
                sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
                session = OnlineAudioSession(
                    sock, ("127.0.0.1", 9), "answer", remote_role,
                    line_mode="v21",
                )
                decoded = bytearray()
                try:
                    # Send 20-ms PCMU frames, padding the final fixture frame
                    # with silence as an RTP source would. This exercises the
                    # production G.711 conversion and OnlineAudioSession RX
                    # path, not only Bell103Receiver directly.
                    for offset in range(0, len(samples), 160):
                        frame = list(samples[offset:offset + 160])
                        frame.extend([0] * (160 - len(frame)))
                        payload = bytes(linear_to_ulaw(sample) for sample in frame)
                        decoded.extend(session.handle_rtp_payload(payload))
                    self.assertEqual(bytes(decoded), expected)
                finally:
                    sock.close()
                    peer.close()

    def test_strict_exhausted_xid_is_not_a_raw_connect(self) -> None:
        engine = types.SimpleNamespace(
            state="connected", v42_mode="failed", v42_resolved=True,
            _v42=types.SimpleNamespace(_xid_sent=4, state="down"),
            is_settled=lambda: True,
        )
        session = object.__new__(OnlineAudioSession)
        session._v22 = engine
        session._auto = None
        session._want_v42 = True
        session._v42_xid_fallback = False
        session._created = time.monotonic()
        session._last_rtp_at = session._created
        self.assertFalse(session.negotiation_complete)
        self.assertTrue(session.failed)

    def test_strict_xid_layer_fallback_is_not_a_raw_connect(self) -> None:
        """Any layer fallback after XID starts is unsafe, not just N400 down."""
        engine = types.SimpleNamespace(
            state="connected", v42_mode="failed", v42_resolved=True,
            _v42=types.SimpleNamespace(_xid_sent=1, state="idle"),
            is_settled=lambda: True,
        )
        session = object.__new__(OnlineAudioSession)
        session._v22 = engine
        session._auto = None
        session._want_v42 = True
        session._v42_xid_fallback = False
        session._created = time.monotonic()
        session._last_rtp_at = session._created
        self.assertFalse(session.negotiation_complete)
        self.assertTrue(session.failed)

    def test_strict_xid_policy_still_allows_no_v42_raw_fallback(self) -> None:
        engine = types.SimpleNamespace(
            state="connected", v42_mode="failed", v42_resolved=True,
            _v42=types.SimpleNamespace(_xid_sent=0, state="detect"),
            is_settled=lambda: True,
        )
        session = object.__new__(OnlineAudioSession)
        session._v22 = engine
        session._auto = None
        session._want_v42 = True
        session._v42_xid_fallback = False
        session._v42_started = True
        session._created = time.monotonic()
        session._last_rtp_at = session._created
        self.assertTrue(session.negotiation_complete)
        self.assertFalse(session.failed)

    def test_raw_bell212_break_control_is_delegated_to_modem(self) -> None:
        engine = types.SimpleNamespace(
            profile="bell212a", _bell103_fallback=False, v42_mode="off",
            bell212_feed_break=mock.Mock(return_value=True),
            bell212_read_break_events=mock.Mock(return_value=[
                {"type": "break_indication", "start_bits": 23, "character_bits": 10},
            ]),
        )
        session = object.__new__(OnlineAudioSession)
        session._v22 = engine

        self.assertTrue(session.send_bell212_break())
        engine.bell212_feed_break.assert_called_once_with()
        self.assertEqual(session.read_bell212_break_events(), [
            {"type": "break_indication", "start_bits": 23, "character_bits": 10},
        ])
        engine.bell212_read_break_events.assert_called_once_with()

    def test_bell212_dte_baud_uses_only_supported_serial_rates(self) -> None:
        self.assertEqual(bell212_dte_baud_for_serial(types.SimpleNamespace(
            get_dte_baud=lambda: 1200,
        )), 1200)
        self.assertEqual(bell212_dte_baud_for_serial(types.SimpleNamespace(
            get_dte_baud=lambda: 1212,
        )), 1212)
        self.assertEqual(bell212_dte_baud_for_serial(types.SimpleNamespace(
            get_dte_baud=lambda: 38400,
        )), 1200)
        self.assertEqual(bell212_dte_baud_for_serial(object()), 1200)

    def test_v42_break_control_is_delegated_to_active_layer(self) -> None:
        from types import SimpleNamespace

        events = [{"type": "break_indication", "length_ms": 120}]
        engine = SimpleNamespace(
            v42_mode="lapm",
            v42_request_break=mock.Mock(return_value=True),
            v42_read_break_events=mock.Mock(side_effect=lambda: list(events)),
            v42_acknowledge_break=mock.Mock(return_value=True),
        )
        session = object.__new__(OnlineAudioSession)
        session._online_control = SimpleNamespace(engine=lambda: engine)

        self.assertTrue(session.v42_break_available())
        self.assertTrue(session.request_v42_break())
        engine.v42_request_break.assert_called_once_with(
            length_ms=None, destructive=False, expedited=False)
        self.assertEqual(session.read_v42_break_events(), events)
        self.assertTrue(session.acknowledge_v42_break())
        engine.v42_acknowledge_break.assert_called_once_with()

        engine.v42_mode = "lost"
        self.assertFalse(session.v42_break_available())

    def test_v42_parameter_request_is_delegated_to_active_control_engine(self) -> None:
        from types import SimpleNamespace

        engine = SimpleNamespace(v42_renegotiate_parameters=mock.Mock(return_value=True))
        session = object.__new__(OnlineAudioSession)
        session._online_control = SimpleNamespace(engine=lambda: engine)

        self.assertTrue(session.request_v42_parameters(
            tx_n401=32, rx_n401=24, tx_k=4, rx_k=3
        ))
        engine.v42_renegotiate_parameters.assert_called_once_with(
            tx_n401=32, rx_n401=24, tx_k=4, rx_k=3
        )

        session._online_control = None
        self.assertFalse(session.request_v42_parameters(
            tx_n401=32, rx_n401=24, tx_k=4, rx_k=3
        ))

    def test_orderly_v42_release_is_delegated_and_has_protocol_bounded_timeout(self) -> None:
        from types import SimpleNamespace

        link = SimpleNamespace(rate=9600, _t401_bits=14_400)
        engine = SimpleNamespace(
            bit_rate=9600, v42_mode="lapm", _v42=link,
            v42_release=mock.Mock(return_value=True),
        )
        session = object.__new__(OnlineAudioSession)
        session._v22 = engine
        session._auto = None
        session._online_control = SimpleNamespace(engine=lambda: engine)

        self.assertTrue(session.request_v42_release())
        engine.v42_release.assert_called_once_with()
        self.assertFalse(session.v42_release_complete())
        self.assertEqual(session.v42_release_timeout(), 17.0)

        engine.v42_mode = "failed"  # V42Layer's raw-mode fallback after release
        self.assertTrue(session.v42_release_complete())

    def test_slow_lapm_release_deadline_does_not_cut_off_t401_retries(self) -> None:
        from types import SimpleNamespace
        from v42_lapm import N400, V42Link

        for rate in (1200, 300):
            with self.subTest(rate=rate):
                link = V42Link(calling_party=True, detect=False,
                               bit_rate=rate, frame_size=128)
                engine = SimpleNamespace(bit_rate=rate, v42_mode="lapm", _v42=link)
                session = object.__new__(OnlineAudioSession)
                session._v22 = engine
                session._auto = None
                session._online_control = SimpleNamespace(engine=lambda: engine)

                expected = (link._t401_bits / link.rate) * (N400 + 1) + 0.5
                self.assertGreater(expected, 20.0)
                self.assertAlmostEqual(session.v42_release_timeout(), expected)

    def test_orderly_v42_release_is_unavailable_when_no_lapm_engine_exists(self) -> None:
        from types import SimpleNamespace

        session = object.__new__(OnlineAudioSession)
        session._online_control = SimpleNamespace(engine=lambda: object())
        self.assertFalse(session.request_v42_release())

    def test_bell212_framing_settings_reach_dte_session_boundary(self) -> None:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode="bell212a", bell212_char_bits=11, bell212_parity="O", bell212_dte_baud=1212,
        )
        try:
            session.feed_terminal_bytes(b"\xc1")
            self.assertEqual(session._bell212_char_bits, 11)
            self.assertEqual(session._v22.bell212_dte_baud, 1212)
            self.assertEqual(session._v22.bell212_parity, "O")
            self.assertEqual(list(session._v22._app_bits), [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1])
        finally:
            session.stop()
            sock.close()

    def test_link_metrics_keep_bounded_v42_failure_context(self) -> None:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode="bell212a", v42=True,
        )
        try:
            session._v22.start_v42(calling_party=True, rate=1200)
            session._v22._v42.events.extend(f"event-{i}" for i in range(40))
            stats = session.link_stats()
            self.assertEqual(stats["v42_mode"], "detect")
            self.assertEqual(stats["v42_state"], "detect")
            self.assertEqual(stats["v42_parameters"], {
                "n401_tx": 128, "n401_rx": 128, "k_tx": 15, "k_rx": 15,
            })
            self.assertEqual(stats["v42_recovery"], {
                "attempts": 0, "completed": 0, "failures": 0,
                "requeued_dte_bytes": 0, "c_error_input_octets": 0,
                "discarded_encoded_octets": 0,
                "last_failure": None,
            })
            self.assertEqual(stats["v42_events"], [f"event-{i}" for i in range(8, 40)])
        finally:
            session.stop()
            sock.close()

    def test_bell212_session_establishes_lapm_and_exchanges_binary_data(self) -> None:
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        originate = OnlineAudioSession(
            o_sock, a_sock.getsockname(), "originate", "answer",
            line_mode="bell212a", v42=True,
        )
        answer = OnlineAudioSession(
            a_sock, o_sock.getsockname(), "answer", "originate",
            line_mode="bell212a", v42=True,
        )
        originate.start()
        answer.start()
        try:
            originate_rx, answer_rx = bytearray(), bytearray()
            payload_o = bytes((i * 73 + 19) & 0xff for i in range(256))
            payload_a = bytes((i * 41 + 0xa7) & 0xff for i in range(256))
            sent = False
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                if originate.negotiation_complete and answer.negotiation_complete and not sent:
                    self.assertEqual((originate.line_speed, answer.line_speed), (1200, 1200))
                    self.assertEqual((originate.connect_suffix, answer.connect_suffix),
                                     ("/LAPM", "/LAPM"))
                    originate.feed_terminal_bytes(payload_o)
                    answer.feed_terminal_bytes(payload_a)
                    sent = True
                for session, sock, received in (
                    (answer, a_sock, answer_rx), (originate, o_sock, originate_rx),
                ):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    received.extend(session.handle_rtp_datagram(source, packet))
                if sent and answer_rx == payload_o and originate_rx == payload_a:
                    break
            self.assertTrue(sent, "Bell 212A RTP sessions did not establish LAPM")
            self.assertEqual((originate._v22.v42_mode, answer._v22.v42_mode),
                             ("lapm", "lapm"))
            self.assertEqual(bytes(answer_rx), payload_o)
            self.assertEqual(bytes(originate_rx), payload_a)

            # The receiver can deliver the last I-frame before its RR reaches
            # the sender. Wait for the protocol window to drain; L-SETPARM is
            # correctly refused while unacknowledged data remains.
            drain_deadline = time.monotonic() + 1.0
            while time.monotonic() < drain_deadline and (
                    originate._v22._v42._unacked or answer._v22._v42._unacked):
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_datagram(source, packet)
            self.assertFalse(originate._v22._v42._unacked)
            self.assertFalse(answer._v22._v42._unacked)

            # Issue an L-SETPARM request through the production DCE/RTP
            # control surface. Drop exactly the first connected XID response
            # at the responder's LAPM boundary; the retry must be clocked by
            # the paced RTP/DSP loop, not by a test-side timer callback.
            from v42_lapm import XID
            answer_link = answer._v22._v42
            original_u = answer_link._u
            dropped_response_at = []

            def drop_first_connected_xid_response(ctrl, command, pf=0, info=b""):
                if (ctrl & 0xEF) == XID and not command and not dropped_response_at:
                    dropped_response_at.append(time.monotonic())
                    return
                return original_u(ctrl, command, pf, info)

            answer_link._u = drop_first_connected_xid_response
            request_started = time.monotonic()
            self.assertTrue(originate.request_v42_parameters(
                tx_n401=32, rx_n401=32, tx_k=4, rx_k=4
            ))
            request_link = originate._v22._v42
            t401_seconds = request_link._t401_bits / request_link.rate
            retry_seen_at = None
            renegotiate_deadline = time.monotonic() + 8.0
            while time.monotonic() < renegotiate_deadline:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_datagram(source, packet)
                if any("connected XID renegotiation retry 1/" in event
                       for event in request_link.events):
                    retry_seen_at = time.monotonic()
                    break
            self.assertEqual(len(dropped_response_at), 1,
                             "the responder response-loss injection did not trigger")
            self.assertIsNotNone(retry_seen_at,
                                 "the production RTP/DSP loop did not reach the T401 retry")
            retry_delay = retry_seen_at - request_started
            self.assertGreaterEqual(retry_delay, t401_seconds - 0.15)
            self.assertLessEqual(retry_delay, t401_seconds + 0.75)

            # The next retry receives the response. Verify completion and
            # subsequent payload on the same live audio session.
            renegotiate_deadline = time.monotonic() + 5.0
            while time.monotonic() < renegotiate_deadline:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_datagram(source, packet)
                if (request_link._renegotiation is None
                        and (request_link.tx_n401, request_link.rx_n401,
                             request_link.tx_k, request_link.rx_k) == (32, 32, 4, 4)
                        and (answer_link.tx_n401, answer_link.rx_n401,
                             answer_link.tx_k, answer_link.rx_k) == (32, 32, 4, 4)):
                    break
            self.assertIsNone(request_link._renegotiation,
                              f"caller events={request_link.events}; "
                              f"answer events={answer_link.events}")
            self.assertEqual((request_link.tx_n401, request_link.rx_n401,
                              request_link.tx_k, request_link.rx_k), (32, 32, 4, 4))
            self.assertEqual((answer_link.tx_n401, answer_link.rx_n401,
                              answer_link.tx_k, answer_link.rx_k), (32, 32, 4, 4))

            post_xid = b"payload after real-time XID retry"
            originate.feed_terminal_bytes(post_xid)
            renegotiated_rx = bytearray()
            data_deadline = time.monotonic() + 5.0
            while (time.monotonic() < data_deadline
                   and len(renegotiated_rx) < len(post_xid)):
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                try:
                    packet, source = a_sock.recvfrom(2048)
                except (socket.timeout, OSError):
                    continue
                renegotiated_rx.extend(answer.handle_rtp_datagram(source, packet))
            self.assertEqual(bytes(renegotiated_rx), post_xid)
        finally:
            originate.stop()
            answer.stop()
            o_sock.close()
            a_sock.close()

    def test_v42bis_c_error_over_rtp_fails_closed_when_peer_does_not_reestablish(self) -> None:
        """Exercise the C-ERROR path through real V.22 PCM, RTP, and LAPM."""
        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.01)
        a_sock.settimeout(0.01)
        recovery_env = mock.patch.dict(os.environ, {"V42_CERROR_RECOVERY": "1"})
        recovery_env.start()
        originate = OnlineAudioSession(
            o_sock, a_sock.getsockname(), "originate", "answer",
            line_mode="bell212a", v42=True, v42bis=True,
            carrier_loss_seconds=2.0,
        )
        answer = OnlineAudioSession(
            a_sock, o_sock.getsockname(), "answer", "originate",
            line_mode="bell212a", v42=True, v42bis=True,
            carrier_loss_seconds=2.0,
        )
        originate.start()
        answer.start()
        try:
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_datagram(source, packet)
                if (originate._v22.v42_mode == "lapm"
                        and answer._v22.v42_mode == "lapm"
                        and originate._v22.v42_compression
                        and answer._v22.v42_compression):
                    break
            else:
                self.fail("V.42bis LAPM did not establish over the RTP audio sessions")

            # Model a control function that accepts orderly release but elects
            # not to re-establish. The modem peer remains online and sends RTP.
            originate._v22._v42_compression_recovery = False
            answer._v22._v42_recovery_timeout_seconds = 0.5

            decoder = answer._v22._v42._dec
            decoder.compressed = True
            invalid = decoder.d.c1.to_bytes(2, "little")
            tx = originate._v22._v42
            tx._queue_i(invalid)
            tx._send_i(*tx._unacked[-1])

            recovery_deadline = time.monotonic() + 6.0
            while time.monotonic() < recovery_deadline and not answer.link_lost:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_datagram(source, packet)

            self.assertTrue(answer._v22.v42_lost,
                            answer._v22._v42.events[-8:])
            self.assertTrue(answer.link_lost)
            self.assertTrue(answer._v22.connected,
                            "the recovery timeout must be distinguished from carrier loss")
            self.assertIsNotNone(answer._last_rtp_at,
                                 "RTP should still be arriving when V.42 recovery expires")
            self.assertEqual(answer._v22._v42_recovery_stage, "failed")
            self.assertTrue(any("re-establishment timed out" in str(event)
                                for event in answer._v22._v42.events),
                            answer._v22._v42.events[-8:])
            self.assertIn("V.42bis recovery failed", " ".join(answer._v22._v42.events))
        finally:
            originate.stop()
            answer.stop()
            o_sock.close()
            a_sock.close()
            recovery_env.stop()

    def test_v42bis_c_error_over_rtp_reestablishes_and_delivers_exact_payload(self) -> None:
        """Exercise successful C-ERROR recovery through both production audio sessions."""
        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.01)
        a_sock.settimeout(0.01)
        recovery_env = mock.patch.dict(os.environ, {"V42_CERROR_RECOVERY": "1"})
        recovery_env.start()
        originate = OnlineAudioSession(
            o_sock, a_sock.getsockname(), "originate", "answer",
            line_mode="bell212a", v42=True, v42bis=True,
            carrier_loss_seconds=2.0,
        )
        answer = OnlineAudioSession(
            a_sock, o_sock.getsockname(), "answer", "originate",
            line_mode="bell212a", v42=True, v42bis=True,
            carrier_loss_seconds=2.0,
        )
        originate.start()
        answer.start()
        try:
            sessions = ((answer, a_sock), (originate, o_sock))

            def exchange_until(predicate, timeout, collected=None):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    originate.poll(now)
                    answer.poll(now)
                    for session, sock in sessions:
                        try:
                            packet, source = sock.recvfrom(2048)
                        except (socket.timeout, OSError):
                            continue
                        data = session.handle_rtp_datagram(source, packet)
                        if collected is not None:
                            collected[session].extend(data)
                    if predicate():
                        return True
                return False

            established = lambda: (
                originate._v22.v42_mode == "lapm"
                and answer._v22.v42_mode == "lapm"
                and originate._v22.v42_compression
                and answer._v22.v42_compression
            )
            self.assertTrue(exchange_until(established, 35),
                            "V.42bis LAPM did not establish over RTP")

            # Inject an invalid compressed codeword into a valid LAPM I-frame;
            # actual V.22 PCM and RTP carry the resulting C-ERROR and restart.
            caller_old_link = originate._v22._v42
            answer_old_link = answer._v22._v42
            decoder = answer._v22._v42._dec
            decoder.compressed = True
            invalid = decoder.d.c1.to_bytes(2, "little")
            tx = originate._v22._v42
            tx._queue_i(invalid)
            tx._send_i(*tx._unacked[-1])

            recovered = lambda: (
                established()
                and originate._v22._v42_recovery_stats["completed"] == 1
                and answer._v22._v42_recovery_stats["completed"] == 1
            )
            def recovery_state(session):
                layer = session._v22
                link = layer._v42
                return {
                    "mode": layer.v42_mode,
                    "stage": layer._v42_recovery_stage,
                    "state": getattr(link, "state", None),
                    "lapm_stats": dict(getattr(link, "stats", {})),
                    "stats": dict(layer._v42_recovery_stats),
                    "events": list(getattr(link, "events", [])),
                }
            old_link_state = lambda link: {
                "state": link.state,
                "stats": dict(link.stats),
                "events": list(link.events),
            }
            self.assertTrue(exchange_until(recovered, 12),
                            "both endpoints did not complete bounded LAPM recovery: "
                            f"caller={recovery_state(originate)!r}, "
                            f"answerer={recovery_state(answer)!r}, "
                            f"caller_old={old_link_state(caller_old_link)!r}, "
                            f"answerer_old={old_link_state(answer_old_link)!r}")
            self.assertIsNot(answer._v22._v42._dec, decoder,
                             "answerer retained the desynchronized V.42bis decoder")

            caller_payload = bytes(range(256)) * 3
            answer_payload = bytes(reversed(range(256))) * 3
            received = {originate: bytearray(), answer: bytearray()}
            originate._v22.v42_feed_bytes(caller_payload)
            answer._v22.v42_feed_bytes(answer_payload)
            delivered = lambda: (
                caller_payload in received[answer]
                and answer_payload in received[originate]
            )
            self.assertTrue(exchange_until(delivered, 12, received),
                            "post-recovery RTP session did not deliver both payloads")
            self.assertEqual(bytes(received[answer]), caller_payload)
            self.assertEqual(bytes(received[originate]), answer_payload)
        finally:
            originate.stop()
            answer.stop()
            o_sock.close()
            a_sock.close()
            recovery_env.stop()

    def test_run_pty_v42bis_c_error_recovers_over_production_rtp(self) -> None:
        """Exercise DTE, DCE, modem DSP, RTP and V.42bis recovery as one call."""
        import modem_dce

        state = {
            "output": bytearray(),
            "peer_rx": bytearray(),
            "peer_errors": [],
            "connect_seen": threading.Event(),
            "peer_payload_seen": threading.Event(),
            "recovered": threading.Event(),
            "stop": threading.Event(),
        }
        initial_payload = bytes((i * 37 + 11) & 0xff for i in range(113))
        post_dte_payload = bytes((i * 53 + 7) & 0xff for i in range(97))
        peer_payload = bytes((i * 29 + 0xd3) & 0xff for i in range(109))

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "v42-cerror-run-pty-socketpair"
                self.supports_modem_control = False
                self.closed = False
                self.post_sent = False
                self.started = time.monotonic()
                self.dte.sendall(b"ATE0\\N3%C1\rATD1001\r")

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                state["output"].extend(data)
                if b"CONNECT 2400" in state["output"] and not state["connect_seen"].is_set():
                    state["connect_seen"].set()
                    self.dte.sendall(initial_payload)

            def service(self, readable, writable=()):
                if time.monotonic() - self.started > 35:
                    raise AssertionError(
                        "run_pty V.42bis C-ERROR recovery timed out: "
                        f"output_tail={bytes(state['output'][-160:])!r}, "
                        f"peer_rx={len(state['peer_rx'])}, errors={state['peer_errors']!r}"
                    )
                if (state["recovered"].is_set() and not self.post_sent):
                    self.post_sent = True
                    self.dte.sendall(post_dte_payload)
                if (state["peer_payload_seen"].is_set()
                        and peer_payload in state["output"]
                        and post_dte_payload in state["peer_rx"]):
                    raise KeyboardInterrupt
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]

            def start(self):
                pass

            def close(self):
                pass

        peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer_sock.bind(("127.0.0.1", 0))
        peer_sock.settimeout(0.01)
        peer = OnlineAudioSession(
            peer_sock, ("127.0.0.1", 0), "answer", "originate",
            line_mode="v22bis", v22_rate=2400, v42=True, v42bis=True,
        )
        serial = FakeSerial()
        original_session = OnlineAudioSession

        class FakeSipCall:
            def __init__(self, *args, **kwargs):
                self.on_connected = kwargs["on_connected"]
                self.on_hangup = kwargs["on_hangup"]
                self.on_failed = kwargs["on_failed"]
                self.hungup = False

            def dial(self, number):
                self.on_connected(peer_sock.getsockname())

            def hangup(self):
                self.hungup = True

        # Let the production session choose and bind its RTP socket, then point
        # the peer back at it from the tracking factory.
        def track_session(sock, remote_rtp, tx_role, rx_role, **kwargs):
            session = original_session(sock, remote_rtp, tx_role, rx_role, **kwargs)
            state["dut_session"] = session
            peer._remote_addr = sock.getsockname()
            peer._send_addr = peer._remote_addr
            peer.start()
            return session

        def peer_media_loop():
            try:
                deadline = time.monotonic() + 35
                injected = False
                sent_peer_payload = False
                while not state["stop"].is_set() and time.monotonic() < deadline:
                    now = time.monotonic()
                    peer.poll(now)
                    try:
                        packet, source = peer_sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        packet = None
                    if packet is not None:
                        state["peer_rx"].extend(peer.handle_rtp_datagram(source, packet))
                    if (not injected and initial_payload in state["peer_rx"]
                            and peer.negotiation_complete and peer._v22.v42_compression
                            and state.get("dut_session") is not None
                            and state["dut_session"]._v22.v42_compression):
                        decoder = state["dut_session"]._v22._v42._dec
                        decoder.compressed = True
                        invalid = decoder.d.c1.to_bytes(2, "little")
                        tx = peer._v22._v42
                        tx._queue_i(invalid)
                        tx._send_i(*tx._unacked[-1])
                        injected = True
                        state["peer_payload_seen"].set()
                    recovered = (
                        peer._v22._v42_recovery_stats["completed"] == 1
                        and getattr(state.get("dut_session"), "_v22", None) is not None
                        and state["dut_session"]._v22._v42_recovery_stats["completed"] == 1
                        and peer._v22.v42_compression
                        and state["dut_session"]._v22.v42_compression
                    )
                    if recovered and not sent_peer_payload:
                        state["recovered"].set()
                        peer._v22.v42_feed_bytes(peer_payload)
                        sent_peer_payload = True
                    if (peer_payload in state["output"]
                            and post_dte_payload in state["peer_rx"]):
                        return
            except Exception as exc:  # surfaced in the test thread below
                state["peer_errors"].append(repr(exc))

        recovery_env = mock.patch.dict(os.environ, {"V42_CERROR_RECOVERY": "1"})
        recovery_env.start()
        peer_thread = threading.Thread(target=peer_media_loop, daemon=True)
        peer_thread.start()

        def track_and_share(*args, **kwargs):
            session = track_session(*args, **kwargs)
            state["dut_session"] = session
            return session

        try:
            with mock.patch.object(modem_dce, "SipIncomingCallServer", FakeSipServer), \
                    mock.patch.object(modem_dce, "SipOriginateCall", FakeSipCall), \
                    mock.patch.object(modem_dce, "OnlineAudioSession", side_effect=track_and_share):
                run_pty(
                    None, sip_host="127.0.0.1", media_host="127.0.0.1", media_port=0,
                    dial_host="127.0.0.1", line_mode="v22bis", v22_rate=2400,
                    negotiate=False, serial_backend=serial,
                )
        finally:
            state["stop"].set()
            peer.stop()
            peer_thread.join(timeout=2)
            peer_sock.close()
            serial.dte.close()
            recovery_env.stop()

        self.assertTrue(serial.closed)
        self.assertFalse(peer_thread.is_alive())
        self.assertEqual(state["peer_errors"], [])
        self.assertTrue(state["connect_seen"].is_set(), bytes(state["output"][-256:]))
        self.assertTrue(state["peer_payload_seen"].is_set(), "peer did not inject C-ERROR")
        self.assertTrue(state["recovered"].is_set(), "both production layers did not recover")
        self.assertEqual(bytes(state["peer_rx"]), initial_payload + post_dte_payload)
        self.assertEqual(bytes(state["output"]).count(peer_payload), 1)
        self.assertIn(b"CONNECT 2400", state["output"])

    def test_bell212_leased_rtp_sessions_establish_lapm_and_exchange_binary_data(self) -> None:
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        originate = OnlineAudioSession(
            o_sock, a_sock.getsockname(), "originate", "answer",
            line_mode="bell212a", bell212_leased_line=True, v42=True,
        )
        answer = OnlineAudioSession(
            a_sock, o_sock.getsockname(), "answer", "originate",
            line_mode="bell212a", bell212_leased_line=True, v42=True,
        )
        originate.start()
        answer.start()
        try:
            originate_rx, answer_rx = bytearray(), bytearray()
            payload_o = bytes((i * 59 + 0x31) & 0xff for i in range(256))
            payload_a = bytes((i * 37 + 0xd3) & 0xff for i in range(256))
            sent = False
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                if originate.negotiation_complete and answer.negotiation_complete and not sent:
                    self.assertEqual((originate.line_speed, answer.line_speed), (1200, 1200))
                    self.assertEqual((originate.connect_suffix, answer.connect_suffix),
                                     ("/LAPM", "/LAPM"))
                    self.assertTrue(originate._v22.bell212_leased_line)
                    self.assertTrue(answer._v22.bell212_leased_line)
                    originate.feed_terminal_bytes(payload_o)
                    answer.feed_terminal_bytes(payload_a)
                    sent = True
                for session, sock, received in (
                    (answer, a_sock, answer_rx), (originate, o_sock, originate_rx),
                ):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    received.extend(session.handle_rtp_datagram(source, packet))
                if sent and answer_rx == payload_o and originate_rx == payload_a:
                    break
            self.assertTrue(sent, "leased Bell 212A RTP sessions did not establish LAPM")
            self.assertEqual((originate._v22.v42_mode, answer._v22.v42_mode),
                             ("lapm", "lapm"))
            self.assertEqual(bytes(answer_rx), payload_o)
            self.assertEqual(bytes(originate_rx), payload_a)
        finally:
            originate.stop()
            answer.stop()
            o_sock.close()
            a_sock.close()

    def test_bell212_raw_break_crosses_rtp_and_preserves_following_data(self) -> None:
        """Raw Mode-B BREAK is an audio event distinct from LAPM L-SIGNAL."""
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        originate = OnlineAudioSession(
            o_sock, a_sock.getsockname(), "originate", "answer", line_mode="bell212a",
        )
        answer = OnlineAudioSession(
            a_sock, o_sock.getsockname(), "answer", "originate", line_mode="bell212a",
        )
        originate.start()
        answer.start()
        try:
            before, after = b"before\x00\xff", b"after\x00\x55"
            received, break_events = bytearray(), []
            sent = False
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                now = time.monotonic()
                originate.poll(now)
                answer.poll(now)
                if originate.negotiation_complete and answer.negotiation_complete and not sent:
                    self.assertEqual((originate.line_speed, answer.line_speed), (1200, 1200))
                    originate.feed_terminal_bytes(before)
                    self.assertTrue(originate.send_bell212_break())
                    originate.feed_terminal_bytes(after)
                    sent = True
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    decoded = session.handle_rtp_datagram(source, packet)
                    if session is answer:
                        received.extend(decoded)
                        break_events.extend(answer.read_bell212_break_events())
                if sent and received.endswith(after) and break_events:
                    break
            self.assertTrue(sent,
                            f"Bell 212A raw RTP session did not connect: "
                            f"orig={originate._v22.events!r} answer={answer._v22.events!r}; "
                            f"settled={originate._v22.is_settled()}/{answer._v22.is_settled()}")
            self.assertEqual(bytes(received), before + after)
            self.assertEqual(break_events, [
                {"type": "break_indication", "start_bits": 23, "character_bits": 10},
            ])
            self.assertEqual((originate._v22.v42_mode, answer._v22.v42_mode), ("off", "off"))
        finally:
            originate.stop()
            answer.stop()
            o_sock.close()
            a_sock.close()

    def test_bell212_dce_ath_release_against_spandsp_over_rtp(self) -> None:
        """Exercise production Bell audio/DCE release against independent LAPM.

        The physical Bell 212A modem and RTP sessions are production code. Only
        the answerer's LAPM endpoint is replaced with SpanDSP, so this covers
        more than the direct bit/audio-pair tests without depending on the
        SmartLink binary peer's non-acknowledging release behavior.
        """
        import socket
        import v22
        import spandsp_v42

        if not spandsp_v42.available():
            self.skipTest("pinned SpanDSP reference library is unavailable")

        SpanDSPV42 = spandsp_v42.V42Link
        ProjectV22Modem = v22.V22Modem

        class SpanDSPBellAnswerer(ProjectV22Modem):
            def __init__(self, **kwargs):
                kwargs.pop("calling_party", None)
                kwargs.pop("bit_rate", None)
                kwargs.pop("profile", None)
                super().__init__(False, 1200, profile="bell212a", **kwargs)
                self.span_link = None

            def start_v42(self, *args, **kwargs):
                self.span_link = SpanDSPV42(False, detect=True, bit_rate=1200)
                self.v42_mode = "detect"

            def _sync_span_mode(self):
                if self.span_link is None:
                    return
                self.v42_mode = {
                    "detect": "detect", "lapm": "lapm",
                    "unsupported": "failed", "down": "failed",
                }[self.span_link.state]

            def v42_tx_bit(self):
                if self.span_link is None:
                    return 1
                bit = self.span_link.tx_bit()
                self._sync_span_mode()
                return bit

            def v42_rx_bits(self, bits):
                if self.span_link is not None:
                    for bit in bits:
                        self.span_link.rx_bit(bit)
                    self._sync_span_mode()
                return b""

            def v42_feed_bytes(self, data):
                if self.span_link is not None:
                    self.span_link.write(data)

            def v42_read(self):
                return self.span_link.read() if self.span_link is not None else b""

            def close(self):
                try:
                    if self.span_link is not None:
                        self.span_link.close()
                finally:
                    super().close()

        def modem_factory(**kwargs):
            if kwargs.get("calling_party"):
                return ProjectV22Modem(**kwargs)
            return SpanDSPBellAnswerer(**kwargs)

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.002)
        a_sock.settimeout(0.002)
        session_events = []
        with mock.patch.object(v22, "V22Modem", modem_factory):
            originate = OnlineAudioSession(
                o_sock, a_sock.getsockname(), "originate", "answer",
                line_mode="bell212a", v42=True,
            )
            answer = OnlineAudioSession(
                a_sock, o_sock.getsockname(), "answer", "originate",
                line_mode="bell212a", v42=True,
            )
        originate.start()
        answer.start()
        try:
            rx_originate, rx_answer = bytearray(), bytearray()
            payload_originate = bytes((i * 29 + 0x51) & 0xff for i in range(128))
            payload_answer = bytes((i * 47 + 0xc3) & 0xff for i in range(128))
            payload_sent = False
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                originate.poll(time.monotonic())
                answer.poll(time.monotonic())
                if (originate._v22.v42_mode == "lapm"
                        and answer._v22.v42_mode == "lapm"
                        and not payload_sent):
                    originate.feed_terminal_bytes(payload_originate)
                    answer.feed_terminal_bytes(payload_answer)
                    payload_sent = True
                for session, sock, received in (
                    (answer, a_sock, rx_answer), (originate, o_sock, rx_originate),
                ):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    received.extend(session.handle_rtp_packet(packet))
                if (payload_sent and rx_originate == payload_answer
                        and rx_answer == payload_originate):
                    break

            self.assertTrue(payload_sent,
                            (originate._v22.v42_mode, answer._v22.v42_mode))
            self.assertEqual(bytes(rx_originate), payload_answer)
            self.assertEqual(bytes(rx_answer), payload_originate)
            answer_link = answer._v22.span_link
            self.assertIsNotNone(answer_link)

            # The real DCE ATH callback requests orderly release. Keep both RTP
            # loops alive until the independent UA is received and both LAPM
            # endpoints leave data mode; only then allow simulated call teardown.
            dce = AtDce(hangup=originate.request_v42_release)
            dce.feed(b"ATE0\r")
            dce.online = True
            self.assertEqual(dce.feed(b"ATH\r"), b"\r\nOK\r\n")
            session_events.append("ATH")
            release_deadline = time.monotonic() + 12
            while time.monotonic() < release_deadline:
                originate.poll(time.monotonic())
                answer.poll(time.monotonic())
                for session, sock in ((answer, a_sock), (originate, o_sock)):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_packet(packet)
                if (originate.v42_release_complete()
                        and answer_link.state == "down"):
                    session_events.append("LAPM_RELEASE_COMPLETE")
                    session_events.append("CALL_TEARDOWN_ALLOWED")
                    break

            self.assertEqual(session_events,
                             ["ATH", "LAPM_RELEASE_COMPLETE", "CALL_TEARDOWN_ALLOWED"])
            self.assertEqual(originate._v22._v42.state, "released",
                             originate._v22._v42.events)
            self.assertTrue(any("DISC acknowledged by UA" in event
                                for event in originate._v22._v42.events),
                            originate._v22._v42.events)
            self.assertIn("LINK_DISCONNECTED", answer_link.events)
        finally:
            originate.stop()
            answer.stop()
            o_sock.close()
            a_sock.close()

    def test_raw_media_payload_refreshes_carrier_liveness(self) -> None:
        session = object.__new__(OnlineAudioSession)
        session._last_rtp_at = None
        session._receive_pcm = lambda pcm: pcm
        result = session.handle_rtp_payload(bytes((0xFF,)) * 160)
        self.assertEqual(len(result), 320)
        self.assertIsNotNone(session._last_rtp_at)

    def test_originate_and_answer_sessions_exchange_bytes_over_loopback_rtp(self) -> None:
        import socket

        originate_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        originate_sock.bind(("127.0.0.1", 0))
        answer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        answer_sock.bind(("127.0.0.1", 0))
        originate_sock.settimeout(2)
        answer_sock.settimeout(2)

        originate_session = OnlineAudioSession(
            originate_sock, answer_sock.getsockname(), "originate", "answer"
        )
        answer_session = OnlineAudioSession(
            answer_sock, originate_sock.getsockname(), "answer", "originate"
        )
        originate_session.start()
        answer_session.start()
        try:
            originate_session.feed_terminal_bytes(b"HI")
            answer_session.feed_terminal_bytes(b"YO")

            decoded_by_answer = bytearray()
            decoded_by_originate = bytearray()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and (
                len(decoded_by_answer) < 2 or len(decoded_by_originate) < 2
            ):
                try:
                    packet, _ = answer_sock.recvfrom(2048)
                    decoded_by_answer.extend(answer_session.handle_rtp_payload(packet[12:]))
                except (socket.timeout, OSError):
                    pass
                try:
                    packet, _ = originate_sock.recvfrom(2048)
                    decoded_by_originate.extend(
                        originate_session.handle_rtp_payload(packet[12:])
                    )
                except (socket.timeout, OSError):
                    pass
            self.assertEqual(bytes(decoded_by_answer), b"HI")
            self.assertEqual(bytes(decoded_by_originate), b"YO")
        finally:
            originate_session.stop()
            answer_session.stop()
            originate_sock.close()
            answer_sock.close()

    def test_flow_control_signal_asserts_and_deasserts_xoff(self) -> None:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        # Point the session at itself; nothing here actually needs the
        # send-thread running, so it is never started.
        session = OnlineAudioSession(sock, sock.getsockname(), "originate", "answer")
        try:
            self.assertIsNone(session.flow_control_signal())
            # 40 bytes * 10 bits/char = 400 queued bits, above the 300-bit
            # high watermark.
            session.feed_terminal_bytes(b"x" * 40)
            self.assertEqual(session.flow_control_signal(), bytes((0x13,)))
            self.assertTrue(session.xoff_asserted)
            # Still above the watermark -- no repeated XOFF.
            self.assertIsNone(session.flow_control_signal())
            # Drain the queue down (simulate the transmitter having sent
            # most of it) by taking samples until under the low watermark.
            while session.transmitter.pending() >= 100:
                session.transmitter.samples(160)
            self.assertEqual(session.flow_control_signal(), bytes((0x11,)))
            self.assertFalse(session.xoff_asserted)
        finally:
            sock.close()

    def test_rtp_gap_fill_prevents_permanent_receiver_desync(self) -> None:
        """A dropped RTP packet used to permanently wedge Bell103Receiver:
        it tracks bit timing as a continuously incrementing sample count,
        so simply skipping the lost packet's samples shifted every later
        sample's position relative to the sender's clock, and the
        in-progress character frame could never reach its 9 expected bit
        positions again (character detection stayed stuck forever).
        handle_rtp_packet's timestamp-gap silence-fill keeps the sample
        clock aligned, and Bell103Receiver now abandons a stuck in-progress
        character if its sampling window is overshot, so a single lost
        packet degrades at most the character(s) spanning the gap rather
        than wedging the receiver for the rest of the call."""
        import socket

        originate_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        originate_sock.bind(("127.0.0.1", 0))
        answer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        answer_sock.bind(("127.0.0.1", 0))
        originate_sock.settimeout(2)
        answer_sock.settimeout(2)

        originate_session = OnlineAudioSession(
            originate_sock, answer_sock.getsockname(), "originate", "answer"
        )
        answer_session = OnlineAudioSession(
            answer_sock, originate_sock.getsockname(), "answer", "originate"
        )
        originate_session.start()
        answer_session.start()
        try:
            originate_session.feed_terminal_bytes(b"BEFORE")
            decoded = bytearray()
            packet_count = 0
            dropped = False
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and len(decoded) < len(b"BEFORE"):
                try:
                    packet, _ = answer_sock.recvfrom(2048)
                    packet_count += 1
                    if packet_count == 3 and not dropped:
                        # Drop exactly one RTP packet mid-transfer, like a
                        # lost UDP datagram.
                        dropped = True
                        continue
                    decoded.extend(answer_session.handle_rtp_packet(packet))
                except (socket.timeout, OSError):
                    pass
            self.assertTrue(dropped)
            # The dropped packet's own samples are gone for good, so the
            # character(s) spanning that gap are not recoverable here --
            # but detection must not be wedged: more data sent afterwards
            # has to keep decoding correctly rather than staying stuck.
            originate_session.feed_terminal_bytes(b"AFTER")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not decoded.endswith(b"AFTER"):
                try:
                    packet, _ = answer_sock.recvfrom(2048)
                    decoded.extend(answer_session.handle_rtp_packet(packet))
                except (socket.timeout, OSError):
                    pass
            self.assertTrue(
                decoded.endswith(b"AFTER"),
                f"receiver never recovered after the dropped packet: {bytes(decoded)!r}",
            )
        finally:
            originate_session.stop()
            answer_session.stop()
            originate_sock.close()
            answer_sock.close()

    def test_arq_layer_round_trips_byte_exact_over_live_audio_path(self) -> None:
        """Confirms the ArqSession CRC/ACK framing introduced for simplified
        error control does not itself corrupt or lose data when wired
        through the real Bell103 send/receive audio path end to end (a
        clean channel -- ArqSession's own resilience to byte-level loss is
        covered in isolation by test_lapm_lite.py).

        NOTE on scope: this does *not* prove the ARQ layer survives real
        RTP packet loss or audio-sample corruption over this project's
        current Bell103 demodulator. Experimentation while building this
        test showed dropped RTP packets or corrupted PCM samples can
        desync the demodulator's bit-level timing badly enough that the
        16-bit CRC/stop-and-wait ARQ above it cannot recover a clean byte
        stream -- the current demodulator lacks the clock-recovery/PLL
        robustness real modems have for this. See lab/asterisk/README.md's
        FEC/LAPM discussion for the full assessment; closing that gap is
        future demodulator robustness work, not an ARQ framing problem."""
        import socket

        originate_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        originate_sock.bind(("127.0.0.1", 0))
        answer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        answer_sock.bind(("127.0.0.1", 0))
        originate_sock.settimeout(2)
        answer_sock.settimeout(2)

        originate_session = OnlineAudioSession(
            originate_sock, answer_sock.getsockname(), "originate", "answer"
        )
        answer_session = OnlineAudioSession(
            answer_sock, originate_sock.getsockname(), "answer", "originate"
        )
        originate_arq = ArqSession(max_payload=32)
        answer_arq = ArqSession(max_payload=32)
        originate_session.start()
        answer_session.start()
        try:
            payload = bytes(range(128))
            originate_arq.feed_terminal_bytes(payload)

            received = bytearray()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and len(received) < len(payload):
                framed = originate_arq.pump_outgoing(time.monotonic())
                if framed:
                    originate_session.feed_terminal_bytes(framed)
                framed = answer_arq.pump_outgoing(time.monotonic())
                if framed:
                    answer_session.feed_terminal_bytes(framed)
                try:
                    packet, _ = answer_sock.recvfrom(2048)
                    decoded = answer_session.handle_rtp_packet(packet)
                    if decoded:
                        received.extend(answer_arq.feed_received(decoded))
                except (socket.timeout, OSError):
                    pass
                try:
                    packet, _ = originate_sock.recvfrom(2048)
                    decoded = originate_session.handle_rtp_packet(packet)
                    if decoded:
                        originate_arq.feed_received(decoded)
                except (socket.timeout, OSError):
                    pass
            self.assertEqual(bytes(received), payload)
        finally:
            originate_session.stop()
            answer_session.stop()
            originate_sock.close()
            answer_sock.close()


class ModulationNegotiationTests(unittest.TestCase):
    """Two OnlineAudioSession endpoints negotiating the modulation over the
    real RTP/audio path (UDP loopback), then exchanging data."""

    def _pair(self, originate_modes=(1, 2), answer_modes=(1, 2)):
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        o = OnlineAudioSession(
            o_sock, a_sock.getsockname(), "originate", "answer",
            negotiate=True, supported_modes=originate_modes,
        )
        a = OnlineAudioSession(
            a_sock, o_sock.getsockname(), "answer", "originate",
            negotiate=True, supported_modes=answer_modes,
        )
        return o, a, o_sock, a_sock

    @staticmethod
    def _pump(o, a, o_sock, a_sock, until, timeout=15.0):
        """Run both sessions until ``until(o_rx, a_rx)`` is true."""
        import socket

        o_rx, a_rx = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            o.poll(time.monotonic())
            a.poll(time.monotonic())
            for session, sock, rx in ((a, a_sock, a_rx), (o, o_sock, o_rx)):
                try:
                    packet, source = sock.recvfrom(2048)
                except (socket.timeout, OSError):
                    continue
                rx.extend(session.handle_rtp_datagram(source,packet))
            if until(o_rx, a_rx):
                break
        return o_rx, a_rx

    def _run(self, originate_modes, answer_modes, expected_mode, speed):
        o, a, o_sock, a_sock = self._pair(originate_modes, answer_modes)
        o.start()
        a.start()
        try:
            self._pump(o, a, o_sock, a_sock,
                       lambda *_: o.negotiation_complete and a.negotiation_complete)
            self.assertTrue(o.negotiation_complete and a.negotiation_complete)
            self.assertEqual((o.mode, a.mode), (expected_mode, expected_mode))
            self.assertEqual((o.line_speed, a.line_speed), (speed, speed))
            # Data typed while negotiating (or just after) must survive the
            # switch and arrive intact, both directions at once.
            o.feed_terminal_bytes(b"from-originate\x00\xff\x55")
            a.feed_terminal_bytes(b"from-answer\xaa\x01")
            o_rx, a_rx = self._pump(
                o, a, o_sock, a_sock,
                lambda o_rx, a_rx: a_rx.endswith(b"\x55") and o_rx.endswith(b"\x01"),
            )
            self.assertEqual(bytes(a_rx), b"from-originate\x00\xff\x55")
            self.assertEqual(bytes(o_rx), b"from-answer\xaa\x01")
        finally:
            o.stop()
            a.stop()
            o_sock.close()
            a_sock.close()

    def test_both_sides_negotiate_bell212_and_exchange_data(self) -> None:
        self._run((1, 2), (1, 2), 2, 1200)

    def test_falls_back_to_bell103_when_one_side_lacks_bell212(self) -> None:
        self._run((1, 2), (1,), 1, 300)
        self._run((1,), (1, 2), 1, 300)

    def test_data_fed_during_negotiation_is_held_not_corrupted(self) -> None:
        o, a, o_sock, a_sock = self._pair()
        o.start()
        a.start()
        try:
            o.feed_terminal_bytes(b"early-bird")  # before any mode is agreed
            _, a_rx = self._pump(
                o, a, o_sock, a_sock, lambda o_rx, a_rx: a_rx.endswith(b"bird")
            )
            self.assertEqual(bytes(a_rx), b"early-bird")
            self.assertEqual(o.mode, 2)
        finally:
            o.stop()
            a.stop()
            o_sock.close()
            a_sock.close()

    def test_silent_peer_times_out_to_bell103(self) -> None:
        import socket

        from negotiation import ModeNegotiator

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer.bind(("127.0.0.1", 0))  # never reads or answers
        session = OnlineAudioSession(
            sock, peer.getsockname(), "originate", "answer", negotiate=True
        )
        session._negotiator = ModeNegotiator(timeout_seconds=0.3)
        session._negotiator.start(time.monotonic())
        session.start()
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not session.negotiation_complete:
                session.poll(time.monotonic())
                time.sleep(0.02)
            self.assertTrue(session.negotiation_complete)
            self.assertEqual((session.mode, session.line_speed), (1, 300))
        finally:
            session.stop()
            sock.close()
            peer.close()

    def test_connect_result_reports_negotiated_speed(self) -> None:
        dce = AtDce()
        dce.connect_speed = 1200
        self.assertEqual(dce._result("CONNECT"), b"\r\nCONNECT 1200\r\n")
        dce.feed(b"ATX0\r")
        self.assertEqual(dce._result("CONNECT"), b"\r\nCONNECT\r\n")

    def test_deferred_connect_is_not_announced_on_answer(self) -> None:
        dce = AtDce()
        dce.defer_connect = True
        dce.attach_call_control(lambda: True, lambda: None)
        dce.begin_incoming_call()
        dce.feed(b"ATA\r")
        self.assertNotIn(b"CONNECT", dce.take_notifications())


class ConnectMessageTests(unittest.TestCase):
    def _dce(self, suffix="/LAPM", speed=2400):
        dce = AtDce()
        dce.connect_speed = speed
        dce.connect_suffix = suffix
        return dce

    def test_lapm_shows_in_connect_at_every_x_level_but_x0(self) -> None:
        for level in (1, 2, 3, 4):
            dce = self._dce()
            dce.feed(f"ATX{level}\r".encode())
            self.assertEqual(dce._result("CONNECT"), b"\r\nCONNECT 2400/LAPM\r\n", level)
        dce = self._dce()
        dce.feed(b"ATX0\r")
        self.assertEqual(dce._result("CONNECT"), b"\r\nCONNECT\r\n")

    def test_default_x_level_shows_lapm_and_no_suffix_without_error_control(self) -> None:
        self.assertEqual(self._dce()._result("CONNECT"), b"\r\nCONNECT 2400/LAPM\r\n")
        self.assertEqual(self._dce(suffix="")._result("CONNECT"), b"\r\nCONNECT 2400\r\n")
        self.assertEqual(self._dce(suffix="", speed=1200)._result("CONNECT"),
                         b"\r\nCONNECT 1200\r\n")

    def test_the_announced_connect_carries_the_suffix(self) -> None:
        # The call layer sets speed/suffix, then announces CONNECT (both the
        # originate and the answer paths go through this).
        dce = self._dce()
        dce.notify_call_progress("CONNECT")
        self.assertEqual(dce.take_notifications(), b"\r\nCONNECT 2400/LAPM\r\n")

    def test_numeric_result_codes_follow_the_speed(self) -> None:
        for speed, code in ((300, "1"), (1200, "5"), (2400, "10")):
            dce = self._dce(speed=speed)
            dce.feed(b"ATV0\r")
            self.assertEqual(dce._result("CONNECT"), f"\r\n{code}\r\n".encode())
        dce = self._dce()
        dce.feed(b"ATV0X0\r")
        self.assertEqual(dce._result("CONNECT"), b"\r\n1\r\n")  # X0: plain CONNECT


class RawByDefaultTests(unittest.TestCase):
    def test_lab_arq_only_where_both_ends_are_lab_modems(self) -> None:
        self.assertEqual(default_error_control_for("bell103", True), 1)
        # Real-modem paths: lab ARQ would discard a real peer's data.
        self.assertEqual(default_error_control_for("bell103", False), 0)
        self.assertEqual(default_error_control_for("v22bis", True), 0)
        self.assertEqual(default_error_control_for("v22bis", False), 0)
        self.assertEqual(default_error_control_for("auto", True), 0)

    def test_atz_returns_to_the_line_mode_default(self) -> None:
        dce = AtDce()
        self.assertEqual(dce.error_control, 1)  # standalone default unchanged
        dce.default_error_control = dce.error_control = 0
        dce.feed(b"AT\\N1\r")
        self.assertEqual(dce.error_control, 1)  # explicit opt-in still works
        dce.feed(b"ATZ\r")
        self.assertEqual(dce.error_control, 0)  # ATZ restores raw

    def test_percent_c_selects_v42bis_and_atz_turns_it_off(self) -> None:
        dce = AtDce()
        self.assertEqual(dce.compression, 0)            # off by default
        self.assertIn(b"OK", dce.feed(b"AT%C1\r"))
        self.assertEqual(dce.compression, 1)
        self.assertIn(b"ERROR", dce.feed(b"AT%C7\r"))
        dce.feed(b"ATZ\r")
        self.assertEqual(dce.compression, 0)

    def test_n2_selects_mnp_error_control(self) -> None:
        dce = AtDce()
        self.assertIn(b"OK", dce.feed(b"AT\\N2\r"))
        self.assertEqual(dce.error_control, 2)
        self.assertIn(b"ERROR", dce.feed(b"AT\\N9\r"))

    def test_k_selects_and_validates_v42_break_policy(self) -> None:
        dce = AtDce()
        self.assertEqual(dce.break_control, 5)  # Hayes default: non-destructive, in-sequence
        for value in range(6):
            self.assertTrue(dce.feed(f"AT\\K{value}\r".encode()).endswith(b"OK\r\n"))
            self.assertEqual(dce.break_control, value)
        self.assertTrue(dce.feed(b"AT\\K6\r").endswith(b"ERROR\r\n"))
        self.assertEqual(dce.break_control, 5)
        self.assertEqual(v42_break_options_for_k(0), None)
        self.assertEqual(v42_break_options_for_k(1), (True, True))
        self.assertEqual(v42_break_options_for_k(2), None)
        self.assertEqual(v42_break_options_for_k(3), (False, True))
        self.assertEqual(v42_break_options_for_k(4), None)
        self.assertEqual(v42_break_options_for_k(5), (False, False))
        with self.assertRaises(ValueError):
            v42_break_options_for_k(6)

    def test_b_transmits_a_300ms_lapm_break_using_manual_k_mapping(self) -> None:
        engine = types.SimpleNamespace(v42_request_break=mock.Mock(return_value=True))
        dce = AtDce()
        dce.attach_v42_break_action(
            lambda **options: engine.v42_request_break(**options))
        dce.online = True
        for value, expected in ((0, (True, True)), (1, (True, True)),
                                (2, (False, True)), (3, (False, True)),
                                (4, (False, False)), (5, (False, False))):
            with self.subTest(k=value):
                dce.feed(f"AT\\K{value}\r".encode())
                self.assertTrue(dce.feed(b"AT\\B9\r").endswith(b"OK\r\n"))
                engine.v42_request_break.assert_called_once_with(
                    length_ms=300, destructive=expected[0], expedited=expected[1])
                engine.v42_request_break.reset_mock()

        self.assertEqual(v42_manual_break_options_for_k(0), (True, True))
        self.assertEqual(v42_manual_break_options_for_k(2), (False, True))
        self.assertEqual(v42_manual_break_options_for_k(4), (False, False))
        disconnected = AtDce()
        self.assertTrue(disconnected.feed(b"AT\\B\r").endswith(b"ERROR\r\n"))

    def test_hardware_and_command_break_requests_share_serial_capacity(self) -> None:
        queue = V42BreakRequestQueue(capacity=2)
        self.assertTrue(queue.enqueue(destructive=False, expedited=False))
        self.assertTrue(queue.enqueue(length_ms=300, destructive=True,
                                      expedited=False))
        self.assertFalse(queue.enqueue(destructive=False, expedited=True))
        request = mock.Mock(side_effect=(True, False, True))

        self.assertTrue(queue.service(request))
        request.assert_called_once_with(length_ms=None, destructive=False,
                                        expedited=False)
        self.assertEqual(queue.available, 0)
        self.assertFalse(queue.service(request))  # never overlap outstanding L-SIGNAL
        self.assertTrue(queue.complete())
        self.assertFalse(queue.complete())
        self.assertFalse(queue.service(request))  # rejected request stays queued
        self.assertEqual(len(queue.pending), 1)
        self.assertEqual(request.call_args,
                         mock.call(length_ms=300, destructive=True,
                                   expedited=False))
        self.assertTrue(queue.service(request))
        self.assertEqual(request.call_args,
                         mock.call(length_ms=300, destructive=True,
                                   expedited=False))
        self.assertTrue(queue.in_flight)
        self.assertEqual(queue.pending, [])
        queue.reset()
        self.assertEqual(queue.pending, [])
        self.assertFalse(queue.in_flight)

    def test_invalid_stored_break_policy_falls_back_to_default(self) -> None:
        dce = AtDce()
        dce.break_control = 9
        dce._normalize_stored_settings()
        self.assertEqual(dce.break_control, 5)

    def test_online_command_mode_keeps_the_call_active(self) -> None:
        dce = AtDce()
        dce.online = True
        dce._call = {"connected": True}
        self.assertTrue(dce.enter_online_command_mode())
        self.assertTrue(dce.online)  # call remains active; the event loop leaves data mode
        self.assertEqual(dce.take_notifications(), b"\r\nOK\r\n")
        self.assertEqual(dce._call, {"connected": True})


class V42ManualBreakIntegrationTests(unittest.TestCase):
    def test_hayes_break_crosses_production_lapm_and_preserves_payload_order(self) -> None:
        caller_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        answer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        caller_sock.bind(("127.0.0.1", 0))
        answer_sock.bind(("127.0.0.1", 0))
        caller_sock.settimeout(0.01)
        answer_sock.settimeout(0.01)
        caller = OnlineAudioSession(
            caller_sock, answer_sock.getsockname(), "originate", "answer",
            line_mode="v22bis", v22_rate=2400, v42=True, v42bis=False,
        )
        answer = OnlineAudioSession(
            answer_sock, caller_sock.getsockname(), "answer", "originate",
            line_mode="v22bis", v22_rate=2400, v42=True, v42bis=False,
        )
        dce = AtDce()
        dce.online = True
        requests = V42BreakRequestQueue()
        dce.attach_v42_break_action(
            lambda **options: requests.enqueue(**options))
        before = bytes((i * 31 + 9) & 0xff for i in range(91))
        after = bytes((i * 47 + 0xa5) & 0xff for i in range(83))
        received = bytearray()
        sent_before = sent_break = sent_after = False
        break_indicated = break_confirmed = False
        caller.start()
        answer.start()
        try:
            deadline = time.monotonic() + 18
            while time.monotonic() < deadline:
                now = time.monotonic()
                caller.poll(now)
                answer.poll(now)
                for session, sock in ((answer, answer_sock), (caller, caller_sock)):
                    try:
                        packet, source = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    delivered = session.handle_rtp_datagram(source, packet)
                    if session is answer:
                        received.extend(delivered)

                if (caller.negotiation_complete and answer.negotiation_complete
                        and not sent_before):
                    caller.feed_terminal_bytes(before)
                    sent_before = True
                if sent_before and received == before and not sent_break:
                    self.assertTrue(dce.feed(b"AT\\B9\r").endswith(b"OK\r\n"))
                    self.assertTrue(requests.service(caller.request_v42_break))
                    caller.feed_terminal_bytes(after)
                    sent_break = sent_after = True

                for event in answer.read_v42_break_events():
                    if event.get("type") == "break_indication":
                        self.assertEqual(event.get("length_ms"), 300)
                        self.assertTrue(answer.acknowledge_v42_break())
                        break_indicated = True
                for event in caller.read_v42_break_events():
                    if event.get("type") == "break_confirmed":
                        self.assertTrue(requests.complete())
                        break_confirmed = True
                    elif event.get("type") == "break_failed":
                        self.fail(f"LAPM BREAK failed: {event}")

                if (sent_after and break_indicated and break_confirmed
                        and received == before + after):
                    break

            self.assertTrue(caller.negotiation_complete)
            self.assertTrue(answer.negotiation_complete)
            self.assertTrue(sent_break)
            self.assertTrue(break_indicated)
            self.assertTrue(break_confirmed)
            self.assertEqual(bytes(received), before + after)
        finally:
            caller.stop()
            answer.stop()
            caller_sock.close()
            answer_sock.close()


def _spandsp_available() -> bool:
    try:
        import spandsp_v22
    except ImportError:
        return False
    return spandsp_v22.available()




class V32SessionTests(unittest.TestCase):
    """The DCE glue for --line-mode v32: a session wrapping our streaming V.32 caller. The far end
    here is the simulator's answerer-style second transmission, packetised as G.711 mu-law RTP."""

    def test_v32_session_reports_rate_and_delivers_data(self) -> None:
        import socket
        import struct as st
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "audio-peer"))
        import v32_sim_rx as sim

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer.bind(("127.0.0.1", 0))
        session = OnlineAudioSession(sock, peer.getsockname(), "originate", "answer", line_mode="v32")
        modem = session._v22
        self.assertTrue(getattr(modem, "is_v32", False))
        # jump into the receive state machine for the answerer's second transmission
        modem.state, modem._tx_started, modem._tx_tm0 = "tx", True, 0
        modem._rx_mode, modem._resume_at, modem._echo_state = "locate2", 0.6, "done"
        audio, text = sim.make_stream(rate=9600)
        got = bytearray()
        seq, ts = 0, 0
        try:
            for i in range(0, len(audio) - 160, 160):
                payload = bytes(linear_to_ulaw(max(-32768, min(32767, int(round(v))))) for v in audio[i:i + 160])
                pkt = st.pack("!BBHII", 0x80, 0, seq, ts, 1234) + payload
                modem._txn += 160
                got += session.handle_rtp_packet(pkt)
                seq, ts = seq + 1, ts + 160
            self.assertTrue(session.negotiation_complete)
            self.assertEqual(session.line_speed, 9600)
            self.assertFalse(session.failed)
            self.assertGreaterEqual(bytes(got).count(text[:50]), 1)
        finally:
            session.stop()
            sock.close()
            peer.close()

    def test_v32_session_delivers_exact_data_across_short_rtp_loss_bursts(self) -> None:
        """Exercise the production V.32 RTP gap filler on known simulator audio.

        A 40 ms reorder window absorbs short reordering; any remaining timestamp
        gap is filled before the V.32 receiver sees later audio. This proves
        sample-clock continuity for these controlled bursts, not LAPM or
        real-network qualification.
        """
        import socket
        import struct as st
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "audio-peer"))
        import v32_sim_rx as sim

        scenarios = {
            "clean": set(),
            "single_at_100": {100},
            "two_at_100": {100, 101},
            "five_at_100": set(range(100, 105)),
            "single_at_200": {200},
            "two_at_200": {200, 201},
            "five_at_200": set(range(200, 205)),
        }
        for rate in (4800, 7200, 9600, 12000, 14400):
            audio, expected = sim.make_stream(
                rate=rate, text=b"V32 controlled RTP loss test payload.\r\n" * 4
            )
            for name, dropped in scenarios.items():
                with self.subTest(rate=rate, name=name):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.bind(("127.0.0.1", 0))
                    peer.bind(("127.0.0.1", 0))
                    session = OnlineAudioSession(
                        sock, peer.getsockname(), "originate", "answer",
                        line_mode="v32", v32_rates=(rate,)
                    )
                    modem = session._v22
                    modem.state, modem._tx_started, modem._tx_tm0 = "tx", True, 0
                    modem._rx_mode, modem._resume_at, modem._echo_state = "locate2", 0.6, "done"
                    got = bytearray()
                    seq = timestamp = 0
                    try:
                        for i in range(0, len(audio) - 160, 160):
                            payload = bytes(
                                linear_to_ulaw(max(-32768, min(32767, int(round(v)))))
                                for v in audio[i : i + 160]
                            )
                            modem._txn += 160
                            if seq not in dropped:
                                packet = st.pack("!BBHII", 0x80, 0, seq, timestamp, 1234) + payload
                                got.extend(session.handle_rtp_packet(packet))
                            seq += 1
                            timestamp += 160
                        self.assertEqual(session.line_speed, rate)
                        self.assertTrue(session.negotiation_complete)
                        self.assertFalse(session.failed)
                        self.assertEqual(bytes(got), expected)
                        self.assertEqual(session._rtp_counts["concealed_samples"], 160 * len(dropped))
                    finally:
                        session.stop()
                        sock.close()
                        peer.close()

    def test_v32_session_can_answer_and_places_calls_with_the_caller(self) -> None:
        import socket

        for role, rx_role, name in (("answer", "originate", "V32Answerer"), ("originate", "answer", "V32Modem")):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                session = OnlineAudioSession(sock, ("127.0.0.1", 9), role, rx_role, line_mode="v32")
                self.assertEqual(type(session._v22).__name__, name)
                session.stop()
            finally:
                sock.close()

    def test_v34_line_mode_builds_caller_and_answerer_engines(self) -> None:
        import socket
        import types
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer.bind(("127.0.0.1", 0))
        try:
            session = OnlineAudioSession(sock, peer.getsockname(), "originate", "answer", line_mode="v34")
            self.assertTrue(getattr(session._v22, "is_v34", False))
            self.assertEqual(type(session._v22).__name__, "V34Modem")
            self.assertFalse(session.negotiation_complete)
            session.stop()
            answer = OnlineAudioSession(sock, peer.getsockname(), "answer", "originate", line_mode="v34")
            self.assertTrue(getattr(answer._v22, "is_v34", False))
            self.assertEqual(type(answer._v22).__name__, "V34AnswerModem")
            self.assertFalse(answer.negotiation_complete)
            answer.stop()
        finally:
            sock.close()
            peer.close()

    def test_v34_direct_mode_preserves_environment_rate_offer(self) -> None:
        import os
        import socket
        from unittest.mock import patch

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer.bind(("127.0.0.1", 0))
        try:
            with patch.dict(os.environ, {"V34_RATE": "33600", "V34_SYMS": "3429"}):
                caller = OnlineAudioSession(sock, peer.getsockname(), "originate", "answer", line_mode="v34")
                answerer = OnlineAudioSession(sock, peer.getsockname(), "answer", "originate", line_mode="v34")
            self.assertEqual(caller._v22._caller_args["data_rate"], 33600)
            self.assertEqual(answerer._v22._answerer_args["data_rate"], 33600)
            self.assertEqual(caller._v22._caller_args["symbol_rates"], [3429])
            self.assertEqual(answerer._v22._answerer_args["symbol_rates"], [3429])
            caller.stop()
            answerer.stop()
        finally:
            sock.close()
            peer.close()


class V22bisRateChangeRtpTests(unittest.TestCase):
    def test_online_ato4_ato5_changes_rate_and_keeps_rtp_payload(self) -> None:
        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.002)
        a_sock.settimeout(0.002)
        o = OnlineAudioSession(o_sock, a_sock.getsockname(), "originate", "answer",
                               line_mode="v22bis", v22_rate=2400)
        a = OnlineAudioSession(a_sock, o_sock.getsockname(), "answer", "originate",
                               line_mode="v22bis", v22_rate=2400)
        dce = AtDce()
        dce.feed(b"ATE0V1X4\r")
        dce.online = True
        dce.attach_online_control(o.request_online_control)
        dce.attach_resume_online(lambda: None)
        o.start()
        a.start()
        try:
            received_o, received_a = bytearray(), bytearray()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (o.negotiation_complete and a.negotiation_complete):
                now = time.monotonic()
                o.poll(now)
                a.poll(now)
                for session, sock, output in ((o, o_sock, received_o), (a, a_sock, received_a)):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    output.extend(session.handle_rtp_packet(packet))
            self.assertTrue(o.negotiation_complete and a.negotiation_complete)
            self.assertEqual((o.line_speed, a.line_speed), (2400, 2400))

            for parameter, target in ((4, 1200), (5, 2400)):
                self.assertEqual(dce.feed(f"ATO{parameter}\r".encode()), b"")
                self.assertTrue(dce.online_return_pending)
                transition_deadline = time.monotonic() + 8
                result = None
                while time.monotonic() < transition_deadline:
                    now = time.monotonic()
                    o.poll(now)
                    a.poll(now)
                    result = o.poll_online_control()
                    for session, sock, output in ((o, o_sock, received_o), (a, a_sock, received_a)):
                        try:
                            packet, _ = sock.recvfrom(2048)
                        except socket.timeout:
                            continue
                        output.extend(session.handle_rtp_packet(packet))
                    if result is not None:
                        break
                self.assertIsNotNone(result, "online rate control did not complete")
                self.assertEqual(result["status"], "rate-exchanged")
                dce.finish_online_control(o.line_speed, o.connect_suffix)
                self.assertIn(f"CONNECT {target}".encode(), dce.take_notifications())
                self.assertEqual((o.line_speed, a.line_speed), (target, target))

                sent_o = bytes((i * 23 + target) & 0xff for i in range(48))
                sent_a = bytes((i * 31 + target) & 0xff for i in range(48))
                received_o.clear()
                received_a.clear()
                o.feed_terminal_bytes(sent_o)
                a.feed_terminal_bytes(sent_a)
                payload_deadline = time.monotonic() + 5
                while time.monotonic() < payload_deadline:
                    now = time.monotonic()
                    o.poll(now)
                    a.poll(now)
                    for session, sock, output in ((o, o_sock, received_o), (a, a_sock, received_a)):
                        try:
                            packet, _ = sock.recvfrom(2048)
                        except socket.timeout:
                            continue
                        output.extend(session.handle_rtp_packet(packet))
                    if len(received_o) >= len(sent_a) and len(received_a) >= len(sent_o):
                        break
                self.assertEqual(bytes(received_o), sent_a)
                self.assertEqual(bytes(received_a), sent_o)
        finally:
            o.stop()
            a.stop()
            o_sock.close()
            a_sock.close()


@unittest.skipUnless(_spandsp_available(), "libspandsp not installed")
class V22bisSessionTests(unittest.TestCase):
    def test_v22bis_sessions_train_and_exchange_data_over_rtp(self) -> None:
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        o = OnlineAudioSession(o_sock, a_sock.getsockname(), "originate", "answer",
                               line_mode="v22bis")
        a = OnlineAudioSession(a_sock, o_sock.getsockname(), "answer", "originate",
                               line_mode="v22bis")
        o.start()
        a.start()
        try:
            o_rx, a_rx = bytearray(), bytearray()
            sent = False
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if o.negotiation_complete and a.negotiation_complete and not sent:
                    self.assertEqual((o.line_speed, a.line_speed), (1200, 1200))
                    o.feed_terminal_bytes(b"ping from originate\r\n")
                    a.feed_terminal_bytes(b"pong from answer\r\n")
                    sent = True
                for session, sock, rx in ((a, a_sock, a_rx), (o, o_sock, o_rx)):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    rx.extend(session.handle_rtp_packet(packet))
                if sent and a_rx.endswith(b"\n") and o_rx.endswith(b"\n"):
                    break
            self.assertTrue(sent, "V.22bis sessions never trained")
            self.assertEqual(bytes(a_rx), b"ping from originate\r\n")
            self.assertEqual(bytes(o_rx), b"pong from answer\r\n")
        finally:
            o.stop()
            a.stop()
            o_sock.close()
            a_sock.close()


    def test_v42_sessions_hold_connect_until_lapm_and_report_v42(self) -> None:
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        o = OnlineAudioSession(o_sock, a_sock.getsockname(), "originate", "answer",
                               line_mode="v22bis", v22_rate=2400, v42=True)
        a = OnlineAudioSession(a_sock, o_sock.getsockname(), "answer", "originate",
                               line_mode="v22bis", v22_rate=2400, v42=True)
        o.start()
        a.start()
        try:
            o_rx, a_rx = bytearray(), bytearray()
            sent = False
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                o.poll(time.monotonic())
                a.poll(time.monotonic())
                if o.negotiation_complete and a.negotiation_complete and not sent:
                    # CONNECT is only announced once error control is resolved.
                    self.assertEqual((o.connect_suffix, a.connect_suffix), ("/LAPM", "/LAPM"))
                    self.assertEqual((o.line_speed, a.line_speed), (2400, 2400))
                    o.feed_terminal_bytes(b"over lapm from originate\r\n")
                    a.feed_terminal_bytes(b"over lapm from answer\r\n")
                    sent = True
                for session, sock, rx in ((a, a_sock, a_rx), (o, o_sock, o_rx)):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    rx.extend(session.handle_rtp_packet(packet))
                if sent and a_rx.endswith(b"\n") and o_rx.endswith(b"\n"):
                    break
            self.assertTrue(sent, "sessions never completed V.42 negotiation")
            self.assertEqual(bytes(a_rx), b"over lapm from originate\r\n")
            self.assertEqual(bytes(o_rx), b"over lapm from answer\r\n")

            # Exercise the end-to-end control path over live DSP/RTP: the
            # receiver applies its DTE-side action before sending BRKACK, and
            # sender data queued after BRK is held until that confirmation.
            self.assertTrue(o.request_v42_break(length_ms=40))
            o.feed_terminal_bytes(b"after break\r\n")
            indication_seen = confirmation_seen = False
            release_deadline = time.monotonic() + 8
            while time.monotonic() < release_deadline:
                o.poll(time.monotonic())
                a.poll(time.monotonic())
                for session, sock, rx in ((a, a_sock, a_rx), (o, o_sock, o_rx)):
                    for event in session.read_v42_break_events():
                        if event["type"] == "break_indication":
                            self.assertEqual(event["length_ms"], 40)
                            indication_seen = True  # stands for successful DTE BREAK assertion
                            self.assertTrue(session.acknowledge_v42_break())
                        elif event["type"] == "break_confirmed":
                            confirmation_seen = True
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    rx.extend(session.handle_rtp_packet(packet))
                if confirmation_seen and a_rx.endswith(b"after break\r\n"):
                    break
            self.assertTrue(indication_seen, "receiver did not expose the L-SIGNAL indication")
            self.assertTrue(confirmation_seen, "sender did not receive BRKACK")
            self.assertTrue(a_rx.endswith(b"after break\r\n"),
                            "post-BREAK data was not released after acknowledgement")

            # Exercise the DCE's local ATH callback against the live media
            # pump. It requests DISC; the media path must stay running long
            # enough for the peer's UA and both LAPM entities to fall back.
            dce = AtDce(hangup=o.request_v42_release)
            dce.feed(b"ATE0\r")
            dce.online = True
            self.assertEqual(dce.feed(b"ATH\r"), b"\r\nOK\r\n")
            self.assertEqual(o._online_control.engine().v42_mode, "lapm")
            release_deadline = time.monotonic() + 8
            while time.monotonic() < release_deadline:
                o.poll(time.monotonic())
                a.poll(time.monotonic())
                for session, sock in ((a, a_sock), (o, o_sock)):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_packet(packet)
                if (o.v42_release_complete()
                        and a._online_control.engine().v42_mode != "lapm"):
                    break
            self.assertEqual(o._online_control.engine().v42_mode, "failed")
            self.assertEqual(a._online_control.engine().v42_mode, "failed")
        finally:
            o.stop()
            a.stop()
            o_sock.close()
            a_sock.close()

    def test_silent_rtp_after_connect_is_link_lost_even_while_the_modem_reports_carrier(self) -> None:
        s = OnlineAudioSession.__new__(OnlineAudioSession)
        s._v22 = types.SimpleNamespace(v42_lost=False, connected=True)
        s._carrier_down_at = None
        s._carrier_loss_seconds = 1.0
        s._created = time.monotonic() - 5
        s._last_rtp_at = None
        # Bounded V.42bis recovery failure marks the data link lost even while
        # RTP and the physical modem carrier remain present.
        s._v22.v42_lost = True
        self.assertTrue(s.link_lost)
        s._v22.v42_lost = False
        with mock.patch.object(OnlineAudioSession, "negotiation_complete", new_callable=mock.PropertyMock, return_value=True):
            self.assertTrue(s.link_lost)
            s._last_rtp_at = time.monotonic()
            self.assertFalse(s.link_lost)
            s._last_rtp_at = time.monotonic() - 2
            self.assertTrue(s.link_lost)
        with mock.patch.object(OnlineAudioSession, "negotiation_complete", new_callable=mock.PropertyMock, return_value=False):
            s._last_rtp_at = time.monotonic() - 2
            self.assertFalse(s.link_lost)

    def test_carrier_loss_after_connect_is_reported_so_the_call_is_dropped(self) -> None:
        import socket

        o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        o_sock.bind(("127.0.0.1", 0))
        a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        a_sock.bind(("127.0.0.1", 0))
        o_sock.settimeout(0.02)
        a_sock.settimeout(0.02)
        o = OnlineAudioSession(o_sock, a_sock.getsockname(), "originate", "answer",
                               line_mode="v22bis", v22_rate=2400,
                               carrier_loss_seconds=1.5)
        a = OnlineAudioSession(a_sock, o_sock.getsockname(), "answer", "originate",
                               line_mode="v22bis", v22_rate=2400)
        o.start()
        a.start()
        try:
            deadline = time.monotonic() + 40
            up = False
            while time.monotonic() < deadline:
                o.poll(time.monotonic())
                a.poll(time.monotonic())
                if o.negotiation_complete and a.negotiation_complete:
                    up = True
                    break
                for session, sock in ((a, a_sock), (o, o_sock)):
                    try:
                        packet, _ = sock.recvfrom(2048)
                    except (socket.timeout, OSError):
                        continue
                    session.handle_rtp_packet(packet)
            self.assertTrue(up, "line never came up")
            self.assertFalse(o.link_lost)
            # Now the far end's audio stops arriving entirely (a dead leg).
            lost_by = time.monotonic() + 15
            while time.monotonic() < lost_by and not o.link_lost:
                o.poll(time.monotonic())
                try:
                    a_sock.recvfrom(2048)  # drain; nothing is delivered to `o`
                except (socket.timeout, OSError):
                    pass
                # Deliver silence so the receiver notices the carrier is gone.
                o.handle_rtp_payload(b"\xff" * 160)
                time.sleep(0.02)
            self.assertTrue(o.link_lost, "carrier loss was never reported")
        finally:
            o.stop()
            a.stop()
            o_sock.close()
            a_sock.close()


class V21DceIntegrationTests(unittest.TestCase):
    def test_run_pty_originates_atd_and_receives_v25_plus_independent_v21(self) -> None:
        """Exercise ATD and the caller-side DCE against an independent V.21 answer waveform."""
        shared = {"session": None, "output": bytearray(), "peer_errors": []}
        fixture = (Path(__file__).parent / "audio-peer" / "fixtures" /
                   "reference" / "independent-fsk" / "v21-answer.raw")
        if not fixture.is_file():
            self.skipTest("optional independent audio fixture is not in the public export")
        independent_data = struct.unpack(
            f"<{fixture.stat().st_size // 2}h", fixture.read_bytes())
        expected = b"ELMOD V21 ANSWER 9876543210\r"
        # Compose the V.25 answer prologue with the independently generated
        # V.21 answer-channel vector; the modulation payload itself is not
        # produced by the DUT.
        remote_audio = [0] * 16000
        for index in range(26400):
            phase = (2 * math.pi * 2100 * index / 8000
                     + math.pi * (index // 3600))
            remote_audio.append(round(7000 * math.sin(phase)))
        remote_audio.extend([0] * 600)
        remote_audio.extend(independent_data)
        peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer_sock.bind(("127.0.0.1", 0))
        peer_sock.settimeout(0.01)
        media_reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        media_reservation.bind(("127.0.0.1", 0))
        media_port = media_reservation.getsockname()[1]
        media_reservation.close()

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "v21-outgoing-run-pty-socketpair"
                self.supports_modem_control = False
                self.closed = False
                self.started = time.monotonic()
                self.dte.sendall(b"ATE0\\N0\rAT+MS=V21,0,300,300\rATD1001\r")

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                shared["output"].extend(data)
                if b"CONNECT 300" in shared["output"] and not shared.get("dte_data_sent"):
                    shared["dte_data_sent"] = True
                    self.dte.sendall(b"LOCAL V.21 CALLER\r")

            def service(self, readable, writable=()):
                if expected in shared["output"]:
                    raise KeyboardInterrupt
                if time.monotonic() - self.started > 20:
                    raise AssertionError(
                        f"V.21 ATD call timed out: output={bytes(shared['output'][-256:])!r}, "
                        f"session={getattr(shared['session'], 'modulation_name', None)!r}"
                    )
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]

            def start(self):
                pass

            def close(self):
                pass

        class FakeSipCall:
            def __init__(self, *args, **kwargs):
                self.on_connected = kwargs["on_connected"]
                self.hungup = False

            def dial(self, number):
                shared["dialed_number"] = number
                self.on_connected(peer_sock.getsockname())

            def hangup(self):
                self.hungup = True

        serial = FakeSerial()
        stop_peer = threading.Event()

        def transmit_answer():
            try:
                deadline = time.monotonic() + 3
                while shared["session"] is None and time.monotonic() < deadline:
                    if stop_peer.wait(0.01):
                        return
                if shared["session"] is None:
                    raise AssertionError("ATD did not create the V.21 caller session")
                sequence = timestamp = 0
                for offset in range(0, len(remote_audio), 160):
                    if stop_peer.is_set():
                        return
                    frame = remote_audio[offset:offset + 160]
                    frame.extend([0] * (160 - len(frame)))
                    packet = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, 0x56323132)
                    packet += bytes(linear_to_ulaw(max(-32768, min(32767, sample)))
                                    for sample in frame)
                    peer_sock.sendto(packet, ("127.0.0.1", media_port))
                    sequence = (sequence + 1) & 0xFFFF
                    timestamp = (timestamp + 160) & 0xFFFFFFFF
                    if stop_peer.wait(0.02):
                        return
            except Exception as exc:
                shared["peer_errors"].append(exc)

        peer_thread = threading.Thread(target=transmit_answer, daemon=True)
        peer_thread.start()
        original_session = OnlineAudioSession

        def track_session(*args, **kwargs):
            session = original_session(*args, **kwargs)
            shared["session"] = session
            return session

        try:
            with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                    mock.patch("modem_dce.SipOriginateCall", FakeSipCall), \
                    mock.patch("modem_dce.OnlineAudioSession", side_effect=track_session):
                run_pty(
                    None, sip_host="127.0.0.1", media_host="127.0.0.1",
                    media_port=media_port, dial_host="127.0.0.1", line_mode="ms",
                    negotiate=False, serial_backend=serial,
                )
        finally:
            stop_peer.set()
            peer_thread.join(timeout=2)
            peer_sock.close()
            serial.dte.close()

        self.assertFalse(peer_thread.is_alive())
        self.assertTrue(serial.closed)
        self.assertEqual(shared["peer_errors"], [])
        self.assertEqual(shared.get("dialed_number"), "1001")
        self.assertTrue(shared.get("dte_data_sent"))
        self.assertIn(b"CONNECT 300", shared["output"])
        self.assertIn(expected, shared["output"])
        self.assertEqual(shared["session"].modulation_name, "V.21")

    def test_run_pty_answers_and_receives_independent_minimodem_waveform(self) -> None:
        """Exercise ATA, V.25 answer timing and V.21 RX through the live DCE loop."""
        shared = {
            "session": None,
            "session_created": None,
            "sip_hungup": False,
            "peer_errors": [],
        }
        fixture = (Path(__file__).parent / "audio-peer" / "fixtures" /
                   "reference" / "independent-fsk" / "v21-orig.raw")
        if not fixture.is_file():
            self.skipTest("optional independent audio fixture is not in the public export")
        reference = struct.unpack(f"<{fixture.stat().st_size // 2}h", fixture.read_bytes())
        expected = b"ELMOD V21 ORIGINATE 0123456789\r"
        peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        peer_sock.bind(("127.0.0.1", 0))
        peer_sock.settimeout(0.01)
        media_reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        media_reservation.bind(("127.0.0.1", 0))
        media_port = media_reservation.getsockname()[1]
        media_reservation.close()

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "v21-run-pty-socketpair"
                self.supports_modem_control = False
                self.closed = False
                self.output = bytearray()
                self.ata_sent = False
                self.data_sent = False
                self.started = time.monotonic()
                self.dte.sendall(b"ATE0\\N0\rAT+MS=V21,0,300,300\r")

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                self.output.extend(data)
                if not self.ata_sent and b"RING" in self.output:
                    self.ata_sent = True
                    self.dte.sendall(b"ATA\r")
                if not self.data_sent and b"CONNECT 300" in self.output:
                    self.data_sent = True
                    self.dte.sendall(b"LOCAL V.21 ANSWER\r")

            def service(self, readable, writable=()):
                if time.monotonic() - self.started > 18:
                    raise AssertionError(
                        f"V.21 DCE call timed out: output={bytes(self.output[-256:])!r}, "
                        f"session={getattr(shared['session'], 'modulation_name', None)!r}"
                    )
                if expected in self.output:
                    raise KeyboardInterrupt
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        class FakeSipServer:
            def __init__(self, *args, **kwargs):
                self.on_incoming = kwargs["on_incoming"]
                self.on_ring = kwargs["on_ring"]
                self.on_hangup = kwargs["on_hangup"]
                self.last_caller = {"number": "v21-test", "technology": "SIP"}

            def start(self):
                self.on_incoming()
                self.on_ring()

            def answer(self):
                return True

            def remote_rtp(self):
                return peer_sock.getsockname()

            def hangup(self):
                shared["sip_hungup"] = True

            def close(self):
                pass

        serial = FakeSerial()
        stop_peer = threading.Event()

        def transmit_independent_vector():
            try:
                deadline = time.monotonic() + 3
                while shared["session"] is None and time.monotonic() < deadline:
                    if stop_peer.wait(0.01):
                        return
                session = shared["session"]
                if session is None:
                    raise AssertionError("run_pty did not create the V.21 answer session")
                # V.25 answer tone consists of 2 s billing silence, 3.3 s tone
                # with reversals and 75 ms post-tone silence. Wait through it
                # before sending this independent minimodem-originated payload.
                send_at = shared["session_created"] + 5.45
                if stop_peer.wait(max(0.0, send_at - time.monotonic())):
                    return
                sequence, timestamp = 0, 0
                for offset in range(0, len(reference), 160):
                    if stop_peer.is_set():
                        return
                    frame = list(reference[offset:offset + 160])
                    frame.extend([0] * (160 - len(frame)))
                    packet = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, 0x56323131)
                    packet += bytes(linear_to_ulaw(sample) for sample in frame)
                    peer_sock.sendto(packet, ("127.0.0.1", media_port))
                    sequence = (sequence + 1) & 0xFFFF
                    timestamp = (timestamp + 160) & 0xFFFFFFFF
                    if stop_peer.wait(0.02):
                        return
            except Exception as exc:
                shared["peer_errors"].append(exc)

        peer_thread = threading.Thread(target=transmit_independent_vector, daemon=True)
        peer_thread.start()
        original_session = OnlineAudioSession

        def track_session(*args, **kwargs):
            session = original_session(*args, **kwargs)
            shared["session"] = session
            shared["session_created"] = time.monotonic()
            return session

        try:
            with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                    mock.patch("modem_dce.OnlineAudioSession", side_effect=track_session):
                run_pty(
                    None, sip_host="127.0.0.1", media_host="127.0.0.1",
                    media_port=media_port, line_mode="ms", negotiate=False,
                    serial_backend=serial,
                )
        finally:
            stop_peer.set()
            peer_thread.join(timeout=2)
            peer_sock.close()
            serial.dte.close()

        output = bytes(serial.output)
        self.assertFalse(peer_thread.is_alive())
        self.assertEqual(shared["peer_errors"], [])
        self.assertTrue(serial.closed)
        self.assertTrue(serial.ata_sent)
        self.assertTrue(serial.data_sent)
        self.assertIn(b"CONNECT 300", output)
        self.assertIn(expected, output)
        self.assertEqual(shared["session"].modulation_name, "V.21")


class V23SessionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("minimodem"), "optional minimodem is not installed")
    def test_run_pty_v23_caller_uses_independent_forward_and_backward_channels(self) -> None:
        """Exercise ATD and both caller channels against minimodem's independent FSK."""
        dte_payload = b"independent V.23 caller forward\r\n"
        peer_payload = b"independent 75 baud caller return\r\n"
        shared = {"output": bytearray(), "session": None, "connected_at": None,
                  "peer_done": False, "peer_errors": [], "captured_pcm": []}
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-caller-interop-") as temp:
            tx_path = Path(temp) / "minimodem-backward.wav"
            generated = subprocess.run(
                [shutil.which("minimodem"), "--tx", "--quiet", "-M", "390", "-S", "450",
                 "-f", str(tx_path), "75"], input=peer_payload,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr.decode(errors="replace"))
            with wave.open(str(tx_path), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (1, 2, 48000))
                raw = wav.readframes(wav.getnframes())
            data_pcm = list(struct.unpack(f"<{len(raw) // 2}h", raw)[::6])
            mark = [round(9000 * math.sin(2 * math.pi * 390 * n / 8000))
                    for n in range(8000)]
            peer_pcm = mark + data_pcm + mark

            peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            peer_sock.bind(("127.0.0.1", 0))
            peer_sock.setblocking(False)
            reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            reservation.bind(("127.0.0.1", 0))
            media_port = reservation.getsockname()[1]
            reservation.close()
            stop_peer = threading.Event()

            class FakeSerial:
                def __init__(self):
                    self.modem, self.dte = socket.socketpair()
                    self.modem.setblocking(False)
                    self.name = "v23-minimodem-caller"
                    self.supports_modem_control = False
                    self.closed = False
                    self.started = time.monotonic()
                    self.dte.sendall(b"ATE0\\N0\rAT+MS=V23,0,1200,1200\rATD1001\r")

                def fds(self):
                    return [self.modem.fileno()]

                def write_fds(self):
                    return []

                def write(self, data):
                    shared["output"].extend(data)
                    if shared["connected_at"] is None and b"CONNECT 1200" in shared["output"]:
                        shared["connected_at"] = time.monotonic()
                        self.dte.sendall(dte_payload)

                def service(self, readable, writable=()):
                    if time.monotonic() - self.started > 14:
                        raise AssertionError(
                            f"V.23 caller timed out: output={bytes(shared['output'][-180:])!r}, "
                            f"captured={len(shared['captured_pcm'])} samples, "
                            f"session={getattr(shared.get('session'), 'link_stats', lambda: None)()}"
                        )
                    while True:
                        try:
                            packet, _ = peer_sock.recvfrom(2048)
                        except BlockingIOError:
                            break
                        if len(packet) >= 12:
                            shared["captured_pcm"].extend(ulaw_to_linear(value) for value in packet[12:])
                    if (peer_payload in shared["output"] and shared["peer_done"]
                            and shared["connected_at"] is not None
                            and time.monotonic() - shared["connected_at"] > 5.5):
                        raise KeyboardInterrupt
                    data = b""
                    if self.modem.fileno() in readable:
                        try:
                            data = self.modem.recv(4096)
                        except BlockingIOError:
                            pass
                    return data, False

                def close(self):
                    self.closed = True
                    self.modem.close()

            class FakeSipServer:
                def __init__(self, *args, **kwargs):
                    pass

                def start(self):
                    pass

                def close(self):
                    pass

            class FakeSipCall:
                def __init__(self, *args, **kwargs):
                    self.on_connected = kwargs["on_connected"]

                def dial(self, number):
                    shared["dialed_number"] = number
                    self.on_connected(peer_sock.getsockname())

                def hangup(self):
                    pass

            def transmit_independent_backward():
                try:
                    deadline = time.monotonic() + 4
                    while shared["session"] is None and time.monotonic() < deadline:
                        if stop_peer.wait(0.01):
                            return
                    if shared["session"] is None:
                        raise AssertionError("ATD did not create the V.23 caller media session")
                    sequence = timestamp = 0
                    for offset in range(0, len(peer_pcm), 160):
                        if stop_peer.is_set():
                            return
                        frame = peer_pcm[offset:offset + 160]
                        frame.extend([0] * (160 - len(frame)))
                        packet = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, 0x56323343)
                        packet += bytes(linear_to_ulaw(sample) for sample in frame)
                        peer_sock.sendto(packet, ("127.0.0.1", media_port))
                        sequence = (sequence + 1) & 0xFFFF
                        timestamp = (timestamp + 160) & 0xFFFFFFFF
                        if stop_peer.wait(0.02):
                            return
                    shared["peer_done"] = True
                except Exception as exc:
                    shared["peer_errors"].append(exc)

            serial = FakeSerial()
            peer_thread = threading.Thread(target=transmit_independent_backward, daemon=True)
            peer_thread.start()
            original_session = OnlineAudioSession

            def track_session(*args, **kwargs):
                session = original_session(*args, **kwargs)
                shared["session"] = session
                return session

            try:
                with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                        mock.patch("modem_dce.SipOriginateCall", FakeSipCall), \
                        mock.patch("modem_dce.OnlineAudioSession", side_effect=track_session):
                    run_pty(None, sip_host="127.0.0.1", media_host="127.0.0.1",
                            media_port=media_port, dial_host="127.0.0.1", line_mode="ms",
                            negotiate=False, serial_backend=serial)
            finally:
                stop_peer.set()
                peer_thread.join(timeout=2)
                peer_sock.close()
                serial.dte.close()

            self.assertFalse(peer_thread.is_alive())
            self.assertTrue(serial.closed)
            self.assertEqual(shared["peer_errors"], [])
            self.assertEqual(shared.get("dialed_number"), "1001")
            self.assertIn(b"CONNECT 1200", shared["output"])
            self.assertIn(peer_payload, shared["output"])
            self.assertEqual(shared["session"].link_stats()["v23_channels"], {
                "forward_baud": 1200, "tx_baud": 1200, "rx_baud": 75,
                "backward_channel": "enabled",
            })

            rx_path = Path(temp) / "production-forward.wav"
            with wave.open(str(rx_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(struct.pack(f"<{len(shared['captured_pcm'])}h",
                                            *shared["captured_pcm"]))
            decoded = subprocess.run(
                [shutil.which("minimodem"), "--rx", "--quiet", "-M", "1300", "-S", "2100",
                 "-R", "8000", "-f", str(rx_path), "1200"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
            self.assertEqual(decoded.stdout, dte_payload)

    @unittest.skipUnless(shutil.which("minimodem"), "optional minimodem is not installed")
    def test_run_pty_v23_answer_uses_independent_forward_and_backward_channels(self) -> None:
        """Exercise ATA and both production V.23 channels against minimodem waveforms."""
        peer_payload = b"independent V.23 forward payload\r\n"
        dte_payload = b"independent 75 baud return\r\n"
        shared = {"output": bytearray(), "session": None, "connected_at": None,
                  "peer_done": False, "peer_errors": [], "captured_pcm": []}
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-dce-interop-") as temp:
            tx_path = Path(temp) / "minimodem-forward.wav"
            generated = subprocess.run(
                [shutil.which("minimodem"), "--tx", "--quiet", "-M", "1300", "-S", "2100",
                 "-f", str(tx_path), "1200"], input=peer_payload,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr.decode(errors="replace"))
            with wave.open(str(tx_path), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (1, 2, 48000))
                raw = wav.readframes(wav.getnframes())
            independent_data_pcm = list(struct.unpack(f"<{len(raw) // 2}h", raw)[::6])
            # V.23 needs the 300..700 ms initial carrier qualification.
            # Supply idle mark before and after the independently modulated
            # minimodem payload, as a real answerer does.
            mark_samples = [round(9000 * math.sin(2 * math.pi * 1300 * n / 8000))
                            for n in range(8000)]
            peer_pcm = mark_samples + independent_data_pcm + mark_samples

            peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            peer_sock.bind(("127.0.0.1", 0))
            peer_sock.setblocking(False)
            reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            reservation.bind(("127.0.0.1", 0))
            media_port = reservation.getsockname()[1]
            reservation.close()
            stop_peer = threading.Event()

            class FakeSerial:
                def __init__(self):
                    self.modem, self.dte = socket.socketpair()
                    self.modem.setblocking(False)
                    self.name = "v23-minimodem-answer"
                    self.supports_modem_control = False
                    self.closed = False
                    self.dte_command_sent = False
                    self.started = time.monotonic()
                    self.dte.sendall(b"ATE0\\N0\rAT+MS=V23,0,1200,1200\r")

                def fds(self):
                    return [self.modem.fileno()]

                def write_fds(self):
                    return []

                def write(self, data):
                    shared["output"].extend(data)
                    if not self.dte_command_sent and b"RING" in shared["output"]:
                        self.dte_command_sent = True
                        self.dte.sendall(b"ATA\r")
                    if shared["connected_at"] is None and b"CONNECT 1200" in shared["output"]:
                        shared["connected_at"] = time.monotonic()
                        self.dte.sendall(dte_payload)

                def service(self, readable, writable=()):
                    if time.monotonic() - self.started > 12:
                        raise AssertionError(
                            f"V.23 DCE interop timed out: output={bytes(shared['output'][-160:])!r}, "
                            f"received={peer_payload in shared['output']}, "
                            f"captured={len(shared['captured_pcm'])} samples, "
                            f"session={getattr(shared.get('session'), 'link_stats', lambda: None)()}"
                        )
                    while True:
                        try:
                            packet, _ = peer_sock.recvfrom(2048)
                        except BlockingIOError:
                            break
                        if len(packet) >= 12:
                            shared["captured_pcm"].extend(ulaw_to_linear(value) for value in packet[12:])
                    if (peer_payload in shared["output"] and shared["peer_done"]
                            and shared["connected_at"] is not None
                            and time.monotonic() - shared["connected_at"] > 5.5):
                        raise KeyboardInterrupt
                    data = b""
                    if self.modem.fileno() in readable:
                        try:
                            data = self.modem.recv(4096)
                        except BlockingIOError:
                            pass
                    return data, False

                def close(self):
                    self.closed = True
                    self.modem.close()

            class FakeSipServer:
                def __init__(self, *args, **kwargs):
                    self.on_incoming = kwargs["on_incoming"]
                    self.on_ring = kwargs["on_ring"]
                    self.last_caller = {"number": "v23-minimodem", "technology": "SIP"}

                def start(self):
                    self.on_incoming()
                    self.on_ring()

                def answer(self):
                    return True

                def remote_rtp(self):
                    return peer_sock.getsockname()

                def hangup(self):
                    pass

                def close(self):
                    pass

            def transmit_independent_forward():
                try:
                    deadline = time.monotonic() + 4
                    while shared["session"] is None and time.monotonic() < deadline:
                        if stop_peer.wait(0.01):
                            return
                    if shared["session"] is None:
                        raise AssertionError("DCE did not create the V.23 media session")
                    sequence = timestamp = 0
                    for offset in range(0, len(peer_pcm), 160):
                        if stop_peer.is_set():
                            return
                        frame = peer_pcm[offset:offset + 160]
                        frame.extend([0] * (160 - len(frame)))
                        packet = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, 0x56323349)
                        packet += bytes(linear_to_ulaw(sample) for sample in frame)
                        peer_sock.sendto(packet, ("127.0.0.1", media_port))
                        sequence = (sequence + 1) & 0xFFFF
                        timestamp = (timestamp + 160) & 0xFFFFFFFF
                        if stop_peer.wait(0.02):
                            return
                    shared["peer_done"] = True
                except Exception as exc:
                    shared["peer_errors"].append(exc)

            serial = FakeSerial()
            peer_thread = threading.Thread(target=transmit_independent_forward, daemon=True)
            peer_thread.start()
            real_session = OnlineAudioSession

            def track_session(*args, **kwargs):
                session = real_session(*args, **kwargs)
                shared["session"] = session
                return session

            try:
                with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                        mock.patch("modem_dce.OnlineAudioSession", side_effect=track_session):
                    run_pty(None, sip_host="127.0.0.1", media_host="127.0.0.1",
                            media_port=media_port, line_mode="ms", negotiate=False,
                            serial_backend=serial)
            finally:
                stop_peer.set()
                peer_thread.join(timeout=2)
                peer_sock.close()
                serial.dte.close()

            self.assertFalse(peer_thread.is_alive())
            self.assertTrue(serial.closed)
            self.assertEqual(shared["peer_errors"], [])
            self.assertIn(b"CONNECT 1200", shared["output"])
            self.assertIn(peer_payload, shared["output"])
            self.assertEqual(shared["session"].link_stats()["v23_channels"], {
                "forward_baud": 1200, "tx_baud": 75, "rx_baud": 1200,
                "backward_channel": "enabled",
            })

            rx_path = Path(temp) / "production-backward.wav"
            with wave.open(str(rx_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(struct.pack(f"<{len(shared['captured_pcm'])}h",
                                            *shared["captured_pcm"]))
            decoded = subprocess.run(
                [shutil.which("minimodem"), "--rx", "--quiet", "-M", "390", "-S", "450",
                 "-R", "8000", "-f", str(rx_path), "75"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
            self.assertEqual(decoded.stdout, dte_payload)

    def test_run_pty_software_audio_loop_uses_dsp_in_both_configurations(self) -> None:
        for rate, role in ((1200, 0), (1200, 1), (600, 0), (600, 1)):
            with self.subTest(forward_baud=rate, role=role):
                self._exercise_run_pty_software_audio_loop(rate, role, "V23")

    def test_run_pty_software_audio_loop_supports_bell103_and_v21(self) -> None:
        for profile in ("B103", "V21"):
            with self.subTest(profile=profile):
                self._exercise_run_pty_software_audio_loop(300, 0, profile)

    def test_run_pty_software_audio_loop_supports_v22_v22bis_and_bell212a(self) -> None:
        for profile, rate in (("V22", 1200), ("V22B", 2400), ("B212", 1200)):
            with self.subTest(profile=profile):
                self._exercise_run_pty_software_audio_loop(rate, 0, profile)

    def test_run_pty_v22bis_software_loop_stops_and_restarts_cleanly(self) -> None:
        self._exercise_run_pty_software_audio_loop(2400, 0, "V22B", cycles=2)

    def test_run_pty_legacy_software_loops_stop_and_restart_cleanly(self) -> None:
        for profile, rate in (("B103", 300), ("V21", 300), ("V23", 1200)):
            with self.subTest(profile=profile):
                self._exercise_run_pty_software_audio_loop(
                    rate, 0, profile, cycles=2)

    def _exercise_run_pty_software_audio_loop(
        self, rate: int, role: int, profile: str = "V23", *, cycles: int = 1
    ) -> None:
        payload = f"{profile}-software-loop".encode("ascii")
        connect_rate = (rate if profile in ("V23", "V22B")
                        else 1200 if profile in ("V22", "B212") else 300)
        role_setting = f"AT+V23DIR={role}\r" if profile == "V23" else ""

        class FakeSerial:
            def __init__(self):
                self.modem, self.dte = socket.socketpair()
                self.modem.setblocking(False)
                self.name = "v23-software-audio-loop"
                self.supports_modem_control = False
                self.closed = False
                self.output = bytearray()
                self.connects_sent = 0
                self.payload_echoes = 0
                self.restarted = False
                self.stop_requested = False
                self.stop_completed = False
                self.escape_at = None
                self.escape_sent = 0
                self.escape_handled = 0
                self.started = time.monotonic()
                self.dte.sendall(
                    f"ATE0\\N0\rAT+MS={profile},0,300,{rate}\r"
                    f"{role_setting}AT&T1\r".encode("ascii"))

            def fds(self):
                return [self.modem.fileno()]

            def write_fds(self):
                return []

            def write(self, data):
                self.output.extend(data)
                connect_marker = f"CONNECT {connect_rate}/SW-LOOP".encode()
                while (self.connects_sent < cycles
                       and self.output.count(connect_marker) > self.connects_sent):
                    self.connects_sent += 1
                    self.dte.sendall(payload)
                observed_echoes = self.output.count(payload)
                if cycles > 1 and observed_echoes > self.payload_echoes:
                    self.escape_at = time.monotonic() + 1.05
                self.payload_echoes = observed_echoes
                if (cycles > 1 and self.escape_sent > self.escape_handled
                        and b"\r\nOK\r\n" in data):
                    self.escape_handled += 1
                    if self.escape_handled == 1:
                        self.restarted = True
                        self.dte.sendall(b"AT&T0\rAT&T1\r")
                    elif self.escape_handled == 2:
                        self.stop_requested = True
                        self.dte.sendall(b"AT&T0\r")
                elif self.stop_requested and b"\r\nOK\r\n" in data:
                    self.stop_completed = True

            def service(self, readable, writable=()):
                if ((cycles == 1 and payload in self.output and self.connects_sent)
                        or self.stop_completed):
                    raise KeyboardInterrupt
                if cycles > 1:
                    if time.monotonic() - self.started > 20:
                        raise TimeoutError("software audio-loop stop/restart timed out")
                    if self.escape_at is not None and time.monotonic() >= self.escape_at:
                        self.escape_at = None
                        self.escape_sent += 1
                        self.dte.sendall(b"+++")
                data = b""
                if self.modem.fileno() in readable:
                    try:
                        data = self.modem.recv(4096)
                    except BlockingIOError:
                        pass
                return data, False

            def close(self):
                self.closed = True
                self.modem.close()

        serial = FakeSerial()
        import modem_dce
        real_socket = modem_dce.socket.socket
        rtp_sends = []
        closed_loop_profiles = []

        class SocketSpy:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            def sendto(self, data, address):
                rtp_sends.append((len(data), address))
                return self._wrapped.sendto(data, address)

        def socket_factory(*args, **kwargs):
            return SocketSpy(real_socket(*args, **kwargs))

        with tempfile.TemporaryDirectory(prefix="elmodem-v23-loop-") as profile_dir:
            profile_file = Path(profile_dir) / "profile.json"
            profile_file.write_text("{\"profiles\": {}}", encoding="utf-8")
            try:
                if profile in ("V22", "V22B", "B212"):
                    import v22_audio_loop
                    original_close = v22_audio_loop.V22SoftwareAudioLoop.close

                    def track_close(loop):
                        closed_loop_profiles.append(loop.profile)
                        original_close(loop)

                    close_patch = mock.patch.object(
                        v22_audio_loop.V22SoftwareAudioLoop, "close", track_close)
                else:
                    close_patch = nullcontext()
                with close_patch:
                    with mock.patch("modem_dce.socket.socket", side_effect=socket_factory):
                        run_pty(None, media_host="127.0.0.1", media_port=0,
                                line_mode="ms", negotiate=False,
                                profile_file=str(profile_file), serial_backend=serial)
            finally:
                serial.dte.close()
        output = bytes(serial.output)
        self.assertTrue(serial.closed)
        self.assertEqual(rtp_sends, [], "software loop leaked audio to RTP")
        self.assertIn(f"CONNECT {connect_rate}/SW-LOOP".encode(), output)
        self.assertIn(payload, output)
        if cycles > 1:
            self.assertEqual(output.count(f"CONNECT {connect_rate}/SW-LOOP".encode()), cycles)
            self.assertEqual(output.count(payload), cycles)
            self.assertIn(b"\r\nOK\r\n", output)
            self.assertTrue(serial.restarted)
            self.assertTrue(serial.stop_completed)
        if profile in ("V22", "V22B", "B212"):
            expected_profile = {"V22": "v22", "V22B": "v22bis",
                                "B212": "bell212a"}[profile]
            self.assertEqual(closed_loop_profiles, [expected_profile] * cycles)

    @unittest.skipUnless(
        shutil.which("minimodem"), "minimodem independent FSK reference not installed"
    )
    def test_v23_production_rtp_audio_interoperates_with_minimodem(self) -> None:
        """Check the production PCMU/RTP boundary against an independent FSK engine."""
        minimodem = shutil.which("minimodem")
        self.assertIsNotNone(minimodem)
        payload = b"independent minimodem through production RTP\r\n"

        with tempfile.TemporaryDirectory(prefix="elmodem-v23-rtp-") as directory:
            directory = Path(directory)
            inbound_wav = directory / "minimodem-tx.wav"
            tx = subprocess.run(
                [minimodem, "--tx", "--quiet", "-M", "1300", "-S", "2100",
                 "-f", str(inbound_wav), "1200"],
                input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(tx.returncode, 0, tx.stderr.decode(errors="replace"))

            # Feed the independent 48-kHz reference waveform through the
            # production RTP/PCMU receive path. The V.23 tones are below the
            # 8-kHz Nyquist limit; this is the same decimation used by the
            # existing independent raw-engine vectors.
            with wave.open(str(inbound_wav), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (1, 2, 48000))
                raw = wav.readframes(wav.getnframes())
            reference_48k = struct.unpack(f"<{len(raw) // 2}h", raw)
            reference_8k = list(reference_48k[::6])

            rtp_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rtp_rx.bind(("127.0.0.1", 0))
            rtp_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rtp_tx.bind(("127.0.0.1", 0))
            answer = OnlineAudioSession(
                rtp_tx, rtp_rx.getsockname(), "answer", "originate",
                line_mode="v23", v23_rate=1200,
            )
            received = bytearray()
            sequence = timestamp = 0
            ssrc = 0x56323352
            try:
                for offset in range(0, len(reference_8k), 160):
                    frame = reference_8k[offset:offset + 160]
                    frame.extend([0] * (160 - len(frame)))
                    packet = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, ssrc)
                    packet += bytes(linear_to_ulaw(sample) for sample in frame)
                    received.extend(answer.handle_rtp_datagram(
                        ("127.0.0.1", 49152), packet,
                        received_at=time.monotonic(),
                    ))
                    sequence = (sequence + 1) & 0xFFFF
                    timestamp = (timestamp + 160) & 0xFFFFFFFF
                self.assertEqual(bytes(received), payload)
                self.assertEqual(answer.link_stats()["v23_channels"]["rx_baud"], 1200)
            finally:
                answer.stop()
                rtp_rx.close()
                rtp_tx.close()

            # In the opposite direction capture the production session's
            # actual RTP/PCMU transmit packets, restore linear PCM, and let
            # minimodem independently decode the resulting 8-kHz waveform.
            sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sink.bind(("127.0.0.1", 0))
            sink.settimeout(0.1)
            session_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            session_sock.bind(("127.0.0.1", 0))
            originate = OnlineAudioSession(
                session_sock, sink.getsockname(), "originate", "answer",
                line_mode="v23", v23_rate=1200,
            )
            originate.feed_terminal_bytes(payload)
            transmitted_pcm = []
            try:
                originate.start()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    try:
                        packet, _ = sink.recvfrom(2048)
                    except socket.timeout:
                        continue
                    self.assertEqual(packet[1] & 0x7F, 0)
                    transmitted_pcm.extend(ulaw_to_linear(value) for value in packet[12:])
            finally:
                originate.stop()
                sink.close()
                session_sock.close()

            outbound_wav = directory / "production-rtp-tx.wav"
            with wave.open(str(outbound_wav), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(struct.pack(f"<{len(transmitted_pcm)}h", *transmitted_pcm))
            rx = subprocess.run(
                [minimodem, "--rx", "--quiet", "-M", "1300", "-S", "2100",
                 "-R", "8000", "-f", str(outbound_wav), "1200"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(rx.returncode, 0, rx.stderr.decode(errors="replace"))
            self.assertEqual(rx.stdout, payload)

    def test_v23_rtp_sessions_exchange_asymmetric_payloads_in_both_modes(self) -> None:
        for rate in (600, 1200):
            with self.subTest(forward_baud=rate):
                o_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                o_sock.bind(("127.0.0.1", 0))
                a_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                a_sock.bind(("127.0.0.1", 0))
                o_sock.settimeout(0.02)
                a_sock.settimeout(0.02)
                originate = OnlineAudioSession(
                    o_sock, a_sock.getsockname(), "originate", "answer",
                    line_mode="v23", v23_rate=rate,
                )
                answer = OnlineAudioSession(
                    a_sock, o_sock.getsockname(), "answer", "originate",
                    line_mode="v23", v23_rate=rate,
                )
                originate.start()
                answer.start()
                try:
                    originate_rx, answer_rx = bytearray(), bytearray()
                    payload_o = bytes(range(80))
                    payload_a = b"return at 75 baud"
                    sent = False
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        originate.poll(time.monotonic())
                        answer.poll(time.monotonic())
                        if originate.negotiation_complete and answer.negotiation_complete and not sent:
                            self.assertEqual((originate.line_speed, answer.line_speed), (rate, rate))
                            self.assertEqual((originate.modulation_name, answer.modulation_name), ("V.23", "V.23"))
                            self.assertEqual(originate.link_stats()["v23_channels"], {
                                "forward_baud": rate, "tx_baud": rate,
                                "rx_baud": 75, "backward_channel": "enabled",
                            })
                            self.assertEqual(answer.link_stats()["v23_channels"], {
                                "forward_baud": rate, "tx_baud": 75,
                                "rx_baud": rate, "backward_channel": "enabled",
                            })
                            originate.feed_terminal_bytes(payload_o)
                            answer.feed_terminal_bytes(payload_a)
                            sent = True
                        for session, sock, received in (
                            (answer, a_sock, answer_rx), (originate, o_sock, originate_rx),
                        ):
                            try:
                                packet, _ = sock.recvfrom(2048)
                            except socket.timeout:
                                continue
                            received.extend(session.handle_rtp_packet(packet))
                        if sent and answer_rx == payload_o and originate_rx == payload_a:
                            break
                    self.assertTrue(sent, "V.23 RTP pair did not detect the continuous mark carriers")
                    self.assertEqual(bytes(answer_rx), payload_o)
                    self.assertEqual(bytes(originate_rx), payload_a)
                finally:
                    originate.stop()
                    answer.stop()
                    o_sock.close()
                    a_sock.close()

    def test_v23_run_pty_hayes_call_carries_forward_and_return_payloads(self) -> None:
        """Exercise +MS selection and asymmetric V.23 data via production run_pty."""
        for rate, backward_enabled, dce_role in (
            (600, True, "answer"), (1200, True, "answer"),
            (1200, False, "answer"), (600, True, "originate"),
            (1200, True, "originate"),
        ):
            with self.subTest(forward_baud=rate, backward_enabled=backward_enabled,
                              dce_role=dce_role):
                shared = {
                    "sip_hungup": False,
                    "session": None,
                    "dce_connect_at": None,
                    "peer_payload_sent": False,
                    "peer_payload_received": bytearray(),
                    "forward_received_at": None,
                    "peer_error": [],
                }
                peer_payload = b"forward V.23 payload at mode %d\r\n" % rate
                dte_payload = b"return-75"
                peer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                peer_sock.bind(("127.0.0.1", 0))
                peer_sock.settimeout(0.01)
                media_reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                media_reservation.bind(("127.0.0.1", 0))
                media_port = media_reservation.getsockname()[1]
                media_reservation.close()

                peer_tx_role = "originate" if dce_role == "answer" else "answer"
                peer_rx_role = "answer" if dce_role == "answer" else "originate"
                peer = OnlineAudioSession(
                    peer_sock, ("127.0.0.1", media_port), peer_tx_role, peer_rx_role,
                    line_mode="v23", v23_rate=rate,
                )

                class FakeSerial:
                    def __init__(self):
                        self.modem, self.dte = socket.socketpair()
                        self.modem.setblocking(False)
                        self.name = f"v23-{rate}-socketpair"
                        self.supports_modem_control = False
                        self.closed = False
                        self.output = bytearray()
                        self.payload_sent = False
                        self.started = time.monotonic()
                        command = "ATA" if dce_role == "answer" else "ATD1001"
                        self.dte.sendall(
                            f"ATE0\\N0\rAT+MS=V23,0,300,{rate}\r"
                            f"AT+V23BC={int(backward_enabled)}\r{command}\r".encode("ascii")
                        )

                    def fds(self):
                        return [self.modem.fileno()]

                    def write_fds(self):
                        return []

                    def write(self, data):
                        self.output.extend(data)
                        shared["output"] = bytes(self.output)
                        if not self.payload_sent and b"CONNECT " in self.output:
                            self.payload_sent = True
                            shared["dce_connect_at"] = time.monotonic()
                            self.dte.sendall(dte_payload)

                    def service(self, readable, writable=()):
                        if shared["sip_hungup"]:
                            raise KeyboardInterrupt
                        if time.monotonic() - self.started > 8:
                            raise AssertionError(
                                f"V.23 run_pty call timed out: output={bytes(self.output[-256:])!r}, "
                                f"peer_up={peer._v22.connected}, "
                                f"peer_tx={shared['peer_payload_sent']}, "
                                f"peer_rx={bytes(shared['peer_payload_received'])!r}, "
                                f"session={getattr(shared.get('session'), 'link_stats', lambda: None)()}"
                            )
                        if peer_payload in self.output:
                            if shared["forward_received_at"] is None:
                                shared["forward_received_at"] = time.monotonic()
                            if backward_enabled:
                                if bytes(shared["peer_payload_received"]) == dte_payload:
                                    raise KeyboardInterrupt
                            elif time.monotonic() - shared["forward_received_at"] >= 1.5:
                                raise KeyboardInterrupt
                        data = b""
                        if self.modem.fileno() in readable:
                            try:
                                data = self.modem.recv(4096)
                            except BlockingIOError:
                                pass
                        return data, False

                    def close(self):
                        self.closed = True
                        self.modem.close()

                class FakeSipServer:
                    def __init__(self, *args, **kwargs):
                        self.on_incoming = kwargs["on_incoming"]
                        self.on_ring = kwargs["on_ring"]
                        self.on_hangup = kwargs["on_hangup"]
                        self.hungup = False
                        self.last_caller = {"number": "v23-test", "technology": "SIP"}

                    def start(self):
                        if dce_role == "answer":
                            self.on_incoming()
                            self.on_ring()

                    def answer(self):
                        return True

                    def remote_rtp(self):
                        return peer_sock.getsockname()

                    def hangup(self):
                        if not self.hungup:
                            self.hungup = True
                            shared["sip_hungup"] = True

                    def close(self):
                        pass

                class FakeSipCall:
                    def __init__(self, *args, **kwargs):
                        self.on_connected = kwargs["on_connected"]
                        self.hungup = False

                    def dial(self, number):
                        shared["dialed_number"] = number
                        self.on_connected(peer_sock.getsockname())

                    def hangup(self):
                        if not self.hungup:
                            self.hungup = True
                            shared["sip_hungup"] = True

                    def close(self):
                        pass

                peer.start()
                stop_peer = threading.Event()

                def pump_peer():
                    try:
                        while not stop_peer.is_set():
                            peer.poll(time.monotonic())
                            try:
                                packet, source = peer_sock.recvfrom(2048)
                            except socket.timeout:
                                continue
                            shared["peer_payload_received"].extend(
                                peer.handle_rtp_datagram(source, packet)
                            )
                            connect_at = shared["dce_connect_at"]
                            if (peer.negotiation_complete and connect_at is not None
                                    and time.monotonic() - connect_at >= 0.02
                                    and not shared["peer_payload_sent"]):
                                peer.feed_terminal_bytes(peer_payload)
                                shared["peer_payload_sent"] = True
                    except Exception as exc:
                        shared["peer_error"].append(exc)

                peer_thread = threading.Thread(target=pump_peer, daemon=True)
                peer_thread.start()
                serial = FakeSerial()
                original_session = OnlineAudioSession

                def track_session(*args, **kwargs):
                    session = original_session(*args, **kwargs)
                    shared["session"] = session
                    return session

                try:
                    with mock.patch("modem_dce.SipIncomingCallServer", FakeSipServer), \
                            mock.patch("modem_dce.SipOriginateCall", FakeSipCall), \
                            mock.patch("modem_dce.OnlineAudioSession", side_effect=track_session):
                        run_pty(
                            None, sip_host="127.0.0.1", media_host="127.0.0.1",
                            media_port=media_port, dial_host="127.0.0.1",
                            line_mode="ms", negotiate=False,
                            serial_backend=serial,
                        )
                finally:
                    stop_peer.set()
                    peer.stop()
                    peer_thread.join(timeout=2)
                    peer_sock.close()
                    serial.dte.close()

                output = bytes(shared.get("output", b""))
                self.assertFalse(peer_thread.is_alive())
                self.assertEqual(shared["peer_error"], [])
                self.assertTrue(serial.closed)
                self.assertTrue(shared["peer_payload_sent"])
                self.assertEqual(bytes(shared["peer_payload_received"]),
                                 dte_payload if backward_enabled else b"")
                self.assertIn(f"CONNECT {rate}".encode(), output)
                self.assertIn(peer_payload, output)
                self.assertIsNotNone(shared["session"])
                self.assertEqual(shared["session"].modulation_name, "V.23")
                self.assertEqual(shared["session"].link_stats()["v23_channels"], {
                    "forward_baud": rate,
                    "tx_baud": 75 if dce_role == "answer" else rate,
                    "rx_baud": rate if dce_role == "answer" else 75,
                    "backward_channel": "enabled" if backward_enabled else "disabled",
                })


class GapConcealmentTests(unittest.TestCase):
    """What stands in for lost RTP packets (see OnlineAudioSession._gap_fill)."""

    @staticmethod
    def _packet(seq, ts, samples):
        header = struct.pack("!BBHII", 0x80, 0, seq, ts, 1234)
        return header + bytes(linear_to_ulaw(v) for v in samples)

    def _session(self, **kw):
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        session = OnlineAudioSession(sock, ("127.0.0.1", 9), "originate", "answer", **kw)
        chunks = []
        session._receive_pcm = lambda pcm: chunks.append(pcm) or b""
        return session, chunks

    def test_automode_fallback_drains_held_audio_before_new_audio(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        for deadline in (False, True):
            for origin in (0, 0xFFFFFF60):
                with self.subTest(deadline=deadline, origin=origin), patch.dict(os.environ):
                    os.environ.pop('EL_MODEM_RTP_JITTER_MS', None)
                    os.environ['EL_MODEM_RTP_DEADLINE_DRAIN'] = '1' if deadline else '0'
                    session, chunks = self._session(line_mode='chain')
                    session._v22 = SimpleNamespace(kind='v32')
                    arrivals = []
                    session._receive_pcm = lambda pcm: arrivals.append(session._rx_arrival) or chunks.append(pcm) or b'D'
                    with patch('modem_dce.time.monotonic', return_value=10.):
                        for frame in (0, 1):
                            self.assertEqual(session.handle_rtp_packet(self._packet(frame, (origin + frame * 160) & 0xFFFFFFFF,
                                [1000 + frame * 100] * 160), received_at=10. + frame * .02), b'')
                        session._v22.kind = 'v22'
                        out = session.handle_rtp_packet(self._packet(2, (origin + 320) & 0xFFFFFFFF, [1200] * 160), received_at=10.04)
                    self.assertEqual(out, b'DDD')
                    self.assertEqual(arrivals, [10., 10.02, 10.04])
                    self.assertEqual(len(chunks), 3)
                    for frame, pcm in enumerate(chunks):
                        expected = ulaw_to_linear(linear_to_ulaw(1000 + frame * 100))
                        self.assertEqual(pcm, struct.pack('<160h', *([expected] * 160)))
                    self.assertFalse(session._rtp_buffer)
                    self.assertFalse(session._rtp_arrivals)
                    self.assertFalse(session._rtp_trace_context)
                    self.assertEqual(session._rtp_counts['concealed_samples'], 0)
                    self.assertEqual(session._rtp_counts['decoded_packets'], 3)

    def test_automode_hold_changes_preserve_reordering_and_existing_clock(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        for deadline in (False, True):
            with self.subTest(deadline=deadline), patch.dict(os.environ):
                os.environ.pop('EL_MODEM_RTP_JITTER_MS', None)
                os.environ['EL_MODEM_RTP_DEADLINE_DRAIN'] = '1' if deadline else '0'
                session, chunks = self._session(line_mode='chain')
                session._v22 = SimpleNamespace(kind='v8')
                def deliver(frame):
                    return session.handle_rtp_packet(self._packet(frame, frame * 160, [1000 + frame * 100] * 160), received_at=10.)
                with patch('modem_dce.time.monotonic', return_value=10.):
                    deliver(0)  # initial V8 with zero hold has already established the PCM clock
                    session._v22.kind = 'v34'
                    for frame in (2, 1, 3, 4):
                        deliver(frame)
                    session._v22.kind = 'v22'
                    deliver(5)
                self.assertEqual(len(chunks), 6)
                for frame, pcm in enumerate(chunks):
                    expected = ulaw_to_linear(linear_to_ulaw(1000 + frame * 100))
                    self.assertEqual(pcm, struct.pack('<160h', *([expected] * 160)))
                self.assertFalse(session._rtp_buffer)
                self.assertEqual(session._last_rx_timestamp, 800)
                self.assertEqual(session._rtp_counts['concealed_samples'], 0)

    def test_deadline_drain_releases_real_audio_during_sender_pause(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'EL_MODEM_RTP_DEADLINE_DRAIN':'1'}):
            session,chunks=self._session()
        session._rtp_jitter_samples=320
        arrivals=[]
        session._receive_pcm=lambda pcm: arrivals.append(session._rx_arrival) or chunks.append(pcm) or b'D'
        with patch('modem_dce.time.monotonic',return_value=10.):
            session.handle_rtp_packet(self._packet(0,0,[1000]*160),received_at=10.)
        with patch('modem_dce.time.monotonic',return_value=10.02):
            session.handle_rtp_packet(self._packet(1,160,[1100]*160),received_at=10.02)
        self.assertEqual(session.drain_rtp_deadline(10.039),b'')
        self.assertAlmostEqual(session.rtp_deadline_delay(10.02),.02)
        self.assertEqual(session.drain_rtp_deadline(10.041),b'D')
        self.assertEqual(session.drain_rtp_deadline(10.061),b'D')
        self.assertEqual(arrivals,[10.,10.02])
        self.assertEqual(len(chunks),2)
        self.assertEqual(session._last_rx_timestamp,160)
        self.assertEqual(session._rtp_counts['concealed_samples'],0)
        self.assertEqual(session._rtp_counts['deadline_released_packets'],2)
        self.assertIsNone(session.rtp_deadline_delay(10.6))
        self.assertEqual(session.drain_rtp_deadline(10.6),b'')

    def test_deadline_preserves_wrap_reordering_and_duplicate_deadlines(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'EL_MODEM_RTP_DEADLINE_DRAIN':'1'}):
            session,chunks=self._session()
        session._rtp_jitter_samples=320
        with patch('modem_dce.time.monotonic',return_value=10.02):
            session.handle_rtp_packet(self._packet(1,0,[1100]*160),received_at=10.)
            session.handle_rtp_packet(self._packet(0,0xFFFFFF60,[1000]*160),received_at=10.02)
            session.handle_rtp_packet(self._packet(0,0xFFFFFF60,[9999]*160),received_at=10.03)
        self.assertEqual(session._rtp_arrivals[0xFFFFFF60],10.02)
        self.assertEqual(session.drain_rtp_deadline(10.041),b'')
        session.drain_rtp_deadline(10.061)
        self.assertEqual(len(chunks),2)
        self.assertEqual(session._last_rx_timestamp,0)
        expected=ulaw_to_linear(linear_to_ulaw(1000))
        self.assertEqual(chunks[0],struct.pack('<160h',*([expected]*160)))
        self.assertEqual(session._rtp_counts['concealed_samples'],0)
        with patch('modem_dce.time.monotonic',return_value=10.1):
            session.handle_rtp_packet(self._packet(0,0xFFFFFF60,[1000]*160),received_at=10.1)
        self.assertEqual(len(chunks),2)
        self.assertEqual(session._rtp_counts['late_or_duplicate_packets'],2)

    def test_deadline_drain_is_disabled_by_default(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'EL_MODEM_RTP_DEADLINE_DRAIN':'0'}):
            session,chunks=self._session()
        session._rtp_jitter_samples=320
        session.handle_rtp_packet(self._packet(0,0,[1000]*160),received_at=10.)
        self.assertIsNone(session.rtp_deadline_delay(100.))
        self.assertEqual(session.drain_rtp_deadline(100.),b'')
        self.assertEqual(chunks,[])

    def test_jitter_window_orders_packets_without_manufacturing_loss(self):
        session, chunks = self._session()
        session._rtp_jitter_samples = 320
        for frame in (0, 2, 1, 4, 3, 5):
            session.handle_rtp_packet(self._packet(frame, frame * 160, [1000 + frame * 100] * 160))
        self.assertEqual(len(chunks), 4)
        for frame, pcm in enumerate(chunks):
            expected = ulaw_to_linear(linear_to_ulaw(1000 + frame * 100))
            self.assertEqual(pcm, struct.pack('<160h', *([expected] * 160)))
        self.assertEqual(session._last_rx_timestamp, 480)
        self.assertEqual(len(session._rtp_buffer), 2)

    def test_jitter_window_conceals_a_gap_after_its_deadline(self):
        session, chunks = self._session()
        session._rtp_jitter_samples = 320
        for frame in (0, 1, 3, 4, 5):
            session.handle_rtp_packet(self._packet(frame, frame * 160, [1000] * 160))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[2], bytes(320))
        self.assertEqual(session._last_rx_timestamp, 480)

    def test_transport_discontinuity_hooks_cover_late_buffered_duplicate_and_source_reset(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        marks=[]
        session,chunks=self._session()
        session.receiver=SimpleNamespace(note_transport_discontinuity_rx=lambda:marks.append(True))
        for frame in (0,2,1,2):session.handle_rtp_packet(self._packet(frame,frame*160,[1000]*160))
        self.assertEqual(len(marks),2)
        packet=self._packet(0,0,[1000]*160)
        session.handle_rtp_packet(packet[:8]+struct.pack('!I',4321)+packet[12:])
        self.assertEqual(len(marks),3)
        with patch.dict(os.environ,{'EL_MODEM_RTP_DEADLINE_DRAIN':'1'}):
            pending,_=self._session()
        pending._rtp_jitter_samples=320
        pending.receiver=SimpleNamespace(note_transport_discontinuity_rx=lambda:marks.append(True))
        with patch('modem_dce.time.monotonic',return_value=10.):
            pending.handle_rtp_packet(packet,received_at=10.)
            pending.handle_rtp_packet(packet,received_at=10.)
        self.assertEqual(len(marks),4)

    def test_late_and_duplicate_packets_never_rewind_the_receiver(self):
        session, chunks = self._session()
        for frame in (0, 2, 1, 2, 3):
            session.handle_rtp_packet(self._packet(frame, frame * 160, [1000] * 160))
        self.assertEqual(len(chunks), 4)  # three real frames and one gap
        self.assertEqual(session._last_rx_timestamp, 480)

    def test_jitter_window_handles_timestamp_wrap_and_initial_reordering(self):
        session, chunks = self._session()
        session._rtp_jitter_samples = 320
        for frame in (1, 0, 3, 2, 4):
            ts = (0xFFFFFF60 + frame * 160) & 0xFFFFFFFF
            session.handle_rtp_packet(self._packet(frame, ts, [1000 + frame * 100] * 160))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(session._last_rx_timestamp, 160)
        for frame, pcm in enumerate(chunks):
            expected = ulaw_to_linear(linear_to_ulaw(1000 + frame * 100))
            self.assertEqual(pcm, struct.pack('<160h', *([expected] * 160)))

    def test_new_rtp_source_resets_timestamp_and_discards_buffered_old_audio(self):
        session, chunks = self._session()
        session._rtp_jitter_samples = 320
        for frame in (0, 1, 2):
            session.handle_rtp_packet(self._packet(frame, 10000 + frame * 160, [1000] * 160))
        self.assertEqual(len(chunks), 1)
        for frame in (0, 1, 2):
            packet = self._packet(frame, frame * 160, [2000] * 160)
            session.handle_rtp_packet(packet[:8] + struct.pack('!I', 4321) + packet[12:])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(session._last_rx_timestamp, 0)
        expected = ulaw_to_linear(linear_to_ulaw(2000))
        self.assertEqual(chunks[-1], struct.pack('<160h', *([expected] * 160)))

    def _tone(self, amp=6000):
        return [round(amp * math.sin(2 * math.pi * 2400 * i / 8000)) for i in range(160)]


    def test_bytes_decoded_from_a_filled_gap_are_not_discarded(self) -> None:
        # With concealment the modem keeps decoding through a lost-packet gap,
        # so an error-controlled frame can complete inside it. Its payload was
        # once thrown away (a whole 32-byte LAPM frame vanished, acked but
        # never delivered). Whatever the fill decodes must be returned.
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        session = OnlineAudioSession(sock, ("127.0.0.1", 9), "originate", "answer")
        calls = []

        def fake_receive(pcm):
            calls.append(len(pcm) // 2)
            # the 320-sample gap fill "decodes" a frame; the real packets nothing
            return b"FROM-THE-GAP" if len(pcm) // 2 == 320 else b""

        session._receive_pcm = fake_receive
        tone = [round(6000 * math.sin(2 * math.pi * 1200 * i / 8000)) for i in range(160)]
        session.handle_rtp_packet(self._packet(0, 0, tone))
        out = session.handle_rtp_packet(self._packet(3, 480, tone))  # 2 frames lost
        self.assertEqual(calls, [160, 320, 160])
        self.assertEqual(out, b"FROM-THE-GAP")

    def test_bell103_keeps_silence_fill(self) -> None:
        session, chunks = self._session()
        session.handle_rtp_packet(self._packet(0, 0, self._tone()))
        session.handle_rtp_packet(self._packet(3, 480, self._tone()))
        fill = chunks[1]  # chunks: frame A, gap fill, frame B
        self.assertEqual(len(fill), 320 * 2)
        self.assertEqual(fill, b"\x00\x00" * 320)

    @unittest.skipUnless(_spandsp_available(), "libspandsp not installed")
    def test_v22_repeats_the_last_frame_over_a_short_gap(self) -> None:
        session, chunks = self._session(line_mode="v22bis")
        a = self._tone()
        session.handle_rtp_packet(self._packet(0, 0, a))
        session.handle_rtp_packet(self._packet(3, 480, self._tone(3000)))
        frame_a = chunks[0]
        fill = chunks[1]
        self.assertEqual(len(fill), 320 * 2)
        self.assertEqual(fill, frame_a * 2)  # two lost frames -> frame A twice

    @unittest.skipUnless(_spandsp_available(), "libspandsp not installed")
    def test_v22_long_gap_falls_back_to_silence(self) -> None:
        session, chunks = self._session(line_mode="v22bis")
        session.handle_rtp_packet(self._packet(0, 0, self._tone()))
        # 8 lost frames (160 ms) is beyond the 5-frame concealment limit.
        session.handle_rtp_packet(self._packet(9, 160 * 9, self._tone()))
        self.assertEqual(chunks[1], b"\x00\x00" * (160 * 8))


class GcPauseTests(unittest.TestCase):
    def session(self, **env):
        import gc
        import socket
        from unittest.mock import patch
        was = gc.isenabled()
        gc.enable()
        self.addCleanup(lambda: gc.enable() if was else gc.disable())
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        with patch.dict("os.environ", env):
            s = OnlineAudioSession(sock, ("127.0.0.1", 9), "originate", "answer")
            s._gc_pause_begin()
        self.addCleanup(s._gc_pause_end)
        return s

    def test_off_by_default(self):
        import gc
        s = self.session()
        self.assertTrue(gc.isenabled())
        self.assertNotIn("gc_pause", s.link_stats())

    def test_collection_is_paused_for_the_call_and_resumed_at_the_end(self):
        import gc
        s = self.session(EL_MODEM_GC_PAUSE="1")
        self.assertFalse(gc.isenabled())
        s._gc_pause_end()
        self.assertTrue(gc.isenabled())
        state = s._gc_pause
        self.assertTrue(state["was_enabled"] and state["resumed"])
        self.assertGreaterEqual(state["paused_seconds"], 0)
        self.assertIn("collected_at_end", state)

    def test_resume_is_idempotent_and_does_not_enable_a_collector_that_was_off(self):
        import gc
        import socket
        from unittest.mock import patch
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        with patch.dict("os.environ", {"EL_MODEM_GC_PAUSE": "1"}):
            s = OnlineAudioSession(sock, ("127.0.0.1", 9), "originate", "answer")
            gc.disable()
            s._gc_pause_begin()
        s._gc_pause_end()
        s._gc_pause_end()
        self.assertFalse(gc.isenabled())
        self.assertFalse(s._gc_pause["was_enabled"])

    def test_stop_resumes_collection(self):
        import gc
        s = self.session(EL_MODEM_GC_PAUSE="1")
        self.assertFalse(gc.isenabled())
        s.stop()
        self.assertTrue(gc.isenabled())


class RtpJitterByLineModeTests(unittest.TestCase):
    """The automode RTP hold follows the engine the chain selects (V.22: none; V.32/V.34: 40 ms)."""

    def _session(self, line_mode, env=None):
        import socket
        from unittest.mock import patch

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        with patch.dict(os.environ, env or {}, clear=False):
            if not env:
                os.environ.pop("EL_MODEM_RTP_JITTER_MS", None)
            return OnlineAudioSession(sock, ("127.0.0.1", 9), "originate", "answer", line_mode=line_mode)

    def test_automode_holds_nothing_until_the_chain_picks_v32_or_v34(self):
        from types import SimpleNamespace
        s = self._session("chain")
        self.assertEqual(s._rtp_jitter_samples, 0)                       # V.8 phase
        s._v22 = SimpleNamespace(kind="v22")
        self.assertEqual(s._rtp_jitter_samples, 0)                       # V.22/V.22bis: nist-12, nist-13
        s._v22 = SimpleNamespace(kind="v34")
        self.assertEqual(s._rtp_jitter_samples, 2 * 160)                 # V.34 keeps 40 ms
        s._v22 = SimpleNamespace(kind="v32")
        self.assertEqual(s._rtp_jitter_samples, 2 * 160)

    def test_plain_v34_line_mode_still_holds_40_ms(self):
        self.assertEqual(self._session("v34")._rtp_jitter_samples, 2 * 160)

    def test_other_line_modes_hold_nothing(self):
        self.assertEqual(self._session("v22bis")._rtp_jitter_samples, 0)

    def test_environment_override_wins_in_every_mode(self):
        s = self._session("chain", env={"EL_MODEM_RTP_JITTER_MS": "40"})
        self.assertEqual(s._rtp_jitter_samples, 320)
        s._v22 = type("K", (), {"kind": "v22"})()
        self.assertEqual(s._rtp_jitter_samples, 320)

    def test_assignment_still_sets_the_hold(self):
        s = self._session("chain")
        s._rtp_jitter_samples = 320
        self.assertEqual(s._rtp_jitter_samples, 320)


if __name__ == "__main__":
    unittest.main()
