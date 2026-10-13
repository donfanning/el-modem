#!/usr/bin/env python3
"""Unit and Linux PTY smoke tests for the lab Hayes DCE."""

import os
import math
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
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
    default_error_control_for,
    EscapeSequenceDetector,
    OnlineAudioSession,
    linear_to_ulaw,
    normalize_dial_number,
    ulaw_to_linear,
)
from call_progress import (
    BusyToneDetector,
    OutgoingCallProgressMonitor,
    RingbackToneDetector,
    VoiceActivityDetector,
)
from lapm_lite import ArqSession

SCRIPT = Path(__file__).with_name("modem_dce.py")


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


class AtProfileTests(unittest.TestCase):
    def test_store_and_recall_profiles(self) -> None:
        import tempfile
        path = tempfile.mktemp(suffix=".json")
        dce = AtDce()
        dce.load_profiles(path)
        dce.feed(b"ATE0S0=2S7=90\\N3%C1&W\r")
        dce.feed(b"ATE1S0=0&F\r")
        self.assertEqual((dce.echo, dce.registers[0]), (True, 0))
        dce.feed(b"ATZ\r")
        self.assertEqual((dce.echo, dce.registers[0], dce.registers[7], dce.error_control, dce.compression),
                         (False, 2, 90, 3, 1))
        again = AtDce()                                  # a new power-up reads the file
        again.load_profiles(path)
        self.assertEqual((again.echo, again.registers[7]), (False, 90))
        self.assertEqual(again.feed(b"ATZ1\r"), b"\r\nOK\r\n")        # profile 1 was never stored: factory defaults
        self.assertTrue(again.echo)

    def test_view_configuration(self) -> None:
        dce = AtDce()
        out = dce.feed(b"AT&V\r")
        self.assertIn(b"El-Modem Active Settings...", out)
        self.assertIn(b"S07=060", out)
        self.assertTrue(out.endswith(b"OK\r\n"))

    def test_identification_and_ms(self) -> None:
        dce = AtDce()
        self.assertIn(b"El-Modem Software Modem", dce.feed(b"ATI3\r"))
        self.assertTrue(dce.feed(b"ATI14\r").endswith(b"ERROR\r\n"))
        self.assertTrue(dce.feed(b"AT+GCAP\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"AT+MS=V32B,1,4800,14400\r").endswith(b"OK\r\n"))
        self.assertEqual(dce.ms, {"mod": "V32B", "auto": 1, "min": 4800, "max": 14400})
        self.assertIn(b"+MS: V32B,1,4800,14400", dce.feed(b"AT+MS?\r"))
        self.assertTrue(dce.feed(b"AT+MS=X99\r").endswith(b"ERROR\r\n"))


class AtiScreenTests(unittest.TestCase):
    def ati(self, dce, n: int) -> str:
        return dce.feed(f"ATI{n}\r".encode()).decode()

    def test_identification_screens(self) -> None:
        dce = AtDce()
        self.assertEqual(self.ati(dce, 0), "ATI0\r\r\n0010\r\n\r\nOK\r\n")        # code, blank line, OK like a Courier
        self.assertRegex(self.ati(dce, 1), r"\r\n[0-9A-F]{4}\r\n\r\nOK\r\n$")
        self.assertIn("\r\nOK\r\nOK\r\n", self.ati(dce, 2))                    # self-test OK, then the command's own OK
        out = self.ati(dce, 3)
        for text in ("El-Modem Software Modem", "Build date:", "Revision: 0.1.0", "https://github.com/donfanning/el-modem"):
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
        self.assertIn("  &C1  &D2  &K3", out)
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
        self.assertTrue(dce.feed(b"AT&Z1=18005550000\r").endswith(b"OK\r\n"))
        self.assertTrue(dce.feed(b"AT&Z4=1\r").endswith(b"ERROR\r\n"))              # only slots 0-3
        out = self.ati(dce, 8)
        self.assertIn("LIST CAPACITIES: MAXIMUM 4 ENTRIES", out)
        self.assertIn("0: 5551234 DELAY: NONE", out)
        self.assertIn("1: 18005550000 DELAY: NONE", out)
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
        dce.begin_incoming_call({"number": "7195550100", "name": "Test Caller"})
        out = self.ati(dce, 15)
        self.assertIn("CALLER ID HISTORIC REGISTER:", out)
        self.assertRegex(out, r"DATE=\d{4}\r\nTIME=\d{4}\r\nNMBR=7195550100\r\nNAME=Test Caller")

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
                                  "blocks": dict(i_tx=100, i_rx=90, i_resent=5, rej_tx=1, rej_rx=2, fcs_errors=3,
                                                 t401_expiries=0, bytes_in=4000, bytes_out=3500, wire_in=2000,
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
        self.assertIn("RX COMPRESSION: 2.00:1", section2)
        self.assertIn("NOT IMPLEMENTED", self.ati(dce, 17))
        dce.notify_call_progress("NO CARRIER")
        dce.take_notifications()
        done = self.ati(dce, 6)
        self.assertIn("El-Modem Link Diagnostics...", done)
        self.assertRegex(done, r"Chars sent\s+120\s+Chars received\s+340")
        self.assertRegex(done, r"Blocks sent\s+100\s+Blocks received\s+90")
        self.assertRegex(done, r"Blocks resent\s+5")
        self.assertRegex(done, r"Last Tx rate\s+33600\s+Last Rx rate\s+33600")
        self.assertRegex(done, r"Modulation\s+V\.34\s+Protocol\s+V\.42")
        self.assertRegex(done, r"Compression\s+V\.42 bis\s+Line Conditioned\s+Ok")
        self.assertIn("Remote Hangup / No Carrier", done)
        self.assertIn("RESENT 5", self.ati(dce, 16))                            # frozen at the end of the call

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
        self.assertIn("NO IAX2 CLIENT", out)
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
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)             # holds a password
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
    def test_ms_maps_to_engines(self) -> None:
        from modem_dce import line_from_ms
        self.assertEqual(line_from_ms({"mod": "V22B", "auto": 0, "min": 300, "max": 2400})["v22_rate"], 2400)
        self.assertEqual(line_from_ms({"mod": "V32B", "auto": 0, "min": 4800, "max": 14400})["v32_rates"],
                         (14400, 12000, 9600, 7200, 4800))
        self.assertEqual(line_from_ms({"mod": "V32", "auto": 0, "min": 4800, "max": 33600})["v32_rates"],
                         (9600, 7200, 4800))
        chain = line_from_ms({"mod": "V34", "auto": 1, "min": 300, "max": 14400})
        self.assertEqual((chain["line_mode"], chain["chain"]["v34_rate"]), ("chain", 14400))
        self.assertEqual(line_from_ms({"mod": "V32B", "auto": 1, "min": 4800, "max": 14400})["chain"]["allow"],
                         (1 << 2) | (1 << 9))
        self.assertEqual(line_from_ms({"mod": "V21", "auto": 0, "min": 300, "max": 300}), {"line_mode": "v21"})


class DialNormalizationTests(unittest.TestCase):
    def test_strips_tone_and_pulse_dial_modifiers(self) -> None:
        self.assertEqual(normalize_dial_number("T106"), "106")
        self.assertEqual(normalize_dial_number("P5551234"), "5551234")

    def test_strips_pauses_flash_and_wait_for_dialtone(self) -> None:
        self.assertEqual(normalize_dial_number("5551234,,,9"), "55512349")
        self.assertEqual(normalize_dial_number("W18005551234"), "18005551234")
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


class OnlineAudioSessionTests(unittest.TestCase):
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
                    packet, _ = sock.recvfrom(2048)
                except (socket.timeout, OSError):
                    continue
                rx.extend(session.handle_rtp_packet(packet))
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
