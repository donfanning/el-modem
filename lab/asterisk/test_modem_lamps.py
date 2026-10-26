"""Front-panel lamps: HS AA CD OH RD SD TR MR DC CS EC ARQ.

Four of these were wrong while looking right. MR was a hardcoded True, so it
showed lit whether or not anything had registered; TR, HS, SYN and ARQ were
wired to None in the dashboard, so they could never light at all. These tests
pin each one to the state it is supposed to reflect, because a lamp that is
wrong but plausible is exactly the kind of defect that survives review.
"""
import json
import os
import socket
import unittest

from modem_control import WorkerControl


class Auto:
    """Stands in for AutoCallerLine (audio-peer/autodetect.py).

    Its state is a plain string, it surfaces bit_rate and connected itself,
    and it holds no v42_mode of its own -- the error control state lives on
    the V.22 engine inside it.
    """

    def __init__(self, state="v22", engine=None):
        self.state = state
        self._v22 = engine

    @property
    def bit_rate(self):
        if self.state == "bell103":
            return 300
        return self._v22.bit_rate

    @property
    def connected(self):
        return self.state == "bell103" or (
            self._v22 is not None and self._v22.is_settled())


class Engine:
    """Stand-in for the SpanDSP engine: bit_rate plus the V.42 state."""

    def __init__(self, bit_rate=0, v42_mode="off", compression=False):
        self.bit_rate = bit_rate
        self.v42_mode = v42_mode
        self.v42_compression = compression

    def is_settled(self):
        return True


class Handover:
    """A V.32/V.34 engine wrapped by its chain; .engine unwraps it."""

    def __init__(self, engine):
        self.engine = engine


class Realtime:
    sums = {}
    counts = {}
    maxima = {}


class Session:
    def __init__(self, engine=None, settled=False, auto=None):
        self._v22 = engine
        self._auto = auto
        self._settled = settled
        self.realtime = Realtime()

    def _line_up(self):
        return self._settled

    @property
    def negotiation_complete(self):
        return self._settled


class Registration:
    def __init__(self, registered=False):
        self.registered = registered


class Dce:
    def __init__(self):
        self._call = {}
        self.registers = [0]
        self.carrier_detect = True
        self.outgoing_pending = False
        self._answering = False
        self.incoming_ringing = False
        self.online = False
        self.error_control = "NONE"
        self.compression = "NONE"
        self.ms = {}


class Serial:
    """Mirrors the pipe backend: driver presence is tracked, but it is NOT a
    terminal being connected, and no modem-control line exists on a byte pipe,
    so TR/RTS/CTS all report None rather than a guess."""

    def __init__(self, present=False, signals=None):
        self._present = present
        self.name = "pipe"
        self._signals = signals

    def signals(self):
        if self._signals is not None:
            return dict(self._signals)
        return dict(TR=None, RTS=None, CTS=None)

    def client_present(self):
        return self._present


class LampTest(unittest.TestCase):
    """poll() is the seam: it builds the lights the dashboard then renders."""

    def lamps(self, *, registration=None, iax_registered=False,
              iax_configured=False, session=None,
              serial=None):
        control_sock, peer = socket.socketpair()
        peer.setblocking(True)
        try:
            control = WorkerControl(control_sock)
            control.poll(Dce(), serial or Serial(), registration, session,
                         iax_registered, iax_configured)
            # Close the writer before draining, or the read below waits for an
            # EOF that only arrives once this test is over.
            control.channel.close()
            data = b""
            while True:
                chunk = peer.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            peer.close()
        events = [json.loads(line) for line in data.splitlines() if line]
        status = [e for e in events if e.get("event") == "status"]
        self.assertTrue(status, "poll() must publish a status event")
        return status[-1]["value"]

    # --- MR: registered to Asterisk, SIP or IAX ---------------------------

    def test_mr_off_when_nothing_has_registered(self):
        self.assertIs(self.lamps()["lights"]["MR"], False)

    def test_mr_off_when_sip_registration_is_only_pending(self):
        value = self.lamps(registration=Registration(registered=False))
        self.assertIs(value["lights"]["MR"], False)
        self.assertEqual(value["registration"], "pending")

    def test_mr_on_when_sip_registered(self):
        value = self.lamps(registration=Registration(registered=True))
        self.assertIs(value["lights"]["MR"], True)
        self.assertEqual(value["registration"], "registered")

    def test_mr_on_when_iax_registered_and_sip_absent(self):
        # IAX registers through the transport's own event stream; the SIP
        # registration object is None because no SIP registrar is configured.
        value = self.lamps(iax_registered=True)
        self.assertIs(value["lights"]["MR"], True)
        self.assertEqual(value["registration"], "registered")

    def test_iax_registration_is_pending_until_it_succeeds(self):
        value = self.lamps(iax_configured=True)
        self.assertIs(value["lights"]["MR"], False)
        self.assertEqual(value["registration"], "pending")

    def test_iax_without_registration_configuration_is_disabled(self):
        value = self.lamps()
        self.assertIs(value["lights"]["MR"], False)
        self.assertEqual(value["registration"], "disabled")

    # --- TR / RS / CS: the terminal end ------------------------------------

    def test_tr_is_unknown_on_a_pipe(self):
        # Regression: TR used to be reported as driver presence, on the
        # reasoning that the driver connecting means a terminal is using the
        # port. It lit before ZOC was ever started. "Can use" is not "is
        # using" -- the driver attaches as soon as the modem serves the pipe,
        # and opening COM3 adds nothing to the broker log, so a terminal
        # attaching is not observable from the modem's side at all.
        #
        # The honest answer is unknown, shown as (.). Lighting TR on a pipe
        # would be inventing a reading the pipe cannot carry.
        lights = self.lamps(serial=Serial(present=True))["lights"]
        self.assertIsNone(lights["TR"])
        # Presence with no driver at all is equally unreportable.
        self.assertIsNone(self.lamps(serial=Serial(present=False))["lights"]["TR"])

    def test_pipe_driver_presence_does_not_claim_external_terminal_owner(self):
        class Pipe(Serial):
            def terminal_present(self):
                return None

        value = self.lamps(serial=Pipe(present=True))
        self.assertEqual(value["owner"], "unknown")

    def test_tr_is_reported_when_the_modem_owns_the_port(self):
        # The COM backend drives a real serial port and can read its control
        # lines, so there TR is a genuine reading. Same dashboard, same lamp.
        # Bound to a stub so the real signals() implementation is exercised
        # without opening a port.
        from modem_serial_com import MS_CTS_ON, MS_DSR_ON, ComSerial

        class Port:
            _rts_asserted = True

            @staticmethod
            def _poll():
                return MS_DSR_ON | MS_CTS_ON

        self.assertEqual(ComSerial.signals.__get__(Port(), ComSerial)(),
                         dict(TR=True, RTS=True, CTS=True))

        class Silent(Port):
            @staticmethod
            def _poll():
                return 0

        silent = ComSerial.signals.__get__(Silent(), ComSerial)()
        self.assertEqual(silent, dict(TR=False, RTS=False, CTS=True))

    def test_unknown_signal_lines_stay_unknown(self):
        # The pipe carries no signal lines. TR/RTS/CTS must report "not known"
        # rather than a fabricated boolean. Driver presence is tracked
        # separately and is not a substitute for any of them.
        lights = self.lamps(serial=Serial(present=True))["lights"]
        self.assertIsNone(lights["TR"])
        self.assertIsNone(lights["RTS"])
        self.assertIsNone(lights["CTS"])

    # --- HS: V.32 (9600) and above ----------------------------------------

    def test_hs_off_below_9600(self):
        for rate in (300, 1200, 2400, 4800):
            with self.subTest(bit_rate=rate):
                session = Session(Engine(bit_rate=rate))
                self.assertIs(self.lamps(session=session)["lights"]["HS"],
                              False)

    def test_hs_on_at_v32_and_above(self):
        # 9600 itself is V.32 and must light: the boundary is >=, not >.
        for rate in (9600, 14400):
            with self.subTest(bit_rate=rate):
                session = Session(Engine(bit_rate=rate))
                self.assertIs(self.lamps(session=session)["lights"]["HS"],
                              True)

    def test_hs_off_for_fsk_where_there_is_no_bit_rate(self):
        # A Bell 103 / V.22 string session has no engine, so no bit_rate.
        # That must read as "not high speed", not as a crash.
        session = Session(engine="bell103")
        self.assertIs(self.lamps(session=session)["lights"]["HS"], False)

    def test_hs_off_on_the_autodetect_line(self):
        # V.22/Bell 103 never reach V.32, so HS stays off whichever way the
        # caller picks a standard.
        for state, rate in (("v22", 2400), ("bell103", 300)):
            with self.subTest(state=state):
                session = Session(auto=Auto(state, Engine(bit_rate=rate)))
                self.assertIs(self.lamps(session=session)["lights"]["HS"],
                              False)

    def test_arq_read_through_the_autodetect_line(self):
        # AutoCallerLine has no v42_mode of its own; the lamp has to reach the
        # V.22 engine inside it or ARQ would never light on a real call.
        session = Session(auto=Auto("v22", Engine(bit_rate=2400,
                                                  v42_mode="lapm")))
        self.assertIs(self.lamps(session=session)["lights"]["ARQ"], True)

    def test_no_rate_before_a_standard_is_picked(self):
        # state == "listen" means no line yet; AutoCallerLine.bit_rate would
        # dereference a None engine, so this must not raise.
        session = Session(auto=Auto("listen", None))
        lights = self.lamps(session=session)["lights"]
        self.assertIs(lights["HS"], False)
        self.assertIs(lights["ARQ"], False)

    def test_hs_reads_through_a_chain_handover(self):
        session = Session(engine=Handover(Engine(bit_rate=14400)))
        self.assertIs(self.lamps(session=session)["lights"]["HS"], True)

    # --- DC / EC: negotiated compression and error correction -------------

    def test_dc_ec_off_without_a_session(self):
        lights = self.lamps()["lights"]
        self.assertIs(lights["DC"], False)
        self.assertIs(lights["EC"], False)

    def test_compression_is_not_error_control(self):
        for compression in (False, True):
            with self.subTest(compression=compression):
                engine = Engine(14400, "lapm", compression)
                lights = self.lamps(session=Session(engine))["lights"]
                self.assertIs(lights["DC"], compression)
                self.assertIs(lights["EC"], True)
                self.assertIs(lights["ARQ"], lights["EC"])

    def test_compression_requires_negotiated_error_control(self):
        lights = self.lamps(session=Session(Engine(14400, "off", True)))["lights"]
        self.assertIs(lights["DC"], False)
        self.assertIs(lights["EC"], False)

    def test_compression_through_autodetect_and_handover(self):
        engine = Engine(2400, "lapm", True)
        for session in (Session(auto=Auto("v22", engine)),
                        Session(engine=Handover(engine))):
            lights = self.lamps(session=session)["lights"]
            self.assertIs(lights["DC"], True)
            self.assertIs(lights["EC"], True)

    def test_unused_v22_engine_cannot_light_compression_or_error_control(self):
        for state in ("listen", "bell103"):
            session = Session(auto=Auto(state, Engine(2400, "lapm", True)))
            lights = self.lamps(session=session)["lights"]
            self.assertIs(lights["DC"], False)
            self.assertIs(lights["EC"], False)
            self.assertIs(lights["ARQ"], False)

    # --- ARQ: error control actually engaged ------------------------------

    def test_arq_off_when_v42_is_not_engaged(self):
        session = Session(Engine(bit_rate=9600, v42_mode="off"), settled=True)
        self.assertIs(self.lamps(session=session)["lights"]["ARQ"], False)

    def test_arq_on_when_lapm_is_engaged(self):
        session = Session(Engine(bit_rate=9600, v42_mode="lapm"), settled=True)
        self.assertIs(self.lamps(session=session)["lights"]["ARQ"], True)


class DashboardLampTest(unittest.TestCase):
    """The mapping is the other half: poll() computing MR is no use if the
    dashboard renders None for it. Both halves have to hold."""

    def test_no_lamp_is_left_hardcoded(self):
        from pathlib import Path
        import modem_tui
        source = Path(modem_tui.__file__).read_text(encoding="utf-8")
        line = next(l for l in source.splitlines() if "mapped={" in l)
        for name in ("HS", "AA", "CD", "OH", "RD", "SD", "TR", "MR",
                     "DC", "CS", "EC", "ARQ"):
            with self.subTest(lamp=name):
                self.assertIn(f"'{name}':", line)
        # Nothing may be a bare None: that is what silently swallowed TR, and
        # then HS, SYN and ARQ. RS/CS legitimately come from RTS/CTS.
        self.assertNotIn(":None", line)
        self.assertIn("'TR':lights.get('TR')", line)
        self.assertIn("'DC':lights.get('DC')", line)
        self.assertIn("'EC':lights.get('EC')", line)
        self.assertNotIn("'RS':", line)
        self.assertNotIn("'SYN':", line)


class PresentedPortTest(unittest.TestCase):
    r"""The dashboard must show the COM port a terminal can actually open.

    It used to show \\.\pipe\ElModemSerial, which is the modem's internal link
    to the driver -- nothing a terminal can open."""

    def test_port_lookup_finds_the_installed_port(self):
        from modem_serial_pipe import presented_port
        port = presented_port()
        if os.name != "nt":
            self.skipTest("the port name is published by the Windows driver")
        if port is None:
            self.skipTest("no El-Modem COM port installed on this machine")
        self.assertRegex(port, r"^COM\d+$")

    def test_port_is_reported_not_the_pipe(self):
        # One port installed here, so the dashboard card should say COM3 and
        # never the pipe path.
        from modem_serial_pipe import presented_port
        if os.name != "nt":
            self.skipTest("Windows only")
        port = presented_port()
        if port is None:
            self.skipTest("no port installed")
        self.assertNotIn("\\pipe\\", port)
        self.assertNotIn("pipe", port.lower())


if __name__ == "__main__":
    unittest.main()
