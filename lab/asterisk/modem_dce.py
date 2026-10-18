#!/usr/bin/env python3
"""Linux PTY and a small Hayes-style DCE command interpreter for the lab."""

import argparse
import cmath
import json
import os
import re
import select
import signal
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable
from queue import Empty, SimpleQueue

# ``bell103.py`` lives alongside this file in the Docker image (everything is
# flattened into /app/) but lives in the audio-peer/ subdirectory in the
# source tree, so fall back to adding that directory to the import path for
# local/non-Docker runs.
_AUDIO_PEER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio-peer")
if os.path.isdir(_AUDIO_PEER_DIR) and _AUDIO_PEER_DIR not in sys.path:
    sys.path.append(_AUDIO_PEER_DIR)

from bell103 import Bell103Receiver, Bell103StreamTransmitter  # noqa: E402
from bell212 import Bell212Receiver, Bell212StreamTransmitter  # noqa: E402
from call_progress import OutgoingCallProgressMonitor  # noqa: E402
from lapm_lite import ArqSession  # noqa: E402
from negotiation import (  # noqa: E402
    MODE_BELL103,
    MODE_BELL212,
    ModeNegotiator,
    line_speed,
)
from pcm_capture import capture_from_env  # noqa: E402
from sip_uac import SipOriginateCall  # noqa: E402
from sip_uas import SipIncomingCallServer  # noqa: E402
from iax_transport import IaxMediaSocket, IaxTransport  # noqa: E402

MAX_COMMAND_BYTES = 256
PRODUCT = "EL-MODEM"
VERSION = "0.1.0"
PRODUCT_CODE = "0010"          # four digits: major, minor, patch (V0.1.0)
STORED_NUMBER_SLOTS = 4        # phone-number slots 0-3 (AT&Zn=, ATDSn)
PNP_ID = "ELM0001"            # three-letter vendor tag + number; not registered with any PnP authority
PROJECT_URL = "https://github.com/donfanning/el-modem"

# Network settings live in string S-registers 100-129 (ATS100=host, ATS100? to read). Passwords read back masked.
NET_REGISTERS = {
    100: ("SIP SERVER", "5060 default port in S101"), 101: ("SIP PORT", ""), 102: ("SIP USER", ""),
    103: ("SIP PASSWORD", ""), 104: ("SIP REGISTRAR", "host[:port]; empty = do not register"),
    105: ("SIP CALLER NUMBER", ""), 106: ("SIP CONTACT HOST", "address providers send incoming calls to"),
    110: ("IAX SERVER", ""), 111: ("IAX PORT", ""), 112: ("IAX USER", ""), 113: ("IAX PASSWORD", ""),
    114: ("IAX CALLER NUMBER", ""), 115: ("IAX REGISTER", "0/1"),
    120: ("TRUNK", "0 = SIP, 1 = IAX"),
}
NET_SECRETS = (103, 113)
NET_DEFAULTS = {101: "5060", 111: "4569", 115: "0", 120: "0"}

CALL_RESULTS = {"RING", "BUSY", "VOICE", "NO ANSWER", "NO CARRIER", "CONNECT", "NO DIALTONE"}
FINAL_CALL_RESULTS = {"BUSY", "VOICE", "NO ANSWER", "NO CARRIER", "CONNECT", "NO DIALTONE"}
RTP_FRAME_SECONDS = 0.02
RTP_SAMPLES_PER_FRAME = 160
XON = 0x11
XOFF = 0x13
# Watermarks are in queued *bits* (10 bits/char at Bell 103 8N1): ~1s of
# buffering triggers a pause request, draining back to ~1/3s resumes. These
# only matter when a DTE pushes data into the PTY faster than the 300 bps
# line can carry it (e.g. a host serial port or terminal emulator left at a
# much higher baud rate than the negotiated line speed).
FLOW_XOFF_HIGH_WATER_BITS = 300
FLOW_XON_LOW_WATER_BITS = 100
# Modulation switch guard times. After the CONFIRM exchange the sender goes
# silent for TX_GUARD_FRAMES (20 ms each) before the new carrier starts, and
# the receiver ignores audio for RX_DEAF_SAMPLES after decoding the peer's
# CONFIRM, so the Bell 212 receiver's carrier-onset detector sees clean
# silence and does not lock onto the tail of the old Bell 103 idle tone.
TX_GUARD_FRAMES = 10
RX_DEAF_SAMPLES = 480


def ulaw_to_linear(sample: int) -> int:
    value = (~sample) & 0xFF
    magnitude = ((value & 0x0F) << 3) + 0x84
    magnitude <<= (value & 0x70) >> 4
    return 0x84 - magnitude if value & 0x80 else magnitude - 0x84


def linear_to_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), 32635) + 0x84
    exponent = 0
    for threshold in (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF):
        if magnitude <= threshold:
            break
        exponent += 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


class EscapeSequenceDetector:
    """Detects the Hayes "+++" guard-time escape sequence without blocking.

    A match requires the configured plus character three times in a row,
    preceded and followed by at least ``guard_time`` seconds with no other
    bytes on the line. ``feed()`` is called with every chunk of terminal
    data (returning the bytes that are *not* part of a candidate sequence,
    so they can still be sent as online data), and ``poll()`` should be
    called periodically (e.g. once per select-loop iteration) so the
    trailing silence can be recognized even when no further bytes arrive.
    """

    def __init__(self, guard_time: float = 1.0, plus_char: int = 0x2B, plus_count: int = 3):
        self.guard_time = guard_time
        self.plus_char = plus_char
        self.plus_count = plus_count
        self._pending_plus = 0
        self._candidate = False
        self._last_byte_time = 0.0

    def feed(self, data: bytes, now: float) -> bytes:
        passthrough = bytearray()
        for byte in data:
            gap = now - self._last_byte_time
            self._last_byte_time = now
            if not self._candidate:
                if byte == self.plus_char and gap >= self.guard_time:
                    self._candidate = True
                    self._pending_plus = 1
                    continue
                passthrough.append(byte)
                continue
            if byte == self.plus_char and self._pending_plus < self.plus_count:
                self._pending_plus += 1
                continue
            # A non-'+' byte, or a 4th '+', cancels the candidate; whatever
            # was tentatively withheld is ordinary data after all.
            passthrough.extend(bytes([self.plus_char]) * self._pending_plus)
            passthrough.append(byte)
            self._candidate = False
            self._pending_plus = 0
        return bytes(passthrough)

    def poll(self, now: float) -> bool:
        if (
            self._candidate
            and self._pending_plus == self.plus_count
            and now - self._last_byte_time >= self.guard_time
        ):
            self._candidate = False
            self._pending_plus = 0
            self._last_byte_time = now
            return True
        return False


def dial_abort_applies(active_outbound, online_session, online, connect_pending, iax_dial_pending=False) -> bool:
    """Whether DTE input before connect must break the chain like a Hayes
    break: hang up and report NO CARRIER. Fires while an originate attempt is
    in flight, or while a session is up but CONNECT is still held for
    negotiation. Never once connected, and never for answer-pending states
    (those never set active_outbound and have no session)."""
    if online:
        return False
    if (active_outbound is not None or iax_dial_pending) and online_session is None and not connect_pending:
        return True
    return online_session is not None and bool(connect_pending)


class EscapeHold:
    """Holds received line bytes while the DCE sits in online command mode
    (after "+++" or &D1), instead of scrolling them over the OK prompt and
    any command being typed. Flushed in order on return-to-online (ATO),
    dropped on hangup or a new call. Bounded: keeps the most recent bytes."""

    LIMIT = 65536

    def __init__(self) -> None:
        self._buf = bytearray()

    def hold(self, data: bytes) -> None:
        if data:
            self._buf += data
            del self._buf[:-self.LIMIT]

    def flush(self) -> bytes:
        data = bytes(self._buf)
        self._buf.clear()
        return data

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


V22_CONNECT_TIMEOUT_SECONDS = 45.0
# Lost-packet concealment for the V.22 modes: a gap of up to this many frames
# (100 ms) is filled by repeating the last received frame; longer gaps are
# filled with silence. See OnlineAudioSession.handle_rtp_packet.
CONCEAL_MAX_FRAMES = 5
# Shown after the speed in CONNECT (ATX1-ATX4) when V.42 LAPM error control is
# up, USR-style: "CONNECT 2400/LAPM". ATX0 stays a bare "CONNECT".
CONNECT_LAPM_SUFFIX = "/LAPM"
CONNECT_V42BIS_SUFFIX = "/V42BIS"
CONNECT_MNP_SUFFIX = "/MNP"
# P0 (both directions), P1 (codewords), P2 (longest string) offered under AT%C1
V42BIS_PARAMS = (3, 1024, 8)
# Hayes numeric result codes for CONNECT at each line speed (ATV0).
NUMERIC_CONNECT_CODES = {300: 1, 1200: 5, 2400: 10, 4800: 11, 9600: 12, 14400: 13}
# Once a V.22 line has connected, losing its carrier for longer than this drops
# the call (NO CARRIER), as a real modem does after S10. Generous, because a
# retrain after a bad patch of packet loss legitimately takes a few seconds.
CARRIER_LOSS_SECONDS = 10.0


class _V22Transmitter:
    """Presents a V22bisModem's transmit side as a stream transmitter."""

    def __init__(self, modem):
        self._modem = modem

    def feed_bytes(self, data: bytes) -> None:
        self._modem.feed_bytes(data)

    def pending(self) -> int:
        return self._modem.pending_bits()

    def feed_bits(self, bits) -> None:
        self._modem.feed_bits(bits)

    def samples(self, count: int) -> list[int]:
        return self._modem.tx_samples(count)


class _V22Receiver:
    """Presents a V22bisModem's receive side as a streaming receiver."""

    def __init__(self, modem):
        self._modem = modem

    def feed_pcm(self, pcm: bytes) -> bytes:
        return self._modem.feed_pcm(pcm)

    def note_concealed_rx(self, n: int) -> None:
        note = getattr(self._modem, 'note_concealed_rx', None)
        if note is not None:note(n)

    def note_transport_discontinuity_rx(self, n: int = 0) -> None:
        note = getattr(self._modem, 'note_transport_discontinuity_rx', None)
        if note is not None:note(n)


class OnlineAudioSession:
    """Bridges a connected call's RTP media to a Bell 103 modem pair.

    ``tx_role``/``rx_role`` follow the Bell 103 convention already used in
    the test suite: the side that originated the call transmits as
    "originate" and must listen for an "answer"-role transmitter, and vice
    versa for the side that answered the call.
    """

    def __init__(
        self,
        sock: socket.socket,
        remote_addr: tuple[str, int],
        tx_role: str,
        rx_role: str,
        negotiate: bool = False,
        supported_modes: tuple[int, ...] = (MODE_BELL103, MODE_BELL212),
        line_mode: str = "bell103",
        v22_rate: int = 1200,
        v42: bool = False,
        v42_frame_size: int = 32,
        v42bis: bool = False,
        mnp: bool = False,
        carrier_loss_seconds: float = CARRIER_LOSS_SECONDS,
        v32_rates: tuple[int, ...] = (9600, 7200, 4800),
        chain: dict | None = None,
        calling_tone: bool = False,
    ):
        self._sock = sock
        self._remote_addr = remote_addr
        # Start at SDP; qualify a consecutive RTP source before learning a
        # symmetric destination. Pin address and SSRC for this call. The
        # probation queue preserves startup samples rather than dropping them.
        self._send_addr = remote_addr
        self._rtp_peer = None
        self._rtp_candidate = []
        self._tx_role = tx_role
        self._rx_role = rx_role
        self._v22 = None
        # Both directions of the linear audio, whatever engine we end up loading: the only way
        # to tell after the fact why a real-PSTN call failed (pcm_capture.py).
        self._capture = capture_from_env(f"{tx_role}-{line_mode}", f"peer={remote_addr}")
        self._last_pcm = b""  # most recent received frame, for concealment
        # Dense V.34 constellations cannot tolerate decoding reordered audio.
        # Hold two frames so short network/sender scheduling jitter can settle
        # before an absent packet is treated as lost.
        # None (automode only): the hold follows the engine the chain selected; see _rtp_jitter_samples.
        self._rtp_jitter_fixed: int | None = None if line_mode == "chain" else (2 * RTP_SAMPLES_PER_FRAME if line_mode == "v34" else 0)
        if os.environ.get('EL_MODEM_RTP_JITTER_MS') is not None:
            milliseconds=int(os.environ['EL_MODEM_RTP_JITTER_MS'])
            if milliseconds not in (0,20,40):raise ValueError('RTP jitter must be 0, 20 or 40 ms')
            self._rtp_jitter_fixed=milliseconds*8
        self._rtp_buffer: dict[int, bytes] = {}
        self._rtp_deadline_drain = os.environ.get('EL_MODEM_RTP_DEADLINE_DRAIN') == '1'
        self._rtp_counts=dict(received_packets=0,decoded_packets=0,late_or_duplicate_packets=0,
            concealed_samples=0,maximum_buffered_packets=0,
            deadline_drain_enabled=self._rtp_deadline_drain,deadline_released_packets=0)
        self._rtp_latest: int | None = None
        self._rtp_ssrc: int | None = None
        self._v22_events_seen = 0
        self._last_auto_state = None
        self._want_v42 = v42 or mnp  # try V.42/LAPM (or MNP, V.42 Annex A) once the line is up
        self._want_mnp = mnp
        self._v42_started = False
        self._v42_frame_size = v42_frame_size
        self._v42bis = v42bis
        self._carrier_loss_seconds = carrier_loss_seconds
        self._was_up = False
        self._carrier_down_at = None
        self._last_rtp_at = None  # monotonic time of the last received RTP packet (None until the first)
        # Debug capture of the V.42 detection phase (see v42_detect.py): send
        # ODP right after CONNECT and log what the peer sends back.
        self._v42_probe = bool(os.environ.get("MODEM_V42_PROBE"))
        self._odp_sent_at = None
        self._probe_rx = bytearray()
        self._probe_logged = 0
        self._adp_verdict = None
        self._created = time.monotonic()
        self.debug = bool(os.environ.get("MODEM_DEBUG"))
        self._auto = None
        if line_mode == "chain":
            # V.8 first, then the best common modulation, else legacy V.22 (autochain.py).
            from autochain import V8Chain  # noqa: PLC0415

            self._v22 = V8Chain(tx_role == "originate", debug=self.debug, **(chain or {}))
            self.transmitter = _V22Transmitter(self._v22)
            self.receiver = _V22Receiver(self._v22)
            self._dsp_capture_pending = bool(os.environ.get('EL_MODEM_DSP_CAPTURE'))
            negotiate = False
        elif line_mode == "v32":
            # Our own streaming V.32 caller (v32_stream.py): 4800/7200/9600 bps with trellis coding.
            # Test-only opt-in: V32_RATES="4800" (comma list) pins the rates offered; unset keeps the default.
            if os.environ.get("V32_RATES"):
                v32_rates = tuple(int(r) for r in os.environ["V32_RATES"].split(",") if r.strip())
            if tx_role == "originate":
                from v32_stream import V32Modem  # noqa: PLC0415

                self._v22 = V32Modem(fs=8000, my_rates=v32_rates, debug=self.debug, echo=False)
            else:
                from v32_answer import V32Answerer  # noqa: PLC0415

                self._v22 = V32Answerer(fs=8000, my_rates=v32_rates, debug=self.debug, echo=False)
            self.transmitter = _V22Transmitter(self._v22)
            self.receiver = _V22Receiver(self._v22)
            negotiate = False
        elif line_mode == "v34":
            # Our V.34 caller and answerer: V.8, phase 2-4 start-up and data mode up to 33600 bit/s.
            if tx_role == "originate":
                from v34_modem import V34Modem  # noqa: PLC0415

                engine = V34Modem
            else:
                from v34_answer_modem import V34AnswerModem  # noqa: PLC0415

                engine = V34AnswerModem
            self._v22 = engine(fs=8000, debug=self.debug)
            from v34_diagnostic import configured as diagnostic_capture
            self._dsp_capture = diagnostic_capture(self._v22)
            self.transmitter = _V22Transmitter(self._v22)
            self.receiver = _V22Receiver(self._v22)
            negotiate = False
        elif line_mode in ("v22bis", "auto"):
            # Our own V.22/V.22bis engine (v22.py): it runs the real
            # handshake itself, so no lab negotiation is used.
            from v22 import V22Modem  # noqa: PLC0415

            self._v22 = V22Modem(
                calling_party=(tx_role == "originate"),
                bit_rate=v22_rate,
                # Bell 212A compatibility only applies at 1200 bps.
                bell212a=(v22_rate == 1200),
            )
            if line_mode == "auto" and tx_role == "originate":
                # Listen to the answerer, then respond in kind (V.22 or
                # Bell 103); see autodetect.py.
                from autodetect import AutoCallerLine  # noqa: PLC0415

                self._auto = AutoCallerLine(
                    self._v22,
                    lambda: Bell103StreamTransmitter(tx_role),
                    lambda: Bell103Receiver(rx_role),
                )
                self.transmitter = _V22Transmitter(self._auto)
                self.receiver = _V22Receiver(self._auto)
            else:
                self.transmitter = _V22Transmitter(self._v22)
                self.receiver = _V22Receiver(self._v22)
            negotiate = False
        else:
            std = "v21" if line_mode == "v21" else "bell103"
            # ITU-T V.21 answerers transmit a continuous 1800 Hz guard tone to
            # disable echo suppressors (matters on international circuits, not
            # domestic digital trunks). Bell 103 has no guard tone.
            guard = (1800.0, 6000.0) if (std == "v21" and tx_role == "answer") else (0.0, 0.0)
            self.transmitter = Bell103StreamTransmitter(tx_role, standard=std,
                                                        guard_hz=guard[0], guard_amp=guard[1],
                                                        v25_answer=(std == "v21" and tx_role == "answer"),
                                                        calling_tone=(3 if (calling_tone and tx_role == "originate"
                                                                            and std in ("v21", "bell103"))
                                                                      else 0))
            self.receiver = Bell103Receiver(rx_role, standard=std)
        # Modulation negotiation (see negotiation.py). Without it the
        # session is plain Bell 103 from the first sample, as before.
        self.mode = MODE_BELL103
        self._negotiator: ModeNegotiator | None = None
        self._pending_tx_mode: int | None = None
        self._tx_drain_done = False
        self._guard_frames = 0
        self._rx_switched = not negotiate
        self._tx_switched = not negotiate
        self._rx_deaf_samples = 0
        self._held = bytearray()
        self._held_lock = threading.Lock()
        if negotiate:
            self._negotiator = ModeNegotiator(supported_modes)
            self.transmitter.feed_bytes(self._negotiator.start(time.monotonic()))
        self._sequence = 0
        self._timestamp = 0
        self._ssrc = int.from_bytes(os.urandom(4), "big")
        self._stop = threading.Event()
        from v34_realtime import RealtimeStats
        self.realtime=RealtimeStats()
        from runtime_timing import TimingTrace
        self.timing_trace = TimingTrace() if os.environ.get('EL_MODEM_TIMING_TRACE') == '1' else None
        self._tx_active = None
        self._tx_watchdog = (threading.Thread(target=self._watch_tx, daemon=True)
            if self.timing_trace is not None and os.environ.get('EL_MODEM_TX_WATCHDOG') == '1'
            else None)
        self._rtp_arrivals={}
        self._rtp_trace_context = {}
        self._rx_context = {}
        self._rx_arrival=None
        from v34_online_control import OnlineControl
        self._online_control = OnlineControl(
            lambda: getattr(self._v22, 'engine', None) or self._v22,
            allow_v34=os.environ.get('V34_CONNECTED_RATE') == '1',
            allow_v32_rate=os.environ.get('V32_CONNECTED_RATE') == '1')
        self._automatic_rate = None
        if os.environ.get('V34_CONNECTED_RATE') == '1':
            if os.environ.get('V34_AUTO_RATE') == '1':
                from v34_auto_rate import AutomaticRate
                self._automatic_rate = AutomaticRate(self._online_control)
                self._auto_traffic = (0, None, 0)
                self._auto_loaded_until = self._auto_transport_until = 0.
                self._auto_transport = (0, 0)
        elif os.environ.get('V34_AUTO_RATE') == '1':
            raise ValueError('V34_AUTO_RATE requires V34_CONNECTED_RATE=1')
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self.xoff_asserted = False
        self._last_rx_timestamp: int | None = None

    @property
    def _rtp_jitter_samples(self) -> int:
        if self._rtp_jitter_fixed is not None:
            return self._rtp_jitter_fixed
        # Automode V.22: no hold. nist-12 and nist-13 connected with 0 ms; nist-9 and nist-10 failed at 40 ms
        # (lab/V22BIS_V8_AUTOMODE_RESEARCH_2026_10_11.md). A V.32 or V.34 engine chosen by the chain keeps 40 ms.
        kind = getattr(self._v22, "kind", None)
        return 2 * RTP_SAMPLES_PER_FRAME if kind in ("v32", "v34") else 0

    @_rtp_jitter_samples.setter
    def _rtp_jitter_samples(self, samples: int) -> None:
        self._rtp_jitter_fixed = samples

    def _gc_pause_begin(self) -> None:
        """Opt-in (--gc-pause): no automatic cyclic collection while the call runs. A full collection takes about 50 ms with the
        diagnostic capture's rows alive and starves the RTP sender (lab/V34_PHASE4_TX_GAP_GC.md); collect now, before the audio starts,
        and again when the call ends."""
        if os.environ.get('EL_MODEM_GC_PAUSE') != '1' or getattr(self, '_gc_pause', None) is not None:
            return
        import gc
        started = time.monotonic()
        self._gc_pause = dict(was_enabled=gc.isenabled(), collected_at_start=gc.collect(),
                              start_collect_ms=round((time.monotonic() - started) * 1000, 1), resumed=False)
        gc.disable()
        self._gc_pause_t0 = time.monotonic()

    def _gc_pause_end(self) -> None:
        state = getattr(self, '_gc_pause', None)
        if state is None or state['resumed']:
            return
        import gc
        state['resumed'] = True
        state['paused_seconds'] = round(time.monotonic() - self._gc_pause_t0, 2)
        if state['was_enabled']:
            gc.enable()
        started = time.monotonic()
        state['collected_at_end'] = gc.collect()
        state['end_collect_ms'] = round((time.monotonic() - started) * 1000, 1)

    def start(self) -> None:
        self._gc_pause_begin()
        self._thread.start()
        if self._tx_watchdog is not None:self._tx_watchdog.start()

    def _watch_tx(self) -> None:
        """Optional bounded stack metadata during a live stall; never retain locals."""
        seen = None
        while not self._stop.wait(.025):
            active = self._tx_active
            if active is None or active[0] == seen:continue
            age = time.monotonic()-active[0]
            if age < .100:continue
            frames = sys._current_frames()
            stacks = []
            for thread in threading.enumerate():
                if thread.ident == threading.get_ident():continue
                frame = frames.get(thread.ident)
                stack = []
                while frame is not None and len(stack)<16:
                    stack.append(dict(file=os.path.basename(frame.f_code.co_filename),
                                      function=frame.f_code.co_name,line=frame.f_lineno))
                    frame = frame.f_back
                stacks.append(dict(thread_id=thread.ident,native_id=thread.native_id,
                                   name=thread.name,stack=stack))
            # Frame references would otherwise keep other threads' locals alive.
            frame = None
            del frames
            # Ignore a frame that finished while its stack metadata was collected.
            current = self._tx_active
            if current is None or current[0] != active[0]:continue
            seen = active[0]
            self.timing_trace.event('tx_watchdog_data' if active[2] else 'tx_watchdog_startup',
                age,dict(started=active[0],sequence=active[1],data=active[2],stage=active[3],
                         observed=time.monotonic(),tx_thread_id=self._thread.ident,
                         threads=stacks))

    @property
    def negotiation_complete(self) -> bool:
        """True once the modulation is settled (immediately when the session
        was created without negotiation). For the SpanDSP V.22bis engine it
        means training has succeeded."""
        if not self._line_up():
            return False
        if self._v42_active():
            # CONNECT is held until error control is resolved, so the result
            # can say /LAPM -- as a real modem in auto-reliable mode does.
            return self._v42_started and self._v22.v42_resolved
        return True

    def _line_up(self) -> bool:
        if self._auto is not None:
            return self._auto.connected
        if self._v22 is not None:
            # Settled, not just trained: a tone can fake a training success,
            # and a 2400 link first trains at 1200 (see spandsp_v22.py).
            return self._v22.is_settled()
        return self._rx_switched and self._tx_switched

    def _v42_active(self) -> bool:
        """Whether V.42 applies to this session (a V.22 line, V.42 wanted)."""
        return (
            self._want_v42
            and self._v22 is not None
            and (self._auto is None or self._auto.state == "v22")
        )

    @property
    def link_lost(self) -> bool:
        """True if the line is gone for good after CONNECT (drop the call):
        the V.42 link failed, or carrier has been missing too long."""
        if self._v22 is None:
            return False
        if self._v22.v42_lost:
            return True
        if self.negotiation_complete and time.monotonic() - (self._last_rtp_at or self._created) > self._carrier_loss_seconds:
            return True  # the far end's media stopped (a dead leg): a modem that hears nothing has lost carrier
        return (
            self._carrier_down_at is not None
            and time.monotonic() - self._carrier_down_at > self._carrier_loss_seconds
        )

    def _track_carrier(self) -> None:
        if self._v22 is None:
            return
        up = self._v22.connected
        if up:
            self._was_up = True
            self._carrier_down_at = None
        elif self._was_up and self._carrier_down_at is None:
            self._carrier_down_at = time.monotonic()

    @property
    def connect_suffix(self) -> str:
        if self._v22 is not None and self._v22.v42_mode == "lapm":
            if self._want_mnp:
                return CONNECT_MNP_SUFFIX
            return CONNECT_LAPM_SUFFIX + (CONNECT_V42BIS_SUFFIX if self._v22.v42_compression else "")
        return ""

    @property
    def failed(self) -> bool:
        """True if a SpanDSP-backed session has not trained in time."""
        if self._v22 is not None and getattr(self._v22, "state", "") == "failed":
            return True
        return (
            self._v22 is not None
            and not self.negotiation_complete
            and time.monotonic() - self._created > V22_CONNECT_TIMEOUT_SECONDS
        )

    @property
    def modulation_name(self) -> str:
        """Which modulation carries the call, for ATI6/ATI11."""
        eng = self._auto.state if self._auto is not None else self._v22
        if isinstance(eng, str):
            return {"v22": "V.22/V.22BIS", "bell103": "BELL 103"}.get(eng, eng.upper())
        if eng is None:
            return {MODE_BELL103: "BELL 103", MODE_BELL212: "BELL 212A"}.get(self.mode, "FSK")
        eng = getattr(eng, "engine", None) or eng                      # V8Chain hands over to a real engine
        name = type(eng).__name__
        if name.startswith("V34"):
            return "V.34"
        if name.startswith("V32"):
            return "V.32BIS" if eng.bit_rate > 9600 else "V.32"
        if name == "V22Modem":
            return "V.22BIS" if eng.bit_rate >= 2400 else "V.22/BELL 212A"
        return name.upper()

    def link_stats(self) -> dict:
        eng = self._v22
        eng = getattr(eng, "engine", None) or eng
        stats: dict = {"error_control": None, "compression": None}
        if eng is None:
            return stats
        stats['realtime']=self.realtime.report()
        if getattr(self, '_gc_pause', None) is not None:stats['gc_pause']=dict(self._gc_pause)
        recorder=getattr(self,'_capture',None)
        recording=recorder.report() if recorder is not None else getattr(self,'_capture_stats',None)
        if recording is not None:stats['pcm_capture']=recording
        if self.timing_trace is not None:
            stats['timing_trace'] = self.timing_trace.report()
        dsp=getattr(eng,'receive_dsp',None)
        if dsp is not None:stats['receive_dsp']=dsp.report()
        distance_backend=getattr(eng,'distance_backend',None)
        if distance_backend is not None:stats['v32_distance_backend']=distance_backend
        echo_stats=getattr(eng,'echo_stats',None)
        if echo_stats is not None:stats['v32_echo']=echo_stats
        if getattr(eng,'is_v32',False):
            stats['v32_retrain']=eng.retrain_stats
            stats['v32_line']=dict(tx_rate=eng.tx.data_rate,rx_rate=eng.bit_rate,
                state=eng.state,profile_rates=list(eng._configured_rates))
        trn_clock=getattr(eng,'_trn_clock',None)
        if trn_clock is not None:stats['v32_trn_clock']=trn_clock.report()
        capture=getattr(self,'_dsp_capture',None)
        if capture is not None:
            stats['dsp_capture']=capture.report()
            stats['engine_state']=getattr(eng,'state',None)
            stats['engine_failed_reason']=getattr(eng,'failed_reason',None)
            stats['engine_events']=getattr(eng,'events',[])
        budget=getattr(eng,'_retrain_budget',None)
        if budget is not None:stats['v34_retrain_budget']=budget.report()
        stats["rx_dbm0"] = getattr(eng, "rx_signal_power_dbm0", None)
        mode = getattr(eng, "v42_mode", "off")
        stats["error_control"] = ("MNP" if self._want_mnp else "V.42 LAPM") if mode == "lapm" else "NONE"
        stats["compression"] = "V.42BIS" if getattr(eng, "v42_compression", None) else "NONE"
        link = getattr(eng, "_v42", None)
        blocks = getattr(link, "stats", None)
        if blocks:
            stats["blocks"] = dict(blocks)
        for part in ("caller", "answerer"):
            core=getattr(eng,part,None)
            baud = getattr(core, "call_baud", None)
            if baud:
                stats["symbol_rate"] = baud
            if core is not None:
                from dataclasses import asdict
                negotiated=getattr(core,'negotiated',None)
                if negotiated is not None:stats['negotiated']=asdict(negotiated)
                stats['rx_qualification']=getattr(core,'rx_qualification',None)
                eq=getattr(getattr(core,'_p4rcv',None),'eq',None)
                if eq is not None:stats['equalizer_backend']=getattr(eq,'arithmetic_backend','legacy')
                guard=getattr(eq,'_input_guard',None)
                if guard is not None:stats['v34_input_power_guard']=guard.report()
                transport=getattr(eq,'_transport_guard',None)
                if transport is not None:stats['v34_gap_feedback']=transport.report()
                phase3=getattr(core,'_p3',None)
                R=getattr(core,'_p4R',None) or (phase3.get('R') if isinstance(phase3,dict) else phase3)
                stats['streaming_storage']=dict(pcm_samples=len(core._rx),pcm_origin=getattr(core,'_rx_origin',0),
                    baseband_samples=len(getattr(R,'_bb_storage',[])))
                state=getattr(core,'_rx_state',None)
                if state is not None:
                    quality = state.get('quality')
                    if quality is not None:stats['v34_data_quality'] = quality.report()
                    vit=state['vit']
                    stats['receive_backend']=vit.backend
                    stats['v34_survivor_state_streaming']=dict(enabled=vit.emit_states,
                        queued_points=len(vit.decided),queued_states=len(vit.decided_states),
                        retained_snapshots=len(vit.snapshots))
        if getattr(self, '_online_control', None) is not None:
            stats['online_control'] = self._online_control.report()
        if getattr(self, '_automatic_rate', None) is not None:
            stats['automatic_rate'] = self._automatic_rate.report()
            received,link,acknowledged=self._auto_traffic
            stats['automatic_rate_activity']=dict(
                received_chars=received,wire_acked=acknowledged,
                source='delivered inbound bytes or valid advancing outbound LAPM acknowledgements',
                activity_hold_seconds=.5)
        stats['rtp']=dict(self._rtp_counts)
        return stats

    def request_online_control(self, parameter):
        owner = getattr(self, '_online_control', None)
        return owner is not None and owner.request(parameter)

    def poll_online_control(self):
        owner = getattr(self, '_online_control', None)
        return owner.poll() if owner is not None else None

    def poll_automatic_rate(self, *, active, mode, received_chars):
        policy = getattr(self, '_automatic_rate', None)
        if policy is None:return None
        now=time.monotonic()
        link=getattr(self._online_control.engine(),'_v42',None)
        acknowledged=getattr(link,'stats',{}).get('wire_acked',0)
        previous_rx,previous_link,previous_ack=self._auto_traffic
        outbound_progress=(acknowledged>previous_ack if link is previous_link else acknowledged>0)
        self._auto_traffic=(received_chars,link,acknowledged)
        if received_chars>previous_rx or outbound_progress:
            self._auto_loaded_until=now+.5
        continuity=tuple(self._rtp_counts.get(k,0) for k in ('concealed_samples','late_or_duplicate_packets'))
        if continuity!=self._auto_transport:
            self._auto_transport=continuity
            self._auto_transport_until=now+2.
        return policy.tick(active=active,mode=mode,loaded=now<self._auto_loaded_until,
                           transport_ok=now>=self._auto_transport_until,now=now)

    @property
    def line_speed(self) -> int:
        if self._auto is not None:
            return self._auto.bit_rate
        if self._v22 is not None:
            return self._v22.bit_rate
        return line_speed(self.mode)

    def poll(self, now: float) -> None:
        """Drives negotiation retries/timeouts; call regularly."""
        if self._v22 is not None and self.debug:
            events = self._v22.events
            while self._v22_events_seen < len(events):
                ev = events[self._v22_events_seen]
                name = ev[1] if isinstance(ev[0], float) else ev[0]     # V32Modem logs (time, text)
                self._v22_events_seen += 1
                print(
                    f"[v22 +{time.monotonic() - self._created:5.2f}s] {name}"
                    f" mode={self._auto.state if self._auto else 'v22'}"
                    f" rate={self._v22.bit_rate}"
                    f" rxpower={self._v22.rx_signal_power_dbm0:.1f}dBm0",
                    file=sys.stderr,
                    flush=True,
                )
        if self._auto is not None and self.debug and self._auto.state != self._last_auto_state:
            self._last_auto_state = self._auto.state
            print(
                f"[auto +{time.monotonic() - self._created:5.2f}s] mode -> "
                f"{self._auto.state}: {self._auto.reason or 'listening'}",
                file=sys.stderr,
                flush=True,
            )
        self._track_carrier()
        if self._v42_probe and self._v22 is not None:
            self._poll_v42_probe()
        if (self._v22 is not None and self._v22.capture_detect_bits
                and self._v22.v42_mode == "failed" and not getattr(self, "_rle_logged", False)):
            self._rle_logged = True
            print(f"[v42] detection failed; rx bits during detection (RLE): "
                  f"{self._v22.detect_bits_rle()}", file=sys.stderr, flush=True)
        if self._v42_active() and not self._v42_started and self._line_up():
            self._v22.start_v42(
                calling_party=(self._tx_role == "originate"),
                frame_size=self._v42_frame_size,
                **({"protocol": "mnp"} if self._want_mnp else
                   {"compression": V42BIS_PARAMS} if self._v42bis else {}),
            )
            self._v42_started = True
            if self.debug:
                print(f"[v42 +{time.monotonic() - self._created:5.2f}s] starting V.42 "
                      f"detection at {self.line_speed} bps", file=sys.stderr, flush=True)
        if getattr(self, '_online_control', None) is not None:
            self._online_control.activate()
        negotiator = self._negotiator
        if negotiator is None or self.negotiation_complete:
            return
        out = negotiator.poll(now)
        if out:
            self.transmitter.feed_bytes(out)
        self._advance_negotiation()

    def _poll_v42_probe(self) -> None:
        from v42_detect import AdpDetector, odp_bits  # noqa: PLC0415

        if self._odp_sent_at is None:
            if self.negotiation_complete:
                self.transmitter.feed_bits(odp_bits(100))  # ~1.5 s at 2400 bps
                self._odp_sent_at = time.monotonic()
                self._adp = AdpDetector()
                print(
                    f"[v42 +{time.monotonic() - self._created:5.2f}s] ODP sent at "
                    f"{self.line_speed} bps",
                    file=sys.stderr, flush=True,
                )
            return
        if len(self._probe_rx) > self._probe_logged and self._probe_logged < 400:
            chunk = bytes(self._probe_rx[self._probe_logged:self._probe_logged + 400])
            self._probe_logged += len(chunk)
            print(
                f"[v42 +{time.monotonic() - self._odp_sent_at:5.2f}s after ODP] rx "
                f"{len(chunk)}B: {chunk.hex(' ')}",
                file=sys.stderr, flush=True,
            )
        verdict = self._adp.result
        if verdict and verdict != self._adp_verdict:
            self._adp_verdict = verdict
            print(f"[v42] ADP detected: {verdict}", file=sys.stderr, flush=True)

    def _advance_negotiation(self) -> None:
        negotiator = self._negotiator
        if negotiator is None or not negotiator.complete:
            return
        mode = negotiator.mode
        assert mode is not None
        if negotiator.fell_back:
            # Nothing was ever agreed: stay on Bell 103 for good.
            self.mode = MODE_BELL103
            self._rx_switched = True
            self._release_held()
            return
        if not self._rx_switched:
            # The peer's CONFIRM has just been decoded; from here it will
            # move to the agreed modulation after its own guard time.
            self.mode = mode
            if mode == MODE_BELL212:
                self.receiver = Bell212Receiver(self._rx_role)
                self._rx_deaf_samples = RX_DEAF_SAMPLES
            self._rx_switched = True
        if self._pending_tx_mode is None and not self._tx_switched:
            self._pending_tx_mode = mode

    def _release_held(self) -> None:
        # Flip the flag under the same lock feed_terminal_bytes uses, so a
        # byte arriving mid-switch is either held (and released here) or
        # sent directly -- never stranded in the buffer.
        with self._held_lock:
            self._tx_switched = True
            held = bytes(self._held)
            self._held.clear()
        if held:
            self.transmitter.feed_bytes(held)

    def stop(self) -> None:
        self._stop.set()
        self._gc_pause_end()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        if self._tx_watchdog is not None and self._tx_watchdog.is_alive():
            self._tx_watchdog.join(timeout=1)
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.close()
            self._capture_stats=capture.report()
        diagnostic=getattr(self,'_dsp_capture',None)
        if diagnostic is not None:diagnostic.close()
        metrics_path=os.environ.get('EL_MODEM_METRICS_FILE')
        if metrics_path:
            import json
            from pathlib import Path
            Path(metrics_path).write_text(json.dumps(self.link_stats(),indent=2)+'\n')

    def feed_terminal_bytes(self, data: bytes) -> None:
        # While modulation is being negotiated the Bell 103 channel belongs
        # to the negotiation frames; hold DTE data until the switch is done.
        with self._held_lock:
            if not self._tx_switched:
                self._held.extend(data)
                return
        self.transmitter.feed_bytes(data)

    def flow_control_signal(self, extra_pending_bits: int = 0) -> bytes | None:
        """Checks the queued-but-not-yet-transmitted bit count against the
        XOFF/XON watermarks and returns a single control byte to send to the
        DTE if a transition is warranted, else ``None``. Only meaningful
        when the caller has confirmed &K selects software flow control
        (AtDce.software_flow_control_enabled()). ``extra_pending_bits`` lets
        a caller fold in backlog sitting in a layer above the transmitter
        (e.g. the ARQ session's own send queue) so flow control reacts to
        the real end-to-end backlog, not just what has reached the
        transmitter's bit queue."""
        pending = self.transmitter.pending() + extra_pending_bits
        if not self.xoff_asserted and pending > FLOW_XOFF_HIGH_WATER_BITS:
            self.xoff_asserted = True
            return bytes((XOFF,))
        if self.xoff_asserted and pending < FLOW_XON_LOW_WATER_BITS:
            self.xoff_asserted = False
            return bytes((XON,))
        return None

    def handle_rtp_payload(self, payload: bytes) -> bytes:
        # The IAX adapter has already removed packet headers and jitter, so it
        # enters here with raw PCMU rather than a complete RTP packet. Keep the
        # same carrier-liveness clock used by handle_rtp_packet(); otherwise
        # every IAX call is falsely declared lost after CARRIER_LOSS_SECONDS.
        self._last_rtp_at = time.monotonic()
        pcm = struct.pack(
            f"<{len(payload)}h", *(ulaw_to_linear(sample) for sample in payload)
        )
        self._last_pcm = pcm
        return self._receive_pcm(pcm)

    def _gap_fill(self, missing: int) -> bytes:
        """PCM standing in for ``missing`` lost samples.

        Silence makes a V.22 modem see its carrier vanish, so it drops carrier
        and retrains (seconds of no data, each time); at only a few percent
        loss that dominated the link. For a short gap, repeating the previous
        frame instead keeps the carrier continuous -- 20 ms is a whole number
        of carrier cycles (1200/2400 Hz) and of 600-baud symbols, so the phase
        is preserved -- and costs only some bit errors, which V.42 repairs.
        (Measured in lossy_sim.py: 11/20 -> 19/20 trials intact at 3% frame
        loss.) Bell 103 keeps the original silence fill. Long gaps get silence
        too: there dropping carrier is the right outcome."""
        zeros = b"\x00\x00" * missing
        if (
            self._v22 is None
            or not self._last_pcm
            or missing > CONCEAL_MAX_FRAMES * RTP_SAMPLES_PER_FRAME
        ):
            return zeros
        frame = self._last_pcm
        repeats = -(-missing * 2 // len(frame))
        return (frame * repeats)[: missing * 2]

    def _receive_pcm(self, pcm: bytes) -> bytes:
        skipped = 0
        capture = self._capture
        if self._rx_deaf_samples > 0:
            skipped = min(self._rx_deaf_samples, len(pcm) // 2)
            self._rx_deaf_samples -= skipped
            pcm = pcm[skipped * 2 :]
            if not pcm:
                if capture is not None:
                    capture.rx(pcm,dict(self._rx_context,skipped_samples=skipped))
                return b""
        if capture is not None:
            capture.rx(pcm,dict(self._rx_context,skipped_samples=skipped))
        started=time.monotonic()
        data_phase=bool(getattr(self._v22,'connected',False))
        age = None
        if self._rx_arrival is not None:
            age=max(0.,started-self._rx_arrival-self._rtp_jitter_samples/8000)
            self.realtime.add('queue_age_seconds',age,data=data_phase)
        diagnostic=getattr(self,'_dsp_capture',None)
        if getattr(self, '_dsp_capture_pending', False) and self._v22.engine is not None:
            # V.8 chooses its engine during the previous receive block. Attach
            # before feeding the first PCM block to the selected V.34 engine.
            self._dsp_capture_pending = False
            if self._v22.kind == 'v34':
                from v34_diagnostic import configured
                self._dsp_capture = diagnostic = configured(self._v22.engine)
        if self._rx_context.get('pcm_source') == 'concealment':
            note = getattr(self.receiver, 'note_concealed_rx', None)
            if note is not None:note(len(pcm)//2)
        decoded = (diagnostic.feed(pcm,self.receiver) if diagnostic is not None
                   else self.receiver.feed_pcm(pcm))
        ended = time.monotonic()
        self.realtime.add('rx_seconds',ended-started,data=data_phase)
        if self.timing_trace is not None:
            row = dict(self._rx_context, started=started, ended=ended,
                       samples=len(pcm)//2, data=data_phase, queue_age=age)
            self.timing_trace.threshold('rx_processing', ended-started, row)
            if age is not None:
                self.timing_trace.threshold('queue', age, row, data=data_phase)
        if self._v42_probe and self._odp_sent_at is not None and decoded:
            self._probe_rx += decoded
            self._adp.feed(decoded)
        negotiator = self._negotiator
        if negotiator is not None and not self._rx_switched:
            # Still on Bell 103 and negotiating: decoded bytes are
            # negotiation frames, not user data.
            reply = negotiator.feed(decoded, time.monotonic())
            if reply:
                self.transmitter.feed_bytes(reply)
            self._advance_negotiation()
            return b""
        return decoded

    def handle_rtp_packet(self, packet: bytes, *, received_at=None, clock_context=None) -> bytes:
        """Order V.34 packets within a 40 ms window before gap concealment."""
        if len(packet) < 12 or packet[0] >> 6 != 2 or packet[1] & 0x7F != 0:
            return b""
        timestamp = struct.unpack_from("!I", packet, 4)[0]
        self._rtp_counts['received_packets']+=1
        self._last_rtp_at = time.monotonic()
        ssrc = struct.unpack_from("!I", packet, 8)[0]
        context = {}
        if self.timing_trace is not None:
            context = dict(clock_context or {}, sequence=struct.unpack_from('!H', packet, 2)[0],
                           timestamp=timestamp, ssrc=ssrc, payload_samples=len(packet)-12,
                           dequeued=time.monotonic(), received_at=received_at)
            self.timing_trace.packet(context)
        if self._rtp_ssrc != ssrc:
            if self._rtp_ssrc is not None:
                note=getattr(self.receiver,'note_transport_discontinuity_rx',None)
                if note is not None:note()
            if self.debug:
                print(f"[rtp] source {ssrc}, initial timestamp {timestamp}", file=sys.stderr, flush=True)
            self._rtp_ssrc = ssrc
            self._rtp_buffer.clear()
            self._rtp_arrivals.clear()
            self._rtp_trace_context.clear()
            self._rtp_latest = None
            self._last_rx_timestamp = None
        if self._last_rx_timestamp is not None:
            advance = (timestamp - self._last_rx_timestamp) & 0xFFFFFFFF
            if advance == 0 or advance >= 0x80000000:
                self._rtp_counts['late_or_duplicate_packets']+=1
                note=getattr(self.receiver,'note_transport_discontinuity_rx',None)
                if note is not None:note()
                return b""  # duplicate/late packets must never rewind the clock
        # A chain fallback can disable the hold with older packets still queued.
        # Include this packet in the ordered drain until that queue is empty;
        # bypassing it would skip real audio and conceal an artificial gap.
        if not self._rtp_jitter_samples and not self._rtp_buffer:
            self._rx_arrival=received_at if received_at is not None else time.monotonic()
            self._rx_context = context
            return self._decode_rtp_packet(packet)
        if self._rtp_deadline_drain and timestamp in self._rtp_buffer:
            self._rtp_counts['late_or_duplicate_packets']+=1
            note=getattr(self.receiver,'note_transport_discontinuity_rx',None)
            if note is not None:note()
            return b""  # a duplicate must not postpone an existing deadline
        if self._rtp_latest is None or 0 < ((timestamp - self._rtp_latest) & 0xFFFFFFFF) < 0x80000000:
            self._rtp_latest = timestamp
        self._rtp_buffer[timestamp] = packet
        self._rtp_counts['maximum_buffered_packets']=max(self._rtp_counts['maximum_buffered_packets'],len(self._rtp_buffer))
        self._rtp_arrivals[timestamp]=received_at if received_at is not None else time.monotonic()
        if self.timing_trace is not None:
            self._rtp_trace_context[timestamp] = context
        return self._drain_rtp(time.monotonic() if self._rtp_deadline_drain else None)

    def rtp_deadline_delay(self, now):
        """Time until the oldest held packet is due; None keeps legacy polling."""
        if not self._rtp_deadline_drain or not self._rtp_buffer:
            return None
        oldest=max(self._rtp_buffer,key=lambda ts:(self._rtp_latest-ts)&0xFFFFFFFF)
        return max(0.,self._rtp_arrivals[oldest]+self._rtp_jitter_samples/8000-now)

    def drain_rtp_deadline(self, now):
        if not self._rtp_deadline_drain or not self._rtp_buffer:
            return b""
        return self._drain_rtp(now)

    def _drain_rtp(self, now=None):
        ordered = sorted(
            ((self._rtp_latest - ts) & 0xFFFFFFFF, ts)
            for ts in self._rtp_buffer
            if ((self._rtp_latest - ts) & 0xFFFFFFFF) < 0x80000000
        )
        out = bytearray()
        for age, ts in reversed(ordered):
            if age < self._rtp_jitter_samples and (now is None or
                    now-self._rtp_arrivals[ts] < self._rtp_jitter_samples/8000):
                break  # never skip an older held packet to drain a newer one
            if age < self._rtp_jitter_samples:
                self._rtp_counts['deadline_released_packets']+=1
            self._rx_arrival=self._rtp_arrivals.pop(ts)
            self._rx_context = self._rtp_trace_context.pop(ts, {})
            out.extend(self._decode_rtp_packet(self._rtp_buffer.pop(ts)))
        return bytes(out)

    def _decode_rtp_packet(self, packet: bytes) -> bytes:
        """Parses a raw RTP packet and feeds its audio to the Bell 103
        receiver, first inserting silence for any gap implied by the RTP
        timestamp.

        The receiver tracks bit timing as a continuously incrementing PCM
        sample count. If an RTP packet is lost in transit, simply decoding
        whatever arrives next -- without accounting for the missing
        samples -- shifts every subsequent sample's position relative to
        the sender's clock by the size of the gap, which permanently
        desyncs start-bit/character framing rather than corrupting only
        the lost bytes. Zero-filling the gap (silence, which looks like no
        carrier rather than a valid tone) keeps the sample clock aligned so
        the demodulator can resynchronize on the next real start bit."""
        if len(packet) < 12 or packet[0] >> 6 != 2 or packet[1] & 0x7F != 0:
            return b""
        _, _, _, timestamp, _ = struct.unpack("!BBHII", packet[:12])
        filled = b""
        if self._last_rx_timestamp is not None:
            gap = (timestamp - self._last_rx_timestamp) & 0xFFFFFFFF
            if self.timing_trace is not None and gap != RTP_SAMPLES_PER_FRAME:
                self.timing_trace.event('rtp_decode_step', gap,
                                       dict(self._rx_context, timestamp_step=gap))
            if self.debug and gap != RTP_SAMPLES_PER_FRAME:
                signed_gap = gap if gap < 0x80000000 else gap - 0x100000000
                print(f"[rtp] timestamp step {signed_gap} samples "
                      f"(expected {RTP_SAMPLES_PER_FRAME})", file=sys.stderr, flush=True)
            missing = gap - RTP_SAMPLES_PER_FRAME
            # Cap how much silence we will manufacture for one gap (1
            # second) so a dropped call/huge reorder doesn't stall the
            # loop computing a multi-minute silence buffer.
            if 0 < missing <= 8000:
                self._rtp_counts['concealed_samples']+=missing
                # Keep what the modem decodes from the filled gap. With
                # concealment the modem keeps running through it, so a whole
                # error-controlled frame can complete -- and be acknowledged --
                # inside the gap; discarding the result silently dropped
                # that frame's payload (32 bytes at a time).
                self._rx_context['pcm_source'] = 'concealment'
                filled = self._receive_pcm(self._gap_fill(missing))
        self._last_rx_timestamp = timestamp
        self._rtp_counts['decoded_packets']+=1
        self._rx_context['pcm_source'] = 'packet'
        return filled + self.handle_rtp_payload(packet[12:])

    def note_rtp_source(self, addr, packet, received_at=None, clock_context=None):
        """Admit PCMU from a qualified address/SSRC, returning buffered packets.

        Three consecutive sequence numbers from one endpoint within one second
        establish the peer. After that, foreign addresses or SSRCs cannot
        redirect the transmitter or inject samples. This is source probation,
        not authentication: a new SIP session is required for address/SSRC
        changes, and deployments must still restrict reachable RTP ports.
        Each returned tuple retains its original arrival time/clock metadata.
        """
        try:
            host, port = addr[0], int(addr[1])
            numeric = socket.inet_pton(socket.AF_INET, host)
        except (TypeError, ValueError, IndexError, OSError):
            return []
        if numeric == bytes(4) or numeric[0] >= 224 or not 1 <= port <= 65535:
            return []
        # The existing decoder consumes a fixed 12-byte header. Do not admit
        # CSRC/extension/padding layouts and interpret their bytes as audio.
        if len(packet) <= 12 or packet[0] != 0x80 or packet[1] & 0x7f != 0:
            return []
        seq, ssrc = struct.unpack_from('!H', packet, 2)[0], packet[8:12]
        peer = ((host, port), ssrc)
        row = (packet, received_at, clock_context)
        if self._rtp_peer is not None:
            return [row] if peer == self._rtp_peer else []
        now = time.monotonic() if received_at is None else received_at
        if self._rtp_candidate:
            old_peer, old_seq, old_at, _ = self._rtp_candidate[-1]
            if peer != old_peer or seq != (old_seq + 1) & 0xffff or not 0 <= now-old_at <= 1:
                self._rtp_candidate.clear()
        self._rtp_candidate.append((peer, seq, now, row))
        if len(self._rtp_candidate) < 3:
            return []
        self._rtp_peer = peer
        self._send_addr = peer[0]
        accepted = [item[3] for item in self._rtp_candidate]
        self._rtp_candidate.clear()
        return accepted

    def handle_rtp_datagram(self, addr, packet, received_at=None, clock_context=None):
        """Apply source admission before PCM/capture/decoder state is touched."""
        output = bytearray()
        for admitted, arrival, context in self.note_rtp_source(addr, packet, received_at, clock_context):
            output.extend(self.handle_rtp_packet(admitted, received_at=arrival, clock_context=context))
        return bytes(output)

    def _next_frame_samples(self) -> list[int]:
        if self._pending_tx_mode is not None and self.transmitter.pending() == 0:
            # The CONFIRM has been fully queued out of the transmitter; give
            # its last bit one more frame to finish, then switch.
            if self._tx_drain_done:
                if self._pending_tx_mode == MODE_BELL212:
                    self.transmitter = Bell212StreamTransmitter(self._tx_role)
                    # Silence lets the peer's Bell 212 receiver see a clean
                    # carrier onset. Staying on Bell 103 needs no guard --
                    # and must not have one: the Bell 103 receiver needs
                    # idle mark before a start bit, or it misses the byte.
                    self._guard_frames = TX_GUARD_FRAMES
                self._pending_tx_mode = None
                self._release_held()
            else:
                self._tx_drain_done = True
        if self._guard_frames > 0:
            self._guard_frames -= 1
            return [0] * RTP_SAMPLES_PER_FRAME
        if getattr(self, '_online_control', None) is not None:
            with self._online_control.lock:
                return self.transmitter.samples(RTP_SAMPLES_PER_FRAME)
        return self.transmitter.samples(RTP_SAMPLES_PER_FRAME)

    def _send_loop(self) -> None:
        next_frame = time.monotonic()
        while not self._stop.is_set():
            started=time.monotonic()
            trace = self.timing_trace
            stamps = [(started,time.thread_time())] if trace is not None else None
            data_phase=bool(getattr(self._v22,'connected',False))
            if self._tx_watchdog is not None:
                self._tx_active = (started,self._sequence,data_phase,'samples')
            linear = self._next_frame_samples()
            if stamps is not None:stamps.append((time.monotonic(),time.thread_time()))
            if self._tx_watchdog is not None:self._tx_active = (started,self._sequence,data_phase,'g711')
            payload = bytes(linear_to_ulaw(sample) for sample in linear)
            if stamps is not None:stamps.append((time.monotonic(),time.thread_time()))
            if self._tx_watchdog is not None:self._tx_active = (started,self._sequence,data_phase,'recording')
            capture = self._capture
            if capture is not None:
                capture.tx(struct.pack(f"<{len(linear)}h", *linear),
                    dict(sequence=self._sequence,timestamp=self._timestamp,
                         ssrc=self._ssrc,deadline=next_frame,pcm_source='before_g711'))
            if stamps is not None:stamps.append((time.monotonic(),time.thread_time()))
            if self._tx_watchdog is not None:self._tx_active = (started,self._sequence,data_phase,'send')
            header = struct.pack(
                "!BBHII", 0x80, 0, self._sequence, self._timestamp, self._ssrc
            )
            try:
                self._sock.sendto(header + payload, self._send_addr)
            except OSError:
                self._tx_active = None
                return
            if stamps is not None:stamps.append((time.monotonic(),time.thread_time()))
            if self._tx_watchdog is not None:self._tx_active = (started,self._sequence,data_phase,'reference')
            engine=getattr(self._v22,'engine',None) or self._v22
            dsp=getattr(engine,'receive_dsp',None)
            if dsp is not None:dsp.commit_tx([ulaw_to_linear(sample) for sample in payload])
            ended = time.monotonic()
            self._tx_active = None
            self.realtime.add('tx_seconds',ended-started,data=data_phase)
            self.realtime.add('sender_lateness_seconds',max(0.,started-next_frame),data=data_phase)
            if self.timing_trace is not None:
                stamps.append((ended,time.thread_time()))
                row = dict(sequence=self._sequence, timestamp=self._timestamp, deadline=next_frame,
                           started=started, ended=ended, data=data_phase)
                names = ('samples','g711','recording','send','reference')
                row['stages_seconds'] = {name:stamps[i+1][0]-stamps[i][0]
                                         for i,name in enumerate(names)}
                row['stages_cpu_seconds'] = {name:stamps[i+1][1]-stamps[i][1]
                                             for i,name in enumerate(names)}
                row['engine_timing'] = getattr(self._v22,'_tx_timing',None)
                self.timing_trace.record('tx', row)
                self.timing_trace.threshold('tx_processing', ended-started, row)
                self.timing_trace.threshold('sender_late', max(0.,started-next_frame), row)
            self._sequence = (self._sequence + 1) & 0xFFFF
            self._timestamp = (self._timestamp + RTP_SAMPLES_PER_FRAME) & 0xFFFFFFFF
            next_frame += RTP_FRAME_SECONDS
            delay = next_frame - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_frame = time.monotonic()


class AtDce:
    def __init__(
        self,
        dial: Callable[[str], bool | str | None] | None = None,
        answer: Callable[[], bool] | None = None,
        hangup: Callable[[], None] | None = None,
    ):
        self._dial = dial
        self._answer = answer
        self._hangup = hangup
        self._resume_online: Callable[[], None] | None = None
        self._online_action = None
        self.online_return_pending = False
        self._input = bytearray()
        self._overflow = False
        self._notifications: SimpleQueue[bytes] = SimpleQueue()
        self._lock = threading.RLock()
        self._answering = False
        self._auto_answer_pending = False
        # Set by the call layer when it will announce CONNECT itself once
        # modulation negotiation finishes (so the result can carry the
        # negotiated speed) instead of AtDce announcing it on answer.
        self.defer_connect = False
        self.connect_speed = 300
        self.connect_suffix = ""  # e.g. "/LAPM" when error control is up
        # What \N resets to (ATZ). 1 = the lab's own ARQ framing, which only
        # works between two lab modems; real-modem line modes set this to 0
        # (raw) so no lab ARQ frames are sent to, and no real data is
        # discarded from, a real peer. See default_error_control_for().
        self.default_error_control = 3 if os.environ.get("EL_MODEM_FACTORY_NATIVE") == "1" else 1
        self.default_auto_rate = 0
        self.profile_path: str | None = None        # where &W keeps the stored profiles (JSON); None = in memory only
        self.profiles: dict[int, dict] = {}
        # Call diagnostics for ATI6/ATI11 and configuration for ATI10, filled in by the call layer.
        self.connect_modulation = ""
        self.link_stats: Callable[[], dict] | None = None
        self.config_lines: list[str] = []
        self.last_call: dict | None = None
        self.terminal_kind = "PTY"                      # shown by ATI4: PTY or TCP
        self.net_defaults: dict[int, str] = {}         # seeded from the command line / environment by run_pty
        self.on_network_change: Callable[[], None] | None = None
        self.stored_numbers: dict[int, str] = {}        # AT&Zn=<number>; dial with ATDSn
        self.last_dialed = ""                           # ATDL redials it
        self.calling_tone_once = False                 # `^` dial modifier: V.25 calling tone, this attempt only
        self.caller_id: dict | None = None             # last incoming call's caller ID (ATI15)
        self._call: dict | None = None
        self._direction = ""
        self._number = ""
        self.extended_settings = {}
        self._factory_reset()

    def _factory_reset(self) -> None:
        # ATZ without a saved profile and AT&F restore the same modulation
        # settings as power-up; _load_profile applies stored settings next.
        self.ms = ({"mod": "V34", "auto": 1, "min": 300, "max": 33600}
                   if os.environ.get("EL_MODEM_FACTORY_NATIVE") == "1" else
                   {"mod": "V32B", "auto": 1, "min": 300, "max": 14400})
        self.extended_settings = {}
        self.net = {r: self.net_defaults.get(r, NET_DEFAULTS.get(r, "")) for r in NET_REGISTERS}
        self.echo = True
        self.quiet = False
        self.verbose = True
        self.result_mode = 4
        self.registers = {
            0: 0, 1: 0, 2: 43, 3: 13, 4: 10, 5: 8, 6: 2, 7: 60, 8: 2, 9: 6, 10: 14, 11: 70, 12: 50, 19: 0, 25: 5,
            32: 17, 33: 19, 36: 7, 37: 0,
        }
        self.online = False
        self.outgoing_pending = False
        self.calling_tone_once = False
        self.online_return_pending = False
        self.incoming_ringing = False
        self.ring_count = 0
        self._answering = False
        self._auto_answer_pending = False
        self.carrier_detect = 1
        self.dtr_behavior = 2
        self.flow_control = 3
        # \N error-control mode: 0 disables the lab's simplified ARQ layer
        # (raw Bell 103 passthrough, as before); 1 (default) enables it.
        # NOTE: this is NOT real ITU-T V.42/LAPM -- see lapm_lite.py's
        # module docstring. It does not negotiate error control with real
        # hardware modems, and it does not yet make this modem
        # tolerant of real RTP packet loss: dropping audio samples can
        # desync the Bell 103 demodulator's bit timing badly enough that
        # CRC/retransmission above it cannot recover a clean byte stream
        # (see test_rtp_gap_fill_prevents_permanent_receiver_desync and
        # test_arq_layer_round_trips_byte_exact_over_live_audio_path in
        # test_modem_dce.py). What it does provide: CRC-checked framing
        # and automatic retransmission over an otherwise-clean channel,
        # and (via handle_rtp_packet's timestamp-gap silence fill plus a
        # receiver watchdog) the receiver no longer wedges permanently
        # after a single lost packet the way it used to.
        self.error_control = self.default_error_control
        self.compression = 0            # %C: V.42bis off by default (ARQ/FEC matter more than compression on VoIP)
        self.auto_rate_mode = self.default_auto_rate

    def software_flow_control_enabled(self) -> bool:
        """&K2/4/5/6 select (some variant of) XON/XOFF software flow
        control. On a bare Linux PTY there are no hardware modem-control
        lines (TIOCMGET/TIOCMSET both fail with ENOTTY on master and slave
        alike -- confirmed experimentally), so &K1/3 (RTS/CTS) are accepted
        without error for terminal-program compatibility but have no
        enforceable effect there. The driver and COM backends drive real
        CTS/RTS lines, so &K1/3 are enforceable on those backends."""
        return self.flow_control in (2, 4, 5, 6)

    def connect_result_text(self) -> str:
        """ATX0 reports a bare CONNECT (classic "blind dial" behavior);
        X1-X4 append the negotiated line speed, matching Hayes/USR
        convention. The speed is whatever modulation negotiation settled on
        (300 for Bell 103, 1200 for Bell 212A)."""
        if self.result_mode == 0:
            return "CONNECT"
        return f"CONNECT {self.connect_speed}{self.connect_suffix}"

    def busy_detect_enabled(self) -> bool:
        """ATX3/X4 enable busy-cadence detection; X0-X2 are "blind dial"
        modes that do not report BUSY (a real USR modem would instead just
        keep waiting and eventually report NO CARRIER/timeout)."""
        return self.result_mode in (3, 4)

    def dialtone_detect_enabled(self) -> bool:
        """ATX2/X4 report NO DIALTONE; X0/X1/X3 do not look for a dial tone, so a missing line shows as NO CARRIER."""
        return self.result_mode in (2, 4)

    def filter_call_result(self, result: str) -> str:
        """Apply the ATX level: results the level does not report become NO CARRIER (X0/X1: neither BUSY nor
        NO DIALTONE; X2: NO DIALTONE only; X3: BUSY only; X4: both)."""
        if result == "BUSY" and not self.busy_detect_enabled():
            return "NO CARRIER"
        if result == "NO DIALTONE" and not self.dialtone_detect_enabled():
            return "NO CARRIER"
        return result

    def feed(self, data: bytes) -> bytes:
        with self._lock:
            return self._feed(data)

    def _feed(self, data: bytes) -> bytes:
        output = bytearray()
        for value in data:
            if self.echo:
                output.append(value)
            if value in (self.registers[3], 10):
                line = bytes(self._input).decode("ascii", "ignore").strip()
                self._input.clear()
                if self._overflow:
                    output.extend(self._result("ERROR"))
                elif line:
                    output.extend(self._execute(line))
                self._overflow = False
            elif value == self.registers[5]:
                if self._input:
                    self._input.pop()
            elif value >= 32:
                if len(self._input) < MAX_COMMAND_BYTES:
                    self._input.append(value)
                else:
                    self._overflow = True
        return bytes(output)

    def take_notifications(self) -> bytes:
        output = bytearray()
        while True:
            try:
                output.extend(self._notifications.get_nowait())
            except Empty:
                return bytes(output)

    def notify_incoming_ring(self) -> None:
        with self._lock:
            self.incoming_ringing = True
            self.ring_count = min(255, self.ring_count + 1)
            self.registers[1] = self.ring_count
            self._notifications.put(self._result("RING"))
            auto_answer_rings = self.registers[0]
            if (
                auto_answer_rings
                and self.ring_count >= auto_answer_rings
                and not self._answering
                and not self._auto_answer_pending
            ):
                self._auto_answer_pending = True
                threading.Thread(target=self._answer_automatically, daemon=True).start()

    def begin_incoming_call(self, caller: dict | None = None) -> None:
        with self._lock:
            self.caller_id = dict(caller or {}, wall=time.time())
            self.incoming_ringing = True
            self.ring_count = 0
            self.registers[1] = 0

    def attach_call_control(
        self,
        answer: Callable[[], bool],
        hangup: Callable[[], None],
    ) -> None:
        with self._lock:
            self._answer = answer
            self._hangup = hangup

    def attach_dial(self, dial: Callable[[str], bool | str | None]) -> None:
        with self._lock:
            self._dial = dial

    def attach_resume_online(self, resume: Callable[[], None]) -> None:
        with self._lock:
            self._resume_online = resume

    def attach_online_control(self, action):
        with self._lock:
            self._online_action = action

    def finish_online_control(self, speed, suffix):
        with self._lock:
            self.online_return_pending = False
            self.connect_speed, self.connect_suffix = speed, suffix
            if self._resume_online is not None:
                self._resume_online()
            if self._call is not None:
                self._call.update(speed=speed, suffix=suffix)
            self._notifications.put(self._result('CONNECT'))

    def notify_call_progress(self, result: str) -> None:
        result = result.upper()
        if result not in CALL_RESULTS:
            raise ValueError(f"unsupported call progress result: {result}")
        with self._lock:
            # ATX0-X2 are "blind dial" modes for BUSY (a real modem would not recognise the busy cadence and
            # would report NO CARRIER); NO DIALTONE needs X2 or X4.
            result = self.filter_call_result(result)
            self._notifications.put(self._result(result))
            if result == "RING":
                return
            self.outgoing_pending = False
            self.online_return_pending = False
            if result == "CONNECT":
                self.online = True
                self._call = {"start": time.monotonic(), "wall": time.time(), "speed": self.connect_speed,
                              "suffix": self.connect_suffix, "modulation": self.connect_modulation,
                              "direction": self._direction, "number": self._number}
            else:
                self.online = False
                self.incoming_ringing = False
                self.call_ended(result)

    def _answer_incoming(self) -> bool:
        with self._lock:
            if not self.incoming_ringing or self._answer is None or self._answering:
                return False
            self._answering = True
            answer = self._answer
        return self._finish_answer(answer)

    def _answer_automatically(self) -> None:
        with self._lock:
            self._auto_answer_pending = False
            if not self.incoming_ringing or self._answer is None or self._answering:
                return
            self._answering = True
            answer = self._answer
        self._finish_answer(answer)

    def _finish_answer(self, answer: Callable[[], bool]) -> bool:
        self._direction, self._number = "answer", ""
        try:
            answered = answer()
        except (OSError, TimeoutError):
            answered = False
        with self._lock:
            self._answering = False
            if answered:
                self.incoming_ringing = False
                self.online = True
                if not self.defer_connect:
                    self._notifications.put(self._result("CONNECT"))
            return answered

    def _result(self, code: str) -> bytes:
        if self.quiet:
            return b""
        numeric = {
            "OK": "0",
            "CONNECT": "1",
            "RING": "2",
            "NO CARRIER": "3",
            "ERROR": "4",
            "NO DIALTONE": "6",
            "BUSY": "7",
            "NO ANSWER": "8",
            "VOICE": "10",
        }
        if code == "CONNECT":
            if self.verbose:
                text = self.connect_result_text()
            elif self.result_mode == 0:
                text = numeric["CONNECT"]
            else:
                text = str(NUMERIC_CONNECT_CODES.get(self.connect_speed, 1))
        else:
            text = code if self.verbose else numeric.get(code, "4")
        return self._line(text) + bytes((self.registers[3], self.registers[4]))

    PROFILE_FIELDS = ("echo", "quiet", "verbose", "result_mode", "dtr_behavior", "flow_control", "error_control",
                      "compression", "carrier_detect", "auto_rate_mode")

    def _profile_snapshot(self) -> dict:
        snap = {name: getattr(self, name) for name in self.PROFILE_FIELDS}
        snap["registers"] = dict(self.registers)
        snap["extended_settings"] = dict(self.extended_settings)
        snap["ms"] = dict(self.ms)
        snap["net"] = {str(k): v for k, v in self.net.items()}
        return snap

    def load_profiles(self, path: str | None) -> None:
        """Remember where stored profiles live and load profile 0 now (power-up behaviour)."""
        self.profile_path = path
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if "profiles" in data:
                    self.profiles = {int(k): v for k, v in data["profiles"].items()}
                    self.stored_numbers = {int(k): v for k, v in data.get("numbers", {}).items()}
                else:                                                  # first layout: profiles only
                    self.profiles = {int(k): v for k, v in data.items()}
            except (OSError, ValueError):
                self.profiles = {}
        self._load_profile(0)

    def _store_profile(self, n: int) -> None:
        previous = self.profiles.get(n)
        self.profiles[n] = self._profile_snapshot()
        try:
            self._save_nvram()
        except ValueError:
            if previous is None: self.profiles.pop(n, None)
            else: self.profiles[n] = previous
            raise

    def _save_nvram(self) -> None:
        """The modem's non-volatile memory: stored profiles and stored phone numbers (a JSON file when --profile-file)."""
        if self.profile_path:
            try:
                from modem_storage import atomic_json
                atomic_json(self.profile_path, {"profiles": {str(k): v for k, v in self.profiles.items()},
                                               "numbers": {str(k): v for k, v in self.stored_numbers.items()}})
            except OSError as exc:
                raise ValueError("profile could not be saved") from exc

    def _load_profile(self, n: int) -> None:
        self._factory_reset()
        snap = self.profiles.get(n)
        if snap:
            for name in self.PROFILE_FIELDS:
                if name in snap:
                    setattr(self, name, snap[name])
            self.registers.update({int(k): v for k, v in snap.get("registers", {}).items()})
            self.extended_settings.update(snap.get("extended_settings", {}))
            self.ms.update(snap.get("ms", {}))
            self.net.update({int(k): v for k, v in snap.get("net", {}).items()})
        self._net_changed()

    def _net_changed(self) -> None:
        if self.on_network_change is not None:
            try:
                self.on_network_change()
            except Exception:                                              # noqa: BLE001 -- settings must still apply
                pass

    def _set_net(self, register: int, value: str) -> bool:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        if register in (101, 111):
            if value and not (value.isdigit() and 1 <= int(value) <= 65535):
                return False
        elif register in (115, 120):
            if value not in ("0", "1"):
                return False
        elif len(value) > 80 or any(ord(c) < 32 for c in value):
            return False
        self.net[register] = value
        self._net_changed()
        return True

    def _net_lines(self, net: dict | None = None) -> list[str]:
        net = self.net if net is None else net
        def shown(r: int) -> str:
            v = net.get(r, "")
            return ("*" * 8 if v else "") if r in NET_SECRETS else v
        return [f"S{r}: {NET_REGISTERS[r][0]} = {shown(r)}" for r in sorted(NET_REGISTERS)]

    def _network_screen(self) -> list[str]:
        sip_host, iax_host = self.net[100], self.net[110]
        lines = ["SIP SETTINGS (S100-S106):"] + self._net_lines()[:7] + ["IAX SETTINGS (S110-S115):"] + self._net_lines()[7:13]
        lines.append(f"TRUNK FOR ATD: {'IAX2' if self.net[120] == '1' else 'SIP'} (S120)")
        lines.append("SIP: " + ("CONFIGURED" if sip_host else "USING COMMAND-LINE DIAL HOST") +
                     (f", REGISTERED AS {self.net[102]}" if self.net[104] and self.net[102] else ", NO REGISTRATION"))
        lines.append("IAX: " + ("CONFIGURED" if iax_host else "NOT CONFIGURED") + " - RFC 5456 / G.711 U-LAW")
        return lines

    # -- call diagnostics and the ATI screens --------------------------------------------------------------------------
    def count_chars(self, sent: int = 0, received: int = 0) -> None:
        """Characters passed between the terminal and the line during the current call (ATI6)."""
        call = self._call
        if call is not None:
            call["chars_sent"] = call.get("chars_sent", 0) + sent
            call["chars_received"] = call.get("chars_received", 0) + received

    def call_ended(self, reason: str, stats: dict | None = None) -> None:
        """Record the call that just ended (connected or not) for ATI6/ATI11. Safe to call more than once."""
        with self._lock:
            self.online_return_pending = False
            call = self._call
            if call is None:
                if self._direction == "originate" and self._number and reason not in ("RING", "ATH"):
                    self.last_call = {"direction": "originate", "number": self._number, "connected": False,
                                      "result": reason, "wall": time.time(), "duration": 0.0}
                    self._number = ""
                return
            self._call = None
            if stats is None and self.link_stats is not None:
                try:
                    stats = self.link_stats()
                except Exception:                                          # noqa: BLE001 -- diagnostics only
                    stats = None
            self.last_call = dict(call, connected=True, result=reason, duration=time.monotonic() - call["start"],
                                  stats=stats or {})
            self._direction = ""

    @staticmethod
    def _crc16(data: bytes) -> int:
        import binascii  # noqa: PLC0415

        return binascii.crc_hqx(data, 0xFFFF)

    def _program_checksum(self) -> str:
        """ATI1: a CRC-16 over the program image (the executable when frozen, else this source file)."""
        if not hasattr(AtDce, "_checksum_cache"):
            path = sys.executable if getattr(sys, "frozen", False) else __file__
            try:
                with open(path, "rb") as fh:
                    AtDce._checksum_cache = f"{self._crc16(fh.read()):04X}"
            except OSError:
                AtDce._checksum_cache = "UNAVAILABLE"
        return AtDce._checksum_cache

    def _self_test(self) -> str:
        """ATI2: known-answer tests standing in for the RAM test: CRC-CCITT check value, a RAM pattern, the DFT."""
        ok = self._crc16(b"123456789") == 0x29B1
        ram = bytearray(65536)
        for pattern in (0x55, 0xAA, 0x00):
            ram[:] = bytes([pattern]) * len(ram)
            ok = ok and ram.count(pattern) == len(ram)
        # a naive DFT rather than numpy's: the point of the check is that we generate a tone and
        # find its bin, which is pure arithmetic. Keeping numpy out of it means ATI2 answers on a
        # host that has no numpy, where the V.22/V.34 engines are simply unavailable.
        tone = [cmath.exp(2j * cmath.pi * 5 * n / 64) for n in range(64)]
        mags = [abs(sum(t * cmath.exp(-2j * cmath.pi * b * n / 64) for n, t in enumerate(tone))) for b in range(64)]
        ok = ok and mags.index(max(mags)) == 5
        return "OK" if ok else "ERROR"

    def _active_settings(self, get) -> str:
        return (f"E{int(get('echo'))} Q{int(get('quiet'))} V{int(get('verbose'))} X{get('result_mode')} "
                f"&C{get('carrier_detect')} &D{get('dtr_behavior')} &K{get('flow_control')} "
                f"\\N{get('error_control')} %C{get('compression')}")

    def _regs_text(self, regs: dict) -> str:
        return " ".join(f"S{int(k):02d}:{v:03d}" for k, v in sorted((int(k), v) for k, v in regs.items()))

    # -- the information screens (laid out like a Courier-class modem's ATI pages; figures are only ever ones we have) --
    SETTINGS_TITLE = f"{PRODUCT.title()}"

    def _register_grid(self, regs: dict) -> list[str]:
        cells = [f"S{int(k):02d}={v:03d}" for k, v in sorted((int(k), v) for k, v in regs.items())]
        return ["  " + "  ".join(cells[i:i + 7]) for i in range(0, len(cells), 7)]

    def _setting_lines(self, get) -> list[str]:
        return [f"  &C{get('carrier_detect')}  &D{get('dtr_behavior')}  &K{get('flow_control')}",
                f"  E{int(get('echo'))}  Q{int(get('quiet'))}  V{int(get('verbose'))}  X{get('result_mode')}",
                f"  %C{get('compression')}  \\N{get('error_control')}"]

    def _number_slots(self) -> list[str]:
        return [f"  {k}= {self.stored_numbers.get(k, '')}".rstrip() if k in self.stored_numbers else f"  {k}= "
                for k in range(STORED_NUMBER_SLOTS)]

    @staticmethod
    def _columns(left: tuple[str, object], right: tuple[str, object] | None = None) -> str:
        text = f"  {left[0]:<22}{str(left[1]):>10}"
        if right:
            text += f"   {right[0]:<22}{str(right[1]):>10}"
        return text

    DISCONNECT_REASONS = {"ATH": "Local Hangup", "HANGUP": "Local Hangup", "NO CARRIER": "Remote Hangup / No Carrier",
                          "BUSY": "Busy", "NO ANSWER": "No Answer", "NO DIALTONE": "No Dial Tone"}

    def _last_call_lines(self) -> list[str]:
        c = self.last_call
        if c is None:
            return [f"{PRODUCT.title()} Link Diagnostics...", "", "  No call yet", ""]
        lines = [f"{PRODUCT.title()} Link Diagnostics...", ""]
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c["wall"]))
        lines.append(f"  {c['direction'].title() or 'Call'} {c.get('number', '')}  started {when}".rstrip())
        if not c["connected"]:
            return lines + ["", f"  Not connected: {self.DISCONNECT_REASONS.get(c['result'], c['result'])}", ""]
        stats = c.get("stats", {})
        b = stats.get("blocks") or {}
        lines.append("")
        lines.append(self._columns(("Chars sent", c.get("chars_sent", 0)), ("Chars received", c.get("chars_received", 0))))
        lines.append(self._columns(("Chars lost", 0)))
        if b:
            lines.append(self._columns(("Octets sent", b["wire_out"]), ("Octets received", b["wire_in"])))
            lines.append(self._columns(("Blocks sent", b["i_tx"]), ("Blocks received", b["i_rx"])))
            lines.append(self._columns(("Blocks resent", b["i_resent"])))
            lines.append(self._columns(("FCS errors", b["fcs_errors"]), ("REJ sent / received", f"{b['rej_tx']}/{b['rej_rx']}")))
        lines.append("")
        lines.append(self._columns(("Last Tx rate", c["speed"]), ("Last Rx rate", c["speed"])))
        protocol = (stats.get("error_control") or "None").replace("V.42 LAPM", "V.42")
        lines.append(self._columns(("Modulation", c.get("modulation") or "Unknown"), ("Protocol", protocol)))
        comp = "V.42 bis" if stats.get("compression") == "V.42BIS" else "None"
        ok = "Ok" if not b or not b["i_tx"] or b["i_resent"] / b["i_tx"] < 0.10 else "Poor"
        lines.append(self._columns(("Compression", comp), ("Line Conditioned", ok)))
        lines.append(self._columns(("Retrains Requested", 0), ("Retrains Granted", 0)))
        lines.append(self._columns(("Fallbacks", 0), ("Disconnect Reason", self.DISCONNECT_REASONS.get(c["result"], c["result"]))))
        mins, secs = divmod(int(c["duration"]), 60)
        lines.append(self._columns(("Duration", f"{mins:02d}:{secs:02d}")))
        return lines + [""]

    def _live_or_last(self) -> tuple[dict | None, dict]:
        """(call record, link stats) for the live call if there is one, else the last finished call."""
        if self._call is not None and self.link_stats is not None:
            try:
                stats = self.link_stats()
            except Exception:                                              # noqa: BLE001
                stats = {}
            return dict(self._call, connected=True, result="ACTIVE"), stats
        c = self.last_call
        return (c, c.get("stats", {})) if c else (None, {})

    def _support_report(self, section: int) -> list[str]:
        c, stats = self._live_or_last()
        head = f"TECHNICAL SUPPORT REPORT - SECTION {section}"
        if c is None or not c.get("connected"):
            return [head, "  NO LINK", ""]
        b = stats.get("blocks") or {}
        if section == 1:
            lines = [head, f"  MODULATION: {c.get('modulation') or 'UNKNOWN'}   TX {c['speed']}  RX {c['speed']}"]
            if stats.get("symbol_rate"):
                lines.append(f"  SYMBOL RATE: {stats['symbol_rate']:.0f}")
            level = stats.get("rx_dbm0")
            lines.append("  RX LEVEL: " + (f"{level:.1f} dBm0" if level is not None else "N/A"))
            lines.append("  SNR: N/A (NOT MEASURED)   ECHO: N/A (NOT MEASURED)")
            if b and b["i_tx"]:
                lines.append(f"  LINE QUALITY: {100.0 * (1 - b['i_resent'] / b['i_tx']):.1f}% OF BLOCKS CLEAN")
            return lines + [""]
        if section == 2:
            lines = [head, "  TRELLIS ERRORS: N/A   FRAME SLIPS: N/A"]
            if b:
                lines.append(f"  V.42 BLOCKS: SENT {b['i_tx']}  RESENT {b['i_resent']}  FCS ERRORS {b['fcs_errors']}  "
                             f"T401 TIMEOUTS {b['t401_expiries']}")
                if b["wire_in"] and b["bytes_in"]:
                    lines.append(f"  RX COMPRESSION: {b['bytes_in'] / b['wire_in']:.2f}:1")
                if b["wire_out"] and b["bytes_out"]:
                    lines.append(f"  TX COMPRESSION: {b['bytes_out'] / b['wire_out']:.2f}:1")
            else:
                lines.append("  NO ERROR-CONTROL LINK")
            return lines + [""]
        return [head, "  PCM / V.90 / V.92: NOT IMPLEMENTED", ""]

    def _help_screen(self, stem: str) -> list[str] | None:
        """The AT$ family: reference screens listing what this modem actually implements."""
        if stem == "":
            return ["HELP - BASIC COMMANDS:",
                    "  A     Answer Call                  O     Return Online",
                    "  Dn    Dial Number (ATDSn stored, ATDL redial)   H     Hang Up",
                    "  En    Command Echo (0=Off, 1=On)   Qn    Result Codes (0=On, 1=Off)",
                    "  Vn    Result Format (0=Num, 1=Word)  Xn   Call Progress Level (0-4)",
                    "  Sr=n  Write Register r           Sr?   Read Register r",
                    "  In    Information screens (ATI0-ATI12, ATI15-ATI17)",
                    "  Z     Reset to Stored Profile",
                    "  $ after a command stem lists its family: AT& AT% ATS ATD AT\\ AT+", ""]
        if stem == "&":
            return ["HELP - AMPERSAND COMMANDS:",
                    "  &Cn   DCD behaviour (stored)       &Dn   DTR drop: 0 ignore 1 command 2 hang up 3 reset",
                    "  &Kn   Flow control (2,4-6 = XON/XOFF)  &F  Factory defaults",
                    "  &V    View active and stored settings  &Wn  Write profile n (0-1) to NVRAM",
                    "  &Zn=s Store phone number n (0-3)", ""]
        if stem == "S":
            return ["HELP - S-REGISTERS:",
                    "  S0   Rings to auto-answer         S7   Carrier wait seconds",
                    "  S1   Ring counter                 S8   Comma pause seconds",
                    "  S2   Escape character (43)        S10  Lost-carrier delay 1/10 s",
                    "  S3   Carriage return (13)         S12  Escape guard time 1/50 s",
                    "  S4   Line feed (10)               S11,S19,S25,S32,S33,S36,S37 stored only",
                    "  S5   Backspace (8)",
                    "  S100-S106 SIP: server, port, user, password, registrar, caller number, contact host",
                    "  S110-S115 IAX: server, port, user, password, caller number, register",
                    "  S120 Trunk for ATD (0=SIP, 1=IAX)  - text values: ATSn=text, ATSn? reads", ""]
        if stem == "%":
            return ["HELP - PERCENT COMMANDS:",
                    "  %C0/%C1   V.42bis data compression off / offered", ""]
        if stem == "\\":
            return ["HELP - BACKSLASH COMMANDS:",
                    "  \\N0 raw   \\N1 lab ARQ   \\N2 MNP 2-4   \\N3 V.42 LAPM (auto-reliable, falls back to raw)", ""]
        if stem == "D":
            return ["HELP - DIAL MODIFIERS:",
                    "  0-9 * #   Digits                   T P   Tone / pulse (accepted, ignored)",
                    "  ,         Pause (S8)               ;     Return to command mode",
                    "  Sn        Dial stored number n     L     Redial last number", ""]
        if stem == "+":
            return ["HELP - EXTENDED COMMANDS:",
                    "  +MS=mod,auto,min,max   Modulation: B103 B212 V21 V22 V22B V32 V32B V34; auto=1 runs V.8",
                    "  +MS? +MS=?             Show / list         +GMI +GMM +GMR +GCAP  Identification",
                    "  +IPR +IFC +ICF         DTE settings (accepted; a PTY or TCP port has no line rate)", ""]
        return None

    def _ati(self, n: int) -> list[str] | None:
        title = PRODUCT.title()
        if n == 0:
            return [PRODUCT_CODE, ""]
        if n == 1:
            return [self._program_checksum(), ""]
        if n == 2:
            return [self._self_test()]                                    # the command's own OK follows
        if n == 3:
            path = sys.executable if getattr(sys, "frozen", False) else __file__
            try:
                built = time.strftime("%m/%d/%y", time.localtime(os.path.getmtime(path)))
            except OSError:
                built = "unknown"
            return [f"{title} Software Modem", f"Build date: {built}", f"Revision: {VERSION}", PROJECT_URL, ""]
        if n == 4:
            dial = "ONLINE" if self.online else "ON-HOOK"
            out = [f"{title} Active Settings...", "", f"  TERMINAL={self.terminal_kind}  DIAL=TONE   {dial}", ""]
            out += self._register_grid(self.registers) + [""]
            out += self._setting_lines(lambda k: getattr(self, k)) + [""]
            out += [f"  +MS={self.ms['mod']},{self.ms['auto']},{self.ms['min']},{self.ms['max']}", ""]
            out += self._net_lines() + [""]
            return out
        if n == 5:
            out = [f"{title} NVRAM Settings...", ""]
            for k in sorted(self.profiles):
                snap = self.profiles[k]
                out += [f"  Stored Profile {k}:"] + self._register_grid(snap["registers"]) + [""]
                out += self._setting_lines(snap.get) + [""]
                if "net" in snap:
                    out += self._net_lines({int(r): v for r, v in snap["net"].items()}) + [""]
            if not self.profiles:
                out += ["  No stored profile", ""]
            return out + ["  Stored Phone Numbers:"] + self._number_slots() + [""]
        if n == 6:
            return self._last_call_lines()
        if n == 7:
            try:
                import numpy as np  # noqa: PLC0415

                numpy_version = np.__version__
            except ImportError:
                numpy_version = "not installed"
            import platform  # noqa: PLC0415

            path = sys.executable if getattr(sys, "frozen", False) else __file__
            try:
                size = f"{os.path.getsize(path) // 1024}KB"
                built = time.strftime("%b %d %Y", time.localtime(os.path.getmtime(path)))
            except OSError:
                size, built = "unknown", "unknown"
            nvram = self.profile_path or "memory only"
            options = "V.34, V.32bis, V.22bis, V.21, V.8, V.42, V.42bis, MNP2-4"
            return [f"{title} Configuration...", "",
                    "  Product Type:            Software",
                    f"  Options:                 {options}",
                    f"  DSP:                     Python {platform.python_version()}, NumPy {numpy_version}",
                    f"  Platform:                {platform.system()} {platform.machine()}",
                    f"  Program Size:            {size}",
                    f"  NVRAM:                   {nvram}",
                    f"  Build Date:              {built}",
                    "  EPROM Checksum:          " + ("Pass" if self._program_checksum() != "UNAVAILABLE" else "Fail"),
                    ""] + ["  " + line for line in self.config_lines] + ([""] if self.config_lines else []) + [
                    f"  {PROJECT_URL}", ""]
        if n == 8:
            out = ["DIAL ARRAY LOG:", f"LIST CAPACITIES: MAXIMUM {STORED_NUMBER_SLOTS} ENTRIES"]
            for k in range(STORED_NUMBER_SLOTS):
                if k in self.stored_numbers:
                    out.append(f"{k}: {self.stored_numbers[k]} DELAY: NONE")
            if not self.stored_numbers:
                out.append("(NO STORED NUMBERS)")
            out.append("REDIAL: ATDL" + (f" ({self.last_dialed})" if self.last_dialed else " (NOTHING DIALED YET)"))
            return out + ["BLACK LIST SCREEN: EMPTY", ""]
        if n == 9:
            return [f"{PNP_ID}\\{title} Software Modem", ""]
        if n == 10:
            return ["DIAL SECURITY ACCOUNT STATUS:", "LOCAL ACCESS:   ENABLED", "AUTOPASSWORD:   DISABLED",
                    "DIALBACK:       DISABLED", "S53 REGISTER MAP: 000", ""]
        if n == 11:
            return self._support_report(1)
        if n == 12:
            return self._network_screen() + [""]
        if n == 15:
            c = self.caller_id
            if not c:
                return ["CALLER ID HISTORIC REGISTER:", "  EMPTY", ""]
            lt = time.localtime(c["wall"])
            return ["CALLER ID HISTORIC REGISTER:", f"DATE={time.strftime('%m%d', lt)}", f"TIME={time.strftime('%H%M', lt)}",
                    f"NMBR={c.get('number', '')}", f"NAME={c.get('name', '')}", ""]
        if n == 16:
            return self._support_report(2)
        if n == 17:
            return self._support_report(3)
        return None

    def _view_config(self) -> bytes:
        """AT&V: the active configuration and every stored profile (the same text as ATI4 and ATI5)."""
        return b"".join(self._line(text) for text in (self._ati(4) or []) + (self._ati(5) or []))

    def _line(self, text: str) -> bytes:
        if self.quiet:
            return b""
        cr = self.registers[3]
        lf = self.registers[4]
        return bytes((cr, lf)) + text.encode("ascii")

    def _execute(self, line: str) -> bytes:
        command = line.upper()
        if command.startswith("AT+CFG"):
            from modem_settings import hayes
            return hayes(self, line)
        if not command.startswith("AT"):
            return self._result("ERROR")
        helped = re.fullmatch(r"AT\s*([&%\\+SD]?)\$", command)
        if helped:                                             # AT$, AT&$, ATS$, AT%$, ATD$, AT\$, AT+$: help screens
            screen = self._help_screen(helped.group(1))
            return b"".join(self._line(t) for t in screen) + self._result("OK") if screen else self._result("ERROR")
        index = 2
        output = bytearray()
        result = "OK"

        while index < len(command):
            current = command[index]
            index += 1
            if current in " \t":
                continue
            if current in "EQVX":
                if index >= len(command) or not command[index].isdigit():
                    return self._result("ERROR")
                value = int(command[index])
                index += 1
                if current == "E" and value in (0, 1):
                    self.echo = bool(value)
                elif current == "Q" and value in (0, 1):
                    self.quiet = bool(value)
                elif current == "V" and value in (0, 1):
                    self.verbose = bool(value)
                elif current == "X" and 0 <= value <= 4:
                    self.result_mode = value
                else:
                    return self._result("ERROR")
            elif current == "S":
                match = re.match(r"\d+", command[index:])
                if not match:
                    return self._result("ERROR")
                register = int(match.group())
                index += len(match.group())
                if register >= 100:                                         # string registers: network settings
                    if register not in NET_REGISTERS:
                        return self._result("ERROR")
                    if index < len(command) and command[index] == "?":
                        index += 1
                        value = self.net[register]
                        output.extend(self._line(("*" * 8 if value else "") if register in NET_SECRETS else value))
                    elif index < len(command) and command[index] == "=":
                        raw = line[index + 1:] if len(line) == len(command) else command[index + 1:]
                        if not self._set_net(register, raw):
                            return self._result("ERROR")
                        index = len(command)                                # the value runs to the end of the line
                    else:
                        return self._result("ERROR")
                    continue
                if not 0 <= register <= 99:
                    return self._result("ERROR")
                if index < len(command) and command[index] == "?":
                    index += 1
                    output.extend(self._line(str(self.registers.get(register, 0))))
                elif index < len(command) and command[index] == "=":
                    index += 1
                    value_match = re.match(r"\d+", command[index:])
                    if not value_match:
                        return self._result("ERROR")
                    value = int(value_match.group())
                    if not 0 <= value <= 255:
                        return self._result("ERROR")
                    self.registers[register] = value
                    index += len(value_match.group())
                else:
                    return self._result("ERROR")
            elif current == "&":
                if index >= len(command) or command[index] not in "FCDKVWYSTGPRLMQZ":
                    return self._result("ERROR")
                setting = command[index]
                index += 1
                if setting == "Z":
                    match = re.match(r"(\d)=(.*)$", command[index:])
                    if not match or int(match.group(1)) >= STORED_NUMBER_SLOTS:
                        return self._result("ERROR")
                    number = match.group(2).strip()
                    if len(number) > 40:
                        return self._result("ERROR")
                    if number:
                        self.stored_numbers[int(match.group(1))] = number
                    else:
                        self.stored_numbers.pop(int(match.group(1)), None)
                    try: self._save_nvram()
                    except ValueError: return self._result("ERROR")
                    index = len(command)
                elif setting in "WYVSTGPRLMQ":
                    match = re.match(r"\d*", command[index:])
                    index += len(match.group())
                    n = int(match.group() or 0)
                    if setting == "V":
                        output.extend(self._view_config())
                    elif setting == "W":
                        if n not in (0, 1):
                            return self._result("ERROR")
                        try: self._store_profile(n)
                        except ValueError: return self._result("ERROR")
                    # &Y (power-up profile), &S/&T/&G/&P/&R/&L/&M/&Q: accepted, nothing to drive on a software line
                elif setting == "F":
                    self._factory_reset()
                    self._net_changed()
                elif setting in "CDK":
                    value_match = re.match(r"\d+", command[index:])
                    if not value_match:
                        return self._result("ERROR")
                    value = int(value_match.group())
                    index += len(value_match.group())
                    if setting == "C" and value in (0, 1):
                        # &C0=force CD always on, &C1=track real carrier
                        # (default). NOTE: both settings are accepted and
                        # stored, but neither can be *signaled* to the DTE
                        # over a bare PTY -- Linux pseudo-terminals have no
                        # hardware modem-control lines at all (TIOCMGET on
                        # either end of a PTY fails with ENOTTY, confirmed
                        # experimentally), so there is no CD wire to drive.
                        # `online`/NO CARRIER result codes are this lab's
                        # substitute carrier indication.
                        self.carrier_detect = value
                    elif setting == "D" and value in (0, 1, 2, 3):
                        # &D0-3 select what a DTR drop should do (ignore,
                        # command mode, hang up, hang up+reset). Same
                        # limitation as &C: a bare PTY has no DTR line for a
                        # real terminal program to drop, so this setting is
                        # stored for compatibility but is not currently
                        # wired to any trigger. A real implementation needs
                        # a genuine serial device or an added sideband
                        # control channel, not a plain PTY.
                        self.dtr_behavior = value
                    elif setting == "K" and 0 <= value <= 6:
                        # 0=none, 1=hardware RTS/CTS, 2=software XON/XOFF,
                        # 3=both, 4=XON/XOFF (TX only), 5/6=transparent
                        # variants. Only the software (XON/XOFF) modes are
                        # enforceable over a bare PTY; see
                        # software_flow_control_enabled().
                        self.flow_control = value
                    else:
                        return self._result("ERROR")
            elif current == "D":
                dial_string = command[index:].strip()
                m = re.match(r"([TP]?)S(\d)$", dial_string)
                if m:                                                  # ATDSn: a stored number
                    dial_string = self.stored_numbers.get(int(m.group(2)), "")
                    if not dial_string:
                        return self._result("ERROR")
                elif re.match(r"[TP]?L$", dial_string):                 # ATDL: redial
                    dial_string = self.last_dialed
                    if not dial_string:
                        return self._result("ERROR")
                self.last_dialed = dial_string.replace("^", "")
                # Commercial `^` dial modifier: V.25 calling tone for this dial
                # attempt only. Assigned fresh every dial (never redialled,
                # never sent to the trunk) and held until the session starts.
                self.calling_tone_once = "^" in dial_string
                if self.calling_tone_once:
                    dial_string = dial_string.replace("^", "")
                self._direction, self._number = "originate", normalize_dial_number(dial_string)
                dial_string = dial_string[:-1] if dial_string.endswith(";") else dial_string
                if not dial_string:
                    result = "ERROR"
                elif self._dial is not None:
                    call_result = self._dial(dial_string)
                    if call_result is None:
                        self.outgoing_pending = True
                        return bytes(output)
                    if isinstance(call_result, bool):
                        call_result = "CONNECT" if call_result else "NO CARRIER"
                    call_result = call_result.upper()
                    if call_result not in FINAL_CALL_RESULTS:
                        call_result = "NO CARRIER"
                    call_result = self.filter_call_result(call_result)
                    self.online = call_result == "CONNECT"
                    result = call_result
                else:
                    self.online = False
                    result = "NO CARRIER"
                break
            elif current == "A":
                if self._answer_incoming():
                    self.online = True
                    return bytes(output)
                else:
                    self.online = False
                    result = "NO CARRIER"
                break
            elif current == "H":
                if index < len(command) and command[index] in "01":
                    index += 1
                if self._hangup is not None:
                    self._hangup()
                self.online = False
                self.call_ended("ATH")
            elif current == "O":
                suffix = command[index:]
                if suffix not in ('', '0', '1', '3', '4', '5') or self.online_return_pending:
                    return self._result('ERROR')
                parameter = int(suffix or 0)
                if not self.online:
                    result = 'NO CARRIER'
                elif parameter:
                    if self._online_action is None or not self._online_action(parameter):
                        return self._result('ERROR')
                    self.online_return_pending = True
                    return bytes(output)  # CONNECT is notified at physical completion.
                else:
                    result = 'CONNECT'
                    if self._resume_online is not None:
                        self._resume_online()
                break
            elif current == "Z":
                # ATZ resets the profile and the rest of the line still runs (ATZE0X4+MS=... must keep its +MS).
                match = re.match(r"\d?", command[index:])
                index += len(match.group())
                self._load_profile(int(match.group() or 0))
            elif current == "I":
                match = re.match(r"\d*", command[index:])
                index += len(match.group())
                lines = self._ati(int(match.group() or 0))
                if lines is None:
                    return self._result("ERROR")
                for text in lines:
                    output.extend(self._line(text))
            elif current == "+":
                rest = command[index:]
                m = re.match(r"(MS|GMI|GMM|GMR|GCAP|IPR|IFC|ICF)(\?|=\?|=.*)?$", rest)
                if not m:
                    return self._result("ERROR")
                name, arg = m.group(1), m.group(2) or ""
                index = len(command)
                if name == "GMI":
                    output.extend(self._line(PRODUCT))
                elif name == "GMM":
                    output.extend(self._line(PRODUCT))
                elif name == "GMR":
                    output.extend(self._line(VERSION))
                elif name == "GCAP":
                    output.extend(self._line("+GCAP: +MS +FCLASS"))
                elif name == "MS":
                    if arg == "?":
                        ms = self.ms
                        output.extend(self._line(f"+MS: {ms['mod']},{ms['auto']},{ms['min']},{ms['max']}"))
                    elif arg == "=?":
                        output.extend(self._line("+MS: (B103,B212,V21,V22,V22B,V32,V32B,V34),(0,1),(300-33600),(300-33600)"))
                    else:
                        fields = arg[1:].split(",")
                        mod = fields[0].strip()
                        if mod not in ("B103", "B212", "V21", "V22", "V22B", "V32", "V32B", "V34"):
                            return self._result("ERROR")
                        try:
                            auto = int(fields[1]) if len(fields) > 1 and fields[1] else self.ms["auto"]
                            lo = int(fields[2]) if len(fields) > 2 and fields[2] else self.ms["min"]
                            hi = int(fields[3]) if len(fields) > 3 and fields[3] else self.ms["max"]
                        except ValueError:
                            return self._result("ERROR")
                        if len(fields) > 4 or auto not in (0, 1) or not 300 <= lo <= hi <= 33600:
                            return self._result("ERROR")
                        self.ms = {"mod": mod, "auto": auto, "min": lo, "max": hi}
                # +IPR/+IFC/+ICF: DTE rate/flow/format -- a PTY or TCP port has no line rate; accepted
                elif arg == "?":
                    output.extend(self._line(f"+{name}: 0" if name == "IPR" else f"+{name}: 3,3" if name == "IFC" else f"+{name}: 3,3"))
                elif arg == "=?":
                    output.extend(self._line(f"+{name}: (0-115200)" if name == "IPR" else f"+{name}: (0-3),(0-3)"))
            elif current == "\\":
                if index >= len(command) or command[index] != "N":
                    return self._result("ERROR")
                index += 1
                # 0 raw, 1 lab ARQ (lab-to-lab only), 2 MNP (V.42 Annex A: MNP 2-4
                # class link request, falling back to raw), 3 auto-reliable: try
                # V.42/LAPM with a real peer, falling back to raw.
                if index >= len(command) or command[index] not in "0123":
                    return self._result("ERROR")
                self.error_control = int(command[index])
                index += 1
            elif current == "%":
                # %C0 / %C1: V.42bis data compression off / offered (only meaningful under \\N3)
                if index >= len(command) or command[index] not in "CE":
                    return self._result("ERROR")
                option=command[index]
                index += 1
                if option=='E' and index<len(command) and command[index]=='?':
                    output.extend(self._line(str(self.auto_rate_mode)))
                    index+=1
                    continue
                if index >= len(command) or command[index] not in ("01" if option=='C' else "012"):
                    return self._result("ERROR")
                if option=='C':self.compression = int(command[index])
                else:self.auto_rate_mode = int(command[index])
                index += 1
            elif current in "TPRWM,;":
                continue
            else:
                return self._result("ERROR")

        return bytes(output) + self._result(result)


_DIAL_MODIFIERS = re.compile(r"[TPRWM@!,;()\-\s^]")


def sip_failure_result(reason: str) -> str:
    """Map why a SIP call attempt failed to the modem result code a user would expect."""
    if reason in ("no line", "no response"):
        return "NO DIALTONE"                    # the trunk/proxy is unreachable or silent: there is no line to dial on
    if reason == "timeout":
        return "NO ANSWER"
    code = reason.removeprefix("sip ")
    if code in ("486", "600", "603"):
        return "BUSY"                           # Busy Here / Busy Everywhere / Decline
    if code in ("503", "502", "504"):
        return "NO DIALTONE"                    # service unavailable, bad gateway, gateway timeout
    if code in ("480", "408"):
        return "NO ANSWER"                      # temporarily unavailable, request timeout
    return "NO CARRIER"


def line_from_ms(ms: dict) -> dict:
    """Turn an AT+MS selection (modulation, automode, min, max rate) into OnlineAudioSession arguments."""
    mod, hi = ms["mod"], ms["max"]
    if ms.get("auto") and mod in ("V22", "V22B", "V32", "V32B", "V34"):
        # Automode: V.8 picks the best modulation both ends offer, no higher than the one selected here.
        from v8 import V8_MOD_V22, V8_MOD_V32, V8_MOD_V34  # noqa: PLC0415

        ceiling = {"V22": 1, "V22B": 1, "V32": 2, "V32B": 2, "V34": 3}[mod]
        allow = V8_MOD_V22 | (V8_MOD_V32 if ceiling >= 2 else 0) | (V8_MOD_V34 if ceiling >= 3 else 0)
        top32 = min(hi, 9600 if mod == "V32" else 14400)
        return {"line_mode": "chain", "chain": {
            "allow": allow, "v34_rate": max(2400, min(hi, 33600)),
            "v32_rates": tuple(r for r in (14400, 12000, 9600, 7200, 4800) if ms["min"] <= r <= top32) or (4800,),
            "v22_rate": 2400 if mod != "V22" and hi >= 2400 else 1200}}
    if mod in ("B103",):
        return {"line_mode": "bell103", "negotiate": False}
    if mod == "V21":
        return {"line_mode": "v21"}
    if mod in ("B212", "V22"):
        return {"line_mode": "v22bis", "v22_rate": 1200}
    if mod == "V22B":
        return {"line_mode": "v22bis", "v22_rate": 2400 if hi >= 2400 else 1200}
    if mod in ("V32", "V32B"):
        ceiling = min(hi, 9600 if mod == "V32" else 14400)
        rates = tuple(r for r in (14400, 12000, 9600, 7200, 4800) if ms["min"] <= r <= ceiling) or (4800,)
        return {"line_mode": "v32", "v32_rates": rates}
    os.environ["V34_RATE"] = str(max(2400, min(hi, 33600)))             # V34
    return {"line_mode": "v34"}


def default_error_control_for(line_mode: str, negotiate: bool) -> int:
    """Whether ``\\N`` defaults to the lab ARQ layer (1) or raw (0).

    The lab ARQ framing (lapm_lite.py) is not V.42, so a real modem neither
    understands it nor sends it: with it on, a real peer's plain data is
    thrown away as malformed ARQ frames (seen live against NIST). It is
    therefore on only where both ends are lab modems by construction -- the
    Bell 103 line mode with lab negotiation -- and off for every real-modem
    path. An explicit ``AT\\N1`` still enables it."""
    return 1 if line_mode == "bell103" and negotiate else 0


def normalize_dial_number(dial_string: str) -> str:
    """Strip Hayes dial modifiers (tone/pulse mode, pauses, flash) from a
    dial string, leaving the digits/*/# an FXO line or SIP extension needs."""
    return _DIAL_MODIFIERS.sub("", dial_string)


def run_pty(
    link: str | None,
    sip_host: str | None = None,
    sip_port: int = 5062,
    media_host: str = "127.0.0.1",
    media_port: int = 16000,
    dial_host: str | None = None,
    dial_port: int = 5060,
    negotiate: bool = True,
    line_mode: str = "bell103",
    v22_rate: int = 1200,
    v42_frame_size: int = 32,
    tcp_listen: str | None = None,
    serial_tcp: str | None = None,
    sip_user: str | None = None,
    caller_number: str | None = None,
    sip_register: str | None = None,
    sip_contact_host: str | None = None,
    profile_file: str | None = None,
    serial_backend=None,
    control=None,
    iax_host: str | None = None,
    iax_port: int = 4569,
    iax_user: str | None = None,
    iax_password: str | None = None,
) -> None:
    # SpanDSP's V.22bis runs its own handshake, so lab negotiation is off,
    # but CONNECT is still deferred until it trains.
    announce_later = negotiate or line_mode in ("v22bis", "auto", "v32", "v34", "ms", "chain")
    negotiate = negotiate and line_mode == "bell103"
    arq_default = 3 if os.environ.get("EL_MODEM_FACTORY_NATIVE") == "1" else default_error_control_for(line_mode, negotiate)
    from serial_port import PtySerial, TcpSerial  # noqa: PLC0415

    if serial_backend is not None:
        ser = serial_backend() if callable(serial_backend) else serial_backend
    elif serial_tcp:
        tcp_h, tcp_p = serial_tcp.rsplit(":", 1)
        ser = TcpSerial(tcp_h, int(tcp_p))
    else:
        ser = PtySerial(link)
    slave_name = ser.name
    terminal_dropped = lambda: None          # replaced once the SIP side exists
    def term_write(data):
        if control is not None and data: control.rd_until = time.monotonic()+.15
        ser.write(data)
    media_socket: socket.socket | None = None
    sip_server: SipIncomingCallServer | None = None
    iax_transport: IaxTransport | None = None
    iax_media_socket: IaxMediaSocket | None = None
    registration = None
    registration_key = None
    final_hangup: Callable[[], None] = lambda: None     # set once the SIP side exists; sends BYE on shutdown
    dce: AtDce
    if tcp_listen and not serial_tcp:
        # Serial port over TCP for terminal programs outside the container (see serial_bridge.py).
        from serial_bridge import serve_pty_over_tcp  # noqa: PLC0415

        tcp_host, tcp_port = tcp_listen.rsplit(":", 1)
        serve_pty_over_tcp(link or slave_name, tcp_host, int(tcp_port))
    if sip_host or iax_host or profile_file:
        media_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        media_socket.bind((media_host, media_port))
        media_socket.setblocking(False)
        kernel_timestamps=False
        if sys.platform.startswith('linux') and hasattr(media_socket,'recvmsg'):
            try:
                media_socket.setsockopt(socket.SOL_SOCKET,35,1)  # SO_TIMESTAMPNS on Linux
                kernel_timestamps=True
            except OSError:pass
        dce = AtDce()
        dce.default_error_control = dce.error_control = arq_default
        dce.default_auto_rate = dce.auto_rate_mode = (2 if os.environ.get('V34_AUTO_RATE') == '1' else 0)
        dce.net_defaults = {
            100: dial_host or "", 101: str(dial_port), 102: sip_user or "",
            103: (os.environ.get("EL_MODEM_SIP_PASSWORD") or os.environ.get("SOFTMODEM_SIP_PASSWORD") or "") if sip_user else "",
            104: sip_register or "", 105: caller_number or "", 106: sip_contact_host or "",
            110: iax_host or "", 111: str(iax_port), 112: iax_user or "",
            113: iax_password or "", 114: caller_number or "", 115: "1" if iax_user else "0",
            120: "1" if os.environ.get("EL_MODEM_NETWORK_TRUNK", "sip").lower() == "iax" else "0",
        }
        dce.load_profiles(profile_file)
        dce.terminal_kind = "TCP" if serial_tcp else ("COM" if getattr(ser, "is_com", False) else "PTY")
        dce.config_lines = [
            f"SERIAL: {ser.name}",
            f"LINE MODE: {line_mode.upper()}" + (f" V.22 {v22_rate}" if line_mode == "v22bis" else ""),
            f"SIP: LISTEN {sip_host}:{sip_port}  RTP {media_host}:{media_port}" if sip_host else "SIP: DISABLED",
            f"IAX: {iax_host}:{iax_port}" if iax_host else "IAX: DISABLED",
            f"DIAL VIA: {dial_host or 'NONE'}:{dial_port}" + (f" AS {sip_user}" if sip_user else ""),
            f"REGISTRAR: {sip_register}" if sip_register else "REGISTRAR: NONE",
            f"V.42 FRAME SIZE: {v42_frame_size}",
        ]
        dce.defer_connect = announce_later
        if sip_host:
            sip_server = SipIncomingCallServer(
                host=sip_host,
                port=sip_port,
                media_host=media_host,
                media_port=media_port,
                on_incoming=lambda: dce.begin_incoming_call(sip_server.last_caller),
                on_ring=dce.notify_incoming_ring,
                on_hangup=lambda: dce.notify_call_progress("NO CARRIER"),
            )

        active_outbound: SipOriginateCall | None = None
        iax_dial_pending = False
        iax_media_active = False
        online_session: OnlineAudioSession | None = None
        arq_session: ArqSession | None = None
        data_mode = False
        connect_pending = False
        escape_detector = EscapeSequenceDetector()

        def start_online_session(
            tx_role: str, rx_role: str, remote_rtp: tuple[str, int]
        ) -> None:
            nonlocal online_session, arq_session, data_mode, connect_pending
            if online_session is not None:
                dce.call_ended("HANGUP", online_session.link_stats() if dce.online else None)
                online_session.stop()
            try:
                session_args = dict(negotiate=negotiate, line_mode=line_mode, v22_rate=v22_rate)
                if line_mode == "ms":                   # AT+MS= picks the modulation for this call
                    session_args.update(line_from_ms(dce.ms))
                if dce.calling_tone_once:               # `^`: consumed here, this call only
                    session_args["calling_tone"] = True
                    dce.calling_tone_once = False
                online_session = OnlineAudioSession(
                    (iax_media_socket if iax_media_active else media_socket),
                    remote_rtp, tx_role, rx_role,
                    **session_args,
                    v42=(dce.error_control == 3), v42_frame_size=v42_frame_size,
                    v42bis=bool(dce.compression), mnp=(dce.error_control == 2),
                )
            except ValueError as exc:               # e.g. v32 cannot answer
                print(f"[modem] {exc}", file=sys.stderr, flush=True)
                online_session = None
                dce.notify_call_progress("NO CARRIER")
                return
            online_session.start()
            escaped.clear()
            # With negotiation on, CONNECT is announced from the main loop
            # once the modulation (and so the line speed) is settled.
            connect_pending = announce_later
            dce.connect_speed = 300
            dce.connect_suffix = ""
            arq_session = ArqSession() if dce.error_control == 1 else None
            data_mode = True
            escape_detector.__init__()

        def end_online_session() -> None:
            nonlocal online_session, arq_session, data_mode, connect_pending, iax_media_active
            session=online_session
            online_session = None
            arq_session = None
            data_mode = False
            connect_pending = False
            iax_media_active = False
            escaped.clear()
            dce.calling_tone_once = False
            if session is not None:
                session.stop()

        def resume_online() -> None:
            nonlocal data_mode
            if online_session is not None:
                held = escaped.flush()
                data_mode = True
                if held:
                    term_write(held)
                    dce.count_chars(received=len(held))

        dce.attach_resume_online(resume_online)
        dce.attach_online_control(lambda parameter: online_session is not None
                                  and online_session.request_online_control(parameter))

        def _terminal_dropped() -> None:
            """The terminal program closed the port: what a DTR drop does on a real modem, per &D."""
            nonlocal data_mode
            mode = dce.dtr_behavior
            busy = online_session is not None or dce.online or dce.outgoing_pending
            if mode == 0 or not busy and mode < 3:
                return
            if mode == 1:
                data_mode = False               # &D1: back to command mode, the connection stays up
                return
            unified_hangup()                    # &D2/&D3: hang up
            dce.online = False
            dce.outgoing_pending = False
            data_mode = False
            if mode == 3:
                dce._factory_reset()

        terminal_dropped = _terminal_dropped

        def report_outgoing_progress(result: str) -> None:
            # BUSY/VOICE are DSP-derived final outcomes from call-progress
            # audio; the underlying SIP INVITE is otherwise still open
            # (Asterisk itself hasn't ended it yet), so tear it down here
            # rather than leaving it dangling until the far end times out.
            nonlocal active_outbound
            if result in ("BUSY", "VOICE") and active_outbound is not None:
                call = active_outbound
                active_outbound = None
                dce.notify_call_progress(result)
                call.hangup()
            else:
                dce.notify_call_progress(result)

        def unified_hangup() -> None:
            nonlocal active_outbound
            try:
                end_online_session()
            finally:
                call=active_outbound
                active_outbound = None
                if call is not None:
                    call.hangup()
                elif sip_server is not None:
                    sip_server.hangup()
                if iax_transport is not None and iax_transport.call_id is not None:
                    iax_transport.hangup()

        def answer_incoming() -> bool:
            nonlocal iax_media_active
            if iax_transport is not None and iax_transport.incoming_call_id is not None:
                answered = iax_transport.answer()
                if answered:
                    iax_media_active = True
                return answered
            answered = sip_server.answer() if sip_server is not None else False
            if answered:
                remote_rtp = sip_server.remote_rtp()
                if remote_rtp:
                    start_online_session("answer", "originate", remote_rtp)
            return answered

        dce.attach_call_control(answer_incoming, unified_hangup)
        final_hangup = unified_hangup
        if sip_server is not None:
            sip_server.on_hangup = lambda: (end_online_session(), dce.notify_call_progress("NO CARRIER"))
            sip_server.start()
        def apply_network() -> None:
            """Apply active SIP registration and IAX endpoint settings."""
            nonlocal registration, registration_key, iax_transport, iax_media_socket
            net = dce.net
            key = (net[104], net[102], net[103], net[106] or sip_host) if net[104] and net[102] else None
            if key != registration_key:
                if registration is not None:
                    registration.stop()
                    registration = None
                registration_key = key
                if key and sip_server is not None:
                    from sip_auth import SipRegistration  # noqa: PLC0415

                    reg_host, _, reg_port = key[0].partition(":")
                    registration = SipRegistration(reg_host, int(reg_port or 5060), key[1], key[2], key[3], sip_port)
                    registration.start(shared_with=sip_server)

            iax_key = (net[110], int(net[111] or 4569), net[112], net[113]) if net[110] and net[112] else None
            current_key = getattr(iax_transport, "config_key", None)
            if iax_key != current_key:
                if iax_transport is not None:
                    iax_transport.close()
                    iax_transport = None
                    iax_media_socket = None
                if iax_key:
                    iax_transport = IaxTransport(*iax_key)
                    iax_transport.config_key = iax_key
                    iax_media_socket = IaxMediaSocket(iax_transport)

        dce.on_network_change = apply_network
        apply_network()

        if True:

            def dial_number(number: str) -> str | None:
                nonlocal active_outbound, iax_dial_pending, iax_media_active
                if active_outbound is not None:
                    return
                dce._call_progress_monitor = None
                destination = normalize_dial_number(number)

                def _clear() -> None:
                    nonlocal active_outbound
                    active_outbound = None

                def _on_connected(remote_rtp: tuple[str, int]) -> None:
                    start_online_session("originate", "answer", remote_rtp)
                    if not announce_later:
                        dce.notify_call_progress("CONNECT")

                def _on_failed(reason: str) -> None:
                    _clear()
                    end_online_session()
                    dce.notify_call_progress(sip_failure_result(reason))

                def _on_hangup() -> None:
                    _clear()
                    end_online_session()
                    dce.notify_call_progress("NO CARRIER")

                net = dce.net
                if net[120] == "1":
                    if iax_transport is None or not net[110] or not net[112]:
                        return "NO DIALTONE"
                    try:
                        iax_media_active = True
                        iax_dial_pending = True
                        iax_transport.dial(destination)
                    except (OSError, ValueError) as exc:
                        iax_dial_pending = False
                        iax_media_active = False
                        print(f"[modem] IAX dial failed: {exc}", file=sys.stderr, flush=True)
                        return "NO CARRIER"
                    return None
                target_host = net[100] or dial_host
                if not target_host or sip_server is None:
                    return "NO DIALTONE"
                call = SipOriginateCall(
                    host=sip_host,
                    port=0,
                    target_host=target_host,
                    target_port=int(net[101] or dial_port),
                    media_host=media_host,
                    media_port=media_port,
                    on_connected=_on_connected,
                    on_failed=_on_failed,
                    on_hangup=_on_hangup,
                    username=net[102] or None,
                    password=net[103] or None,
                    caller_number=net[105] or None,
                )
                active_outbound = call
                threading.Thread(
                    target=call.dial, args=(destination,), daemon=True
                ).start()

            dce.attach_dial(dial_number)
    else:
        dce = AtDce()
        online_session = None
        arq_session = None
        data_mode = False
        iax_dial_pending = False
        iax_media_active = False
        iax_transport = None
        iax_media_socket = None
        # Bound on every path: the main loop reads these unconditionally, and
        # the SIP suite below only assigns them when SIP exists. Without this
        # the standalone daemon dies on the first keystroke (UnboundLocalError).
        active_outbound = None
        connect_pending = False
        escape_detector = EscapeSequenceDetector()
    if control is not None:
        dce.control_settings = True
    print(slave_name, flush=True)
    escaped = EscapeHold()
    def deliver_received(decoded):
        # Timer and packet drains use the same DTE/ARQ delivery and prompt ACK.
        if not decoded or connect_pending:
            return
        if arq_session is not None:
            delivered = arq_session.feed_received(decoded)
            framed = arq_session.pump_outgoing(time.monotonic())
            if framed:
                online_session.feed_terminal_bytes(framed)
            if not delivered:
                return
        else:
            delivered = decoded
        if dce.online and not data_mode and online_session is not None:
            # Online command mode (after "+++" or &D1): hold the bytes instead
            # of scrolling them over the OK prompt. Counted on flush.
            escaped.hold(delivered)
            return
        term_write(delivered)
        if os.environ.get("MODEM_DEBUG") == "1" and online_session is not None and data_mode:
            print(f"[dce] terminal receive bytes={len(delivered)}", file=sys.stderr, flush=True)
        dce.count_chars(received=len(delivered))
    try:
        while True:
            if control is not None:
                control.poll(dce, ser, registration, online_session)
                ser.update_lines(dce.online or not dce.carrier_detect, dce.incoming_ringing, True,
                                 dce.flow_control in (1,3),
                                 ((online_session.transmitter.pending() if online_session else 0) +
                                  (arq_session.pending_bytes()*10 if arq_session else 0)))
            readers = list(ser.fds())
            if control is not None:
                readers.append(control.channel.sock)

            if media_socket:
                readers.append(media_socket.fileno())
            if iax_transport is not None:
                # IAX media shares the DSP event loop; include its UDP socket
                # so startup audio wakes us immediately instead of batching
                # it behind the 250 ms idle timeout.
                readers.append(iax_transport.fileno())
            timeout = 0.05 if data_mode else 0.25
            if online_session is not None:
                due = online_session.rtp_deadline_delay(time.monotonic())
                if due is not None:
                    timeout = min(timeout,due)
            if iax_transport is not None and iax_media_active and online_session is not None:
                audio_due = iax_transport.audio_delay()
                if audio_due is not None:
                    timeout = min(timeout, audio_due)
            loop_trace = online_session.timing_trace if online_session is not None else None
            select_started = time.monotonic() if loop_trace is not None else None
            readable, writable, _ = select.select(readers, ser.write_fds(), [], timeout)
            control_started = time.monotonic() if loop_trace is not None else None
            data, dropped = ser.service(readable, writable)
            if control is not None and data: control.sd_until = time.monotonic()+.15
            serial_ended = time.monotonic() if loop_trace is not None else None
            if dropped:
                terminal_dropped()
            if iax_transport is not None:
                for event in iax_transport.poll():
                    kind = event.get("type")
                    if kind == "incoming":
                        dce.begin_incoming_call({"number": event.get("from", ""), "technology": "IAX2"})
                        dce.notify_incoming_ring()
                    elif kind == "answered":
                        iax_dial_pending = False
                        iax_media_active = True
                        if online_session is None:
                            roles = ("answer", "originate") if iax_transport.call_direction == "incoming" else ("originate", "answer")
                            start_online_session(*roles, (iax_host or "127.0.0.1", iax_port))
                            if not announce_later:
                                dce.notify_call_progress("CONNECT")
                    elif kind == "ended":
                        was_pending = iax_dial_pending
                        iax_dial_pending = False
                        iax_media_active = False
                        if os.environ.get("MODEM_DEBUG") == "1":
                            rx_frames, tx_frames, jitter_stats = iax_transport.audio_stats()
                            print(f"[iax] call ended reason={event.get('reason', 'unknown')} "
                                  f"audio_rx={rx_frames} audio_tx={tx_frames} "
                                  f"jitter(played,concealed,late,overflow,target_ms,buffered)={jitter_stats}",
                                  file=sys.stderr, flush=True)
                        if online_session is not None:
                            end_online_session()
                        if was_pending or dce.online or dce.outgoing_pending:
                            dce.notify_call_progress("NO CARRIER")
                    elif kind == "register_lost":
                        print(f"[iax] registration lost: {event.get('reason', 'unknown')}", file=sys.stderr, flush=True)
                if online_session is not None and iax_media_active:
                    for ulaw in iax_transport.drain_audio():
                        deliver_received(online_session.handle_rtp_payload(ulaw))
            if online_session is not None:
                online_session.poll(time.monotonic())
                if dce.online_return_pending:
                    completed_control = online_session.poll_online_control()
                    if completed_control is not None:
                        if completed_control['status'] == 'failed':
                            unified_hangup()
                            dce.notify_call_progress('NO CARRIER')
                            continue
                        else:
                            dce.finish_online_control(online_session.line_speed, online_session.connect_suffix)
                automatic_result=online_session.poll_automatic_rate(
                    active=data_mode and dce.online and not connect_pending and not dce.online_return_pending,
                    mode=dce.auto_rate_mode,received_chars=(dce._call or {}).get('chars_received',0))
                if automatic_result is not None:
                    if automatic_result['status']=='failed':
                        unified_hangup();dce.notify_call_progress('NO CARRIER');continue
                    dce.connect_speed=online_session.line_speed
                    if dce._call is not None:dce._call['speed']=online_session.line_speed
                if connect_pending and not online_session.negotiation_complete:
                    # Why the CONNECT hold is still in force. A 33600/LAPM call reported a bare
                    # "CONNECT 33600" seconds after V.42 detection started, which negotiation_complete
                    # should forbid while v42_mode is "detect" -- v42_resolved is False in that state.
                    # The suspicion is _v42_active(), which is not a stable predicate: it also requires
                    # _auto.state == "v22", so a multi-standard session whose _auto state moves off
                    # "v22" after start_v42 would flip _v42_active() False and release the hold while
                    # V.42 is still detecting. Logging the four values that decide it makes that
                    # visible instead of inferred.
                    _ev = os.environ.get("V42_DEBUG")
                    if _ev:
                        print(f"[v42 +{time.monotonic() - online_session._created:5.2f}s] CONNECT held: "
                              f"v42_mode={getattr(online_session._v22, 'v42_mode', None)!r} "
                              f"v42_started={online_session._v42_started} "
                              f"v42_active={online_session._v42_active()} "
                              f"want_v42={online_session._want_v42} "
                              f"auto_state={getattr(online_session._auto, 'state', None)!r} "
                              f"line_up={online_session._line_up()}",
                              file=sys.stderr, flush=True)
                if connect_pending and online_session.negotiation_complete:
                    connect_pending = False
                    dce.connect_speed = online_session.line_speed
                    dce.connect_suffix = online_session.connect_suffix
                    if os.environ.get("V42_DEBUG"):
                        print(f"[v42 +{time.monotonic() - online_session._created:5.2f}s] CONNECT sent: "
                              f"suffix={dce.connect_suffix!r} "
                              f"v42_mode={getattr(online_session._v22, 'v42_mode', None)!r} "
                              f"v42_active={online_session._v42_active()} "
                              f"auto_state={getattr(online_session._auto, 'state', None)!r}",
                              file=sys.stderr, flush=True)
                    dce.connect_modulation = online_session.modulation_name
                    dce.link_stats = online_session.link_stats
                    dce.notify_call_progress("CONNECT")
                elif not connect_pending and online_session.link_lost:
                    # The error-controlled link died: drop the call like a
                    # real modem instead of passing on a damaged stream.
                    if os.environ.get("MODEM_DEBUG") == "1":
                        engine = online_session._v22
                        print(f"[modem] link lost: v42_mode={getattr(engine, 'v42_mode', None)!r} "
                              f"v42_lost={getattr(engine, 'v42_lost', None)!r} "
                              f"carrier_down_at={online_session._carrier_down_at!r} "
                              f"last_media_age={time.monotonic() - (online_session._last_rtp_at or online_session._created):.3f}s",
                              file=sys.stderr, flush=True)
                    unified_hangup()
                    dce.notify_call_progress("NO CARRIER")
                elif connect_pending and online_session.failed:
                    # The line never trained: give up like a real modem.
                    connect_pending = False
                    unified_hangup()
                    dce.notify_call_progress("NO CARRIER")
            notifications = dce.take_notifications()
            if notifications:
                term_write(notifications)
            escape_detector.guard_time = max(0.02, dce.registers[12] / 50.0)      # S12 in 1/50 s
            escape_detector.plus_char = dce.registers[2]                          # S2
            if data:
                if dial_abort_applies(active_outbound, online_session, dce.online, connect_pending, iax_dial_pending):
                    # Break before connect: hang up everything in flight and
                    # report NO CARRIER, ready for the next command. Both legs
                    # are torn down unconditionally once committed: the SIP
                    # attempt object lingers after answer (no clearing runs on
                    # connect) while a session negotiates, so either/or would
                    # leave half a call behind to swallow the next command.
                    # Neuter an orphaned SIP attempt's callbacks first, so a
                    # late answer or failure cannot resurrect it after the
                    # result.
                    if active_outbound is not None:
                        call, active_outbound = active_outbound, None
                        call.on_connected = call.on_failed = call.on_hangup = lambda *a: None
                        call.hangup()
                        dce._call_progress_monitor = None
                    if online_session is not None:
                        unified_hangup()
                        connect_pending = False
                    elif iax_transport is not None and iax_dial_pending:
                        iax_transport.hangup()
                        iax_dial_pending = False
                        iax_media_active = False
                    dce.outgoing_pending = False
                    dce.notify_call_progress("NO CARRIER")
                elif data_mode and online_session is not None:
                    passthrough = escape_detector.feed(data, time.monotonic())
                    if passthrough:
                        if os.environ.get("MODEM_DEBUG") == "1":
                            print(f"[dce] terminal transmit bytes={len(passthrough)}", file=sys.stderr, flush=True)
                        if arq_session is not None:
                            arq_session.feed_terminal_bytes(passthrough)
                            framed = arq_session.pump_outgoing(time.monotonic())
                            if framed:
                                online_session.feed_terminal_bytes(framed)
                        else:
                            online_session.feed_terminal_bytes(passthrough)
                        dce.count_chars(sent=len(passthrough))
                    if dce.software_flow_control_enabled():
                        extra = (
                            arq_session.pending_bytes() * 10
                            if arq_session is not None
                            else 0
                        )
                        signal = online_session.flow_control_signal(extra)
                        if signal:
                            term_write(signal)
                else:
                    response = dce.feed(data)
                    if response:
                        term_write(response)
            if (
                data_mode
                and online_session is not None
                and escape_detector.poll(time.monotonic())
            ):
                data_mode = False
                term_write(dce._result("OK"))
            if (
                data_mode
                and online_session is not None
                and online_session.xoff_asserted
                and dce.software_flow_control_enabled()
            ):
                # Re-check even if the DTE sent nothing this cycle: once
                # paused with XOFF it won't send more until it sees XON.
                extra = arq_session.pending_bytes() * 10 if arq_session is not None else 0
                signal = online_session.flow_control_signal(extra)
                if signal:
                    term_write(signal)
            if data_mode and online_session is not None and arq_session is not None:
                # Drives retransmission timeouts and queued acks even when
                # the DTE sends nothing this cycle.
                framed = arq_session.pump_outgoing(time.monotonic())
                if framed:
                    online_session.feed_terminal_bytes(framed)
            if loop_trace is not None:
                control_ended = time.monotonic()
                loop_row = dict(select_started=select_started, control_started=control_started,
                                serial_ended=serial_ended, control_ended=control_ended,
                                data_mode=data_mode, serial_bytes=len(data),
                                media_ready=bool(media_socket and media_socket.fileno() in readable))
                loop_trace.record('main', loop_row)
                loop_trace.threshold('main_control', control_ended-control_started, loop_row)
            if media_socket and media_socket.fileno() in readable:
                try:
                    received_at=time.monotonic()
                    clock_context = None
                    if kernel_timestamps:
                        packet,ancillary,_,rtp_from=media_socket.recvmsg(2048,socket.CMSG_SPACE(16))
                        for level,kind,value in ancillary:
                            if level==socket.SOL_SOCKET and kind==35 and len(value)>=16:
                                sec,nsec=struct.unpack('qq',value[:16])
                                clock_mono = time.monotonic()
                                clock_wall = time.time()
                                received_at=clock_mono-max(0.,clock_wall-sec-nsec/1e9)
                                if online_session is not None and online_session.timing_trace is not None:
                                    clock_context = dict(kernel_seconds=sec, kernel_nanoseconds=nsec,
                                        clock_mono_before=clock_mono, clock_wall=clock_wall,
                                        clock_mono_after=time.monotonic())
                    else:packet,rtp_from=media_socket.recvfrom(2048)
                except BlockingIOError:
                    continue
                if len(packet) >= 12 and packet[0] >> 6 == 2 and packet[1] & 0x7F == 0:
                    payload = packet[12:]
                    if online_session is not None:
                        decoded = online_session.handle_rtp_datagram(rtp_from,packet,received_at=received_at,
                                                                     clock_context=clock_context)
                        deliver_received(decoded)
                    elif dce.outgoing_pending:
                        pcm = struct.pack(
                            f"<{len(payload)}h",
                            *(ulaw_to_linear(sample) for sample in payload),
                        )
                        monitor = getattr(dce, "_call_progress_monitor", None)
                        if monitor is None:
                            monitor = OutgoingCallProgressMonitor(report_outgoing_progress)
                            dce._call_progress_monitor = monitor
                        monitor.feed_pcm(pcm)
            if online_session is not None:
                deliver_received(online_session.drain_rtp_deadline(time.monotonic()))
            if loop_trace is not None:
                loop_ended = time.monotonic()
                loop_trace.threshold('main_active', loop_ended-control_started,
                                     dict(loop_row, ended=loop_ended))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            final_hangup()                      # never leave a call up at the far end (BYE / CANCEL)
        except Exception as exc:                # noqa: BLE001 -- shutting down regardless
            print(f"[modem] hangup on shutdown failed: {exc}", file=sys.stderr, flush=True)
        if registration is not None:
            registration.stop()
        if iax_transport is not None:
            iax_transport.close()
        if sip_server:
            sip_server.close()
        if online_session:
            online_session.stop()
        if media_socket:
            media_socket.close()
        if control is not None: control.close()
        ser.close()


def preimport_v34_modules() -> None:
    """Move known first-call imports off the RTP generation/receive threads."""
    from importlib import import_module
    for name in ('mnp_link', 'v34_preemphasis', 'v34_preemphasis_coefficients',
                 'v34_streaming', 'v42_lapm', 'v42bis'):
        import_module(name)


def main(serial_backend=None, control=None) -> None:
    parser = argparse.ArgumentParser(prog="el-modem", description=f"{PRODUCT} V{VERSION} - open-source software modem ({PROJECT_URL})")
    parser.add_argument("--self-test", action="store_true", help="exercise all offline engine factories without opening network or serial ports")
    parser.add_argument("--self-test-payload", action="store_true", help="also verify exact local PCM payload through the all-in-one profile (no network)")
    parser.add_argument("--link", help="create a symlink to the allocated PTY")
    parser.add_argument("--tcp-listen", metavar="HOST:PORT",
                        help="also serve the modem's serial port over TCP (for serial_bridge.py on the host)")
    parser.add_argument("--serial-tcp", metavar="HOST:PORT",
                        help="use a TCP listener as the serial port instead of a PTY (the only mode on Windows; "
                        "a client disconnecting is a DTR drop)")
    parser.add_argument("--sip-user", help="digest username for outbound calls to a SIP trunk provider; the password is "
                        "read from the EL_MODEM_SIP_PASSWORD environment variable (SOFTMODEM_SIP_PASSWORD also works) (never from the command line)")
    parser.add_argument("--sip-register", metavar="HOST[:PORT]",
                        help="register with this SIP registrar (needs --sip-user) so a provider can deliver incoming calls")
    parser.add_argument("--sip-contact-host", help="address the provider should send calls to (default: --sip-host)")
    parser.add_argument("--profile-file", default=os.environ.get("EL_MODEM_PROFILE") or os.environ.get("SOFTMODEM_PROFILE"),
                        help="where AT&W stores profiles (JSON); default: kept in memory only")
    parser.add_argument("--caller-number", help="number to present as the calling party")
    parser.add_argument("--sip-host", help="enable the incoming SIP user agent")
    parser.add_argument("--sip-port", type=int, default=5062)
    parser.add_argument("--media-host", default="127.0.0.1")
    parser.add_argument("--media-port", type=int, default=16000)
    parser.add_argument(
        "--dial-host",
        help="Asterisk (or other SIP proxy) address used as the ATD originate target",
    )
    parser.add_argument("--dial-port", type=int, default=5060)
    parser.add_argument("--iax-host", default=os.environ.get("EL_MODEM_IAX_HOST"), help="Asterisk IAX2 server (UDP; credentials from --iax-user and EL_MODEM_IAX_PASSWORD)")
    parser.add_argument("--iax-port", type=int, default=int(os.environ.get("EL_MODEM_IAX_PORT", "4569")))
    parser.add_argument("--iax-user", default=os.environ.get("EL_MODEM_IAX_USER"), help="IAX2 endpoint username")
    parser.add_argument(
        "--line-mode",
        choices=["bell103", "v21", "v22bis", "auto", "v32", "v34", "ms", "chain"],
        default="bell103",
        help="bell103: lab Bell 103 (+ lab negotiation); v22bis: our own "
        "V.22/V.22bis/Bell 212A; auto: caller listens to "
        "the answerer and picks V.22 or Bell 103; v32: our own "
        "V.32/V.32bis caller or answerer, 4800-14400 bps; v34: our own V.34 caller or answerer (needs numpy; rate via "
        "V34_RATE); v21: ITU-T V.21 300 bps FSK (the Bell 103 "
        "engine on the V.21 tones); chain: V.8 picks the best of V.34/V.32bis/V.22bis, with a legacy V.22 fallback; ms: the modulation is chosen per call by AT+MS=",
    )
    parser.add_argument(
        "--v22-rate",
        type=int,
        choices=[1200, 2400],
        default=1200,
        help="highest V.22/V.22bis rate to offer (with --line-mode v22bis)",
    )
    parser.add_argument(
        "--v42-frame-size",
        type=int,
        default=32,
        help="LAPM information-field size (N401, 1-128 bytes) to offer under \\N3; "
        "smaller frames survive packet loss better, at ~14%% cost on a clean line",
    )
    parser.add_argument(
        "--no-negotiate",
        action="store_true",
        help="stay on Bell 103 (300 bps) with no modulation negotiation",
    )
    parser.add_argument('--v34-native-trellis',action='store_true',help='use the optional compiled V.34 receiver')
    parser.add_argument('--v34-cancel',action='store_true',help='enable guarded actual-TX V.34 cancellation')
    parser.add_argument('--v34-trn-gradient-auto',action='store_true',help='activate startup derivative tracking on repeated TRN errors after a CRC grace interval')
    parser.add_argument('--v34-trn-gradient',action='store_true',help='enable derivative-based phase-four startup timing only')
    parser.add_argument('--v34-trn-clock',action='store_true',help='enable guarded TRN-only timing and tap handoff')
    parser.add_argument('--v34-b1-noise-predictive',action='store_true',help='enable guarded TRN-trained B1 prediction')
    parser.add_argument('--v34-preimport', action='store_true',
                        help='pre-import V.34/data-link modules before accepting calls (opt-in)')
    parser.add_argument('--gc-pause',action='store_true',help='pause automatic garbage collection for the duration of a call (collect before it starts and when it ends); opt-in')
    parser.add_argument('--metrics-file',help='write final per-call timing and DSP measurements as JSON')
    parser.add_argument('--timing-trace',action='store_true',help='retain bounded packet/clock/event-loop metadata around timing outliers')
    parser.add_argument('--dsp-capture',help='export live V.34 TRN/E/B1 and initial-data sampling state to this prefix')
    parser.add_argument('--capture',help='record both local PCM directions to this prefix')
    parser.add_argument('--capture-seconds',type=float,default=120,help='PCM recording cap per direction (default 120 seconds)')
    args = parser.parse_args()
    for enabled,name in ((args.v34_native_trellis,'V34_NATIVE_TRELLIS'),
            (args.v34_trn_gradient,'V34_TRN_GRADIENT'),(args.v34_trn_clock,'V34_TRN_CLOCK'),(args.v34_cancel,'V34_DUPLEX_CANCEL'),(args.v34_b1_noise_predictive,'V34_B1_NOISE_PREDICTIVE')):
        if enabled:os.environ[name]='1'
    if args.v34_trn_gradient_auto:os.environ['V34_TRN_GRADIENT']='auto'
    if args.gc_pause:os.environ['EL_MODEM_GC_PAUSE']='1'
    if args.self_test or args.self_test_payload:
        from modem_selftest import run as self_test
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            report = self_test(OnlineAudioSession, line_from_ms, AtDce,
                               with_payload=args.self_test_payload)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report['status'] == 'PASS' else 1)
    if args.metrics_file:os.environ['EL_MODEM_METRICS_FILE']=args.metrics_file
    if args.timing_trace:os.environ['EL_MODEM_TIMING_TRACE']='1'
    if args.dsp_capture:os.environ['EL_MODEM_DSP_CAPTURE']=args.dsp_capture
    if args.capture:
        if args.capture_seconds<=0:parser.error('capture seconds must be positive')
        os.environ['MODEM_CAPTURE']=args.capture
        os.environ['MODEM_CAPTURE_SECONDS']=str(args.capture_seconds)
        os.environ['MODEM_CAPTURE_REQUIRED']='1'
        os.environ['MODEM_CAPTURE_TIMELINE']='1'
    if args.dsp_capture or args.capture:
        from pathlib import Path
        import tempfile
        for prefix in (args.dsp_capture,args.capture):
            if prefix:
                directory=Path(prefix).parent
                directory.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryFile(dir=directory) as probe:
                    probe.write(b'capture preflight\n');probe.flush()
        if args.dsp_capture:
            from v34_diagnostic import _install_hooks
            _install_hooks()
            Path(args.dsp_capture+'.armed.json').write_text(json.dumps(dict(
                schema=1,dsp_export_available=True,pcm_capture_required=bool(args.capture),
                pcm_capture_limit_seconds=args.capture_seconds))+'\n')
    if os.environ.get('V34_DATA_TRACE'):
        from v34_data_trace import install as install_data_trace
        install_data_trace()
    from v34_equalizer_backend import warmup as warm_equalizer_backend
    warm_equalizer_backend()  # Fail/prepare explicit opt-in before opening DTE or media.
    if os.environ.get('V34_NATIVE_FEEDBACK')=='1':
        from v34_feedback_native import warmup as warm_feedback
        warm_feedback()
    if (os.environ.get('V34_TRN_GRADIENT') in ('1','auto')
            or os.environ.get('V34_TRN_CLOCK_NATIVE')=='1'
            or os.environ.get('V34_SYMBOL_INTERP_NATIVE')=='1'):
        from v34_interp_native import warmup as warm_interpolation
        warm_interpolation()
    if os.environ.get('V34_NATIVE_TRELLIS')=='1':
        from v34_trellis_native import warmup
        warmup()
    if os.environ.get('V34_B1_NOISE_PREDICTIVE')=='1':
        from v34_noise_kernel import warmup
        warmup()
    if args.v34_preimport or os.environ.get("V34_PREIMPORT") == "1":     # V34_PREIMPORT=1 is the same switch for runners that cannot pass a flag
        preimport_v34_modules()
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # docker stop, kill: clean up
    try:
        run_pty(
            args.link,
            args.sip_host,
            args.sip_port,
            args.media_host,
            args.media_port,
            args.dial_host,
            args.dial_port,
            negotiate=not args.no_negotiate,
            line_mode=args.line_mode,
            v22_rate=args.v22_rate,
            v42_frame_size=args.v42_frame_size,
            tcp_listen=args.tcp_listen,
            sip_user=args.sip_user,
            profile_file=args.profile_file,
            sip_register=args.sip_register,
            sip_contact_host=args.sip_contact_host,
            caller_number=args.caller_number,
            serial_tcp=args.serial_tcp or ("127.0.0.1:9100" if os.name == "nt" else None),
            serial_backend=serial_backend, control=control,
            iax_host=args.iax_host,
            iax_port=args.iax_port,
            iax_user=args.iax_user,
            iax_password=os.environ.get("EL_MODEM_IAX_PASSWORD", ""),
        )
    except FileExistsError as exc:
        parser.error(f"refusing to replace existing path: {exc.filename}")
    except OSError as exc:
        print(f"could not create modem PTY: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
