#!/usr/bin/env python3
"""Linux PTY and a small Hayes-style DCE command interpreter for the lab."""

import argparse
import cmath
import copy
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
import textwrap
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
from g711 import linear_to_ulaw, ulaw_to_linear  # noqa: E402
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

from windows_timer import TimerResolutionRequest  # noqa: E402
from modem_build import build_timestamp  # noqa: E402
from modem_pnp import identification as pnp_identification  # noqa: E402
from modem_diagnostics import finite, line_metrics  # noqa: E402
from modem_help import help_screen  # noqa: E402


MAX_COMMAND_BYTES = 256
PRODUCT = "EL-MODEM"
VERSION = "0.4.0-rc1"
PRODUCT_CODE = "0040"          # four digits: major, minor, patch (V0.4.0)
STORED_NUMBER_SLOTS = 4        # phone-number slots 0-3 (AT&Zn=, ATDSn)
PNP_ID = "ELM0001"            # three-letter vendor tag + number; not registered with any PnP authority
PROJECT_URL = "https://github.com/donfanning/el-modem"

_RUNTIME_CLOCK_INFO = {
    "source": "time.monotonic",
    "implementation": time.get_clock_info("monotonic").implementation,
    "resolution_seconds": time.get_clock_info("monotonic").resolution,
}


def _configure_runtime_clock(*, platform=None, monotonic_info=None, perf_counter_info=None):
    """Use a fine-grained monotonic clock for the 20 ms modem runtime.

    CPython 3.11/3.12 on Windows implements ``time.monotonic`` with
    ``GetTickCount64`` (typically 15.625 ms resolution). ``perf_counter`` is
    monotonic and uses the high-resolution performance counter there. Select
    it only when it is measurably finer; changing the shared ``time`` module
    keeps every dynamically imported modem component on one clock domain.
    """
    global _RUNTIME_CLOCK_INFO
    platform = os.name if platform is None else platform
    monotonic_info = monotonic_info or time.get_clock_info("monotonic")
    perf_counter_info = perf_counter_info or time.get_clock_info("perf_counter")
    if platform == "nt" and perf_counter_info.resolution < monotonic_info.resolution:
        previous_monotonic = time.monotonic
        time.monotonic = time.perf_counter
        # threading imports ``monotonic`` into its private _time alias. If it
        # was imported before this runtime setup, timed Event/Condition waits
        # would otherwise keep using GetTickCount64 while modem deadlines use
        # QueryPerformanceCounter. Keep those timeout calculations aligned.
        threading_clock_updated = (
            getattr(threading, "_time", None) is previous_monotonic
        )
        if threading_clock_updated:
            threading._time = time.perf_counter
        _RUNTIME_CLOCK_INFO = {
            "source": "time.perf_counter",
            "implementation": perf_counter_info.implementation,
            "resolution_seconds": perf_counter_info.resolution,
            "replaced_implementation": monotonic_info.implementation,
            "replaced_resolution_seconds": monotonic_info.resolution,
            "threading_clock_updated": threading_clock_updated,
        }
    else:
        _RUNTIME_CLOCK_INFO = {
            "source": "time.monotonic",
            "implementation": monotonic_info.implementation,
            "resolution_seconds": monotonic_info.resolution,
        }
    return dict(_RUNTIME_CLOCK_INFO)


def _sip_setup_trace(event: str, **fields: object) -> None:
    """Opt-in call-setup diagnostics; never include numbers, headers, or SDP."""
    if os.environ.get("EL_MODEM_SIP_TRACE") != "1":
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"SIPTRACE side=dce event={event} {details}".rstrip(), file=sys.stderr, flush=True)

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
RTP_CLOCK_RATE = 8000
V23_RTP_SAMPLES_PER_FRAME = 80  # 10 ms cadence for V.23 circuit-109 timing
XON = 0x11
XOFF = 0x13
# Watermarks are in queued *bits* (10 bits/char at Bell 103 8N1): ~1s of
# buffering triggers a pause request, draining back to ~1/3s resumes. These
# only matter when a DTE pushes data into the PTY faster than the 300 bps
# line can carry it (e.g. a host serial port or terminal emulator left at a
# much higher baud rate than the negotiated line speed).
FLOW_XOFF_HIGH_WATER_BITS = 300
FLOW_XON_LOW_WATER_BITS = 100
BELL212_DTE_BAUD_MIN = 1170
BELL212_DTE_BAUD_MAX = 1212


def bell212_dte_baud_for_serial(serial) -> int:
    """Return a supported Bell 212A DTE rate, defaulting to 1200.

    Backends without baud metadata retain the historical 1200-baud model.
    Only the narrow 1170..1212 rate mismatch is currently modeled; a
    high-speed buffered DTE is not a physical clock mismatch. PTY termios
    reports the application's nominal setting, not a measured or enforced
    physical clock.
    """
    get_baud = getattr(serial, "get_dte_baud", None)
    try:
        baud = get_baud() if callable(get_baud) else None
    except (OSError, ValueError):
        baud = None
    if isinstance(baud, int) and BELL212_DTE_BAUD_MIN <= baud <= BELL212_DTE_BAUD_MAX:
        return baud
    return 1200


def prepare_v42_break_dte_order(event: dict, serial, undelivered: bytearray) -> bool:
    """Apply the D/S buffer action that can be decided when BRK is indicated.

    D=1 discards data accumulated but not delivered. S=1,D=0 moves native
    serial output still queued in the OS behind the next BREAK. Byte-only
    transports cannot provide that ordering and return False.
    """
    if event.get("destructive"):
        discard = getattr(serial, "discard_output", None)
        try:
            applied = discard is not None and discard()
        except OSError:
            applied = False
        if applied:
            undelivered.clear()
            event["dte_action_prepared"] = True
        return applied
    if event.get("expedited"):
        reorder = getattr(serial, "place_break_before_pending_output", None)
        try:
            applied = reorder is not None and reorder()
        except OSError:
            applied = False
        if applied:
            event["dte_action_prepared"] = True
        return applied
    return True


def v42_break_can_be_delivered(event: dict, serial, undelivered: bytearray) -> bool:
    """Check that a non-destructive in-sequence BREAK follows prior DTE data."""
    if event.get("dte_action_failed"):
        return False
    if event.get("destructive") or event.get("expedited"):
        return bool(event.get("dte_action_prepared"))
    pending = getattr(serial, "output_pending", None)
    try:
        queued = pending() if pending is not None else None
    except OSError:
        queued = None
    return queued == 0 and not undelivered


def v42_break_options_for_k(value: int) -> tuple[bool, bool] | None:
    """Map Hayes ``\\K`` to V.42 (destructive, expedited) BREAK options."""
    if value in (0, 2, 4):
        return None
    if value == 1:
        return True, True
    if value == 3:
        return False, True
    if value == 5:
        return False, False
    raise ValueError("break control must be 0..5")


def v42_manual_break_options_for_k(value: int) -> tuple[bool, bool]:
    """Map Hayes ``\\B`` handling (destructive, expedited) for an online link."""
    if value in (0, 1):
        return True, True
    if value in (2, 3):
        return False, True
    if value in (4, 5):
        return False, False
    raise ValueError("break control must be 0..5")


class V42BreakRequestQueue:
    """Bound and serialize local DTE BREAK requests on one LAPM link."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = max(1, int(capacity))
        self.pending: list[tuple[int | None, bool, bool]] = []
        self.in_flight = False

    @property
    def available(self) -> int:
        return max(0, self.capacity - len(self.pending) - int(self.in_flight))

    def enqueue(self, *, length_ms: int | None = None,
                destructive: bool, expedited: bool) -> bool:
        if self.available <= 0:
            return False
        self.pending.append((None if length_ms is None else int(length_ms),
                             bool(destructive), bool(expedited)))
        return True

    def service(self, request: Callable[..., bool]) -> bool:
        if self.in_flight or not self.pending:
            return False
        length_ms, destructive, expedited = self.pending[0]
        if not request(length_ms=length_ms, destructive=destructive,
                       expedited=expedited):
            return False
        self.pending.pop(0)
        self.in_flight = True
        return True

    def complete(self) -> bool:
        if not self.in_flight:
            return False
        self.in_flight = False
        return True

    def reset(self) -> None:
        self.pending.clear()
        self.in_flight = False


def v34_rx_timing_slips(engine) -> dict | None:
    """Snapshot timing slips and optional sampled equalizer-stage timings."""
    by_core = {}
    step_profile_by_core = {}
    slip_events_by_core = {}
    timing_state_by_core = {}
    timing_anomalies_by_core = {}
    timing_low_power_by_core = {}
    for name in ("caller", "answerer"):
        core = getattr(engine, name, None)
        receiver = getattr(core, "_p4rcv", None) if core is not None else None
        equalizer = getattr(receiver, "eq", None) if receiver is not None else None
        slips = getattr(equalizer, "slips", None) if equalizer is not None else None
        if isinstance(slips, int) and not isinstance(slips, bool):
            by_core[name] = slips
        profile = getattr(equalizer, "_step_profile", None) if equalizer is not None else None
        if profile is not None and profile.count:
            step_profile_by_core[name] = profile.report()
        trace = getattr(equalizer, "_timing_slip_trace", None) if equalizer is not None else None
        if trace is not None:
            slip_events_by_core[name] = list(trace)
        state_trace = getattr(equalizer, "_timing_state_trace", None) if equalizer is not None else None
        if state_trace is not None:
            timing_state_by_core[name] = list(state_trace)
        anomalies = getattr(equalizer, "_timing_anomaly_trace", None) if equalizer is not None else None
        if anomalies is not None:
            timing_anomalies_by_core[name] = list(anomalies)
        low_power = getattr(equalizer, "_timing_low_power_trace", None) if equalizer is not None else None
        if low_power is not None:
            timing_low_power_by_core[name] = list(low_power)
    if not by_core:
        return None
    result = {"by_core": by_core, "total": sum(by_core.values())}
    if step_profile_by_core:
        result['symbol_eq_step_profile_by_core'] = step_profile_by_core
    if slip_events_by_core:
        result['slip_events_by_core'] = slip_events_by_core
    if timing_state_by_core:
        result['timing_state_samples_by_core'] = timing_state_by_core
    if timing_anomalies_by_core:
        result['timing_anomalies_by_core'] = timing_anomalies_by_core
    if timing_low_power_by_core:
        result['timing_low_power_events_by_core'] = timing_low_power_by_core
    return result


# Modulation switch guard times. After the CONFIRM exchange the sender lets
# the final old-mode frame drain, then goes silent for TX_GUARD_FRAMES (20 ms
# each) before the new carrier starts. The receive deaf interval covers the
# tail of the old-mode CONFIRM; Bell 212 then waits for the guard silence before
# acquiring its new carrier.
TX_GUARD_FRAMES = 10
RX_DEAF_SAMPLES = 480


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


def _env_enabled(name: str) -> bool:
    """Treat common explicit false strings as disabled environment flags."""
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"}


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

    def set_tx_media_deadline(self, deadline: float) -> None:
        setter = getattr(self._modem, "set_tx_media_deadline", None)
        if setter is not None:
            setter(deadline)


class _V22Receiver:
    """Presents a V22bisModem's receive side as a streaming receiver."""

    def __init__(self, modem):
        self._modem = modem

    def feed_pcm(self, pcm: bytes) -> bytes:
        return self._modem.feed_pcm(pcm)

    def set_rx_media_deadline(self, deadline: float) -> None:
        setter = getattr(self._modem, "set_rx_media_deadline", None)
        if setter is not None:
            setter(deadline)

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
        v23_rate: int = 1200,
        v42: bool = False,
        v42_frame_size: int = 32,
        v42bis: bool = False,
        v42_xid_fallback: bool = True,
        mnp: bool = False,
        carrier_loss_seconds: float = CARRIER_LOSS_SECONDS,
        v32_rates: tuple[int, ...] = (9600, 7200, 4800),
        bell212_char_bits: int = 10,
        bell212_parity: str = "E",
        bell212_dte_baud: int = 1200,
        bell212_leased_line: bool = False,
        chain: dict | None = None,
        calling_tone: bool = False,
        v25_call_sequence: bool | None = None,
        v23_backward_channel: bool = True,
        guard_tone: int = 0,
        v22_min_rate: int | None = None,
        v34_rate: int | None = None,
        v34_min_rate: int = 2400,
    ):
        self._sock = sock
        self._remote_addr = remote_addr
        # Start at SDP; qualify a consecutive RTP source before learning a
        # symmetric destination. Pin the address/port for this call; permit an
        # SSRC restart from that same endpoint only after fresh packet probation.
        # The probation queue preserves startup samples rather than dropping them.
        self._send_addr = remote_addr
        self._rtp_peer = None
        self._rtp_candidate = []
        self._tx_role = tx_role
        self._rx_role = rx_role
        self._line_mode = line_mode
        # V.23's forward received-line-signal detector must release in
        # 5–15 ms. Use 10-ms RTP packets so packet delivery does not impose a
        # 20-ms minimum observation delay. Other modes retain 20-ms packets.
        self._rtp_frame_samples = (
            V23_RTP_SAMPLES_PER_FRAME if line_mode == "v23" else RTP_SAMPLES_PER_FRAME
        )
        self._rtp_frame_seconds = self._rtp_frame_samples / RTP_CLOCK_RATE
        self._v22 = None
        self._v25_caller = None
        self._v25_answer = False
        self._bell212_char_bits = bell212_char_bits if line_mode == "bell212a" else 10
        self._bell212_dte_baud = bell212_dte_baud if line_mode == "bell212a" else 1200
        if bell212_leased_line and line_mode != "bell212a":
            raise ValueError("Bell 212A leased-line startup requires the Bell 212A line mode")
        self._bell212_leased_line = bool(bell212_leased_line)
        # Both directions of the linear audio, whatever engine we end up loading: the only way
        # to tell after the fact why a real-PSTN call failed (pcm_capture.py).
        self._capture = capture_from_env(f"{tx_role}-{line_mode}", f"peer={remote_addr}")
        self._last_pcm = b""  # most recent received frame, for concealment
        self._last_rx_payload_samples: int | None = None
        self._v23_detector_transitions: list[dict] = []
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
        if line_mode == "v23" and self._want_v42:
            raise ValueError("V.23 currently supports raw asymmetric channels; V.42/MNP is not implemented")
        self._v42_started = False
        self._v42_frame_size = v42_frame_size
        self._v42bis = v42bis
        self._v42_xid_fallback = bool(v42_xid_fallback)
        self._carrier_loss_seconds = carrier_loss_seconds
        self._was_up = False
        self._carrier_down_at = None
        self._last_rtp_at = None  # monotonic time of the last received RTP packet (None until the first)
        # Debug capture of the V.42 detection phase (see v42_detect.py): send
        # ODP right after CONNECT and log what the peer sends back.
        self._v42_probe = _env_enabled("MODEM_V42_PROBE")
        self._odp_sent_at = None
        self._probe_rx = bytearray()
        self._probe_logged = 0
        self._adp_verdict = None
        self._created = time.monotonic()
        self.debug = _env_enabled("MODEM_DEBUG")
        self._auto = None
        if line_mode == "chain":
            # V.8 first, then the best common modulation, else legacy V.22 (autochain.py).
            from autochain import V8Chain  # noqa: PLC0415

            self._v22 = V8Chain(tx_role == "originate", debug=self.debug, **(chain or {}))
            self.transmitter = _V22Transmitter(self._v22)
            self.receiver = _V22Receiver(self._v22)
            self._dsp_capture_pending = _env_enabled('EL_MODEM_DSP_CAPTURE')
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
            self._v22 = engine(fs=8000, data_rate=v34_rate, min_data_rate=v34_min_rate, debug=self.debug)
            from v34_diagnostic import configured as diagnostic_capture
            self._dsp_capture = diagnostic_capture(self._v22)
            self.transmitter = _V22Transmitter(self._v22)
            self.receiver = _V22Receiver(self._v22)
            negotiate = False
        elif line_mode in ("v22bis", "bell212a", "auto"):
            # Our own V.22/V.22bis engine (v22.py): it runs the real
            # handshake itself, so no lab negotiation is used.
            from v22 import V22Modem  # noqa: PLC0415

            self._v22 = V22Modem(
                calling_party=(tx_role == "originate"),
                bit_rate=v22_rate,
                min_rate=v22_min_rate,
                profile="bell212a" if line_mode == "bell212a" else ("v22" if v22_rate < 2400 else "v22bis"),
                guard_tone=guard_tone,
                bell212_char_bits=bell212_char_bits,
                bell212_parity=bell212_parity,
                bell212_dte_baud=self._bell212_dte_baud,
                bell212_leased_line=self._bell212_leased_line,
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
        elif line_mode == "v23":
            from v23 import V23Modem  # noqa: PLC0415

            self._v22 = V23Modem(
                calling_party=(tx_role == "originate"), forward_baud=v23_rate,
                backward_channel=v23_backward_channel,
            )
            self.transmitter = _V22Transmitter(self._v22)
            self.receiver = _V22Receiver(self._v22)
            negotiate = False
        else:
            std = "v21" if line_mode == "v21" else "bell103"
            # ITU-T V.21 answerers transmit a continuous 1800 Hz guard tone to
            # disable echo suppressors (matters on international circuits, not
            # domestic digital trunks). Bell 103 has no guard tone.
            guard = (1800.0, 6000.0) if (std == "v21" and tx_role == "answer") else (0.0, 0.0)
            use_v25_call = (
                std == "v21" and tx_role == "originate"
                if v25_call_sequence is None else bool(v25_call_sequence)
            )
            if use_v25_call:
                from v25 import V25CallerProcedure  # noqa: PLC0415
                self._v25_caller = V25CallerProcedure()
            self._v25_answer = std == "v21" and tx_role == "answer"
            self.transmitter = Bell103StreamTransmitter(tx_role, standard=std,
                                                        guard_hz=guard[0], guard_amp=guard[1],
                                                        v25_answer=(std == "v21" and tx_role == "answer"),
                                                        v25_call_sequence=use_v25_call,
                                                        calling_tone=(3 if (calling_tone and not use_v25_call
                                                                            and tx_role == "originate"
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
        if self._v25_caller is not None:
            # The DTE may send before ANS is complete; buffer it until the
            # interrupted answer tone and its trailing silence have ended.
            self._tx_switched = False
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
        self._timer_resolution_request = TimerResolutionRequest.from_environment()
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
        if self._timer_resolution_request is not None:
            self._timer_resolution_request.start()
        try:
            self._gc_pause_begin()
            self._thread.start()
            if self._tx_watchdog is not None:self._tx_watchdog.start()
        except BaseException:
            self._stop.set()
            if self._thread.is_alive():self._thread.join(timeout=1)
            self._gc_pause_end()
            if self._timer_resolution_request is not None:
                self._timer_resolution_request.close()
            raise

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
        v25_caller = getattr(self, "_v25_caller", None)
        if v25_caller is not None and v25_caller.state != "complete":
            return False
        if (getattr(self, "_v25_answer", False)
                and self.transmitter.prologue_remaining):
            return False
        if self._strict_xid_failed():
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

    def dte_dcd_state(self, online: bool, force_on: bool = False, output_pending: bool = False) -> bool:
        """Return the DCD level exposed to a connected DTE.

        V.23 has an explicit receive-carrier detector (circuit 109 or 122,
        depending on the receiving channel). Keep that detector visible on
        DCD when &C1 is selected instead of holding DCD high for the whole
        SIP call. V.250 §6.2.8 also requires DCD to remain on until previously
        received bytes have been delivered to the DTE, so a falling carrier
        cannot clear an asserted DCD while output is pending.
        """
        engine = getattr(self._v22, "engine", None) or self._v22
        if type(engine).__name__ == "V23Modem":
            carrier = bool(engine.connected)
        else:
            carrier = bool(online)
        if force_on or carrier:
            level = True
        elif getattr(self, "_dte_dcd_asserted", False) and output_pending:
            level = True
        else:
            level = False
        self._dte_dcd_asserted = level
        return level

    def _v42_active(self) -> bool:
        """Whether V.42 applies to this session (a V.22 line, V.42 wanted)."""
        return (
            self._want_v42
            and self._v22 is not None
            and getattr(self._v22, "supports_v42", True)
            and not getattr(self._v22, "bell103_fallback", False)
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
        if self._strict_xid_failed():
            return True
        v25_caller = getattr(self, "_v25_caller", None)
        if v25_caller is not None and v25_caller.state == "failed":
            return True
        return (
            self._v22 is not None
            and not self.negotiation_complete
            and time.monotonic() - self._created > V22_CONNECT_TIMEOUT_SECONDS
        )

    def _strict_xid_failed(self) -> bool:
        link = getattr(self._v22, "_v42", None) if self._v22 is not None else None
        # Any raw fallback after XID was sent is unsafe under strict policy,
        # whether the LAPM link ended in `down` or another layer failure path
        # left it idle/released. A peer that never offered V.42 has sent us no
        # XID and retains the ordinary raw fallback.
        return (self._want_v42 and not self._v42_xid_fallback
                and getattr(self._v22, "v42_mode", None) == "failed"
                and getattr(link, "_xid_sent", 0) > 0)

    @property
    def modulation_name(self) -> str:
        """Which modulation carries the call, for ATI6/ATI11."""
        if getattr(self, "_line_mode", None) == "v21":
            return "V.21"
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
        if name == "V23Modem":
            return "V.23"
        if name == "V22Modem" and getattr(eng, "bell103_fallback", False):
            return "BELL 103 (212A FALLBACK)"
        if name == "V22Modem":
            profile = getattr(eng, "profile", None)
            return {"v22": "V.22", "v22bis": "V.22BIS", "bell212a": "BELL 212A"}.get(
                profile, "V.22BIS" if eng.bit_rate >= 2400 else "V.22"
            )
        return name.upper()

    def link_stats(self) -> dict:
        eng = self._v22
        eng = getattr(eng, "engine", None) or eng
        if getattr(self, '_auto', None) is not None and self._auto.state != 'v22':
            eng = None  # Do not report a dormant V.22 engine on another call mode.

        stats: dict = {"error_control": None, "compression": None,
                       "rtp": dict(self._rtp_counts), "realtime": self.realtime.report()}
        stats["rtp_packetization_ms"] = self._rtp_frame_seconds * 1000
        stats["runtime_clock"] = dict(_RUNTIME_CLOCK_INFO)
        if self._v23_detector_transitions:
            stats["v23_detector_transitions"] = list(self._v23_detector_transitions)
        if self._timer_resolution_request is not None:
            stats['windows_timer_resolution'] = self._timer_resolution_request.report()
        if self.timing_trace is not None:
            stats['timing_trace'] = self.timing_trace.report()
        if eng is None:
            return stats
        timing_slips = v34_rx_timing_slips(eng)
        if timing_slips is not None:
            stats["v34_rx_timing_slips"] = timing_slips
        stats.update(line_metrics(eng))
        if getattr(self, '_gc_pause', None) is not None:stats['gc_pause']=dict(self._gc_pause)
        recorder=getattr(self,'_capture',None)
        recording=recorder.report() if recorder is not None else getattr(self,'_capture_stats',None)
        if recording is not None:stats['pcm_capture']=recording
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
        stats["v42_mode"] = mode
        # error_control is the negotiated call result (the protocol reported
        # by CONNECT), not merely whether the link object is currently in its
        # connected state. Preserve it after a post-connect C-ERROR or release;
        # v42_session.final_mode reports the final live state separately.
        v42_ever_established = bool(getattr(eng, "_v42_ever_established", mode == "lapm"))
        stats["error_control"] = (
            ("MNP" if self._want_mnp else "V.42 LAPM")
            if mode == "lapm" or v42_ever_established else "NONE"
        )
        stats["compression"] = "V.42BIS" if getattr(eng, "v42_compression", None) else "NONE"
        if type(eng).__name__ == "V23Modem":
            stats["v23_channels"] = dict(
                forward_baud=eng.forward_baud, tx_baud=eng.tx_baud,
                rx_baud=eng.rx_baud,
                backward_channel=("enabled" if eng.backward_channel_enabled else "disabled"),
            )
        link = getattr(eng, "_v42", None)
        if link is not None:
            stats["v42_session"] = {
                "requested": "MNP" if self._want_mnp else ("V.42 LAPM" if self._want_v42 else None),
                "ever_established": v42_ever_established,
                "final_mode": mode,
                "final_state": getattr(link, "state", None),
            }
        recovery_stats = getattr(eng, "_v42_recovery_stats", None)
        if recovery_stats is not None:
            stats["v42_recovery"] = dict(recovery_stats)
        # Keep a compact snapshot of the ODP/ADP detector.  The detailed
        # V42_DEBUG trace is too expensive on a real-time call; these bounded
        # counters and the last marker characters are enough to tell whether
        # detection was never armed, rejected the received ADP, or timed out.
        if link is not None and hasattr(eng, "_v42_bits"):
            detector = getattr(eng, "_v42_adp_detector", None)
            stats["v42_detection"] = {
                "calling": getattr(eng, "_v42_calling", None),
                "mode": mode,
                "bits_emitted": getattr(eng, "_v42_bits", None),
                "deadline_bits": getattr(eng, "_v42_detect_deadline", None),
                "adp_armed": getattr(eng, "_v42_adp_armed", None),
                "adp_detector_result": getattr(detector, "result", None),
                "adp_marker_tail_hex": bytes(getattr(detector, "_buf", b""))[-16:].hex(),
                "adp_gap_tail_bits": list(getattr(detector, "_gaps", ()))[-15:],
                "link_phase": getattr(link, "_phase", None),
                "link_bits_sent": getattr(link, "bits_sent", None),
                "link_detect_bits": getattr(link, "_detect_bit_count", None),
                "link_marker_tail_hex": bytes(getattr(link, "_chars", ()))[-16:].hex(),
                "link_gap_tail_bits": list(getattr(link, "_char_gaps", ()))[-15:],
                "answerer_adp_count": getattr(link, "_adp_count", None),
                "answerer_adp_started": getattr(link, "_answer_adp", None),
                "adp_seen": getattr(link, "adp_seen", None),
            }
        if link is not None:
            stats["v42_state"] = getattr(link, "state", None)
            stats["v42_parameters"] = {
                "n401_tx": getattr(link, "tx_n401", None),
                "n401_rx": getattr(link, "rx_n401", None),
                "k_tx": getattr(link, "tx_k", None),
                "k_rx": getattr(link, "rx_k", None),
            }
            events = getattr(link, "events", None)
            if events:
                stats["v42_events"] = list(events[-32:])
        trace = getattr(eng, "v42_adp_trace", None)
        if trace is not None:
            trace = trace()
            if trace is not None:
                stats["v42_adp_detector_trace"] = trace
        blocks = getattr(link, "stats", None)
        if blocks:
            stats["blocks"] = dict(blocks)
        lapm_trace = getattr(link, "bad_frame_trace_report", None)
        if callable(lapm_trace):
            report = lapm_trace()
            if report is not None:
                stats["v42_lapm_bad_frame_trace"] = report
        for part in ("caller", "answerer"):
            core=getattr(eng,part,None)
            baud = getattr(core, "call_baud", None)
            if baud:
                stats["symbol_rate"] = baud
            if core is not None:
                from dataclasses import asdict
                negotiated=getattr(core,'negotiated',None)
                if negotiated is not None:stats['negotiated']=asdict(negotiated)
                connected_rate=getattr(core,'_connected_rate',None)
                if connected_rate is not None:
                    exchange=getattr(connected_rate,'exchange',None)
                    active_origin=(None if exchange is None else
                                   ('local' if exchange.initiating else 'peer'))
                    stats.setdefault('v34_connected_rate', {})[part]=dict(
                        phase=connected_rate.phase,
                        active_origin=active_origin,
                        completed=[{key:event[key] for key in
                                    ('tx_rate','rx_rate','origin','completed_monotonic','b1_symbol')
                                    if key in event}
                                   for event in connected_rate.completed])
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
                    data_trace=getattr(core,'_data_state_trace',None)
                    if data_trace is not None:stats['v34_data_state_trace']=data_trace.report()
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

    def request_v42_release(self) -> bool:
        """Start an orderly LAPM release when this session has an active V.42 link."""
        owner = getattr(self, '_online_control', None)
        if owner is None:
            return False
        engine = owner.engine()
        release = getattr(engine, 'v42_release', None)
        return bool(release is not None and release())

    def request_v42_parameters(self, *, tx_n401: int, rx_n401: int,
                                tx_k: int, rx_k: int) -> bool:
        """Issue the control-function L-SETPARM request on an active LAPM link."""
        owner = getattr(self, '_online_control', None)
        engine = owner.engine() if owner is not None else None
        request = getattr(engine, 'v42_renegotiate_parameters', None)
        return bool(request is not None and request(
            tx_n401=tx_n401, rx_n401=rx_n401, tx_k=tx_k, rx_k=rx_k
        ))

    def request_v42_break(self, *, length_ms: int | None = None,
                          destructive: bool = False, expedited: bool = False) -> bool:
        """Request a DTE BREAK transfer through the active LAPM control function."""
        owner = getattr(self, '_online_control', None)
        engine = owner.engine() if owner is not None else None
        request = getattr(engine, 'v42_request_break', None)
        return bool(request is not None and request(length_ms=length_ms,
                                                    destructive=destructive,
                                                    expedited=expedited))

    def read_v42_break_events(self) -> list[dict]:
        """Read deferred L-SIGNAL events for the DCE/terminal bridge."""
        owner = getattr(self, '_online_control', None)
        engine = owner.engine() if owner is not None else None
        read = getattr(engine, 'v42_read_break_events', None)
        return read() if read is not None else []

    def acknowledge_v42_break(self) -> bool:
        """Confirm one remote L-SIGNAL after its DTE action has been applied."""
        owner = getattr(self, '_online_control', None)
        engine = owner.engine() if owner is not None else None
        acknowledge = getattr(engine, 'v42_acknowledge_break', None)
        return bool(acknowledge is not None and acknowledge())

    def v42_break_available(self) -> bool:
        owner = getattr(self, '_online_control', None)
        engine = owner.engine() if owner is not None else None
        return getattr(engine, 'v42_mode', None) == 'lapm'

    def send_bell212_break(self) -> bool:
        """Transmit a raw Bell 212A Mode-B BREAK on the active modem."""
        engine = getattr(self, '_v22', None)
        send = getattr(engine, 'bell212_feed_break', None)
        return bool(send is not None and send())

    def read_bell212_break_events(self) -> list[dict]:
        """Drain raw Bell 212A Mode-B BREAK detections from the modem engine."""
        engine = getattr(self, '_v22', None)
        read = getattr(engine, 'bell212_read_break_events', None)
        return read() if read is not None else []

    def v42_release_complete(self) -> bool:
        """Whether LAPM has completed its local release and returned to raw mode."""
        owner = getattr(self, '_online_control', None)
        if owner is None:
            return True
        engine = owner.engine()
        return getattr(engine, 'v42_mode', 'off') != 'lapm'

    def v42_release_timeout(self) -> float:
        """Bound the wait through T401/N400 retries, with a small scheduling margin."""
        owner = getattr(self, '_online_control', None)
        engine = owner.engine() if owner is not None else None
        link = getattr(engine, '_v42', None)
        rate = max(1, int(getattr(link, 'rate', self.line_speed)))
        t401_bits = max(0, int(getattr(link, '_t401_bits', 0)))
        retries = max(0, int(getattr(link, 'N400', 10)))
        if not t401_bits:
            return 2.0
        # This deadline is a final DCE safety bound, not a shorter policy than
        # LAPM's retry procedure. At low negotiated rates, maximum N401 makes
        # the required (N400 + 1) T401 interval exceed 20 seconds.
        return (t401_bits / rate) * (retries + 1) + 0.5

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
        v25_caller = getattr(self, "_v25_caller", None)
        if v25_caller is not None:
            # V25CallerProcedure timestamps ANS transitions with
            # time.monotonic() while processing received PCM.  Keep its
            # watchdog in that same clock domain; `now` may be a simulated
            # modem-time value in offline PCM replay.
            v25_caller.poll()
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
                xid_fallback=self._v42_xid_fallback,
                compression_recovery=(
                    self._v42bis and os.environ.get("V42_CERROR_RECOVERY", "0") == "1"
                ),
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
                self.receiver = Bell212Receiver(
                    self._rx_role, require_silence_before_carrier=True
                )
                self._rx_deaf_samples = RX_DEAF_SAMPLES
            self._rx_switched = True
        if self._pending_tx_mode is None and not self._tx_switched:
            self._pending_tx_mode = mode

    def _release_held(self) -> None:
        # Flip the mode flag under the same lock feed_terminal_bytes uses.
        with self._held_lock:
            self._tx_switched = True
        self._flush_held_terminal_bytes()

    def _dte_transmit_ready(self) -> bool:
        ready = getattr(self._v22, "dte_transmit_ready", None)
        return True if ready is None else bool(ready)

    def _flush_held_terminal_bytes(self) -> None:
        # V.22bis circuit 106 remains OFF during retraining. Retain terminal
        # writes until the transmit training interval has ended and receive
        # requalification is complete, then release them in order.
        if not self._tx_switched or not self._dte_transmit_ready():
            return
        with self._held_lock:
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
        if self._timer_resolution_request is not None:
            self._timer_resolution_request.close()
        metrics_path=os.environ.get('EL_MODEM_METRICS_FILE')
        if metrics_path:
            import json
            from pathlib import Path
            Path(metrics_path).write_text(json.dumps(self.link_stats(),indent=2)+'\n')

    def feed_terminal_bytes(self, data: bytes) -> None:
        # While modulation is being negotiated the Bell 103 channel belongs
        # to the negotiation frames; hold DTE data until the switch is done.
        if self._bell212_char_bits == 9:
            data = bytes(value & 0x7f for value in data)
        with self._held_lock:
            if not self._tx_switched or not self._dte_transmit_ready():
                self._held.extend(data)
                return
        self.transmitter.feed_bytes(data)

    def flow_control_signal(self, extra_pending_bits: int = 0) -> bytes | None:
        """Checks the queued-but-not-yet-transmitted bit count against the
        XOFF/XON watermarks and returns a single control byte to send to the
        DTE if a transition is warranted, else ``None``. Only meaningful
        when the caller has confirmed that Hayes ``&K`` or V.250 ``+IFC``
        selects software flow control. ``extra_pending_bits`` lets
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

    def set_dte_receive_busy(self, busy: bool) -> None:
        """Propagate a paused DTE receive path to LAPM as RNR/RR."""
        modem = self._v22
        set_busy = getattr(modem, "v42_set_local_busy", None)
        if set_busy is not None:
            set_busy(busy)

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
        rx_media_deadline = self._rx_context.get("packet_playout_deadline_monotonic")
        set_rx_media_deadline = getattr(self.receiver, "set_rx_media_deadline", None)
        if rx_media_deadline is not None and set_rx_media_deadline is not None:
            input_samples = len(pcm) // 2 + skipped
            packet_samples = int(self._rx_context.get("payload_samples", input_samples))
            if self._rx_context.get("pcm_source") == "concealment":
                # A gap-fill block ends at the current packet's playout time.
                media_start = float(rx_media_deadline) - input_samples / 8000 + skipped / 8000
            else:
                # _drain_rtp can join several contiguous packets. Its context
                # describes the last packet, so back up over preceding PCM.
                media_start = (float(rx_media_deadline)
                               - max(0, input_samples - packet_samples) / 8000
                               + skipped / 8000)
            set_rx_media_deadline(media_start)
        v25_caller = getattr(self, "_v25_caller", None)
        if v25_caller is not None and v25_caller.state not in ("complete", "failed"):
            previous = v25_caller.state
            state = v25_caller.feed_pcm(pcm)
            if previous == "waiting_answer" and state == "answer_tone":
                self.transmitter.stop_calling_tone()
            elif state == "complete":
                self.transmitter.release_call_sequence()
                self._release_held()
        started=time.monotonic()

        rx_thread_started=(time.thread_time()
                           if getattr(self.realtime, 'thread_timing', False) is True else None)
        # Wall time alone cannot tell whether a long receive interval was DSP
        # work or a descheduled Python worker. Capture thread CPU only when
        # the opt-in trace is active.
        thread_cpu_started=time.thread_time() if self.timing_trace is not None else None

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
        v23_engine = self._v22 if type(self._v22).__name__ == "V23Modem" else None
        v23_samples_before = getattr(v23_engine, "_rx_samples", 0)
        decoded = (diagnostic.feed(pcm,self.receiver) if diagnostic is not None
                   else self.receiver.feed_pcm(pcm))
        thread_cpu_seconds=(time.thread_time()-thread_cpu_started
                            if thread_cpu_started is not None else None)
        ended = time.monotonic()
        if v23_engine is not None:
            for transition in v23_engine.take_carrier_transitions():
                row = dict(transition)
                row.update(
                    sample_offset_in_packet=transition["sample_index"] - v23_samples_before,
                    rtp_timestamp=self._rx_context.get("timestamp"),
                    pcm_source=self._rx_context.get("pcm_source", "unknown"),
                    packet_received_monotonic=self._rx_arrival,
                    state_processed_monotonic=ended,
                )
                if self._rx_arrival is not None:
                    row["packet_to_state_ms"] = max(0.0, (ended - self._rx_arrival) * 1000.0)
                self._v23_detector_transitions.append(row)
            if len(self._v23_detector_transitions) > 64:
                del self._v23_detector_transitions[:-64]
        self.realtime.add('rx_seconds',ended-started,data=data_phase)
        if rx_thread_started is not None:
            self.realtime.add('rx_thread_cpu_seconds',time.thread_time()-rx_thread_started,
                              data=data_phase)
        if self.timing_trace is not None:
            row = dict(self._rx_context, started=started, ended=ended,
                       samples=len(pcm)//2, data=data_phase, queue_age=age)
            row['thread_cpu_seconds']=thread_cpu_seconds
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
        if self._bell212_char_bits == 9:
            decoded = bytes(value & 0x7f for value in decoded)
        return decoded

    def handle_rtp_packet(self, packet: bytes, *, received_at=None, clock_context=None) -> bytes:
        """Order V.34 packets within a 40 ms window before gap concealment."""
        if len(packet) < 12 or packet[0] >> 6 != 2 or packet[1] & 0x7F != 0:
            return b""
        timestamp = struct.unpack_from("!I", packet, 4)[0]
        self._rtp_counts['received_packets']+=1
        self._last_rtp_at = time.monotonic()
        ssrc = struct.unpack_from("!I", packet, 8)[0]
        context = dict(clock_context or {})
        context.update(sequence=struct.unpack_from('!H', packet, 2)[0],
                       timestamp=timestamp, ssrc=ssrc,
                       payload_samples=len(packet)-12, received_at=received_at)
        if self.timing_trace is not None:
            context["dequeued"] = time.monotonic()
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
            self._last_rx_payload_samples = None
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
            self._rx_context.update(
                packet_release_monotonic=time.monotonic(),
                packet_playout_deadline_monotonic=self._rx_arrival,
                jitter_buffer_samples=0,
            )
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
        # Keep packet identity and arrival metadata through the reorder/jitter
        # buffer even when the optional timing trace is disabled. PCM capture
        # needs these fields to relate an RX sample offset to RTP time.
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
            self._rx_context.update(
                packet_release_monotonic=time.monotonic(),
                packet_playout_deadline_monotonic=(
                    self._rx_arrival + self._rtp_jitter_samples / 8000
                ),
                jitter_buffer_samples=self._rtp_jitter_samples,
            )
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
            # A peer may use a different packetization interval than our TX.
            # Compare timestamp spacing with its previous payload duration.
            expected = self._last_rx_payload_samples or self._rtp_frame_samples
            if self.timing_trace is not None and gap != expected:
                self.timing_trace.event('rtp_decode_step', gap,
                                       dict(self._rx_context, timestamp_step=gap))
            if self.debug and gap != expected:
                signed_gap = gap if gap < 0x80000000 else gap - 0x100000000
                print(f"[rtp] timestamp step {signed_gap} samples "
                      f"(expected {expected})", file=sys.stderr, flush=True)
            missing = gap - expected
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
        self._last_rx_payload_samples = len(packet) - 12
        self._rtp_counts['decoded_packets']+=1
        self._rx_context['pcm_source'] = 'packet'
        return filled + self.handle_rtp_payload(packet[12:])

    def note_rtp_source(self, addr, packet, received_at=None, clock_context=None):
        """Admit PCMU from a qualified address/SSRC, returning buffered packets.

        Three consecutive sequence numbers from one endpoint within one second
        establish the peer. After that, a different address/port is rejected;
        a new SSRC from the established address/port must pass the same
        probation before its audio is admitted. This permits legitimate RTP
        source restarts without allowing an SSRC change to redirect the
        transmitter. This is source probation, not authentication: deployments
        must still restrict reachable RTP ports. Each returned tuple retains
        its original arrival time/clock metadata.
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
            if peer[0] != self._rtp_peer[0]:
                return []
            if peer == self._rtp_peer:
                return [row]
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

    def _next_frame_samples(self, media_deadline: float | None = None) -> list[int]:
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
            return [0] * self._rtp_frame_samples
        if media_deadline is not None:
            set_tx_media_deadline = getattr(self.transmitter, "set_tx_media_deadline", None)
            if set_tx_media_deadline is not None:
                set_tx_media_deadline(media_deadline)
        if getattr(self, '_online_control', None) is not None:
            with self._online_control.lock:
                samples = self.transmitter.samples(self._rtp_frame_samples)
        else:
            samples = self.transmitter.samples(self._rtp_frame_samples)
        self._flush_held_terminal_bytes()
        return samples

    def _send_loop(self) -> None:
        next_frame = time.monotonic()
        while not self._stop.is_set():
            started=time.monotonic()
            tx_thread_started=(time.thread_time()
                               if getattr(self.realtime, 'thread_timing', False) is True else None)
            trace = self.timing_trace
            stamps = [(started,time.thread_time())] if trace is not None else None
            data_phase=bool(getattr(self._v22,'connected',False))
            if self._tx_watchdog is not None:
                self._tx_active = (started,self._sequence,data_phase,'samples')
            linear = self._next_frame_samples(next_frame)
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
            if tx_thread_started is not None:
                self.realtime.add('tx_thread_cpu_seconds',time.thread_time()-tx_thread_started,
                                  data=data_phase)
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
            self._timestamp = (self._timestamp + self._rtp_frame_samples) & 0xFFFFFFFF
            next_frame += self._rtp_frame_seconds
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
        supports_hardware_flow_control: bool = False,
        set_dte_baud: Callable[[int], bool] | None = None,
        get_dte_baud: Callable[[], int | None] | None = None,
        dte_baud_rates: tuple[int, ...] = (),
        set_dte_character_format: Callable[[int, int], bool] | None = None,
        dte_character_formats: tuple[int, ...] = (),
        dte_parity_values: tuple[int, ...] = (),
    ):
        self._dial = dial
        self._answer = answer
        self._hangup = hangup
        self.supports_hardware_flow_control = bool(supports_hardware_flow_control)
        self._set_dte_baud = set_dte_baud
        self._get_dte_baud = get_dte_baud
        self._dte_baud_rates = tuple(sorted(set(int(rate) for rate in dte_baud_rates if int(rate) > 0)))
        self._pending_dte_baud: int | None = None
        self._set_dte_character_format = set_dte_character_format
        self._dte_character_formats = tuple(sorted(set(dte_character_formats)))
        self._dte_parity_values = tuple(sorted(set(dte_parity_values)))
        self._pending_dte_character_format: tuple[int, int] | None = None
        self._resume_online: Callable[[], None] | None = None
        self._online_action = None
        self._v42_break_action: Callable[..., bool] | None = None
        self._software_test_loop_action: Callable[[bool], bool] | None = None
        self.software_test_loop_state = "off"
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
        self.break_control = 5
        self.icf = (3, 3)  # V.250 recommended 8N1 default
        # Project extensions for preliminary Bell 212A async DTE framing:
        # 9/10 = 7/8 data bits without parity; 11/12 = 8 data + parity +
        # one/two stop bits. +BPAR chooses the parity type for 11/12.
        self.bell212_char_bits = 10
        self.bell212_parity = "E"
        # Hayes &L: 0=switched line; 1=two-wire leased line. The media path
        # still has to be established by the host (e.g. SIP/IAX); this selects
        # Bell 212A's role-led startup once that audio circuit is present.
        self.bell212_line_type = 0
        # Hayes &G: national V.22/V.22bis guard-tone option (0 off, 1 550 Hz,
        # 2 1800 Hz). The engine applies it only in high-channel answer mode.
        self.guard_tone = 0
        # Project extension: the V.23 75-baud backward channel is optional.
        # When disabled, the answerer continues to transmit idle mark.
        self.v23_backward_channel = True
        # Software-loop endpoint role: 0=caller (forward DTE TX),
        # 1=answerer (75-baud backward DTE TX).
        self.v23_software_loop_role = 0
        # Project extension: allow a legacy peer that ignores XID to establish
        # LAPM with defaults, or require the XID exchange to complete.
        self.v42_xid_fallback = os.environ.get("V42_XID_FALLBACK", "1") != "0"
        # V.250 +IFC: DCE_by_DTE controls modem-to-terminal delivery;
        # DTE_by_DCE controls terminal-to-modem transmission. Use the
        # standard RTS/CTS default only where the endpoint exposes both lines.
        self.ifc = (2, 2) if self.supports_hardware_flow_control else (0, 0)
        self.ifc_dte_paused = False
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

    def dte_flow_control_enabled(self) -> bool:
        """Whether +IFC's second field lets the DCE throttle DTE-to-DCE input."""
        return self.ifc[1] == 1

    def _normalize_stored_settings(self) -> None:
        """Discard stored DTE options unsupported by this modem endpoint."""
        try:
            icf = tuple(int(value) for value in self.icf)
        except (TypeError, ValueError):
            icf = (3, 3)
        formats = self._dte_character_formats or (3,)
        parities = self._dte_parity_values or (3,)
        valid_icf = (len(icf) == 2 and icf[0] in formats and icf[1] in (0, 1, 2, 3)
                     and (icf[0] not in (2, 5) or icf[1] in parities))
        self.icf = icf if valid_icf else (3, 3)
        fallback = (2, 2) if self.supports_hardware_flow_control else (0, 0)
        try:
            values = tuple(int(value) for value in self.ifc)
        except (TypeError, ValueError):
            values = fallback
        if len(values) != 2:
            values = fallback
        first = (0, 1, 2, 3) if self.supports_hardware_flow_control else (0, 1, 3)
        second = (0, 1, 2) if self.supports_hardware_flow_control else (0, 1)
        self.ifc = (values[0] if values[0] in first else fallback[0],
                    values[1] if values[1] in second else fallback[1])
        try:
            char_bits = int(self.bell212_char_bits)
        except (TypeError, ValueError):
            char_bits = 10
        self.bell212_char_bits = char_bits if char_bits in (9, 10, 11, 12) else 10
        parity = str(getattr(self, "bell212_parity", "E")).upper()
        self.bell212_parity = parity if parity in ("E", "O", "M", "S") else "E"
        try:
            line_type = int(getattr(self, "bell212_line_type", 0))
        except (TypeError, ValueError):
            line_type = 0
        self.bell212_line_type = line_type if line_type in (0, 1) else 0
        try:
            guard_tone = int(self.guard_tone)
        except (TypeError, ValueError):
            guard_tone = 0
        self.guard_tone = guard_tone if guard_tone in (0, 1, 2) else 0
        try:
            break_control = int(getattr(self, "break_control", 5))
        except (TypeError, ValueError):
            break_control = 5
        self.break_control = break_control if break_control in range(6) else 5
        self.v42_xid_fallback = bool(self.v42_xid_fallback)
        self.v23_backward_channel = bool(self.v23_backward_channel)
        try:
            loop_role = int(self.v23_software_loop_role)
        except (TypeError, ValueError):
            loop_role = 0
        self.v23_software_loop_role = loop_role if loop_role in (0, 1) else 0

    def filter_ifc_input(self, data: bytes) -> bytes:
        """Apply DTE XON/XOFF for +IFC's DCE-output direction.

        The first +IFC field selects whether the DTE can pause data flowing
        from the DCE to the DTE. These control bytes arrive as input from the
        DTE, but govern the opposite direction's output queue.
        """
        mode = self.ifc[0]
        if mode not in (1, 3):
            return data
        out = bytearray()
        for value in data:
            if value == XOFF:
                self.ifc_dte_paused = True
                if mode == 3:
                    out.append(value)
            elif value == XON:
                self.ifc_dte_paused = False
                if mode == 3:
                    out.append(value)
            else:
                out.append(value)
        return bytes(out)

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

    def take_pending_dte_baud(self) -> int | None:
        """Return a requested +IPR change after its result is sent at old speed."""
        rate = self._pending_dte_baud
        self._pending_dte_baud = None
        return rate

    def take_pending_dte_character_format(self) -> tuple[int, int] | None:
        """Return a +ICF change after its result is queued with old framing."""
        value = self._pending_dte_character_format
        self._pending_dte_character_format = None
        return value

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

    def attach_v42_break_action(self, action: Callable[..., bool]) -> None:
        with self._lock:
            self._v42_break_action = action

    def request_v42_break(self, *, length_ms: int | None = None,
                          destructive: bool = False, expedited: bool = False) -> bool:
        action = self._v42_break_action
        if action is None or not self.online:
            return False
        return bool(action(length_ms=length_ms, destructive=destructive,
                           expedited=expedited))

    def attach_software_test_loop(self, action: Callable[[bool], bool]) -> None:
        """Attach the explicitly software-only diagnostic audio-loop control."""
        with self._lock:
            self._software_test_loop_action = action

    def enter_online_command_mode(self) -> bool:
        """Return the DTE to command mode without releasing the active call."""
        with self._lock:
            if not self.online or self.online_return_pending:
                return False
            self._notifications.put(self._result("OK"))
            return True

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

    PROFILE_FIELDS = ("echo", "quiet", "verbose", "result_mode", "dtr_behavior", "flow_control", "break_control", "ifc", "icf", "error_control",
                      "compression", "carrier_detect", "auto_rate_mode", "bell212_char_bits", "bell212_parity", "bell212_line_type", "guard_tone", "v42_xid_fallback",
                      "v23_backward_channel", "v23_software_loop_role")

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
            self._normalize_stored_settings()
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
                     (f", REGISTRATION USER {self.net[102]}" if self.net[104] and self.net[102] else ", NO REGISTRATION CONFIGURED"))
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
                                  stats=copy.deepcopy(stats or {}))
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
                f"&C{get('carrier_detect')} &D{get('dtr_behavior')} &K{get('flow_control')} &L{get('bell212_line_type')} "
                f"\\K{get('break_control')} "
                f"+IFC={get('ifc')[0]},{get('ifc')[1]} +ICF={get('icf')[0]},{get('icf')[1]} "
                f"+BDF={get('bell212_char_bits')} +BPAR={get('bell212_parity')} "
                f"+XID={int(get('v42_xid_fallback'))} +V23BC={int(get('v23_backward_channel'))} "
                f"+V23DIR={get('v23_software_loop_role')} "
                f"\\N{get('error_control')} %C{get('compression')}")

    def _regs_text(self, regs: dict) -> str:
        return " ".join(f"S{int(k):02d}:{v:03d}" for k, v in sorted((int(k), v) for k, v in regs.items()))

    # -- the information screens (laid out like a Courier-class modem's ATI pages; figures are only ever ones we have) --
    SETTINGS_TITLE = f"{PRODUCT.title()}"

    def _register_grid(self, regs: dict) -> list[str]:
        cells = [f"S{int(k):02d}={v:03d}" for k, v in sorted((int(k), v) for k, v in regs.items())]
        return ["  " + "  ".join(cells[i:i + 7]) for i in range(0, len(cells), 7)]

    def _setting_lines(self, get) -> list[str]:
        return [f"  &C{get('carrier_detect')}  &D{get('dtr_behavior')}  &G{get('guard_tone')}  &K{get('flow_control')}  &L{get('bell212_line_type')}  \\K{get('break_control')}",
                f"  E{int(get('echo'))}  Q{int(get('quiet'))}  V{int(get('verbose'))}  X{get('result_mode')}",
                f"  +IFC={get('ifc')[0]},{get('ifc')[1]}",
                f"  +ICF={get('icf')[0]},{get('icf')[1]}  (command/result character format)",
                f"  +BDF={get('bell212_char_bits')}  (Bell 212A character bits)",
                f"  +BPAR={get('bell212_parity')}  (Bell 212A parity: E/O/M/S)",
                f"  +XID={int(get('v42_xid_fallback'))}  (V.42 unanswered-XID compatibility fallback)",
                f"  +V23BC={int(get('v23_backward_channel'))}  (V.23 75-baud backward channel)",
                f"  +V23DIR={get('v23_software_loop_role')}  (software-loop role: 0 caller, 1 answerer)",
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

        # These are not maintained end-to-end counters.  A literal zero would
        # claim that the receive path detected no lost characters, which the
        # current byte-stream and modem engines cannot establish.
        lines.append(self._columns(("Chars lost", "N/A")))

        if b:
            lines.append(self._columns(("Octets sent", b["wire_out"]), ("Octets received", b["wire_in"])))
            lines.append(self._columns(("Blocks sent", b["i_tx"]), ("Blocks received", b["i_rx"])))
            lines.append(self._columns(("Blocks resent", b["i_resent"])))
            lines.append(self._columns(("FCS errors", b["fcs_errors"]), ("REJ sent / received", f"{b['rej_tx']}/{b['rej_rx']}")))
        lines.append("")
        lines.append(self._columns(("Last Tx rate", stats.get('tx_rate') or c["speed"]),
                                   ("Last Rx rate", stats.get('rx_rate') or c["speed"])))
        protocol = (stats.get("error_control") or "None").replace("V.42 LAPM", "V.42")
        lines.append(self._columns(("Modulation", c.get("modulation") or "Unknown"), ("Protocol", protocol)))
        comp = "V.42 bis" if stats.get("compression") == "V.42BIS" else "None"

        lines.append(self._columns(("Compression", comp), ("Line Conditioned", 'N/A')))
        retrain = stats.get('v32_retrain') or {}
        events = retrain.get('events', [])
        completed = sum('completed_seconds' in event for event in events) if 'events' in retrain else 'N/A'
        lines.append(self._columns(("Retrain attempts", retrain.get('attempts', stats.get('retrain_attempts', 'N/A'))),
                                   ("Retrains completed", completed)))
        lines.append(self._columns(("Fallbacks", stats.get('fallbacks', 'N/A'))))
        lines.append(self._columns(("Disconnect Reason", self.DISCONNECT_REASONS.get(c["result"], c["result"]))))

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
            tx_rate, rx_rate = stats.get('tx_rate') or c['speed'], stats.get('rx_rate') or c['speed']
            lines = [head, f"  MODULATION: {c.get('modulation') or 'UNKNOWN'}   TX {tx_rate}  RX {rx_rate}"]
            if stats.get("symbol_rate"):
                lines.append(f"  SYMBOL RATE: {stats['symbol_rate']:.0f}")
            level = stats.get("rx_dbm0")
            lines.append("  RX LEVEL: " + (f"{level:.1f} dBm0" if finite(level) else "N/A"))
            lines.append("  SNR: N/A (NOT MEASURED)   ECHO: N/A (NOT MEASURED)")
            probe_snr = stats.get('probe_snr_db')
            if finite(probe_snr):
                lines.append(f"  PROBE SNR: {probe_snr:.1f} dB (STARTUP ESTIMATE, NOT LIVE DATA SNR)")
            if b and b["i_tx"]:
                lines.append(f"  RETRANSMISSIONS / I-FRAME TX: {100.0 * b['i_resent'] / b['i_tx']:.1f}%")
            lines.extend(self._dsp_report(stats))
            return lines + [""]
        if section == 2:
            timing_slips = stats.get("v34_rx_timing_slips")
            total_slips = timing_slips.get("total") if isinstance(timing_slips, dict) else None
            slip_count = (total_slips if isinstance(total_slips, int)
                          and not isinstance(total_slips, bool) else "N/A")
            lines = [head, f"  TRELLIS ERRORS: N/A   RX TIMING SLIPS: {slip_count}"]
            if b:
                lines.append(f"  V.42 BLOCKS: SENT {b['i_tx']}  RESENT {b['i_resent']}  FCS ERRORS {b['fcs_errors']}  "
                             f"T401 TIMEOUTS {b['t401_expiries']}")
                lines.append(
                    "  RELEASE: DISC TX/RX "
                    f"{b.get('disc_tx', 0)}/{b.get('disc_rx', 0)}  "
                    f"UA TX/RX {b.get('ua_tx', 0)}/{b.get('ua_rx', 0)}  "
                    f"DM TX/RX {b.get('dm_tx', 0)}/{b.get('dm_rx', 0)}"
                )
                if b["wire_in"] and b["bytes_in"]:
                    lines.append(f"  RX COMPRESSION: {b['bytes_in'] / b['wire_in']:.2f}:1")
                if b["wire_out"] and b["bytes_out"]:
                    lines.append(f"  TX COMPRESSION: {b['bytes_out'] / b['wire_out']:.2f}:1")
            else:
                lines.append("  NO ERROR-CONTROL LINK")
            return lines + [""]
        return [head, "  PCM / V.90 / V.92: NOT IMPLEMENTED", ""]

    @staticmethod
    def _dsp_report(stats: dict) -> list[str]:
        """Observable line/DSP facts, without inventing SNR, BER or echo loss."""
        lines = []
        if stats.get('line_state') is not None:
            lines.append(f"  DSP STATE: {stats['line_state']}")
        retrain = stats.get('v32_retrain') or {}
        attempts = retrain.get('attempts', stats.get('retrain_attempts'))
        if attempts is not None:
            active = retrain.get('active', stats.get('retraining', False))
            lines.append(f"  RETRAIN ATTEMPTS: {attempts}   ACTIVE: {'YES' if active else 'NO'}")
        echo = stats.get('v32_echo') or {}
        if echo:
            lines.append(f"  ECHO CANCELLER: {'ENABLED' if echo.get('enabled') else 'DISABLED'}")
            if echo.get('detected_delay') is not None:
                lines.append(f"  ECHO DETECTED DELAY: {echo['detected_delay']} {echo.get('delay_units', '')}")
            if finite(echo.get('input_power')) and finite(echo.get('output_power')):
                lines.append(f"  EC BASEBAND POWER: IN {echo['input_power']:.6g} OUT {echo['output_power']:.6g}")
                lines.append('  Power includes remote signal; not echo-only loss or SNR.')
        quality = stats.get('v34_data_quality') or {}
        if finite(stats.get('decision_mse')):
            lines.append(f"  V.32 DECISION RESIDUAL MSE (EWMA): {stats['decision_mse']:.6g}")
            lines.append(f"  DECISION SYMBOLS: {stats.get('decision_symbols', 'N/A')}; not SNR or BER.")
        cost = quality.get('path_cost_per_symbol')
        if finite(cost):
            lines.append(f"  V.34 DETECTOR PATH COST / SYMBOL: {cost:.6g}")
            lines.append('  Squared waveform-grid units; not SNR or bit error rate.')
            lines.append(f"  QUALITY SYMBOLS: {quality.get('symbols', 'N/A')}   WINDOWS: {quality.get('windows', 'N/A')}")
        rtp = stats.get('rtp') or {}
        if rtp:
            lines.append(f"  RTP PACKETS: RX {rtp.get('received_packets', 'N/A')} DECODED {rtp.get('decoded_packets', 'N/A')}")
            lines.append(f"  RTP LATE/DUPLICATE: {rtp.get('late_or_duplicate_packets', 'N/A')}   CONCEALED SAMPLES: {rtp.get('concealed_samples', 'N/A')}")
        measurements = (stats.get('realtime') or {}).get('measurements') or {}
        for name, label in (('data_rx_seconds', 'RX DSP'), ('data_tx_seconds', 'TX DSP'),
                            ('data_sender_lateness_seconds', 'TX SENDER LATENESS')):
            row = measurements.get(name) or {}
            if finite(row.get('mean')) and finite(row.get('maximum')):
                lines.append(f"  {label}: MEAN {row['mean']*1000:.3f} ms MAX {row['maximum']*1000:.3f} ms")
        return lines

    def _help_screen(self, stem: str) -> list[str] | None:
        """The AT$ family: reference screens listing what this modem actually implements."""

        return help_screen(stem)

    @staticmethod
    def _screen_lines(lines: list[str]) -> list[str]:
        """80-column human-readable pages, unlimited rows; never truncate values."""
        out = []
        for text in lines:
            text = str(text).encode('ascii', 'backslashreplace').decode('ascii')
            text = ''.join(c if 32 <= ord(c) < 127 else f'\\x{ord(c):02x}' for c in text)
            if len(text) <= 80:
                out.append(text)
            else:
                out.extend(textwrap.wrap(text, width=80, subsequent_indent='  ',
                                         replace_whitespace=False, drop_whitespace=False,
                                         break_on_hyphens=False))
        return out


    def _ati(self, n: int) -> list[str] | None:
        lines = self._ati_page(n)
        # ATI9 is a machine packet; preserve its revision bytes and checksum.
        return lines if lines is None or n == 9 else self._screen_lines(lines)

    def _ati_page(self, n: int) -> list[str] | None:
        title = PRODUCT.title()
        if n == 666:
            from modem_easter_egg import SCREEN
            return ['', *SCREEN, '']
        if n == 0:
            return [PRODUCT_CODE, ""]
        if n == 1:
            return [self._program_checksum(), ""]
        if n == 2:
            return [self._self_test()]                                    # the command's own OK follows
        if n == 3:
            built = build_timestamp()
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
            # A profile managed by the supervisor may omit Hayes fields. On
            # recall, _load_profile() first restores factory defaults and then
            # overlays the stored values, so AT&V must display that same
            # effective profile rather than borrowing the live modem state.
            factory = AtDce()
            factory.default_error_control = self.default_error_control
            factory.default_auto_rate = self.default_auto_rate
            factory.net_defaults = dict(self.net_defaults)
            factory._factory_reset()
            for k in sorted(self.profiles):
                snap = self.profiles[k]
                registers = dict(factory.registers)
                registers.update({int(reg): value for reg, value in snap.get("registers", {}).items()})
                out += [f"  Stored Profile {k}:"] + self._register_grid(registers) + [""]
                out += self._setting_lines(lambda name: snap.get(name, getattr(factory, name))) + [""]
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
            except OSError:
                size = "unknown"
            built = build_timestamp()
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
                    "  Program CRC-16:          " + self._program_checksum(),
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
            return [pnp_identification(), ""]
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
        if n == 18:
            return [f"SOFTWARE TEST LOOP: {self.software_test_loop_state.upper()}",
                    "Bell 103, V.21, V.22/V.22bis, Bell 212A, V.23: paired DSP + PCMU",
                    "Software audio test only; not a physical V.54 loop", ""]
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
            return b"".join(self._line(t) for t in self._screen_lines(screen)) + self._result("OK") if screen else self._result("ERROR")
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
                if index >= len(command) or command[index] not in "FCDKVWYSTGPRLMQZT":
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
                elif setting == "T":
                    match = re.match(r"\d?", command[index:])
                    mode = int(match.group() or 0)
                    index += len(match.group())
                    if mode == 1:
                        action = self._software_test_loop_action
                        if action is None or not action(True):
                            return self._result("ERROR")
                        self.software_test_loop_state = "software-audio-loop"
                        self.online = True
                        result = "CONNECT"
                    elif mode == 0:
                        action = self._software_test_loop_action
                        if action is not None and not action(False):
                            return self._result("ERROR")
                        self.software_test_loop_state = "off"
                        self.online = False
                    else:
                        return self._result("ERROR")
                elif setting in "WYVSGPRMQ":
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
                    elif setting == "G":
                        if n not in (0, 1, 2):
                            return self._result("ERROR")
                        self.guard_tone = n
                    # &Y (power-up profile), &S/&P/&R/&M/&Q: accepted,
                    # but have no additional behavior on a software line.
                elif setting == "L":
                    match = re.match(r"\d*", command[index:])
                    index += len(match.group())
                    n = int(match.group() or 0)
                    if n not in (0, 1):
                        # &L2 is four-wire leased service. The present engine
                        # models only the two-wire full-duplex interface.
                        return self._result("ERROR")
                    self.bell212_line_type = n
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
                        # command mode, hang up, hang up+reset). The DCE
                        # stores this policy; run_pty applies it when a serial
                        # backend reports a real DTR transition. A bare PTY
                        # cannot report DTR and therefore does not trigger it.
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
                    _sip_setup_trace("atd_dispatch", callback="attached", digits=len(self._number))
                    call_result = self._dial(dial_string)
                    if call_result is None:
                        _sip_setup_trace("atd_pending", result="async")
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
                    _sip_setup_trace("atd_dispatch", callback="missing", digits=len(self._number))
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
                m = re.match(r"(MS|GMI|GMM|GMR|GCAP|IPR|IFC|ICF|BDF|BPAR|XID|V23BC|V23DIR)(\?|=\?|=.*)?$", rest)
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
                        output.extend(self._line("+MS: (B103,B212,V21,V22,V22B,V23,V32,V32B,V34),(0,1),(300-33600),(300-33600)"))
                    else:
                        fields = arg[1:].split(",")
                        mod = fields[0].strip()
                        if mod not in ("B103", "B212", "V21", "V22", "V22B", "V23", "V32", "V32B", "V34"):
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
                # V.250 +ICF controls asynchronous command/result framing.
                elif name == "ICF":
                    if arg == "?":
                        output.extend(self._line(f"+ICF: {self.icf[0]},{self.icf[1]}"))
                    elif arg == "=?":
                        formats = self._dte_character_formats or (3,)
                        parities = self._dte_parity_values or (3,)
                        output.extend(self._line(
                            f"+ICF: ({','.join(map(str, formats))}),"
                            f"({','.join(map(str, parities))})"))
                    elif arg.startswith("="):
                        fields = arg[1:].split(",")
                        if len(fields) > 2 or not fields[0]:
                            return self._result("ERROR")
                        try:
                            format_code = int(fields[0])
                            parity_code = int(fields[1]) if len(fields) == 2 and fields[1] else self.icf[1]
                        except ValueError:
                            return self._result("ERROR")
                        formats = self._dte_character_formats or (3,)
                        parities = self._dte_parity_values or (3,)
                        if (format_code not in formats or parity_code not in (0, 1, 2, 3)
                                or (format_code in (2, 5) and parity_code not in parities)):
                            return self._result("ERROR")
                        if (format_code, parity_code) != self.icf:
                            self._pending_dte_character_format = (format_code, parity_code)
                        self.icf = (format_code, parity_code)
                    else:
                        return self._result("ERROR")
                elif name == "IFC":
                    if arg == "?":
                        output.extend(self._line(f"+IFC: {self.ifc[0]},{self.ifc[1]}"))
                    elif arg == "=?":
                        first = [0, 1, 2, 3] if self.supports_hardware_flow_control else [0, 1, 3]
                        second = [0, 1] + ([2] if self.supports_hardware_flow_control else [])
                        output.extend(self._line(
                            f"+IFC: ({','.join(map(str, first))}),({','.join(map(str, second))})"))
                    elif arg.startswith("="):
                        fields = arg[1:].split(",")
                        if len(fields) > 2:
                            return self._result("ERROR")
                        values = list(self.ifc)
                        try:
                            for field_index, field in enumerate(fields):
                                if field.strip():
                                    values[field_index] = int(field)
                        except ValueError:
                            return self._result("ERROR")
                        first_supported = (0, 1, 2, 3) if self.supports_hardware_flow_control else (0, 1, 3)
                        second_supported = (0, 1, 2) if self.supports_hardware_flow_control else (0, 1)
                        if values[0] not in first_supported or values[1] not in second_supported:
                            return self._result("ERROR")
                        self.ifc = tuple(values)
                        self.ifc_dte_paused = False
                    else:
                        return self._result("ERROR")
                elif name == "BDF":
                    if arg == "?":
                        output.extend(self._line(f"+BDF: {self.bell212_char_bits}"))
                    elif arg == "=?":
                        output.extend(self._line("+BDF: (9,10,11,12)"))
                    elif arg.startswith("="):
                        try:
                            value = int(arg[1:])
                        except ValueError:
                            return self._result("ERROR")
                        if value not in (9, 10, 11, 12):
                            return self._result("ERROR")
                        self.bell212_char_bits = value
                    else:
                        return self._result("ERROR")
                elif name == "BPAR":
                    if arg == "?":
                        output.extend(self._line(f"+BPAR: {self.bell212_parity}"))
                    elif arg == "=?":
                        output.extend(self._line("+BPAR: (E,O,M,S)"))
                    elif arg.upper() in ("=E", "=O", "=M", "=S"):
                        self.bell212_parity = arg[1:].upper()
                    else:
                        return self._result("ERROR")
                elif name == "V23BC":
                    if arg == "?":
                        output.extend(self._line(f"+V23BC: {int(self.v23_backward_channel)}"))
                    elif arg == "=?":
                        output.extend(self._line("+V23BC: (0,1)"))
                    elif arg in ("=0", "=1"):
                        self.v23_backward_channel = arg == "=1"
                    else:
                        return self._result("ERROR")
                elif name == "V23DIR":
                    if arg == "?":
                        output.extend(self._line(f"+V23DIR: {self.v23_software_loop_role}"))
                    elif arg == "=?":
                        output.extend(self._line("+V23DIR: (0,1)"))
                    elif arg in ("=0", "=1"):
                        self.v23_software_loop_role = int(arg[1])
                    else:
                        return self._result("ERROR")
                elif name == "XID":
                    if arg == "?":
                        output.extend(self._line(f"+XID: {int(self.v42_xid_fallback)}"))
                    elif arg == "=?":
                        output.extend(self._line("+XID: (0,1)"))
                    elif arg in ("=0", "=1"):
                        self.v42_xid_fallback = arg == "=1"
                    else:
                        return self._result("ERROR")
                elif name == "IPR":
                    if self._set_dte_baud is None or not self._dte_baud_rates:
                        return self._result("ERROR")
                    if arg == "?":
                        try:
                            current = self._get_dte_baud() if self._get_dte_baud else None
                        except (OSError, ValueError):
                            current = None
                        if current not in self._dte_baud_rates:
                            return self._result("ERROR")
                        output.extend(self._line(f"+IPR: {current}"))
                    elif arg == "=?":
                        output.extend(self._line(
                            f"+IPR: ({','.join(map(str, self._dte_baud_rates))})"))
                    elif arg.startswith("="):
                        try:
                            rate = int(arg[1:])
                        except ValueError:
                            return self._result("ERROR")
                        if rate not in self._dte_baud_rates:
                            return self._result("ERROR")
                        # The event loop applies this after writing the final
                        # result so the terminal receives OK at the old speed.
                        self._pending_dte_baud = rate
                    else:
                        return self._result("ERROR")
            elif current == "\\":
                if index >= len(command) or command[index] not in "NKB":
                    return self._result("ERROR")
                option = command[index]
                index += 1
                if option == "N":
                    # 0 raw, 1 lab ARQ (lab-to-lab only), 2 MNP (V.42 Annex A:
                    # MNP 2-4 class link request, falling back to raw), 3
                    # auto-reliable: try V.42/LAPM with a real peer, falling
                    # back to raw.
                    if index >= len(command) or command[index] not in "0123":
                        return self._result("ERROR")
                    self.error_control = int(command[index])
                    index += 1
                elif option == "K":
                    if index >= len(command) or command[index] not in "012345":
                        return self._result("ERROR")
                    self.break_control = int(command[index])
                    index += 1
                elif option == "B":
                    match = re.match(r"\d?", command[index:])
                    index += len(match.group())
                    if not self.online:
                        return self._result("ERROR")
                    destructive, expedited = v42_manual_break_options_for_k(
                        self.break_control)
                    # Hayes reliable-link behavior uses a fixed 300 ms BREAK,
                    # regardless of the raw-mode 100 ms multiplier n.
                    if not self.request_v42_break(length_ms=300,
                                                  destructive=destructive,
                                                  expedited=expedited):
                        return self._result("ERROR")
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
        lo = int(ms["min"])
        supported_v22 = tuple(rate for rate in (2400, 1200) if lo <= rate <= hi)
        supported_v32 = tuple(
            rate for rate in (14400, 12000, 9600, 7200, 4800)
            if lo <= rate <= min(hi, 9600 if mod == "V32" else 14400)
        )
        v34_max = min(hi, 33600) // 2400 * 2400
        v34_min = (lo + 2399) // 2400 * 2400
        v34_in_range = 2400 <= v34_min <= v34_max
        if not supported_v22:
            allow &= ~V8_MOD_V22
        if not supported_v32:
            allow &= ~V8_MOD_V32
        if not v34_in_range:
            allow &= ~V8_MOD_V34
        if not allow:
            raise ValueError(f"{mod} automode range contains no supported modulation rate")
        if mod == "V22" and not supported_v22:
            raise ValueError("V.22 automode currently supports 1200 bit/s; requested range excludes it")
        if mod == "V22B" and not supported_v22:
            raise ValueError("V.22bis automode range excludes supported rates 1200 and 2400 bit/s")
        v22_rate = supported_v22[0] if supported_v22 else (2400 if hi >= 2400 else 1200)
        return {"line_mode": "chain", "chain": {
            "allow": allow, "v34_rate": max(2400, v34_max), "v34_min_rate": max(2400, v34_min),
            "v32_rates": supported_v32,
            "v22_rate": v22_rate, "v22_min_rate": lo}}
    if mod in ("B103",):
        if not int(ms["min"]) <= 300 <= hi:
            raise ValueError("Bell 103 supports 300 bit/s; requested range excludes it")
        return {"line_mode": "bell103", "negotiate": False}
    if mod == "V21":
        if not int(ms["min"]) <= 300 <= hi:
            raise ValueError("V.21 supports 300 bit/s; requested range excludes it")
        return {"line_mode": "v21"}
    if mod == "V23":
        if ms.get("auto"):
            raise ValueError("V.23 has no V.8 automode negotiation")
        lo = int(ms["min"])
        if lo > 1200 or hi < 600:
            raise ValueError("V.23 supports a 600- or 1200-baud forward channel")
        rate = 1200 if hi >= 1200 and lo <= 1200 else 600
        if lo > rate:
            raise ValueError("V.23 selected baud is below the configured minimum")
        return {"line_mode": "v23", "v23_rate": rate}
    if mod == "B212":
        lo = int(ms["min"])
        if lo > 1200 or hi < 1200:
            raise ValueError("Bell 212A currently requires a 1200 bit/s startup; requested range excludes it")
        return {"line_mode": "bell212a", "v22_rate": 1200, "v22_min_rate": lo}
    if mod == "V22":
        lo = int(ms["min"])
        rates = tuple(rate for rate in (1200, 600) if lo <= rate <= hi)
        if not rates:
            raise ValueError("V.22 requested range excludes supported rates 600 and 1200 bit/s")
        return {"line_mode": "v22bis", "v22_rate": rates[0], "v22_min_rate": lo}
    if mod == "V22B":
        lo = int(ms["min"])
        rates = tuple(rate for rate in (2400, 1200) if lo <= rate <= hi)
        if not rates:
            raise ValueError("V.22bis requested range excludes supported rates 1200 and 2400 bit/s")
        return {"line_mode": "v22bis", "v22_rate": rates[0], "v22_min_rate": lo}
    if mod in ("V32", "V32B"):
        lo = int(ms["min"])
        ceiling = min(hi, 9600 if mod == "V32" else 14400)
        rates = tuple(r for r in (14400, 12000, 9600, 7200, 4800) if lo <= r <= ceiling)
        if not rates:
            raise ValueError(f"{mod} requested range excludes supported rates 4800..{ceiling} bit/s")
        return {"line_mode": "v32", "v32_rates": rates}
    lo = int(ms["min"])
    v34_max = min(33600, hi) // 2400 * 2400
    v34_min = max(2400, (lo + 2399) // 2400 * 2400)
    if v34_min > v34_max:
        raise ValueError("V.34 requested range contains no supported data rate")
    return {"line_mode": "v34", "v34_rate": v34_max, "v34_min_rate": v34_min}


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


class DeferredV42Release:
    """Track when a local call may be torn down after LAPM DISC release."""

    def __init__(self) -> None:
        self.deadline: float | None = None

    @property
    def pending(self) -> bool:
        return self.deadline is not None

    def request(self, session, *, now: float | None = None) -> bool:
        """Start orderly release, returning False when immediate teardown is needed."""
        if session is None or not session.request_v42_release():
            self.deadline = None
            return False
        started = time.monotonic() if now is None else now
        self.deadline = started + max(0.0, float(session.v42_release_timeout()))
        return True

    def ready(self, session, *, now: float | None = None) -> bool:
        """Whether release completed or its bounded safety deadline expired."""
        if self.deadline is None:
            return False
        current = time.monotonic() if now is None else now
        return (
            session is None
            or session.v42_release_complete()
            or current >= self.deadline
        )

    def clear(self) -> None:
        self.deadline = None


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
    tcp_baud_control: bool = False,
    serial_tcp: str | None = None,
    serial_tcp_baud_control: bool = False,
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
    announce_later = negotiate or line_mode in (
        "v21", "v22bis", "bell212a", "auto", "v32", "v34", "ms", "chain"
    )
    negotiate = negotiate and line_mode == "bell103"
    arq_default = 3 if os.environ.get("EL_MODEM_FACTORY_NATIVE") == "1" else default_error_control_for(line_mode, negotiate)
    from serial_port import PtySerial, TcpSerial  # noqa: PLC0415

    if serial_backend is not None:
        ser = serial_backend() if callable(serial_backend) else serial_backend
    elif serial_tcp:
        tcp_h, tcp_p = serial_tcp.rsplit(":", 1)
        ser = TcpSerial(tcp_h, int(tcp_p),
                        baud_control_port=int(tcp_p) + 1 if serial_tcp_baud_control else None)
    else:
        ser = PtySerial(link)
    slave_name = ser.name
    terminal_dropped = lambda: None          # replaced once the SIP side exists
    def term_write(data):
        if control is not None and data: control.rd_until = time.monotonic()+.15
        ser.write(data)

    def new_dce():
        supported_rates = getattr(ser, "supported_dte_baud_rates", None)
        supported_formats = getattr(ser, "supported_dte_character_formats", None)
        set_rate = getattr(ser, "set_dte_baud", None)
        get_rate = getattr(ser, "get_dte_baud", None)
        set_format = getattr(ser, "set_dte_character_format", None)
        try:
            rates = tuple(supported_rates()) if callable(supported_rates) else ()
        except (OSError, ValueError):
            rates = ()
        try:
            formats, parities = supported_formats() if callable(supported_formats) else ((), ())
        except (OSError, ValueError):
            formats, parities = (), ()
        return AtDce(
            supports_hardware_flow_control=bool(getattr(ser, "supports_modem_control", False)),
            set_dte_baud=set_rate if callable(set_rate) and rates else None,
            get_dte_baud=get_rate if callable(get_rate) else None,
            dte_baud_rates=rates,
            set_dte_character_format=set_format if callable(set_format) else None,
            dte_character_formats=tuple(formats),
            dte_parity_values=tuple(parities),
        )
    media_socket: socket.socket | None = None
    sip_server: SipIncomingCallServer | None = None
    iax_transport: IaxTransport | None = None
    iax_media_socket: IaxMediaSocket | None = None
    registration = None
    registration_key = None
    final_hangup: Callable[[], None] = lambda: None     # set once the SIP side exists; sends BYE on shutdown
    orderly_release = DeferredV42Release()
    dce: AtDce
    if tcp_listen and not serial_tcp:
        # Serial port over TCP for terminal programs outside the container (see serial_bridge.py).
        from serial_bridge import serve_pty_over_tcp  # noqa: PLC0415

        tcp_host, tcp_port = tcp_listen.rsplit(":", 1)
        serve_pty_over_tcp(link or slave_name, tcp_host, int(tcp_port),
                           baud_control_port=int(tcp_port) + 1 if tcp_baud_control else None)
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
        dce = new_dce()
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
        def sip_packetization_ms() -> int:
            selected = dce.ms.get("mod", "") if line_mode == "ms" else line_mode
            return 10 if selected.upper() == "V23" else 20

        if dce._set_dte_character_format is not None:
            if not dce._set_dte_character_format(*dce.icf):
                print(f"[dce] failed to apply saved DTE framing {dce.icf}",
                      file=sys.stderr, flush=True)
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
                packetization_ms=sip_packetization_ms(),
                on_incoming=lambda: dce.begin_incoming_call(sip_server.last_caller),
                on_ring=dce.notify_incoming_ring,
                on_hangup=lambda: dce.notify_call_progress("NO CARRIER"),
            )

        active_outbound: SipOriginateCall | None = None
        iax_dial_pending = False
        iax_media_active = False
        iax_registered = False      # drives the MR lamp on the IAX path
        online_session: OnlineAudioSession | None = None
        software_loop = None
        arq_session: ArqSession | None = None
        data_mode = False
        connect_pending = False
        bell_dte_framing_active = False
        escape_detector = EscapeSequenceDetector()

        def set_bell_dte_framing(enabled: bool) -> bool:
            """Switch a native UART between command and Bell 212A data framing."""
            nonlocal bell_dte_framing_active
            if enabled:
                if online_session is None or getattr(online_session._v22, "profile", None) != "bell212a":
                    return True
                setter = getattr(ser, "set_bell212_dte_format", None)
                if not callable(setter):
                    return True  # byte-stream and PTY endpoints have no UART framing bits
                if not setter(dce.bell212_char_bits, dce.bell212_parity):
                    return False
                bell_dte_framing_active = True
                return True
            if not bell_dte_framing_active:
                return True
            setter = getattr(ser, "set_dte_character_format", None)
            restored = callable(setter) and setter(*dce.icf)
            if restored:
                bell_dte_framing_active = False
            return bool(restored)

        def start_online_session(
            tx_role: str, rx_role: str, remote_rtp: tuple[str, int]
        ) -> None:
            nonlocal online_session, arq_session, data_mode, connect_pending
            nonlocal remote_break_deadline
            nonlocal remote_break_waiting, remote_break_post_data
            nonlocal break_transport_warning
            if online_session is not None:
                if not set_bell_dte_framing(False):
                    print("[dce] failed to restore command-mode DTE framing before retrain",
                          file=sys.stderr, flush=True)
                dce.call_ended("HANGUP", online_session.link_stats() if dce.online else None)
                online_session.stop()
            try:
                session_args = dict(negotiate=negotiate, line_mode=line_mode, v22_rate=v22_rate)
                if line_mode == "ms":                   # AT+MS= picks the modulation for this call
                    session_args.update(line_from_ms(dce.ms))
                if session_args.get("line_mode") == "v23":
                    session_args["v23_backward_channel"] = dce.v23_backward_channel
                if session_args.get("line_mode") in ("v22bis", "auto"):
                    session_args["guard_tone"] = dce.guard_tone
                if session_args.get("line_mode") == "bell212a":
                    session_args["bell212_char_bits"] = dce.bell212_char_bits
                    session_args["bell212_parity"] = dce.bell212_parity
                    session_args["bell212_dte_baud"] = bell212_dte_baud_for_serial(ser)
                    session_args["bell212_leased_line"] = dce.bell212_line_type == 1
                if dce.calling_tone_once:               # `^`: consumed here, this call only
                    # V.21 already has the mandatory automatic V.25 caller
                    # sequence; the legacy `^` option only adds the tone to
                    # Bell 103 calls, where no V.25 procedure is implicit.
                    if session_args.get("line_mode") != "v21":
                        session_args["calling_tone"] = True
                    dce.calling_tone_once = False
                online_session = OnlineAudioSession(
                    (iax_media_socket if iax_media_active else media_socket),
                    remote_rtp, tx_role, rx_role,
                    **session_args,
                    v42=(dce.error_control == 3), v42_frame_size=v42_frame_size,
                    v42bis=bool(dce.compression), v42_xid_fallback=dce.v42_xid_fallback,
                    mnp=(dce.error_control == 2),
                )
            except ValueError as exc:               # e.g. v32 cannot answer
                print(f"[modem] {exc}", file=sys.stderr, flush=True)
                online_session = None
                dce.notify_call_progress("NO CARRIER")
                return
            online_session.start()
            escaped.clear()
            ifc_output_hold.clear()
            local_break_requests.reset()
            remote_break_waiting.clear()
            remote_break_deadline = 0.0
            remote_break_post_data.clear()
            break_transport_warning = False
            dce.ifc_dte_paused = False
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
            nonlocal remote_break_deadline
            nonlocal remote_break_waiting, remote_break_post_data
            nonlocal break_transport_warning
            session=online_session
            if not set_bell_dte_framing(False):
                print("[dce] failed to restore command-mode DTE framing", file=sys.stderr, flush=True)
            online_session = None
            arq_session = None
            data_mode = False
            connect_pending = False
            orderly_release.clear()
            iax_media_active = False
            escaped.clear()
            ifc_output_hold.clear()
            local_break_requests.reset()
            remote_break_waiting.clear()
            remote_break_deadline = 0.0
            remote_break_post_data.clear()
            break_transport_warning = False
            dce.calling_tone_once = False
            if session is not None:
                session.set_dte_receive_busy(False)
                session.stop()

        def resume_online() -> None:
            nonlocal data_mode
            if online_session is not None:
                if not set_bell_dte_framing(True):
                    print("[dce] failed to restore Bell 212A data framing", file=sys.stderr, flush=True)
                    unified_hangup()
                    dce.notify_call_progress("NO CARRIER")
                    return
                held = escaped.flush()
                data_mode = True
                if held:
                    term_write(held)
                    dce.count_chars(received=len(held))

        dce.attach_resume_online(resume_online)
        dce.attach_online_control(lambda parameter: online_session is not None
                                  and online_session.request_online_control(parameter))
        dce.attach_v42_break_action(
            lambda **options: (online_session is not None and dce.online
                               and online_session.v42_break_available()
                               and local_break_requests.enqueue(**options)))

        def software_test_loop_action(enabled: bool) -> bool:
            nonlocal software_loop, data_mode
            if enabled:
                if online_session is not None or software_loop is not None:
                    return False
                try:
                    selected = (line_from_ms(dce.ms) if line_mode == "ms"
                                else {"line_mode": line_mode})
                    profile = selected.get("line_mode")
                    if profile == "v23":
                        from v23 import V23SoftwareAudioLoop  # noqa: PLC0415

                        software_loop = V23SoftwareAudioLoop(
                            forward_baud=selected.get("v23_rate", 1200),
                            calling_party=(dce.v23_software_loop_role == 0),
                            backward_channel=dce.v23_backward_channel,
                        )
                    elif profile in ("v21", "bell103"):
                        from bell103_engine import SoftwareFskAudioLoop  # noqa: PLC0415

                        software_loop = SoftwareFskAudioLoop(
                            standard=profile, calling_party=True)
                    elif profile in ("v22bis", "bell212a"):
                        from v22_audio_loop import V22SoftwareAudioLoop  # noqa: PLC0415

                        loop_rate = (1200 if profile == "bell212a"
                                     else selected.get("v22_rate", v22_rate))
                        software_loop = V22SoftwareAudioLoop(
                            bit_rate=loop_rate,
                            calling_party=True,
                            profile="bell212a" if profile == "bell212a" else None,
                        )
                    else:
                        return False
                except (ImportError, ValueError):
                    software_loop = None
                    return False
                dce.connect_speed = getattr(
                    software_loop, "forward_baud",
                    getattr(software_loop, "bit_rate", 300))
                dce.connect_suffix = "/SW-LOOP"
                modulation = (
                    software_loop.status()["standard"]
                    if profile in ("v22bis", "bell212a") else {
                        "v23": "V.23", "v21": "V.21", "bell103": "Bell 103",
                    }[profile]
                )
                dce.connect_modulation = f"{modulation} software audio loop"
                dce.software_test_loop_state = "software-audio-loop"
                data_mode = True
                return True
            if software_loop is not None:
                close_loop = getattr(software_loop, "close", None)
                if close_loop is not None:
                    close_loop()
                software_loop = None
                data_mode = False
                dce.connect_suffix = ""
                dce.connect_modulation = ""
                dce.software_test_loop_state = "off"
                escape_detector.__init__()
            return True

        dce.attach_software_test_loop(software_test_loop_action)

        def _terminal_dropped() -> None:
            """The terminal program closed the port: what a DTR drop does on a real modem, per &D."""
            nonlocal data_mode
            mode = dce.dtr_behavior
            busy = online_session is not None or dce.online or dce.outgoing_pending
            if mode == 0 or not busy and mode < 3:
                return
            if mode == 1:
                if not set_bell_dte_framing(False):
                    print("[dce] failed to restore command-mode DTE framing", file=sys.stderr, flush=True)
                    request_local_hangup()
                    return
                data_mode = False               # &D1: back to command mode, the connection stays up
                term_write(dce._result("OK"))
                return
            request_local_hangup()              # &D2/&D3: orderly LAPM release, then hang up
            dce.online = False
            dce.outgoing_pending = False
            set_bell_dte_framing(False)
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
            nonlocal software_loop, data_mode
            orderly_release.clear()
            try:
                end_online_session()
            finally:
                close_loop = getattr(software_loop, "close", None)
                if close_loop is not None:
                    close_loop()
                software_loop = None
                data_mode = False
                dce.software_test_loop_state = "off"
                call=active_outbound
                active_outbound = None
                if call is not None:
                    call.hangup()
                elif sip_server is not None:
                    sip_server.hangup()
                if iax_transport is not None and iax_transport.call_id is not None:
                    iax_transport.hangup()

        def request_local_hangup() -> None:
            """Let LAPM's DISC/UA exchange finish before ending a local call."""
            nonlocal data_mode, arq_session
            session = online_session
            if orderly_release.request(session):
                # ATH/DTR has returned the DTE to command mode. Keep the media
                # pump alive so DISC and its response traverse the modem DSP.
                if not set_bell_dte_framing(False):
                    print("[dce] failed to restore command-mode DTE framing", file=sys.stderr, flush=True)
                    unified_hangup()
                    return
                data_mode = False
                arq_session = None
                return
            unified_hangup()

        def answer_incoming() -> bool:
            nonlocal iax_media_active
            if iax_transport is not None and iax_transport.incoming_call_id is not None:
                answered = iax_transport.answer()
                if answered:
                    iax_media_active = True
                return answered
            if sip_server is not None:
                # AT+MS can be changed after startup but before ATA; advertise
                # the selected mode's packet interval in the 200 OK SDP.
                sip_server.packetization_ms = sip_packetization_ms()
            answered = sip_server.answer() if sip_server is not None else False
            if answered:
                remote_rtp = sip_server.remote_rtp()
                if remote_rtp:
                    start_online_session("answer", "originate", remote_rtp)
            return answered

        dce.attach_call_control(answer_incoming, request_local_hangup)
        final_hangup = unified_hangup
        if sip_server is not None:
            sip_server.on_hangup = lambda: (end_online_session(), dce.notify_call_progress("NO CARRIER"))
            sip_server.start()
        def apply_network() -> None:
            """Apply active SIP registration and IAX endpoint settings."""
            nonlocal registration, registration_key, iax_transport, iax_media_socket, iax_registered
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
                # Do not carry the old endpoint's registration into the new
                # configuration while its REGISTER exchange is pending.
                iax_registered = False
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
                _sip_setup_trace("dial_callback", trunk="iax" if dce.net[120] == "1" else "sip")
                if active_outbound is not None:
                    _sip_setup_trace("dial_rejected", reason="already_active")
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
                        _sip_setup_trace("dial_rejected", reason="iax_unconfigured")
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
                    _sip_setup_trace(
                        "dial_rejected", reason="sip_unconfigured",
                        has_target=bool(target_host), sip_listener=bool(sip_server),
                    )
                    return "NO DIALTONE"
                _sip_setup_trace("originate_scheduled", target=f"{target_host}:{int(net[101] or dial_port)}")
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
                    packetization_ms=sip_packetization_ms(),
                )
                active_outbound = call
                threading.Thread(
                    target=call.dial, args=(destination,), daemon=True
                ).start()

            dce.attach_dial(dial_number)
    else:
        dce = new_dce()
        online_session = None
        software_loop = None
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
    ifc_output_hold = bytearray()
    ifc_output_hold_limit = 1 << 20
    local_break_requests = V42BreakRequestQueue()
    remote_break_waiting = []
    remote_break_deadline = 0.0
    remote_break_post_data = bytearray()
    break_transport_warning = False

    def flush_ifc_output_hold():
        if not ifc_output_hold:
            return
        if remote_break_deadline > time.monotonic():
            return
        if dce.ifc_dte_paused:
            return
        # +IFC=2,0 uses the DTE's RTS input as receive backpressure. A
        # missing/unknown line is not interpreted as a pause.
        if dce.ifc[0] == 2:
            try:
                rts = ser.signals().get("RTS")
            except (AttributeError, OSError):
                rts = None
            if rts is False:
                return
        # Serial backends have bounded write queues. Move a small amount per
        # loop so the backend can service its fd between chunks.
        chunk = bytes(ifc_output_hold[:4096])
        try:
            term_write(chunk)
        except BufferError:
            return
        del ifc_output_hold[:len(chunk)]

    def update_ifc_dte_backpressure():
        if not data_mode or online_session is None:
            return
        mode = dce.ifc[0]
        paused = dce.ifc_dte_paused
        if mode == 2:
            try:
                paused = paused or ser.signals().get("RTS") is False
            except (AttributeError, OSError):
                pass
        if mode not in (1, 2, 3):
            online_session.set_dte_receive_busy(False)
            return
        busy = paused or len(ifc_output_hold) >= 32768
        if not busy and len(ifc_output_hold) > 16384:
            busy = getattr(online_session, "_dte_receive_busy", False)
        online_session._dte_receive_busy = busy
        online_session.set_dte_receive_busy(busy)

    def deliver_received(decoded):
        # Timer and packet drains use the same DTE/ARQ delivery and prompt ACK.
        if not decoded or connect_pending:
            return
        if remote_break_deadline > time.monotonic():
            if len(remote_break_post_data) + len(decoded) > ifc_output_hold_limit:
                print("[v42] post-BREAK DTE queue exhausted; dropping call", file=sys.stderr, flush=True)
                remote_break_post_data.clear()
                unified_hangup()
                dce.notify_call_progress("NO CARRIER")
                return
            remote_break_post_data.extend(decoded)
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
        rts_paused = False
        if data_mode and dce.ifc[0] == 2:
            try:
                rts_paused = ser.signals().get("RTS") is False
            except (AttributeError, OSError):
                pass
        if data_mode and (dce.ifc_dte_paused or rts_paused or ifc_output_hold):
            if len(ifc_output_hold) + len(delivered) > ifc_output_hold_limit:
                print("[modem] DTE receive-flow buffer exhausted; dropping call", file=sys.stderr, flush=True)
                ifc_output_hold.clear()
                unified_hangup()
                dce.notify_call_progress("NO CARRIER")
                return
            ifc_output_hold.extend(delivered)
            dce.count_chars(received=len(delivered))
            update_ifc_dte_backpressure()
            return
        flush_ifc_output_hold()
        update_ifc_dte_backpressure()
        term_write(delivered)
        if os.environ.get("MODEM_DEBUG") == "1" and online_session is not None and data_mode:
            print(f"[dce] terminal receive bytes={len(delivered)}", file=sys.stderr, flush=True)
        dce.count_chars(received=len(delivered))

    def service_v42_breaks():
        """Bridge serial BREAK sideband and deferred V.42 L-SIGNAL actions."""
        nonlocal data_mode
        nonlocal remote_break_deadline, break_transport_warning
        mode = (getattr(online_session._v22, "v42_mode", None)
                if online_session is not None else None)
        engine = getattr(online_session, "_v22", None) if online_session is not None else None
        bell_raw = (getattr(engine, "profile", None) == "bell212a"
                    and not getattr(engine, "_bell103_fallback", False)
                    and mode not in ("detect", "lapm"))
        take_breaks = getattr(ser, "take_break_events", None)
        if take_breaks is not None:
            try:
                count = max(0, int(take_breaks()))
            except (OSError, TypeError, ValueError):
                count = 0
            if count:
                if mode == "lapm":
                    options = v42_break_options_for_k(dce.break_control)
                    if options is None:
                        if data_mode and dce.enter_online_command_mode():
                            data_mode = False
                            print(f"[v42] DTE BREAK \\K{dce.break_control}: online command mode; no remote BREAK",
                                  file=sys.stderr, flush=True)
                        else:
                            print("[v42] DTE BREAK ignored outside online data mode",
                                  file=sys.stderr, flush=True)
                    elif not dce.online:
                        print("[v42] DTE BREAK ignored outside online data mode",
                              file=sys.stderr, flush=True)
                    else:
                        destructive, expedited = options
                        accepted = 0
                        for _ in range(count):
                            if local_break_requests.enqueue(
                                    destructive=destructive, expedited=expedited):
                                accepted += 1
                        if accepted < count:
                            print(f"[v42] dropped {count - accepted} DTE BREAK event(s): queue full",
                                  file=sys.stderr, flush=True)
                elif bell_raw and data_mode:
                    accepted = 0
                    for _ in range(count):
                        if online_session.send_bell212_break():
                            accepted += 1
                    if accepted < count:
                        print(f"[bell212] could not queue {count - accepted} DTE BREAK event(s)",
                              file=sys.stderr, flush=True)
                else:
                    print(f"[modem] ignored {count} DTE BREAK event(s): no active raw Bell/LAPM data path",
                          file=sys.stderr, flush=True)

        if online_session is None:
            return

        # Consume all link events before starting another local BREAK. Duplicate
        # BRK frames have already been suppressed by LAPM, so each indication
        # here represents one DTE-side action that still needs confirmation.
        for event in online_session.read_v42_break_events():
            kind = event.get("type")
            if kind == "break_indication":
                remote_break_waiting.append(event)
                # D/S data handling happens before BRKACK is allowed. For S=1
                # this requeues any OS-held bytes after BREAK; for D=1 it
                # discards accumulated undelivered data.
                if not prepare_v42_break_dte_order(event, ser, ifc_output_hold):
                    action = "destructive" if event.get("destructive") else "expedited"
                    print(f"[v42] {action} BREAK ordering unavailable; leaving BRK unacknowledged",
                          file=sys.stderr, flush=True)
                    event["dte_action_failed"] = True
            elif kind in ("break_confirmed", "break_failed"):
                local_break_requests.complete()
                if kind == "break_failed":
                    print(f"[v42] local BREAK failed: {event.get('reason', 'no BRKACK')}",
                          file=sys.stderr, flush=True)

        if bell_raw and data_mode:
            # Bell Mode-B BREAK is an in-band start-polarity run, not a LAPM
            # L-SIGNAL. Once decoded, reproduce it as a physical DTE BREAK
            # where the selected serial backend supports that sideband.
            for event in online_session.read_bell212_break_events():
                send_break = getattr(ser, "send_break", None)
                bit_count = max(1, int(event.get("start_bits", 0)))
                duration_ms = max(1, (bit_count * 1000 + 1199) // 1200)
                try:
                    applied = send_break is not None and send_break(duration_ms)
                except (OSError, ValueError, TypeError):
                    applied = False
                if not applied and not break_transport_warning:
                    print("[bell212] terminal transport cannot represent raw Bell BREAK; "
                          "continuing without sideband delivery", file=sys.stderr, flush=True)
                    break_transport_warning = True

        # Hardware DTE BREAKs and command-mode AT\B share one bounded queue;
        # only one LAPM BRK transaction may be outstanding at a time.
        if mode == "lapm" and online_session is not None:
            local_break_requests.service(online_session.request_v42_break)
        elif local_break_requests.pending or local_break_requests.in_flight:
            local_break_requests.reset()

        if remote_break_waiting and time.monotonic() >= remote_break_deadline:
            event = remote_break_waiting[0]
            can_apply = v42_break_can_be_delivered(event, ser, ifc_output_hold)
            if (not can_apply and not event.get("destructive")
                    and not event.get("expedited") and not break_transport_warning):
                pending = getattr(ser, "output_pending", None)
                try:
                    queued = pending() if pending is not None else None
                except OSError:
                    queued = None
                if queued is None:
                    print("[v42] terminal cannot report queued output; BREAK remains "
                          "unacknowledged", file=sys.stderr, flush=True)
                    break_transport_warning = True
            send_break = getattr(ser, "send_break", None)
            length_ms = event.get("length_ms")
            duration_ms = 250 if length_ms is None else max(1, int(length_ms))
            if can_apply and not event.get("dte_action_applied"):
                try:
                    applied = send_break is not None and send_break(duration_ms)
                except OSError:
                    applied = False
                if applied:
                    event["dte_action_applied"] = True
                    remote_break_deadline = max(remote_break_deadline,
                                                time.monotonic() + duration_ms / 1000.0)
                elif not break_transport_warning:
                    print("[v42] terminal transport has no BREAK signal; "
                          "BRK remains unacknowledged", file=sys.stderr, flush=True)
                    break_transport_warning = True
            if event.get("dte_action_applied") and online_session.acknowledge_v42_break():
                remote_break_waiting.pop(0)

        now = time.monotonic()
        if remote_break_deadline and now >= remote_break_deadline:
            remote_break_deadline = 0.0
            if remote_break_post_data:
                queued = bytes(remote_break_post_data)
                remote_break_post_data.clear()
                deliver_received(queued)

    try:
        while True:
            if control is not None:

                control.poll(dce, ser, registration, online_session, iax_registered,
                             iax_configured=iax_transport is not None)
                ser.update_lines(dce.online or not dce.carrier_detect, dce.incoming_ringing, True,

                                 dce.flow_control in (1,3),
                                 ((online_session.transmitter.pending() if online_session else 0) +
                                  (arq_session.pending_bytes()*10 if arq_session else 0)),
                                 rts_flow=dce.ifc[0] == 2, cts_flow=dce.ifc[1] == 2)
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
            if software_loop is not None:
                timeout = min(timeout, software_loop.FRAME_SAMPLES / RTP_CLOCK_RATE)
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
            service_v42_breaks()
            if software_loop is not None:
                deliver_received(software_loop.step())
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
                    elif kind == "registered":
                        iax_registered = True
                    if kind == "answered":
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
                        iax_registered = False
                        print(f"[iax] registration lost: {event.get('reason', 'unknown')}", file=sys.stderr, flush=True)
                if online_session is not None and iax_media_active:
                    for ulaw in iax_transport.drain_audio():
                        deliver_received(online_session.handle_rtp_payload(ulaw))
            if online_session is not None:
                online_session.poll(time.monotonic())
                service_v42_breaks()
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
                    if not set_bell_dte_framing(True):
                        print("[dce] Bell 212A DTE framing is unsupported by this serial device",
                              file=sys.stderr, flush=True)
                        unified_hangup()
                        dce.notify_call_progress("NO CARRIER")
                        continue
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
            if orderly_release.pending:
                session = online_session
                if orderly_release.ready(session):
                    # The peer acknowledged DISC, LAPM exhausted its bounded
                    # retry procedure, or the safety deadline expired.
                    unified_hangup()
            notifications = dce.take_notifications()
            if notifications:
                term_write(notifications)
            escape_detector.guard_time = max(0.02, dce.registers[12] / 50.0)      # S12 in 1/50 s
            escape_detector.plus_char = dce.registers[2]                          # S2
            if data:
                if orderly_release.pending:
                    # Do not allow a new ATD/ATA to replace the still-active
                    # media session while its orderly LAPM release is draining.
                    term_write(dce._result("ERROR"))
                elif dial_abort_applies(active_outbound, online_session, dce.online, connect_pending, iax_dial_pending):
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
                elif data_mode and (online_session is not None or software_loop is not None):
                    passthrough = escape_detector.feed(dce.filter_ifc_input(data), time.monotonic())
                    if passthrough:
                        if os.environ.get("MODEM_DEBUG") == "1":
                            print(f"[dce] terminal transmit bytes={len(passthrough)}", file=sys.stderr, flush=True)
                        if software_loop is not None:
                            software_loop.feed_bytes(passthrough)
                        elif arq_session is not None:
                            arq_session.feed_terminal_bytes(passthrough)
                            framed = arq_session.pump_outgoing(time.monotonic())
                            if framed:
                                online_session.feed_terminal_bytes(framed)
                        else:
                            online_session.feed_terminal_bytes(passthrough)
                        dce.count_chars(sent=len(passthrough))
                    if (online_session is not None
                            and (dce.software_flow_control_enabled() or dce.dte_flow_control_enabled())):
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
                    pending_baud = dce.take_pending_dte_baud()
                    if pending_baud is not None:
                        if not ser.set_dte_baud(pending_baud):
                            print(f"[dce] failed to apply requested DTE baud {pending_baud}",
                                  file=sys.stderr, flush=True)
                    pending_format = dce.take_pending_dte_character_format()
                    if pending_format is not None:
                        if not ser.set_dte_character_format(*pending_format):
                            print(f"[dce] failed to apply requested DTE framing {pending_format}",
                                  file=sys.stderr, flush=True)
            if (
                data_mode
                and (online_session is not None or software_loop is not None)
                and escape_detector.poll(time.monotonic())
            ):
                if online_session is not None and not set_bell_dte_framing(False):
                    print("[dce] failed to restore command-mode DTE framing", file=sys.stderr, flush=True)
                    unified_hangup()
                    dce.notify_call_progress("NO CARRIER")
                    continue
                data_mode = False
                term_write(dce._result("OK"))
            if (
                data_mode
                and online_session is not None
                and online_session.xoff_asserted
                and (dce.software_flow_control_enabled() or dce.dte_flow_control_enabled())
            ):
                # Re-check even if the DTE sent nothing this cycle: once
                # paused with XOFF it won't send more until it sees XON.
                extra = arq_session.pending_bytes() * 10 if arq_session is not None else 0
                signal = online_session.flow_control_signal(extra)
                if signal:
                    term_write(signal)
            flush_ifc_output_hold()
            update_ifc_dte_backpressure()
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
    _configure_runtime_clock()
    parser = argparse.ArgumentParser(prog="el-modem", description=f"{PRODUCT} V{VERSION} - open-source software modem ({PROJECT_URL})")

    def v42_frame_size(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer from 1 to 128") from exc
        if not 1 <= parsed <= 128:
            raise argparse.ArgumentTypeError("must be from 1 to 128 bytes")
        return parsed

    parser.add_argument("--self-test", action="store_true", help="exercise all offline engine factories without opening network or serial ports")
    parser.add_argument("--self-test-payload", action="store_true", help="also verify exact local PCM payload through the all-in-one profile (no network)")
    parser.add_argument("--link", help="create a symlink to the allocated PTY")
    parser.add_argument("--tcp-listen", metavar="HOST:PORT",
                        help="also serve the modem's serial port over TCP (for serial_bridge.py on the host)")
    parser.add_argument("--tcp-baud-control", action="store_true",
                        help="with --tcp-listen, also mirror DTE baud on TCP data-port+1")
    parser.add_argument("--serial-tcp", metavar="HOST:PORT",
                        help="use a TCP listener as the serial port instead of a PTY (the only mode on Windows; "
                        "a client disconnecting is a DTR drop)")
    parser.add_argument("--serial-tcp-baud-control", action="store_true",
                        help="with --serial-tcp, also mirror nominal DTE baud on TCP data-port+1")
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
        choices=[600, 1200, 2400],
        default=1200,
        help="V.22/V.22bis rate to use (600 is initial asynchronous Mode iv)",
    )
    parser.add_argument(
        "--v42-frame-size",
        type=v42_frame_size,
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
    if args.tcp_baud_control and not args.tcp_listen:
        parser.error("--tcp-baud-control requires --tcp-listen")
    if args.serial_tcp_baud_control and not args.serial_tcp:
        parser.error("--serial-tcp-baud-control requires --serial-tcp")
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
    if os.environ.get('V34_NATIVE_DECISION_FEEDBACK')=='1':
        import numpy as np
        from v34_data import QUARTER
        from v34_decision_feedback_native import warmup as warm_decision_feedback
        warm_decision_feedback(np.asarray(QUARTER[:4],dtype=np.complex128))
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
            tcp_baud_control=args.tcp_baud_control,
            sip_user=args.sip_user,
            profile_file=args.profile_file,
            sip_register=args.sip_register,
            sip_contact_host=args.sip_contact_host,
            caller_number=args.caller_number,
            serial_tcp=args.serial_tcp or ("127.0.0.1:9100" if os.name == "nt" else None),
            serial_tcp_baud_control=args.serial_tcp_baud_control,
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
