"""Bell 103/V.21 (300 bit/s) FSK line engine.

Wraps the stream transmitter and receiver of ``bell103.py`` and adds remote-
carrier detection. The default Bell 103 profile is used by ``V8Chain`` fallback;
the V.21 profile is also used for software audio-loop diagnostics. For Bell 103,
the detector rejects the 2100-Hz V.25 answer tone. Async 8N1 only; no error
control (V.42 stays off).
"""
import math
import struct

from bell103 import SAMPLE_RATE, STANDARDS, Bell103Receiver, Bell103StreamTransmitter
from g711 import linear_to_ulaw, ulaw_to_linear
from audio_path_impairment import AudioPathImpairment
from v42_layer import V42Layer

BLOCK = 160                                   # 20 ms detector blocks


def _energy(block, freq):
    w = 2.0 * math.pi * freq / SAMPLE_RATE
    re = sum(s * math.cos(w * i) for i, s in enumerate(block))
    im = sum(s * math.sin(w * i) for i, s in enumerate(block))
    return (re * re + im * im) / len(block)


class Bell103Engine(V42Layer):
    is_v32 = False
    is_v22 = False
    supports_v42 = False
    capture_detect_bits = False
    bit_rate = 300

    def __init__(self, calling_party: bool, *, amplitude: float = 12000.0,
                 standard: str = "bell103"):
        if standard not in STANDARDS:
            raise ValueError(f"unsupported FSK standard: {standard}")
        self.calling = calling_party
        self.standard = standard
        self.events: list[tuple[str, int]] = []
        self.connected = False
        # V.21 answer-channel transmitters send the 1800-Hz guard with the
        # data carrier. V.25 answer sequencing is handled by OnlineAudioSession
        # for direct calls; a V.8Chain hands over after V.8 has completed.
        guard_hz, guard_amp = ((1800.0, 6000.0)
                               if standard == "v21" and not calling_party else (0.0, 0.0))
        self._tx = Bell103StreamTransmitter(
            "originate" if calling_party else "answer", amplitude=amplitude,
            standard=standard, guard_hz=guard_hz, guard_amp=guard_amp)
        self._rx = Bell103Receiver("answer" if calling_party else "originate",
                                   standard=standard)
        self._peer = STANDARDS[standard]["answer" if calling_party else "originate"]
        self._buf: list[int] = []
        self._on = self._off = 0
        self._power = 0.0
        self._v42_setup()

    @property
    def rx_signal_power_dbm0(self) -> float:
        return 10 * math.log10(self._power / 6.76e6 + 1e-12) if self._power > 0 else -99.0

    def is_settled(self) -> bool:
        return self.connected

    def feed_bytes(self, data: bytes) -> None:
        self._tx.feed_bytes(data)

    def pending_bits(self) -> int:
        return self._tx.pending()

    def select_after_qualified_mark(self) -> None:
        """Clear stale frames while carrying the already-qualified mark edge state."""
        self._rx.prime_after_mark()

    def tx_samples(self, count: int) -> list[int]:
        return self._tx.samples(count)

    def feed_pcm(self, pcm: bytes) -> bytes:
        self._buf.extend(s for (s,) in struct.iter_unpack("<h", pcm))
        while len(self._buf) >= BLOCK:
            block, self._buf = self._buf[:BLOCK], self._buf[BLOCK:]
            self._carrier(block)
        decoded = self._rx.feed_pcm(pcm)
        # The async tone slicer can frame random mark/space decisions during
        # silence or noise. A modem must not expose those as DTE data while its
        # receive carrier is down. Continue feeding the slicer so mark timing
        # is ready when carrier is acquired, but gate its bytes on carrier.
        return decoded if self.connected else b""

    def _carrier(self, block) -> None:
        self._power = sum(s * s for s in block) / len(block)
        mark, space = (_energy(block, f) for f in self._peer)
        between = _energy(block, 2100.0) if self.calling else 0.0     # the V.25 answer tone is not a Bell 103 carrier
        # Correlator energy grows with the window length, while broadband
        # noise also produces nonzero mark/space energy. Normalize the
        # strongest tone to the full-window power before applying the carrier
        # threshold; the previous unnormalized comparison accepted noise as
        # carrier and allowed its random async decisions to escape to the DTE.
        purity = (4.0 * max(mark, space) / (len(block) * self._power)
                  if self._power > 0 else 0.0)
        present = (self._power > 80.0 ** 2 and purity > 0.25
                   and max(mark, space) > 3.0 * between)
        if present:
            self._on, self._off = self._on + 1, 0
            if not self.connected and self._on >= 5:
                self.connected = True
                self.events.append(("CARRIER_UP", -2))
        else:
            self._off, self._on = self._off + 1, 0
            if self.connected and self._off >= 25:
                self.connected = False
                self.events.append(("CARRIER_DOWN", -1))


class SoftwareFskAudioLoop:
    """Paired FSK DSP/PCMU test loop; not a physical V.54 Loop 3.

    This is an end-to-end data-path diagnostic for the implemented Bell 103
    and V.21 FSK engines. A complementary software endpoint reflects decoded
    DTE bytes so the test traverses both modulators, demodulators and a PCMU
    boundary without using RTP or claiming analog-line coverage. Optional
    sample impairments are synthetic and disabled by default.
    """

    FRAME_SAMPLES = 160  # 20 ms at the modem's 8-kHz audio rate.

    def __init__(self, *, standard: str = "bell103", calling_party: bool = True,
                 local_to_peer_impairment: AudioPathImpairment | None = None,
                 peer_to_local_impairment: AudioPathImpairment | None = None):
        if standard not in ("bell103", "v21"):
            raise ValueError(f"unsupported software FSK loop profile: {standard}")
        self.endpoint = Bell103Engine(calling_party, standard=standard)
        self.peer = Bell103Engine(not calling_party, standard=standard)
        self.standard = standard
        self.calling_party = bool(calling_party)
        self.local_to_peer_impairment = local_to_peer_impairment
        self.peer_to_local_impairment = peer_to_local_impairment
        self.frames = 0
        self.tx_bytes = 0
        self.reflected_bytes = 0
        self.rx_bytes = 0
        self._pending_tx = bytearray()

    @staticmethod
    def _pcmu_roundtrip(samples: list[int]) -> bytes:
        encoded = bytes(linear_to_ulaw(sample) for sample in samples)
        return b"".join(struct.pack("<h", ulaw_to_linear(value)) for value in encoded)

    def feed_bytes(self, data: bytes) -> None:
        # A real DTE starts writing after CONNECT, when the opposite FSK
        # carrier has already qualified. This diagnostic loop reports CONNECT
        # before its two software endpoints have exchanged a carrier lead, so
        # hold early DTE bytes until both receive detectors are up instead of
        # sending them into the carrier-acquisition interval.
        self._pending_tx.extend(data)
        self.tx_bytes += len(data)

    def step(self) -> bytes:
        local = self.endpoint.tx_samples(self.FRAME_SAMPLES)
        peer = self.peer.tx_samples(self.FRAME_SAMPLES)
        local_path = (self.local_to_peer_impairment.process(local)
                      if self.local_to_peer_impairment else local)
        peer_path = (self.peer_to_local_impairment.process(peer)
                     if self.peer_to_local_impairment else peer)
        peer_rx = self.peer.feed_pcm(self._pcmu_roundtrip(local_path))
        local_rx = self.endpoint.feed_pcm(self._pcmu_roundtrip(peer_path))
        if (self._pending_tx and self.endpoint.connected and self.peer.connected):
            self.endpoint.feed_bytes(bytes(self._pending_tx))
            self._pending_tx.clear()
        if peer_rx:
            self.peer.feed_bytes(peer_rx)
            self.reflected_bytes += len(peer_rx)
        self.frames += 1
        self.rx_bytes += len(local_rx)
        return local_rx

    def status(self) -> dict:
        return {
            "kind": "software-fsk-audio-loop",
            "physical_v54_loop": False,
            "standard": self.standard,
            "calling_party": self.calling_party,
            "frames": self.frames,
            "endpoint_carrier": self.endpoint.connected,
            "peer_carrier": self.peer.connected,
            "tx_bytes": self.tx_bytes,
            "reflected_bytes": self.reflected_bytes,
            "rx_bytes": self.rx_bytes,
            "local_to_peer_impairment": (self.local_to_peer_impairment.report()
                                          if self.local_to_peer_impairment else None),
            "peer_to_local_impairment": (self.peer_to_local_impairment.report()
                                         if self.peer_to_local_impairment else None),
        }
