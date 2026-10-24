"""ITU-T V.23 asymmetric FSK data modem.

The calling terminal uses the forward channel (600 or 1200 baud, mark 1300 Hz,
space 1700 or 2100 Hz). The answering terminal uses the optional backward
channel (75 baud, mark 390 Hz, space 450 Hz). Each endpoint receives the other
channel. There is no training exchange in V.23; the continuous mark idles are
the line signals. This module is the physical/DTE data path only: it does not
claim the V.24 electrical circuits or timing behavior of a hardware DCE.
"""

from __future__ import annotations

import math
import struct

from bell103 import Bell103Receiver, Bell103StreamTransmitter, SAMPLE_RATE
from g711 import linear_to_ulaw, ulaw_to_linear
from audio_path_impairment import AudioPathImpairment

FORWARD_MARK_HZ = 1300.0
FORWARD_SPACE_HZ = {600: 1700.0, 1200: 2100.0}
BACKWARD_MARK_HZ = 390.0
BACKWARD_SPACE_HZ = 450.0
BACKWARD_DETECT_SAMPLES = 320  # 40 ms; resolves the 60-Hz separation
FORWARD_DETECT_SAMPLES = 80  # 10 ms; enough cycles for the forward tone plan
FORWARD_TONE_SUBWINDOW = 10  # noncoherent accumulation avoids FSK phase cancellation
FORWARD_TONE_QUALITY = 0.65
FORWARD_COHERENT_QUALITY = 0.005
BACKWARD_TONE_QUALITY = 0.03
DBM0_REFERENCE_POWER = 6.76e6  # dBm0 normalization used by rx_signal_power_dbm0
CARRIER_ON_DBM0 = -43.0
CARRIER_OFF_DBM0 = -48.0
CARRIER_ON_POWER = DBM0_REFERENCE_POWER * 10 ** (CARRIER_ON_DBM0 / 10.0)
CARRIER_OFF_POWER = DBM0_REFERENCE_POWER * 10 ** (CARRIER_OFF_DBM0 / 10.0)


def _tone_energy(samples: list[int], frequency: float) -> float:
    phase = 2.0 * math.pi * frequency / SAMPLE_RATE
    real = sum(sample * math.cos(phase * i) for i, sample in enumerate(samples))
    imag = sum(sample * math.sin(phase * i) for i, sample in enumerate(samples))
    return (real * real + imag * imag) / max(1, len(samples))


class V23Modem:
    """One V.23 endpoint with one forward or backward DTE byte channel.

    ``calling_party=True`` sends/receives on the forward channel. The answerer
    transmits and receives on the 75-baud backward channel. The default
    backward channel is always present and idles on mark as required by §4;
    its use for application data remains available through the normal DTE byte
    interface. Select 600 or 1200 baud for the forward channel.
    """

    is_v22 = False
    is_v32 = False
    capture_detect_bits = False
    supports_v42 = False
    v42_mode = "off"
    v42_lost = False
    v42_compression = None

    def __init__(self, calling_party: bool, *, forward_baud: int = 1200,
                 amplitude: float = 12000.0, backward_channel: bool = True):
        if forward_baud not in FORWARD_SPACE_HZ:
            raise ValueError("V.23 forward baud must be 600 or 1200")
        self.calling = bool(calling_party)
        self.forward_baud = forward_baud
        self.backward_channel_enabled = bool(backward_channel)
        self.tx_baud = forward_baud if self.calling else 75
        self.rx_baud = 75 if self.calling else forward_baud
        if self.calling:
            tx_freqs = (FORWARD_MARK_HZ, FORWARD_SPACE_HZ[forward_baud])
            rx_freqs = (BACKWARD_MARK_HZ, BACKWARD_SPACE_HZ)
            tx_standard_role = rx_standard_role = "originate"
        else:
            tx_freqs = (BACKWARD_MARK_HZ, BACKWARD_SPACE_HZ)
            rx_freqs = (FORWARD_MARK_HZ, FORWARD_SPACE_HZ[forward_baud])
            tx_standard_role = rx_standard_role = "answer"
        self._tx = Bell103StreamTransmitter(
            tx_standard_role, baud=self.tx_baud, frequencies=tx_freqs,
            amplitude=amplitude,
        )
        self._rx = Bell103Receiver(
            rx_standard_role, baud=self.rx_baud, frequencies=rx_freqs,
            track_clock=not self.calling,
        )
        self.events: list[tuple[str, int]] = []
        self.connected = False
        self._rx_samples = 0
        self._carrier_samples_processed = 0
        self._carrier_transitions: list[dict] = []
        self._carrier_buffer: list[int] = []
        self._carrier_on = self._carrier_off = 0
        self._carrier_block_samples = (
            BACKWARD_DETECT_SAMPLES if self.calling else FORWARD_DETECT_SAMPLES
        )
        # V.23 §8.3: backward detector 122 must assert in <80 ms; forward
        # detector 109 uses the 300-700 ms acquisition interval at call setup.
        # Subsequent acquisition uses the steady-state 10-20 ms interval.
        self._carrier_first_acquisition = True
        self._power = 0.0

    @property
    def bit_rate(self) -> int:
        # CONNECT reports the forward-channel rate at both ends; ATI/DSP stats
        # expose tx_baud/rx_baud to preserve the asymmetry.
        return self.forward_baud

    @property
    def rx_signal_power_dbm0(self) -> float:
        return (10 * math.log10(self._power / DBM0_REFERENCE_POWER + 1e-12)
                if self._power > 0 else -99.0)

    @property
    def v42_resolved(self) -> bool:
        return True

    def is_settled(self) -> bool:
        return self.connected

    def feed_bytes(self, data: bytes) -> None:
        if not self.calling and not self.backward_channel_enabled:
            # An absent optional backward DTE signal is transmitted as mark.
            return
        self._tx.feed_bytes(data)

    def pending_bits(self) -> int:
        return self._tx.pending()

    def tx_samples(self, count: int) -> list[int]:
        return self._tx.samples(count)

    def feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        # Keep byte demodulation ahead of status metering so carrier-quality
        # work cannot delay the receive decoder on a live RTP callback.
        decoded = self._rx.feed_pcm(pcm)
        samples = [value for (value,) in struct.iter_unpack("<h", pcm)]
        self._rx_samples += len(samples)
        self._carrier_buffer.extend(samples)
        while len(self._carrier_buffer) >= self._carrier_block_samples:
            block = self._carrier_buffer[:self._carrier_block_samples]
            del self._carrier_buffer[:self._carrier_block_samples]
            self._update_carrier(block)
            self._carrier_samples_processed += len(block)
        if self.calling and not self.backward_channel_enabled:
            return b""
        return decoded

    def take_carrier_transitions(self) -> list[dict]:
        """Drain bounded circuit-109/122 transitions with exact PCM sample indices."""
        rows = self._carrier_transitions
        self._carrier_transitions = []
        return rows

    def _record_carrier_transition(self, event: str, code: int, state: bool,
                                   block_samples: int) -> None:
        self.events.append((event, code))
        self._carrier_transitions.append({
            "event": event,
            "state": bool(state),
            "sample_index": self._carrier_samples_processed + block_samples,
        })
        if len(self._carrier_transitions) > 64:
            del self._carrier_transitions[:-64]

    def _update_carrier(self, block: list[int]) -> None:
        self._power = sum(sample * sample for sample in block) / len(block)
        mark_hz, space_hz = ((BACKWARD_MARK_HZ, BACKWARD_SPACE_HZ)
                              if self.calling else
                              (FORWARD_MARK_HZ, FORWARD_SPACE_HZ[self.forward_baud]))
        mark, space = _tone_energy(block, mark_hz), _tone_energy(block, space_hz)
        # V.23 §8.3 defines RLSD ON above -43 dBm0 and OFF below -48 dBm0,
        # with the intervening range left to implementation as long as the
        # threshold hysteresis is at least 2 dB.
        power_threshold = CARRIER_OFF_POWER if self.connected else CARRIER_ON_POWER
        power_qualified = self._power >= power_threshold
        hard_power_loss = self._power < (CARRIER_OFF_POWER if self.connected else CARRIER_ON_POWER)
        if self.calling:
            # The 75-baud channel's tones are only 60 Hz apart, so evaluate
            # coherent tone energy over the full 40-ms detector window.
            # Normalize by window length and signal power: unnormalized
            # correlator energy scales with N and broadband noise otherwise
            # passes the old fixed fraction-of-power comparison.
            tone_quality = 2.0 * max(mark, space) / (
                len(block) * self._power) if self._power else 0.0
            tone_present = tone_quality >= BACKWARD_TONE_QUALITY
        else:
            # Forward FSK changes tone at symbol boundaries. Summing the
            # allowed-tone energies from short subwindows noncoherently avoids
            # cancellation between mark and space phases, while normalization
            # rejects broadband energy that is not concentrated in this
            # modem's two-tone plan.
            tone_energy = 0.0
            for offset in range(0, len(block), FORWARD_TONE_SUBWINDOW):
                subwindow = block[offset:offset + FORWARD_TONE_SUBWINDOW]
                if len(subwindow) != FORWARD_TONE_SUBWINDOW:
                    continue
                tone_energy += (_tone_energy(subwindow, mark_hz)
                                + _tone_energy(subwindow, space_hz))
            tone_quality = (2.0 * tone_energy / (len(block) * self._power)
                            if self._power else 0.0)
            # Short windows distinguish broadband noise from tones but have
            # broad frequency response. During acquisition, also require
            # coherent energy at one of the configured frequencies over the
            # full 10-ms window; this rejects strong single tones between or
            # outside the V.23 mark/space pair. Once qualified, tolerate the
            # phase cancellations of valid FSK data and use hysteresis.
            coherent_quality = (
                2.0 * max(mark, space) / (len(block) * self._power)
                if self._power else 0.0
            )
            tone_present = tone_quality >= FORWARD_TONE_QUALITY and (
                self.connected or coherent_quality >= FORWARD_COHERENT_QUALITY
            )
        present = power_qualified and tone_present
        if present:
            self._carrier_on += 1
            self._carrier_off = 0
            if self.calling:
                required_on = 1  # 40 ms, within circuit 122's <80 ms limit
            elif self._carrier_first_acquisition:
                required_on = 30  # 300 ms, lower edge of circuit 109 setup range
            else:
                required_on = 2  # 20 ms, steady-state circuit 109 response
            if not self.connected and self._carrier_on >= required_on:
                self.connected = True
                self._carrier_first_acquisition = False
                self._record_carrier_transition("CARRIER_UP", -2, True, len(block))
        else:
            self._carrier_off += 1
            self._carrier_on = 0
            # Silence is an immediate loss. During valid FSK data a short
            # block can have weak matched-tone energy at a phase transition;
            # tolerate those spectral nulls without hiding actual silence.
            # Two 40-ms backward intervals remain within circuit 122's
            # 15..80-ms release range; three 10-ms forward intervals avoid
            # false circuit-109 drops on ordinary 1200-baud data.
            spectral_miss_limit = 3 if not self.calling else 2
            required_off = 1 if hard_power_loss else spectral_miss_limit
            if self.connected and self._carrier_off >= required_off:
                self.connected = False
                # Do not let a partially sampled UART character straddle a
                # circuit-109/122 carrier interruption and corrupt recovery.
                self._rx.reset_frame_sync()
                self._record_carrier_transition("CARRIER_DOWN", -1, False, len(block))


class V23SoftwareAudioLoop:
    """Exercise V.23 TX/RX through a paired software audio path.

    This is a diagnostic loop, not V.54 Loop 3: it uses a complementary
    V.23 software endpoint and a PCMU encode/decode boundary, but no analog
    line interface or hybrid. Optional bounded sample impairments are synthetic
    digital tests, not an analog-line model. The peer reflects decoded DTE
    bytes on its own asymmetric channel, so returned bytes have traversed
    both DSP transmitters, both receivers, and G.711 quantization.
    """

    FRAME_SAMPLES = 80  # V.23 circuit-109 receive timing uses a 10 ms cadence.

    def __init__(self, *, forward_baud: int = 1200, calling_party: bool = True,
                 backward_channel: bool = True,
                 local_to_peer_impairment: AudioPathImpairment | None = None,
                 peer_to_local_impairment: AudioPathImpairment | None = None):
        self.endpoint = V23Modem(
            calling_party, forward_baud=forward_baud,
            backward_channel=backward_channel)
        self.peer = V23Modem(
            not calling_party, forward_baud=forward_baud,
            backward_channel=backward_channel)
        self.forward_baud = forward_baud
        self.calling_party = bool(calling_party)
        self.backward_channel_enabled = bool(backward_channel)
        self.local_to_peer_impairment = local_to_peer_impairment
        self.peer_to_local_impairment = peer_to_local_impairment
        self.frames = 0
        self.tx_bytes = 0
        self.reflected_bytes = 0
        self.rx_bytes = 0

    @staticmethod
    def _pcmu_roundtrip(samples: list[int]) -> list[int]:
        return [ulaw_to_linear(linear_to_ulaw(sample)) for sample in samples]

    def feed_bytes(self, data: bytes) -> None:
        """Queue DTE bytes on the local endpoint's transmit channel."""
        self.endpoint.feed_bytes(data)
        self.tx_bytes += len(data)

    def step(self) -> bytes:
        """Advance one 10 ms audio frame and return newly looped-back bytes."""
        local_audio = self.endpoint.tx_samples(self.FRAME_SAMPLES)
        peer_audio = self.peer.tx_samples(self.FRAME_SAMPLES)
        local_path = (self.local_to_peer_impairment.process(local_audio)
                      if self.local_to_peer_impairment else local_audio)
        peer_path = (self.peer_to_local_impairment.process(peer_audio)
                     if self.peer_to_local_impairment else peer_audio)
        peer_rx = self.peer.feed_pcm(struct.pack(
            f"<{self.FRAME_SAMPLES}h", *self._pcmu_roundtrip(local_path)))
        local_rx = self.endpoint.feed_pcm(struct.pack(
            f"<{self.FRAME_SAMPLES}h", *self._pcmu_roundtrip(peer_path)))
        if peer_rx:
            self.peer.feed_bytes(peer_rx)
            self.reflected_bytes += len(peer_rx)
        self.frames += 1
        if local_rx:
            self.rx_bytes += len(local_rx)
        return local_rx

    def status(self) -> dict:
        """Expose diagnostic state without implying physical V.54 coverage."""
        return {
            "kind": "software-audio-loop",
            "physical_v54_loop": False,
            "forward_baud": self.forward_baud,
            "calling_party": self.calling_party,
            "backward_channel": self.backward_channel_enabled,
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


__all__ = [
    "V23Modem", "FORWARD_MARK_HZ", "FORWARD_SPACE_HZ",
    "BACKWARD_MARK_HZ", "BACKWARD_SPACE_HZ", "V23SoftwareAudioLoop",
]
