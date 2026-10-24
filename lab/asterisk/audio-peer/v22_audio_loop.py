"""Paired V.22/V.22bis DSP audio loop for software diagnostics.

This routes two production V.22-family DSP engines through PCMU. It does not
run the V.25 call procedure, RTP, an analog hybrid, or a physical V.54 loop.
"""
import struct

from audio_path_impairment import AudioPathImpairment
from g711 import linear_to_ulaw, ulaw_to_linear
from v22 import V22Modem


class V22SoftwareAudioLoop:
    FRAME_SAMPLES = 160

    def __init__(self, *, bit_rate: int = 2400, calling_party: bool = True,
                 profile: str | None = None,
                 local_to_peer_impairment: AudioPathImpairment | None = None,
                 peer_to_local_impairment: AudioPathImpairment | None = None):
        if bit_rate not in (600, 1200, 2400):
            raise ValueError("software V.22 loop supports 600, 1200 or 2400 bit/s")
        profile = profile or ("v22bis" if bit_rate == 2400 else "v22")
        if profile not in ("v22", "v22bis", "bell212a"):
            raise ValueError("unsupported software V.22-family profile")
        if ((profile == "v22bis" and bit_rate != 2400)
                or (profile == "bell212a" and bit_rate != 1200)
                or (profile == "v22" and bit_rate not in (600, 1200))):
            raise ValueError("software profile and line rate are incompatible")
        self.profile = profile
        self.endpoint = V22Modem(calling_party, bit_rate, profile=profile)
        self.peer = V22Modem(not calling_party, bit_rate, profile=profile)
        self.bit_rate = bit_rate
        self.calling_party = bool(calling_party)
        self.local_to_peer_impairment = local_to_peer_impairment
        self.peer_to_local_impairment = peer_to_local_impairment
        self.frames = 0
        self.tx_bytes = 0
        self.reflected_bytes = 0
        self.rx_bytes = 0
        self.training_frames = 0
        for frame in range(1500):
            self._exchange_audio()
            if self.endpoint.is_settled() and self.peer.is_settled():
                self.training_frames = frame + 1
                break
        else:
            self.close()
            raise ValueError("paired V.22 software loop did not train")

    @staticmethod
    def _pcmu_roundtrip(samples):
        return [ulaw_to_linear(linear_to_ulaw(sample)) for sample in samples]

    def feed_bytes(self, data: bytes) -> None:
        self.endpoint.feed_bytes(data)
        self.tx_bytes += len(data)

    def feed_peer_bytes(self, data: bytes) -> None:
        """Queue data from the local software peer, for duplex DSP tests."""
        self.peer.feed_bytes(data)

    def step(self) -> bytes:
        local_rx, _peer_rx = self.step_duplex()
        return local_rx

    def _exchange_audio(self) -> tuple[bytes, bytes]:
        local = self.endpoint.tx_samples(self.FRAME_SAMPLES)
        peer = self.peer.tx_samples(self.FRAME_SAMPLES)
        local_path = (self.local_to_peer_impairment.process(local)
                      if self.local_to_peer_impairment else local)
        peer_path = (self.peer_to_local_impairment.process(peer)
                     if self.peer_to_local_impairment else peer)
        peer_rx = self.peer.feed_pcm(struct.pack(
            "<160h", *self._pcmu_roundtrip(local_path)))
        local_rx = self.endpoint.feed_pcm(struct.pack(
            "<160h", *self._pcmu_roundtrip(peer_path)))
        return local_rx, peer_rx

    def step_duplex(self) -> tuple[bytes, bytes]:
        """Advance the DSP pair and return bytes received at local/peer DTEs."""
        local_rx, peer_rx = self._exchange_audio()
        if peer_rx:
            self.peer.feed_bytes(peer_rx)
            self.reflected_bytes += len(peer_rx)
        self.frames += 1
        self.rx_bytes += len(local_rx)
        return local_rx, peer_rx

    def status(self) -> dict:
        return {
            "kind": "software-audio-loop",
            "physical_v54_loop": False,
            "standard": {"v22": "V.22", "v22bis": "V.22bis",
                         "bell212a": "Bell 212A"}[self.profile],
            "profile": self.profile,
            "bit_rate": self.bit_rate,
            "calling_party": self.calling_party,
            "frames": self.frames,
            "training_frames": self.training_frames,
            "endpoint_carrier": self.endpoint.connected,
            "peer_carrier": self.peer.connected,
            "endpoint_settled": self.endpoint.is_settled(),
            "peer_settled": self.peer.is_settled(),
            "tx_bytes": self.tx_bytes,
            "reflected_bytes": self.reflected_bytes,
            "rx_bytes": self.rx_bytes,
            "local_to_peer_impairment": (self.local_to_peer_impairment.report()
                                          if self.local_to_peer_impairment else None),
            "peer_to_local_impairment": (self.peer_to_local_impairment.report()
                                         if self.peer_to_local_impairment else None),
        }

    def close(self) -> None:
        self.endpoint.close()
        self.peer.close()


__all__ = ["V22SoftwareAudioLoop"]
