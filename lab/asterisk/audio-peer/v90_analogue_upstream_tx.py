"""V.90 analogue-caller upstream B1 and V.34 data-symbol transmitter.

The V.90 analog modem uses the answer-role GPA scrambler convention for its
V.34 upstream stream. This component maps caller/LAPM bits into continuous
V.34 symbols; carrier pulse shaping and sample-clock pacing remain with
``StreamQam``.
"""
from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np

import v34_data
import v34_tx


class V90AnalogueUpstreamTransmitter:
    """Generate upstream B1 then payload symbols with continuous state.

    Call :meth:`encode_b1` once at the Phase 4 boundary. Its all-ones input is
    scrambled and encoded using the selected V.34 data profile. Subsequent
    :meth:`encode_bits` calls accept unscrumbled line bits (normally LAPM) and
    preserve both scrambler and V.34 mapper state across frame boundaries.
    Returned symbols are normalized to mean energy two for the shared V.34
    ``StreamQam`` pulse-shaper convention.
    """

    def __init__(self, profile, *, power_estimate=None):
        self.profile = profile
        self.mapper = v34_data.DataTx(
            int(profile.rate), int(round(profile.baud)),
            expanded=bool(profile.shaping),
            precoder=v34_data.precoder_from_mp(profile.precoder),
            nonlinear=bool(profile.nonlinear),
            trellis=int(profile.trellis),
            power_estimate=power_estimate)
        if not np.isfinite(self.mapper.avg_power) or self.mapper.avg_power <= 0:
            raise ValueError("V.90 upstream profile has invalid transmit power")
        self._scale = float(np.sqrt(2.0 / self.mapper.avg_power))
        self._scrambler = v34_tx.Scrambler("answer")
        self._b1_sent = False

    @property
    def bits_per_data_frame(self) -> int:
        return int(self.mapper.bits_per_data_frame)

    @property
    def symbol_power(self) -> float:
        return 2.0

    @property
    def b1_sent(self) -> bool:
        return self._b1_sent

    def encode_b1(self) -> list[complex]:
        """Return one spec-pattern B1 frame and advance all TX memories."""
        if self._b1_sent:
            raise RuntimeError("V.90 analogue upstream B1 was already sent")
        scrambled = [self._scrambler.step(1)
                     for _ in range(self.bits_per_data_frame)]
        self._b1_sent = True
        return self._map_scrambled_frame(scrambled)

    def encode_bits(self, bits: Iterable[int]) -> list[complex]:
        """Encode complete, unscrumbled upstream data frames after B1."""
        if not self._b1_sent:
            raise RuntimeError("V.90 analogue upstream data cannot precede B1")
        values = list(bits)
        if any(isinstance(bit, bool) or not isinstance(bit, Integral)
               or bit not in (0, 1) for bit in values):
            raise ValueError("V.90 upstream input must contain bits")
        if len(values) % self.bits_per_data_frame:
            raise ValueError("V.90 upstream input must contain complete V.34 data frames")
        output: list[complex] = []
        for offset in range(0, len(values), self.bits_per_data_frame):
            scrambled = [self._scrambler.step(int(bit))
                         for bit in values[offset:offset + self.bits_per_data_frame]]
            output.extend(self._map_scrambled_frame(scrambled))
        return output

    def _map_scrambled_frame(self, bits: list[int]) -> list[complex]:
        return [complex(symbol) * self._scale
                for symbol in self.mapper.data_frame(bits)]


def upstream_power_estimate_job(profile) -> v34_data.DataTxPowerEstimate:
    """Return the time-sliceable power estimate ``DataTx`` would compute inline.

    The 40-frame fixed-seed simulation takes about 100 ms. Run in one piece at
    MP acceptance, it stalls the caller's media thread, and the far end hears
    a gap in the middle of CP.
    """
    rate, baud = int(profile.rate), int(round(profile.baud))
    return v34_data.DataTxPowerEstimate(
        rate, baud, expanded=bool(profile.shaping),
        precoder=v34_data.precoder_from_mp(profile.precoder),
        nonlinear=bool(profile.nonlinear), trellis=int(profile.trellis),
        estimate_frames=v34_data._power_estimate_frames(rate, baud))
