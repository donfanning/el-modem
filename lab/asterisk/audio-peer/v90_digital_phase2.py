"""Digital-modem V.90 Phase 2 INFO frame and waveform primitives.

The frame contents come from V.90 Tables 7 and 8. INFO1d has the same
information-bit layout as V.34 INFO1c. The caller owns the Phase 2 event
timeline (Tone B, probing, ranging, INFO0d repetition/acknowledgement, and
INFO1a receive); this module only builds and modulates one frame at a time.
"""

from __future__ import annotations

import math

import numpy as np

import v34_info
import v90_info_reference as v90


class V90DigitalPhase2Signals:
    """Build digital-side INFO0d and INFO1d signals at the V.90 line rate."""

    def __init__(self, *, info0d: dict | None = None,
                 info1d: dict | None = None, amplitude: float = 3000.0):
        self.info0d_fields = dict(info0d or {})
        self.info1d_fields = dict(info1d or {})
        self.amplitude = float(amplitude)
        self._carrier_phase = 0.0
        self._modulator = v34_info.InfoModulator("call", amplitude=self.amplitude,
                                                  guard=False)
        self._validate()

    def _validate(self) -> None:
        info0_ranges = {
            "rate2743": (0, 1), "rate2800": (0, 1), "rate3429": (0, 1),
            "lo3000": (0, 1), "hi3000": (0, 1), "lo3200": (0, 1),
            "hi3200": (0, 1), "disallow3429": (0, 1),
            "power_reduction": (0, 1), "max_asym": (0, 5), "cme": (0, 1),
            "pts1664": (0, 1), "reserved": (0, 0), "ack_info0a": (0, 1),
            "nominal_power": (0, 15), "max_power": (0, 31),
            "power_at_codec": (0, 1), "alaw": (0, 1),
            "v90_3429_upstream": (0, 1), "reserved_41": (0, 0),
        }
        for name, value in self.info0d_fields.items():
            if name not in info0_ranges:
                raise ValueError(f"unknown INFO0d field {name!r}")
            lo, hi = info0_ranges[name]
            if not lo <= int(value) <= hi:
                raise ValueError(f"INFO0d field {name} must be in {lo}..{hi}")
            if name == "reserved" and int(value) & 1:
                raise ValueError("INFO0d bit 26 (short Phase 2) is not supported")

        allowed_info1 = {"power_min", "power_extra", "md_len", "blocks", "freq_offset"}
        unknown = set(self.info1d_fields) - allowed_info1
        if unknown:
            raise ValueError(f"unknown INFO1d fields: {sorted(unknown)}")
        for key, maximum in (("power_min", 7), ("power_extra", 7), ("md_len", 127)):
            value = int(self.info1d_fields.get(key, 0))
            if not 0 <= value <= maximum:
                raise ValueError(f"INFO1d field {key} must be in 0..{maximum}")
        if not -512 <= int(self.info1d_fields.get("freq_offset", 0)) <= 511:
            raise ValueError("INFO1d freq_offset must be in -512..511")
        blocks = self.info1d_fields.get("blocks", [{}] * 6)
        if len(blocks) != 6:
            raise ValueError("INFO1d requires six rate blocks")
        for block in blocks:
            if not 0 <= int(block.get("hi_carrier", 0)) <= 1:
                raise ValueError("INFO1d hi_carrier must be in 0..1")
            if not 0 <= int(block.get("preemph", 0)) <= 10:
                raise ValueError("INFO1d preemph must be in 0..10")
            if not 0 <= int(block.get("max_rate", 0)) <= 14:
                raise ValueError("INFO1d max_rate must be in 0..14")


    def info0d_bits(self, *, ack_info0a: int | None = None) -> list[int]:
        fields = dict(self.info0d_fields)
        if ack_info0a is not None:
            if ack_info0a not in (0, 1):
                raise ValueError("ack_info0a must be 0 or 1")
            fields["ack_info0a"] = ack_info0a
        return v90.build_info0d(**fields)

    def info1d_bits(self) -> list[int]:
        return v34_info.build_info1c(**self.info1d_fields)

    def _modulate(self, bits: list[int]) -> np.ndarray:
        samples = self._modulator.modulate(bits, tail_bits=8,
                                           start_phase=self._carrier_phase)
        self._carrier_phase = (self._carrier_phase
                               + 2 * math.pi * v34_info.CARRIER["call"]
                               * len(samples) / v34_info.FS) % (2 * math.pi)
        return samples

    def advance_silence(self, sample_count: int) -> None:
        """Keep the carrier phase continuous across an externally generated interval."""
        if not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        self._carrier_phase = (self._carrier_phase
                               + 2 * math.pi * v34_info.CARRIER["call"]
                               * sample_count / v34_info.FS) % (2 * math.pi)

    def set_phase_at_sample(self, sample: int) -> None:
        """Set the INFO carrier phase for an absolute 8 kHz media sample."""
        if not isinstance(sample, int) or sample < 0:
            raise ValueError("media sample must be a nonnegative integer")
        self._carrier_phase = (2 * math.pi * v34_info.CARRIER["call"] * sample
                               / v34_info.FS) % (2 * math.pi)

    def modulate_info0d(self, *, ack_info0a: int | None = None) -> np.ndarray:
        """Return one linear-PCM INFO0d frame; caller controls repeat/turnaround."""
        return self._modulate(self.info0d_bits(ack_info0a=ack_info0a))

    def modulate_info1d(self) -> np.ndarray:
        """Return one linear-PCM INFO1d frame using the V.34 INFO1c bit layout."""
        return self._modulate(self.info1d_bits())


def constrain_info1d_to_info0a(info1d_fields: dict,
                                info0a_fields: dict) -> dict:
    """Return INFO1d with rate blocks consistent with the peer's INFO0a.

    INFO1d is sent by the digital modem, but its six rate blocks recommend
    analogue-modem-to-digital-modem upstream profiles. The peer's INFO0a is
    therefore authoritative for which rates and carrier choices it can use.
    Preserve our probe-derived profile where supported and suppress an
    unsupported choice by setting its projected maximum rate to zero.
    """
    from copy import deepcopy

    result = deepcopy(info1d_fields)
    # The shared V.34 waveform parser calls INFO0a bit 19 ``allow3429``;
    # the V.90 table parser calls the same raw bit ``disallow3429``. Raw 0
    # disallows 3429 and raw 1 permits it.
    allow_3429 = bool(info0a_fields.get(
        "allow3429", info0a_fields.get("disallow3429", 0)))
    if not bool(info0a_fields.get("power_reduction", 0)):
        # Table 9 requires both power-reduction requests to be zero when
        # INFO0a says the analogue transmitter cannot reduce power.
        result["power_min"] = 0
        result["power_extra"] = 0
    blocks = list(result.get("blocks", [{} for _ in range(6)]))
    if len(blocks) != 6:
        raise ValueError("INFO1d requires six rate blocks")

    def supports(index: int, carrier: int) -> bool:
        if index == 0:  # 2400 symbols/s is the V.34 base rate.
            return True
        if index == 1:
            return bool(info0a_fields.get("rate2743", 0))
        if index == 2:
            return bool(info0a_fields.get("rate2800", 0))
        if index == 3:
            return bool(info0a_fields.get("hi3000" if carrier else "lo3000", 0))
        if index == 4:
            return bool(info0a_fields.get("hi3200" if carrier else "lo3200", 0))
        if index == 5:
            # Table 8 says bit 19 = 0 means 3429 symbols/s is disallowed.
            return bool(info0a_fields.get("rate3429", 0)) and allow_3429
        return False

    for index, raw in enumerate(blocks):
        block = dict(raw)
        carrier = int(block.get("hi_carrier", 0))
        if index in (3, 4) and not supports(index, carrier):
            alternate = 1 - carrier
            if supports(index, alternate):
                carrier = alternate
                block["hi_carrier"] = carrier
        if not supports(index, carrier):
            block["max_rate"] = 0
        elif int(block.get("max_rate", 0)) > 12 and not bool(
                info0a_fields.get("pts1664", 0)):
            block["max_rate"] = 12
        blocks[index] = block
    result["blocks"] = blocks
    return result


def project_info1d_rates_from_probe(info1d_fields: dict,
                                    probe_quality: dict | None) -> dict:
    """Limit configured INFO1d rate caps using measured L2 probe quality.

    The six configured ``max_rate`` values are treated as local capability
    ceilings. When per-tone measurements are available, estimate each block
    from a lower-quartile SNR score inside that V.34 carrier's occupied band;
    this lets selective attenuation reduce rates that need the damaged band.
    The 7.8 dB margin and Shannon-style estimate match El-Modem's existing V.34
    projection policy, while the lower-quartile selection is an additional
    conservative implementation heuristic. ITU-T V.90 defines the fields and
    exchange but does not mandate this estimator. If no usable measurements
    are available, leave the configured ceilings intact.
    """
    from copy import deepcopy

    result = deepcopy(info1d_fields)
    if not probe_quality or "snr_db" not in probe_quality:
        return result
    snr_db = float(probe_quality["snr_db"])
    if not math.isfinite(snr_db):
        return result
    snr_db = max(-120.0, min(120.0, snr_db))
    rates = (2400, 2743, 2800, 3000, 3200, 3429)
    blocks = list(result.get("blocks", [{} for _ in rates]))
    if len(blocks) != len(rates):
        raise ValueError("INFO1d requires six rate blocks")
    tone_snr = probe_quality.get("tone_snr_db_by_hz")
    if isinstance(tone_snr, dict) and tone_snr:
        from v34_tx import CARRIERS, SYMBOL_RATES

        def rate_snr(baud: float, block: dict) -> float:
            carrier_index = int(block.get("hi_carrier", 0))
            carrier_set = CARRIERS.get(round(baud))
            if (carrier_set is None
                    or not 0 <= carrier_index < len(carrier_set)):
                return snr_db
            carrier = carrier_set[carrier_index]
            half_band = baud * (1.0 + 0.15) / 2.0
            low, high = max(0.0, carrier - half_band), min(4000.0,
                                                            carrier + half_band)
            band = [float(value) for freq, value in tone_snr.items()
                    if low <= float(freq) <= high]
            if len(band) < 4:
                return snr_db
            return float(np.percentile(band, 25))

        baud_rates = tuple(float(SYMBOL_RATES[index]) for index in range(6))
        get_snr = rate_snr
    else:
        baud_rates = tuple(float(rate) for rate in rates)
        get_snr = lambda baud, block: snr_db

    for index, (baud, raw) in enumerate(zip(rates, blocks)):
        block = dict(raw)
        configured_cap = int(block.get("max_rate", 0))
        effective_snr = get_snr(baud_rates[index], block)
        bits_per_symbol = math.floor(
            math.log2(1.0 + 10.0 ** ((effective_snr - 7.8) / 10.0)))
        measured_limit = max(0, min(14, math.floor(
            bits_per_symbol * baud_rates[index] / 2400)))
        block["max_rate"] = min(configured_cap, measured_limit)
        blocks[index] = block
    result["blocks"] = blocks
    return result
