"""V.90 digital-side Phase 2 INFO frame and modulator tests."""

import unittest

import numpy as np
import v34_info
import v90_info_reference as v90
from v90_digital_phase2 import (
    V90DigitalPhase2Signals,
    constrain_info1d_to_info0a,
    project_info1d_rates_from_probe,
)
from v90_digital_phase2_rx import measure_probe_quality
from v34_call import probing_signal


class V90DigitalPhase2Tests(unittest.TestCase):
    def test_info1d_rate_blocks_follow_peer_info0a_capabilities(self):
        blocks = [
            {"hi_carrier": 0, "preemph": 1, "max_rate": rate}
            for rate in (8, 9, 10, 11, 12, 14)
        ]
        blocks[3]["hi_carrier"] = 1  # peer only offers the low 3000-baud carrier
        fields = constrain_info1d_to_info0a(
            {"blocks": blocks, "power_min": 3, "power_extra": 4},
            {
                "rate2743": 0, "rate2800": 1, "rate3429": 1,
                "lo3000": 1, "hi3000": 0,
                "lo3200": 0, "hi3200": 1,
                "disallow3429": 1, "pts1664": 0,
                "power_reduction": 0,
            },
        )
        got = fields["blocks"]
        self.assertEqual((fields["power_min"], fields["power_extra"]), (0, 0))
        self.assertEqual(got[0]["max_rate"], 8)
        self.assertEqual(got[1]["max_rate"], 0)  # 2743 not offered
        self.assertEqual(got[2]["max_rate"], 10)
        self.assertEqual((got[3]["hi_carrier"], got[3]["max_rate"]), (0, 11))
        self.assertEqual(got[4]["max_rate"], 12)  # no 1664-point support
        self.assertEqual(got[5]["max_rate"], 12)
        self.assertEqual(blocks[3]["hi_carrier"], 1)  # input is not mutated

    def test_info1d_accepts_shared_v34_parser_name_for_raw_3429_bit(self):
        constrained = constrain_info1d_to_info0a(
            {"blocks": [{"max_rate": 0} for _ in range(5)]
                        + [{"max_rate": 12}]},
            {"rate3429": 1, "allow3429": 1},
        )
        self.assertEqual(constrained["blocks"][5]["max_rate"], 12)
        denied = constrain_info1d_to_info0a(
            {"blocks": [{"max_rate": 0} for _ in range(5)]
                        + [{"max_rate": 12}]},
            {"rate3429": 1, "allow3429": 0},
        )
        self.assertEqual(denied["blocks"][5]["max_rate"], 0)

    def test_probe_quality_lowers_per_rate_projection_with_configured_caps(self):
        configured = {"blocks": [{"max_rate": 14} for _ in range(6)]}
        low = project_info1d_rates_from_probe(configured, {"snr_db": 20.0})
        high = project_info1d_rates_from_probe(configured, {"snr_db": 50.0})
        self.assertEqual([b["max_rate"] for b in low["blocks"]],
                         [4, 4, 4, 5, 5, 5])
        self.assertEqual([b["max_rate"] for b in high["blocks"]],
                         [14] * 6)
        capped = project_info1d_rates_from_probe(
            {"blocks": [{"max_rate": 8} for _ in range(6)]},
            {"snr_db": 50.0},
        )
        self.assertEqual([b["max_rate"] for b in capped["blocks"]], [8] * 6)
        self.assertEqual(configured["blocks"][0]["max_rate"], 14)

    def test_tilted_noisy_l2_lowers_rate_projections(self):
        fs = 8000
        n = fs // 2
        probe = probing_signal(.5, 400)
        frequencies = np.fft.rfftfreq(n, 1 / fs)
        spectrum = np.fft.rfft(probe)
        response = np.ones_like(frequencies)
        response[(frequencies > 1500) & (frequencies < 1800)] = .1
        response[frequencies >= 1800] = .04
        impaired = np.fft.irfft(spectrum * response, n=n)
        impaired += np.random.default_rng(21).normal(0, 20, n)
        clean = probe + np.random.default_rng(1).normal(0, 3, n)

        caps = {"blocks": [{"max_rate": 14} for _ in range(6)]}
        clean_projection = project_info1d_rates_from_probe(
            caps, measure_probe_quality(clean, frequency_offset_units=0))
        impaired_quality = measure_probe_quality(
            impaired, frequency_offset_units=0)
        impaired_projection = project_info1d_rates_from_probe(caps, impaired_quality)
        clean_rates = [block["max_rate"] for block in clean_projection["blocks"]]
        impaired_rates = [block["max_rate"] for block in impaired_projection["blocks"]]
        self.assertEqual(clean_rates, [14] * 6)
        self.assertTrue(all(got < clean for got, clean in
                            zip(impaired_rates, clean_rates)))
        self.assertLess(impaired_quality["tone_snr_db_by_hz"][3750],
                        impaired_quality["tone_snr_db_by_hz"][750] - 15)

    def test_info1d_rejects_out_of_standard_field_values(self):
        with self.assertRaisesRegex(ValueError, "preemph"):
            V90DigitalPhase2Signals(info1d={"blocks": [
                {"preemph": 11}, {}, {}, {}, {}, {},
            ]})
        with self.assertRaisesRegex(ValueError, "max_rate"):
            V90DigitalPhase2Signals(info1d={"blocks": [
                {"max_rate": 15}, {}, {}, {}, {}, {},
            ]})
        with self.assertRaisesRegex(ValueError, "max_asym"):
            V90DigitalPhase2Signals(info0d={"max_asym": 6})

    def test_info0d_ack_is_clear_then_set_and_both_frames_decode(self):
        signals = V90DigitalPhase2Signals(info0d={
            "rate2743": 1, "rate2800": 1, "rate3429": 1,
            "lo3000": 1, "hi3000": 1, "lo3200": 1, "hi3200": 1,
            "disallow3429": 0, "max_asym": 3, "nominal_power": 6,
            "max_power": 23, "power_at_codec": 1, "alaw": 1,
        })
        first = signals.modulate_info0d(ack_info0a=0)
        signals.advance_silence(8)  # short interval stands in for external Tone B
        second = signals.modulate_info0d(ack_info0a=1)
        decoded = v90.decode_info0d(np.concatenate((first, second)),
                                    sender_role="call", phases=24)
        fields = [item for item in decoded if "_start_sample" in item]
        unique = []
        for item in fields:
            if not unique or abs(item["_start_sample"] - unique[-1]["_start_sample"]) > 20:
                unique.append(item)
        self.assertGreaterEqual(len(unique), 2)
        self.assertEqual(unique[0]["ack_info0a"], 0)
        self.assertEqual(unique[1]["ack_info0a"], 1)
        self.assertEqual(unique[1]["alaw"], 1)

    def test_info1d_uses_v34_info1c_frame_fields_and_waveform(self):
        settings = {
            "power_min": 2, "power_extra": 3, "md_len": 0,
            "blocks": [
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 1, "max_rate": 12},
                {"hi_carrier": 0, "preemph": 1, "max_rate": 13},
                {"hi_carrier": 0, "preemph": 1, "max_rate": 14},
            ],
            "freq_offset": -2,
        }
        signals = V90DigitalPhase2Signals(info1d=settings)
        bits = signals.info1d_bits()
        parsed = v34_info.parse_info1c(bits)
        self.assertEqual(parsed["power_min"], 2)
        self.assertEqual(parsed["blocks"][5]["max_rate"], 14)
        self.assertEqual(parsed["freq_offset"], -2)
        decoded = v34_info.InfoDemodulator("call", phases=24).decode(
            signals.modulate_info1d(), "info1c")
        self.assertTrue(any(item["freq_offset"] == -2
                            and item["blocks"][5]["max_rate"] == 14
                            for item in decoded))

    def test_reserved_info0d_bits_and_bad_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            V90DigitalPhase2Signals(info0d={"reserved": 1})
        signals = V90DigitalPhase2Signals()
        with self.assertRaisesRegex(ValueError, "ack_info0a"):
            signals.info0d_bits(ack_info0a=2)
        with self.assertRaisesRegex(ValueError, "six rate blocks"):
            V90DigitalPhase2Signals(info1d={"blocks": [{}]})


if __name__ == "__main__":
    unittest.main()
