import unittest
import wave
from pathlib import Path

import numpy as np

import v34_info
import v90_info_reference as v90
from v34_call import probing_signal
from v90_digital_phase2 import project_info1d_rates_from_probe
from v90_digital_phase2_rx import V90DigitalPhase2FrameRx
from v90_digital_phase2_rx import (V90DigitalPhase2SignalRx,
                                   estimate_probe_frequency_offset_units,
                                   measure_probe_quality)
from v90_digital_phase2_adapter import V90DigitalPhase2Adapter
from v90_digital_phase2_control import V90DigitalPhase2Controller
from v90_digital_phase2 import V90DigitalPhase2Signals
from v90_digital_phase2_control import Phase2Action
from v90_digital_phase2_tx import V90DigitalPhase2Transmitter


class DigitalPhase2FrameRxTests(unittest.TestCase):
    def test_incremental_baseband_matches_full_window_filter_across_wrap(self):
        rng = np.random.default_rng(20261103)
        signal = rng.normal(0.0, 1000.0, 16000)
        rx = V90DigitalPhase2FrameRx(phases=8)

        for start in range(0, len(signal), 160):
            rx.feed(signal[start:start + 160])
            if rx._until_scan == 0:
                cached = rx._baseband_window()
                reference = rx._demod._baseband(rx._rx) * np.exp(
                    -2j * np.pi * rx._demod.fc * rx._base / 8000)
                np.testing.assert_allclose(cached, reference, rtol=1e-10,
                                           atol=1e-7)

    def test_chunked_analog_info_frames_emit_crc_valid_deduplicated_events(self):
        # The analog V.90 caller uses the shared API's 2400 Hz "answer" carrier
        # label and sends its 1800 Hz guard tone alongside it.
        mod = v34_info.InfoModulator("answer", amplitude=3000, guard=True)
        info0a = mod.modulate(v34_info.build_info0(rate3429=1, ack=0), tail_bits=2)
        info1a = mod.modulate(v90.build_info1a_v90(
            md_len=0, analog_symbol_rate=5, digital_symbol_rate=6,
            freq_offset=-3), tail_bits=2)
        signal = np.concatenate((np.zeros(800), info0a, np.zeros(2400),
                                 info1a, np.zeros(1600)))

        rx = V90DigitalPhase2FrameRx(phases=8)
        events = []
        for start in range(0, len(signal), 320):
            events.extend(rx.feed(signal[start:start + 320]))

        self.assertEqual([event.kind for event in events], ["info0a", "info1a"])
        self.assertEqual(events[0].fields["rate3429"], 1)
        self.assertEqual(events[0].fields["ack"], 0)
        self.assertEqual(events[1].fields["mode"], 6)
        self.assertEqual(events[1].fields["v90_analog_symbol_rate"], 5)
        self.assertEqual(events[1].fields["v90_digital_symbol_rate"], 6)
        self.assertEqual(events[1].fields["v90_freq_offset"], -3)
        self.assertLess(abs(events[0].at - (800 + 13)), 16)
        self.assertLess(abs(events[1].at - (800 + len(info0a) + 2400 + 13)), 16)
        self.assertGreater(events[0].complete_at, events[0].at)
        self.assertGreater(events[1].complete_at, events[1].at)
        self.assertEqual(rx.samples_seen, len(signal))

    def test_invalid_crc_is_not_emitted(self):
        bits = v34_info.build_info0(rate2743=1)
        bits[29] ^= 1
        signal = v34_info.InfoModulator("answer", guard=True).modulate(bits, tail_bits=2)
        rx = V90DigitalPhase2FrameRx()
        events = []
        for start in range(0, len(signal), 400):
            events.extend(rx.feed(signal[start:start + 400], kinds=("info0",)))
        self.assertEqual(events, [])


class DigitalPhase2SignalRxTests(unittest.TestCase):
    def test_gough_phase2_probe_capture_regression(self):
        # These are paired Gough handset/ISP captures already in the fixture
        # set. Channel 1 is the analogue caller in both, verified from its
        # CRC-valid V.90 INFO1a selector. Start just before the decoded INFO0a
        # and stop at INFO1a; the receiver must locate L1/L2 within that
        # protocol-bounded interval. This checks the estimator on recorded
        # modem PCM; the captures do not reveal the peer's internal rate fit.
        fixtures = (
            ("Conexant-HCF-smooth-crescendo.wav", 66078, 90596,
             [14, 14, 14, 14, 14, 14]),
            ("Agere-Mars1648C-bipbipbipbip.wav", 57296, 85138,
             [13, 14, 14, 14, 14, 14]),
        )
        directory = Path(__file__).parent / "fixtures" / "stereo"
        for filename, start, stop, expected_caps in fixtures:
            with self.subTest(capture=filename):
                with wave.open(str(directory / filename), "rb") as wav:
                    self.assertEqual(wav.getframerate(), 8000)
                    self.assertEqual(wav.getnchannels(), 2)
                    self.assertEqual(wav.getsampwidth(), 2)
                    pcm = np.frombuffer(wav.readframes(wav.getnframes()),
                                        dtype="<i2").reshape(-1, 2)
                analogue_channel = pcm[start:stop, 1].astype(float)
                receiver = V90DigitalPhase2SignalRx()
                events = []
                for offset in range(0, len(analogue_channel), 160):
                    events.extend(receiver.feed(
                        analogue_channel[offset:offset + 160],
                        expect="probe"))
                completed = [event for event in events
                             if event.kind == "l2_complete"]
                self.assertEqual(len(completed), 1)
                self.assertIsNotNone(receiver.probe_quality)
                projected = project_info1d_rates_from_probe(
                    {"blocks": [{"max_rate": 14} for _ in range(6)]},
                    receiver.probe_quality)
                self.assertEqual([block["max_rate"]
                                  for block in projected["blocks"]],
                                 expected_caps)

    def test_phase2_receiver_projects_lower_rates_on_tilted_noisy_l2(self):
        def capture(*, tilted, noise_rms, seed):
            def channel(samples):
                frequencies = np.fft.rfftfreq(len(samples), 1 / 8000)
                response = np.ones_like(frequencies)
                if tilted:
                    response[(frequencies > 1500) & (frequencies < 1800)] = .1
                    response[frequencies >= 1800] = .04
                return np.fft.irfft(np.fft.rfft(samples) * response,
                                    n=len(samples))

            pcm = np.concatenate((
                np.zeros(800),
                channel(probing_signal(.160, 1200)),
                channel(probing_signal(.500, 600)),
                np.zeros(800),
            ))
            pcm += np.random.default_rng(seed).normal(0, noise_rms, len(pcm))
            receiver = V90DigitalPhase2SignalRx()
            events = []
            for start in range(0, len(pcm), 160):
                events.extend(receiver.feed(pcm[start:start + 160],
                                            expect="probe"))
            l2 = next(event for event in events if event.kind == "l2_complete")
            self.assertIn("probe_quality", l2.fields)
            profile = project_info1d_rates_from_probe(
                {"blocks": [{"max_rate": 14} for _ in range(6)]},
                receiver.probe_quality,
            )
            return [block["max_rate"] for block in profile["blocks"]]

        clean_rates = capture(tilted=False, noise_rms=3, seed=1)
        impaired_rates = capture(tilted=True, noise_rms=20, seed=21)
        self.assertTrue(all(impaired < clean for impaired, clean
                            in zip(impaired_rates, clean_rates)))

    def test_probe_quality_uses_comb_and_omitted_bins(self):
        fs = 8000
        n = 4000
        t = np.arange(n) / fs
        tones = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950,
                 2100, 2250, 2550, 2700, 2850, 3000, 3150, 3300, 3450,
                 3600, 3750)
        probe = sum(np.cos(2 * np.pi * f * t) for f in tones)
        rng = np.random.default_rng(9201)
        clean = measure_probe_quality(
            probe + rng.normal(0, 5, n), frequency_offset_units=0)
        noisy = measure_probe_quality(
            probe + np.random.default_rng(9201).normal(0, 40, n),
            frequency_offset_units=0)
        self.assertIsNotNone(clean)
        self.assertIsNotNone(noisy)
        self.assertEqual(len(clean["tone_rms_by_hz"]), 21)
        self.assertEqual(len(clean["omitted_rms_by_hz"]), 4)
        self.assertGreater(clean["snr_db"], noisy["snr_db"] + 10)
        self.assertIsNone(measure_probe_quality(np.zeros(n)))

    def test_probe_offset_estimator_reports_signed_info1d_units(self):
        fs = 8000
        tones = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950,
                 2100, 2250, 2550, 2700, 2850, 3000, 3150, 3300, 3450,
                 3600, 3750)
        t = np.arange(4000) / fs
        for offset, expected in ((0.10, 5), (-0.10, -5), (0.0, 0)):
            signal = sum(np.cos(2 * np.pi * (f + offset * f / 1050.0) * t)
                         for f in tones)
            signal *= 600 / np.sqrt(np.mean(signal ** 2))
            with self.subTest(offset=offset):
                self.assertEqual(estimate_probe_frequency_offset_units(signal), expected)
                if offset == 0.10:
                    noisy = signal + np.random.default_rng(42).normal(0, 30, signal.size)
                    self.assertEqual(estimate_probe_frequency_offset_units(noisy), expected)

        self.assertEqual(estimate_probe_frequency_offset_units(np.zeros(4000)), -512)

    def test_first_probe_offset_is_attached_to_l2_completion(self):
        fs = 8000
        tones = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950,
                 2100, 2250, 2550, 2700, 2850, 3000, 3150, 3300, 3450,
                 3600, 3750)
        def probe(duration, offset_hz):
            t = np.arange(int(duration * fs)) / fs
            signal = sum(np.cos(2 * np.pi * (f + offset_hz * f / 1050.0) * t)
                         for f in tones)
            return signal * (600 / np.sqrt(np.mean(signal ** 2)))

        signal = np.concatenate((np.zeros(800), probe(.160, 0.10),
                                 probe(.500, 0.10), np.zeros(400)))
        rx = V90DigitalPhase2SignalRx()
        events = []
        for start in range(0, len(signal), 160):
            events.extend(rx.feed(signal[start:start + 160], expect="probe"))
        done = next(event for event in events if event.kind == "l2_complete")
        self.assertEqual(done.fields["freq_offset"], 5)
        self.assertGreater(done.fields["probe_quality"]["snr_db"], 50)
        self.assertEqual(rx.probe_frequency_offset_units, 5)

    def test_guarded_tone_a_phase_reversal_survives_odd_chunk_sizes(self):
        fs = 8000
        n = 8000
        t = np.arange(n)
        phase = np.where(t >= 3200, np.pi, 0.0)
        signal = (1000 * np.cos(2 * np.pi * 2400 * t / fs + phase)
                  + 300 * np.cos(2 * np.pi * 1800 * t / fs))
        rx = V90DigitalPhase2SignalRx()
        events = []
        for start in range(0, n, 73):
            events.extend(rx.feed(signal[start:start + 73], expect="tone_a"))
        tone = next(event for event in events if event.kind == "tone_a")
        reversal = next(event for event in events if event.kind == "tone_a_reversal")
        self.assertEqual(tone.at, 0)
        self.assertLessEqual(abs(reversal.at - 3200), 1)
        self.assertLess(reversal.fields["normalized_fit_error"], 1e-8)

    def test_guarded_tone_a_phase_reversal_fit_handles_noise(self):
        fs = 8000
        start = 2000
        split = 33
        n = 60
        t = start + np.arange(n)
        phase = np.where(np.arange(n) >= split, np.pi, 0.0)
        rng = np.random.default_rng(90210)
        signal = (900 * np.cos(2 * np.pi * 2400 * t / fs + phase + 0.31)
                  + 240 * np.cos(2 * np.pi * 1800 * t / fs - 0.6)
                  + rng.normal(0, 40, n))
        rx = V90DigitalPhase2SignalRx()
        rx._tone_audio_history = signal
        fitted = rx._fit_a_reversal(start)
        self.assertIsNotNone(fitted)
        self.assertLessEqual(abs(fitted[0] - start - split), 1)
        self.assertLess(fitted[1], 0.02)

    def test_comb_probe_reports_160ms_l1_and_bounded_500ms_l2(self):
        fs = 8000
        tones = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950,
                 2100, 2250, 2550, 2700, 2850, 3000, 3150, 3300, 3450,
                 3600, 3750)
        t = np.arange(int(.16 * fs)) / fs
        l1 = sum(np.cos(2 * np.pi * f * t) for f in tones)
        l1 *= 1200 / np.sqrt(np.mean(l1 ** 2))
        t2 = np.arange(int(.5 * fs)) / fs
        l2 = sum(np.cos(2 * np.pi * f * t2) for f in tones)
        l2 *= 600 / np.sqrt(np.mean(l2 ** 2))
        signal = np.concatenate((np.zeros(800), l1, l2, np.zeros(400)))
        rx = V90DigitalPhase2SignalRx()
        events = []
        for start in range(0, len(signal), 333):
            events.extend(rx.feed(signal[start:start + 333], expect="probe"))
        l1_done = next(event for event in events if event.kind == "l1_complete")
        l2_done = next(event for event in events if event.kind == "l2_complete")
        self.assertLessEqual(abs(l1_done.at - (800 + 1280)), 160)
        self.assertEqual(l2_done.at - l1_done.at, 4000)

    def test_tone_only_and_silence_do_not_open_probe_interval(self):
        fs = 8000
        t = np.arange(4000)
        tone = 1000 * np.cos(2 * np.pi * 2400 * t / fs)
        rx = V90DigitalPhase2SignalRx()
        events = rx.feed(tone, expect="probe")
        self.assertEqual(events, [])

    def test_l1_completion_is_not_early_across_detector_window_phases(self):
        fs = 8000
        tones = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950,
                 2100, 2250, 2550, 2700, 2850, 3000, 3150, 3300, 3450,
                 3600, 3750)
        t = np.arange(5000) / fs
        probe = sum(np.cos(2 * np.pi * f * t) for f in tones)
        probe *= 900 / np.sqrt(np.mean(probe ** 2))
        for onset_phase in range(0, 160, 10):
            with self.subTest(onset_phase=onset_phase):
                onset = 800 + onset_phase
                signal = np.concatenate((np.zeros(onset), probe, np.zeros(5000)))
                rx = V90DigitalPhase2SignalRx()
                events = []
                for start in range(0, len(signal), 137):
                    events.extend(rx.feed(signal[start:start + 137], expect="probe"))
                l1_done = next(event for event in events if event.kind == "l1_complete")
                expected = onset + int(.160 * fs)
                self.assertGreaterEqual(l1_done.at, expected)
                self.assertLessEqual(l1_done.at - expected, fs // 200)


class DigitalPhase2AdapterTests(unittest.TestCase):
    def test_info_scans_are_limited_to_controller_state(self):
        controller = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(
            controller, rx_to_media=lambda sample: sample)
        self.assertEqual(adapter._expected_frame_kinds(), ("info0",))
        controller.state = controller.WAIT_A_REV2
        self.assertEqual(adapter._expected_frame_kinds(), ("info0",))
        controller.state = controller.RX_PROBE
        self.assertEqual(adapter._expected_frame_kinds(), ("info0",))
        controller.state = controller.WAIT_A_AFTER_PROBE
        self.assertEqual(adapter._expected_frame_kinds(), ())
        controller.state = controller.TX_PROBE
        self.assertEqual(adapter._expected_frame_kinds(), ())
        controller.state = controller.WAIT_INFO1A
        self.assertEqual(adapter._expected_frame_kinds(), ("info1a",))
        controller.state = controller.WAIT_INFO1A_RETRAIN
        self.assertEqual(adapter._expected_frame_kinds(), ("info1a",))
        self.assertEqual(adapter._expected_signal(), "tone_a")
        info1a = type("Event", (), {
            "kind": "info1a", "at": 1, "complete_at": 2,
            "fields": {"mode": 6},
        })()
        reversal = type("Event", (), {
            "kind": "tone_a_reversal", "at": 3, "complete_at": 4,
            "fields": {},
        })()
        self.assertEqual(adapter._route(info1a), ("info1a", info1a.fields))
        controller.info0a = {"ack": 0}
        controller.state = controller.WAIT_RETRAIN_A_REVERSAL
        self.assertEqual(adapter._route(reversal),
                         ("retrain_tone_a_reversal", reversal.fields))

    def test_second_probe_wait_routes_returned_l2_and_keeps_tone_a_detection(self):
        controller = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(controller, rx_to_media=lambda sample: sample)
        controller.state = controller.TX_PROBE
        self.assertEqual(adapter._expected_signal(), "all")
        event = type("Event", (), {
            "kind": "l2_complete", "at": 900, "complete_at": 4900,
            "fields": {"probe": "returned"},
        })()
        self.assertEqual(adapter._route(event), ("l2_echo_complete", event.fields))

    def test_maps_valid_info0a_to_common_media_clock(self):
        controller = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(
            controller, rx_to_media=lambda sample: sample + 100_000)
        self.assertEqual([a.signal for a in adapter.start(100_000)],
                         ["info0d", "tone_b"])
        mod = v34_info.InfoModulator("answer", amplitude=3000, guard=True)
        info0a = mod.modulate(v34_info.build_info0(rate3429=1, ack=0), tail_bits=2)
        signal = np.concatenate((np.zeros(800), info0a, np.zeros(1200)))
        offset = 0
        for start in range(0, len(signal), 320):
            chunk = signal[start:start + 320]
            adapter.feed(chunk, rx_at=5000 + offset, tx_now=200_000)
            offset += len(chunk)
        self.assertIsNotNone(controller.info0a)
        self.assertEqual(controller.info0a["ack"], 0)
        self.assertEqual(adapter.frames.samples_seen, len(signal))

    def test_adapter_and_renderer_answer_duplicate_info0a_then_resume_tone_b(self):
        """Exercise the production event-to-PCM path, including frame timing."""
        config = {"rate3429": 1, "alaw": 0}
        controller = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(
            controller, rx_to_media=lambda sample: sample, trace_events=True)
        tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(info0d=config, amplitude=1000),
            probe_builder=lambda seconds, rms: np.zeros(
                int(round(seconds * 8000)), dtype=float),
            tone_b_peak=700, probe_rms=300)
        tx.submit(adapter.start(0))

        mod = v34_info.InfoModulator("answer", amplitude=3000, guard=True)
        info0a = mod.modulate(v34_info.build_info0(rate3429=1, ack=0),
                              tail_bits=2)
        silence = np.zeros(4000)
        received = np.concatenate((np.zeros(800), info0a, silence,
                                   info0a, np.zeros(5000)))
        rendered = []
        ack_starts = []
        for start in range(0, len(received), 160):
            chunk = received[start:start + 160]
            actions = adapter.feed(chunk, rx_at=start, tx_now=tx.cursor)
            if any(action.signal == "info0d" and action.value == 1
                   for action in actions):
                ack_starts.append(tx.cursor)
            tx.submit(actions)
            rendered.append(tx.render(len(chunk)))
        rendered.append(tx.render(1600))
        rendered = np.concatenate(rendered)

        routed = [row for row in adapter.event_trace
                  if row["event"] == "info0a"]
        self.assertEqual(len(routed), 2, adapter.event_trace)
        self.assertEqual(len(ack_starts), 1, adapter.event_trace)
        ack_at = ack_starts[0]
        ack_signals = V90DigitalPhase2Signals(info0d=config, amplitude=1000)
        ack_signals.set_phase_at_sample(ack_at)
        expected_ack = ack_signals.modulate_info0d(ack_info0a=1)
        np.testing.assert_array_equal(
            rendered[ack_at:ack_at + len(expected_ack)], expected_ack)
        tone_at = ack_at + len(expected_ack)
        expected_tone = 700 * np.cos(
            2 * np.pi * 1200 * np.arange(tone_at, tone_at + 80) / 8000)
        np.testing.assert_allclose(rendered[tone_at:tone_at + 80],
                                   expected_tone, atol=1e-9)

    def test_duplicate_info0a_restarts_ranging_before_probe_receive(self):
        """Reactive PCM path: restart A/B ranging after INFO0a recovery."""
        config = {"rate3429": 1, "alaw": 0}
        controller = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(
            controller, rx_to_media=lambda sample: sample, trace_events=True)
        tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(info0d=config, amplitude=1000),
            probe_builder=lambda seconds, rms: np.zeros(
                int(round(seconds * 8000)), dtype=float),
            tone_b_peak=700, probe_rms=300)
        tx.submit(adapter.start(0))

        mod = v34_info.InfoModulator("answer", amplitude=3000, guard=True)
        info0a = mod.modulate(v34_info.build_info0(rate3429=1, ack=0),
                              tail_bits=2)
        tone_samples = np.arange(8000)
        phase = np.where(tone_samples >= 400, np.pi, 0.0)
        tone_a_with_reversal = (
            1000 * np.cos(2 * np.pi * 2400 * tone_samples / 8000 + phase)
            + 300 * np.cos(2 * np.pi * 1800 * tone_samples / 8000))
        rx_at = 0
        rendered = []

        def deliver(signal):
            nonlocal rx_at
            actions = []
            for offset in range(0, len(signal), 160):
                chunk = signal[offset:offset + 160]
                produced = adapter.feed(
                    chunk, rx_at=rx_at, tx_now=tx.cursor)
                rx_at += len(chunk)
                actions.extend(produced)
                tx.submit(produced)
                rendered.append(tx.render(len(chunk)))
            return actions

        deliver(np.concatenate((np.zeros(800), info0a, np.zeros(2000))))
        first = deliver(tone_a_with_reversal)
        self.assertEqual(controller.state, controller.WAIT_A_REV2)
        self.assertEqual(controller._a_rev_count, 1)
        self.assertEqual([action.signal for action in first],
                         ["reverse-tone-b", "silence"])

        retry = deliver(np.concatenate((np.zeros(500), info0a,
                                        np.zeros(5000))))
        self.assertEqual([action.signal for action in retry],
                         ["info0d", "tone_b"])
        self.assertEqual(controller._a_rev_count, 0)
        self.assertIsNone(controller.b_reversal_at)

        restarted_first = deliver(tone_a_with_reversal)
        self.assertEqual(controller.state, controller.WAIT_A_REV2)
        self.assertEqual(controller._a_rev_count, 1)
        self.assertEqual([action.signal for action in restarted_first],
                         ["reverse-tone-b", "silence"])

        second = deliver(tone_a_with_reversal)
        self.assertEqual(controller.state, controller.RX_PROBE)
        self.assertEqual(controller._a_rev_count, 2)
        self.assertEqual(second, [])

        # Real SmartLink repeats INFO0a after the server has already entered
        # first-probe receive. The adapter must still scan and route it, abort
        # the incomplete probe, and return to the fresh ranging exchange.
        repeated_after_probe = deliver(np.concatenate((np.zeros(500), info0a,
                                                       np.zeros(5000))))
        self.assertEqual([action.signal for action in repeated_after_probe],
                         ["info0d", "tone_b"])
        self.assertEqual(controller.state, controller.WAIT_A_REV2)
        self.assertEqual(controller._a_rev_count, 0)

        post_recovery_first = deliver(tone_a_with_reversal)
        self.assertEqual([action.signal for action in post_recovery_first],
                         ["reverse-tone-b", "silence"])
        self.assertEqual(controller.state, controller.WAIT_A_REV2)
        post_recovery_second = deliver(tone_a_with_reversal)
        self.assertEqual(post_recovery_second, [])
        self.assertEqual(controller.state, controller.RX_PROBE)

        rendered.append(tx.render(1600))
        self.assertEqual(len(np.concatenate(rendered)), tx.cursor)
        reversals = [row for row in adapter.event_trace
                     if row["event"] == "tone_a_reversal"]
        self.assertEqual(len(reversals), 5, adapter.event_trace)
        self.assertEqual([row["state_after"] for row in reversals],
                         [controller.WAIT_A_REV2,
                          controller.WAIT_A_REV2,
                          controller.RX_PROBE,
                          controller.WAIT_A_REV2,
                          controller.RX_PROBE])
        self.assertEqual([row["actions"] for row in reversals],
                         [["reverse-tone-b", "silence"],
                          ["reverse-tone-b", "silence"], [],
                          ["reverse-tone-b", "silence"], []])

    def test_reversal_detected_before_crc_frame_is_replayed_after_info0a(self):
        """A slow CRC scan must not erase a later, valid Tone A reversal."""
        controller = V90DigitalPhase2Controller()
        adapter = V90DigitalPhase2Adapter(
            controller, rx_to_media=lambda sample: sample, trace_events=True)
        adapter.start(0)

        mod = v34_info.InfoModulator("answer", amplitude=3000, guard=True)
        info0a = mod.modulate(v34_info.build_info0(rate3429=1, ack=0),
                              tail_bits=2)
        after_info = len(info0a)
        tone_a = np.arange(8000) / 8000
        phase = np.where(np.arange(8000) >= 400, np.pi, 0.0)
        following_tone = (1000 * np.cos(2 * np.pi * 2400 * tone_a + phase)
                          + 300 * np.cos(2 * np.pi * 1800 * tone_a))
        signal = np.concatenate((np.zeros(800), info0a, following_tone,
                                 np.zeros(2400)))

        actions = []
        for start in range(0, len(signal), 160):
            chunk = signal[start:start + 160]
            actions.extend(adapter.feed(chunk, rx_at=start,
                                        tx_now=start + len(chunk)))

        self.assertIsNotNone(controller.info0a)
        self.assertEqual(controller.info0a["ack"], 0)
        self.assertEqual(controller.state, controller.WAIT_A_REV2)
        self.assertEqual(controller._a_rev_count, 1)
        self.assertTrue(any(action.signal == "reverse-tone-b" for action in actions))

        trace_kinds = [entry["event"] for entry in adapter.event_trace]
        self.assertIn("tone_a_reversal", trace_kinds)
        self.assertIn("info0a", trace_kinds)
        self.assertTrue(adapter.signal_trace)
        self.assertTrue(all(
            all(entry[key] >= 0.0 for key in
                ("signal_cpu_ms", "frame_cpu_ms", "route_cpu_ms"))
            for entry in adapter.signal_trace))
        self.assertTrue(any(entry["frame_cpu_ms"] > 0.0
                            for entry in adapter.signal_trace))

    def test_rejects_unreported_receive_clock_gap(self):
        adapter = V90DigitalPhase2Adapter(
            V90DigitalPhase2Controller(), rx_to_media=lambda sample: sample)
        adapter.start(0)
        adapter.feed(np.zeros(80), rx_at=1000, tx_now=1000)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            adapter.feed(np.zeros(80), rx_at=1200, tx_now=1200)


class DigitalPhase2TransmitterTests(unittest.TestCase):
    @staticmethod
    def _probe(duration, rms):
        count = round(duration * 8000)
        t = np.arange(count) / 8000
        return rms * np.sqrt(2) * np.cos(2 * np.pi * 900 * t)

    def test_renders_info_then_continuous_tone_b_and_exact_reversal(self):
        config = {"rate3429": 1, "alaw": 0}
        expected_builder = V90DigitalPhase2Signals(info0d=config, amplitude=1000)
        expected_builder.set_phase_at_sample(100)
        expected = expected_builder.modulate_info0d(ack_info0a=0)
        tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(info0d=config, amplitude=1000),
            probe_builder=self._probe, tone_b_peak=700, probe_rms=300,
        )
        tx.submit([Phase2Action(100, "info0d", 0),
                   Phase2Action(None, "tone_b", after="info0d-complete")])
        pcm = tx.render(100 + len(expected) + 100)
        np.testing.assert_array_equal(pcm[100:100 + len(expected)], expected)
        tone_start = 100 + len(expected)
        np.testing.assert_allclose(pcm[tone_start:tone_start + 80],
                                   700 * np.cos(2 * np.pi * 1200
                                                * np.arange(tone_start, tone_start + 80) / 8000),
                                   atol=1e-9)

        tx.submit([Phase2Action(tx.cursor + 40, "reverse-tone-b")])
        before = tx.render(40)
        after = tx.render(1)
        sample_at = tx.cursor - 1
        self.assertEqual(len(before), 40)
        self.assertAlmostEqual(after[0], -700 * np.cos(2 * np.pi * 1200
                                                       * sample_at / 8000), places=8)

    def test_repeated_info0a_ack_resumes_tone_b_after_actual_frame_end(self):
        config = {"rate3429": 1, "alaw": 0}
        controller = V90DigitalPhase2Controller()
        tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(info0d=config, amplitude=1000),
            probe_builder=self._probe, tone_b_peak=700, probe_rms=300,
        )
        tx.submit(controller.start(0))
        initial_end = tx._last_info0d_complete
        tx.render(initial_end + 80)

        controller.feed("info0a", 10, payload={"ack_info0d": 0})
        actions = controller.feed("info0a", 20,
                                  payload={"ack_info0d": 0})
        ack_at = tx.cursor
        expected_signals = V90DigitalPhase2Signals(info0d=config, amplitude=1000)
        expected_signals.set_phase_at_sample(ack_at)
        expected_ack = expected_signals.modulate_info0d(ack_info0a=1)
        tx.submit(actions)

        # Cross the finite-frame boundary with irregular render sizes. The
        # Tone B onset must follow the actual INFO0d completion, not a poll or
        # the nominal frame duration.
        pieces = [tx.render(137), tx.render(503),
                  tx.render(len(expected_ack) + 80 - 640)]
        rendered = np.concatenate(pieces)
        np.testing.assert_array_equal(rendered[:len(expected_ack)], expected_ack)
        tone_at = ack_at + len(expected_ack)
        expected_tone = 700 * np.cos(
            2 * np.pi * 1200 * np.arange(tone_at, tone_at + 80) / 8000)
        np.testing.assert_allclose(rendered[len(expected_ack):], expected_tone,
                                   atol=1e-9)

    def test_rapid_duplicate_info0a_acks_serialize_before_resuming_tone_b(self):
        config = {"rate3429": 1, "alaw": 0}
        controller = V90DigitalPhase2Controller()
        tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(info0d=config, amplitude=1000),
            probe_builder=self._probe, tone_b_peak=700, probe_rms=300,
        )
        tx.submit(controller.start(0))
        tx.render((tx._last_info0d_complete or 0) + 80)

        # Two duplicate frames can be reported before the first recovery ACK
        # reaches its scheduled boundary. Their INFO0d responses must queue,
        # and Tone B must resume only after the final actual frame completion.
        controller.feed("info0a", 10, payload={"ack_info0d": 0})
        first_actions = controller.feed(
            "info0a", 20, payload={"ack_info0d": 0})
        tx.submit(first_actions)
        second_actions = controller.feed(
            "info0a", 30, payload={"ack_info0d": 0})
        tx.submit(second_actions)

        first_ack_at = tx.cursor
        first_signals = V90DigitalPhase2Signals(info0d=config, amplitude=1000)
        first_signals.set_phase_at_sample(first_ack_at)
        first_ack = first_signals.modulate_info0d(ack_info0a=1)
        second_ack_at = first_ack_at + len(first_ack)
        second_signals = V90DigitalPhase2Signals(info0d=config, amplitude=1000)
        second_signals.set_phase_at_sample(second_ack_at)
        second_ack = second_signals.modulate_info0d(ack_info0a=1)

        rendered = tx.render(len(first_ack) + len(second_ack) + 80)
        np.testing.assert_array_equal(rendered[:len(first_ack)], first_ack)
        np.testing.assert_array_equal(
            rendered[len(first_ack):len(first_ack) + len(second_ack)],
            second_ack)
        tone_start = first_ack_at + len(first_ack) + len(second_ack)
        expected_tone = 700 * np.cos(
            2 * np.pi * 1200 * np.arange(tone_start, tone_start + 80) / 8000)
        np.testing.assert_allclose(rendered[-80:], expected_tone, atol=1e-9)

    def test_finite_probe_actions_have_exact_durations_and_return_to_silence(self):
        tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(amplitude=1000),
            probe_builder=self._probe, tone_b_peak=700, probe_rms=300,
        )
        tx.submit([Phase2Action(0, "l1"),
                   Phase2Action(1280, "l2", 400),
                   Phase2Action(1680, "silence")])
        pcm = tx.render(1800)
        self.assertTrue(np.any(pcm[:1280]))
        self.assertTrue(np.any(pcm[1280:1680]))
        np.testing.assert_array_equal(pcm[1680:], np.zeros(120))


if __name__ == "__main__":
    unittest.main()
