"""V.23 asymmetric forward/backward-channel tests."""

import math
import random
import struct
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from bell103 import Bell103StreamTransmitter
from audio_path_impairment import AudioPathImpairment
from v23 import (
    BACKWARD_MARK_HZ, BACKWARD_SPACE_HZ, FORWARD_MARK_HZ,
    FORWARD_SPACE_HZ, DBM0_REFERENCE_POWER, V23Modem,
)
from v23 import V23SoftwareAudioLoop


class V23Tests(unittest.TestCase):
    @staticmethod
    def tone_block(frequency, start_sample, count, amplitude=12000):
        return struct.pack(
            f"<{count}h",
            *(round(amplitude * math.sin(2 * math.pi * frequency *
                                    (start_sample + i) / 8000))
              for i in range(count)),
        )

    @staticmethod
    def tone_peak_for_dbm0(level_dbm0):
        return round(math.sqrt(2 * 6.76e6 * 10 ** (level_dbm0 / 10)))

    def test_v23_carrier_detector_response_times(self):
        # Circuit 109: initial OFF->ON uses the 300-700 ms call-establishment
        # interval; ON->OFF uses the 5-15 ms steady-state interval.
        forward_rx = V23Modem(False, forward_baud=1200)
        for block in range(29):
            forward_rx.feed_pcm(self.tone_block(1300, block * 80, 80))
        self.assertFalse(forward_rx.connected)
        forward_rx.feed_pcm(self.tone_block(1300, 29 * 80, 80))
        self.assertTrue(forward_rx.connected)
        self.assertEqual(forward_rx.events[-1], ("CARRIER_UP", -2))
        self.assertEqual(forward_rx.take_carrier_transitions(), [{
            "event": "CARRIER_UP", "state": True, "sample_index": 2400,
        }])
        forward_rx.feed_pcm(bytes(80 * 2))
        self.assertFalse(forward_rx.connected)
        self.assertEqual(forward_rx.events[-1], ("CARRIER_DOWN", -1))
        self.assertEqual(forward_rx.take_carrier_transitions(), [{
            "event": "CARRIER_DOWN", "state": False, "sample_index": 2480,
        }])

        # Circuit 122: the optional backward-channel detector has a <80 ms
        # assertion and a 15-80 ms release interval.
        backward_rx = V23Modem(True, forward_baud=1200)
        backward_rx.feed_pcm(self.tone_block(390, 0, 320))
        self.assertTrue(backward_rx.connected)
        self.assertEqual(backward_rx.events[-1], ("CARRIER_UP", -2))
        backward_rx.feed_pcm(bytes(320 * 2))
        self.assertFalse(backward_rx.connected)
        self.assertEqual(backward_rx.events[-1], ("CARRIER_DOWN", -1))

        # After the initial setup interval, forward reacquisition uses the
        # steady-state 10-20 ms response rather than repeating call setup.
        for block in range(2):
            forward_rx.feed_pcm(self.tone_block(1300, 30 * 80 + block * 80, 80))
        self.assertTrue(forward_rx.connected)

    def test_recommendation_channel_frequencies_and_asymmetry(self):
        self.assertEqual(FORWARD_MARK_HZ, 1300.0)
        self.assertEqual(FORWARD_SPACE_HZ, {600: 1700.0, 1200: 2100.0})
        self.assertEqual((BACKWARD_MARK_HZ, BACKWARD_SPACE_HZ), (390.0, 450.0))

    def test_forward_receiver_recovers_slow_known_byte_stream(self):
        # The Minitel compatibility WAV's non-oracle baud estimate was near
        # 1161. This known-byte vector verifies frame-edge timing recovery at
        # that offset without treating the WAV or minimodem as a byte oracle.
        payload = random.Random(231161).randbytes(256)
        tx = Bell103StreamTransmitter(
            "originate", baud=1161,
            frequencies=(FORWARD_MARK_HZ, FORWARD_SPACE_HZ[1200]),
        )
        rx = V23Modem(False, forward_baud=1200)
        received = bytearray()

        def feed(count):
            for offset in range(0, count, 160):
                block = tx.samples(min(160, count - offset))
                received.extend(rx.feed_pcm(struct.pack(f"<{len(block)}h", *block)))

        feed(3200)  # 400-ms answerer carrier-acquisition interval
        self.assertTrue(rx.connected)
        tx.feed_bytes(payload)
        feed(int((len(payload) * 10 + 30) * 8000 / tx.baud) + 1600)
        self.assertEqual(bytes(received), payload)
        self.assertAlmostEqual(rx._rx.baud, 1161, delta=1)

    def test_v23_carrier_thresholds_and_hysteresis_use_dbm0_limits(self):
        # The Recommendation requires assertion above -43 dBm0 and release
        # below -48 dBm0; behavior between the two levels is unspecified.
        acquire = V23Modem(False, forward_baud=1200)
        for block in range(30):
            acquire.feed_pcm(self.tone_block(1300, block * 80, 80,
                                             self.tone_peak_for_dbm0(-42)))
        self.assertTrue(acquire.connected, acquire.rx_signal_power_dbm0)
        self.assertGreater(acquire.rx_signal_power_dbm0, -43.0)

        reject = V23Modem(False, forward_baud=1200)
        for block in range(40):
            reject.feed_pcm(self.tone_block(1300, block * 80, 80,
                                            self.tone_peak_for_dbm0(-44)))
        self.assertFalse(reject.connected)

        # An established detector holds in the unspecified 5-dB hysteresis
        # band, then drops on the first hard-low-power block.
        held = V23Modem(False, forward_baud=1200)
        mark_amp = self.tone_peak_for_dbm0(-40)
        for block in range(30):
            held.feed_pcm(self.tone_block(1300, block * 80, 80, mark_amp))
        self.assertTrue(held.connected)
        start = 30 * 80
        for block in range(5):
            held.feed_pcm(self.tone_block(1300, start + block * 80, 80,
                                          self.tone_peak_for_dbm0(-46)))
        self.assertTrue(held.connected)
        held.feed_pcm(self.tone_block(1300, start + 5 * 80, 80,
                                     self.tone_peak_for_dbm0(-49)))
        self.assertFalse(held.connected)

    def test_broadband_noise_above_on_power_threshold_is_not_carrier(self):
        # The power criterion alone is not sufficient: white noise at this
        # level is above the V.23 -43 dBm0 ON threshold, but has no stable
        # concentration in the configured mark/space pair.
        noise_rms = math.sqrt(DBM0_REFERENCE_POWER * 10 ** (-42.0 / 10.0))
        for calling, baud in ((False, 1200), (False, 600), (True, 1200)):
            with self.subTest(calling=calling, forward_baud=baud):
                modem = V23Modem(calling, forward_baud=baud)
                rng = random.Random(23042 + baud + int(calling))
                for _ in range(100):  # 2 s, enough to exceed initial acquisition
                    samples = [max(-32768, min(32767, round(rng.gauss(0, noise_rms))))
                               for _ in range(160)]
                    modem.feed_pcm(struct.pack("<160h", *samples))
                self.assertGreater(modem.rx_signal_power_dbm0, -43.0)
                self.assertFalse(modem.connected, "broadband noise asserted carrier")

    def test_forward_off_plan_tones_above_on_power_threshold_are_not_carrier(self):
        signal_rms = math.sqrt(DBM0_REFERENCE_POWER * 10 ** (-42.0 / 10.0))
        for baud, frequencies in ((1200, (1100, 1500)), (600, (1500, 1900))):
            for frequency in frequencies:
                with self.subTest(baud=baud, frequency=frequency):
                    modem = V23Modem(False, forward_baud=baud)
                    amplitude = math.sqrt(2.0) * signal_rms
                    samples = [round(amplitude * math.sin(
                        2.0 * math.pi * frequency * i / 8000.0
                    )) for i in range(16000)]
                    for offset in range(0, len(samples), 160):
                        modem.feed_pcm(struct.pack(
                            "<160h", *samples[offset:offset + 160]
                        ))
                    self.assertGreater(modem.rx_signal_power_dbm0, -43.0)
                    self.assertFalse(
                        modem.connected,
                        f"off-plan {frequency}-Hz tone asserted carrier at {baud} baud",
                    )

    def test_1200_forward_and_75_backward_carry_independent_payloads(self):
        caller = V23Modem(True, forward_baud=1200)
        answerer = V23Modem(False, forward_baud=1200)
        forward = bytes(range(96))
        backward = b"return-channel"
        sent = False
        settle_frames = 0

        received_by_caller = bytearray()
        received_by_answerer = bytearray()
        for _ in range(180):  # 3.6 seconds: enough for both 8N1 directions
            if caller.connected and answerer.connected and not sent:
                if settle_frames >= 2:
                    caller.feed_bytes(forward)
                    answerer.feed_bytes(backward)
                    sent = True
                else:
                    settle_frames += 1
            caller_audio = caller.tx_samples(160)
            answerer_audio = answerer.tx_samples(160)
            received_by_caller.extend(caller.feed_pcm(struct.pack("<160h", *answerer_audio)))
            received_by_answerer.extend(answerer.feed_pcm(struct.pack("<160h", *caller_audio)))

        self.assertTrue(caller.connected)
        self.assertTrue(answerer.connected)
        self.assertEqual(bytes(received_by_answerer), forward)
        self.assertEqual(bytes(received_by_caller), backward)
        self.assertEqual((caller.tx_baud, caller.rx_baud), (1200, 75))
        self.assertEqual((answerer.tx_baud, answerer.rx_baud), (75, 1200))
        self.assertEqual(caller.bit_rate, answerer.bit_rate)
        self.assertNotIn(("CARRIER_DOWN", -1), answerer.events,
                         "forward FSK data must not look like loss of circuit 109")
        self.assertNotIn(("CARRIER_DOWN", -1), caller.events,
                         "backward FSK data must not look like loss of circuit 122")

    def test_carrier_loss_discards_partial_frame_and_reacquires_both_channels(self):
        """Neither FSK receive direction may join a pre-loss UART character to recovered data."""
        scenarios = (
            (False, "originate", 1200, (1300.0, 2100.0), 40),
            (True, "answer", 75, (390.0, 450.0), 640),
        )
        for receiver_calling, tx_role, baud, frequencies, partial_samples in scenarios:
            with self.subTest(rx_baud=75 if receiver_calling else 1200):
                receiver = V23Modem(receiver_calling, forward_baud=1200)
                interrupted = Bell103StreamTransmitter(
                    tx_role, baud=baud, frequencies=frequencies)
                # Establish a valid carrier first; the forward channel uses
                # its 300-ms call-setup interval, the backward one 40 ms.
                received = bytearray()
                for _ in range(32):
                    received.extend(receiver.feed_pcm(struct.pack(
                        "<160h", *interrupted.samples(160))))
                self.assertTrue(receiver.connected)

                interrupted.feed_bytes(b"X")
                pcm = interrupted.samples(partial_samples)
                for offset in range(0, len(pcm), 160):
                    received.extend(receiver.feed_pcm(struct.pack(
                        f"<{min(160, len(pcm) - offset)}h", *pcm[offset:offset + 160])))
                self.assertEqual(bytes(received), b"")  # first UART character is incomplete.

                # Drop carrier long enough to assert OFF and clear candidate
                # framing, then let a fresh independent transmitter reacquire.
                for _ in range(8):
                    received.extend(receiver.feed_pcm(bytes(320)))
                self.assertFalse(receiver.connected)
                self.assertIn(("CARRIER_DOWN", -1), receiver.events)
                self.assertEqual(bytes(received), b"")

                recovered = Bell103StreamTransmitter(
                    tx_role, baud=baud, frequencies=frequencies)
                for _ in range(16):
                    received.extend(receiver.feed_pcm(struct.pack(
                        "<160h", *recovered.samples(160))))
                self.assertTrue(receiver.connected)
                payload = b"clean-after-reacquire"
                recovered.feed_bytes(payload)
                for _ in range(320):  # 6.4 s also covers the 75-baud direction.
                    received.extend(receiver.feed_pcm(struct.pack(
                        "<160h", *recovered.samples(160))))
                    if payload in received:
                        break
                self.assertEqual(bytes(received), payload,
                                 f"rx_baud={receiver.rx_baud}, events={receiver.events}, "
                                 f"decoded={bytes(received)!r}")
                transitions = [name for name, _ in receiver.events]
                self.assertGreaterEqual(transitions.count("CARRIER_UP"), 2)

    def test_600_baud_mode_selection_transmits_at_600_on_forward_path(self):
        caller = V23Modem(True, forward_baud=600)
        answerer = V23Modem(False, forward_baud=600)
        payload = b"V23 mode 1"
        caller.feed_bytes(payload)
        received = bytearray()
        for _ in range(100):
            tx = caller.tx_samples(160)
            answerer.tx_samples(160)
            received.extend(answerer.feed_pcm(struct.pack("<160h", *tx)))
        self.assertEqual(bytes(received), payload)
        self.assertEqual(caller.tx_baud, 600)
        self.assertEqual(answerer.rx_baud, 600)

    def test_optional_backward_channel_can_be_disabled_at_either_endpoint(self):
        payload = b"forward stays available"
        backward = b"must be suppressed"
        for disabled_endpoint in ("caller", "answerer"):
            with self.subTest(disabled_endpoint=disabled_endpoint):
                caller = V23Modem(
                    True, forward_baud=1200,
                    backward_channel=disabled_endpoint != "caller",
                )
                answerer = V23Modem(
                    False, forward_baud=1200,
                    backward_channel=disabled_endpoint != "answerer",
                )
                caller.feed_bytes(payload)
                answerer.feed_bytes(backward)
                at_caller, at_answerer = bytearray(), bytearray()
                for _ in range(180):
                    caller_audio = caller.tx_samples(160)
                    answer_audio = answerer.tx_samples(160)
                    at_caller.extend(caller.feed_pcm(struct.pack("<160h", *answer_audio)))
                    at_answerer.extend(answerer.feed_pcm(struct.pack("<160h", *caller_audio)))
                self.assertTrue(caller.connected)
                self.assertTrue(answerer.connected)
                self.assertEqual(bytes(at_answerer), payload)
                self.assertEqual(bytes(at_caller), b"")
                self.assertEqual(caller.backward_channel_enabled,
                                 disabled_endpoint != "caller")
                self.assertEqual(answerer.backward_channel_enabled,
                                 disabled_endpoint != "answerer")

    def test_invalid_forward_rate_is_rejected(self):
        for rate in (75, 300, 1200, 2400):
            if rate == 1200:
                continue
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                V23Modem(True, forward_baud=rate)

    def test_software_audio_loop_reflects_data_through_pcmu_and_both_channels(self):
        payload = b"xy"
        for rate in (600, 1200):
            for calling_party in (True, False):
                with self.subTest(forward_baud=rate, calling_party=calling_party):
                    loop = V23SoftwareAudioLoop(
                        forward_baud=rate, calling_party=calling_party)
                    for _ in range(35):
                        loop.step()  # detector setup, no network or external line
                    self.assertTrue(loop.endpoint.connected)
                    self.assertTrue(loop.peer.connected)

                    loop.feed_bytes(payload)
                    returned = bytearray()
                    for _ in range(100):
                        returned.extend(loop.step())
                        if returned == payload:
                            break

                    self.assertEqual(bytes(returned), payload)
                    self.assertGreaterEqual(loop.reflected_bytes, len(payload))
                    self.assertEqual(loop.status()["kind"], "software-audio-loop")
                    self.assertFalse(loop.status()["physical_v54_loop"])

    def test_software_audio_loop_roundtrips_under_bounded_sample_impairment(self):
        payload = bytes(range(16))
        for rate in (600, 1200):
            for calling_party in (True, False):
                with self.subTest(rate=rate, calling_party=calling_party):
                    impairment = AudioPathImpairment(
                        gain_db=3, dc_offset=256, noise_rms=64,
                        clip_level=10000, seed=7)
                    loop = V23SoftwareAudioLoop(
                        forward_baud=rate, calling_party=calling_party,
                        local_to_peer_impairment=impairment)
                    for _ in range(35):
                        loop.step()
                    loop.feed_bytes(payload)
                    received = bytearray()
                    for _ in range(600):
                        received.extend(loop.step())
                        if payload in received:
                            break

                    self.assertIn(payload, received)
                    decoded = received[received.index(payload):][:len(payload)]
                    bit_errors = sum((sent ^ got).bit_count()
                                     for sent, got in zip(payload, decoded))
                    self.assertEqual(bit_errors, 0)
                    self.assertGreater(impairment.report()["clipped_samples"], 0)


@unittest.skipUnless(shutil.which("minimodem"), "minimodem independent FSK reference not installed")
class V23IndependentReferenceTests(unittest.TestCase):
    def decode_minimodem_wave_with_noise(self, wav_path, *, calling, baud,
                                         noise_dbm0, seed):
        """Decode an independent waveform at a fixed signal/noise level.

        The waveform is first converted to 8-kHz PCM, then globally normalized
        to -13 dBm0. AWGN is added at a fixed digital dBm0 level after
        resampling, avoiding accidental noise-bandwidth changes from the WAV
        decimation.
        """
        with wave.open(str(wav_path), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(),
                              wav.getframerate()), (1, 2, 48000))
            high_rate = [value for (value,) in struct.iter_unpack(
                "<h", wav.readframes(wav.getnframes()))]
        samples = high_rate[::6]
        self.assertTrue(samples)
        measured_rms = math.sqrt(sum(value * value for value in samples) / len(samples))
        target_rms = math.sqrt(DBM0_REFERENCE_POWER * 10 ** (-13.0 / 10.0))
        scale = target_rms / measured_rms
        noise_rms = math.sqrt(DBM0_REFERENCE_POWER * 10 ** (noise_dbm0 / 10.0))
        rng = random.Random(seed)
        noisy = [max(-32768, min(32767, round(value * scale + rng.gauss(0, noise_rms))))
                 for value in samples]
        rx = V23Modem(calling, forward_baud=baud if baud in (600, 1200) else 1200)
        decoded = bytearray()
        for offset in range(0, len(noisy), 160):
            block = noisy[offset:offset + 160]
            decoded.extend(rx.feed_pcm(struct.pack(f"<{len(block)}h", *block)))
        return bytes(decoded), rx

    def test_independent_waveforms_decode_with_controlled_awgn_both_channels(self):
        payload = b"V23 independent noise envelope 0123456789\r\n" * 2
        channels = (
            (1200, FORWARD_MARK_HZ, FORWARD_SPACE_HZ[1200], False),
            (75, BACKWARD_MARK_HZ, BACKWARD_SPACE_HZ, True),
        )
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-awgn-") as tmp:
            for baud, mark, space, calling in channels:
                for noise_dbm0 in (-36.0, -30.0):
                    with self.subTest(baud=baud, noise_dbm0=noise_dbm0):
                        wav_path = Path(tmp) / f"v23-{baud}-{noise_dbm0}.wav"
                        generated = subprocess.run(
                            ["minimodem", "--tx", "--quiet", "-M", str(mark),
                             "-S", str(space), "-f", str(wav_path), str(baud)],
                            input=payload, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False,
                        )
                        self.assertEqual(generated.returncode, 0,
                                         generated.stderr.decode(errors="replace"))
                        decoded, rx = self.decode_minimodem_wave_with_noise(
                            wav_path, calling=calling, baud=baud,
                            noise_dbm0=noise_dbm0, seed=2300 + baud + int(-noise_dbm0),
                        )
                        self.assertEqual(decoded, payload)
                        self.assertTrue(rx.connected)

    def test_forward_clock_tracking_survives_combined_slow_clock_and_awgn(self):
        # The Minitel WAV motivated testing a slow forward channel. Minimodem
        # is configured for 1161 baud here, but its generated file is not a
        # calibrated 1161-baud clock vector (the production estimator sees
        # about 1170 baud). This remains an independent-waveform known-byte
        # regression; the exact-clock diagnostic is in the sweep tool.
        payload = b"".join(random.Random(7400 + i).randbytes(128) for i in range(10))
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-slow-noise-") as tmp:
            wav_path = Path(tmp) / "slow-clock.wav"
            generated = subprocess.run(
                ["minimodem", "--tx", "--quiet", "-M", str(FORWARD_MARK_HZ),
                 "-S", str(FORWARD_SPACE_HZ[1200]), "-f", str(wav_path), "1161"],
                input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(generated.returncode, 0,
                             generated.stderr.decode(errors="replace"))
            for seed in range(74000, 74010):
                with self.subTest(noise_seed=seed):
                    decoded, rx = self.decode_minimodem_wave_with_noise(
                        wav_path, calling=False, baud=1200,
                        noise_dbm0=-30.0, seed=seed,
                    )
                    self.assertEqual(decoded, payload)
                    self.assertTrue(rx.connected)

    def test_forward_clock_tracker_acquires_before_first_slow_clock_error(self):
        # Exact 1161-baud control at -30 dBm0 AWGN. This fixed noise
        # realization lost a character in the baseline and now decodes exactly.
        # The broader ten-realization result remains in the sweep tool/note.
        payload = b"".join(
            random.Random(7400 + i).randbytes(128) for i in range(10)
        )
        tx = Bell103StreamTransmitter(
            "originate", baud=1161,
            frequencies=(FORWARD_MARK_HZ, FORWARD_SPACE_HZ[1200]),
        )
        samples = []
        for _ in range(20):  # 400 ms mark lead for carrier acquisition
            samples.extend(tx.samples(160))
        tx.feed_bytes(payload)
        remaining = math.ceil((len(payload) * 10 + 30) * 8000 / 1161) + 1600
        while remaining:
            count = min(160, remaining)
            samples.extend(tx.samples(count))
            remaining -= count

        measured_rms = math.sqrt(sum(value * value for value in samples) / len(samples))
        target_rms = math.sqrt(DBM0_REFERENCE_POWER * 10 ** (-13.0 / 10.0))
        noise_rms = math.sqrt(DBM0_REFERENCE_POWER * 10 ** (-30.0 / 10.0))
        rng = random.Random(74004)
        noisy = [
            max(-32768, min(32767, round(value * target_rms / measured_rms
                                          + rng.gauss(0, noise_rms))))
            for value in samples
        ]
        rx = V23Modem(False, forward_baud=1200)
        decoded = bytearray()
        for offset in range(0, len(noisy), 160):
            block = noisy[offset:offset + 160]
            decoded.extend(rx.feed_pcm(struct.pack(f"<{len(block)}h", *block)))
        self.assertEqual(bytes(decoded), payload)
        self.assertTrue(rx.connected)

    def test_backward_receiver_accepts_r35_frequency_limits_from_minimodem(self):
        payload = b"Independent V23 backward frequency envelope\r\n"
        # V.23 §5 refers backward-channel frequency tolerances to R.35.
        # R.35 §§3, 5-6 give mean-frequency error <=2 Hz and shift error
        # <=3 Hz; include the V.23 §5 assumed connection drift of +/-6 Hz.
        # Thus the received mean may be +/-8 Hz from nominal while the
        # mark/space difference remains within +/-3 Hz of 60 Hz.
        offsets = ((-8, -8), (8, 8), (-9.5, -6.5), (-6.5, -9.5),
                   (6.5, 9.5), (9.5, 6.5))
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-backward-drift-") as tmp:
            tmp_path = Path(tmp)
            for mark_offset, space_offset in offsets:
                with self.subTest(mark=mark_offset, space=space_offset):
                    wav_path = tmp_path / f"backward-{mark_offset}-{space_offset}.wav"
                    generated = subprocess.run(
                        ["minimodem", "--tx", "--quiet",
                         "-M", str(BACKWARD_MARK_HZ + mark_offset),
                         "-S", str(BACKWARD_SPACE_HZ + space_offset),
                         "-f", str(wav_path), "75"],
                        input=payload, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, check=False,
                    )
                    self.assertEqual(generated.returncode, 0,
                                     generated.stderr.decode(errors="replace"))
                    rx = V23Modem(True, forward_baud=1200)
                    decoded = bytearray()
                    with wave.open(str(wav_path), "rb") as wav:
                        self.assertEqual((wav.getnchannels(), wav.getsampwidth(),
                                          wav.getframerate()), (1, 2, 48000))
                        while audio := wav.readframes(960):
                            high_rate = struct.unpack(f"<{len(audio) // 2}h", audio)
                            samples = high_rate[::6]
                            decoded.extend(rx.feed_pcm(
                                struct.pack(f"<{len(samples)}h", *samples)))
                    self.assertEqual(bytes(decoded), payload)

    def test_forward_receiver_accepts_frequency_drift_envelope_from_minimodem(self):
        payload = b"Independent V23 frequency envelope 0123456789\r\n" * 2
        # V.23 §3 permits ±10 Hz transmitter error per tone and assumes up
        # to ±6 Hz line drift. Its separate frequency-difference tolerance
        # is ±20 Hz. Exercise common ±16-Hz offsets and the differential
        # boundary while preserving those limits.
        offsets = ((-16, -16), (16, 16), (-10, 10), (10, -10),
                   (16, -4), (-16, 4))
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-drift-") as tmp:
            tmp_path = Path(tmp)
            for rate in (600, 1200):
                for mark_offset, space_offset in offsets:
                    with self.subTest(rate=rate, mark=mark_offset, space=space_offset):
                        wav_path = tmp_path / f"v23-{rate}-{mark_offset}-{space_offset}.wav"
                        command = [
                            "minimodem", "--tx", "--quiet",
                            "-M", str(FORWARD_MARK_HZ + mark_offset),
                            "-S", str(FORWARD_SPACE_HZ[rate] + space_offset),
                            "-f", str(wav_path), str(rate),
                        ]
                        generated = subprocess.run(
                            command, input=payload, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False,
                        )
                        self.assertEqual(generated.returncode, 0,
                                         generated.stderr.decode(errors="replace"))
                        rx = V23Modem(False, forward_baud=rate)
                        decoded = bytearray()
                        with wave.open(str(wav_path), "rb") as wav:
                            self.assertEqual((wav.getnchannels(), wav.getsampwidth(),
                                              wav.getframerate()), (1, 2, 48000))
                            while audio := wav.readframes(960):
                                high_rate = struct.unpack(f"<{len(audio) // 2}h", audio)
                                samples = high_rate[::6]
                                decoded.extend(rx.feed_pcm(
                                    struct.pack(f"<{len(samples)}h", *samples)))
                        self.assertEqual(bytes(decoded), payload)

    def roundtrip_reference(self, *, baud, mark, space, calling, payload):
        with tempfile.TemporaryDirectory(prefix="elmodem-v23-") as tmp:
            reference_wav = Path(tmp) / "reference.wav"
            result = subprocess.run(
                ["minimodem", "--tx", "--quiet", "-M", str(mark), "-S", str(space),
                 "-f", str(reference_wav), str(baud)],
                input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            with wave.open(str(reference_wav), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (1, 2, 48000))
                rx = V23Modem(calling, forward_baud=baud if baud in (600, 1200) else 1200)
                received = bytearray()
                while audio := wav.readframes(960):
                    high_rate = struct.unpack(f"<{len(audio) // 2}h", audio)
                    decimated = high_rate[::6]
                    received.extend(rx.feed_pcm(struct.pack(f"<{len(decimated)}h", *decimated)))
            self.assertEqual(bytes(received), payload)

            ours = V23Modem(not calling, forward_baud=baud if baud in (600, 1200) else 1200)
            ours.feed_bytes(payload)
            audio = []
            # Include idle mark after payload so the independent receiver can
            # validate the final stop bit instead of ending at the file edge.
            samples = max(2 * 8000, int((len(payload) * 10 + 30) * 8000 / baud))
            for offset in range(0, samples, 160):
                audio.extend(ours.tx_samples(min(160, samples - offset)))
            ours_wav = Path(tmp) / "ours.wav"
            with wave.open(str(ours_wav), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(struct.pack(f"<{len(audio)}h", *audio))
            decoded = subprocess.run(
                ["minimodem", "--rx", "--quiet", "-M", str(mark), "-S", str(space),
                 "-R", "8000", "-f", str(ours_wav), str(baud)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
            self.assertEqual(decoded.stdout, payload)

    def test_forward_1200_channel_against_minimodem_both_directions(self):
        self.roundtrip_reference(
            baud=1200, mark=1300, space=2100, calling=False,
            payload=b"V23 forward 1200\r\n",
        )

    def test_forward_600_channel_against_minimodem_both_directions(self):
        self.roundtrip_reference(
            baud=600, mark=1300, space=1700, calling=False,
            payload=b"V23 mode 1 600\r\n",
        )

    def test_backward_75_channel_against_minimodem_both_directions(self):
        self.roundtrip_reference(
            baud=75, mark=390, space=450, calling=True,
            payload=b"V23 return 75\r\n",
        )


if __name__ == "__main__":
    unittest.main()
