import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import v34_g711
import v92_ja
import v90_digital_tx
import v92_analogue_session as asn
import v92_phase3_exchange as ex
import v92_phase4_controller as p4c

_TESTDATA = Path(__file__).with_name("testdata")


def _needs_capture(*names):
    """Skip a test whose recorded capture is private and absent from this checkout."""
    missing = [n for n in names if not (_TESTDATA / n).exists()]
    return unittest.skipIf(missing, "private capture fixture not present: " + ", ".join(missing))


def descriptor(rates=(28000.0, 32000.0, 36000.0), *, dil_ucodes=()):
    return v92_ja.build_ja_descriptor(N=len(dil_ucodes), Ucodes=list(dil_ucodes), upstream_rates=rates)


def loop(law="pcmu", *, delay=240, silence=560, blocks=900, rate_mask=0x3FFFFF, lu=1000.0,
         ja_descriptor=None, fractional_tx_timing=False, channel_fractional_delay=0.0):
    analogue = ex.AnalogueSide(ja_descriptor or descriptor(), lu=lu, law=law, silence_samples=silence,
                               fractional_tx_timing=fractional_tx_timing)
    digital = ex.DigitalSide(law, rate_mask=rate_mask)
    timing = (asn.V92AnalogueSession(law=law, fractional_tx_timing=True)
              if fractional_tx_timing else None)
    channel_timing = (asn.V92AnalogueSession(law=law, fractional_tx_timing=True)
                      if channel_fractional_delay else None)
    # a simple sample delay line per direction
    up_line = list(np.zeros(delay))
    down_line = list(np.zeros(delay))
    for _ in range(blocks):
        a = analogue.tx(160)
        if timing is not None:
            a = timing._apply_tx_fractional_timing(a, analogue._last_tx_delay_trace)
        if channel_timing is not None:
            a = channel_timing._apply_tx_fractional_timing(
                a, np.full(len(a), float(channel_fractional_delay), dtype=float))
        a = v34_g711.roundtrip(np.rint(a), law).astype(float)
        up_line.extend(a)
        digital.feed(np.array(up_line[:160])); del up_line[:160]
        d = digital.tx(160)
        d = v34_g711.decode(np.frombuffer(d, dtype=np.uint8), law).astype(float)
        down_line.extend(d)
        analogue.feed(np.array(down_line[:160])); del down_line[:160]
    return analogue, digital


def measure_trn1u_residual_phase(digital):
    """Fit the received TRN1u#2 PCM to fractional sinc templates near 0 mod 1."""
    start = digital.trn1u2_start - digital.eq_delay
    reference = np.where(np.asarray(ex._trn1u_bits(900), dtype=bool), -1.0, 1.0)
    observed = digital._raw
    best_score, best_phase, best_integer_offset = -1.0, None, None
    # Search both sides of the modulo-one boundary. Integer timing is allowed
    # to move independently so a 1T template and a 0T template are equivalent.
    phases = sorted(set(range(0, 65)) | set(range(ex.TX_DELAY_PHASES - 64,
                                                   ex.TX_DELAY_PHASES + 1)))
    for phase in phases:
        fractional_delay = phase / ex.TX_DELAY_PHASES
        taps = ex.TX_DELAY_BANK[phase]
        delayed = np.convolve(reference, taps, mode="full")[:len(reference)]
        expected = delayed[ex.TX_DELAY_BULK + 60:ex.TX_DELAY_BULK + 560]
        expected -= float(expected.mean())
        expected_norm = float(np.linalg.norm(expected))
        for integer_offset in (-1, 0, 1):
            at = start + integer_offset + 60
            actual = observed[at:at + len(expected)]
            if len(actual) != len(expected):
                continue
            actual = actual - float(actual.mean())
            score = float(actual @ expected) / (float(np.linalg.norm(actual)) * expected_norm + 1e-12)
            if score > best_score:
                best_score, best_phase, best_integer_offset = score, fractional_delay, integer_offset
    if best_phase is None:
        raise AssertionError("TRN1u#2 receive window is incomplete")
    return min(best_phase, 1.0 - best_phase), best_score, best_integer_offset


class PhaseThreeExchangeTests(unittest.TestCase):
    def test_first_su_bar_requires_expected_phase3_position(self):
        digital = ex.DigitalSide("pcmu")
        digital._begin_downstream()
        self.assertTrue(digital.su_tracker.fractional_timing)
        self.assertEqual(digital.su_tracker.min_correlation, 0.76)

        su = np.asarray(ex.PATTERNS["su"], dtype=float)
        su_bar = np.asarray(ex.PATTERNS["su-bar"], dtype=float)
        # A short early lookalike must not trigger Jp. The real first bar is
        # at 144 PCM intervals after Su, as specified for the 144T Su segment.
        received = np.concatenate((
            np.tile(su, 10), np.tile(su_bar, 3), np.tile(su, 11),
            np.tile(su_bar, 4), np.tile(su, 30),
        )) * 300.0
        for start in range(0, len(received), 160):
            digital._su_events(received[start:start + 160])

        outside = [value for name, value in digital.events
                   if name == "su-bar-outside-first-window"]
        accepted = [value for name, value in digital.events
                    if name == "su-flip-1"]
        self.assertEqual(len(outside), 1, digital.events)
        self.assertEqual(outside[0], {"sample": 60, "expected": 144})
        self.assertEqual(accepted, [144], digital.events)

    def test_fractional_rx_phase_estimator_uses_known_trn1u_only(self):
        training = np.asarray(ex._trn1u_bits(1800), dtype=float)
        ideal = np.where(training == 1, -1200.0, 1200.0)
        ideal = np.concatenate((np.zeros(16), ideal))
        shifted = ex._windowed_sinc_sample(
            ideal, np.arange(len(ideal), dtype=float) - 0.237)
        phase, fit = ex._estimate_trn1u_fractional_phase(
            shifted, 16, training_symbols=1000)
        self.assertAlmostEqual(phase, 0.237, delta=0.02)
        self.assertGreater(fit, 0.99)

    def test_second_trn1u_reacquires_fractional_rx_phase(self):
        digital = ex.DigitalSide("pcmu", fractional_rx_timing=True)
        start = 300
        bits = np.asarray(ex._trn1u_bits(700), dtype=bool)
        source = np.zeros(1200, dtype=float)
        source[start:start + len(bits)] = np.where(bits, -1200.0, 1200.0)
        # Simulate the new TRN burst arriving at a different fractional phase
        # from the phase fitted on TRN1u #1.
        digital._rx = ex._windowed_sinc_sample(
            source, np.arange(len(source), dtype=float) - 0.45)
        digital._raw = digital._rx.copy()
        digital.trn1u_start = 20
        digital._flip3 = True
        digital._flip3_at = start - 24
        digital.state = "sending"
        digital.rx_fractional_phase = 0.15
        digital._fractional_rx = ex._windowed_sinc_sample(
            digital._rx, np.arange(len(digital._rx), dtype=float) + 0.15)

        for _ in range(len(np.arange(0.0, 1.0, 0.05))):
            digital._find_trn1u2()
            if digital.trn1u2_start is not None:
                break

        self.assertEqual(digital.trn1u2_start, start, digital.events)
        # A hard-sign TRN sequence has a broad fractional-phase eye.  The
        # acquisition objective is the sampling eye with a strong training
        # match, not recovery of the synthetic channel's exact delay.
        self.assertGreater(digital.rx_fractional_phase, 0.20)
        self.assertLess(digital.rx_fractional_phase, 0.66)
        self.assertGreater(abs(digital.rx_fractional_phase - 0.15), 0.05)
        phase_event = next(value for name, value in digital.events
                           if name == "trn1u2-fractional-phase")
        self.assertGreaterEqual(phase_event["match"], 0.95)
        recovered = digital._fractional_rx[start:start + 600] < 0.0
        self.assertGreaterEqual(float(np.mean(recovered == bits[:600])), 0.99)

    @_needs_capture('v92_smartlink_ja_capture.pcm.gz')
    def test_opt_in_fractional_rx_timing_recovers_ja_after_training_fit(self):
        fixture = Path(__file__).with_name("testdata") / "v92_smartlink_ja_capture.pcm.gz"
        captured = np.frombuffer(gzip.decompress(fixture.read_bytes()), dtype="<i2").astype(float)
        prefix_bits = np.asarray(ex._trn1u_bits(16))
        prefix = np.where(prefix_bits == 1, -1500.0, 1500.0)
        # Preserve the PRBS phase at the splice, add a guard before the first
        # known symbol, then create a sub-sample channel delay without using Ja.
        source = np.concatenate((np.zeros(16), prefix, captured[16:]))
        received = ex._windowed_sinc_sample(
            source, np.arange(len(source), dtype=float) - 0.16)
        digital = ex.DigitalSide("pcmu", uinfo=78, rate_mask=0x3FFFFF,
                                 fractional_rx_timing=True)
        digital.state = "wait-ja"
        digital.trn1u_start = 16
        for start in range(0, len(received), 160):
            digital.feed(received[start:start + 160])
            if digital.ja is not None or digital.state == "failed":
                break
        self.assertEqual(digital.state, "sending", digital.events)
        self.assertIsNotNone(digital.ja)
        self.assertAlmostEqual(digital.rx_fractional_phase, 0.314, delta=0.01)
        self.assertTrue(any(name == "trn1u-fractional-phase" for name, _ in digital.events))
        timing = digital.fractional_rx_timing_diagnostics()
        self.assertGreater(timing["count"], 1)
        self.assertGreaterEqual(timing["p99_ms"], 0.0)
        self.assertGreater(timing["max_ms"], 0.0)
        self.assertGreater(timing["max_location"]["feed_block"], 0)
        self.assertGreater(timing["max_location"]["sample_position"], 0)
        self.assertIn(timing["max_location"]["stage"],
                      ("fit-and-backfill", "incremental-resample"))

    @_needs_capture('v92_smartlink_ja_capture.pcm.gz')
    def test_smartlink_capture_finds_late_overlapped_ja_with_valid_majority_crc(self):
        fixture = Path(__file__).with_name("testdata") / "v92_smartlink_ja_capture.pcm.gz"
        samples = np.frombuffer(gzip.decompress(fixture.read_bytes()), dtype="<i2").astype(float)
        digital = ex.DigitalSide("pcmu", uinfo=78, rate_mask=0x3FFFFF, equalise=False)
        # The fixture starts exactly at the captured TRN1u anchor. Bypass Ru
        # only so this regression isolates the Ja acquisition/parser behavior.
        digital.state = "wait-ja"
        digital.trn1u_start = 0
        for start in range(0, len(samples), 160):
            digital.feed(samples[start:start + 160])
            if digital.ja is not None or digital.state == "failed":
                break
        self.assertEqual(digital.state, "sending", digital.events)
        self.assertEqual(digital.ja_start, 20021, digital.events)
        self.assertEqual(digital.ja["N"], 144)
        self.assertEqual(digital.ja["LSP"], 120)
        self.assertEqual(digital.ja["LTP"], 120)
        self.assertEqual(digital.ja["upstream_rate_mask"], 0xFFFF)
        self.assertEqual(digital.ja["repeat_bits"], 1736)
        self.assertEqual(digital.ja["repetitions"], 5)
        self.assertIn(("ja", 20021), digital.events)

    @_needs_capture('v92_smartlink_cpt_symbols.json', 'v92_smartlink_cpt_symbols.bin')
    def test_smartlink_captured_cpt_sync_and_crc_acquire_without_exact_preamble(self):
        testdata = Path(__file__).with_name("testdata")
        meta = json.loads((testdata / "v92_smartlink_cpt_symbols.json").read_text())
        packed = np.fromfile(testdata / "v92_smartlink_cpt_symbols.bin", dtype=np.uint8)
        symbols = np.unpackbits(packed, bitorder=meta["bitorder"])[:meta["symbol_count"]]
        received = np.where(symbols, -1200.0, 1200.0)
        digital = ex.DigitalSide("pcmu", fractional_rx_timing=True)
        digital.state = "trn1u-2"
        digital.trn1u2_start = meta["trn1u2_start"]
        digital.rx_fractional_phase = 0.0
        digital._rx = received.copy()
        digital._fractional_rx = received.copy()

        digital._find_cpt()

        self.assertIsNotNone(digital.cpt, digital.events)
        self.assertEqual(digital.cpt_at, meta["expected_cpt_preamble_offset"])
        self.assertEqual(digital.cpt["kind"], meta["expected_kind"])
        self.assertEqual(digital.cpt["drn"], meta["expected_drn"])
        self.assertEqual(digital.cpt["frame_bits"], meta["expected_frame_bits"])
        preamble = next(value for name, value in digital.events if name == "cpt-preamble")
        self.assertLess(preamble["ones"], preamble["expected"])

    @_needs_capture('v92_smartlink_e1u_boundary.json', 'v92_smartlink_e1u_boundary.npz')
    def test_smartlink_captured_noisy_e1u_window_enters_phase4(self):
        testdata = Path(__file__).with_name("testdata")
        meta = json.loads((testdata / "v92_smartlink_e1u_boundary.json").read_text())
        with np.load(testdata / "v92_smartlink_e1u_boundary.npz") as capture:
            samples = capture["decision_samples"].astype(float)
            reg = capture["reg"].astype(int).tolist()
            prev = int(capture["prev"][0])
        digital = ex.DigitalSide("pcmu", fractional_rx_timing=True)
        digital.state = "cpt-received"
        digital.cpt = {"kind": "cpt"}
        digital.cpt_at = 0
        digital._cpt_len = meta["cpt_frame_symbols"]
        digital._cpt_dec = (0, prev, reg)
        digital.ri_units = 24
        digital.rx_fractional_phase = 0.4
        digital._rx = samples.copy()
        digital._fractional_rx = samples.copy()

        digital._find_e1u()

        self.assertEqual(digital.trn2u_start, 12)
        self.assertEqual(digital.state, "cpt-received")
        self.assertIn(("e1u", 0), digital.events)
        qualified = next(value for name, value in digital.events
                         if name == "e1u-qualified")
        self.assertEqual(qualified["zero_bits"], 10)
        self.assertEqual(qualified["threshold"], ex.E1U_MIN_ZERO_BITS)
        window = next(value for name, value in digital.events
                      if name == "e1u-window")
        self.assertEqual(window["classification"], "e1u")
        self.assertEqual(window["bits"][:36], meta["decoded_first_36_bits"])

    @_needs_capture('v92_smartlink_e1u_boundary.npz')
    def test_ambiguous_cpt_followup_keeps_ri_until_e1u_timeout(self):
        testdata = Path(__file__).with_name("testdata")
        with np.load(testdata / "v92_smartlink_e1u_boundary.npz") as capture:
            prev = int(capture["prev"][0])
            reg = capture["reg"].astype(int).tolist()

        # Encode an alternating decoded stream through the stateful
        # differential/scrambler mapper. No 12-symbol window is E1u-like, so
        # the receiver must keep scanning and time out instead of going silent
        # on the first ambiguous frame.
        wire = []
        previous = prev
        state = list(reg)
        for decoded in ([1, 0] * 12) * (24_000 // 24):
            scrambled = decoded ^ state[4] ^ state[22]
            transmitted = scrambled ^ previous
            wire.append(transmitted)
            previous = transmitted
            state = [scrambled] + state[:-1]
        samples = np.where(np.asarray(wire, dtype=bool), -1000.0, 1000.0)

        digital = ex.DigitalSide("pcmu")
        digital.state = "cpt-received"
        digital.cpt = {"kind": "cpt"}
        digital._cpt_len = 432
        digital._cpt_dec = (0, prev, reg)
        digital.ri_units = 24
        digital._rx = samples[:432].copy()
        digital._fractional_rx = digital._rx.copy()
        digital._dil = SimpleNamespace(stopped=True)
        digital._jprime_sent = True

        digital._find_e1u()
        self.assertEqual(digital.state, "cpt-received")
        self.assertTrue(any(byte != digital._silence for byte in digital.tx(160)))

        digital._rx = samples
        digital._fractional_rx = samples.copy()
        digital._find_e1u()
        self.assertEqual(digital.state, "failed")
        timeout = next(value for name, value in digital.events
                       if name == "e1u-timeout")
        self.assertGreaterEqual(timeout["waited_symbols"],
                                ex.E1U_WAIT_TIMEOUT_SYMBOLS)

    def test_e1u_classifier_waits_for_a_complete_ri_before_scanning(self):
        digital = ex.DigitalSide("pcmu")
        digital.state = "cpt-received"
        digital.cpt = {"kind": "cpt"}
        digital._cpt_len = 432
        digital._cpt_dec = (0, 0, [0] * 23)
        digital._rx = np.full(432, -1000.0)

        digital._find_e1u()

        self.assertEqual(digital.state, "cpt-received")
        self.assertEqual(digital._e1u_windows_seen, 0)
        self.assertIsNone(digital._e1u_wait_start)

    def test_first_phase_second_trn_arms_cpt_receive_without_reacquisition(self):
        digital = ex.DigitalSide(
            "pcmu", fractional_rx_timing=True,
            fractional_rx_reacquisition=False)
        digital._flip3_at = 1000
        digital._rx = np.zeros(2200)
        digital.rx_fractional_phase = 0.16
        digital._best_near_alignment = lambda base, radius=1: (0.91, base - 6)

        digital._find_trn1u2()

        self.assertEqual(digital.trn1u2_start, 1018)
        self.assertEqual(digital.state, "trn1u-2")
        event = next(value for name, value in digital.events
                     if name == "trn1u-2-first-phase")
        self.assertEqual(event["match"], 0.91)

    def test_cpt_lookahead_above_jd_advertisement_is_rejected_with_event(self):
        digital = ex.DigitalSide("pcmu")

        digital._accept_cpt(0, 100, {"kind": "cpt", "lookahead": 2}, [])

        self.assertEqual(digital.state, "failed")
        self.assertIsNone(digital.cpt)
        self.assertIn(("unsupported-cpt-lookahead", {
            "requested": 2,
            "advertised": ex.v92_phase3.MAX_ADVERTISED_LOOKAHEAD,
        }), digital.events)

    def test_missing_ja_search_remains_bounded_at_v92_wait_limit(self):
        digital = ex.DigitalSide("pcmu")
        digital.state = "wait-ja"
        digital.trn1u_start = 0
        digital.feed(np.zeros(ex.JA_SEARCH_TIMEOUT_SYMBOLS + 160))
        self.assertEqual(digital.state, "failed")
        no_ja = [value for name, value in digital.events if name == "no-ja"]
        self.assertEqual(len(no_ja), 1)
        self.assertGreaterEqual(no_ja[0], ex.JA_SEARCH_TIMEOUT_SYMBOLS)

    def test_su_phase_estimator_recovers_fractional_delay_and_jp_complement(self):
        pattern = ex.PATTERNS["su"]
        reference, start = 5, 17
        n = np.arange(start, start + ex.SU_PHASE_WINDOW) - reference
        for delay in (0.0, 0.25, 0.5, 0.75):
            with self.subTest(delay=delay):
                phase = round(delay * ex.TX_DELAY_PHASES)
                taps = ex.TX_DELAY_BANK[phase]
                sample_phases = n % 6
                source_indices = ((sample_phases[:, None] - ex.TX_DELAY_BULK
                                   - np.arange(ex.TX_DELAY_TAPS)[None, :]) % 6)
                samples = np.sum(pattern[source_indices] * taps[None, :], axis=1)
                phase, fit = ex._estimate_su_fractional_phase(
                    samples, sample_start=start, su_reference_start=reference)
                jp = int(round(((1.0 - phase) % 1.0) * 65536.0)) & 0xFFFF
                self.assertGreater(fit, 0.30)
                self.assertAlmostEqual(phase, delay, delta=0.001)
                self.assertLess(min((jp - round((1.0 - delay) % 1.0 * 65536)) & 0xFFFF,
                                    (round((1.0 - delay) % 1.0 * 65536) - jp) & 0xFFFF), 8)

    def test_fractional_phase_jp_aligns_second_trn_with_and_without_channel_delay(self):
        for channel_delay, expected_phase, expected_jp in (
                (0.0, 0.5, 0x8000), (0.25, 0.75, 0x4000)):
            with self.subTest(channel_delay=channel_delay):
                analogue, digital = loop("pcmu", blocks=500, fractional_tx_timing=True,
                                         channel_fractional_delay=channel_delay)
                phase_event = next(value for name, value in digital.events if name == "su-phase")
                self.assertEqual(phase_event["measurement_point"], "post-su-bar")
                self.assertGreater(phase_event["fit"], 0.70)
                if expected_phase is not None:
                    self.assertAlmostEqual(phase_event["delay_samples"], expected_phase, delta=0.01)
                self.assertLess(abs(phase_event["jp_fraction"] - expected_jp), 512)
                self.assertEqual(analogue.jp["phase_fraction"], phase_event["jp_fraction"])
                self.assertIsNotNone(digital.trn1u2_start, digital.events)
                residual, residual_fit, _ = measure_trn1u_residual_phase(digital)
                self.assertLess(residual, 0.03, (residual, residual_fit, digital.events))
                self.assertGreater(residual_fit, 0.95, digital.events)
                # The pattern detector's flip event is quantized to a six-sample
                # window; compare the recovered start to the paired sender's exact
                # transition sample and the channel's integer delay instead.
                trn_offset = digital.trn1u2_start - (analogue.trn1u2_start + 240)
                self.assertEqual(trn_offset, 19 + (18 if channel_delay else 0), digital.events)
                self.assertAlmostEqual(analogue._tx_fractional_delay + channel_delay,
                                       1.0, delta=0.01)
                self.assertEqual(digital.state, "phase4-ready", digital.events)
                self.assertTrue(analogue.e1u_sent, analogue.events)

    def test_fractional_tx_timing_applies_245t_and_the_jp_fraction(self):
        analogue = ex.AnalogueSide(descriptor(), fractional_tx_timing=True)
        analogue._su_start = 0
        analogue._su_state, analogue._su_n = "bar1", 23
        analogue.tx(1)
        np.testing.assert_array_equal(analogue._last_tx_delay_trace, [0.0])
        self.assertEqual(analogue._tx_fractional_delay, 0.5)
        analogue.tx(1)
        np.testing.assert_array_equal(analogue._last_tx_delay_trace, [0.5])
        analogue.jp = {"phase_fraction": 0x4000}
        analogue._su_state, analogue._su_n = "bar2", 23
        analogue.tx(1)
        np.testing.assert_array_equal(analogue._last_tx_delay_trace, [0.5])
        self.assertEqual(analogue._tx_fractional_delay, 0.5 + 0x4000 / 65536)
        analogue.tx(1)
        np.testing.assert_allclose(analogue._last_tx_delay_trace, [0.5 + 0x4000 / 65536])
        self.assertEqual([name for name, _ in analogue.events],
                         ["su-bar1-fractional-end", "su-bar2-fractional-end"])

    def test_fractional_tx_sinc_keeps_cadence_and_fixed_bulk_delay(self):
        session = asn.V92AnalogueSession(law="pcmu", fractional_tx_timing=True)
        samples = np.zeros(80)
        samples[10] = 1.0
        delayed = session._apply_tx_fractional_timing(samples, np.zeros(len(samples)))
        self.assertEqual(len(delayed), len(samples))
        self.assertEqual(int(np.argmax(delayed)), 10 + asn.TX_DELAY_BULK)
        self.assertAlmostEqual(float(np.max(delayed)), 1.0, places=12)
        self.assertEqual(session._tx_delay_samples_seen, len(samples))

    def test_fractional_tx_sinc_frequency_response_is_flat_and_linear_phase(self):
        n = np.arange(asn.TX_DELAY_TAPS, dtype=float)
        omega = 2.0 * np.pi * np.linspace(300.0, 3400.0, 2001) / 8000.0
        d_values = (0.1, 0.25, 0.5, 0.75, 0.9)
        for delay in d_values:
            with self.subTest(delay=delay):
                scaled = delay * asn.TX_DELAY_PHASES
                lo = int(np.floor(scaled))
                alpha = scaled - lo
                taps = ((1.0 - alpha) * asn.TX_DELAY_BANK[lo]
                        + alpha * asn.TX_DELAY_BANK[lo + 1])
                response = np.exp(-1j * omega[:, None] * n) @ taps
                magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-15))
                group_delay = -np.gradient(np.unwrap(np.angle(response)), omega)
                expected = asn.TX_DELAY_BULK + delay
                self.assertLess(float(np.max(np.abs(group_delay - expected))), 0.02)
                self.assertLess(float(np.ptp(magnitude_db)), 0.2)

    def test_noninteger_jp_keeps_analog_trn2u_level(self):
        reference = asn.V92AnalogueSession(law="pcmu", fractional_tx_timing=True)
        phase4 = p4c.analogue_side(reference.profile, "pcmu", lu=1000.0,
                                   analogue_trn=12096, initial_sign=0)
        trn2u = np.asarray(phase4.transmit(10000), dtype=float)
        self.assertTrue(phase4.tx_trace)
        self.assertTrue(all(event["kind"] == "TRN" for event in phase4.tx_trace))

        rms = {}
        for label, delay in (("integer", 1.0), ("jp_5000", 0.5 + 0x5000 / 65536.0)):
            session = asn.V92AnalogueSession(law="pcmu", fractional_tx_timing=True)
            output = session._apply_tx_fractional_timing(
                trn2u, np.full(len(trn2u), delay, dtype=float))
            steady = output[asn.TX_DELAY_BULK + 32:]
            rms[label] = float(np.sqrt(np.mean(steady * steady)))
        level_delta_db = 20.0 * np.log10(rms["jp_5000"] / rms["integer"])
        self.assertLess(abs(level_delta_db), 0.2)

    def test_trn1d_scan_history_extends_incrementally_with_spec_sequence(self):
        analogue = ex.AnalogueSide(descriptor())
        analogue._ensure_trn1d_history(ex.TRN_MIN + 6)
        prefix = list(analogue._trn1d_history)
        analogue._ensure_trn1d_history(ex.TRN_MIN + 6 + 6 * 500)
        self.assertEqual(analogue._trn1d_history[:len(prefix)], prefix)
        self.assertEqual(analogue._trn1d_history,
                         ex._trn1d_bits(ex.TRN_MIN + 6 + 6 * 500))

    def test_analogue_phase3_rejects_decoy_sd_flip_then_accepts_real_jd(self):
        law = "pcmu"
        decoy = (v90_digital_tx.sd_codewords(78, law)
                 + v90_digital_tx.sd_prime_codewords(78, law))
        filler = bytes([0xFF]) * (ex.TRN_MIN + 600)  # silence: no valid TRN1d follows the decoy

        tx = v90_digital_tx.V90DigitalPhase3Transmitter(78, law)
        real = (v90_digital_tx.sd_codewords(78, law)
                + v90_digital_tx.sd_prime_codewords(78, law)
                + tx.trn1d(ex.TRN_MIN)
                + tx.jd_frame(0x3FFFFF))
        pcm = np.concatenate((
            v34_g711.decode(np.frombuffer(decoy, dtype=np.uint8), law),
            v34_g711.decode(np.frombuffer(filler, dtype=np.uint8), law),
            v34_g711.decode(np.frombuffer(real, dtype=np.uint8), law),
        )).astype(float)

        analogue = ex.AnalogueSide(descriptor())
        for start in range(0, len(pcm), 160):
            analogue.feed(pcm[start:start + 160])

        rejected = [value for name, value in analogue.events
                    if name == "sd-candidate-rejected"]
        self.assertEqual(len(rejected), 1, analogue.events)
        self.assertLess(rejected[0]["trn1d_match"], ex.TRN1D_MIN_MATCH)
        self.assertEqual(analogue.state, "jd-received", analogue.events)
        self.assertEqual(analogue.jd["rate_mask"], 0x3FFFFF)
        self.assertEqual(sum(1 for name, _ in analogue.events if name == "jd"), 1)

    def test_nonzero_dil_phase3_completes_over_both_g711_laws(self):
        for ucodes in ((37,), (37, 82), (1, 18, 35, 52, 69, 86, 103)):
            ja = descriptor(dil_ucodes=ucodes)
            parsed = v92_ja.parse_ja_descriptor(ja)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["N"], len(ucodes))
            self.assertEqual(parsed["Ucodes"], list(ucodes))
            self.assertEqual(parsed["bits"], len(ja))

            for law in ("pcmu", "pcma"):
                with self.subTest(law=law, dil_count=len(ucodes)):
                    analogue, digital = loop(law, ja_descriptor=ja, blocks=500)
                    self.assertEqual(digital.ja["N"], len(ucodes), digital.events)
                    self.assertGreater(digital.dil_symbols_sent, 2040, digital.events)
                    names = [name for name, _ in digital.events]
                    for event in ("dil-stop-request", "dil-stopped", "cpt", "ri-sent", "e1u", "r-bar-i-sent"):
                        self.assertIn(event, names, digital.events)
                    self.assertTrue(analogue.e1u_sent, analogue.events)
                    self.assertEqual(digital.state, "phase4-ready")

    def test_both_sides_complete_their_half_under_both_laws_and_delays(self):
        for law in ("pcmu", "pcma"):
            for delay, silence in ((0, 560), (240, 560), (241, 563), (600, 571)):
                with self.subTest(law=law, delay=delay, silence=silence):
                    analogue, digital = loop(law, delay=delay, silence=silence,
                                             rate_mask=0b1010101010101010101010)
                    names = [n for n, _ in digital.events]
                    self.assertEqual(digital.ja, v92_ja.parse_ja_descriptor(descriptor()))
                    # The digital modem saw Su, its flip, Su again, and the last flip; sent Jp and Jp'.
                    for step in ("su", "su-flip-1", "su-back", "su-flip-3", "jp-prime-sent", "trn1u-2"):
                        self.assertIn(step, names, digital.events)
                    self.assertGreaterEqual(digital.jp_sent, 1)
                    self.assertGreater(digital.scr_symbols, 0)
                    # The analogue modem parsed Jd (rate mask exact), then Jp, then Jp'.
                    self.assertEqual(analogue.jd["rate_mask"], 0b1010101010101010101010)
                    self.assertEqual(analogue.jp["kind"], "jp")
                    jp_event = next(value for name, value in analogue.events
                                    if name == "jp-decoded")
                    self.assertEqual(jp_event["phase_fraction"],
                                     analogue.jp["phase_fraction"])
                    self.assertEqual(jp_event["phase_symbols"],
                                     analogue.jp["phase_symbols"])
                    self.assertEqual(jp_event["training_8point"],
                                     analogue.jp["training_8point"])
                    # The default transmitter emits an integer 24T first
                    # Su-bar. The 24.5T option is tested separately above.
                    self.assertAlmostEqual(analogue.jp["phase_symbols"], 0.0, delta=1 / 65536)
                    self.assertTrue(analogue.jp_prime_seen, analogue.events)
                    self.assertEqual(analogue.state, "jp-prime-seen")
                    # The tail of Phase 3 (N = 0): Ri, CPt, R-bar-i, E1u.
                    for step in ("ri-sent", "cpt", "r-bar-i-sent"):
                        self.assertIn(step, [n for n, _ in digital.events], digital.events)
                    self.assertEqual(digital.state, "phase4-ready")
                    self.assertEqual(digital.cpt["kind"], "cpt")
                    self.assertEqual(digital.cpt["drn"], 9)
                    self.assertEqual(digital.cpt["constellations"], ((10, 20, 30),))
                    for step in ("ri", "r-bar-i"):
                        self.assertIn(step, [n for n, _ in analogue.events], analogue.events)
                    self.assertTrue(analogue.e1u_sent)
                    # The digital modem also saw E1u end: its sample is where the analogue modem's Phase 4 begins.
                    self.assertIsNotNone(digital.trn2u_start)
                    self.assertIn("e1u", [n for n, _ in digital.events])
                    self.assertIn(digital.e1u_last_sign, (0, 1))
                    self.assertEqual(digital.e1u_last_sign,
                                     int(digital._rx[digital.trn2u_start - 1] < 0))
                    self.assertGreaterEqual(analogue.cpt_frames_sent, 1)

    def test_the_analogue_su_sequence_has_the_figure_10_segments(self):
        analogue = ex.AnalogueSide(descriptor(), lu=1000.0)
        analogue._su_state, analogue._su_n, analogue._su_start = "su1", 0, 0
        x = []
        for i in range(400):
            if i == 250:
                analogue._jp_seen_flag = True          # Jp decoded part-way through the hold
            x.append(analogue._su_sample())
        x = np.array(x)
        a = 1000.0 * 1.5 ** 0.5
        su = np.array([a, 0, a, -a, 0, -a])
        self.assertTrue(np.allclose(x[:144], np.tile(su, 24)))             # Su, 144T
        self.assertTrue(np.allclose(x[144:168], -np.tile(su, 4)))          # Su-bar, 24T
        hold_end = 168 + (250 - 168) // 6 * 6 + 6                          # held to the period end
        self.assertTrue(np.allclose(x[168:hold_end], np.tile(su, (hold_end - 168) // 6)))
        self.assertTrue(np.allclose(x[hold_end:hold_end + 24], -np.tile(su, 4)))   # Su-bar, 24T
        self.assertEqual(set(np.abs(x[hold_end + 24:hold_end + 24 + 100])), {1000.0})  # TRN1u

    def test_the_digital_modem_waits_in_silence_until_it_has_ja(self):
        analogue = ex.AnalogueSide(descriptor())
        digital = ex.DigitalSide("pcmu")
        silence = bytes(digital.tx(160))
        self.assertEqual(set(silence), {digital._silence})

    def test_ja_stops_at_a_12_bit_boundary_and_goes_silent(self):
        analogue, digital = loop("pcmu")
        ja0 = analogue.plan.boundaries["ja"][0]
        self.assertEqual((analogue._ja_end - ja0) % 12, 0)
        x = analogue.plan.samples
        # After the end the analogue modem sends nothing.
        out = ex.AnalogueSide(descriptor())
        out._pos = analogue._ja_end + 10
        out._ja_end = analogue._ja_end
        self.assertTrue(np.all(out.tx(160) == 0))

    def test_no_ru_means_no_progress(self):
        digital = ex.DigitalSide("pcmu")
        for _ in range(100):
            digital.feed(np.zeros(160))
        self.assertEqual(digital.state, "wait-ru")
        self.assertEqual(set(digital.tx(160)), {digital._silence})


if __name__ == "__main__":
    unittest.main()
