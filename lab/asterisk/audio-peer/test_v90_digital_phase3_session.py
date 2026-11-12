"""Ja-to-digital-Phase-3 integration checks without RTP or a provider."""

import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

import v34_tx
import v34_g711
import v34_data
from v90_cp import parse_cp
from v90_digital_phase4_session import V90DigitalAnswererPhase4Session
from v90_digital_phase3_session import V90DigitalAnswererPhase3Session
from v90_mp_policy import answerer_mp_profile, select_analog_upstream_profile
from v90_upstream_data import V90UpstreamDataReceiver
from v90_digital_phase4 import V90DigitalPhase4Encoder
from v90_digital_phase4_rx import V90DigitalPhase4Receiver
from v90_lapm import V90LapmLink
from v90_analogue_phase4 import V90AnaloguePhase4Controller, qam_symbols

import v90_test_ja_waveform as control
from test_v90_cp import _encode_symbols, _frame  # noqa: E402


def _encode_continuous_4point_segments(segments):
    """Independent 4-point scrambler/differential mapper across signal runs."""
    register = [0] * 23
    rotation = 0
    mapped = []
    for bits in segments:
        run = []
        padded = list(bits)
        padded += [0] * (len(padded) % 2)
        for offset in range(0, len(padded), 2):
            scrambled = []
            for bit in padded[offset:offset + 2]:
                value = bit ^ register[4] ^ register[22]
                register = [value] + register[:-1]
                scrambled.append(value)
            rotation = (rotation + 2 * scrambled[1] + scrambled[0]) % 4
            run.append(rotation)
        mapped.append(np.asarray([v34_tx.P0 * v34_tx.ROT_CW[value]
                                  for value in run]))
    return mapped


def _encode_continuous_16point_segments(segments):
    """Encode V.34/V.90 16-point symbol decisions across protocol segments."""
    register = [0] * 23
    rotation = 0
    mapped = []
    for bits in segments:
        padded = list(bits)
        padded += [0] * ((-len(padded)) % 4)
        run = []
        for offset in range(0, len(padded), 4):
            scrambled = []
            for bit in padded[offset:offset + 4]:
                value = int(bit) ^ register[4] ^ register[22]
                register = [value] + register[:-1]
                scrambled.append(value)
            rotation = (rotation + 2 * scrambled[1] + scrambled[0]) % 4
            quarter = 2 * scrambled[3] + scrambled[2]
            x, y = v34_tx.C16[quarter]
            run.append(complex(x, y) * v34_tx.ROT_CW[rotation])
        mapped.append(np.asarray(run, dtype=np.complex128))
    return mapped


def _paired_lapm_caller_bits(upstream_rate, downstream_profile, law,
                             data_frame_bits):
    """Capture a valid caller LAPM bit stream and its DTE payload from a pair."""
    downstream_tx = V90DigitalPhase4Encoder(downstream_profile, law)
    downstream_rx = V90DigitalPhase4Receiver(downstream_profile, law)
    frame_bits = downstream_tx.s + downstream_tx.k
    up_count = 3 * data_frame_bits
    down_count = int(round(up_count * downstream_profile["rate"] /
                           upstream_rate))
    if down_count % frame_bits:
        raise AssertionError("paired LAPM slice is not V.90-frame aligned")

    caller = V90LapmLink(rate=upstream_rate, calling_party=True)
    answerer = V90LapmLink(
        rate=int(round(downstream_profile["rate"])), calling_party=False)
    caller_payload = bytes((index * 29 + 7) & 0xFF for index in range(192))
    answer_payload = bytes((index * 53 + 11) & 0xFF for index in range(128))
    caller.feed_bytes(caller_payload)
    answerer.feed_bytes(answer_payload)
    received = bytearray()
    caller_received = bytearray()
    sent_bits = []

    # The real session emits B1d before its LAPM data source is read. Advance
    # the source mapper/decoder and the caller's receive path over the same
    # all-ones training frames before recording LAPM traffic.
    b1d_bits = [1] * (48 * frame_bits)
    b1d = downstream_tx.encode_bits(b1d_bits)
    if downstream_rx.decode_codewords(b1d) != b1d_bits:
        raise AssertionError("paired LAPM B1d preamble did not round-trip")
    caller.feed_line_bits(b1d_bits)

    def exchange():
        up_bits = caller.read_tx_bits(up_count)
        sent_bits.extend(up_bits)
        down_bits = answerer.read_tx_bits(down_count)
        down_codewords = downstream_tx.encode_bits(down_bits)
        answerer.feed_line_bits(up_bits)
        caller.feed_line_bits(downstream_rx.decode_codewords(down_codewords))
        received.extend(answerer.take_received())
        caller_received.extend(caller.take_received())

    for _ in range(200):
        exchange()
        if caller.v42_mode == answerer.v42_mode == "lapm":
            break
    else:
        raise AssertionError("paired LAPM setup did not establish")

    for _ in range(3000):
        exchange()
        if (bytes(received) == caller_payload
                and bytes(caller_received) == answer_payload):
            break
    if bytes(received) != caller_payload or bytes(caller_received) != answer_payload:
        raise AssertionError("paired LAPM source did not deliver both payloads")
    return sent_bits, caller_payload, answer_payload, up_count, down_count


class V90DigitalAnswererPhase3SessionTests(unittest.TestCase):
    def test_explicit_type1_mp_is_used_by_digital_answerer_phase4(self):
        mp = {
            "mp_type": 1, "max_a2d_drn": 12, "trellis": 0,
            "nonlinear": 1, "shaping": 1, "rate_mask": 0x1FFF,
            "precoder": [520, -52, -119, 1, 116, 137],
        }
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcmu",
            uinfo=67, rate_mask=0xFFFF, phase4_mp_profile=mp)
        with patch.object(session, "_arm_phase4_receiver"):
            session._start_phase4()
        self.assertEqual(session.phase4.transmitter.mp_type, 1)
        self.assertEqual(session.phase4.transmitter._mp_fields, {
            "max_a2d_drn": 12, "trellis": 0, "nonlinear": 1,
            "shaping": 1, "rate_mask": 0x1FFF,
            "precoder": (520, -52, -119, 1, 116, 137),
        })
        self.assertEqual(session.events[-1][0], "phase4-started")

    def test_stage_timing_report_correlates_slow_callback_costs(self):
        with patch.dict("os.environ", {"V90_RX_STAGE_TIMING": "1"}):
            session = V90DigitalAnswererPhase3Session(
                baud=control.BAUD, carrier=control.CARRIER, law="pcmu",
                uinfo=67, rate_mask=0xFFFF)
        session._stage_timing_rows.append({
            "total": 20_000_000, "symbols": 51, "rx_sample": 17_458,
            "observation": 400_000, "phase4_control": 8_000_000,
            "phase4_scanner": 1_000_000, "cpt_candidate_search": 0,
            "cpt_candidate_replay": 0, "cpt_candidate_advance": 0,
            "phase4_profile_e": 0, "profile_prepare": 0,
            "profile_select": 0, "receiver_construct": 0,
            "phase4_fsm": 0, "lapm_setup": 0,
            "upstream_data_lapm": 7_000_000,
        })
        report = session.stage_timing_report()
        self.assertEqual(report["slow_callback_threshold_ms"], 8.0)
        self.assertEqual(report["slow_callback_count"], 1)
        row = report["slow_callbacks"][0]
        self.assertEqual(row["rx_sample"], 17_458)
        self.assertEqual(row["symbols"], 51)
        self.assertEqual(row["total_ms"], 20.0)
        self.assertEqual(row["phase4_control_ms"], 8.0)
        self.assertEqual(row["upstream_data_lapm_ms"], 7.0)

    def test_cpt_candidate_trace_is_opt_in_and_reports_worst_batch(self):
        with patch.dict("os.environ", {"V90_CPT_TRACE": "1"}):
            session = V90DigitalAnswererPhase3Session(
                baud=control.BAUD, carrier=control.CARRIER, law="pcmu",
                uinfo=67, rate_mask=0xFFFF)
        self.assertTrue(session._cpt_trace_enabled)
        self.assertEqual(session.cpt_candidate_trace_report()["count"], 0)
        session._record_cpt_candidate_timing(
            "advance", 1234, candidate_count=12, work_symbols=12,
            started_ns=0)
        session._record_cpt_candidate_timing(
            "replay", 1300, candidate_count=24, work_symbols=300,
            started_ns=0)
        report = session.cpt_candidate_trace_report()
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["max_kind"], "replay")
        self.assertEqual(report["max_candidate_count"], 24)
        self.assertEqual(report["max_work_symbols"], 300)
        self.assertEqual(report["rows"][0]["receive_symbol"], 1234)

        session.upstream.receiver = SimpleNamespace(
            eq=SimpleNamespace(k=1350))
        session._record_cpt_feed_batch(
            started_ns=0, trace_before=0, start_symbol=1300)
        batch = session.cpt_candidate_trace_report()["rows"][-1]
        self.assertEqual(batch["kind"], "feed_batch")
        self.assertEqual(batch["candidate_calls"], 2)
        self.assertEqual(batch["batch_symbols"], 50)
        self.assertEqual(batch["work_symbols"], 312)

    def test_s_transition_search_stops_after_phase4_receiver_is_armed(self):
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcmu",
            uinfo=67, rate_mask=0xFFFF)
        session._detector = SimpleNamespace(
            feed=lambda *_args, **_kwargs: self.fail(
                "Phase 3 S detector ran after Phase 4 receive was armed"))
        session._phase4_receive_armed = True
        fed = []
        session.upstream.feed = lambda samples: fed.append(samples.copy()) or []
        session.phase4 = object()

        session.feed_pcm(np.zeros(160), sample_at=0)

        self.assertEqual(len(fed), 1)
        self.assertEqual(len(fed[0]), 160)

    def test_data_tracking_absorbs_b1_gain_without_refined_taps(self):
        session = object.__new__(V90DigitalAnswererPhase3Session)
        session._upstream_data_tracking = False
        gain = 0.052 - 0.00005j
        original = np.asarray([1 + 2j, -3 + 4j], dtype=np.complex128)
        eq = SimpleNamespace(h=original.copy())
        data_receiver = SimpleNamespace(
            _gain=gain, equalizer_taps=None, receiver=object())
        session.upstream_data = data_receiver
        session.upstream = SimpleNamespace(
            receiver=SimpleNamespace(eq=eq),
            begin_data_tracking=lambda receiver: None)

        session._activate_upstream_data_tracking()

        np.testing.assert_allclose(eq.h, original / gain)
        self.assertEqual(data_receiver._gain, 1 + 0j)
        self.assertTrue(session._upstream_data_tracking)

    def _minimum_trn_ja(self):
        descriptor = control._ja_descriptor(7, 19, 31)
        sequence, _, _ = control._preamble_and_trn()
        scrambled, state = control._scramble([1] * (2 * 512))
        trn = []
        for offset in range(0, len(scrambled), 2):
            i1, i2 = scrambled[offset:offset + 2]
            trn.append((1 + 1j) * [1, -1j, -1, 1j][(2 * i2 + i1) % 4])
        final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
        ja, _ = control._differential(descriptor * 3, final_rotation, state)
        return control._modulate(np.concatenate((sequence[:432], trn, ja)))

    def test_crc_valid_ja_starts_phase3_and_later_s_events_reach_fsm(self):
        samples = self._minimum_trn_ja()
        pair = np.concatenate((v34_tx.s_signal(128), v34_tx.s_signal(16, True)))
        pulse = control._modulate(pair)
        silence = np.zeros(8000)
        later = np.concatenate((pulse, silence, pulse))
        for law in ("pcmu", "pcma"):
            for phase4_16point in (False, True):
                with self.subTest(law=law, phase4_16point=phase4_16point):
                    session = V90DigitalAnswererPhase3Session(
                        baud=control.BAUD, carrier=control.CARRIER, law=law,
                        uinfo=67, rate_mask=0xFFFF,
                        phase3_16point=phase4_16point)
                    descriptors = []
                    for offset in range(0, len(samples), 160):
                        descriptors.extend(session.feed_pcm(
                            samples[offset:offset + 160], sample_at=offset))

                    self.assertEqual(len(descriptors), 1)
                    self.assertEqual(session.descriptor["N"], 7)
                    output = session.read_g711(4096, law)
                    self.assertEqual(len(output), 4096)
                    self.assertEqual(session.transmitter.state, session.transmitter.JD)

                    # Ja is followed by protocol phases where the caller is
                    # silent. Those samples are not QAM decisions: feedback
                    # must hold the trained equalizer until Phase 4 CP begins.
                    eq = session.upstream.receiver.eq
                    self.assertFalse(eq.adapt)
                    self.assertTrue(eq._input_guard.armed)
                    self.assertTrue(eq._input_guard.holding)
                    held_taps = eq.h.copy()
                    held_position = eq.dt
                    post_ja_silence = np.zeros(8000)
                    silence_start = len(samples)
                    for offset in range(0, len(post_ja_silence), 160):
                        session.feed_pcm(post_ja_silence[offset:offset + 160],
                                         sample_at=silence_start + offset)
                    np.testing.assert_array_equal(eq.h, held_taps)
                    self.assertEqual(eq.dt, held_position)

                    events = []
                    observed_transitions = 0
                    start = silence_start + len(post_ja_silence)
                    for offset in range(0, len(later), 160):
                        events.extend(session.feed_pcm(
                            later[offset:offset + 160], sample_at=start + offset))
                        count = len(session.transitions)
                        if count > observed_transitions:
                            observed_transitions = count
                            if count == 1:
                                session.read_g711(4096)
                                self.assertEqual(session.transmitter.state,
                                                 session.transmitter.DIL)
                            elif count == 2:
                                self.assertEqual(session.transmitter.state,
                                                 session.transmitter.DIL)
                                self.assertTrue(session.transmitter._dil._stop_requested)
                                tail = session.read_g711(100000)
                                self.assertEqual(len(tail), 100000)
                                self.assertTrue(session.transmitter._dil.stopped)
                                self.assertEqual(
                                    session.transmitter.state,
                                    session.transmitter.PHASE4_READY)
                                self.assertIsNotNone(session.phase4)
                                self.assertEqual(session.phase4.state,
                                                 session.phase4.transmitter.WAIT_CPT)
                                expected_constellation = 16 if phase4_16point else 4
                                self.assertEqual(session.phase4_constellation,
                                                 expected_constellation)
                                self.assertEqual(session.phase4_receiver.constellation,
                                                 expected_constellation)
                                self.assertEqual(
                                    session.upstream.receiver._phase4_constellation,
                                    expected_constellation)
                                self.assertFalse(eq.adapt)
                                self.assertFalse(eq._input_guard.release_enabled)
                                self.assertEqual(session.phase4_mp_profile["trellis"], 0)
                    self.assertEqual(events, [])  # transitions are controller events
                    self.assertEqual(len(session.transitions), 2)
                    self.assertEqual(session.transmitter.s_transitions, 2)
                    self.assertEqual([name for name, _ in session.events].count("ja-accepted"), 1)
                    self.assertEqual([name for name, _ in session.events].count("phase4-started"), 1)
                    phase4_taps = eq.h.copy()
                    phase4_silence_start = start + len(later)
                    for offset in range(0, 8000, 160):
                        session.feed_pcm(np.zeros(160),
                                         sample_at=phase4_silence_start + offset)
                    np.testing.assert_array_equal(eq.h, phase4_taps)
                    self.assertTrue(eq._input_guard.holding)
                    self.assertEqual(session.phase4_receiver.bits_seen, 0)
                    cpt_bits = _frame(
                        kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                        tx_masks=(tuple(range(128)),) * 3,
                        law=int(law == "pcma"), ack=0,
                        spectral_sign_bits=0)
                    reference_power = eq._input_guard.reference
                    # The input-power threshold leads the CPt symbol center;
                    # these 13 active symbols model that measured offset.
                    for index in range(13):
                        decision = ((index * 3) % 4, 0)
                        session._acquire_phase4_cpt(decision, reference_power)
                    repeated_cpt = _encode_symbols(
                        cpt_bits + cpt_bits,
                        constellation=expected_constellation)
                    for index, decision in enumerate(repeated_cpt):
                        session._acquire_phase4_cpt(decision, reference_power)
                        if index == len(repeated_cpt) // 2 - 1:
                            self.assertFalse(session._phase4_cpt_acquired)
                    self.assertTrue(session._phase4_cpt_acquired)
                    self.assertTrue(session.phase4_profiles)
                    self.assertEqual(session.phase4_profiles[0]["kind"], "cpt")
                    self.assertTrue(eq.adapt)
                    self.assertTrue(eq._input_guard.release_enabled)

    def test_rejects_receive_gap(self):
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcma",
            uinfo=67, rate_mask=0xFFFF)
        session.feed_pcm(np.zeros(160), sample_at=1000)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            session.feed_pcm(np.zeros(160), sample_at=1320)

    def test_parsed_cpt_profile_starts_phase4_transmitter(self):
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcma",
            uinfo=67, rate_mask=0xFFFF)
        samples = self._minimum_trn_ja()
        for offset in range(0, len(samples), 160):
            session.feed_pcm(samples[offset:offset + 160], sample_at=offset)
        session.read_g711(4096)

        pair = np.concatenate((v34_tx.s_signal(128), v34_tx.s_signal(16, True)))
        pulse = control._modulate(pair)
        later = np.concatenate((pulse, np.zeros(8000), pulse,
                                 np.zeros(8000), pulse))
        start = len(samples)
        transitions = 0
        for offset in range(0, len(later), 160):
            session.feed_pcm(later[offset:offset + 160], sample_at=start + offset)
            if len(session.transitions) != transitions:
                transitions = len(session.transitions)
                if transitions == 1:
                    session.read_g711(4096)
                elif transitions == 2:
                    session.read_g711(1)
                elif transitions == 3:
                    session.read_g711(100000)

        self.assertIsNotNone(session.phase4)
        self.assertEqual(session.phase4.state, session.phase4.transmitter.WAIT_CPT)
        profile = parse_cp(_frame(
            kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=1, ack=0,
            spectral_sign_bits=0))
        self.assertIsNotNone(profile)
        self.assertTrue(session.receive_phase4_profile(profile), session.phase4_error)
        self.assertEqual(session.phase4.transmitter.state,
                         session.phase4.transmitter.WAIT_CP)
        self.assertEqual(session.phase4_mp_profile["trellis"], 0)
        self.assertEqual(len(session.read_g711(4096)), 4096)
        self.assertGreater(session.phase4.transmitter.mp_frames_sent, 0)

    def test_4point_cpt_pcm_reacquires_at_sync_and_enters_phase4(self):
        # Keep one pulse-shaped stream so carrier phase and filter history
        # remain continuous across TRN, Ja, silent DIL, S transitions, Ri,
        # and repeated CPt frames.
        descriptor = control._ja_descriptor(7, 19, 31)
        prefix, _, _ = control._preamble_and_trn()
        pre = prefix[:432]
        scrambled, state = control._scramble([1] * (2 * 512))
        trn = np.asarray([
            (1 + 1j) * (1, -1j, -1, 1j)[
                (2 * scrambled[i + 1] + scrambled[i]) % 4]
            for i in range(0, len(scrambled), 2)
        ])
        final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
        ja, _ = control._differential(descriptor * 3, final_rotation, state)
        s_signal = np.concatenate((v34_tx.s_signal(128),
                                   v34_tx.s_signal(16, True)))
        one_second = int(round(control.BAUD))
        # Agere 1648C's measured DIL pass is 29,760 8-kHz codewords = 3.72 s.
        dil_silence = int(round(3.72 * control.BAUD))

        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                cpt_bits = _frame(
                    kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                    tx_masks=(tuple(range(128)),) * 3,
                    law=int(law == "pcma"), ack=0, spectral_sign_bits=0)
                cpt_symbols = _encode_symbols(cpt_bits * 3, constellation=4)
                cpt = np.asarray([
                    v34_tx.P0 * v34_tx.ROT_CW[rotation]
                    for rotation, _quarter in cpt_symbols
                ])
                symbols = np.concatenate((
                    pre, trn, ja,
                    np.zeros(one_second), s_signal,
                    np.zeros(dil_silence), s_signal,
                    np.zeros(192), cpt, np.zeros(32)))
                waveform = v34_tx.qam_modulate(
                    symbols, control.BAUD, control.CARRIER, .1,
                    beta=.15, span=8, power=2.0)
                waveform = v34_g711.roundtrip(waveform * 32768.0,
                                              law) / 32768.0

                session = V90DigitalAnswererPhase3Session(
                    baud=control.BAUD, carrier=control.CARRIER, law=law,
                    uinfo=67, rate_mask=0xFFFF, phase3_16point=False)
                seen_transitions = 0
                ja_seen = False
                dil_clocked = 0
                for offset in range(0, len(waveform), 160):
                    descriptors = session.feed_pcm(
                        waveform[offset:offset + 160], sample_at=offset)
                    if descriptors and not ja_seen:
                        ja_seen = True
                        session.read_g711(4096, law)
                    transition_count = len(session.transitions)
                    if transition_count > seen_transitions:
                        seen_transitions = transition_count
                        if seen_transitions == 1:
                            session.read_g711(4096, law)
                            self.assertEqual(session.transmitter.state,
                                             session.transmitter.DIL)
                        elif seen_transitions == 2:
                            self.assertEqual(session.transmitter.state,
                                             session.transmitter.DIL)
                            self.assertTrue(
                                session.transmitter._dil._stop_requested)
                    if session.transmitter.state == session.transmitter.DIL:
                        tx_chunk = session.read_g711(160, law)
                        self.assertEqual(len(tx_chunk), 160)
                        dil_clocked += len(tx_chunk)

                self.assertEqual(seen_transitions, 2)
                self.assertTrue(ja_seen)
                self.assertGreater(dil_clocked, 0)
                self.assertTrue(session.transmitter._dil.stopped)
                self.assertEqual(session.transmitter.state,
                                 session.transmitter.PHASE4_READY)
                self.assertIsNotNone(session.phase4)
                self.assertTrue(session._phase4_feedback_released_early)
                self.assertTrue(session._phase4_cpt_acquired)
                self.assertGreaterEqual(
                    sum(profile.get("kind") == "cpt"
                        for profile in session.phase4_profiles), 1)
                self.assertIsNotNone(session.phase4)
                self.assertEqual(session.phase4.state,
                                 session.phase4.transmitter.WAIT_CP)
                # Accepted CPt drives Ri(24T) and all 2040 TRN2d codewords.
                self.assertEqual(len(session.read_g711(2064, law)), 2064)

    def test_pcm_control_exchange_reaches_b1d_and_lapm_payloads(self):
        # Preserve one symbol stream and one scrambler/differential state
        # through CPt, allowed SCR, repeated CP, CP-prime, and E. Silence here
        # would corrupt the continuous control decoder rather than represent
        # an equivalent idle interval.
        descriptor = control._ja_descriptor(7, 19, 31)
        prefix, _, _ = control._preamble_and_trn()
        pre = prefix[:432]
        scrambled, state = control._scramble([1] * (2 * 512))
        trn = np.asarray([
            (1 + 1j) * (1, -1j, -1, 1j)[
                (2 * scrambled[i + 1] + scrambled[i]) % 4]
            for i in range(0, len(scrambled), 2)
        ])
        final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
        ja, _ = control._differential(descriptor * 3, final_rotation, state)
        s_signal = np.concatenate((v34_tx.s_signal(128),
                                   v34_tx.s_signal(16, True)))
        one_second = int(round(control.BAUD))
        dil_silence = int(round(3.72 * control.BAUD))
        common = dict(drn=7, selectors=(0, 1, 2, 0, 1, 2),
                      tx_masks=(tuple(range(128)),) * 3,
                      spectral_sign_bits=0)
        for law in ("pcmu", "pcma"):
            for completion in ("cp-prime", "e"):
                for phase4_constellation in (4, 16):
                    with self.subTest(law=law, completion=completion,
                                      phase4_constellation=phase4_constellation):
                        law_bit = int(law == "pcma")
                        cpt_bits = _frame(kind="cpt", ack=0, law=law_bit,
                                          **common)
                        cp_bits = _frame(kind="cp", ack=0, law=law_bit,
                                         **common)
                        cp_prime_bits = _frame(kind="cp", ack=1, law=law_bit,
                                               **common)
                        # CPt starts after its protocol reset. SCR and following
                        # CP messages retain that state across message frames.
                        scr_symbols = int(round(0.75 * control.BAUD))
                        bits_per_symbol = 4 if phase4_constellation == 16 else 2
                        segments = [cpt_bits * 3,
                                    [1] * (bits_per_symbol * scr_symbols),
                                    cp_bits * 12]
                        if completion == "cp-prime":
                            segments.append(cp_prime_bits * 4)
                        segments.append([1] * 20)
                        encoder = (_encode_continuous_16point_segments
                                   if phase4_constellation == 16
                                   else _encode_continuous_4point_segments)
                        cpt_run, scr_run, cp_run, *tail_runs = encoder(segments)
                        if completion == "cp-prime":
                            cp_prime_run, e_run = tail_runs
                            b1_run = np.asarray([1 + 0j] * 32)
                            payload_bits = []
                        else:
                            e_run = tail_runs[0]
                            cp_prime_run = np.empty(0, dtype=np.complex128)

                            # Build an actual analogue-upstream V.34 B1 and
                            # payload using the profile selected from this CP
                            # and the answerer's MP. This exercises the same
                            # production B1 receiver/data decoder reached by
                            # the session after E; the prior all-ones tail was
                            # only a control-boundary rejection check.
                            cp_profile = parse_cp(cp_bits)
                            upstream_profile = select_analog_upstream_profile(
                                cp_profile, answerer_mp_profile(control.BAUD),
                                baud=control.BAUD,
                                carrier=control.CARRIER)
                            upstream_rx = V90UpstreamDataReceiver(
                                upstream_profile, remote_role="answer")
                            upstream_tx = v34_data.DataTx(
                                upstream_profile.rate,
                                int(round(upstream_profile.baud)),
                                expanded=upstream_profile.shaping,
                                precoder=upstream_profile.precoder,
                                nonlinear=upstream_profile.nonlinear,
                                trellis=upstream_profile.trellis)
                            (payload_bits, expected_dte, expected_server_dte,
                             up_slice_bits, down_slice_bits) = _paired_lapm_caller_bits(
                                 upstream_profile.rate, cp_profile, law,
                                 upstream_tx.bits_per_data_frame)
                            upstream_scrambler = v34_tx.Scrambler("answer")
                            b1_bits = [upstream_scrambler.step(1)
                                       for _ in range(
                                           upstream_tx.bits_per_data_frame)]
                            self.assertEqual(b1_bits,
                                             upstream_rx.receiver.expected)
                            b1_run = np.asarray(upstream_tx.data_frame(
                                b1_bits),
                                dtype=np.complex128)
                            payload_run = []
                            for offset in range(
                                    0, len(payload_bits),
                                    upstream_tx.bits_per_data_frame):
                                scrambled = [upstream_scrambler.step(bit)
                                             for bit in payload_bits[
                                                 offset:offset +
                                                 upstream_tx.bits_per_data_frame]]
                                payload_run.extend(upstream_tx.data_frame(
                                    scrambled))
                            payload_run = np.asarray(
                                payload_run, dtype=np.complex128)
                            # The control mapper uses unit-scale points while
                            # V.34's absolute constellation has much higher
                            # numeric energy. Normalize the synthetic channel
                            # level once across B1 and payload; qam_modulate's
                            # explicit power=2 describes the control stream's
                            # nominal symbol power.
                            data_symbols = np.concatenate((b1_run,
                                                           payload_run))
                            data_symbols *= np.sqrt(
                                2.0 / np.mean(np.abs(data_symbols) ** 2))
                            b1_run = data_symbols[:len(b1_run)]
                            payload_run = data_symbols[len(b1_run):]
                        symbols = np.concatenate((
                            pre, trn, ja,
                            np.zeros(one_second), s_signal,
                            np.zeros(dil_silence), s_signal,
                            np.zeros(192), cpt_run,
                            scr_run, cp_run, cp_prime_run, e_run, b1_run,
                            payload_run if completion == "e" else
                            np.empty(0, dtype=np.complex128),
                            np.zeros(32)))
                        waveform = v34_tx.qam_modulate(
                            symbols, control.BAUD, control.CARRIER, .1,
                            beta=.15, span=8, power=2.0)
                        waveform = v34_g711.roundtrip(waveform * 32768.0,
                                                      law) / 32768.0
                        session = V90DigitalAnswererPhase3Session(
                            baud=control.BAUD, carrier=control.CARRIER, law=law,
                            uinfo=67, rate_mask=0xFFFF,
                            enable_lapm=True,
                            phase3_16point=phase4_constellation == 16)
                        if completion == "e":
                            session.feed_bytes(expected_server_dte)
                        transmit = bytearray()
                        transition_count = 0
                        ja_seen = False
                        for offset in range(0, len(waveform), 160):
                            descriptors = session.feed_pcm(
                                waveform[offset:offset + 160], sample_at=offset)
                            if descriptors and not ja_seen:
                                ja_seen = True
                            transition_count = len(session.transitions)
                            if session.transmit_ready:
                                transmit.extend(session.read_g711(160, law))

                        # Drain the final short TX remainder; the loop above polls
                        # in fixed 20 ms blocks, while the 16-point control stream
                        # can end between those block boundaries.
                        if session.transmit_ready:
                            transmit.extend(session.read_g711(160, law))
                        # Continue the digital answerer clock after the caller
                        # waveform ends so queued LAPM data can drain.
                        for _ in range(100):
                            if not session.transmit_ready:
                                break
                            transmit.extend(session.read_g711(160, law))

                        profiles = session.phase4_profiles
                        self.assertTrue(ja_seen)
                        self.assertEqual(transition_count, 2)
                        self.assertTrue(session._phase4_cpt_acquired)
                        self.assertEqual(session.analog_upstream_rate, 33600)
                        if phase4_constellation == 16:
                            self.assertTrue(
                                session._phase4_feedback_released_early)
                            self.assertTrue(
                                session.upstream.receiver.eq.adapt_nlms)
                        self.assertTrue(any(
                            p.get("kind") == "cp" and p.get("ack") == 0
                            for p in profiles))
                        if completion == "cp-prime":
                            self.assertTrue(any(
                                p.get("kind") == "cp" and p.get("ack") == 1
                                for p in profiles))
                        else:
                            self.assertFalse(any(
                                p.get("kind") == "cp" and p.get("ack") == 1
                                for p in profiles))
                        self.assertIsNotNone(session.phase4)
                        tx = session.phase4.transmitter
                        self.assertEqual(tx.state, tx.DATA_READY)
                        if completion == "e":
                            self.assertIsNotNone(session.upstream_data)
                            # V.90 assigns the analogue modem GPA even though
                            # this fixture models the analogue caller.
                            self.assertEqual(
                                session.upstream_data.remote_role, "answer")
                            self.assertEqual(session.upstream_data.state,
                                             session.upstream_data.DATA)
                            self.assertTrue(
                                session.upstream_data.receiver.diagnostics[
                                    "qualified"])
                            self.assertIsNotNone(session.lapm)
                            self.assertEqual(session.lapm.v42_mode, "lapm")
                            self.assertEqual(session.read_received_data(),
                                             expected_dte)
                            self.assertTrue(any(
                                name == "upstream-b1-start"
                                for name, _value in session.events))
                        self.assertEqual(tx._end_signal, completion)
                        self.assertGreater(tx.mp_frames_sent, 0)
                        self.assertGreater(tx.mp_prime_frames_sent, 0)
                        if completion == "e":
                            self.assertTrue(any(
                                name == "phase4-e-detected"
                                for name, _ in session.events))
                        # Ed is followed by 48 B1d frames (288 exact G.711
                        # octets). Check those codewords against a separate
                        # encoder instance initialized from the parsed CP profile.
                        from v90_digital_phase4 import V90DigitalPhase4Encoder
                        expected_encoder = V90DigitalPhase4Encoder(tx.cp_profile,
                                                                   law)
                        expected_bits = [1] * (48 * (expected_encoder.s
                                                      + expected_encoder.k))
                        expected_b1d = expected_encoder.encode_bits(expected_bits)
                        self.assertEqual(len(expected_b1d), 288)
                        b1d_at = bytes(transmit).find(expected_b1d)
                        self.assertGreaterEqual(b1d_at, 0,
                                                "B1d codewords not present in output")
                        self.assertEqual(
                            bytes(transmit[b1d_at:b1d_at + len(expected_b1d)]),
                            expected_b1d)

                        if completion == "e":
                            # Decode the answerer's downstream PCM after B1d.
                            # The replay caller must recover the exact answer
                            # payload from these received bits.
                            downstream_rx = V90DigitalPhase4Receiver(
                                tx.cp_profile, law)
                            encoded_tail = bytes(transmit[b1d_at:])
                            encoded_tail = encoded_tail[:
                                len(encoded_tail) // 6 * 6]
                            downstream_bits = downstream_rx.decode_codewords(
                                encoded_tail)
                            b1d_bit_count = 48 * (expected_encoder.s
                                                   + expected_encoder.k)
                            self.assertEqual(
                                downstream_bits[:b1d_bit_count], expected_bits)
                            session_downlink_bits = downstream_bits[
                                b1d_bit_count:]
                            replay_caller = V90LapmLink(
                                rate=session.analog_upstream_rate,
                                calling_party=True)
                            replay_caller.feed_bytes(expected_dte)
                            replay_caller.feed_line_bits(expected_bits)
                            received_server = bytearray()
                            down_offset = 0
                            for _up_offset in range(0, len(payload_bits),
                                                    up_slice_bits):
                                if bytes(received_server) == expected_server_dte:
                                    break
                                replay_caller.read_tx_bits(up_slice_bits)
                                down_chunk = session_downlink_bits[
                                    down_offset:down_offset + down_slice_bits]
                                if not down_chunk:
                                    break
                                replay_caller.feed_line_bits(down_chunk)
                                down_offset += down_slice_bits
                                received_server.extend(
                                    replay_caller.take_received())
                            self.assertEqual(bytes(received_server),
                                             expected_server_dte)

    def test_reactive_pcm_lapm_request_response_uses_one_media_clock(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                self._run_reactive_pcm_lapm_exchange(law)

    def _run_reactive_pcm_lapm_exchange(self, law):
        """Drive a caller response from the answerer's received G.711 data."""
        descriptor = control._ja_descriptor(7, 19, 31)
        prefix, _, _ = control._preamble_and_trn()
        pre = prefix[:432]
        scrambled_trn, scrambler_state = control._scramble([1] * (2 * 512))
        trn = np.asarray([
            (1 + 1j) * (1, -1j, -1, 1j)[
                (2 * scrambled_trn[index + 1]
                 + scrambled_trn[index]) % 4]
            for index in range(0, len(scrambled_trn), 2)
        ])
        final_rotation = (2 * scrambled_trn[-1]
                          + scrambled_trn[-2]) % 4
        ja, _ = control._differential(
            descriptor * 3, final_rotation, scrambler_state)
        s_signal = np.concatenate((v34_tx.s_signal(128),
                                   v34_tx.s_signal(16, True)))
        common = dict(drn=7, selectors=(0, 1, 2, 0, 1, 2),
                      tx_masks=(tuple(range(128)),) * 3,
                      spectral_sign_bits=0)
        cpt_fields = dict(kind="cpt", ack=0, law=law,
                          analog_rate_mask=0x1081, codec_masks=None,
                          lookahead=1, gain_q13=0x4000,
                          shape_q6=(0x10, -16, 0x08, -24), **common)
        cp_fields = dict(kind="cp", ack=0, law=law,
                         analog_rate_mask=0x1081, codec_masks=None,
                         lookahead=1, gain_q13=0x4000,
                         shape_q6=(0x10, -16, 0x08, -24), **common)
        phase4 = V90AnaloguePhase4Controller(
            cpt_fields=cpt_fields, cp_fields=cp_fields, law=law,
            constellation=4, baud=control.BAUD, carrier=control.CARRIER)
        caller_decoded_frames = []
        original_decode = phase4._receiver.decode_codewords

        def capture_caller_frames(codewords):
            bits = original_decode(codewords)
            if bits:
                caller_decoded_frames.append((phase4.state, tuple(bits)))
            return bits

        phase4._receiver.decode_codewords = capture_caller_frames
        cp_profile = phase4.cp_profile
        scr_symbols = int(round(0.75 * control.BAUD))
        upstream_profile = select_analog_upstream_profile(
            cp_profile, answerer_mp_profile(control.BAUD),
            baud=control.BAUD, carrier=control.CARRIER)
        upstream_tx = v34_data.DataTx(
            upstream_profile.rate, int(round(upstream_profile.baud)),
            expanded=upstream_profile.shaping,
            precoder=upstream_profile.precoder,
            nonlinear=upstream_profile.nonlinear,
            trellis=upstream_profile.trellis)

        prefix_symbols = np.concatenate((
            pre, trn, ja,
            np.zeros(int(round(control.BAUD))), s_signal,
            np.zeros(int(round(3.72 * control.BAUD))), s_signal,
            np.zeros(192)))

        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law=law,
            uinfo=67, rate_mask=0xFFFF, enable_lapm=True)
        # Tap only the actual Phase 4 G.711 output. Phase3Session may return a
        # packet containing the tail of Phase 3 and the start of Phase 4; the
        # caller must receive the exact Phase 4 suffix on the same tick.
        phase4_wire = bytearray()
        start_phase4 = session._start_phase4
        phase4_read_tapped = [False]

        def tracked_start_phase4():
            start_phase4()
            if phase4_read_tapped[0]:
                return
            actual_read = session.phase4.read_g711

            def tracked_phase4_read(*args, **kwargs):
                payload = actual_read(*args, **kwargs)
                phase4_wire.extend(payload)
                return payload

            session.phase4.read_g711 = tracked_phase4_read
            phase4_read_tapped[0] = True

        session._start_phase4 = tracked_start_phase4
        server_prompt = bytes((index * 53 + 11) & 0xFF for index in range(192))
        caller_request = bytes((index * 29 + 7) & 0xFF for index in range(384))
        session.feed_bytes(server_prompt)

        qam = v34_tx.StreamQam(control.BAUD, control.CARRIER, 0.1,
                               beta=0.15, span=8, power=2.0)
        qam.push(prefix_symbols)
        qam.push(qam_symbols(phase4.start(), 4))
        qam.push(qam_symbols(
            phase4._encoder.encode_bits(phase4.cpt_bits * 2), 4))
        qam.push(qam_symbols(
            phase4._encoder.encode_bits([1] * (2 * scr_symbols)), 4))

        def queue_repeated_response(decisions, frame_bits, repeats):
            qam.push(qam_symbols(decisions, 4))
            for _ in range(repeats - 1):
                qam.push(qam_symbols(
                    phase4._encoder.encode_bits(frame_bits), 4))

        def fill_upstream(stream):
            required = stream.chunk + stream.span
            while len(stream.queue) < required:
                if (cp_repetitions_queued
                        and phase4.state == phase4.WAIT_MP_PRIME):
                    # Repeat CP only as media is consumed. When MP′ arrives,
                    # the queued tail is at most one CP frame instead of a
                    # preloaded burst of stale responses.
                    stream.push(qam_symbols(
                        phase4._encoder.encode_bits(phase4.cp_bits), 4))
                    continue
                if (cp_prime_tail_queued
                        and phase4.state == phase4.WAIT_ED):
                    # CP′ repeats until the answerer's Ed arrives. This is
                    # essential over a delayed RTP/PSTN return path.
                    stream.push(qam_symbols(
                        phase4._encoder.encode_bits(phase4.cp_prime_bits), 4))
                    continue
                if phase4.lapm is None or not e_and_b1_queued:
                    # Hold the caller carrier with the all-one SCR pattern
                    # until MP/MP′ have caused the control response and B1.
                    stream.push(qam_symbols(
                        phase4._encoder.encode_bits([1] * 64), 4))
                    continue
                raw = phase4.lapm.read_tx_bits(
                    upstream_tx.bits_per_data_frame)
                if len(raw) != upstream_tx.bits_per_data_frame:
                    raise AssertionError("caller LAPM returned a partial V.34 frame")
                transmitter = phase4.upstream_transmitter
                if transmitter is None:
                    raise AssertionError("caller upstream profile is not negotiated")
                stream.push(transmitter.encode_bits(raw))

        qam.fill = fill_upstream
        caller_received = bytearray()
        answerer_received = bytearray()
        caller_request_queued = False
        cp_repetitions_queued = False
        cp_prime_tail_queued = False
        e_and_b1_queued = False
        # StreamQam starts its first symbol center at sample zero; the block
        # transmitter's reference waveform reserves span symbols before that
        # center. Add the same rounded media delay before processing 20 ms
        # slices, keeping the existing sample clock continuous.
        pcm_queue = np.zeros(int(round(
            8 * 8000 / control.BAUD)), dtype=np.float64)

        # One 160-sample tick advances both media directions at 8 kHz. The
        # caller creates the next V.34 interval from its current LAPM state;
        # the answerer returns the actual Phase 4 codewords for that tick.
        for tick in range(1800):
            while len(pcm_queue) < 160:
                pcm_queue = np.concatenate((pcm_queue, qam.read(160)))
            pcm, pcm_queue = pcm_queue[:160], pcm_queue[160:]
            pcm = v34_g711.roundtrip(pcm * 32768.0, law) / 32768.0
            session.feed_pcm(pcm, sample_at=tick * 160)
            answerer_received.extend(session.read_received_data())

            if session.transmit_ready:
                down_chunk = session.read_g711(160, law)
                new_phase4_octets = bytes(phase4_wire)
                phase4_wire.clear()
                # Feed from the very first Phase 4 R-i. The caller controller
                # now recognizes and drains repeated R-i through the actual
                # TRN2d boundary, including the initial 192T wait.
                caller_phase4_input = new_phase4_octets
                # The production V.90 media engine fills pre-data wait states
                # with exact-law silence to preserve every RTP sample tick.
                if len(down_chunk) < 160:
                    silence = 0xFF if law == "pcmu" else 0xD5
                    down_chunk += bytes((silence,)) * (160 - len(down_chunk))
                self.assertEqual(
                    len(down_chunk), 160,
                    f"short RTP tick={tick}, phase3={session.transmitter.state}, "
                    f"phase4={None if session.phase4 is None else session.phase4.state}")
                decisions = (phase4.feed_downstream(caller_phase4_input)
                             if caller_phase4_input else [])
                if decisions:
                    if phase4.state == phase4.WAIT_MP_PRIME:
                        self.assertFalse(cp_repetitions_queued)
                        queue_repeated_response(decisions, phase4.cp_bits, 1)
                        cp_repetitions_queued = True
                    elif phase4.state == phase4.WAIT_ED:
                        self.assertFalse(cp_prime_tail_queued)
                        # MP′ starts CP′ repetition; E/B1 wait for Ed.
                        qam.push(qam_symbols(decisions, 4))
                        cp_prime_tail_queued = True
                    elif (phase4.state in (phase4.WAIT_B1D, phase4.DATA)
                          and ("e-sent", 20) in phase4.events
                          and not e_and_b1_queued):
                        qam.push(qam_symbols(decisions, 4))
                        self.assertIsNotNone(phase4.upstream_transmitter)
                        qam.push(phase4.encode_upstream_b1())
                        e_and_b1_queued = True
                caller_received.extend(phase4.take_received())

            if (not caller_request_queued
                    and bytes(caller_received) == server_prompt):
                phase4.feed_bytes(caller_request)
                caller_request_queued = True
            if (caller_request_queued
                    and bytes(answerer_received) == caller_request):
                break

        self.assertTrue(session.transmit_ready,
                        f"phase3 events={session.events[-12:]}, "
                        f"upstream state={session.upstream.state}/"
                        f"{session.upstream.failure_reason}, "
                        f"phase4 profiles={session.phase4_profiles[-4:]}, "
                        f"phase4 error={session.phase4_error}, "
                        f"qam underruns={getattr(qam, 'underrun_symbols', 0)}")
        phase4_failures = [event[1] for event in phase4.events
                           if event[0] == "phase4-failed"]
        self.assertEqual(phase4.state, phase4.DATA,
                         f"caller failed={phase4_failures}, "
                         f"caller events={[event[0] for event in phase4.events]}, "
                         f"answerer state="
                         f"{None if session.phase4 is None else session.phase4.state}, "
                         f"last decoded frames={caller_decoded_frames[-6:]}, "
                         f"trn2d={None if phase4._trn2d_start is None else phase4._trn2d_start.hex()}, "
                         f"last caller G.711={caller_phase4_input[:48].hex()}")
        upstream_state = (None if session.upstream_data is None else
                          f"{session.upstream_data.state}/"
                          f"{session.upstream_data.failure_reason}")
        b1_diagnostics = (None if session.upstream_data is None else
                          session.upstream_data.receiver.diagnostics)
        self.assertTrue(caller_request_queued,
                        f"caller did not decode server prompt: "
                        f"received={len(caller_received)} bytes, "
                        f"prefix={bytes(caller_received[:32])!r}, "
                        f"phase4 events={[event[0] for event in phase4.events]}, "
                        f"caller LAPM={phase4.lapm.v42_mode if phase4.lapm else None}, "
                        f"server LAPM={session.lapm.v42_mode if session.lapm else None}, "
                        f"upstream data={upstream_state}, "
                        f"B1 diagnostics={b1_diagnostics}")
        self.assertEqual((phase4.lapm.v42_mode, session.lapm.v42_mode),
                         ("lapm", "lapm"))
        self.assertEqual(bytes(caller_received), server_prompt)
        self.assertEqual(bytes(answerer_received), caller_request)
        self.assertEqual(session.upstream_data.state,
                         session.upstream_data.DATA)
        self.assertTrue(session.upstream_data.receiver.diagnostics["qualified"])
        self.assertEqual(getattr(qam, "underrun_symbols", 0), 0)

    def test_e_is_descrambled_and_gated_by_cp_and_transmitted_mp_prime(self):
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcma",
            uinfo=67, rate_mask=0xFFFF)
        phase4 = V90DigitalAnswererPhase4Session(
            uinfo=67, law="pcma", constellation=4,
            mp=session.phase4_mp_profile)
        phase4.start()
        session.phase4 = phase4
        cpt = parse_cp(_frame(
            kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=1, ack=0,
            spectral_sign_bits=0))
        cp = parse_cp(_frame(
            kind="cp", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=1, ack=0,
            spectral_sign_bits=0))
        self.assertTrue(phase4.receive_profile(cpt))
        phase4.transmitter.mp_frames_sent = 1
        self.assertTrue(phase4.receive_profile(cp))
        phase4.transmitter.next_mp()  # Complete an MP′ transmission.
        self.assertEqual(phase4.transmitter.mp_prime_frames_sent, 1)
        cp_end = 1000
        session._cp_end_bit = cp_end

        self.assertFalse(session._detect_phase4_e([1] * 20, cp_end - 1))
        self.assertTrue(session._detect_phase4_e([1], cp_end + 19))
        self.assertEqual(phase4.state, phase4.transmitter.DATA_READY)
        self.assertIn(("phase4-e-detected", cp_end), session.events)

    def test_a_second_run_of_ones_after_e_is_not_another_e(self):
        # Regression: line idle (a run of 20 ones) during data once re-triggered
        # the E handler and attempted to restart B1 after the data transition.
        from unittest import mock
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcma",
            uinfo=67, rate_mask=0xFFFF)
        phase4 = V90DigitalAnswererPhase4Session(
            uinfo=67, law="pcma", constellation=4, mp=session.phase4_mp_profile)
        phase4.start()
        session.phase4 = phase4
        cpt = parse_cp(_frame(kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                              tx_masks=(tuple(range(128)),) * 3, law=1, ack=0,
                              spectral_sign_bits=0))
        cp = parse_cp(_frame(kind="cp", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                             tx_masks=(tuple(range(128)),) * 3, law=1, ack=0,
                             spectral_sign_bits=0))
        phase4.receive_profile(cpt)
        phase4.transmitter.mp_frames_sent = 1
        phase4.receive_profile(cp)
        phase4.transmitter.next_mp()
        session._cp_end_bit = 1000
        session._phase4_symbol_origin = 0
        session.upstream_data = mock.Mock()
        session.upstream = mock.Mock()
        self.assertFalse(session._detect_phase4_e([1] * 20, 999))
        self.assertTrue(session._detect_phase4_e([1], 1019))
        self.assertEqual(session.upstream_data.set_equalizer_snapshot.call_count, 0)
        self.assertEqual(session.upstream_data.start_b1.call_count, 1)
        # Idle ones later in the call: must neither detect nor touch B1 again.
        self.assertFalse(session._detect_phase4_e([1] * 40, 5000))
        self.assertFalse(session._detect_phase4_e([1] * 20, 6000))
        self.assertEqual(session.upstream_data.set_equalizer_snapshot.call_count, 0)
        self.assertEqual(session.upstream_data.start_b1.call_count, 1)
        self.assertEqual(sum(1 for e in session.events if e[0] == "phase4-e-detected"), 1)

    def test_optional_lapm_uses_cp_downstream_rate_and_queues_dte_data(self):
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcmu",
            uinfo=67, rate_mask=0xFFFF, enable_lapm=True)
        session.feed_bytes(b"queued before CP")
        phase4 = V90DigitalAnswererPhase4Session(
            uinfo=67, law="pcmu", constellation=4,
            mp=session.phase4_mp_profile)
        phase4.start()
        session.phase4 = phase4
        cpt = parse_cp(_frame(
            kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=0, ack=0,
            spectral_sign_bits=0))
        cp = parse_cp(_frame(
            kind="cp", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=0, ack=0,
            spectral_sign_bits=0))
        self.assertTrue(phase4.receive_profile(cpt))
        phase4.read_g711(50000)  # Finish TRN2d and transmit MP before CP.
        self.assertTrue(session.receive_phase4_profile(cp))
        self.assertIsNotNone(session.lapm)
        self.assertEqual(session.lapm._v42_params[0], int(cp["rate"]))
        self.assertEqual(session.analog_upstream_rate, 33600)
        self.assertEqual(session.lapm._v42_hold, b"queued before CP")
        self.assertIn("lapm-started", [name for name, _ in session.events])

    def test_repeated_cp_does_not_rebuild_upstream_receiver(self):
        session = V90DigitalAnswererPhase3Session(
            baud=control.BAUD, carrier=control.CARRIER, law="pcmu",
            uinfo=67, rate_mask=0xFFFF)
        phase4 = V90DigitalAnswererPhase4Session(
            uinfo=67, law="pcmu", constellation=4,
            mp=session.phase4_mp_profile)
        phase4.start()
        session.phase4 = phase4
        cpt = parse_cp(_frame(
            kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=0, ack=0,
            spectral_sign_bits=0))
        cp = parse_cp(_frame(
            kind="cp", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=0, ack=0,
            spectral_sign_bits=0))
        self.assertTrue(phase4.receive_profile(cpt))
        phase4.transmitter.mp_frames_sent = 1

        with patch("v90_digital_phase3_session.V90UpstreamDataReceiver") as build:
            self.assertTrue(session.receive_phase4_profile(cp))
            self.assertEqual(build.call_count, 1)
            self.assertFalse(session.receive_phase4_profile(dict(cp)))
            self.assertEqual(build.call_count, 1)


if __name__ == "__main__":
    unittest.main()
