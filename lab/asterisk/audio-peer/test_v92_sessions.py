import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import v34_g711
import v92_analogue_session as asn
import v92_data_mode as dm
import v92_digital_session as dsn
import v92_fast_parameter_exchange as fpe
import v92_phase3_exchange as p3x
import v92_pcm_upstream as pu
import v92_cp
import v90_digital_phase4 as dp4

BLOCK = 160
# A mild dispersive upstream line. Heavier ISI (a 12-tap 0.55 first echo) breaks Phase 3, which decides on signs with no equaliser.
LINE = [1.0, 0.15, -0.06, 0.03, -0.02, 0.01]
PHASE3_ISI_LINE = [1.0, 0.6, 0.3, 0.2] + [0.0] * 8


class FactoryConfigurationTests(unittest.TestCase):
    def test_empty_precoder_policy_defaults_to_omit_and_can_be_overridden(self):
        make = dsn.factory(law="pcmu")
        session = make(info1a={}, law="pcmu", handoff_at=0, media_origin=0.0)
        self.assertEqual(session._empty_precoder_policy, "omit")
        with patch.dict(os.environ, {"EL_MODEM_V92_CPD_EMPTY_PRECODER_POLICY": "include"}):
            make = dsn.factory(law="pcmu")
            session = make(info1a={}, law="pcmu", handoff_at=0, media_origin=0.0)
        self.assertEqual(session._empty_precoder_policy, "include")


def pair(law="pcmu", delay=240, seconds=40, up=b"", down=b"", channel=None, equalise=False,
         phase3_equalise=False, enable_rate_renegotiation=False,
         enable_rate_renegotiation_response=False, enable_fpe_response=False,
         dil_ucodes=(), lapm_detect=True, analogue_lu=1000.0, first_block_offset=0,
         phase3_cpt_drn=None):
    a = asn.V92AnalogueSession(law=law, lu=analogue_lu,
                               enable_rate_renegotiation=enable_rate_renegotiation,
                               dil_ucodes=tuple(dil_ucodes),
                               lapm_detect=lapm_detect)
    if phase3_cpt_drn is not None:
        cpt_bits = v92_cp.build_cpu(
            kind=v92_cp.CPT, drn=phase3_cpt_drn,
            constellations=asn._masks(), interval_sets=(0, 1, 2, 3, 4, 5),
            alaw=(law == "pcma"))
        a._p3._cpt_bits = cpt_bits
        a.profile = dm.downstream_profile_from_cpu(v92_cp.parse_cpu(cpt_bits))
    d = dsn.V92DigitalSession(law=law, equalise=equalise, phase3_equalise=phase3_equalise,
                              enable_rate_renegotiation_response=enable_rate_renegotiation_response,
                              enable_fast_parameter_exchange_response=enable_fpe_response,
                              lapm_detect=lapm_detect)
    channel_hist = np.zeros(0 if channel is None else len(channel) - 1)
    a.feed_bytes(up)
    d.feed_bytes(down)
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    to_d = [0.0] * delay
    to_a = [float(v34_g711.decode(np.array([silence], dtype=np.uint8), law)[0])] * delay
    got_up = bytearray()
    got_down = bytearray()
    handoff_blocks = []
    elapsed = 0
    first_count = first_block_offset if first_block_offset else BLOCK
    while elapsed < seconds * 8000:
        count = min(first_count if elapsed == 0 else BLOCK, seconds * 8000 - elapsed)
        tx_start = a._p3._pos if a._p3 is not None else a._tx_after
        event_start = len(a.events)
        x = a.read(count)
        for name, value in a.events[event_start:]:
            if name == "phase4-tx-start":
                tail = int(value["sample"]) - tx_start
                handoff_blocks.append((value, np.asarray(x[max(0, tail):]).copy()))
        if channel is not None:
            joined = np.concatenate((channel_hist, x))
            x = np.convolve(joined, channel, mode="valid")
            channel_hist = joined[len(x):]
        to_d.extend(v34_g711.roundtrip(np.rint(x), law).astype(float))
        wire = d.read_g711(count, law)
        to_a.extend(v34_g711.decode(np.frombuffer(wire, dtype=np.uint8), law).astype(float))
        d.feed_pcm(np.array(to_d[:count]))
        del to_d[:count]
        a.feed(np.array(to_a[:count]))
        del to_a[:count]
        got_up.extend(d.read_received_data())
        got_down.extend(a.take_received())
        if len(got_up) >= len(up) and len(got_down) >= len(down) and (up or down):
            break
        elapsed += count
        if first_count != BLOCK and elapsed >= first_count:
            first_count = BLOCK
    a._test_phase4_handoffs = handoff_blocks
    return a, d, bytes(got_up), bytes(got_down)


class DownstreamMarkFillTests(unittest.TestCase):
    """Astound fills a 56000 downstream with 42-bit runs of marks, inside HDLC frames too (2026-11-20, batch i)."""

    FLAG = [0, 1, 1, 1, 1, 1, 1, 0]

    def _frames(self, bits):
        import v42_lapm
        hdlc = v42_lapm._Hdlc()
        return [(f, ok) for bit in bits for f, ok in hdlc.push(bit)]

    def test_marks_inserted_in_a_frame_are_removed_and_the_frame_decodes(self):
        import v42_lapm
        frame = bytes((0x01, 0x00, 0x00)) + b"\r" + b"\n" * 30
        body = v42_lapm._stuffed(frame)
        for at in (0, 8, 80):                    # after the opening flag, after the address, mid-information
            line = self.FLAG * 3 + body[:at] + [1] * 42 + body[at:] + self.FLAG * 3
            self.assertEqual(self._frames(line), [])
            fill = asn.DownstreamMarkFill()
            self.assertEqual(self._frames(fill.push(line, 42)), [(frame, True)])
            self.assertEqual(fill.removed, 1)

    def test_idle_marks_pass_through_unchanged_and_are_not_held(self):
        fill = asn.DownstreamMarkFill()
        self.assertEqual(fill.push([1] * 100, 42), [1] * 100)
        self.assertEqual(fill.push([1] * 5 + [0], 42), [1] * 5 + [0])
        self.assertEqual(fill.removed, 0)

    def test_a_run_split_across_pushes_is_removed_and_short_runs_are_data(self):
        fill = asn.DownstreamMarkFill()
        out = fill.push([0] + [1] * 20, 42) + fill.push([1] * 22 + [0, 1, 1, 0], 42)
        self.assertEqual(out, [0, 0, 1, 1, 0])
        self.assertEqual(fill.push([1] * 41 + [0], 42), [1] * 41 + [0])
        self.assertEqual(fill.push([1] * 53 + [0], 42), [1] * 53 + [0])
        self.assertEqual(fill.removed, 1)

    def test_the_session_option_is_off_by_default(self):
        self.assertIsNone(asn.V92AnalogueSession(law="pcmu").mark_fill)
        self.assertIsInstance(asn.V92AnalogueSession(law="pcmu", downstream_mark_fill=True).mark_fill,
                              asn.DownstreamMarkFill)


class FilterLimitTests(unittest.TestCase):
    def test_info1a_limits_decode_table18_bits_14_to_17(self):
        # v90_reserved_12_17 holds INFO1a bits 12..17, bit 12 as its LSB.
        self.assertEqual(dsn.filter_limits_from_info1a({"v90_reserved_12_17": 0}), (128, 192))
        self.assertEqual(dsn.filter_limits_from_info1a({"v90_reserved_12_17": 0b111100}), (320, 384))
        self.assertEqual(dsn.filter_limits_from_info1a({"v90_reserved_12_17": 0b011000}), (192, 320))
        self.assertEqual(dsn.filter_limits_from_info1a({}), (128, 192))
        self.assertEqual(dsn.filter_limits_from_info1a({"v90_reserved_12_17": True}), (128, 192))

    def test_tap_budget_stays_within_lmax_and_ltot(self):
        import v92_pair_session as pair
        for lmax, ltot in ((128, 192), (192, 256), (256, 320), (320, 384), (16, 20)):
            with self.subTest(lmax=lmax, ltot=ltot):
                z2, p1 = pair.filter_tap_budget(lmax, ltot)
                self.assertGreaterEqual(z2, 1)
                self.assertLessEqual(max(z2, p1), lmax)
                self.assertLessEqual(z2 + p1, ltot)
        self.assertEqual(pair.filter_tap_budget(), (128, pair.POSTCURSORS))

    def test_line_design_respects_minimum_info1a_limits(self):
        rng = np.random.default_rng(3)
        h = np.array([0.0, 0.25, 1.0, 0.42, -0.18, 0.07])
        lu = 1000.0
        import v92_phase3_exchange as p3
        bits = p3._trn1u_bits(16320)
        reference = lu * np.where(np.array(bits) == 1, -1.0, 1.0)
        received = np.convolve(reference, h)[:len(reference)] + rng.normal(0, 2.0, len(reference))
        received = v34_g711.roundtrip(np.rint(received), "pcmu").astype(float)
        designed = dsn.design_for_line(received, "pcmu", lu=lu, filter_limits=(128, 192))
        self.assertIsNotNone(designed)
        parsed = v92_cp.parse_cpd(designed[2](False))
        pre = parsed["precoder"]
        lengths = [len(pre.get(k, ())) for k in ("z1", "p1", "z2", "p2")]
        self.assertLessEqual(max(lengths), 128)
        self.assertLessEqual(sum(lengths), 192)
        self.assertGreater(len(pre["z2"]), 1)


class Phase4DiagnosticCoordinateTests(unittest.TestCase):
    def test_trn2u_index_is_also_reported_in_absolute_receive_samples(self):
        session = dsn.V92DigitalSession.__new__(dsn.V92DigitalSession)
        session._p4_fed_from = 91_294
        session._p4 = SimpleNamespace(diagnostics=lambda: {
            "decoder": {"trn2u_at": 12, "acquisition_state": "locked"}})
        diagnostics = session.phase4_diagnostics()
        self.assertEqual(diagnostics["decoder"]["receive_sample_origin"], 91_294)
        self.assertEqual(diagnostics["decoder"]["trn2u_absolute_sample"], 91_306)


class SessionTests(unittest.TestCase):
    def test_phase4_diagnostics_exception_does_not_interrupt_receive(self):
        digital = dsn.V92DigitalSession(law="pcmu")
        received = []

        class RaisingDiagnosticsSide:
            connected = False
            peer_cp = None

            class DataRx:
                qualified = False
                def take_bits(self):
                    return []
            data_rx = DataRx()

            def diagnostics(self):
                raise RuntimeError("diagnostic snapshot failed")

            def receive(self, samples):
                received.extend(np.asarray(samples).tolist())

        digital.stage = "phase4"
        digital._p4 = RaisingDiagnosticsSide()
        frame = np.arange(BLOCK, dtype=float)

        digital.feed_pcm(frame, sample_at=12345)

        self.assertEqual(received, frame.tolist())
        self.assertEqual(digital.stage, "phase4")
        errors = [value for name, value in digital.events if name == "diagnostics-error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["source"], "phase4-diagnostics")
        self.assertEqual(errors[0]["error_type"], "RuntimeError")
        self.assertIn("diagnostic snapshot failed", errors[0]["traceback"])
        # A persistent diagnostics defect is reported once, while subsequent
        # media callbacks continue feeding the decoder.
        digital.feed_pcm(frame + BLOCK, sample_at=12345 + BLOCK)
        self.assertEqual(received, np.concatenate((frame, frame + BLOCK)).tolist())
        self.assertEqual(sum(name == "diagnostics-error" for name, _ in digital.events), 1)

    def test_digital_trn_length_is_opt_in_and_reaches_initial_phase4(self):
        with patch.dict(os.environ, {"EL_MODEM_V92_DIGITAL_TRN_SYMBOLS": "8000"}):
            make = dsn.factory(law="pcmu")
        digital = make(info1a={}, law="pcmu", handoff_at=0, media_origin=0.0)
        self.assertEqual(digital._digital_trn_symbols, 8000)

        digital._p3.cpt = {"kind": "cpt"}
        fake_side = type("FakeSide", (), {})()
        with patch.object(dsn.dm, "downstream_profile_from_cpu", return_value={"rate": 32000}), \
                patch.object(dsn.p4c, "digital_side", return_value=fake_side) as build_side:
            digital._initialize_phase4()
        self.assertEqual(build_side.call_args.kwargs["digital_trn"], 8000)
        self.assertEqual(digital.events[-1], ("phase4-start", {
            "cpt_rate": 32000, "digital_trn_symbols": 8000,
        }))

    def test_digital_trn_length_rejects_values_below_protocol_minimum(self):
        with self.assertRaisesRegex(ValueError, "integer >= 2040"):
            dsn.V92DigitalSession(law="pcmu", digital_trn_symbols=2039)

    def test_phase4_metrics_distinguish_not_started_from_failed_initialization(self):
        digital = dsn.V92DigitalSession(law="pcmu")
        self.assertEqual(digital.phase4.state, "not-started")
        digital._p3.cpt = {"kind": "cpt"}
        with patch.object(dsn.dm, "downstream_profile_from_cpu", return_value={"test": True}), \
                patch.object(dsn.p4c, "digital_side",
                             side_effect=NotImplementedError("unsupported profile")):
            with self.assertRaisesRegex(NotImplementedError, "unsupported profile"):
                digital._begin_phase4()
        self.assertEqual(digital.phase4.state, "failed")
        self.assertEqual(digital.phase4_error, "unsupported profile")
        self.assertIn(("phase4-init-failed", "unsupported profile"), digital.events)

    def test_phase4_tx_follows_rbar_i_tail_in_same_media_block(self):
        digital = dsn.V92DigitalSession(law="pcmu")
        digital._p3.state = "phase4-ready"
        digital._p3._queue = bytearray([0x20, 0x21, 0x22, 0x23])
        digital._p3.rbar_i_start_sample = -20  # this is the final four symbols of a 24T R-bar-i
        digital._p3.sd_start_sample = -122  # 17 full frames before TRN2d starts

        class Phase4Tx:
            def transmit(self, count):
                return bytes([0xA5]) * count

        digital._begin_phase4 = lambda: setattr(digital, "_p4", Phase4Tx())

        output = digital.read_g711(160, "pcmu")

        self.assertEqual(output[:4], bytes([0x20, 0x21, 0x22, 0x23]))
        self.assertEqual(output[4:], bytes([0xA5]) * 156)

    def test_phase3_rbar_i_handoff_fills_same_block_without_gap_or_frame_slip(self):
        for end_offset in (0, 44, 159):
            with self.subTest(end_offset=end_offset):
                digital = dsn.V92DigitalSession(law="pcmu")
                p3 = digital._p3
                p3.state = "cpt-received"
                p3._jprime_sent = True
                p3.ja = {"N": 0}
                p3.cpt = {"kind": "cpt"}
                prefix_count = (BLOCK - 24) if end_offset == 0 else end_offset - 24
                # Put this media block at a later global sample position; Sd
                # is 17 complete six-symbol frames before the R-bar-i start.
                block_start = 1600
                p3.tx_symbols_emitted = block_start
                p3.sd_start_sample = block_start + prefix_count - 6 * 17
                digital._tx_sample_count = block_start
                p3._queue = bytearray([0x22] * prefix_count)

                class Phase4Tx:
                    def transmit(self, count):
                        return bytes([0x55]) * count

                digital._begin_phase4 = lambda: setattr(digital, "_p4", Phase4Tx())
                output = digital.read_g711(BLOCK, "pcmu")
                expected_rbar = dp4.rbar_i_codewords(67, "pcmu", duration_symbols=24)
                start = prefix_count
                end = start + len(expected_rbar)

                self.assertEqual(len(output), BLOCK)
                self.assertEqual(output[:start], bytes([0x22]) * start)
                self.assertEqual(output[start:end], expected_rbar)
                self.assertEqual(output[end:], bytes([0x55]) * (BLOCK - end))
                self.assertNotIn(digital._silence(), output[start:])
                handoff = next(value for name, value in digital.events
                               if name == "phase4-tx-handoff")
                self.assertEqual(handoff["trn2d_start_sample"], handoff["rbar_i_end_sample"])
                self.assertEqual(handoff["rbar_relative_offset"], 24)
                self.assertEqual(handoff["frame_origin_name"], "Sd")
                self.assertEqual(handoff["frame_phase_from_sd"], 0)
                self.assertTrue(handoff["frame_phase_verified"])
                self.assertEqual(handoff["same_media_block_tail"], BLOCK - end)

    def test_phase3_cpt_e1u_window_diagnostics_reach_session_events(self):
        digital = dsn.V92DigitalSession(law="pcmu")
        raw_window = {"at": 91282, "classification": "ambiguous",
                      "bits": "000000110000", "slips": []}
        digital._p3.events.extend([
            ("cpt", 89098),
            ("ri-sent", "cpt"),
            ("e1u-window", raw_window),
        ])

        digital._capture_phase3_events()

        self.assertIn(("phase3-cpt", 89098), digital.events)
        self.assertIn(("phase3-ri-sent", "cpt"), digital.events)
        self.assertIn(("phase3-e1u-window", raw_window), digital.events)

    def test_phase4_sign_seed_handoff_survives_both_dil_modes_and_packet_offsets(self):
        up = bytes((i * 37 + 5) & 0xFF for i in range(16))
        down = bytes((i * 91 + 17) & 0xFF for i in range(16))
        for dil in ((), (37,)):
            for cpt_drn, expected_sign in ((1, 1), (2, 0)):
                for first_block_offset in (1, 79, 159):
                    with self.subTest(dil=len(dil), cpt_drn=cpt_drn,
                                      first_block_offset=first_block_offset):
                        analogue, digital, got_up, got_down = pair(
                            law="pcmu", delay=240, seconds=20, up=up, down=down,
                            dil_ucodes=dil, phase3_cpt_drn=cpt_drn,
                            first_block_offset=first_block_offset)
                        tx_start = next(value for name, value in analogue.events
                                        if name == "phase4-tx-start")
                        rx_seed = next(value for name, value in digital.events
                                       if name == "phase4-e1u-sign")
                        suvu_start = next(value for name, value in analogue.events
                                          if name == "phase4-suvu-start")
                        self.assertEqual(tx_start["initial_sign"], expected_sign)
                        self.assertEqual(rx_seed["sign"], expected_sign)
                        if dil:
                            self.assertFalse(tx_start["rx_attached_at_tx_start"])
                            rx_start = next(value for name, value in analogue.events
                                            if name == "phase4-start")
                            self.assertGreater(rx_start, tx_start["sample"])
                        self.assertEqual(tx_start["offset_mod12"], 0)
                        self.assertEqual(suvu_start["offset_mod12"], 0)
                        self.assertTrue(analogue._test_phase4_handoffs)
                        first_handoff = analogue._test_phase4_handoffs[0][1]
                        self.assertGreater(len(first_handoff), 0)
                        self.assertTrue(np.all(first_handoff != 0.0))
                        self.assertEqual(analogue.stage, "data")
                        self.assertEqual(digital.state, "data")
                        self.assertEqual(got_up, up)
                        self.assertEqual(got_down, down)
                        self.assertIn("suvu", [kind for kind, _ in digital._p4.frames_seen])
                        self.assertIn("cpu", [kind for kind, _ in digital._p4.frames_seen])

    def test_phase4_upstream_lu_matches_trn2u_in_decoder_domain(self):
        up = bytes((i * 37 + 5) & 0xff for i in range(128))
        down = bytes((i * 91 + 17) & 0xff for i in range(128))
        for law in ("pcmu", "pcma"):
            for equalise in (False, True):
                with self.subTest(law=law, equalise=equalise):
                    analogue, digital, got_up, got_down = pair(
                        law=law, up=up, down=down, analogue_lu=988.0,
                        equalise=equalise, phase3_equalise=equalise)
                    self.assertEqual(got_up, up)
                    self.assertEqual(got_down, down)
                    self.assertAlmostEqual(analogue.lu, 988.0)
                    # Calibrate from TRN2u in the actual decoder input
                    # domain, after G.711 and any Phase-4 receive filter.
                    decoder = digital._p4._decoder
                    self.assertFalse(decoder._auto_lu)
                    # The equalised A-law probe is quantised and its first
                    # 1200 active samples vary slightly with the valid E1u
                    # differential seed. Payload qualification below is the
                    # functional check; keep this as a coarse LU sanity bound.
                    self.assertAlmostEqual(decoder.lu, 988.0, delta=40.0)

                    # The production analogue side's mapper now emits the
                    # Table 28 point set at this same LU; compare quantized
                    # levels instead of ideal table values.
                    symbols = analogue._p4._encode(
                        [0, 0, 0, 1, 1, 0, 1, 1] * 20)
                    quantized = v34_g711.roundtrip(np.rint(symbols), law)
                    actual_levels = sorted(set(np.abs(np.rint(quantized)).astype(int)))
                    expected_levels = ([428, 1308] if law == "pcmu"
                                       else [440, 1312])
                    self.assertEqual(actual_levels, expected_levels)

    def test_direct_lapm_startup_carries_exact_payload_in_both_laws(self):
        up = bytes((i * 37 + 5) & 0xff for i in range(128))
        down = bytes((i * 91 + 17) & 0xff for i in range(128))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                analogue, digital, got_up, got_down = pair(
                    law=law, up=up, down=down, lapm_detect=False)
                self.assertEqual(got_up, up)
                self.assertEqual(got_down, down)
                self.assertTrue(analogue.connected)
                self.assertEqual(digital.state, "data")
                self.assertTrue(analogue.lapm._v42_direct)
                self.assertTrue(digital.lapm._v42_direct)
                self.assertNotIn("ADP received", analogue.lapm._v42.events)
                self.assertNotIn("ODP received", digital.lapm._v42.events)

    def test_equalised_fpe_retains_trained_model_and_lapm(self):
        for law in ("pcmu", "pcma"):
            for initiator in ("digital", "analogue"):
                with self.subTest(law=law, initiator=initiator):
                    self._equalised_fpe_case(law, initiator)

    def _equalised_fpe_case(self, law, initiator):
        before_up = bytes((i * 37 + 5) & 0xFF for i in range(96))
        before_down = bytes((i * 91 + 17) & 0xFF for i in range(96))
        after_up = bytes((i * 13 + 101) & 0xFF for i in range(96))
        after_down = bytes((i * 53 + 7) & 0xFF for i in range(96))
        analogue = asn.V92AnalogueSession(law=law, enable_fast_parameter_exchange=True)
        digital = dsn.V92DigitalSession(
            law=law, equalise=True, phase3_equalise=True,
            enable_fast_parameter_exchange=True,
            enable_fast_parameter_exchange_response=True)
        analogue.feed_bytes(before_up)
        digital.feed_bytes(before_down)
        silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
        to_digital = [0.0] * 240
        to_analogue = [float(v34_g711.decode(np.array([silence], dtype=np.uint8), law)[0])] * 240
        taps = np.asarray([1.0, 0.08, -0.03, 0.015])
        history = np.zeros(len(taps) - 1)
        got_up, got_down = bytearray(), bytearray()

        def exchange():
            nonlocal history
            tx = analogue.read(BLOCK)
            joined = np.concatenate((history, tx))
            impaired = np.convolve(joined, taps, mode="valid")
            history = joined[len(impaired):]
            to_digital.extend(v34_g711.roundtrip(np.rint(impaired), law).astype(float))
            wire = digital.read_g711(BLOCK, law)
            to_analogue.extend(v34_g711.decode(np.frombuffer(wire, dtype=np.uint8), law).astype(float))
            digital.feed_pcm(np.asarray(to_digital[:BLOCK])); del to_digital[:BLOCK]
            analogue.feed(np.asarray(to_analogue[:BLOCK])); del to_analogue[:BLOCK]
            got_up.extend(digital.read_received_data())
            got_down.extend(analogue.take_received())

        for _ in range(1600):
            exchange()
            if (got_up == before_up and got_down == before_down and analogue.connected
                    and digital.state == "data" and analogue.lapm and digital.lapm):
                break
        self.assertEqual(bytes(got_up), before_up)
        self.assertEqual(bytes(got_down), before_down)
        self.assertIsNotNone(digital._p4.rx_filter)
        digital_filter = digital._p4.rx_filter
        digital_lapm, analogue_lapm = digital.lapm, analogue.lapm
        analogue.feed_bytes(after_up)
        digital.feed_bytes(after_down)
        if initiator == "digital":
            digital.request_fast_parameter_exchange()
        else:
            analogue.request_fast_parameter_exchange()

        for _ in range(500):
            exchange()
            if (bytes(got_up) == before_up + after_up
                    and bytes(got_down) == before_down + after_down
                    and analogue.connected and digital.state == "data"
                    and analogue._reneg is None and digital._reneg is None):
                break

        dnew_state = None
        if digital._reneg:
            dnew = digital._reneg["new_side"]
            dnew_state = (dnew.phase, dnew.frames_seen[-8:], dnew.data_rx.b1_errors)
        self.assertEqual(bytes(got_up), before_up + after_up,
                         f"digital={digital.state}/{digital.events[-8:]} "
                         f"dnew={dnew_state} "
                         f"analog={analogue.stage}/{analogue.events[-8:]}")
        self.assertEqual(bytes(got_down), before_down + after_down,
                         f"digital={digital.state}/{digital.events[-8:]} "
                         f"analog={analogue.stage}/{analogue.events[-8:]}")
        self.assertIs(analogue.lapm, analogue_lapm)
        self.assertIs(digital.lapm, digital_lapm)
        self.assertIs(digital._p4.rx_filter, digital_filter)
        self.assertTrue(analogue._p4.data_rx.qualified)
        self.assertTrue(digital._p4.data_rx.qualified)
        self.assertEqual(analogue._p4.data_rx.b1_errors, 0)
        self.assertEqual(digital._p4.data_rx.b1_errors, 0)
        self.assertIn("v92-fpe-rm-detected", [name for name, _ in digital.events])
        self.assertIn("v92-fpe-complete", [name for name, _ in analogue.events])
        self.assertIn("v92-fpe-complete", [name for name, _ in digital.events])

        # A second, reverse-initiated exchange exercises listener reset and
        # decoder/filter handoff on an already reconfigured live session.
        reverse_up = bytes((i * 29 + 3) & 0xFF for i in range(64))
        reverse_down = bytes((i * 61 + 11) & 0xFF for i in range(64))
        analogue.feed_bytes(reverse_up)
        digital.feed_bytes(reverse_down)
        if initiator == "digital":
            analogue.request_fast_parameter_exchange()
        else:
            digital.request_fast_parameter_exchange()
        for _ in range(500):
            exchange()
            if (bytes(got_up) == before_up + after_up + reverse_up
                    and bytes(got_down) == before_down + after_down + reverse_down
                    and analogue.connected and digital.state == "data"
                    and analogue._reneg is None and digital._reneg is None):
                break
        self.assertEqual(
            bytes(got_up), before_up + after_up + reverse_up,
            f"reverse FPE up failed: digital={digital.state} "
            f"events={digital.events[-12:]} "
            f"reneg={None if digital._reneg is None else digital._reneg.get('phase')} "
            f"probe={len(digital._peer_reneg_probe)} "
            f"analog={analogue.stage}/{analogue.events[-12:]}")
        self.assertEqual(bytes(got_down), before_down + after_down + reverse_down)
        self.assertTrue(analogue._p4.data_rx.qualified)
        self.assertTrue(digital._p4.data_rx.qualified)
        self.assertEqual(analogue._p4.data_rx.b1_errors, 0)
        self.assertEqual(digital._p4.data_rx.b1_errors, 0)
        self.assertGreaterEqual(
            sum(name == "v92-fpe-complete" for name, _ in analogue.events), 2)
        self.assertGreaterEqual(
            sum(name == "v92-fpe-complete" for name, _ in digital.events), 2)


    def test_live_digital_initiated_fpe_preserves_lapm_and_exact_payload(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 7, 240, 247):
                with self.subTest(law=law, delay=delay):
                    self._check_live_fpe(law, delay, "digital")

    def test_live_analogue_initiated_fpe_preserves_lapm_and_exact_payload(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 7, 240, 247):
                with self.subTest(law=law, delay=delay):
                    self._check_live_fpe(law, delay, "analogue")

    def test_validated_receiver_fpe_carries_the_trn2u_fit_through_two_exchanges(self):
        # FPE sends no TRN2u: the old mode's fitted FIR decodes RM, the control and FB1u, and the new mode's receiver
        # starts from it. The precursor channel makes the FIR's delay nonzero (31 / 9 samples).
        stack = {"b1u_rx_mode": "trn2u-fit-viterbi", "cpd_constellation_mode": "adaptive"}
        for law in ("pcmu", "pcma"):
            for initiator in ("digital", "analogue"):
                with self.subTest(law=law, initiator=initiator):
                    digital = self._check_live_fpe(law, 247, initiator, stack=stack, rounds=2,
                                                   channel=[0.0, 0.25, 1.0, 0.42, -0.18, 0.07])
                    data_rx = digital._p4.data_rx
                    self.assertEqual(data_rx.fb1_errors, 0)
                    self.assertEqual(data_rx.decoder_mode, "viterbi32")
                    self.assertEqual(data_rx.rx_filter_diagnostic["state"], "active")
                    self.assertGreater(data_rx.rx_filter_diagnostic["delay_samples"], 0)
                    self.assertGreater(data_rx.rx_filter_diagnostic["nlms"]["updates"], 0)
                    self.assertTrue(digital._p4.diagnostic_metadata["cpd_constellation"]["final"])

    def _check_live_fpe(self, law, delay, initiator, *, stack=None, channel=None, rounds=1):
        payloads = [(bytes((i * 37 + 5) & 0xFF for i in range(64)), bytes((i * 91 + 17) & 0xFF for i in range(64))),
                    (bytes((i * 13 + 101) & 0xFF for i in range(80)), bytes((i * 53 + 7) & 0xFF for i in range(80))),
                    (bytes((i * 7 + 3) & 0xFF for i in range(96)), bytes((i * 11 + 1) & 0xFF for i in range(96)))]
        analogue = asn.V92AnalogueSession(law=law, enable_fast_parameter_exchange=True)
        digital = dsn.V92DigitalSession(
            law=law, **({"enable_fast_parameter_exchange": True} if initiator == "digital"
                        else {"enable_fast_parameter_exchange_response": True}), **(stack or {}))
        silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
        to_digital = [0.0] * delay
        to_analogue = [float(v34_g711.decode(
            np.array([silence], dtype=np.uint8), law)[0])] * delay
        channel_history = np.zeros(0 if channel is None else len(channel) - 1)
        got_up, got_down = bytearray(), bytearray()

        def exchange():
            nonlocal channel_history
            a_out = analogue.read(BLOCK)
            d_out = digital.read_g711(BLOCK, law)
            if channel is not None:
                joined = np.concatenate((channel_history, np.asarray(a_out, dtype=float)))
                a_out = np.convolve(joined, channel, mode="valid")
                channel_history = joined[len(a_out):]
            to_digital.extend(v34_g711.roundtrip(np.rint(a_out), law).astype(float))
            to_analogue.extend(v34_g711.decode(
                np.frombuffer(d_out, dtype=np.uint8), law).astype(float))
            digital.feed_pcm(np.asarray(to_digital[:BLOCK]))
            del to_digital[:BLOCK]
            analogue.feed(np.asarray(to_analogue[:BLOCK]))
            del to_analogue[:BLOCK]
            got_up.extend(digital.read_received_data())
            got_down.extend(analogue.take_received())

        want_up, want_down = b"", b""
        lapms = None
        for round_ in range(rounds + 1):
            up, down = payloads[round_]
            want_up, want_down = want_up + up, want_down + down
            analogue.feed_bytes(up)
            digital.feed_bytes(down)
            if round_:
                (digital if initiator == "digital" else analogue).request_fast_parameter_exchange()
            for _ in range(500):
                exchange()
                if (bytes(got_up) == want_up and bytes(got_down) == want_down
                        and analogue.connected and digital.state == "data" and analogue.lapm and digital.lapm
                        and analogue._reneg is None and digital._reneg is None):
                    break
            self.assertEqual(bytes(got_up), want_up)
            self.assertEqual(bytes(got_down), want_down)
            if lapms is None:
                lapms = analogue.lapm, digital.lapm
            self.assertIs(analogue.lapm, lapms[0])
            self.assertIs(digital.lapm, lapms[1])
        self.assertTrue(analogue._p4.data_rx.qualified)
        self.assertTrue(digital._p4.data_rx.qualified)
        self.assertEqual(analogue._p4.data_rx.b1_errors, 0)
        self.assertEqual(digital._p4.data_rx.b1_errors, 0)
        digital_events = [name for name, _ in digital.events]
        analogue_events = [name for name, _ in analogue.events]
        self.assertIn("v92-fpe-rm-detected", digital_events)
        if initiator == "digital":
            self.assertIn("v92-fpe-rf-start", digital_events)
            self.assertIn("v92-fpe-rf-detected", analogue_events)
        else:
            self.assertIn("v92-fpe-rf-detected", analogue_events)
        self.assertEqual(digital_events.count("v92-fpe-complete"), rounds)
        self.assertIn("v92-fpe-complete", analogue_events)
        return digital

    def test_live_digital_session_detects_rm_and_transmits_rf_response(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                up = bytes((i * 37 + 5) & 0xFF for i in range(64))
                down = bytes((i * 91 + 17) & 0xFF for i in range(64))
                analogue, digital, got_up, got_down = pair(
                    law=law, delay=240, seconds=20, up=up, down=down, enable_fpe_response=True)
                self.assertEqual(got_up, up)
                self.assertEqual(got_down, down)
                self.assertTrue(analogue.connected and digital.state == "data")

                profile = digital._p4.data_rx.rx.profile
                tx = pu.PcmUpstreamTransmitter(profile)
                rm, rm_prime = pu.rm_intervals(profile.moduli), pu.rm_intervals(profile.moduli, prime=True)
                waveform = np.concatenate(
                    [np.asarray(tx.encode_intervals(rm), dtype=float) for _ in range(fpe.R_FRAMES)]
                    + [np.asarray(tx.encode_intervals(rm_prime), dtype=float)
                       for _ in range(fpe.RBAR_FRAMES)])
                pcm = v34_g711.roundtrip(np.rint(waveform), law).astype(float)
                # A peer's RTP clock need not share our local frame phase. If
                # the listener searches all offsets, start RM at the next sample.
                phase = digital._fpe_detector.sample_phase
                lead = (0 if phase is None else
                        (phase - digital._peer_reneg_total) % pu.INTERVALS)
                stream = np.concatenate((np.zeros(lead), pcm))
                for at in range(0, len(stream), BLOCK):
                    digital.feed_pcm(stream[at:at + BLOCK])

                names = [name for name, _ in digital.events]
                self.assertIn("v92-fpe-rm-detected", names)
                self.assertIn("v92-fpe-rf-response-start", names)
                self.assertTrue(digital._reneg["fpe"])
                expected_rf = digital._reneg["signal"]
                response = b""
                for _ in range(10):
                    response += digital.read_g711(BLOCK, law)
                    if digital._reneg.get("phase") == "TRN2":
                        break
                self.assertEqual(digital._reneg["phase"], "TRN2")
                self.assertIn(expected_rf, response)

    def test_nonzero_dil_phase3_reaches_lapm_and_transfers_payload(self):
        up = bytes((i * 37 + 5) & 0xFF for i in range(256))
        down = bytes((i * 91 + 17) & 0xFF for i in range(256))
        analogue, digital, got_up, got_down = pair(
            law="pcma", delay=240, seconds=20, up=up, down=down,
            dil_ucodes=(1, 18, 35, 52, 69, 86, 103))
        self.assertEqual(got_up, up)
        self.assertEqual(got_down, down)
        self.assertTrue(analogue.connected and digital.state == "data")
        self.assertIsNotNone(analogue.lapm)
        self.assertIsNotNone(digital.lapm)
        ja_events = [value for name, value in digital.events if name == "phase3-ja"]
        self.assertEqual(ja_events[-1]["N"], 7)
        self.assertEqual(ja_events[-1]["Ucodes"], [1, 18, 35, 52, 69, 86, 103])
        dils = [(name, value) for name, value in digital.events if name.startswith("phase3-dil-")]
        self.assertEqual([name for name, _ in dils], [
            "phase3-dil-tx-start", "phase3-dil-start",
            "phase3-dil-stop-request", "phase3-dil-stopped"])
        self.assertEqual(dils[0][1]["frame_phase_from_sd"], 0)
        self.assertGreater(dils[-1][1], 0)
        for marker in ("phase3-sd-tx-start", "phase3-jd-tx-start",
                       "phase3-dil-tx-start", "phase3-ri-tx-start",
                       "phase3-r-bar-i-tx-start"):
            values = [value for name, value in digital.events if name == marker]
            self.assertTrue(values, (marker, digital.events))
            self.assertTrue(all(value["frame_phase_from_sd"] == 0
                                for value in values), (marker, values))
        handoff = next(value for name, value in digital.events
                       if name == "phase4-tx-handoff")
        self.assertEqual(handoff["frame_origin_name"], "Sd")
        self.assertEqual(handoff["frame_phase_from_sd"], 0)
        self.assertTrue(handoff["frame_phase_verified"])

    def test_rate_renegotiation_listener_stays_passive_during_bulk_data(self):
        up = bytes((i * 37 + 5) & 0xFF for i in range(4096))
        down = bytes((i * 91 + 17) & 0xFF for i in range(4096))
        analogue, digital, got_up, got_down = pair(
            law="pcmu", delay=240, seconds=40, up=up, down=down,
            enable_rate_renegotiation=True)
        self.assertEqual(got_up, up)
        self.assertEqual(got_down, down)
        self.assertTrue(analogue.connected and digital.state == "data")
        self.assertNotIn("v92-rate-Rd-detected", [name for name, _ in analogue.events])
        self.assertIsNone(analogue._reneg)

    def test_tone_b_in_phase3_requests_a_retrain(self):
        # Astound sent Tone B 0.9 s into our Phase 3 (2026-11-20); 9.5.2.2 has the analogue modem answer it.
        n = np.arange(8000)
        tone_b = 3000.0 * np.sin(2 * np.pi * 1200 * n / 8000)
        ri_like = np.tile([2000.0] * 3 + [-2000.0] * 3, 1400)[:8000]
        sd_like = np.tile([2000.0, 0.0, 2000.0, -2000.0, 0.0, -2000.0], 1400)[:8000]
        for name, signal, length, expected in (("tone-b-60ms", tone_b, 480, True), ("tone-b-40ms", tone_b, 320, False),
                                               ("ri-like", ri_like, 8000, False), ("sd-like", sd_like, 8000, False)):
            with self.subTest(name=name):
                analogue = asn.V92AnalogueSession(law="pcmu")
                x = np.concatenate((signal[:length], np.zeros(BLOCK * 4)))
                for start in range(0, len(x), BLOCK):
                    analogue.feed(x[start:start + BLOCK])
                names = [event for event, _ in analogue.events]
                self.assertEqual(analogue.retrain_requested, expected, names)
                if expected:
                    self.assertEqual(analogue.stage, "failed")
                    self.assertIn("v92-phase3-tone-b", names)
                    self.assertIn("Tone B", analogue.retrain_reason)

    def test_rate_renegotiation_waits_for_the_end_of_r_bar(self):
        # A chunk that ends inside R-bar leaves TRN2d's start beyond the received history; the responder used to
        # fail the call there ("history expired"), as one live Astound call did (2026-11-20).
        from types import SimpleNamespace

        class Tracker:
            def __init__(self, script):
                self.script = list(script)

            def feed(self, samples):
                return self.script.pop(0) if self.script else []

        analogue = asn.V92AnalogueSession(law="pcmu", enable_rate_renegotiation=True)
        analogue.stage = "data"
        analogue._e1u_sign = 0
        old = SimpleNamespace(receive=lambda codes: None)
        analogue._p4 = old
        analogue._pump = lambda: None
        analogue._reneg_tracker = Tracker([[], [("ru", 100)], [("ru-bar", 470)], []])
        codes = bytes([0xFF]) * BLOCK
        for _ in range(3):
            analogue._feed_rate_renegotiation_probe(np.zeros(BLOCK), codes)
        # total 480: R-bar ends at 494, after what has arrived
        self.assertNotEqual(analogue.stage, "failed")
        self.assertIsNone(analogue._reneg)
        analogue._feed_rate_renegotiation_probe(np.zeros(BLOCK), codes)
        names = [event for event, _ in analogue.events]
        self.assertEqual(analogue.stage, "renegotiating", names)
        self.assertIn("v92-rate-Rd-detected", names)
        self.assertIsNot(analogue._p4, old)

    def test_unacknowledged_cpt_requests_a_retrain(self):
        # Astound twice held Ri through 200+ CPt frames (2026-11-19); a good exchange needs 6.
        limit = asn.V92AnalogueSession.CPT_UNACKNOWLEDGED_FRAMES
        for acknowledged in (False, True):
            with self.subTest(acknowledged=acknowledged):
                analogue = asn.V92AnalogueSession(law="pcmu")
                p3 = analogue._p3
                p3._ri_seen, p3._rbar_seen = True, acknowledged
                p3.cpt_frames_sent = limit - 1
                analogue.feed(np.zeros(BLOCK))
                self.assertEqual(analogue.stage, "phase3")
                p3.cpt_frames_sent = limit
                analogue.feed(np.zeros(BLOCK))
                names = [name for name, _ in analogue.events]
                if acknowledged:
                    self.assertEqual(analogue.stage, "phase3")
                    self.assertFalse(analogue.retrain_requested)
                else:
                    self.assertEqual(analogue.stage, "failed")
                    self.assertTrue(analogue.retrain_requested)
                    self.assertIn("v92-phase3-cpt-unacknowledged", names)

    def test_live_digital_initiated_rate_renegotiation_keeps_lapm_and_payload(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 240):
                with self.subTest(law=law, delay=delay):
                    self._check_live_rate_renegotiation(law, delay)

    def test_validated_receiver_digital_initiated_rate_renegotiation_refits_and_keeps_payload(self):
        for law in ("pcmu", "pcma"):
            for silent in (False, True):
                with self.subTest(law=law, silent=silent):
                    self._check_live_rate_renegotiation(law, 240, silent_period=silent, validated=True)

    def test_live_digital_initiated_silent_rate_renegotiation_keeps_lapm_and_payload(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                self._check_live_rate_renegotiation(law, 240, silent_period=True)

    def test_equalised_digital_rate_change_keeps_precoder_lapm_and_exact_payload(self):
        for law in ("pcmu", "pcma"):
            for early in (None, 4008):
                with self.subTest(law=law, early=early):
                    self._check_live_rate_renegotiation(
                        law, 240, channel=LINE, equalise=True,
                        silent_period=early is not None, silent_early_rt_after=early)

    def test_live_early_silent_rate_change_with_peer_listener_keeps_payload(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                self._check_live_rate_renegotiation(
                    law, 240, silent_period=True, silent_early_rt_after=4008)

    def test_live_analogue_initiated_silent_rate_renegotiation_keeps_lapm_and_payload(self):
        stacks = (("equalised", {"equalise": True, "phase3_equalise": True}),
                  ("trn2u-fit-viterbi", {"b1u_rx_mode": "trn2u-fit-viterbi", "cpd_constellation_mode": "adaptive"}))
        for law in ("pcmu", "pcma"):
            for name, stack in stacks:
                with self.subTest(law=law, stack=name):
                    self._check_analogue_initiated_silent_rate_renegotiation(law, stack)

    def _check_analogue_initiated_silent_rate_renegotiation(self, law, stack):
        up1 = bytes((i * 37 + 5) & 0xFF for i in range(80))
        down1 = bytes((i * 91 + 17) & 0xFF for i in range(80))
        up2 = bytes((i * 13 + 101) & 0xFF for i in range(80))
        down2 = bytes((i * 53 + 7) & 0xFF for i in range(80))
        analogue = asn.V92AnalogueSession(law=law, enable_rate_renegotiation=True)
        digital = dsn.V92DigitalSession(
            law=law, enable_rate_renegotiation_response=True, **stack)
        analogue.feed_bytes(up1)
        digital.feed_bytes(down1)
        silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
        to_digital = [0.0] * 240
        to_analogue = [float(v34_g711.decode(np.array([silence], dtype=np.uint8), law)[0])] * 240
        line_taps = np.asarray([1.0, 0.08, -0.03, 0.015], dtype=float)
        line_history = np.zeros(len(line_taps) - 1, dtype=float)
        got_up, got_down = bytearray(), bytearray()

        def exchange(a_out_override=None):
            nonlocal line_history
            a_out = analogue.read(BLOCK) if a_out_override is None else a_out_override
            d_out = digital.read_g711(BLOCK, law)
            joined = np.concatenate((line_history, a_out))
            impaired = np.convolve(joined, line_taps, mode="valid")
            line_history = joined[len(impaired):]
            to_digital.extend(v34_g711.roundtrip(np.rint(impaired), law).astype(float))
            to_analogue.extend(v34_g711.decode(np.frombuffer(d_out, dtype=np.uint8), law).astype(float))
            digital.feed_pcm(np.asarray(to_digital[:BLOCK])); del to_digital[:BLOCK]
            analogue.feed(np.asarray(to_analogue[:BLOCK])); del to_analogue[:BLOCK]
            got_up.extend(digital.read_received_data())
            got_down.extend(analogue.take_received())

        for _ in range(1200):
            exchange()
            if got_up[:len(up1)] == up1 and got_down[:len(down1)] == down1 \
                    and analogue.connected and digital.state == "data":
                break
        self.assertEqual(got_up, up1)
        self.assertEqual(got_down, down1)
        old_up_bits = digital._p4.data_rx.rx.profile.bits_per_frame
        old_down_bits = analogue._p4.data_rx.fb
        old_downstream_rate = dm.downstream_profile_from_cpu(digital._p4.peer_cp)["rate"]
        old_analogue_lapm_rate = analogue.lapm._v42.rate
        old_digital_lapm_rate = digital.lapm._v42.rate
        old_side = analogue._p4
        if old_side.units_sent % 12 == 0:
            exchange()
        old_units = old_side.units_sent
        self.assertNotEqual(old_units % 12, 0)
        samples_to_boundary = (-old_units) % 12
        analogue.request_rate_renegotiation(drn_down=1, silent_period=True)
        first_renegotiation_chunk = analogue.read(BLOCK)
        self.assertEqual(old_side.units_sent, old_units + samples_to_boundary)
        self.assertEqual(old_side.units_sent % 12, 0)
        self.assertEqual(analogue._reneg["phase"], "Ru")
        self.assertEqual(analogue._reneg["signal_at"], BLOCK - samples_to_boundary)
        exchange(first_renegotiation_chunk)
        analogue.feed_bytes(up2)
        digital.feed_bytes(down2)
        for _ in range(1500):
            exchange()
            if got_up[80:160] == up2 and got_down[80:160] == down2 \
                    and analogue.connected and digital.state == "data" \
                    and analogue._reneg is None and digital._reneg is None:
                break

        self.assertEqual(got_up, up1 + up2)
        self.assertEqual(got_down, down1 + down2)
        if stack.get("b1u_rx_mode") == "trn2u-fit-viterbi":
            # The new side fitted the resent TRN2u and designed its CPd again.
            self.assertEqual(digital._p4.data_rx.rx_filter_diagnostic["state"], "active")
            self.assertTrue(digital._p4.diagnostic_metadata["cpd_constellation"]["final"])
        else:
            self.assertEqual(digital._p4.data_rx.rx.profile.bits_per_frame, old_up_bits)
        self.assertLess(analogue._p4.data_rx.fb, old_down_bits)
        self.assertLess(dm.downstream_profile_from_cpu(digital._p4.peer_cp)["rate"],
                        old_downstream_rate)
        if stack.get("b1u_rx_mode") != "trn2u-fit-viterbi":
            self.assertEqual(analogue.lapm._v42.rate, old_analogue_lapm_rate)
        self.assertLess(digital.lapm._v42.rate, old_digital_lapm_rate)
        self.assertIn("v92-rate-Ru-detected", [name for name, _ in digital.events])
        self.assertIn("v92-rate-Rd-start", [name for name, _ in digital.events])
        self.assertIn("v92-rate-Rd-detected", [name for name, _ in analogue.events])
        self.assertIn("SILENT:trn2u-complete", analogue._p4.sent)
        self.assertEqual(analogue._p4._silent_trn_generated, 8004)
        self.assertIn("SILENT:rt-sent", digital._p4.sent)
        self.assertIn(("suvu", True), digital._p4.silent_period_events)
        self.assertIn(("suvu", False), digital._p4.silent_period_events)

    def _check_live_rate_renegotiation(self, law, delay, silent_period=False,
                                       silent_early_rt_after=None, channel=None, equalise=False,
                                       validated=False):
        up1 = bytes((i * 37 + 5) & 0xFF for i in range(80))
        down1 = bytes((i * 91 + 17) & 0xFF for i in range(80))
        up2 = bytes((i * 13 + 101) & 0xFF for i in range(80))
        down2 = bytes((i * 53 + 7) & 0xFF for i in range(80))
        analogue = asn.V92AnalogueSession(law=law, enable_rate_renegotiation=True)
        digital = dsn.V92DigitalSession(
            law=law, equalise=equalise, enable_rate_renegotiation_response=True,
            **({"b1u_rx_mode": "trn2u-fit-viterbi", "cpd_constellation_mode": "adaptive"} if validated else {}))
        analogue.feed_bytes(up1)
        digital.feed_bytes(down1)
        silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
        to_digital = [0.0] * delay
        to_analogue = [float(v34_g711.decode(np.array([silence], dtype=np.uint8), law)[0])] * delay
        got_up, got_down = bytearray(), bytearray()

        channel_history = np.zeros(0 if channel is None else len(channel) - 1)

        def exchange():
            nonlocal channel_history
            a_out = analogue.read(BLOCK)
            if channel is not None:
                joined = np.concatenate((channel_history, a_out))
                a_out = np.convolve(joined, channel, mode="valid")
                channel_history = joined[len(a_out):]
            d_out = digital.read_g711(BLOCK, law)
            to_digital.extend(v34_g711.roundtrip(np.rint(a_out), law).astype(float))
            to_analogue.extend(v34_g711.decode(np.frombuffer(d_out, dtype=np.uint8), law).astype(float))
            digital.feed_pcm(np.asarray(to_digital[:BLOCK])); del to_digital[:BLOCK]
            analogue.feed(np.asarray(to_analogue[:BLOCK])); del to_analogue[:BLOCK]
            got_up.extend(digital.read_received_data())
            got_down.extend(analogue.take_received())

        for _ in range(1200):
            exchange()
            if got_up[:len(up1)] == up1 and got_down[:len(down1)] == down1 and analogue.connected \
                    and digital.state == "data":
                break
        self.assertEqual(got_up, up1)
        self.assertEqual(got_down, down1)
        if validated:
            # The adaptive CPd's rate, designed from the TRN2u fit
            designed = digital._p4.diagnostic_metadata["cpd_constellation"]["rate_bps"]
            self.assertEqual(analogue.lapm._v42.rate, round(designed))
        elif equalise:
            # The line design's rate for this channel (34,667 or more)
            designed = digital._p4.data_rx.rx.profile.bits_per_frame * 8000 / 12
            self.assertEqual(analogue.lapm._v42.rate, round(designed))
            self.assertGreaterEqual(analogue.lapm._v42.rate, 34667)
        else:
            self.assertEqual(analogue.lapm._v42.rate, 34667)
        self.assertGreater(digital.lapm._v42.rate, 0)
        old_up_bits = digital._p4.data_rx.rx.profile.bits_per_frame
        old_digital_lapm_rate = digital.lapm._v42.rate
        old_down_bits = analogue._p4.data_rx.fb

        old_filter = digital._p4.rx_filter
        old_cpd = digital._p4._build_cp(False)
        # The first CPd fixed G = 1/LU of this call's TRN2u; renegotiated CPds keep it.
        lu_gain4 = digital._p4.lu_gain4
        if not validated:                       # the adaptive CPd sends a fixed G instead
            self.assertEqual(lu_gain4, round(4 * 65536 / digital._p4._decoder.lu))
        if equalise:
            with self.assertRaisesRegex(ValueError, "drn_up"):
                digital.request_rate_renegotiation(drn_up=True)
            with self.assertRaisesRegex(ValueError, "capacity"):
                digital.request_rate_renegotiation(drn_up=19)
            self.assertIsNone(digital._reneg)
            self.assertIs(digital._p4.rx_filter, old_filter)
        analogue.set_next_rate_renegotiation(drn_down=1)
        digital.request_rate_renegotiation(
            drn_up=8, silent_period=silent_period, silent_early_rt_after=silent_early_rt_after)
        renegotiation_decoder = digital._reneg["new_side"]._decoder if digital._reneg else None
        analogue.feed_bytes(up2)
        digital.feed_bytes(down2)
        with patch.object(asn.p4c, "analogue_side", wraps=asn.p4c.analogue_side) as built:
            for _ in range(1500):
                exchange()
                if got_up[80:160] == up2 and got_down[80:160] == down2 \
                        and analogue.connected and digital.state == "data" \
                        and analogue._reneg is None and digital._reneg is None:
                    break
        # 8.7.6: the renegotiation's TRN2u differential sign starts from E1u's last sign on both sides, as in
        # Phase 4. Astound trained on an inverted TRN2u when ours started from 0 (call 2, 2026-11-19).
        self.assertEqual(built.call_args.kwargs.get("initial_sign"), analogue._e1u_sign)
        self.assertEqual(digital._e1u_sign, analogue._e1u_sign)
        if renegotiation_decoder is not None and hasattr(renegotiation_decoder, "_initial_sign"):
            self.assertEqual(renegotiation_decoder._initial_sign, digital._e1u_sign)

        self.assertEqual(got_up, up1 + up2)
        self.assertEqual(got_down, down1 + down2)
        if equalise:
            self.assertIs(digital._p4.rx_filter, old_filter)
            import dataclasses
            import v92_cp
            self.assertEqual(v92_cp.with_cpd_gain(digital._cpd_builder(False), lu_gain4),
                             digital._p4._build_cp(False))
            on_wire = pu.profile_from_cpd(v92_cp.parse_cpd(digital._p4._build_cp(False)))
            self.assertEqual(digital._p4.data_rx.profile,
                             dataclasses.replace(on_wire, gain=1.0))
        self.assertIn("v92-rate-Rd-detected", [name for name, _ in analogue.events])
        self.assertIn("v92-rate-ru-start", [name for name, _ in analogue.events])
        self.assertIn("v92-rate-Rd-start", [name for name, _ in digital.events])
        self.assertIn("v92-rate-trn2d-start", [name for name, _ in digital.events])
        self.assertIn("v92-rate-renegotiated", [name for name, _ in analogue.events])
        self.assertIn("v92-rate-renegotiated", [name for name, _ in digital.events])
        self.assertLess(digital._p4.data_rx.rx.profile.bits_per_frame, old_up_bits)
        self.assertLess(analogue._p4.data_rx.fb, old_down_bits)
        if validated:
            # A fresh TRN2u fit and CPd design, capped at the requested DRN 8 (33,333)
            designed = digital._p4.diagnostic_metadata["cpd_constellation"]
            self.assertTrue(designed["final"])
            self.assertLessEqual(designed["drn"], 8)
            self.assertEqual(analogue.lapm._v42.rate, round(designed["rate_bps"]))
            self.assertEqual(digital._p4.data_rx.rx_filter_diagnostic["state"], "active")
        else:
            self.assertEqual(analogue.lapm._v42.rate, 33333)
        self.assertEqual(digital.lapm._v42.rate, 28000)
        self.assertNotEqual(digital.lapm._v42.rate, old_digital_lapm_rate)
        self.assertEqual(digital.phase4.transmitter.cp_profile["rate"], 28000.0)
        if silent_period:
            self.assertIn("SILENT:rt-sent", digital._p4.sent)
            if silent_early_rt_after is None:
                self.assertIn("SILENT:trn2u-complete", analogue._p4.sent)
                self.assertEqual(analogue._p4._silent_trn_generated, 8004)
            else:
                self.assertIn("SILENT:trn2u-terminated-by-rt", analogue._p4.sent)
                self.assertLess(analogue._p4._silent_trn_generated, 8004)

        if equalise:
            # A second exchange exercises rate increases and builder reuse;
            # the fitted channel and its quantised filter must remain intact.
            links = (analogue.lapm, digital.lapm)
            up3, down3 = b"restore-up" * 8, b"restore-down" * 8
            analogue.set_next_rate_renegotiation(drn_down=2)
            digital.request_rate_renegotiation(drn_up=old_up_bits // 2 - 17)
            analogue.feed_bytes(up3)
            digital.feed_bytes(down3)
            for _ in range(1500):
                exchange()
                if (got_up == up1 + up2 + up3 and got_down == down1 + down2 + down3
                        and analogue._reneg is None and digital._reneg is None):
                    break
            self.assertEqual(got_up, up1 + up2 + up3)
            self.assertEqual(got_down, down1 + down2 + down3)
            self.assertEqual(digital._p4.data_rx.rx.profile.bits_per_frame, old_up_bits)
            self.assertEqual(analogue._p4.data_rx.fb, old_down_bits)
            self.assertIs(digital._p4.rx_filter, old_filter)
            self.assertEqual(digital._p4._build_cp(False), old_cpd)
            self.assertIs(analogue.lapm, links[0])
            self.assertIs(digital.lapm, links[1])

    def test_lapm_bytes_each_way_over_phase_3_phase_4_and_data(self):
        up = bytes((i * 37 + 5) & 0xFF for i in range(300))
        down = bytes((i * 91 + 17) & 0xFF for i in range(300))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                a, d, got_up, got_down = pair(law, delay=240, up=up, down=down)
                self.assertEqual(d.state, "data", (d.events[-3:], getattr(d, "failure", None)))
                self.assertTrue(a.connected)
                self.assertEqual(got_up[:len(up)], up)
                self.assertEqual(got_down[:len(down)], down)
                self.assertEqual(d.lapm.v42_mode, "lapm")

    def test_a_dispersive_upstream_line_with_the_equaliser_and_a_designed_precoder(self):
        up = bytes((i * 37 + 5) & 0xFF for i in range(400))
        down = bytes((i * 91 + 17) & 0xFF for i in range(400))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                a, d, got_up, got_down = pair(law, delay=240, seconds=60, up=up, down=down,
                                              channel=LINE, equalise=True)
                self.assertEqual(d.state, "data", (d.events[-4:], d.failure))
                self.assertIn("line-designed", [n for n, _ in d.events])
                self.assertEqual(got_up[:len(up)], up)
                self.assertEqual(got_down[:len(down)], down)

    def test_a_precursor_line_whose_prefilter_is_scaled_into_q015_carries_data(self):
        # The designed prefilter's largest tap is above 1 here (the line of
        # test_line_design_respects_minimum_info1a_limits); its scaled taps must still give exact data.
        channel = [0.0, 0.25, 1.0, 0.42, -0.18, 0.07]
        up = bytes((i * 37 + 5) & 0xFF for i in range(400))
        down = bytes((i * 91 + 17) & 0xFF for i in range(400))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                a, d, got_up, got_down = pair(law, delay=240, seconds=60, up=up, down=down,
                                              channel=channel, equalise=True)
                self.assertEqual(d.state, "data", (d.events[-4:], d.failure))
                designed = next(v for n, v in d.events if n == "line-designed")
                self.assertLess(designed["z2_scale"], 1.0)
                self.assertEqual(got_up[:len(up)], up)
                self.assertEqual(got_down[:len(down)], down)

    def test_phase3_equaliser_recovers_trn1u_and_ja_on_12_sample_isi_line(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                def acquire(use_equaliser):
                    analogue = asn.V92AnalogueSession(law=law)
                    digital = dsn.V92DigitalSession(law=law, phase3_equalise=use_equaliser)
                    self.assertEqual(digital._p3._equalise, use_equaliser)
                    history = np.zeros(len(PHASE3_ISI_LINE) - 1)
                    channel = np.asarray(PHASE3_ISI_LINE, dtype=float)
                    for _ in range(80):
                        samples = analogue.read(BLOCK)
                        joined = np.concatenate((history, samples))
                        impaired = np.convolve(joined, channel, mode="valid")
                        history = joined[len(impaired):]
                        received = v34_g711.roundtrip(np.rint(impaired), law).astype(float)
                        digital.feed_pcm(received)
                        if digital._p3.ja is not None or digital.state == "failed":
                            break
                    return digital

                without = acquire(False)
                self.assertEqual(without.state, "failed")
                self.assertIn("no-trn1u", without.failure)

                with_equaliser = acquire(True)
                self.assertIsNotNone(with_equaliser._p3.ja)
                self.assertEqual(with_equaliser._p3.state, "sending")
                eq_event = next(v for k, v in with_equaliser._p3.events if k == "phase3-equaliser")
                self.assertLess(eq_event["match_before"], 0.75)
                self.assertGreaterEqual(eq_event["match_after"], 0.95)

    def test_heavy_isi_line_runs_end_to_end_with_both_equalisers(self):
        up = bytes((i * 37 + 5) & 0xFF for i in range(300))
        down = bytes((i * 91 + 17) & 0xFF for i in range(300))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                a, d, got_up, got_down = pair(law, seconds=60, up=up, down=down, channel=PHASE3_ISI_LINE,
                                              equalise=True, phase3_equalise=True)
                self.assertEqual(d.state, "data", (d.events[-3:], d.failure))
                self.assertEqual(got_up[:len(up)], up)
                self.assertEqual(got_down[:len(down)], down)


if __name__ == "__main__":
    unittest.main()
