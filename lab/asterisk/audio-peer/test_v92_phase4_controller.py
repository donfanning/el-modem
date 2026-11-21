import unittest
from unittest import mock

import numpy as np

import v34_g711
import v92_pcm_upstream as pu
import v92_phase4_controller as ctl
import v92_phase4_downstream as down
import v92_phase4_upstream as up
import v92_cp
import v92_phase3

UP = bytes((i * 37 + 5) & 0xFF for i in range(120))        # analogue to digital
DOWN = bytes((i * 91 + 17) & 0xFF for i in range(120))     # digital to analogue


def profile(law, drn=2):
    masks = tuple(tuple(range(16 + 16 * index, 32 + 16 * index)) for index in range(6))
    return {"kind": "cpt", "codec": law, "ack": 0, "rate": (drn + 8) * 8000 / 6,
            "spectral_sign_bits": 0, "lookahead": 0, "shape": (),
            "interval_constellations": (0, 1, 2, 3, 4, 5),
            "constellations": masks, "codec_constellations": masks}


def _data_side(frame):
    def more(_side, count):
        n = ((count + frame - 1) // frame) * frame
        return [0] * n

    side = ctl.Side("test", encode=lambda bits: list(bits), decoder=None, parsers=(),
                    build_suv=lambda ack, silent: [], build_cp=lambda ack: [], e_bits=0,
                    peer_e_bits=0, trn_units=0, early=False, ack_wait=0,
                    trn_msg_bits=0, align_bits=1, peer_align_bits=1, data_rx=object(),
                    make_data_units=lambda _peer: [], idle_unit=0, data_more=more)
    side.phase = "DATA"
    side.sent = ["DATA"]
    return side


def run(law="pcmu", *, delay=240, block=160, max_units=90000, ack_wait=None, payloads=(UP, DOWN)):
    """analogue <-> digital through a delay each way; the upstream also crosses G.711."""
    wait = 2 * delay + 400 if ack_wait is None else ack_wait
    analogue = ctl.analogue_side(profile(law), law, ack_wait=wait, payload=payloads[0])
    digital = ctl.digital_side(profile(law), law, ack_wait=wait, payload=payloads[1])
    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
    to_digital = [0.0] * delay
    to_analogue = [silence] * delay
    for t in range(0, max_units, block):
        a = np.array(analogue.transmit(block))
        a = v34_g711.roundtrip(np.rint(a), law).astype(float)
        d = bytes(digital.transmit(block))
        to_digital.extend(a)
        to_analogue.extend(d)
        digital.receive(np.array(to_digital[:block])); del to_digital[:block]
        analogue.receive(bytes(to_analogue[:block])); del to_analogue[:block]
        if (len(digital.data_rx.payload()) >= len(payloads[0])
                and len(analogue.data_rx.payload()) >= len(payloads[1])):
            return analogue, digital, t
    return analogue, digital, None


def shaped_profile(law, lookahead, drn=2):
    """A CPt with S_r=1 spectral shaping and the given look-ahead depth."""
    return dict(profile(law, drn), spectral_sign_bits=1, lookahead=lookahead,
                shape=(1.0, 0.0, 0.0, 0.0))


class ControllerTests(unittest.TestCase):
    def test_ed_is_fully_transmitted_with_shaping_lookahead(self):
        # A look-ahead mapper holds its last frames until flushed. Ed is the training
        # mapper's final block, so all of Ed must reach the wire before B1d starts;
        # SmartLink needs both Ed frames (2*(S+K) zeros) to leave the CPt slicer.
        for law in ("pcmu", "pcma"):
            for depth in (0, 1, 2):
                with self.subTest(law=law, lookahead=depth):
                    prof = shaped_profile(law, depth)
                    side = ctl.digital_side(prof, law)
                    sent_bits = []
                    encode = side._encode
                    side._encode = lambda bits: (sent_bits.extend(bits), encode(bits))[1]
                    side._make_data_units = lambda _peer: []
                    side._data_more = None
                    # Some SUVd first, so pipeline state crosses message boundaries.
                    side.phase = "SUV"
                    wire = list(side.transmit(600))
                    side.saw_suv = side.saw_cp = side.saw_ack = True
                    side.sent_ack = True
                    side.cp_sent_at = 0
                    wire += list(side.transmit(1200))
                    self.assertIn("E", side.sent)
                    self.assertEqual(side.phase, "DATA")

                    ref = ctl.dp4.V90DigitalPhase4Encoder(prof, law)
                    ref.reset()
                    fb = ref.s + ref.k
                    self.assertEqual(len(sent_bits) % fb, 0)
                    expected = list(ref.encode_bits(sent_bits)) + list(ref.flush_lookahead())
                    self.assertEqual(len(expected), 6 * len(sent_bits) // fb)
                    self.assertEqual(sent_bits[-2 * fb:], [0] * (2 * fb))
                    # Every training-mapper symbol, including Ed's last frame(s), precedes
                    # the first data-phase unit.
                    self.assertEqual(wire[:len(expected)], expected)

    def test_analogue_side_takes_a_five_frame_ed_and_a_zero_frame_before_b1d(self):
        # Astound's V.92 server (live calls 2 and 5, 2026-11-19): Ed is 5 TRN2d frames, not 2, and one data frame
        # of zeros precedes B1d. Ending E after 2 frames read the rest of Ed as B1d (57-60 errors in its first 4 frames).
        make_digital = ctl.digital_side

        def astound_side(prof, law, **kwargs):
            side = make_digital(prof, law, **kwargs)
            side._e_bits = side._e_bits // 2 * 5
            make = side._make_data_units

            def with_zero_frame(peer_cpu):
                enc = ctl.dp4.V90DigitalPhase4Encoder(ctl.dm.downstream_profile_from_cpu(peer_cpu), law)
                enc.reset()
                return list(enc.encode_bits([0] * (enc.s + enc.k))) + list(make(peer_cpu))

            side._make_data_units = with_zero_frame
            return side

        for law in ("pcmu", "pcma"):
            with self.subTest(law=law), mock.patch.object(ctl, "digital_side", astound_side):
                analogue, digital, done = run(law)
                self.assertIsNotNone(done)
                self.assertEqual(analogue.data_rx.b1_errors, 0)
                self.assertTrue(analogue.data_rx.qualified)
                self.assertEqual(analogue.data_rx.payload()[:len(DOWN)], DOWN)
                self.assertEqual(digital.data_rx.payload()[:len(UP)], UP)

    def test_empty_precoder_policy_changes_only_the_optional_filter_section(self):
        included = ctl.digital_side(profile("pcmu"), "pcmu", empty_precoder_policy="include")
        default = ctl.digital_side(profile("pcmu"), "pcmu")
        omitted = ctl.digital_side(profile("pcmu"), "pcmu", empty_precoder_policy="omit")
        identity = ctl.digital_side(profile("pcmu"), "pcmu", empty_precoder_policy="identity")
        identity_z2_two = ctl.digital_side(
            profile("pcmu"), "pcmu", empty_precoder_policy="identity", empty_precoder_z2=[32767, 0])

        with_part = included._build_cp(False)
        default_part = default._build_cp(False)
        without_part = omitted._build_cp(False)
        identity_part = identity._build_cp(False)
        identity_z2_two_part = identity_z2_two._build_cp(False)
        parsed_with = v92_cp.parse_cpd(with_part)
        parsed_without = v92_cp.parse_cpd(without_part)
        parsed_identity = v92_cp.parse_cpd(identity_part)
        parsed_identity_z2_two = v92_cp.parse_cpd(identity_z2_two_part)

        self.assertEqual((with_part[19], with_part[20], with_part[21]), (1, 1, 1))
        self.assertEqual((without_part[19], without_part[20], without_part[21]), (1, 0, 1))
        self.assertEqual(default_part, without_part)
        self.assertEqual(len(with_part) - len(without_part), 4 * 17)
        self.assertEqual(parsed_with["precoder"], {"z1": (), "p1": (), "z2": (), "p2": ()})
        self.assertNotIn("precoder", parsed_without)
        self.assertEqual(parsed_identity["precoder"], {
            "z1": (), "p1": (), "z2": (32767 / 32768.0,), "p2": ()})
        self.assertEqual(parsed_identity_z2_two["precoder"], {
            "z1": (), "p1": (), "z2": (32767 / 32768.0, 0.0), "p2": ()})
        for parsed in (parsed_without, parsed_identity, parsed_identity_z2_two):
            self.assertEqual(parsed["drn"], parsed_with["drn"])
            self.assertEqual(parsed["moduli"], parsed_with["moduli"])
            self.assertEqual(parsed["interval_sets"], parsed_with["interval_sets"])
            self.assertEqual(parsed["constellations"], parsed_with["constellations"])
            self.assertEqual(parsed["ack"], parsed_with["ack"])

        with self.assertRaises(ValueError):
            ctl.digital_side(profile("pcmu"), "pcmu", empty_precoder_policy="unknown")

    def test_expanded10_cpd_dry_run_uses_approved_arm8_profile(self):
        side = ctl.digital_side(
            profile("pcmu"), "pcmu", drn=1, empty_precoder_policy="omit",
            cpd_constellation_mode="expanded10", cpd_gain4=0x0300)
        parsed = v92_cp.parse_cpd(side._build_cp(False))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["drn"], 1)
        self.assertEqual(parsed["trellis"], 1)
        self.assertEqual(parsed["gain"], 0x0300 / (4 * 65536))
        self.assertEqual(parsed["moduli"], (10, 10, 10, 5) * 3)
        self.assertEqual(parsed["interval_sets"], (0,) * 6)
        self.assertEqual(parsed["constellations"][0],
                         (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236))
        self.assertNotIn("precoder", parsed)
        rx_profile = pu.profile_from_cpd(parsed)
        self.assertEqual(rx_profile.constellations[0], parsed["constellations"][0])
        self.assertEqual(rx_profile.moduli, (10, 10, 10, 5) * 3)
        self.assertEqual(rx_profile.bits_per_frame, 36)
        self.assertEqual(side.diagnostic_metadata["cpd_constellation"]["level_gaps"],
                         [196, 200, 224, 272, 256, 256, 256, 224, 256])

    def test_adaptive_cpd_designs_from_the_trn2u_fit_with_peer_safe_moduli(self):
        import v34_g711
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                side = ctl.digital_side(
                    profile(law), law, drn=1, empty_precoder_policy="omit", b1u_rx_mode="trn2u-fit-viterbi",
                    cpd_constellation_mode="adaptive", cpd_gain4=0x0300)
                provisional = side.diagnostic_metadata["cpd_constellation"]
                self.assertFalse(provisional["final"])
                grid = {int(v) for v in v34_g711.decode(np.arange(256), law) if v > 0}
                # A cleaner fit than the provisional one at the first CPd that is sent.
                side._decoder.eq_residual, side._decoder.fit_target_rms = 0.009, 1.0
                parsed = v92_cp.parse_cpd(side._build_cp(False))
                final = side.diagnostic_metadata["cpd_constellation"]
                self.assertTrue(final["final"])
                levels = parsed["constellations"][0]
                self.assertEqual(list(levels), final["levels_positive"])
                self.assertTrue(set(levels) <= grid)
                lc = len(levels)
                moduli = parsed["moduli"]
                self.assertTrue(all(m <= lc for i, m in enumerate(moduli) if i % 4 < 3))
                # SmartLink sends a zero for one position-3 index unless 2 * M_3 == LC (measured, see design_for_peer).
                self.assertEqual(lc % 2, 0)
                self.assertTrue(all(2 * m == lc for i, m in enumerate(moduli) if i % 4 == 3))
                self.assertLessEqual(final["rms"], ctl.ADAPTIVE_MAX_RMS)
                self.assertGreater(final["rate_bps"], provisional["rate_bps"])
                self.assertEqual(parsed["gain"], 0x0300 / (4 * 65536))
                # The receiver decodes the constellation that was sent; the ack repeat does not redesign.
                self.assertEqual(side.data_rx.profile.constellations[0], levels)
                self.assertEqual(side.data_rx.rx.profile.bits_per_frame, 2 * (parsed["drn"] + 17))
                side._decoder.eq_residual = 0.05
                self.assertEqual(v92_cp.parse_cpd(side._build_cp(True))["constellations"][0], levels)

    def test_adaptive_cpd_without_a_fit_keeps_the_provisional_plan(self):
        side = ctl.digital_side(profile("pcmu"), "pcmu", drn=1, b1u_rx_mode="trn2u-fit-viterbi",
                                cpd_constellation_mode="adaptive", cpd_gain4=0x0300)
        before = list(side.data_rx.profile.constellations[0])
        parsed = v92_cp.parse_cpd(side._build_cp(False))
        self.assertEqual(list(parsed["constellations"][0]), before)
        self.assertIn("final_note", side.diagnostic_metadata["cpd_constellation"])

    def test_cp_wire_capture_is_opt_in_and_records_serialized_frame(self):
        with mock.patch.dict("os.environ", {"EL_MODEM_V92_CAPTURE_CP_BITS": "1"}):
            side = ctl.digital_side(profile("pcmu"), "pcmu")
        side.phase = "SUV"
        side.saw_suv = True
        side.transmit(6)
        captures = side.diagnostics()["cp_tx_captures"]
        self.assertEqual(len(captures), 1)
        frame = captures[0]
        bits = [int(bit) for bit in frame["bits"]]
        self.assertEqual(frame["frame_bits"], len(bits))
        self.assertEqual(bits[:18], [1] * 17 + [0])
        parsed = v92_cp.parse_cpd(bits)
        self.assertIsNotNone(parsed)
        self.assertFalse(parsed["ack"])
        side.saw_cp = True
        for _ in range(2000):
            side.transmit(6)
            if len(side.cp_tx_captures) == 2:
                break
        acked = side.diagnostics()["cp_tx_captures"][1]
        self.assertTrue(acked["ack"])
        self.assertTrue(v92_cp.parse_cpd([int(bit) for bit in acked["bits"]])["ack"])

    def test_cp_wire_capture_is_empty_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            side = ctl.digital_side(profile("pcmu"), "pcmu")
        side.phase = "SUV"
        side.saw_suv = True
        side.transmit(6)
        self.assertEqual(side.diagnostics()["cp_tx_captures"], [])

    def test_upstream_decoder_waits_for_trn2u_and_equalizes_before_frames(self):
        points, initial_sign, trn_symbols = 4, 1, 6000
        group_bits = 12 * 2  # 12 symbols per group, two bits per 4-point symbol
        frames = [v92_cp.build_suvu(wait_for_cpu=True),
                  v92_cp.build_cpu(drn=2, constellations=((0, 3, 6, 9, 12, 15),),
                                   interval_sets=(0,) * 6)]
        bits = [1] * (trn_symbols * 2)
        for frame in frames:
            bits.extend(frame)
            bits.extend([0] * (-len(frame) % group_bits))
        tx = ctl.up.SymbolCoder(points, initial_sign=initial_sign)
        sent = np.asarray(tx.encode(bits), dtype=float) * 1800.0
        channel = np.array([0.03, 0.15, 0.62, -0.22, 0.09, 0.025])
        rng = np.random.default_rng(4)
        prefix = rng.normal(0.0, 5.0, 700)
        received = np.r_[prefix,
                         np.convolve(sent, channel, mode="full")[:len(sent)]
                         + rng.normal(0.0, 2.0, len(sent))]

        decoder = ctl._UpstreamDecoder(points, None, initial_sign=initial_sign,
                                       acquire_trn2u=True)
        decoded = []
        for start in range(0, len(received), 160):
            decoded.extend(decoder.feed(received[start:start + 160]))
        if decoder._fit_thread is not None:
            decoder._fit_thread.join(timeout=2.0)
        decoded.extend(decoder.feed([]))

        self.assertEqual(decoder.acquisition_state, "locked")
        self.assertLessEqual(abs(decoder.trn2u_at - len(prefix)), 8)
        self.assertLess(decoder.eq_residual, 0.02)
        self.assertEqual(decoder.fit_samples, decoder.TRN_FIT_SYMBOLS)
        self.assertGreater(decoder.fit_wall_seconds, 0.0)
        self.assertEqual(decoder.acquisition_attempts, 1)
        found = []
        for offset in range(len(decoded)):
            if decoded[offset:offset + 18] != [1] * 17 + [0]:
                continue
            frame = v92_cp.parse_suvu(decoded[offset:]) or v92_cp.parse_cpu(decoded[offset:])
            if frame is not None:
                found.append(frame["kind"])
        self.assertEqual(found[:2], ["suvu", "cpu"])

    def test_upstream_acquisition_retries_once_after_a_rejected_fit(self):
        import v92_equaliser

        points, initial_sign = 4, 1
        train = np.asarray(v92_phase3.Trn2uModulator(
            points, initial_sign=initial_sign).symbols(1500), dtype=float) * 1800.0
        frames = [v92_cp.build_suvu(wait_for_cpu=True),
                  v92_cp.build_cpu(drn=2, constellations=((0, 3, 6, 9, 12, 15),),
                                   interval_sets=(0,) * 6)]
        bits = [1] * (1500 * 2)
        for frame in frames:
            bits.extend(frame)
            bits.extend([0] * (-len(frame) % 24))
        tail_tx = up.SymbolCoder(points, initial_sign=initial_sign)
        tail = np.asarray(tail_tx.encode(bits[1500 * 2:]), dtype=float) * 1800.0
        channel = np.array([0.03, 0.15, 0.62, -0.22, 0.09, 0.025])
        first = np.convolve(train, channel, mode="full")[:len(train)]
        second = np.convolve(np.r_[train, tail], channel, mode="full")[:len(train) + len(tail)]
        stream = np.r_[np.zeros(80), first, np.zeros(1400), second]

        original_train = v92_equaliser.train
        calls = 0

        def reject_once(received, sent, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.zeros(ctl._UpstreamDecoder.TRN_TAPS), 0, 0.75
            return original_train(received, sent, **kwargs)

        decoder = ctl._UpstreamDecoder(points, None, initial_sign=initial_sign,
                                       acquire_trn2u=True, async_fit=False)
        with mock.patch.object(v92_equaliser, "train", side_effect=reject_once):
            for start in range(0, len(stream), 160):
                decoder.feed(stream[start:start + 160])

        self.assertEqual(calls, 2)
        self.assertEqual(decoder.acquisition_state, "locked")
        self.assertEqual(decoder.acquisition_attempts, 2)
        self.assertEqual(decoder.acquisition_retries, 1)
        self.assertEqual(len(decoder.acquisition_failures), 1)

    def test_fast_upstream_decoder_accepts_only_predecode_sign_seed(self):
        prof = pu.PcmUpstreamProfile(
            constellations=((1.0, 2.0),), interval_set=(0,) * 12,
            moduli=(2,) * 12)
        decoder = ctl._FastUpstreamDecoder(prof, "pcmu")
        decoder.set_initial_sign(1)
        self.assertEqual(decoder.receiver._rx._sign, 1)
        decoder.feed([0.0])
        with self.assertRaisesRegex(RuntimeError, "after decoding starts"):
            decoder.set_initial_sign(0)

    def test_b1_alignment_and_payload_do_not_depend_on_scoring_budget(self):
        import v92_pcm_upstream as pu
        payload = bytes((i * 37 + 11) & 255 for i in range(1024))
        for law in ("pcmu", "pcma"):
            prof = ctl.digital_side(profile(law), law, drn=8).data_rx.profile
            with self.assertRaisesRegex(ValueError, "DRN"):
                ctl.digital_side(profile(law), law, drn=9, upstream_data_profile=prof)
            bits = ctl._scrambled_ones_and_payload(prof.bits_per_frame, payload)
            tx = pu.PcmUpstreamTransmitter(prof)
            samples = [0.0] * 17
            for i in range(0, len(bits), prof.bits_per_frame):
                samples.extend(tx.encode_frame(bits[i:i + prof.bits_per_frame]))
            for budget in (3, 6):
                with self.subTest(law=law, budget=budget):
                    rx = ctl._UpstreamDataRx(
                        prof, law, align_search=24, max_lost_frames=1,
                        max_data_lost_frames=1)
                    rx.SCORES_PER_CALL = budget
                    for start in range(0, len(samples), 160):
                        rx.feed(samples[start:start + 160])
                    while rx._held:
                        rx.feed([])
                    self.assertEqual(rx.alignment, 17)
                    self.assertTrue(rx.qualified)
                    self.assertEqual(rx.b1_errors, 0)
                    self.assertEqual(rx.payload()[:len(payload)], payload)
                    # The configured burst allowance must survive construction
                    # of the receiver at the selected sample alignment.
                    from unittest.mock import patch
                    with patch.object(rx.rx._rx, "decode_intervals",
                                      side_effect=ValueError("outside constellation")):
                        rx.feed([0.0] * 24)
                    self.assertEqual(rx.rx.state, rx.rx.FAILED)

    def test_scanner_reacquires_only_on_complete_sync_after_midstream_reset(self):
        def parse_frame(bits):
            if len(bits) >= 18 and bits[:18] == [1] * 17 + [0]:
                return {"kind": "test", "frame_bits": 18}
            return None

        scanner = ctl._Scanner((parse_frame,), e_bits=4, align_bits=8)
        self.assertEqual(scanner.feed([0, 1, 0] + [1] * 8), [])
        self.assertEqual(scanner.feed([1] * 9 + [0] + [1] * 6),
                         [{"kind": "test", "frame_bits": 18}])
        self.assertEqual(scanner.feed([1] * 17 + [0] + [1] * 6),
                         [{"kind": "test", "frame_bits": 18}])
        self.assertFalse(scanner.failed)

    def test_scanner_trims_long_training_ones_incrementally(self):
        def parse_frame(bits):
            if len(bits) >= 18 and bits[:18] == [1] * 17 + [0]:
                return {"kind": "test", "frame_bits": 18}
            return None

        scanner = ctl._Scanner((parse_frame,), e_bits=4, align_bits=8)
        started = ctl.time.perf_counter()
        worst_feed = 0.0
        for _ in range(60_000 // 160):
            tick = ctl.time.perf_counter()
            self.assertEqual(scanner.feed([1] * 160), [])
            worst_feed = max(worst_feed, ctl.time.perf_counter() - tick)
        elapsed = ctl.time.perf_counter() - started
        self.assertLessEqual(len(scanner.bits), 17)
        self.assertEqual(scanner._bit_base, 60_000 - 17)
        self.assertLess(elapsed, 1.0)
        self.assertLess(worst_feed, 0.05)
        self.assertEqual(scanner.feed([1] * 17 + [0] + [1] * 6),
                         [{"kind": "test", "frame_bits": 18}])
        self.assertFalse(scanner.failed)

    def test_scanner_recovers_suvu_and_cpu_after_long_training_ones(self):
        scanner = ctl._Scanner((ctl.v92_cp.parse_suvu, ctl.v92_cp.parse_cpu),
                               e_bits=64, align_bits=1)
        elapsed = []
        for _ in range(60_000 // 160):
            started = ctl.time.perf_counter()
            self.assertEqual(scanner.feed([1] * 160), [])
            elapsed.append(ctl.time.perf_counter() - started)

        suvu = ctl.v92_cp.build_suvu(level_db=-1.0)
        cpu = ctl.v92_cp.build_cpu(
            drn=3, interval_sets=(0,) * 6,
            constellations=[tuple(range(8))])
        frames = []
        wire = suvu + cpu
        for start in range(0, len(wire), 160):
            started = ctl.time.perf_counter()
            frames.extend(scanner.feed(wire[start:start + 160]))
            elapsed.append(ctl.time.perf_counter() - started)

        self.assertEqual([frame["kind"] for frame in frames], ["suvu", "cpu"])
        self.assertFalse(scanner.failed)
        self.assertLess(max(elapsed), 0.01)

    def test_scanner_follows_control_frames_whose_fill_is_shorter_than_the_alignment(self):
        # SmartLink in an FPE from a 40-bit (26,667) data mode: each SUVu padded to 72 bits, then CPu, then E.
        scanner = ctl._Scanner((ctl.v92_cp.parse_suvu, ctl.v92_cp.parse_cpu), e_bits=40, align_bits=40)
        suvu = ctl.v92_cp.build_suvu(level_db=-1.0)
        cpu = ctl.v92_cp.build_cpu(drn=3, interval_sets=(0,) * 6, constellations=[tuple(range(8))])
        padded = suvu + [0] * (72 - len(suvu))
        wire = padded * 4 + cpu + [0] * (-len(cpu) % 36) + [0] * 40
        frames = []
        for start in range(0, len(wire), 160):
            frames.extend(scanner.feed(wire[start:start + 160]))
        self.assertEqual([frame["kind"] for frame in frames], ["suvu"] * 4 + ["cpu"])
        self.assertTrue(scanner.e_seen)
        self.assertFalse(scanner.failed)

    def test_scanner_candidate_parser_work_is_bounded_per_feed(self):
        calls = 0

        def parse(bits):
            nonlocal calls
            calls += 1
            if len(bits) >= 20 and bits[:18] == [1] * 17 + [0] and bits[18:20] == [0, 1]:
                return {"kind": "valid", "frame_bits": 20}
            return None

        scanner = ctl._Scanner((parse,), e_bits=64, align_bits=1)
        false_candidate = [1] * 17 + [0] + [1] * 143
        stream = false_candidate * 300 + [1] * 17 + [0, 0, 1]
        max_calls_per_feed = 0
        max_pending_candidates = 0
        found = []
        for start in range(0, len(stream), 160):
            before = calls
            found.extend(scanner.feed(stream[start:start + 160]))
            max_calls_per_feed = max(max_calls_per_feed, calls - before)
            max_pending_candidates = max(max_pending_candidates,
                                         len(scanner._later_candidates))
        # Candidate processing is incremental. Drain a bounded amount of
        # already discovered false syncs until the trailing valid frame is
        # reached; no feed may invoke an unbounded number of parsers.
        for _ in range(100):
            if found:
                break
            before = calls
            found.extend(scanner.feed([]))
            max_calls_per_feed = max(max_calls_per_feed, calls - before)
        self.assertEqual(found, [{"kind": "valid", "frame_bits": 20}])
        self.assertLessEqual(max_calls_per_feed, 5)
        self.assertLessEqual(max_pending_candidates, scanner.MAX_FRAME_BITS // 18 + 16)

    def test_scanner_never_returns_to_an_abandoned_candidate(self):
        # Ones, a zero that completes a false sync, then more than MAX_FRAME_BITS that never parse: the scanner
        # used to step past the candidate and find the same sync again behind its position, forever.
        calls = 0

        def parse(bits):
            nonlocal calls
            calls += 1
            return None

        scanner = ctl._Scanner((parse,), e_bits=20, align_bits=10, e_to_run_end=True)
        rng = np.random.default_rng(9)
        stream = [1] * 27 + [0, 0, 0, 1] + [int(b) for b in rng.integers(0, 2, ctl._Scanner.MAX_FRAME_BITS + 200)]
        started = ctl.time.perf_counter()
        for start in range(0, len(stream), 160):
            scanner.feed(stream[start:start + 160])
        self.assertLess(ctl.time.perf_counter() - started, 2.0)
        self.assertLess(calls, 20000)

    def test_scanner_ends_e_where_the_zero_run_stops(self):
        def parse_frame(bits):
            if len(bits) >= 18 and bits[:18] == [1] * 17 + [0]:
                return {"kind": "test", "frame_bits": 18}
            return None

        frame = [1] * 17 + [0]                       # 18 bits, filled to 20
        for zeros, after, end in ((20, [1], 40), (50, [1], 70), (52, [1, 1], 70), (29, [0, 1], 50)):
            with self.subTest(zeros=zeros, after=after):
                scanner = ctl._Scanner((parse_frame,), e_bits=20, align_bits=10, e_to_run_end=True)
                scanner.feed(frame + [0, 0] + [0] * zeros)
                self.assertFalse(scanner.e_seen)     # the run has not stopped yet
                scanner.feed(after)
                self.assertEqual(scanner.e_end_bit, end)
        # The decoder stops at B1d's codewords, so no one arrives.
        scanner = ctl._Scanner((parse_frame,), e_bits=20, align_bits=10, e_to_run_end=True)
        scanner.feed(frame + [0, 0] + [0] * 30)
        scanner.end_of_stream()
        self.assertEqual(scanner.e_end_bit, 50)
        # A run that never stops ends E after E_RUN_MAX_FRAMES frames.
        scanner = ctl._Scanner((parse_frame,), e_bits=20, align_bits=10, e_to_run_end=True)
        scanner.feed(frame + [0, 0] + [0] * 200)
        self.assertEqual(scanner.e_end_bit, 20 + 10 * ctl._Scanner.E_RUN_MAX_FRAMES)
        # Fewer than e_bits zeros is still fill before another frame, and the default ends E at e_bits.
        scanner = ctl._Scanner((parse_frame,), e_bits=20, align_bits=10, e_to_run_end=True)
        self.assertEqual(len(scanner.feed(frame + [0, 0] + [0] * 8 + frame + [0, 0] + [0] * 22 + [1])), 2)
        self.assertEqual(scanner.e_end_bit, 70)
        scanner = ctl._Scanner((parse_frame,), e_bits=20, align_bits=10)
        scanner.feed(frame + [0, 0] + [0] * 50)
        self.assertEqual(scanner.e_end_bit, 40)

    def test_scanner_skips_false_training_sync_to_find_later_valid_frame(self):
        marker = [0, 1, 0, 1, 1, 0]

        def parse_frame(bits):
            if len(bits) >= 24 and bits[:18] == [1] * 17 + [0] and bits[18:24] == marker:
                return {"kind": "valid", "frame_bits": 24}
            return None

        scanner = ctl._Scanner((parse_frame,), e_bits=4, align_bits=1)
        training_false_sync = [1] * 17 + [0] + [0] * 64
        valid_frame = [1] * 17 + [0] + marker
        self.assertEqual(scanner.feed(training_false_sync + valid_frame),
                         [{"kind": "valid", "frame_bits": 24}])
        self.assertFalse(scanner.failed)

    def test_side_emits_silent_period_request_in_mapped_suv(self):
        for role, make, recover in (
                ("analogue", lambda law: ctl.analogue_side(profile(law), law),
                 lambda wire, law: up.recover_frames(wire, points=4, lu=2000.0)),
                ("digital", lambda law: ctl.digital_side(profile(law), law),
                 lambda wire, law: down.recover_downstream(wire, profile(law), law))):
            for law in ("pcmu", "pcma"):
                with self.subTest(role=role, law=law):
                    side = make(law)
                    side._trn_units = 0
                    side.set_silent_period_request(True)
                    first = side.transmit(160)
                    side.set_silent_period_request(False)
                    second = side.transmit(160)
                    wire = (np.concatenate((first, second)) if role == "analogue"
                            else bytes(first) + bytes(second))
                    frames = recover(wire, law)
                    self.assertTrue(frames, side.diagnostics())
                    self.assertIn(True, [frame["silent_period"] for frame in frames])
                    self.assertIn(False, [frame["silent_period"] for frame in frames])

    def test_phase4_controller_preserves_suv_silent_period_transitions(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                suvd = [v92_cp.build_suvd(silent_period=True),
                        v92_cp.build_suvd(silent_period=False)]
                analogue = ctl.analogue_side(profile(law), law)
                analogue.receive(down.build_downstream(suvd, profile(law), law,
                                                       trn_symbols=2040))
                self.assertEqual(analogue.silent_period_events,
                                 [("suvd", True), ("suvd", False)])
                self.assertFalse(analogue.silent_period_requested)

                suvu = [v92_cp.build_suvu(silent_period=True),
                        v92_cp.build_suvu(silent_period=False)]
                digital = ctl.digital_side(profile(law), law)
                digital.receive(up.build_burst(suvu, points=4, lu=2000.0,
                                               trn_symbols=1200))
                self.assertEqual(digital.silent_period_events,
                                 [("suvu", True), ("suvu", False)])
                self.assertFalse(digital.silent_period_requested)

    def test_analogue_silent_trn_terminates_on_fragmented_rt_and_clears_request(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                p = profile(law)
                side = ctl.analogue_side(p, law)
                side._silent_state = "analogue-trn"
                side._silent_active = True
                side.silent_period_tx = True
                side._silent_trn_generated = 2400
                rt = ctl.rt_t_pair_codewords(p, law)

                # Receive the raw R-t/R-bar-t codeword pair split across RTP-sized
                # chunks. The analogue modem must stop its second TRN2u at the
                # detected pair and return to SUVu with bit 32 clear.
                side.receive(rt[:137])
                self.assertEqual(side._silent_state, "analogue-trn")
                side.receive(rt[137:])
                self.assertIsNone(side._silent_state)
                self.assertEqual(side._silent_trn_generated, 2400)
                self.assertFalse(side.silent_period_tx)
                self.assertIn("SILENT:trn2u-terminated-by-rt", side.sent)
                self.assertIn("SILENT:rt-detected", side.sent)

                wire = side.transmit(160)
                frames = up.recover_frames(wire, points=4, lu=2000.0)
                self.assertTrue(frames, side.diagnostics())
                self.assertFalse(frames[0]["silent_period"])

    def test_sparse_cpd_diagnostic_uses_designer_and_clause_limits(self):
        default = ctl.digital_side(profile("pcmu"), "pcmu")
        sparse = ctl.digital_side(profile("pcmu"), "pcmu", cpd_constellation_mode="sparse")
        self.assertEqual(default.data_rx.rx_filter_mode, "off")
        self.assertEqual(default.diagnostic_metadata["cpd_constellation"], {"mode": "default"})

        parsed = v92_cp.parse_cpd(sparse._build_cp(False))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["drn"], 1)
        self.assertEqual(parsed["drn"], sparse.diagnostic_metadata["cpd_constellation"]["drn"])
        self.assertEqual(parsed["trellis"], 1)
        self.assertEqual(sparse.data_rx.rx_filter_mode, "off")
        self.assertEqual(sparse.diagnostic_metadata["cpd_constellation"]["mode"], "sparse")

        positive = parsed["constellations"][0]
        self.assertEqual(tuple(positive), tuple(sorted(set(positive))))
        self.assertLess(len(positive), 48)
        n = 2 * len(positive)
        moduli = parsed["moduli"]
        self.assertEqual(len(moduli), 12)
        for i, modulus in enumerate(moduli):
            if i % 4 == 3:
                self.assertLessEqual(2 * modulus, n)
            else:
                self.assertLessEqual(modulus, n)
        k = 2 * (parsed["drn"] + 17)
        product = 1
        for modulus in moduli:
            product *= modulus
        self.assertGreaterEqual(product, 1 << k)
        metadata = sparse.diagnostic_metadata["cpd_constellation"]
        self.assertEqual(metadata["levels_positive"], list(positive))
        self.assertEqual(metadata["moduli"], list(moduli))
        self.assertEqual(metadata["min_gap_sigmas"], 5.0)
        self.assertEqual(metadata["designer"], "v92_modulation_design.design")

    def test_digital_side_applies_configured_upstream_lost_frame_allowance(self):
        side = ctl.digital_side(profile("pcmu"), "pcmu", max_lost_frames=16)
        self.assertEqual(side.data_rx.rx._max_lost, 16)
        with self.assertRaisesRegex(ValueError, "max_lost_frames"):
            ctl.digital_side(profile("pcmu"), "pcmu", max_lost_frames=True)

    def test_live_data_transmit_keeps_downstream_mapper_frame_buffer_bounded(self):
        frame = 6
        side = _data_side(frame)
        for _ in range(1000):
            self.assertEqual(len(side.transmit(160)), 160)
        self.assertLess(len(side._units_out), frame)
        self.assertEqual(len(side._units_out), (-side.units_sent) % frame)

    def test_live_data_transmit_keeps_pcm_upstream_frame_buffer_bounded(self):
        frame = 12
        side = _data_side(frame)
        for _ in range(1000):
            self.assertEqual(len(side.transmit(160)), 160)
        self.assertLess(len(side._units_out), frame)
        self.assertEqual(len(side._units_out), (-side.units_sent) % frame)

    def test_analogue_trn_stops_at_the_configured_frame_count_at_multiple_read_sizes(self):
        for read_size in (7, 160, 1000, 4096):
            with self.subTest(read_size=read_size):
                side = ctl.analogue_side(profile("pcmu"), "pcmu", analogue_trn=12096)
                original_next = side._next_message
                first_suv_at = None

                def traced_next():
                    nonlocal first_suv_at
                    message = original_next()
                    if message[0] == "SUV" and first_suv_at is None:
                        first_suv_at = side.units_sent + len(side._units_out)
                    return message

                side._next_message = traced_next
                while first_suv_at is None:
                    side.transmit(read_size)

                self.assertEqual(side.tx_counts["TRN"], 1008)
                self.assertEqual(first_suv_at, 12096)

    def test_digital_trn_stops_at_the_configured_frame_count_at_multiple_read_sizes(self):
        for trn_symbols in (2040, 8000, 24000):
            for read_size in (7, 160, 1000, 4096):
                with self.subTest(trn_symbols=trn_symbols, read_size=read_size):
                    side = ctl.digital_side(profile("pcmu"), "pcmu", digital_trn=trn_symbols)
                    original_next = side._next_message
                    first_suv_at = None

                    def traced_next():
                        nonlocal first_suv_at
                        message = original_next()
                        if message[0] == "SUV" and first_suv_at is None:
                            first_suv_at = side.units_sent + len(side._units_out)
                        return message

                    side._next_message = traced_next
                    while first_suv_at is None:
                        side.transmit(read_size)

                    expected_trn = ((trn_symbols + 5) // 6) * 6
                    self.assertEqual(side.tx_counts["TRN"], expected_trn // 6)
                    self.assertEqual(first_suv_at, expected_trn)

    def test_live_data_transmit_can_drain_exactly_to_a_mapper_frame_boundary(self):
        for frame in (6, 12):
            with self.subTest(frame=frame):
                side = _data_side(frame)
                for _ in range(1000):
                    side.transmit(160)
                expected_tail = (-side.units_sent) % frame
                tail = side.transmit_to_frame_boundary(160, frame)
                self.assertEqual(len(tail), expected_tail)
                self.assertEqual(side.units_sent % frame, 0)
                self.assertEqual(side._units_out, [])

    def test_both_sides_connect_and_carry_a_payload_each_way(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 240, 800):
                with self.subTest(law=law, delay=delay):
                    a, d, t = run(law, delay=delay)
                    self.assertIsNotNone(
                        t, (a.sent[-4:], d.sent[-4:], a.frames_seen[-3:], d.frames_seen[-3:],
                            a.scanner.failed, d.scanner.failed))
                    self.assertTrue(a.connected and d.connected)
                    # Each side saw the other's SUVs and one CP; the digital CP' acknowledges, the
                    # analogue modem acknowledges only in its E.
                    self.assertIn("cpd", [k for k, _ in a.frames_seen])
                    self.assertIn("cpu", [k for k, _ in d.frames_seen])
                    self.assertTrue(a.saw_ack and not d.saw_ack)
                    # B1 in each side's data mode qualified, and the payloads are exact.
                    self.assertTrue(a.data_rx.qualified and d.data_rx.qualified)
                    self.assertEqual(a.data_rx.b1_errors, 0)
                    self.assertEqual(d.data_rx.b1_errors, 0)
                    self.assertEqual(d.data_rx.payload()[:len(UP)], UP)
                    self.assertEqual(a.data_rx.payload()[:len(DOWN)], DOWN)

    def test_digital_sends_g_of_one_over_lu_and_the_analogue_sends_g_lu_times_u(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                a, d, t = run(law)
                self.assertIsNotNone(t)
                gain4 = round(4 * 65536 / d._decoder.lu)
                self.assertEqual(d.lu_gain4, gain4)
                self.assertEqual(d.diagnostic_metadata["cpd_gain"]["gain4"], gain4)
                self.assertEqual(v92_cp.parse_cpd(d._build_cp(False))["gain"], gain4 / 65536 / 4)
                # analogue_side's LU is 2000: the line carries the CPd points times G*LU, about 1 (LU as
                # measured over the G.711-quantised TRN2u)
                self.assertAlmostEqual(a.upstream_scale, gain4 / 65536 / 4 * 2000.0)
                self.assertAlmostEqual(a.upstream_scale, 1.0, delta=0.03)

    def test_a_fixed_cpd_gain_is_sent_as_given(self):
        side = ctl.digital_side(profile("pcmu"), "pcmu", cpd_gain4=0x0300)
        self.assertIsNone(side.lu_gain4)
        self.assertEqual(v92_cp.parse_cpd(side._build_cp(False))["gain"], 0x0300 / 65536 / 4)
        with self.assertRaises(ValueError):
            ctl.digital_side(profile("pcmu"), "pcmu", cpd_gain4=0)

    def test_the_analogue_modem_cuts_its_training_short_when_it_hears_an_suv(self):
        a, d, t = run("pcmu", delay=240)
        self.assertLess(a.sent.count("TRN"), 1008)            # a full 12096T is 1008 12-symbol messages
        self.assertGreaterEqual(d.sent.count("TRN"), 2040 // 6)   # digital: at least 2040T

    def test_e_is_sent_once_and_followed_by_data(self):
        a, d, t = run("pcma", delay=240)
        for side in (a, d):
            self.assertEqual(side.sent.count("E"), 1)
            self.assertEqual(side.sent[side.sent.index("E") + 1], "DATA")

    def test_a_cp_is_repeated_when_the_acknowledge_is_slower_than_the_wait(self):
        a, d, t = run("pcmu", delay=800, ack_wait=200)
        self.assertIsNotNone(t)
        self.assertGreater(a.sent.count("CP"), 1)
        self.assertGreater(d.sent.count("CP"), 1)
        self.assertEqual(a.sent.count("E"), 1)
        self.assertEqual(d.sent.count("E"), 1)
        self.assertEqual(d.data_rx.payload()[:len(UP)], UP)
        self.assertEqual(a.data_rx.payload()[:len(DOWN)], DOWN)


class DownstreamErasureTests(unittest.TestCase):
    def test_an_rtp_erasure_keeps_the_cadence_and_decoding_resumes(self):
        law = "pcmu"
        prof = dict(profile(law), kind="cp")
        encoder = ctl.dp4.V90DigitalPhase4Encoder(prof, law)
        fb = encoder.s + encoder.k
        rng = np.random.default_rng(3)
        source = [1] * (ctl.B1_FRAMES * fb) + [int(b) for b in rng.integers(0, 2, 60 * fb)]
        codewords = encoder.encode_bits(source)
        silence = bytes([0xFF]) * 162           # one 20 ms packet concealed as silence: 27 frames
        rx = ctl._DownstreamDataRx(prof, law)
        cut = (ctl.B1_FRAMES + 10) * 6
        rx.feed(codewords[:cut])
        rx.feed(silence)
        rx.feed(codewords[cut + 162:])
        self.assertFalse(rx.ended)
        self.assertEqual(rx.erased_frames, 27)
        self.assertEqual(len(rx.bits), len(source))
        # The descrambler needs its 23-bit span (plus one sign bit) after the gap: 3 frames of 10 bits.
        resume = (ctl.B1_FRAMES + 10 + 27) * fb + 3 * fb
        self.assertEqual(rx.bits[resume:], source[resume:])

    def test_a_1200ms_outage_is_ridden_out(self):
        # A 400 ms ZeroTier outage ended a relay call at the old 300 ms limit (2026-11-19).
        law = "pcmu"
        prof = dict(profile(law), kind="cp")
        encoder = ctl.dp4.V90DigitalPhase4Encoder(prof, law)
        fb = encoder.s + encoder.k
        rng = np.random.default_rng(4)
        source = [1] * (ctl.B1_FRAMES * fb) + [int(b) for b in rng.integers(0, 2, 2000 * fb)]
        codewords = encoder.encode_bits(source)
        gap = 9600                               # 1.2 s concealed as silence: 1600 frames
        rx = ctl._DownstreamDataRx(prof, law)
        cut = (ctl.B1_FRAMES + 10) * 6
        rx.feed(codewords[:cut])
        rx.feed(bytes([0xFF]) * gap)
        rx.feed(codewords[cut + gap:])
        self.assertFalse(rx.ended)
        self.assertEqual(rx.erased_frames, gap // 6)
        resume = (ctl.B1_FRAMES + 10 + gap // 6) * fb + 3 * fb
        self.assertEqual(rx.bits[resume:], source[resume:])

    def test_a_lone_leading_frame_before_b1d_is_dropped(self):
        law = "pcmu"
        prof = dict(profile(law), kind="cp")
        encoder = ctl.dp4.V90DigitalPhase4Encoder(prof, law)
        fb = encoder.s + encoder.k
        payload = [int(b) for b in np.random.default_rng(5).integers(0, 2, 8 * fb)]
        for name, lead, expected_lead in (("zeros", [0] * fb, 1), ("ones", [1] * fb, 0), ("none", [], 0)):
            with self.subTest(lead=name):
                encoder.reset()
                rx = ctl._DownstreamDataRx(prof, law)
                rx.feed(encoder.encode_bits(lead + [1] * (ctl.B1_FRAMES * fb) + payload))
                self.assertEqual(rx.lead_frames, expected_lead)
                self.assertTrue(rx.qualified)
                # A leading frame of ones is simply B1d's first frame; one more B1d frame then precedes the data.
                self.assertEqual(rx.take_bits(), [1] * fb + payload if name == "ones" else payload)

    def test_a_bad_first_b1d_frame_is_not_skipped_when_the_next_is_bad_too(self):
        law = "pcmu"
        prof = dict(profile(law), kind="cp")
        encoder = ctl.dp4.V90DigitalPhase4Encoder(prof, law)
        fb = encoder.s + encoder.k
        encoder.reset()
        rx = ctl._DownstreamDataRx(prof, law)
        rx.feed(encoder.encode_bits([0] * (2 * fb) + [1] * (ctl.B1_FRAMES * fb)))
        self.assertEqual(rx.lead_frames, 0)
        self.assertEqual(rx.b1_errors, 2 * fb)
        self.assertFalse(rx.qualified)

    def test_a_peer_that_stops_sending_data_still_ends(self):
        law = "pcmu"
        prof = dict(profile(law), kind="cp")
        encoder = ctl.dp4.V90DigitalPhase4Encoder(prof, law)
        fb = encoder.s + encoder.k
        rx = ctl._DownstreamDataRx(prof, law)
        rx.feed(encoder.encode_bits([1] * (ctl.B1_FRAMES * fb)))
        rx.feed(bytes([0xFF]) * 6 * (ctl._DownstreamDataRx.MAX_ERASED_FRAMES + 1))
        self.assertTrue(rx.ended)
        self.assertEqual(len(rx.bits), ctl.B1_FRAMES * fb)


class DownstreamConstellationTests(unittest.TestCase):
    def test_drn_10_and_below_keep_the_16_point_blocks(self):
        for drn in (1, 2, 10):
            masks, sets = ctl.downstream_cpu_constellations(drn, "pcmu")
            self.assertEqual(masks, [tuple(range(16 + 16 * i, 32 + 16 * i)) for i in range(6)])
            self.assertEqual(sets, (0, 1, 2, 3, 4, 5))

    def test_higher_drn_constellations_carry_the_rate_within_the_power_limit(self):
        from v90_dil import g711_ucode_codewords
        for law in ("pcmu", "pcma"):
            levels = np.abs(v34_g711.decode(np.asarray(g711_ucode_codewords(law), dtype=np.uint8), law).astype(float))
            distances = {}
            for drn in range(11, 23):
                with self.subTest(law=law, drn=drn):
                    masks, sets = ctl.downstream_cpu_constellations(drn, law)
                    self.assertEqual(sets, (0,) * 6)
                    (points,) = masks
                    self.assertGreaterEqual(len(points) ** 6, 2 ** (drn + 14))
                    self.assertNotIn(0, points)
                    self.assertLessEqual(max(points), 127)
                    self.assertEqual(list(points), sorted(points))
                    rms = np.sqrt(np.mean(levels[list(points)] ** 2))
                    self.assertLessEqual(20 * np.log10(rms / 32768) + 3.14, ctl._DOWNSTREAM_MAX_DBM0 + 0.05)
                    distances[drn] = min([2 * levels[points[0]]] + list(np.diff(levels[list(points)])))
                    # The encoder and receiver accept it.
                    profile = ctl.dm.downstream_profile_from_cpu(ctl.v92_cp.parse_cpu(ctl.v92_cp.build_cpu(
                        drn=drn, constellations=masks, interval_sets=sets, alaw=(law == "pcma"))))
                    ctl.dp4.V90DigitalPhase4Encoder(profile, law)
                    ctl.rx4.V90DigitalPhase4Receiver(profile, law)
            # Spread across the allowed power, not packed at the lowest Ucodes (8 apart at DRN 22): SmartLink's
            # 54,667 constellation for Astound keeps 40 between points.
            self.assertGreaterEqual(distances[22], 64)
            self.assertGreaterEqual(min(distances.values()), 64)
            self.assertEqual(sorted(distances.values(), reverse=True), [distances[d] for d in range(11, 23)])

    def test_higher_drn_constellation_is_the_widest_spacing_under_the_power_limit(self):
        from v90_dil import g711_ucode_codewords
        levels = np.abs(v34_g711.decode(np.asarray(g711_ucode_codewords("pcmu"), dtype=np.uint8), "pcmu").astype(float))
        (points,) = ctl.downstream_cpu_constellations(22, "pcmu")[0]
        self.assertEqual((points[0], points[-1], len(points)), (4, 98, 64))
        # One step wider spacing from the same start would exceed the limit.
        wider = [points[0]]
        for u in range(points[0] + 1, 128):
            if levels[u] - levels[wider[-1]] > 64 and len(wider) < 64:
                wider.append(u)
        rms = np.sqrt(np.mean(levels[wider] ** 2))
        self.assertTrue(len(wider) < 64 or 20 * np.log10(rms / 32768) + 3.14 > ctl._DOWNSTREAM_MAX_DBM0)


class UpstreamRateMarginTests(unittest.TestCase):
    """Options for a digital modem that rates our upstream above what our data reaches (Astound, 2026-11-19)."""

    def test_pad_aware_samples_land_nearest_after_the_path_gain(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                gain = 1.19
                samples = np.linspace(-6000, 6000, 997)
                wire = np.asarray(ctl.pad_aware_samples(samples, gain, law))
                # Every output is an exact G.711 value, so our encoder sends the chosen code unchanged.
                np.testing.assert_array_equal(v34_g711.roundtrip(wire, law), wire)
                received = v34_g711.roundtrip(np.rint(wire * gain), law)
                plain = v34_g711.roundtrip(np.rint(v34_g711.roundtrip(np.rint(samples), law) * gain), law)
                target = samples * gain
                # No reachable received value is nearer the target than the one chosen.
                values = v34_g711.decode(np.arange(256, dtype=np.uint8), law).astype(float)
                reachable = np.unique(v34_g711.roundtrip(np.rint(values * gain), law))
                best = np.min(np.abs(reachable[None, :] - target[:, None]), axis=1)
                np.testing.assert_allclose(np.abs(received - target), best)
                self.assertLess(np.mean(np.abs(received - target)), np.mean(np.abs(plain - target)))

    def test_cpd_path_gain_comes_from_the_prefilter_main_tap(self):
        self.assertAlmostEqual(ctl.cpd_path_gain({"precoder": {"z2": (0.0, -0.84, 0.001)}}), 1 / 0.84)
        self.assertIsNone(ctl.cpd_path_gain({"precoder": {"z2": (0.995,)}}))
        self.assertIsNone(ctl.cpd_path_gain({}))

    def test_trn2u_noise_is_off_by_default_and_bounded(self):
        def trn(**kwargs):
            side = ctl.analogue_side(profile("pcmu"), "pcmu", lu=1000.0, **kwargs)
            return np.asarray(side.transmit(1200))
        plain, quiet = trn(), trn(trn2u_noise=0.0)
        np.testing.assert_array_equal(plain, quiet)
        noisy = trn(trn2u_noise=0.015)
        np.testing.assert_array_equal(noisy, trn(trn2u_noise=0.015))      # seeded
        noise = noisy - plain
        self.assertAlmostEqual(float(np.std(noise[plain != 0])) / 1000.0, 0.015, delta=0.003)
        with self.assertRaises(ValueError):
            trn(trn2u_noise=0.6)

    def test_pair_carries_data_with_both_options_on(self):
        make = ctl.analogue_side

        def with_options(prof, law, **kwargs):
            return make(prof, law, trn2u_noise=0.015, pad_aware_upstream=True, **kwargs)

        for law in ("pcmu", "pcma"):
            with self.subTest(law=law), mock.patch.object(ctl, "analogue_side", with_options):
                analogue, digital, done = run(law)
                self.assertIsNotNone(done)
                self.assertEqual(digital.data_rx.payload()[:len(UP)], UP)
                self.assertEqual(analogue.data_rx.payload()[:len(DOWN)], DOWN)


class EBoundaryCutTests(unittest.TestCase):
    def test_trn2u_fit_taps_are_opt_in_and_keep_delay_bound(self):
        with mock.patch.dict("os.environ", {"EL_MODEM_V92_TRN2U_FIT_TAPS": "96"}):
            side = ctl.digital_side(
                profile("pcmu", drn=1), "pcmu", digital_trn=2040, drn=1,
                keep_payload=True, async_upstream_equalizer=False,
                b1u_rx_mode="trn2u-fit-viterbi",
                cpd_constellation_mode="expanded10", cpd_gain4=0x0300)
        self.assertEqual(side._decoder.trn_fit_taps, 96)
        self.assertEqual(side._decoder.trn_fit_max_delay, 40)

    def test_trn2u_fit_rejects_g711_octets_instead_of_filtering_byte_values(self):
        side = ctl.digital_side(
            profile("pcmu"), "pcmu", b1u_rx_mode="trn2u-fit-viterbi")
        side.receive(bytes([0xFF] * 160))
        self.assertEqual(side.raw_in, 0)
        self.assertEqual(side.data_rx.rx_filter_diagnostic["state"], "failed")
        self.assertIn("linear PCM samples", side.data_rx.rx_filter_diagnostic["error"])
        self.assertIsNotNone(side.data_rx.rx.failure)

    def test_cut_diagnostics_are_safe_before_e_boundary(self):
        with mock.patch.dict("os.environ", {"EL_MODEM_V92_B1_CUT_DIAGNOSTICS": "1"}):
            side = ctl.digital_side(
                profile("pcmu", drn=1), "pcmu", digital_trn=2040, drn=1,
                keep_payload=True, async_upstream_equalizer=False,
                b1u_rx_mode="trn2u-fit-viterbi",
                cpd_constellation_mode="expanded10", cpd_gain4=0x0300)
            try:
                self.assertIsNone(side._b1u_cut_diagnostic)
                diagnostics = side.diagnostics()
                self.assertIsNone(diagnostics["b1u_cut_diagnostic"])
                self.assertEqual(side.data_mode_rx, False)
            finally:
                close = getattr(side.data_rx, "close", None)
                if callable(close):
                    close()

    def test_exclusive_e_end_recovers_cross_chunk_raw_history_and_keeps_fit_delay(self):
        scanner = ctl._Scanner((), e_bits=3, align_bits=1)
        scanner.mode = "tail"
        scanner.feed([0, 0, 0])
        self.assertEqual(scanner.e_end_bit, 3)  # pos advances past the final E bit

        class Decoder:
            start_raw = 1009
            eq_delay = 30

            @staticmethod
            def raw_of_bit(bit_index):
                return 1009 + int(bit_index) // 2

        bit_index, raw_cut, cut = ctl._e_boundary_cut(
            Decoder(), 0, 52536, base=27289, rx_filter_delay=0)
        self.assertEqual((bit_index, raw_cut, cut), (52536, 27277, -12))

        current = np.arange(27289, 27449, dtype=float)
        prior = np.arange(27242, 27289, dtype=float)  # 47-sample FIR prehistory
        handed, first_sample, used_history = ctl._units_from_raw_cut(
            current, 27289, cut, (np.concatenate((prior, current)), 27242))
        self.assertTrue(used_history)
        self.assertEqual(first_sample, 27277)
        np.testing.assert_array_equal(handed, np.arange(27277, 27449, dtype=float))
        clipped, clipped_first, used_history = ctl._units_from_raw_cut(current, 27289, cut)
        self.assertFalse(used_history)
        self.assertEqual(clipped_first, 27289)
        np.testing.assert_array_equal(clipped, current)

        # The fitted FIR delay is handled after the raw cut by the receiver's
        # output drop; it must not shift the E-to-B1 raw boundary itself.
        side = ctl.digital_side(profile("pcmu"), "pcmu")
        parsed = v92_cp.parse_cpd(side._build_cp(False))
        rx = ctl._UpstreamDataRx(
            pu.profile_from_cpd(parsed), "pcmu", align_search=0,
            rx_filter_mode="trn2u-fit", fitted_fractional_delay_samples=0.6554263218584089)
        filt = mock.Mock()
        filt.w = np.zeros(48, dtype=float)
        filt._history = np.zeros(47, dtype=float)
        rx.install_trn2u_filter(
            filt, delay=Decoder.eq_delay, gain=1.0, target_rms=1.0,
            training_target_rms=1.0, fit_residual=0.07059,
            fractional_delay_samples=0.6554263218584089)
        self.assertEqual(rx._trn2u_rx_drop, 30)
        self.assertEqual(rx.rx_filter_diagnostic["filter_history_samples"], 47)
        self.assertEqual(rx.alignment, 0)
        self.assertIsNone(rx._calibrate_gain)


if __name__ == "__main__":
    unittest.main()
