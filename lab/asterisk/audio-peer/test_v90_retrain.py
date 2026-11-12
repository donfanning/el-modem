"""V.90 retrain tone handshake (clause 9.5), sample-level, both G.711 laws."""

import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_g711
    import v90_retrain as rt


def run_pair(initiator: str, *, law: str = "pcmu", delay: int = 0, max_samples: int = 16000):
    parties = {"digital": rt.RetrainPreamble("digital", initiating=initiator == "digital", peak=800),
               "analogue": rt.RetrainPreamble("analogue", initiating=initiator == "analogue", peak=1200)}
    lines = {"digital": np.zeros(0), "analogue": np.zeros(0)}
    queue = {"digital": [], "analogue": []}
    for start in range(0, max_samples, 160):
        out = {name: p.render(160) for name, p in parties.items()}
        for name, peer in (("digital", "analogue"), ("analogue", "digital")):
            wire = v34_g711.roundtrip(out[name], law)
            queue[peer].append(wire)
        for name, p in parties.items():
            block = queue[name].pop(0) if len(queue[name]) > delay // 160 else np.zeros(160)
            p.feed(block)
        if all(p.phase2_start_at is not None for p in parties.values()):
            return parties, start + 160
    return parties, None


@unittest.skipIf(np is None, "numpy not installed")
class RetrainTests(unittest.TestCase):
    def test_either_modem_can_initiate_under_both_laws(self):
        for initiator in ("digital", "analogue"):
            for law in ("pcmu", "pcma"):
                parties, done = run_pair(initiator, law=law)
                self.assertIsNotNone(done, (initiator, law))
                for p in parties.values():
                    self.assertTrue(p.ready)

    def test_initiator_sends_70_ms_of_silence_before_its_tone(self):
        parties, _ = run_pair("digital")
        # The initiator's tone starts at 560 samples; nothing is sent before that.
        p = rt.RetrainPreamble("digital", initiating=True, peak=800)
        first = p.render(560)
        self.assertEqual(float(np.max(np.abs(first))), 0.0)
        self.assertGreater(float(np.max(np.abs(p.render(160)))), 700.0)

    def test_responder_waits_for_more_than_50_ms_then_sends_70_ms_of_silence(self):
        parties, _ = run_pair("digital")
        responder = parties["analogue"]
        heard = next(at for at, name in responder.events if name == "peer tone")
        responding = next(at for at, name in responder.events if name == "responding")
        # Heard no sooner than 560 (initiator silence) + 400 (50 ms) samples into the call.
        self.assertGreaterEqual(heard, 560 + rt.DETECT_SAMPLES)
        # The responder's own tone starts 560 samples after it responded.
        self.assertEqual(responder._tone_from - responding, rt.SILENCE_SAMPLES)

    def test_a_short_tone_noise_and_the_wrong_frequency_do_not_trigger(self):
        p = rt.RetrainPreamble("digital", initiating=False)
        rng = np.random.default_rng(3)
        n = np.arange(8000)
        p.feed(rng.normal(0, 300, 4000))                                      # noise
        p.feed(500 * np.cos(2 * np.pi * 1200.0 * n[:3200] / 8000))           # wrong tone (own)
        p.feed(500 * np.cos(2 * np.pi * 2400.0 * n[:320] / 8000))            # 40 ms of Tone A
        p.feed(np.zeros(1600))
        self.assertIsNone(p.peer_heard_at)
        p.feed(500 * np.cos(2 * np.pi * 2400.0 * n[:800] / 8000))            # 100 ms
        self.assertIsNotNone(p.peer_heard_at)

    def test_a_return_delay_only_shifts_the_handshake(self):
        base, base_done = run_pair("analogue")
        late, late_done = run_pair("analogue", delay=1600)
        self.assertIsNotNone(late_done)
        self.assertGreater(late_done, base_done)

    def test_digital_initiator_continues_from_a_reversal_without_new_phase1(self):
        parties, _ = run_pair("digital", max_samples=24000)
        for p in parties.values():
            self.assertEqual(p.tone_ends_at - p.ready_at, rt.HOLD_SAMPLES)
        # The digital initiator resumes at the detected Tone A reversal;
        # §9.5.1.1 sends Tone B reversal 40 ± 1 ms after that boundary.
        digital = parties["digital"]
        self.assertEqual(digital.phase2_start_at, digital.peer_ended_tx)
        # The analogue responder still performs its response silence before
        # entering its own continuation.
        analogue = parties["analogue"]
        self.assertEqual(
            analogue.phase2_start_at,
            max(analogue.tone_ends_at, analogue.peer_ended_tx)
            + rt.PHASE1_SILENCE_SAMPLES)
        # The responder is ready first; the initiator must still hear its tone for 50 ms.
        digital, analogue = parties["digital"], parties["analogue"]
        self.assertLess(analogue.ready_at, digital.ready_at)
        # At zero path delay the responder may end its Tone A in the same
        # sample that the initiating side confirms the phase reversal; the
        # detector's >50 ms qualification has already completed then.
        self.assertGreaterEqual(analogue.tone_ends_at, digital.peer_heard_at)

    def test_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            rt.RetrainPreamble("modem", initiating=True)
        with self.assertRaises(ValueError):
            rt.RetrainPreamble("digital", initiating=True, peak=0)
        with self.assertRaises(ValueError):
            rt.RetrainPreamble("digital", initiating=True).feed([float("nan")])



# -- the production digital engine retraining against a caller ----------------------------------
if np is not None:
    import time

    import v34_call
    import v90_info_reference as v90
    from test_v90_analogue_caller_phase2 import _delayed_echo
    from v8 import V8_PCM_DIGITAL
    from v90_analogue_caller_phase2 import V90AnalogueCallerPhase2
    from v90_analogue_phase2_rx import V90AnaloguePhase2Adapter
    from v90_analogue_phase2_tx import V90AnaloguePhase2Transmitter
    from v90_digital_phase2_engine import V90DigitalAnswererPhase2Engine


LAST_HISTORIES = None


def _caller(start_at=0, begin=None):
    """Fresh caller Phase 2: transmitter clock at ``start_at`` (the loop time it first renders),
    first action at ``begin`` (defaults to ``start_at``)."""
    begin = start_at if begin is None else begin
    controller = V90AnalogueCallerPhase2()
    adapter = V90AnaloguePhase2Adapter(controller, rx_to_media=lambda sample: sample)
    tx = V90AnaloguePhase2Transmitter(
        info0a_fields={"rate3429": 1, "disallow3429": 1},
        info1a_builder=lambda mode: v90.build_info1a_v90(
            md_len=0, uinfo=90, analog_symbol_rate=5, digital_symbol_rate=6, freq_offset=0),
        probe_builder=v34_call.probing_signal, nominal_peak=1200, probe_rms=300,
        start_at=start_at)
    tx.submit(adapter.start(begin))
    return controller, adapter, tx


def _engine_phase2_start(engine):
    """Global media sample at which the engine's fresh Phase 2 begins, once it is known."""
    for name, detail in reversed(engine.events):
        if name == "retrain-phase2-restart":
            return detail["at"]
    pre = engine._retrain
    if pre is None or pre.phase2_start_at is None:
        return None
    started = next(d["at"] for n, d in reversed(engine.events) if n == "retrain-start")
    return started + -(-pre.phase2_start_at // 160) * 160


def run_engine_retrain(law: str, initiator: str, *, retrain_at: int = 6 * 8000,
                       total: int = 26 * 8000, initial_skew: int = 0, align: bool = True,
                       one_way_delay: int = 240):
    """Phase 2 against the engine, then a retrain started by ``initiator``, then Phase 2 again."""
    media_origin = time.monotonic()
    engine = V90DigitalAnswererPhase2Engine(
        role="digital-answerer", negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
        media_at=0, media_origin=media_origin, law=law, info0d_fields={"rate3429": 1},
        info1d_fields={"blocks": [{"hi_carrier": 0} for _ in range(6)], "freq_offset": 0},
        tone_b_peak=800, probe_rms=300, probe_builder=v34_call.probing_signal,
        phase3_rate_mask=0xFFFF, phase3_enable_lapm=True)
    stage = "phase2"
    caller, adapter, tx = _caller(0, initial_skew)
    pre = None
    info0a_done = False
    tx_hist = np.zeros(total + 160)
    eng_hist = np.zeros_like(tx_hist)
    global LAST_HISTORIES
    LAST_HISTORIES = (tx_hist, eng_hist)       # for diagnostics only
    first_done_at = None
    for start in range(0, total, 160):
        now = start + 160
        if stage in ("retrain", "listen"):
            if stage == "listen" and start >= begin:       # decided before rendering this block
                stage = "phase2b"
                caller_pcm = tx.render(160)
            else:
                caller_pcm = pre.render(160)
        else:
            caller_pcm = tx.render(160)
        wire = engine.tx_g711(160, law)
        eng_pcm = v34_g711.decode(np.frombuffer(wire, dtype=np.uint8), law).astype(float)
        tx_hist[start:start + 160] = caller_pcm
        eng_hist[start:start + 160] = eng_pcm
        # Each direction takes ``one_way_delay`` samples (240 = 30 ms, a round trip of 60 ms);
        # with none, the measured round-trip delay is about zero and a one-sample jitter makes it
        # negative, which the caller's 40 ms turnaround rule rejects.
        caller_rx_pcm = (_delayed_echo(eng_hist, start, 160, one_way_delay, 1.0)
                         + _delayed_echo(tx_hist, start, 160))
        engine_rx_pcm = (_delayed_echo(tx_hist, start, 160, one_way_delay, 1.0)
                         + _delayed_echo(eng_hist, start, 160))
        engine.feed_g711(bytes(v34_g711.encode(engine_rx_pcm, law)), law, {
            "packet_playout_deadline_monotonic": media_origin + start / 8000,
            "payload_samples": 160, "timestamp": start})
        if stage == "retrain":
            pre.feed(caller_rx_pcm)
            planned = _engine_phase2_start(engine) if align else None
            if pre.receive_ready and pre.phase2_start_at is not None and (
                    planned is not None or not align):
                # Packets are 160 samples, so the switch happens on the first block boundary at
                # or after the planned start; the transmitter clock must start exactly there.
                # ``align`` starts the caller in the same packet as the engine: Phase 2 only
                # tolerates a small start offset (see the notes), and this test is about the
                # engine's retrain logic, not that tolerance.
                begin = (planned if align else
                         retrain_base + -(-pre.phase2_start_at // 160) * 160)
                caller, adapter, tx = _caller(begin, begin)
                stage, info0a_done = "listen", False
        elif stage == "listen":
            pre.feed(caller_rx_pcm)
            adapter.feed(caller_rx_pcm, rx_at=start, tx_now=now)
        else:
            done = tx._completion.get("info0a-complete")
            if not info0a_done and done is not None and done <= now:
                info0a_done = True
                tx.submit(adapter.feed_tx_complete("info0a_complete", media_at=done, tx_now=now))
            for action in adapter.feed(caller_rx_pcm, rx_at=start, tx_now=now):
                tx.submit([action])
            for action in adapter.tick(media_now=now):
                tx.submit([action])
        if stage == "phase2" and first_done_at is None and caller.done and engine._phase2.done:
            first_done_at = now
        if stage == "phase2" and first_done_at is not None and now >= retrain_at:
            stage = "retrain"
            retrain_base = now          # the handshake object counts from zero at this loop time
            pre = rt.RetrainPreamble("analogue", initiating=initiator == "analogue", peak=1200)
            if initiator == "digital":
                assert engine.request_retrain("test")
        if stage == "phase2b" and caller.done and engine._phase2.done and engine.retrain_count:
            break
    return engine, caller, pre, first_done_at


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skip(
    "retired synthetic driver restarts with fresh INFO0a; production paired "
    "continuation through the production analogue-caller fallback retrain is outside the curated V.90 slice")
class EngineRetrainTests(unittest.TestCase):
    """The production digital engine retraining against an analogue caller, 30 ms each way.

    The harness needs a propagation delay: with none, the measured round-trip delay is about
    zero and one sample of jitter makes it negative, which the caller's 40 ms turnaround rule
    rejects (this made the first versions of these tests fail). Starts are tested both aligned
    to the engine's packet and left to the caller's own handshake plan."""

    def run_case(self, initiator, law, align):
        engine, caller, pre, first = run_engine_retrain(law, initiator, align=align,
                                                        total=24 * 8000)
        self.addCleanup(engine.close)
        self.assertIsNotNone(first, "the first Phase 2 never completed")
        return engine, caller

    def check(self, engine, caller, *, initiating, reason):
        events = {}
        for name, detail in engine.events:
            events.setdefault(name, detail)
        self.assertEqual(engine.retrain_count, 1)
        self.assertEqual(events["retrain-start"]["initiating"], initiating)
        self.assertEqual(events["retrain-start"]["reason"], reason)
        self.assertIn("retrain-phase2-restart", events)
        self.assertFalse(engine.retraining)
        self.assertFalse(engine._failed)
        self.assertEqual(len(engine._retired_phase3), 1)       # the old data link was retired
        names = [name for name, _ in engine.events]
        self.assertEqual(names.count("phase3-handoff"), 2, names)
        self.assertEqual(engine._phase2.state, engine._phase2.COMPLETE)
        self.assertEqual(caller.state, caller.COMPLETE)
        self.assertEqual(caller.mode, 6)

    def test_engine_initiated_retrain_completes_a_second_phase_2(self):
        for law in ("pcmu", "pcma"):
            for align in (True, False):
                with self.subTest(law=law, align=align):
                    engine, caller = self.run_case("digital", law, align)
                    self.check(engine, caller, initiating=True, reason="test")

    def test_engine_answers_a_peer_retrain_heard_in_data_mode(self):
        for law in ("pcmu", "pcma"):
            for align in (True, False):
                with self.subTest(law=law, align=align):
                    engine, caller = self.run_case("analogue", law, align)
                    self.check(engine, caller, initiating=False, reason="peer tone")

    def test_peer_starting_before_the_engine_is_not_missed(self):
        # The peer's INFO0a is sent once; the engine's receive side is built as soon as the
        # peer's tone is gone, not at its own Phase 2 start, so an early INFO0a is still seen.
        engine, caller = self.run_case("analogue", "pcmu", align=False)
        self.check(engine, caller, initiating=False, reason="peer tone")
        self.assertIsNone(engine._phase2_prebuilt_at)


@unittest.skipIf(np is None, "numpy not installed")
class RetrainTriggerTests(unittest.TestCase):
    """retrain_on_failure: the engine starts a retrain itself when the data link dies."""

    def engine(self, **kw):
        engine = V90DigitalAnswererPhase2Engine(
            role="digital-answerer", negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=time.monotonic(), law="pcmu",
            info0d_fields={"rate3429": 1},
            info1d_fields={"blocks": [{"hi_carrier": 0} for _ in range(6)], "freq_offset": 0},
            tone_b_peak=800, probe_rms=300, probe_builder=v34_call.probing_signal,
            phase3_rate_mask=0xFFFF, phase3_enable_lapm=True, **kw)
        self.addCleanup(engine.close)
        return engine

    @staticmethod
    def stub(*, state="running", lost=False):
        import types
        lapm = types.SimpleNamespace(v42_lost=lost, v42_mode="lost" if lost else "lapm")
        return types.SimpleNamespace(state=state, transmit_ready=False, lapm=lapm)

    def test_a_failed_data_session_starts_a_retrain_that_sends_tone_b(self):
        engine = self.engine(retrain_on_failure=True)
        engine._phase3 = self.stub(state="failed")
        out = b"".join(engine.tx_g711(160, "pcmu") for _ in range(8))
        start = next(d for n, d in engine.events if n == "retrain-start")
        self.assertEqual((start["initiating"], start["reason"]), (True, "data link failed"))
        self.assertTrue(engine.retraining)
        self.assertEqual(engine.retrain_count, 1)
        self.assertFalse(engine._failed)
        self.assertIsNone(engine._phase3)
        self.assertEqual(len(engine._retired_phase3), 1)
        self.assertNotEqual(out[-160:], bytes((0xFF,)) * 160)       # Tone B after the silence

    def test_failure_retrain_snapshots_lapm_state_before_retiring_phase3(self):
        import types
        engine = self.engine(retrain_on_failure=True)
        stats = {"t401_expiries": 11, "i_tx": 12, "i_rx": 3,
                 "i_resent": 10, "fcs_errors": 2}
        lapm = types.SimpleNamespace(
            state="down", stats=stats, _t401_bits=12000,
            _tries=11, bits_sent=132000, rate=64000,
            va=7, vs=9, vr=5, _unacked=[[7, b"x"]], _retx=[],
            _poll_out=True, _ack_pending=False, _peer_busy=False,
            events=["poll", "retransmit", "N400 retransmissions exhausted"])
        link = types.SimpleNamespace(v42_mode="lost", v42_lost=True, _v42=lapm)
        timing = {"symbol": 12345, "slips": 0, "timing_velocity": 1.0}
        eq = types.SimpleNamespace(timing_snapshot=lambda: timing)
        upstream = types.SimpleNamespace(
            state="tracking", failure_reason=None, samples_seen=100000,
            receiver=types.SimpleNamespace(eq=eq))
        upstream_data = types.SimpleNamespace(
            state="data", failure_reason=None, next_symbol=12345)
        pcm_rx = types.SimpleNamespace(
            state="data", frames=75, lost_frames=0, consecutive_lost_frames=0,
            max_consecutive_lost_frames=0,
            b1_errors=0,
            trellis_violations=0, failure=None, _pending=[1.0, 2.0])
        p4 = types.SimpleNamespace(data_rx=types.SimpleNamespace(rx=pcm_rx))
        engine._phase3 = types.SimpleNamespace(
            state="running", stage="data", lapm=link,
            upstream=upstream, upstream_data=upstream_data, _p4=p4)

        engine._begin_retrain(initiating=True, reason="data link failed")

        start = next(detail for name, detail in engine.events if name == "retrain-start")
        self.assertEqual(start["lapm"], {
            "mode": "lost", "lost": True, "state": "down",
            "stats": stats,
            "t401_expiries": 11, "t401_bits": 12000, "tries": 11,
            "bits_sent": 132000, "rate": 64000,
            "va": 7, "vs": 9, "vr": 5, "unacked_frames": 1,
            "retransmit_queue_frames": 0, "poll_out": True,
            "ack_pending": False, "peer_busy": False,
            "last_events": ["poll", "retransmit", "N400 retransmissions exhausted"],
        })
        self.assertEqual(start["phase3"], {
            "state": "running", "stage": "data",
            "upstream_phase3_state": "tracking",
            "upstream_failure_reason": None, "upstream_samples_seen": 100000,
            "upstream_acquisition": None, "upstream_training_mse": None,
            "upstream_detected_trn_role": None,
            "upstream_startup_trace": None,
            "upstream_data_state": "data",
            "upstream_data_failure_reason": None, "upstream_next_symbol": 12345,
            "phase4_error": None, "phase4_state": None,
            "phase4_tx_state": None, "phase4_events": [],
            "phase4_tx_output_octets": None,
            "phase4_initial_ri_remaining": None,
            "phase4_mp_frames_sent": None,
            "phase4_mp_prime_frames_sent": None,
            "phase4_rx_bits_seen": None, "phase4_profiles": [],
            "phase4_receive_armed": False, "phase4_cpt_acquired": False,
            "timing": timing,
            "pcm_upstream_rx": {
                "state": "data", "frames": 75, "lost_frames": 0,
                "consecutive_lost_frames": 0,
                "max_consecutive_lost_frames": 0,
                "b1_errors": 0, "trellis_violations": 0,
                "failure": None, "pending_symbols": 2,
            },
        })
        self.assertIsNone(engine._phase3)
        self.assertEqual(engine._retired_phase3[0].lapm, link)

    def test_a_gap_in_received_audio_during_a_retrain_does_not_fail_the_engine(self):
        engine = self.engine(retrain_on_failure=True)
        engine.request_retrain("test")
        engine.tx_g711(160, "pcmu")
        self.assertTrue(engine.retraining)
        engine.feed_gap(8160, "pcmu", {})
        self.assertTrue(engine.retraining)
        self.assertFalse(engine._failed)
        self.assertIn("retrain-rx-gap", [name for name, _ in engine.events])
        engine.tx_g711(160, "pcmu")                      # still transmitting the handshake

    def test_a_gap_outside_a_retrain_still_fails_phase_2(self):
        engine = self.engine(retrain_on_failure=True)
        engine.feed_gap(8160, "pcmu", {})
        self.assertTrue(engine._failed)

    def test_a_connected_data_session_is_not_retrained_for_taking_long(self):
        import types
        engine = self.engine(retrain_on_failure=True)
        engine._phase3 = types.SimpleNamespace(
            state="running", transmit_ready=False, lapm=None,
            upstream_data=types.SimpleNamespace(state="data"))
        engine._phase3_started_at = 0
        for _ in range(25 * 8000 // 160):
            engine.tx_g711(160, "pcmu")
        self.assertFalse(engine.retraining)
        self.assertEqual(engine.retrain_count, 0)

    def test_off_by_default(self):
        engine = self.engine()
        engine._phase3 = self.stub(state="failed")
        engine.tx_g711(160, "pcmu")
        self.assertFalse(engine.retraining)
        self.assertEqual(engine.retrain_count, 0)

    def test_a_lost_lapm_link_is_hidden_from_the_host_while_a_retrain_is_possible(self):
        engine = self.engine(retrain_on_failure=True)
        engine._phase3 = self.stub(lost=True)
        self.assertFalse(engine.v42_lost)          # the host would hang up if this were True
        engine.tx_g711(160, "pcmu")
        self.assertTrue(engine.retraining)
        plain = self.engine()
        plain._phase3 = self.stub(lost=True)
        self.assertTrue(plain.v42_lost)

    def test_the_retrain_limit_ends_the_call(self):
        engine = self.engine(retrain_on_failure=True, max_retrains=1)
        engine._phase3 = self.stub(lost=True)
        engine.tx_g711(160, "pcmu")
        self.assertEqual(engine.retrain_count, 1)
        engine._finish_retrain()                    # as if the retrain had completed
        engine._phase3 = self.stub(lost=True)       # and the new link died too
        self.assertTrue(engine.v42_lost)            # now the host may hang up
        engine.tx_g711(160, "pcmu")
        self.assertEqual(engine.retrain_count, 1)
        self.assertFalse(engine.retraining)

    def test_a_peer_that_never_answers_ends_the_call_after_20_seconds(self):
        engine = self.engine(retrain_on_failure=True)
        engine._phase3 = self.stub(state="failed")
        for _ in range(20 * 8000 // 160 + 3):
            engine.tx_g711(160, "pcmu")
        self.assertIn("retrain-timeout", [n for n, _ in engine.events])
        self.assertFalse(engine.retraining)
        self.assertTrue(engine._failed)
        self.assertEqual(engine.tx_g711(160, "pcmu"), bytes((0xFF,)) * 160)

    def test_bad_settings_are_rejected(self):
        for kw in ({"retrain_on_failure": 1}, {"max_retrains": 0}, {"max_retrains": 6},
                   {"max_retrains": True}):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.engine(**kw)


if __name__ == "__main__":
    unittest.main()
