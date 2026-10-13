"""Real PCM through both DCEs, preserving data across section 11.6.

Local functional tests, not independent-peer or realtime qualification.
"""
import random
import hashlib
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from v34_modem import V34Modem
from v34_answer_modem import V34AnswerModem


class ConnectedRateTests(unittest.TestCase):
    @patch.dict('os.environ', {'V34_CONNECTED_RATE':'1', 'V34_PHASE4_EQ_SPAN':'24',
                              'V34_NATIVE_TRELLIS':'1', 'V34_CALLER_RETRAIN':'1',
                              'V34_ANSWERER_RETRAIN':'1'})
    def exercise(self, lapm=False, first_role='call', capture_dir=None, retrain_cycle=False, failure_role=None, legacy_retrain=True, online_commands=False, lower_resync=False, directional=False, fallback_rates=None):
        if not legacy_retrain:
            import os
            os.environ['V34_CALLER_RETRAIN'] = os.environ['V34_ANSWERER_RETRAIN'] = '0'
        c, a = V34Modem(data_rate=33600), V34AnswerModem(data_rate=33600)
        cr, ar = bytearray(), bytearray()
        quality_epochs = {'call': [], 'answer': []}
        capture = None
        if capture_dir is not None:
            from v34_diagnostic import DspCapture
            capture = DspCapture(Path(capture_dir)/'caller', c)
            self.addCleanup(capture.close)

        def tick():
            ct, at = c.tx_samples(160), a.tx_samples(160)
            ar.extend(a.feed_pcm(struct.pack('<160h', *ct)))
            pcm = struct.pack('<160h', *at)
            cr.extend(capture.feed(pcm, c) if capture is not None else c.feed_pcm(pcm))
            self.assertNotEqual(c.state, 'failed', c.events[-8:])
            self.assertNotEqual(a.state, 'failed', a.events[-8:])
            for role, core in (('call', c.caller), ('answer', a.answerer)):
                monitor = (getattr(core, '_rx_state', None) or {}).get('quality')
                if monitor is not None and monitor not in quality_epochs[role]:
                    quality_epochs[role].append(monitor)

        for _ in range(1500):
            tick()
            if c.connected and a.connected:
                break
        self.assertTrue(c.connected and a.connected)
        self.assertEqual((c.bit_rate,a.bit_rate),(33600,33600))
        if lapm:
            c.start_v42(True, rate=33600, compression=(3,1024,8))
            a.start_v42(False, rate=33600, compression=(3,1024,8))
            for _ in range(500):
                tick()
                if c.v42_mode == a.v42_mode == 'lapm':
                    break
            self.assertEqual((c.v42_mode,a.v42_mode),('lapm','lapm'))
        links = (c._v42, a._v42)
        dictionaries = (getattr(c._v42,'_enc',None),getattr(a._v42,'_dec',None))
        rng = random.Random(34116)
        cp = bytes(rng.randrange(256) for _ in range(16384))
        ap = bytes(rng.randrange(256) for _ in range(16384))
        c.feed_bytes(cp)
        a.feed_bytes(ap)
        for _ in range(8):
            tick()
        self.assertGreater(c.pending_bits() + a.pending_bits(), 0)
        first = c if first_role == 'call' else a
        second = a if first_role == 'call' else c
        if failure_role is not None:
            failing = c if failure_role == 'call' else a
            core = c.caller if failure_role == 'call' else a.answerer
            old_c, old_a = c.caller, a.answerer
            failing.request_rate_change(24000)
            for _ in range(400):
                tick()
                if core._connected_rate.phase == 'mp':
                    break
            self.assertEqual(core._connected_rate.phase,'mp')
            ex = core._connected_rate.exchange
            self.assertIsNotNone(ex.transition_at)
            if fallback_rates is not None:failing._rate_fallback_rates=fallback_rates
            # Expire the actual controller timer before its next TX read.
            self.assertEqual(ex.check_timeout(ex.transition_at + ex.timeout),
                             'E timeout; full retrain required')
            for _ in range(2000):
                tick()
                if (c.caller is not old_c and a.answerer is not old_a
                        and not c._retraining and not a._retraining):
                    break
            self.assertIsNot(c.caller,old_c);self.assertIsNot(a.answerer,old_a)
            self.assertFalse(c._retraining or a._retraining)
            self.assertTrue(any('connected rate failed: E timeout' in text for _,text in failing.events))
            self.assertIs(c._v42,links[0]);self.assertIs(a._v42,links[1])
            self.assertIs(c._v42._enc,dictionaries[0]);self.assertIs(a._v42._dec,dictionaries[1])
            if fallback_rates is not None:
                wanted=(fallback_rates if failure_role=='call' else fallback_rates[::-1])
                self.assertEqual((c.bit_rate,c.rx_bit_rate),wanted)
                self.assertEqual((a.bit_rate,a.rx_bit_rate),wanted[::-1])
                for renewed in (c.caller,a.answerer):
                    self.assertEqual(renewed._local_mp['max_c2a'],14)
                    self.assertEqual(renewed._local_mp['max_a2c'],14)
                    self.assertEqual(renewed.rx_qualification['bit_errors'],0)
                del failing._rate_fallback_rates
            else:self.assertEqual((c.bit_rate,a.bit_rate),(33600,33600))
            self.assertEqual((c._retrains,a._retrains),(1,1))
        if online_commands:
            from modem_dce import AtDce
            from v34_online_control import OnlineControl
            controls = {m:OnlineControl(lambda m=m:m) for m in (c,a)}
            dces = {}
            for m in (c,a):
                dce=AtDce();dce.feed(b'ATE0V1X4\r');dce.online=True
                dce.attach_online_control(controls[m].request)
                dces[m]=dce
            def change(modem, parameter):
                self.assertEqual(dces[modem].feed(f'ATO{parameter}\r'.encode()),b'')
                self.assertTrue(dces[modem].online_return_pending)
            change(first,4)
        else:
            if directional:first.request_rate_change(rates=(33600,24000))
            else:first.request_rate_change(24000)
        for epoch in range(1,5 if lower_resync else 4):
            for _ in range(400):
                tick()
                if getattr(c.caller,'_rate_epoch',0) >= epoch and getattr(a.answerer,'_rate_epoch',0) >= epoch:
                    break
            self.assertEqual((c.caller._rate_epoch,a.answerer._rate_epoch),(epoch,epoch))
            rate = (31200 if online_commands else 24000) if epoch == 1 or (lower_resync and epoch == 2) else 33600
            expected=((33600,24000,24000,33600) if first_role=='call' else (24000,33600,33600,24000)) if directional and epoch==1 else (rate,)*4
            self.assertEqual((c.bit_rate,a.bit_rate,c.rx_bit_rate,a.rx_bit_rate),expected)
            self.assertIs(c._v42,links[0]);self.assertIs(a._v42,links[1])
            if lapm:
                self.assertEqual((c._v42.rate,a._v42.rate),(c.bit_rate,a.bit_rate))
                self.assertEqual((c.v42_mode,a.v42_mode),('lapm','lapm'))
                self.assertIs(c._v42._enc,dictionaries[0]);self.assertIs(a._v42._dec,dictionaries[1])
            for core in (c.caller,a.answerer):
                self.assertEqual(core.rx_qualification['bit_errors'],0)
                self.assertTrue(core.rx_qualification['standard_qualification'])
                self.assertEqual(core._local_mp['max_c2a'],14)
            if online_commands:
                initiator=second if epoch==(3 if lower_resync else 2) else first
                result=controls[initiator].poll()
                self.assertEqual(result['status'],'rate-exchanged')
                dces[initiator].finish_online_control(rate,'/LAPM')
                self.assertIn(f'CONNECT {rate}/LAPM'.encode(),dces[initiator].take_notifications())
                self.assertFalse(dces[initiator].online_return_pending)
            if epoch == 1:
                self.assertGreater(c.pending_bits() + a.pending_bits(), 0)
                if lower_resync:
                    change(first,3)
                else:
                    change(second,5) if online_commands else second.request_rate_change(33600)
            elif lower_resync and epoch == 2:
                change(second,5)
            elif epoch == (3 if lower_resync else 2):
                change(first,3) if online_commands else first.request_rate_change()  # Same-rate resynchronization.
        for _ in range(1500):
            tick()
            if len(cr) >= len(ap) and len(ar) >= len(cp):
                break
        self.assertEqual(bytes(cr),ap)
        self.assertEqual(bytes(ar),cp)
        self.assertTrue(c.connected and a.connected)
        if retrain_cycle:
            # A new physical core restarts its local epoch counter. The wrapper
            # must still notice the first rate change on that replacement core.
            old_c, old_a = c.caller, a.answerer
            c._carrier_stream_lost()
            for _ in range(2000):
                tick()
                if not c._retraining and a.answerer is not old_a and not a._retraining:
                    break
            self.assertIsNot(c.caller,old_c)
            self.assertIsNot(a.answerer,old_a)
            self.assertFalse(c._retraining or a._retraining)
            self.assertEqual((c._rate_epoch_seen,a._rate_epoch_seen),(0,0))
            self.assertIs(c._v42,links[0]);self.assertIs(a._v42,links[1])
            c.request_rate_change(24000)
            for _ in range(400):
                tick()
                if getattr(c.caller,'_rate_epoch',0)==getattr(a.answerer,'_rate_epoch',0)==1:
                    break
            self.assertEqual((c.bit_rate,a.bit_rate,c.rx_bit_rate,a.rx_bit_rate),(24000,)*4)
            self.assertEqual((c._v42.rate,a._v42.rate),(24000,24000))
            suffix_c = b'caller after full retrain and another rate change\r\n' * 20
            suffix_a = b'answerer after full retrain and another rate change\r\n' * 20
            c.feed_bytes(suffix_c);a.feed_bytes(suffix_a)
            for _ in range(400):
                tick()
                if len(cr) >= len(ap)+len(suffix_a) and len(ar) >= len(cp)+len(suffix_c):
                    break
            self.assertEqual(bytes(cr),ap+suffix_a)
            self.assertEqual(bytes(ar),cp+suffix_c)
            self.assertIs(c._v42._enc,dictionaries[0]);self.assertIs(a._v42._dec,dictionaries[1])
        if capture is not None:
            capture.close()
            self.assertEqual(capture.epoch,3)
            self.assertEqual([item['qualification']['bit_errors'] for item in capture.epochs],[0,0,0,0])
            self.assertFalse(capture.truncated)
            self.assertGreater(c.caller._rx_origin,0)
            self.assertGreater(len(c.caller._b1_inputs['raw']),0)
            self.assertEqual(len(c.caller._b1_inputs['raw']),len(c.caller._b1_inputs['baseband']))
            from verify_v34_dsp_capture import verify
            result = verify(capture.prefix)
            self.assertTrue(result['valid'],result)
        return dict(first_role=first_role, protocol='lapm-bis' if lapm else 'raw',
                    received_bytes_each_way=len(cr), caller_payload_sha256=hashlib.sha256(cp).hexdigest(),
                    answer_payload_sha256=hashlib.sha256(ap).hexdigest(),
                    caller_epochs=list(c.caller._connected_rate.completed),
                    answer_epochs=list(a.answerer._connected_rate.completed),
                    capture_verification=result if capture is not None else None,
                    quality_epochs={role:[m.report() for m in monitors]
                                    for role,monitors in quality_epochs.items()},
                    caller_connected=c.connected,answer_connected=a.connected)

    @patch.dict('os.environ', {'V34_QUALITY_MONITOR':'1'})
    def test_quality_rebaselines_across_real_pcm_rate_epochs(self):
        result = self.exercise(lapm=True)
        for reports in result['quality_epochs'].values():
            self.assertEqual(len(reports), 4)
            self.assertEqual([r['profile']['rate'] for r in reports], [33600, 24000, 33600, 33600])
            self.assertGreater(sum(r['symbols'] for r in reports), 0)
            for report in reports:
                if not report['symbols']:
                    self.assertIsNone(report['path_cost_per_symbol'])
                self.assertEqual(report['invalid_batches'], 0)
                self.assertFalse(report['selection_enabled'])
                self.assertLessEqual(len(report['recent']), 32)

    def test_raw_inflight_down_then_up_caller_initiates_first(self):
        self.exercise(first_role='call')

    def test_directional_receive_fallback_preserves_payload_both_roles(self):
        for role in ('call','answer'):self.exercise(lapm=True,first_role=role,directional=True)

    def test_automatic_failure_retains_receive_offer_through_full_retrain(self):
        for role in ('call','answer'):
            self.exercise(lapm=True,failure_role=role,fallback_rates=(33600,24000))

    def test_raw_inflight_down_then_up_answerer_initiates_first(self):
        self.exercise(first_role='answer')

    def test_lapm_compression_and_inflight_bytes_survive_both_rates(self):
        self.exercise(lapm=True)

    def test_rate_epochs_and_trimmed_b1_capture_reproduce(self):
        with tempfile.TemporaryDirectory() as directory:
            self.exercise(capture_dir=directory)

    def test_full_retrain_then_new_core_rate_epoch_updates_retained_link(self):
        self.exercise(lapm=True,retrain_cycle=True)

    def test_caller_timeout_retrains_with_inflight_compressed_payload(self):
        self.exercise(lapm=True,failure_role='call')

    def test_answerer_timeout_retrains_with_inflight_compressed_payload(self):
        self.exercise(lapm=True,failure_role='answer',legacy_retrain=False)

    def test_online_at_controls_wait_for_pcm_rate_completion(self):
        self.exercise(lapm=True,online_commands=True,lower_resync=True)


if __name__ == '__main__':
    unittest.main()
