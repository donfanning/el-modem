"""ATI output: bounded width, preserved data, build identity and measured DSP facts."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import os
import runpy
import sys
import types
from unittest.mock import patch

import modem_build
from modem_dce import AtDce, OnlineAudioSession, linear_to_ulaw, ulaw_to_linear
from modem_diagnostics import line_metrics, pcm_dbfs, recorded_fallbacks
from modem_pnp import HARDWARE_ID, COMPATIBLE_ID, identification


class InformationTests(unittest.TestCase):
    def test_pcm_dbfs_uses_full_scale_sine_reference(self):
        import math
        n = 800
        samples = [round(32767 * math.sin(2 * math.pi * i / 80)) for i in range(n)]
        power = sum(value * value for value in samples)
        self.assertAlmostEqual(pcm_dbfs(power, n), 0.0, delta=0.001)
        self.assertAlmostEqual(pcm_dbfs(power / 100, n), -20.0, delta=0.001)
        self.assertIsNone(pcm_dbfs(0, n))
        self.assertIsNone(pcm_dbfs(1, 0))

    def test_online_session_accumulates_decoded_media_pcm_before_dsp(self):
        session = OnlineAudioSession.__new__(OnlineAudioSession)
        session._rx_pcm_sum_squares = 0
        session._rx_pcm_samples = 0
        session._last_pcm = b''
        session._receive_pcm = lambda _pcm: b'ok'
        payload = bytes([linear_to_ulaw(8000)] * 12)
        self.assertEqual(session.handle_rtp_payload(payload), b'ok')
        decoded = ulaw_to_linear(payload[0])
        self.assertEqual(session._rx_pcm_sum_squares, 12 * decoded * decoded)
        self.assertEqual(session._rx_pcm_samples, 12)

    def test_retrain_metrics_and_recorded_fallbacks_are_explicit(self):
        engine = SimpleNamespace(
            retrain_stats={'attempts': 2, 'active': False, 'local_started': 1,
                           'peer_started': 1, 'automatic_started': 0,
                           'completed': 1, 'failed': 1},
            bell103_fallback=False,
        )
        stats = line_metrics(engine)
        self.assertEqual(stats['retrain']['peer_started'], 1)
        self.assertEqual(stats['retrain_attempts'], 2)
        self.assertEqual(recorded_fallbacks(engine), [])
        self.assertEqual(recorded_fallbacks(SimpleNamespace()), None)
        fallbacks = recorded_fallbacks(SimpleNamespace(bell103_fallback=True))
        self.assertEqual(fallbacks[0]['kind'], 'bell212a-to-bell103')
        online = recorded_fallbacks(SimpleNamespace(is_v34=True), {
            'transactions': [{'status': 'fallback-retrained', 'initial_tx_rate': 33600,
                              'tx_rate': 28800}]})
        self.assertEqual(online, [{'kind': 'online-control-fallback-retrain',
                                   'from_rate': 33600, 'to_rate': 28800,
                                   'source': 'online-control'}])

    def test_all_supported_pages_fit_80_columns_with_long_values(self):
        dce = AtDce()
        dce.echo = False
        dce.profile_path = 'C:/' + 'long-directory/' * 24 + 'profiles.json'
        dce.config_lines = ['CONFIG ' + 'x' * 600]
        dce.stored_numbers = {0: '1' * 150}
        dce.last_dialed = '2' * 160
        dce.net[100] = 'h' * 80
        dce.net[103] = 'do-not-print-this-password'
        dce.caller_id = dict(wall=0, number='3' * 160, name='caller' * 40)
        for n in range(18):
            with self.subTest(page=n):
                lines = dce._ati(n)
                if n in (13, 14):
                    self.assertIsNone(lines)
                    continue
                self.assertIsNotNone(lines)
                self.assertTrue(all(len(line) <= 80 for line in lines))
                reply = dce.feed(f'ATI{n}\r'.encode()).decode('ascii')
                self.assertTrue(reply.endswith('OK\r\n'))
                self.assertTrue(all(len(line) <= 80 for line in reply.split('\r\n')))
                self.assertNotIn('do-not-print-this-password', reply)
        lines = dce._ati(7)
        self.assertGreater(len(lines), 24, 'no fixed screen-height cap')
        self.assertEqual(''.join(AtDce._screen_lines(['x' * 600])).replace(' ', ''), 'x' * 600)
        slots = AtDce._screen_lines([dce._number_slots()[0]])
        self.assertEqual(sum(line.count('1') for line in slots), 150)

    def test_help_and_view_config_are_also_80_columns(self):
        dce = AtDce()
        dce.echo = False
        dce.net[100] = 'h' * 80
        for command in ('AT$', 'AT&$', 'ATS$', 'AT%$', 'AT\\$', 'ATD$', 'AT+$', 'AT&V'):
            with self.subTest(command=command):
                reply = dce.feed((command+'\r').encode()).decode('ascii')
                self.assertTrue(reply.endswith('OK\r\n'))
                self.assertTrue(all(len(line) <= 80 for line in reply.split('\r\n')))

    def test_help_pages_align_columns_and_explain_noop_commands(self):
        from modem_help import PAGES, help_screen
        for stem, (_, entries, _) in PAGES.items():
            lines = help_screen(stem)
            self.assertIn('=' * 78, lines)
            self.assertTrue(all(len(line) <= 80 for line in lines))
            self.assertIn('  Command               Description', lines)
            for command, _ in entries:
                self.assertTrue(any(line.startswith(f'  {command:<20}  ') for line in lines))
        text = '\n'.join(help_screen('&'))
        self.assertIn('compatibility only', text)
        self.assertIn('do not enable synchronous', ' '.join(text.split()))

    def test_unicode_and_terminal_controls_do_not_crash_or_escape(self):
        dce = AtDce()
        dce.caller_id = dict(wall=0, name='Jos\u00e9\x1b[31m', number='123')
        lines = dce._ati(15)
        text = '\n'.join(lines)
        self.assertIn(r'Jos\xe9\x1b[31m', text)
        self.assertNotIn('\x1b', text)
        text.encode('ascii')

    def test_identification_packet_is_machine_readable_and_not_wrapped(self):
        packet = identification()
        self.assertTrue(packet.startswith('(\x01$'+HARDWARE_ID))
        self.assertIn('\\\\MODEM\\'+COMPATIBLE_ID+'\\', packet)
        self.assertEqual(int(packet[-3:-1], 16), (sum(packet[:-3].encode('ascii'))+ord(')')) & 255)
        self.assertLessEqual(len(packet), 80)
        self.assertEqual(AtDce()._ati(9), [packet, ''])

    def test_ati666_easter_egg_is_ascii_fits_and_is_not_in_help(self):
        from modem_easter_egg import FIST, SCREEN, WIDTH
        dce = AtDce()
        dce.echo = False
        reply = dce.feed(b'ATI666\r').decode('ascii')
        self.assertTrue(reply.endswith('OK\r\n'))
        self.assertEqual(dce._ati(666), ['', *SCREEN, ''])
        self.assertEqual(SCREEN[0], 'BELL SYSTEMS PROPERTY'.center(WIDTH).rstrip())
        self.assertEqual(SCREEN[-1], 'NOT FOR SALE'.center(WIDTH).rstrip())
        self.assertEqual(tuple(SCREEN[2:-2]), tuple(FIST.splitlines()))
        self.assertTrue(all(len(line) <= 80 for line in reply.split('\r\n')))
        self.assertNotIn('666', '\n'.join(dce._help_screen('')))


class BuildIdentityTests(unittest.TestCase):
    def setUp(self):
        modem_build.build_timestamp.cache_clear()
        self.addCleanup(modem_build.build_timestamp.cache_clear)

    def test_source_does_not_misrepresent_mtime_as_compilation(self):
        with patch.object(modem_build, 'sys', SimpleNamespace(frozen=False)):
            self.assertEqual(modem_build.build_timestamp(), 'uncompiled source')

    def test_stamp_is_utc_and_both_ati_pages_use_it(self):
        stamp = modem_build.compile_timestamp()
        self.assertRegex(stamp, r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / 'el-modem-build.json'
            manifest.write_text(json.dumps({'compiled_at_utc': '2026-10-19T03:04:05Z'}), encoding='utf-8')
            with patch.object(modem_build, 'sys', SimpleNamespace(frozen=True, _MEIPASS=root)):
                dce = AtDce()
                self.assertIn('2026-10-19T03:04:05Z', '\n'.join(dce._ati(3)))
                self.assertIn('2026-10-19T03:04:05Z', '\n'.join(dce._ati(7)))
                # A copied executable's mtime or the current clock never supplies this date.
                with patch('os.path.getmtime', side_effect=AssertionError('mtime is not build time')):
                    self.assertEqual(modem_build.build_timestamp(), '2026-10-19T03:04:05Z')

    def test_legacy_or_bad_manifest_is_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(modem_build, 'sys', SimpleNamespace(frozen=True, _MEIPASS=root)):
                for contents in ('{}', '{bad-json', '{"compiled_at_utc":"yesterday"}'):
                    modem_build.build_timestamp.cache_clear()
                    (Path(root)/'el-modem-build.json').write_text(contents, encoding='utf-8')
                    self.assertIn('unknown', modem_build.build_timestamp())

    def test_build_spec_embeds_timestamp_in_the_bundled_manifest(self):
        # Execute the actual specification with packaging operations stubbed:
        # exercises timestamp generation, dependency discovery and data wiring,
        # without making a large executable or touching the running modem.
        root = Path(__file__).resolve().parents[2]
        hooks = types.ModuleType('PyInstaller.utils.hooks')
        hooks.collect_data_files = lambda name: []
        hooks.collect_submodules = lambda name: []
        modules = {name: types.ModuleType(name) for name in ('PyInstaller', 'PyInstaller.utils')}
        modules['PyInstaller.utils.hooks'] = hooks
        analysis = lambda *args, **kwargs: SimpleNamespace(binaries=[], datas=kwargs['datas'], pure=[], scripts=[])
        with tempfile.TemporaryDirectory() as folder:
            fake_extension = Path(folder)/'el_modem_iax.pyd'
            fake_extension.write_bytes(b'test stub, not a native build')
            with patch.dict(sys.modules, modules), patch('importlib.util.find_spec',
                    return_value=SimpleNamespace(origin=str(fake_extension))), \
                    patch.dict(os.environ, {'EL_MODEM_LIBPYTHON':''}):
                state = runpy.run_path(str(root/'packaging'/'el_modem.spec'), init_globals={
                    'SPECPATH':str(root/'packaging'), 'Analysis':analysis,
                    'PYZ':lambda *args: None, 'EXE':lambda *args, **kwargs: None})
                try:
                    manifest = state['manifest']
                    metadata = json.loads(manifest.read_text(encoding='utf-8'))
                    self.assertRegex(metadata['compiled_at_utc'], r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')
                    self.assertIn((str(manifest), '.'), state['a'].datas)
                    self.assertIn('lab/asterisk/modem_build.py',
                                  {key.replace('\\','/') for key in metadata['local_modules']})
                finally:
                    state['manifest_directory'].cleanup()


class DspInformationTests(unittest.TestCase):
    def test_engine_metrics_preserve_directional_rates_and_probe_scope(self):
        engine = SimpleNamespace(bit_rate=33600, rx_bit_rate=28800, state='data',
                                 caller=SimpleNamespace(call_baud=3429, probe_result={'snr_db': 28.5}),
                                 _retrains=2, _retraining=False)
        stats = line_metrics(engine)
        self.assertEqual((stats['tx_rate'], stats['rx_rate']), (33600, 28800))
        self.assertEqual(stats['probe_snr_db'], 28.5)
        self.assertNotIn('snr_db', stats, 'startup estimate must not masquerade as live SNR')
        self.assertEqual(stats['retrain_attempts'], 2)

    def test_v32_directional_rates(self):
        engine = SimpleNamespace(is_v32=True, tx=SimpleNamespace(data_rate=14400), bit_rate=9600,
                                 _mse=0.25, _sym_count=2000)
        stats = line_metrics(engine)
        self.assertEqual((stats['tx_rate'], stats['rx_rate'], stats['symbol_rate']), (14400,9600,2400))
        self.assertEqual(stats['decision_mse'], 0.25)
        self.assertIn('RESIDUAL MSE (EWMA): 0.25', '\n'.join(AtDce._dsp_report(stats)))

    def test_report_displays_measured_facts_and_freezes_them_at_disconnect(self):
        dce = AtDce()
        stats = dict(tx_rate=33600, rx_rate=28800, probe_snr_db=28.5,
                     rx_pcm_dbfs=-18.2, rx_pcm_samples=8000,
                     line_state='data', v32_retrain={'attempts':2, 'active':False,
                                                   'events':[{'completed_seconds':3}]},
                     v32_echo={'enabled':True, 'input_power':12.5, 'output_power':3.0,
                               'detected_delay':40, 'delay_units':'T/2 samples'},
                     v34_data_quality={'path_cost_per_symbol':0.125, 'symbols':1024, 'windows':2},
                     rx_qualification={'held_out_nmse':0.0025},
                     rtp={'received_packets':100, 'decoded_packets':98,
                          'late_or_duplicate_packets':2, 'concealed_samples':160},
                     realtime={'measurements':{'data_rx_seconds':{'mean':0.002, 'maximum':0.005}}})
        dce.link_stats = lambda: stats
        dce.connect_speed = 33600
        dce.feed(b'ATDT123\r')
        dce.notify_call_progress('CONNECT')
        dce.count_chars(sent=12, received=34)
        dce.count_dte_rx_dropped(7)
        live = '\n'.join(dce._ati(11))
        for fact in ('TX 33600  RX 28800', 'RX PCM LEVEL: -18.2 dBFS RMS (8000 samples)',
                     'not calibrated analog line level/dBm0', 'PROBE SNR: 28.5 dB',
                     'STARTUP ESTIMATE, NOT LIVE DATA SNR', 'RETRAIN ATTEMPTS: 2',
                     'EC BASEBAND POWER: IN 12.5 OUT 3', 'PATH COST / SYMBOL: 0.125',
                     'V.34 B1 HOLDOUT RESIDUAL: 5.00% RMS OF REFERENCE',
                     'Includes noise/model error; not live SNR or BER.',
                     'CONCEALED SAMPLES: 160', 'RX DSP: MEAN 2.000 ms MAX 5.000 ms',
                     'DATA OCTETS: TX 12  RX 34'):
            self.assertIn(fact, live)
        self.assertIn('SNR: N/A (NOT MEASURED)', live)
        dce.call_ended('NO CARRIER')
        stats['v32_retrain']['attempts'] = 99
        stats['rtp']['concealed_samples'] = 999
        final = '\n'.join(dce._ati(11))
        self.assertIn('RETRAIN ATTEMPTS: 2', final)
        self.assertIn('CONCEALED SAMPLES: 160', final)
        self.assertIn('DATA OCTETS: TX 12  RX 34', final)
        self.assertRegex(final, r'CALL DURATION: \d+ seconds')
        summary = '\n'.join(dce._ati(6))
        self.assertRegex(summary, r'Retrain attempts\s+2\s+Retrains completed\s+1')
        self.assertRegex(summary, r'Chars lost\s+N/A')
        self.assertRegex(summary, r'DTE RX bytes dropped\s+7')
        self.assertRegex(summary, r'Last Tx rate\s+33600\s+Last Rx rate\s+28800')

    def test_missing_or_nonfinite_measurements_do_not_become_good_line_claims(self):
        self.assertNotIn('probe_snr_db', line_metrics(SimpleNamespace(caller=SimpleNamespace(
            probe_result={'snr_db':float('nan')}))))
        lines = AtDce._dsp_report({'v34_data_quality':{'path_cost_per_symbol':float('inf')}})
        self.assertEqual(lines, [])
        lines = AtDce._dsp_report({'rx_qualification':{'held_out_nmse':float('nan')}})
        self.assertNotIn('B1 HOLDOUT RESIDUAL', '\n'.join(lines))


if __name__ == '__main__':
    unittest.main()
