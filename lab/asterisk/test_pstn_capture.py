import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

from capture_io import TranscriptSocket
from capture_tools import inspect_pcap, pcm_inventory
from pcm_capture import PcmCapture, capture_from_env
from run_pstn_capture import EMSI_INQ, connection_observed, probe, reply_list


def packet(payload):
    udp=struct.pack('!HHHH',16112,10000,8+len(payload),0)+payload
    ip=bytes([0x45,0])+struct.pack('!H',20+len(udp))+bytes(4)+bytes([64,17])+bytes(2)
    ip+=bytes([172,29,103,20,172,29,103,10])+udp
    return struct.pack('!HHI',0x800,0,2)+bytes(12)+ip


class PstnCaptureTests(unittest.TestCase):
    def test_automatic_profile_is_applied_after_reset_before_dial(self):
        class Connection:
            sent=[]
            def sendall(self,data):self.sent.append(data)
        connection=Connection()
        replies=[b'\r\nOK\r\n']*7+[b'\r\nNO CARRIER\r\n',b'\r\nOK\r\n']
        with patch('run_pstn_capture.read_until',side_effect=replies):
            result=probe(connection,'611',modulation_profile='V34,1,300,33600')
        self.assertLess(connection.sent.index(b'ATZ\r'),connection.sent.index(b'AT+MS=V34,1,300,33600\r'))
        self.assertLess(connection.sent.index(b'AT+MS=V34,1,300,33600\r'),connection.sent.index(b'ATDT611\r'))
        self.assertEqual(result['modulation_profile'],'V34,1,300,33600')
        self.assertTrue(result['hangup_confirmed'])

    def test_control_connect_is_taken_from_the_actual_dce_result(self):
        control={'scope':'local control','control_matrix':{'results':[
            {'result':{'dce_connect':'CONNECT 33600/LAPM'}}]}}
        self.assertTrue(connection_observed(control))
        control['control_matrix']['results'][0]['result']['dce_connect']='NO CARRIER'
        self.assertFalse(connection_observed(control))
        self.assertFalse(connection_observed({'scope':'local control'}))
        self.assertFalse(connection_observed({'scope':'VSP/PSTN startup observation'}))
        self.assertTrue(connection_observed({'scope':'VSP/PSTN startup observation',
            'session':{'connected':True}}))

    def test_pcap_retains_sequence_timing_and_detects_truncated_packets(self):
        rtp=struct.pack('!BBHII',128,0,55,3200,99)+bytes([255])*160
        frame=packet(rtp)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wire.pcap'
            header=struct.pack('<IHHIIII',0xa1b2c3d4,2,4,0,0,262144,276)
            path.write_bytes(header+struct.pack('<IIII',100,250000,len(frame),len(frame))+frame)
            timeline=Path(directory)/'rtp.jsonl'
            report=inspect_pcap(path,timeline)
            self.assertTrue(report['valid'],report)
            row=json.loads(timeline.read_text())
            self.assertEqual((row['sequence'],row['timestamp'],row['ssrc']),(55,3200,99))
            self.assertEqual(row['arrival_wall_time'],100.25)
            path.write_bytes(header+struct.pack('<IIII',100,0,len(frame)-1,len(frame))+frame[:-1])
            self.assertFalse(inspect_pcap(path)['valid'])
            path.write_bytes(header+b'partial')
            self.assertFalse(inspect_pcap(path)['valid'])

    def test_transcript_preserves_binary_bytes_and_offsets(self):
        first,second=socket.socketpair()
        try:
            with tempfile.TemporaryDirectory() as directory:
                prefix=Path(directory)/'dte'
                client=TranscriptSocket(first,prefix)
                data=b'\x7e\xff\x03\x00\x1b[31mLogin:\x7e'
                second.sendall(data)
                self.assertEqual(client.recv(3),data[:3])
                self.assertEqual(client.recv(100),data[3:])
                client.sendall(b'AT\r');self.assertEqual(second.recv(100),b'AT\r')
                client.close()
                self.assertEqual(Path(str(prefix)+'.rx.bin').read_bytes(),data)
                rows=[json.loads(line) for line in Path(str(prefix)+'.jsonl').read_text().splitlines()]
                self.assertEqual([r['byte_offset'] for r in rows if r['direction']=='rx'],[0,3])
        finally:
            first.close();second.close()

    def test_partial_send_records_only_the_bytes_accepted_by_the_socket(self):
        class Connection:
            def send(self,data,flags):
                self.accepted=bytes(data[:3])
                return 3
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            conn=Connection();prefix=Path(directory)/'dte'
            client=TranscriptSocket(conn,prefix)
            self.assertEqual(client.send(memoryview(b'abcdef')),3)
            client.close()
            self.assertEqual(Path(str(prefix)+'.tx.bin').read_bytes(),conn.accepted)
            row=json.loads(Path(str(prefix)+'.jsonl').read_text())
            self.assertEqual(row['bytes'],3)
            self.assertEqual(row['display'],"b'abc'")

    def test_pcm_timeline_marks_skipped_and_concealed_samples(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'MODEM_CAPTURE_TIMELINE':'1'}):
            cap=PcmCapture(str(Path(directory)/'call.pcm'),'originate-v34',120)
            cap.rx(b'',dict(skipped_samples=160))
            cap.rx(bytes(320),dict(timestamp=100,pcm_source='packet'))
            cap.rx(bytes(640),dict(timestamp=580,pcm_source='concealment'))
            cap.tx(bytes(320),dict(timestamp=0,pcm_source='before_g711'))
            cap.close()
            inventory=pcm_inventory(directory)
            self.assertTrue(inventory['valid'],inventory)
            self.assertEqual(inventory['captures'][0]['concealed_samples'],320)
            self.assertEqual(inventory['captures'][0]['skipped_samples'],160)
            with open(cap.stem+'.rx.raw','ab') as file:file.write(bytes(2))
            self.assertFalse(pcm_inventory(directory)['valid'])

    def test_required_capture_does_not_silently_disable_on_creation_error(self):
        with patch.dict(os.environ,{'MODEM_CAPTURE':'/dev/null/call','MODEM_CAPTURE_REQUIRED':'1'}):
            with self.assertRaises(OSError):capture_from_env('originate-v34')

    def test_prompt_split_at_connect_stops_without_login_and_hangs_up(self):
        class Connection:
            def __init__(self):self.sent=[]
            def sendall(self,data):self.sent.append(data)
        conn=Connection()
        replies=[b'\r\nOK\r\n']*6+[b'\r\nCONNECT 24000/LAPM\r\nWelcome\r\nLog',
            b'in:',b'',b'\r\nOK\r\n',b'\r\nOK\r\n']
        with patch('run_pstn_capture.read_until',side_effect=replies):
            result=probe(conn,'619123',prompts=[r'login\s*:'])
        self.assertEqual(result['stop_reason'],'target_prompt')
        self.assertTrue(result['hangup_confirmed'])
        self.assertFalse(result['payload_verified'])
        self.assertEqual(conn.sent[-3:],[b'ATDT619123\r',b'+++',b'ATH0\r'])

    def test_error_control_option_changes_only_the_N_command_and_defaults_to_lapm(self):
        class Connection:
            def __init__(self):self.sent=[]
            def sendall(self,data):self.sent.append(data)
        def run(**kw):
            conn=Connection()
            replies=[b'\r\nOK\r\n']*6+[b'\r\nNO CARRIER\r\n']+[b'\r\nOK\r\n']*4
            with patch('run_pstn_capture.read_until',side_effect=replies):
                probe(conn,'619123',**kw)
            return conn.sent
        default=run()
        zero=run(error_control=0)
        self.assertIn(b'AT\\N3\r',default)
        self.assertNotIn(b'AT\\N0\r',default)
        self.assertIn(b'AT\\N0\r',zero)
        self.assertNotIn(b'AT\\N3\r',zero)
        self.assertEqual([c for c in default if not c.startswith(b'AT\\N')],[c for c in zero if not c.startswith(b'AT\\N')])

    def test_error_control_is_forwarded_to_the_session_subprocess(self):
        import inspect
        import run_pstn_capture
        src=inspect.getsource(run_pstn_capture)
        self.assertIn("driver+=['--error-control',str(args.error_control)]",src)       # launcher -> session subprocess
        self.assertIn("error_control=args.error_control",src)                         # session subprocess -> probe

    def test_emsi_reply_is_sent_once_when_the_request_appears_and_nothing_else_is(self):
        class Connection:
            def __init__(self):self.sent=[]
            def sendall(self,data):self.sent.append(data)
        conn=Connection()
        replies=[b'\r\nOK\r\n']*6+[b'\r\nCONNECT 31200/LAPM\r\nBanner\r\n',b'**EMSI_',b'REQA77E\r',b'more',b'NO CARRIER',b'\r\nOK\r\n']
        with patch('run_pstn_capture.read_until',side_effect=replies):
            result=probe(conn,'619123',observe_seconds=60,replies=[(r'EMSI_REQ',EMSI_INQ)])
        self.assertEqual(conn.sent.count(EMSI_INQ),1)
        self.assertEqual(EMSI_INQ,b'**EMSI_INQC816\r')
        self.assertEqual(len(result['replies_sent']),1)
        self.assertEqual(result['replies_sent'][0]['trigger'],'EMSI_REQ')
        self.assertFalse(result['payload_verified'])
        self.assertIn('listed replies',result['session_scope'])
        self.assertEqual([c for c in conn.sent if c.startswith(b'AT')][-2:],[b'ATDT619123\r',b'ATH0\r'])

    def test_no_reply_is_sent_without_its_trigger_or_without_the_option(self):
        class Connection:
            def __init__(self):self.sent=[]
            def sendall(self,data):self.sent.append(data)
        for replies in ([(r'EMSI_REQ',EMSI_INQ)],None):
            conn=Connection()
            script=[b'\r\nOK\r\n']*6+[b'CONNECT 31200\r\nWelcome\r\n',b'NO CARRIER',b'',b'\r\nOK\r\n']
            with patch('run_pstn_capture.read_until',side_effect=script):
                probe(conn,'619123',observe_seconds=5,replies=replies)
            self.assertNotIn(EMSI_INQ,conn.sent)

    def test_reply_list_parses_the_cli_options(self):
        import argparse
        out=reply_list(argparse.Namespace(reply_emsi=True,reply=[r'ready=GO\r\n']))
        self.assertEqual(out,[(r'EMSI_REQ',EMSI_INQ),('ready',b'GO\r\n')])
        self.assertEqual(reply_list(argparse.Namespace(reply_emsi=False,reply=None)),[])

    def test_failed_escape_preserves_the_result_without_sending_ath_as_data(self):
        class Connection:
            def __init__(self):self.sent=[]
            def sendall(self,data):self.sent.append(data)
        conn=Connection()
        replies=[b'\r\nOK\r\n']*6+[b'CONNECT 24000\r\n',b'',b'',b'']
        with patch('run_pstn_capture.read_until',side_effect=replies):
            result=probe(conn,'619123')
        self.assertTrue(result['connected'])
        self.assertEqual(result['stop_reason'],'observation_timeout')
        self.assertFalse(result['hangup_confirmed'])
        self.assertEqual(conn.sent[-1],b'+++')


class OptInSettingsTests(unittest.TestCase):
    def test_defaults_add_nothing_and_options_expand_to_the_validated_set(self):
        import argparse
        from run_pstn_capture import optin_settings
        env, extra = optin_settings(argparse.Namespace(opt_in=None, info_hole_tolerant=False))
        self.assertEqual((env, extra), ([], []))
        env, extra = optin_settings(argparse.Namespace(opt_in='tested', info_hole_tolerant=False))
        self.assertEqual(extra, ['--v34-trn-gradient-auto', '--v34-trn-clock'])
        for item in ('V34_TRN_CLOCK_NATIVE=1', 'V34_TRAJECTORY_GRID=1', 'V34_NATIVE_FEEDBACK=1', 'V34_TIMING_MODE=data-gradient'):
            self.assertIn(item, env)
        self.assertNotIn('V34_INFO_HOLE_TOLERANT=1', env)
        env, _ = optin_settings(argparse.Namespace(opt_in=None, info_hole_tolerant=True))
        self.assertEqual(env, ['V34_INFO_HOLE_TOLERANT=1'])


class DutCommandTests(unittest.TestCase):
    def test_automatic_profile_reaches_both_pstn_and_control_dut(self):
        from run_pstn_capture import dut_docker_command,local_control_command
        args=self.args(automode=True)
        command=dut_docker_command(args,'dut','/out')
        self.assertEqual(command[command.index('--line-mode')+1],'ms')
        self.assertIn('--automode',local_control_command(args,'/out/control'))

    def args(self, **kw):
        import argparse
        d = dict(rate=33600, dut_image='img', opt_in=None, info_hole_tolerant=False)
        d.update(kw)
        return argparse.Namespace(**d)

    def test_default_command_carries_no_opt_in_settings(self):
        from run_pstn_capture import dut_docker_command, local_control_command
        c = dut_docker_command(self.args(), 'dut1', '/out')
        self.assertNotIn('--v34-trn-clock', c)
        self.assertFalse([x for x in c if x.startswith('V34_') and x not in ('V34_RATE=33600', 'V34_SYMS=3429')])
        control=local_control_command(self.args(phase4_eq_span=12),'/out/control')
        self.assertFalse([x for x in control if x.startswith('V34_PHASE4_EQ_SPAN=')])

    def test_cli_span_default_and_override(self):
        from run_pstn_capture import main
        for option,expected in (([],12),(['--phase4-eq-span','24'],24)):
            with patch('sys.argv',['capture','--output','/unused',*option]), \
                 patch('run_pstn_capture.run',return_value=0) as run:
                self.assertEqual(main(),0)
                self.assertEqual(run.call_args.args[0].phase4_eq_span,expected)

    def test_opt_in_command_has_the_environment_before_the_image_and_the_arguments_after_it(self):
        from run_pstn_capture import dut_docker_command
        c = dut_docker_command(self.args(opt_in='tested', info_hole_tolerant=True), 'dut1', '/out')
        image = c.index('img')
        for item in ('V34_TRN_CLOCK_NATIVE=1', 'V34_TRAJECTORY_GRID=1', 'V34_NATIVE_FEEDBACK=1', 'V34_TIMING_MODE=data-gradient', 'V34_INFO_HOLE_TOLERANT=1'):
            self.assertIn(item, c[:image])
            self.assertEqual(c[c.index(item)-1], '-e')
        self.assertIn('--v34-trn-clock', c[image:])
        self.assertIn('--v34-trn-gradient-auto', c[image:])
        self.assertIn('--v34-native-trellis', c[image:])

    def test_span24_reaches_both_duts_without_enabling_other_features(self):
        from run_pstn_capture import dut_docker_command, local_control_command
        args=self.args(phase4_eq_span=24)
        docker=dut_docker_command(args,'dut1','/out')
        control=local_control_command(args,'/out/control')
        self.assertIn('V34_PHASE4_EQ_SPAN=24',docker[:docker.index('img')])
        self.assertEqual(docker[docker.index('V34_PHASE4_EQ_SPAN=24')-1],'-e')
        self.assertEqual(control[control.index('V34_PHASE4_EQ_SPAN=24')-1],'--dut-env')
        self.assertIn('OPENBLAS_NUM_THREADS=1',control)
        self.assertIn('--native-trellis',control)
        self.assertNotIn('--v34-trn-clock',docker)
        self.assertNotIn('--trn-clock',control)

    def test_clock_option_is_only_the_trn_clock_fit(self):
        import argparse
        from run_pstn_capture import dut_docker_command, local_control_command, optin_settings
        env, extra = optin_settings(argparse.Namespace(opt_in='clock', info_hole_tolerant=False))
        self.assertEqual((env, extra), ([], ['--v34-trn-clock']))
        docker = dut_docker_command(self.args(opt_in='clock', phase4_eq_span=24), 'dut1', '/out')
        image = docker.index('img')
        self.assertIn('--v34-trn-clock', docker[image:])
        self.assertNotIn('--v34-trn-gradient-auto', docker)
        self.assertEqual([x for x in docker[:image] if x.startswith('V34_') and x not in ('V34_RATE=33600', 'V34_SYMS=3429')],
                         ['V34_PHASE4_EQ_SPAN=24'])
        control = local_control_command(self.args(opt_in='clock', phase4_eq_span=24), '/out/control')
        self.assertIn('--trn-clock', control)
        self.assertNotIn('--trn-gradient-auto', control)

    def test_v34_preimport_reaches_the_dut_command_only_when_asked(self):
        import argparse
        from run_pstn_capture import dut_docker_command, optin_settings
        _, extra = optin_settings(argparse.Namespace(opt_in=None, info_hole_tolerant=False))
        self.assertNotIn('--v34-preimport', extra)
        _, extra = optin_settings(argparse.Namespace(opt_in=None, info_hole_tolerant=False, v34_preimport=True))
        self.assertEqual(extra, ['--v34-preimport'])
        self.assertIn('--v34-preimport', dut_docker_command(self.args(v34_preimport=True), 'dut1', '/out'))

    def test_gc_pause_reaches_the_dut_command_only_when_asked(self):
        import argparse
        from run_pstn_capture import dut_docker_command, optin_settings
        _, extra = optin_settings(argparse.Namespace(opt_in=None, info_hole_tolerant=False))
        self.assertNotIn('--gc-pause', extra)
        self.assertIn('--gc-pause', dut_docker_command(self.args(gc_pause=True), 'dut1', '/out'))

    def test_local_control_preserves_both_startup_mitigations(self):
        from run_pstn_capture import local_control_command
        args = self.args(v34_preimport=True, gc_pause=True, opt_in='clock')
        control = local_control_command(args, '/out/control')
        self.assertIn('--v34-preimport', control)
        self.assertIn('--gc-pause', control)
        self.assertIn('--trn-clock', control)
        default = local_control_command(self.args(), '/out/control')
        self.assertNotIn('--v34-preimport', default)
        self.assertNotIn('--gc-pause', default)

    def test_hold_seconds_lengthens_the_capture_cap_only_when_asked(self):
        from run_pstn_capture import dut_docker_command
        def cap(**kw):
            c = dut_docker_command(self.args(**kw), 'dut1', '/out')
            return c[c.index('--capture-seconds') + 1]
        self.assertEqual(cap(), '120')
        self.assertEqual(cap(hold_seconds=0), '120')
        self.assertEqual(cap(hold_seconds=240), '310')
        self.assertEqual(cap(hold_seconds=20), '120')

    def test_dut_env_reaches_the_dut_and_the_control_and_bad_names_are_rejected(self):
        import argparse
        from run_pstn_capture import dut_docker_command, dut_env_arg, local_control_command, optin_settings
        env, extra = optin_settings(argparse.Namespace(opt_in=None, info_hole_tolerant=False, dut_env=['V42_ASSUME_BIS=1']))
        self.assertEqual((env, extra), (['V42_ASSUME_BIS=1'], []))
        docker = dut_docker_command(self.args(dut_env=['V42_ASSUME_BIS=1']), 'dut1', '/out')
        image = docker.index('img')
        self.assertIn('V42_ASSUME_BIS=1', docker[:image])
        self.assertEqual(docker[docker.index('V42_ASSUME_BIS=1') - 1], '-e')
        control = local_control_command(self.args(dut_env=['V42_ASSUME_BIS=1']), '/out/control')
        self.assertEqual(control[control.index('V42_ASSUME_BIS=1') - 1], '--dut-env')
        self.assertEqual(dut_env_arg('V34_X=1'), 'V34_X=1')
        for bad in ('PATH=/x', 'V42_X', '=1', 'V42_A B=1', 'LD_PRELOAD=x'):
            with self.assertRaises(argparse.ArgumentTypeError):
                dut_env_arg(bad)

    def test_low_rates_leave_the_symbol_rate_to_the_answerer(self):
        from run_pstn_capture import dut_docker_command
        low = dut_docker_command(self.args(rate=9600), 'dut1', '/out')
        self.assertIn('V34_RATE=9600', low)
        self.assertFalse(any(x.startswith('V34_SYMS=') for x in low))
        high = dut_docker_command(self.args(rate=33600), 'dut1', '/out')
        self.assertIn('V34_SYMS=3429', high)

    def test_cli_accepts_clock(self):
        from run_pstn_capture import main
        with patch('sys.argv', ['capture', '--output', '/unused', '--opt-in', 'clock']), \
             patch('run_pstn_capture.run', return_value=0) as run:
            self.assertEqual(main(), 0)
            self.assertEqual(run.call_args.args[0].opt_in, 'clock')

    def test_control_forwards_tested_features_using_its_own_flag_names(self):
        from run_pstn_capture import local_control_command
        command=local_control_command(self.args(opt_in='tested',phase4_eq_span=24),'/out/control')
        self.assertIn('--trn-clock',command)
        self.assertIn('--trn-gradient-auto',command)
        self.assertNotIn('--v34-trn-clock',command)
        self.assertIn('V34_TIMING_MODE=data-gradient',command)


if __name__=='__main__':
    unittest.main()
