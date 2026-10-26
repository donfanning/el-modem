"""Bounds, clock evidence, and PCM parity of optional timing instrumentation."""
import socket
import struct
import time
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch
from runtime_timing import TimingTrace
from modem_dce import OnlineAudioSession

from el_modem import worker_passthrough_environment
from v34_realtime import RealtimeStats


class TimingTraceTests(unittest.TestCase):
    def test_opt_in_timer_setting_survives_supervisor_worker_scrub(self):
        settings = worker_passthrough_environment({
            'EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS': '1',
            'EL_MODEM_TIMING_TRACE': '1',
            'EL_MODEM_UNRELATED_SETTING': 'must-not-leak',
            'V34_EQUALIZER_BACKEND': 'python',
            'V34_GAP_RECOVERY': '1',
            'V32_NUMPY_METRICS': '1',
        })
        self.assertEqual(settings, {
            'EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS': '1',
            'EL_MODEM_TIMING_TRACE': '1',
            'V34_EQUALIZER_BACKEND': 'python',
            'V34_GAP_RECOVERY': '1',
            'V32_NUMPY_METRICS': '1',
        })

    def test_scoped_windows_timer_request_is_visible_and_released_on_stop(self):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        with patch.dict('os.environ', {'EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS':'1'}):
            session=OnlineAudioSession(sock,('127.0.0.1',9),'originate','answer')
        session.start()
        try:
            self.assertIn(session._timer_resolution_request.report()['status'],
                          ('active','unsupported_platform'))
        finally:
            session.stop()
        self.assertIn(session.link_stats()['windows_timer_resolution']['status'],
                      ('released','unsupported_platform'))
    def test_receive_timing_trace_separates_thread_cpu_from_wall_time(self):
        session=object.__new__(OnlineAudioSession)
        session._rx_deaf_samples=0
        session._bell212_char_bits=10
        session._capture=None
        session._v22=SimpleNamespace(connected=False)
        session._rx_arrival=None
        session._rtp_jitter_samples=0
        session._rx_context={'sequence':7}
        session._dsp_capture_pending=False
        def delayed_receive(pcm):
            time.sleep(.03)
            return b''
        session.receiver=SimpleNamespace(feed_pcm=delayed_receive)
        session._v42_probe=False
        session._negotiator=None
        session._rx_switched=True
        session.realtime=RealtimeStats()
        session.timing_trace=TimingTrace()

        self.assertEqual(session._receive_pcm(b'\xff'*320),b'')

        event=session.timing_trace.report()['events']['rx_processing']['first']
        self.assertEqual(event['row']['sequence'],7)
        self.assertGreaterEqual(event['row']['thread_cpu_seconds'],0.0)
        self.assertGreater(event['score'],0.02)
        self.assertLess(event['row']['thread_cpu_seconds'],event['score']/2)

    def test_link_stats_keeps_trace_when_automode_has_switched_from_v22(self):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        session=OnlineAudioSession(sock,('127.0.0.1',9),'originate','answer',line_mode='chain')
        session._auto=SimpleNamespace(state='v34')
        session.timing_trace=TimingTrace()
        session.timing_trace.threshold('rx_processing',.025,dict(sequence=17))

        stats=session.link_stats()

        self.assertIn('timing_trace',stats)
        self.assertEqual(stats['timing_trace']['events']['rx_processing']['worst']['row']['sequence'],17)


    def test_watchdog_captures_live_blocked_stage_and_preserves_packet(self):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sink=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addCleanup(sock.close);self.addCleanup(sink.close)
        sink.bind(('127.0.0.1',0));sink.settimeout(1)
        with patch.dict('os.environ',{'EL_MODEM_TIMING_TRACE':'1','EL_MODEM_TX_WATCHDOG':'1'}):
            session=OnlineAudioSession(sock,sink.getsockname(),'originate','answer')
        release=threading.Event();captured=threading.Event()
        def blocked_samples(count):
            release.wait(1.5)
            return [0]*count
        session.transmitter.samples=blocked_samples
        original=session.timing_trace.event
        def event(kind,score,row):
            original(kind,score,row)
            if kind=='tx_watchdog_startup':captured.set()
        session.timing_trace.event=event
        session.start()
        try:
            self.assertTrue(captured.wait(1))
            row=session.timing_trace.report()['events']['tx_watchdog_startup']['first']['row']
            self.assertEqual(row['stage'],'samples')
            frames=[frame for thread in row['threads'] for frame in thread['stack']]
            self.assertTrue(any(f['function']=='blocked_samples' for f in frames))
            self.assertTrue(all(set(f)=={'file','function','line'} for f in frames))
            release.set()
            packet,_=sink.recvfrom(2048)
            self.assertEqual(packet[12:],b'\xff'*160)
        finally:
            release.set();session.stop()
        self.assertFalse(session._tx_watchdog.is_alive())

    def test_startup_queue_does_not_hide_worst_data_queue(self):
        trace=TimingTrace()
        trace.threshold('queue', .6, dict(data=False, sequence=1))
        trace.threshold('queue', .24, dict(data=True, sequence=2), data=True)
        trace.threshold('queue', .12, dict(data=True, sequence=3), data=True)
        events=trace.report()['events']
        self.assertEqual(events['queue']['worst']['row']['sequence'],1)
        self.assertEqual(events['queue_startup']['worst']['score'],.6)
        self.assertEqual(events['queue_data']['worst']['row']['sequence'],2)
        self.assertEqual(events['queue_data']['count'],2)

    def test_context_is_bounded_and_first_and_worst_survive_a_cascade(self):
        trace = TimingTrace()
        for i in range(10000):
            trace.record('main', dict(index=i))
            trace.threshold('main_control', .030 if i != 50 else .500, dict(index=i))
        report = trace.report()
        self.assertEqual(report['counts']['main'], 10000)
        self.assertEqual(len(report['recent']['main']), trace.LIMIT)
        event = report['events']['main_control']
        self.assertEqual(event['count'], 10000)
        self.assertEqual(event['first']['row']['index'], 0)
        self.assertEqual(event['worst']['row']['index'], 50)
        self.assertLessEqual(len(event['worst']['preceding']['main']), 32)
        report['recent']['main'].clear()
        self.assertEqual(len(trace.report()['recent']['main']), trace.LIMIT)

    def test_wrapping_packet_clocks_and_raw_clock_sampling_evidence(self):
        trace = TimingTrace()
        for i in range(3):
            trace.packet(dict(sequence=(65535+i)&65535, timestamp=(0xffffff60+i*160)&0xffffffff,
                              ssrc=1234, clock_mono_before=10+i, clock_wall=1000+i,
                              clock_mono_after=10+i+.000001, kernel_seconds=1000+i,
                              kernel_nanoseconds=0))
        self.assertFalse(trace.report()['events'])
        trace.packet(dict(sequence=2, timestamp=320, ssrc=1234,
                          clock_mono_before=13, clock_wall=1003.05, clock_mono_after=13.02))
        events = trace.report()['events']
        self.assertIn('clock_pair_span', events)
        self.assertIn('clock_offset_step', events)
        self.assertAlmostEqual(events['clock_pair_span']['first']['score'], .02)
        self.assertNotIn('rtp_arrival_step', events)
        trace.packet(dict(sequence=20, timestamp=5000, ssrc=5678))
        self.assertNotIn('rtp_arrival_step', trace.report()['events'])

    def test_trace_preserves_pcm_gap_returns_order_and_source_reset(self):
        sessions = []
        chunks = []
        contexts = []
        for enabled in (False, True):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addCleanup(sock.close)
            session = OnlineAudioSession(sock, ('127.0.0.1', 9), 'originate', 'answer')
            session.timing_trace = TimingTrace() if enabled else None
            session._rtp_jitter_samples = 320
            captured, ctx = [], []
            def receive(pcm, s=session, c=captured, x=ctx):
                c.append(pcm)
                x.append(dict(s._rx_context))
                return b'gap' if len(pcm) == 640 else b'packet'
            session._receive_pcm = receive
            sessions.append(session); chunks.append(captured); contexts.append(ctx)
        streams = [(frame, frame*160, 1234) for frame in (0, 2, 1, 4, 5, 6)]
        streams += [(frame, frame*160, 5678) for frame in (0, 1, 2, 3)]
        outputs = [[], []]
        for frame, timestamp, ssrc in streams:
            packet = struct.pack('!BBHII', 0x80, 0, frame, timestamp, ssrc) + bytes([128+frame])*160
            for i, s in enumerate(sessions):
                outputs[i].append(s.handle_rtp_packet(packet, received_at=timestamp/8000,
                    clock_context=dict(kernel_seconds=100, kernel_nanoseconds=timestamp)))
        self.assertEqual(chunks[0], chunks[1])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(sessions[0]._rtp_counts, sessions[1]._rtp_counts)
        self.assertEqual(sessions[1]._rtp_counts['concealed_samples'], 160)
        self.assertEqual([c['timestamp'] for c in contexts[1]], [0,160,320,640,640,0,160])
        self.assertEqual([c['pcm_source'] for c in contexts[1]],
                         ['packet','packet','packet','concealment','packet','packet','packet'])
        self.assertEqual(contexts[1][3]['kernel_nanoseconds'], 640)
        self.assertEqual(len(sessions[1]._rtp_trace_context), 2)


if __name__ == '__main__':
    unittest.main()
