"""Real DCE/automode adapters must preserve transport validity for the engine."""
import os
import socket
import struct
import unittest
from unittest.mock import patch

from modem_dce import OnlineAudioSession, _V22Receiver, linear_to_ulaw
from autochain import V8Chain


class ReceiveTransportTests(unittest.TestCase):
    def make_session(self, calling, mode):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        session=OnlineAudioSession(sock,('127.0.0.1',9),
            'originate' if calling else 'answer', 'answer' if calling else 'originate',
            line_mode=mode)
        self.addCleanup(session.stop)
        session._rtp_jitter_samples=0
        if mode=='chain':session._v22._choose('v34')
        engine=getattr(session._v22,'engine',None) or session._v22
        engine._phase='p2'
        return session,engine

    @staticmethod
    def packet(seq, timestamp):
        return struct.pack('!BBHII',128,0,seq,timestamp,1234)+bytes([linear_to_ulaw(1000)])*160

    def test_rtp_erasure_marks_only_its_chunk_through_actual_wrappers_both_roles(self):
        for calling in (True,False):
            for mode in ('v34','chain'):
                with self.subTest(calling=calling,mode=mode),patch.dict(os.environ,{'EL_MODEM_DSP_CAPTURE':''}):
                    session,engine=self.make_session(calling,mode)
                    module='v34_modem' if calling else 'v34_answer_modem'
                    with patch(module+'.observe_retrain_budget') as budget:
                        session.handle_rtp_packet(self.packet(0,0))
                        session.handle_rtp_packet(self.packet(2,320))
                        self.assertEqual([c.kwargs['concealed'] for c in budget.call_args_list],[False,True,False])
                        self.assertEqual([c.args[2] for c in budget.call_args_list],[160,160,160])
                        self.assertEqual(engine._conceal_pending,0)
                    self.assertEqual(session._rtp_counts['concealed_samples'],160)
                    if calling:self.assertEqual(engine.caller._concealed,[(160,160)])

    def test_v32_discontinuity_and_concealment_reach_actual_engine(self):
        for calling in (True,False):
            for chain in (True,False):
                with self.subTest(calling=calling,chain=chain):
                    if chain:
                        modem=V8Chain(calling);modem._choose('v32');engine=modem.engine
                    else:
                        if calling:
                            from v32_stream import V32Modem
                            engine=V32Modem(echo=False)
                        else:
                            from v32_answer import V32Answerer
                            engine=V32Answerer(echo=False)
                        modem=engine
                    close=getattr(modem,'close',None)
                    if close is not None:self.addCleanup(close)
                    receiver=_V22Receiver(modem)
                    receiver.note_concealed_rx(160)
                    self.assertGreaterEqual(engine._retrain_untrusted_samples,160)
                    engine._retrain_untrusted_samples=0
                    receiver.note_transport_discontinuity_rx()
                    self.assertGreater(engine._retrain_untrusted_samples,0)

    def test_v8_marks_are_not_replayed_into_a_later_engine(self):
        chain=V8Chain(True);receiver=_V22Receiver(chain)
        receiver.note_concealed_rx(160);receiver.note_transport_discontinuity_rx()
        chain._choose('v34');self.addCleanup(chain.close)
        self.assertEqual(getattr(chain.engine,'_conceal_pending',0),0)
        self.assertEqual(chain.engine.caller._concealed,[])

    def test_engines_without_validity_hooks_keep_their_pcm(self):
        class Engine:
            def feed_pcm(self,pcm):return pcm
        receiver=_V22Receiver(Engine())
        receiver.note_concealed_rx(160);receiver.note_transport_discontinuity_rx()
        self.assertEqual(receiver.feed_pcm(b'unchanged'),b'unchanged')

    def test_answerer_legacy_notification_name_remains_compatible(self):
        from v34_answer_modem import V34AnswerModem
        engine=V34AnswerModem()
        close=getattr(engine,'close',None)
        if close is not None:self.addCleanup(close)
        engine.note_concealed(160)
        self.assertEqual(engine._conceal_pending,160)


if __name__=='__main__':unittest.main()
