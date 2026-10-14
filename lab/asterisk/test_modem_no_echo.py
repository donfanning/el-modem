"""All production V32 entry points explicitly disable echo cancellation."""
import unittest
from modem_dce import AtDce, OnlineAudioSession, line_from_ms
from modem_selftest import make_session, profile_options


class NoEchoFactoryTests(unittest.TestCase):
    def test_forced_and_deferred_caller_answerer_factories(self):
        for profile in ('V32B,0,300,14400', 'V34,1,300,33600'):
            for calling in (True, False):
                with self.subTest(profile=profile,calling=calling):
                    session=make_session(OnlineAudioSession,
                        profile_options(AtDce,line_from_ms,profile),calling)
                    try:
                        if profile.startswith('V34'):
                            session._v22._choose('v32')
                            engine=session._v22.engine
                        else:engine=session._v22
                        self.assertFalse(engine.use_echo)
                        self.assertFalse(engine.echo_stats['requested'])
                        self.assertFalse(engine.echo_stats['enabled'])
                    finally:session.stop()
