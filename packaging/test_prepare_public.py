import unittest
from prepare_public import findings, selected


class PublicBoundaryTests(unittest.TestCase):
    def test_private_material_is_outside_allowlist(self):
        for path in ('lab/LLM_HANDOFF_V34.md', 'lab/asterisk/asterisk/etc/asterisk/pjsip.conf',
                     'lab/asterisk/audio-peer/fixtures/reference/call.rx.raw',
                     'lab/asterisk/audio-peer/fixtures/b1.npz', '.gitmodules',
                     'spandsp', 'lab/slmodem-peer/v34_dce_peer.py', '.env',
                     'lab/asterisk/audio-peer/analyze_second_leg.py'):
            self.assertFalse(selected(path), path)

    def test_actual_runtime_and_driver_source_are_retained(self):
        for path in ('lab/asterisk/el_modem.py', 'lab/asterisk/modem_dce.py',
                     'lab/asterisk/audio-peer/v34_rx.py', 'packaging/el_modem.spec',
                     'packaging/Dockerfile.lto-libpython', 'packaging/serial/serialsim-flow.patch'):
            self.assertTrue(selected(path), path)

    def test_reviewed_serial_evidence_is_retained_without_private_fixtures(self):
        for path in ('docs/MACOS_PTY_QUALIFICATION.json',
                     'docs/macos-pty-evidence/macos-arm64/first-capture.txt',
                     'docs/windows-com-evidence/code-integrity.json',
                     'packaging/windows/qualify_com_terminal.py',
                     '.github/workflows/macos-pty.yml'):
            self.assertTrue(selected(path), path)
        for path in ('docs/macos-pty-evidence/macos-arm64/provider.conf',
                     'docs/macos-pty-evidence/unknown/result.json',
                     'docs/windows-com-evidence/call.wav',
                     'lab/windows-com/guest-credential.json'):
            self.assertFalse(selected(path), path)

    def test_findings_do_not_leak_values(self):
        secret = 'ghp_' + 'x' * 36
        result = findings('example.py', ('\nvalue="' + secret + '"').encode())
        self.assertEqual(result, [dict(path='example.py', line=2, rule='github-token')])
        self.assertNotIn(secret, repr(result))


if __name__ == '__main__':
    unittest.main()
