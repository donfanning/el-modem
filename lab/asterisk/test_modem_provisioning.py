"""Focused safety and reconciliation tests for the Asterisk adapter."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modem_provision import AsteriskPjsipAdapter, ProvisionError, INCLUDE_NAME
from modem_provisioning import (
    ModemSipAccount, ProvisionRequest, ProvisionTarget,
    read_saved_network_profile,
)


class AsteriskProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='elmodem-provision-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main = self.root / 'pjsip.conf'
        self.extensions = self.root / 'extensions.conf'
        self.original_main = (
            '[udp-transport]\n'
            'type=transport\n'
            'protocol=udp\n'
            'bind=127.0.0.1:5060\n'
        ).encode()
        self.main.write_bytes(self.original_main)
        self.main.chmod(0o640)
        self.original_extensions = b'[from-modems]\n'
        self.extensions.write_bytes(self.original_extensions)
        self.adapter = AsteriskPjsipAdapter()
        self.target = ProvisionTarget('asterisk', 'pbx.example.lan:5060', {
            'host': 'pbx.example.lan', 'port': 5060,
            'context': 'from-modems', 'transport': 'udp-transport',
            'config_dir': str(self.root),
        })
        self.account = ModemSipAccount(
            modem_id='modem-1', username='modemuser',
            password='test-password-value',
            registrar='pbx.example.lan:5060',
        )
        self.request = ProvisionRequest(self.target, (self.account,))

    def plan(self):
        return self.adapter.plan(self.request)

    def test_apply_check_and_rollback_are_limited_to_owned_files(self):
        plan = self.plan()
        self.assertNotIn(self.account.password, repr(plan))
        self.assertNotIn(self.account.password, repr(self.request))
        self.assertIn('endpoint modemuser', plan.summary[0])

        backups = self.adapter.apply(plan)
        fragment = self.root / INCLUDE_NAME
        self.assertEqual(fragment.read_text(encoding='utf-8').count(self.account.password), 1)
        self.assertEqual(fragment.stat().st_mode & 0o027, 0)
        self.assertEqual(self.extensions.read_bytes(), self.original_extensions)
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in backups))
        self.assertTrue(self.adapter.check(plan).consistent)

        restored = self.adapter.rollback(self.target)
        self.assertCountEqual(restored, [self.main, fragment])
        self.assertEqual(self.main.read_bytes(), self.original_main)
        self.assertFalse(fragment.exists())
        self.assertEqual(self.extensions.read_bytes(), self.original_extensions)

    def test_refuses_unowned_include_and_unmarked_fragment(self):
        self.main.write_text(self.main.read_text() + f'#include {INCLUDE_NAME}\n')
        with self.assertRaisesRegex(ProvisionError, 'without El-Modem ownership markers'):
            self.plan()

        self.main.write_bytes(self.original_main)
        fragment = self.root / INCLUDE_NAME
        fragment.write_text('[handwritten]\ntype=endpoint\n')
        with self.assertRaisesRegex(ProvisionError, 'not marked as El-Modem-managed'):
            self.plan()

    def test_apply_revalidates_context_after_plan(self):
        plan = self.plan()
        self.extensions.write_text('[changed-context]\n')
        with self.assertRaisesRegex(ProvisionError, 'context .* disappeared'):
            self.adapter.apply(plan)
        self.assertFalse((self.root / INCLUDE_NAME).exists())

    def test_live_check_requires_loaded_endpoint_and_contact(self):
        plan = self.plan()
        self.adapter.apply(plan)
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        cli = bin_dir / 'asterisk'
        cli.write_text(
            '#!/usr/bin/env python3\n'
            'import sys\n'
            'command = sys.argv[-1]\n'
            'if command.startswith("pjsip show aor"):\n'
            ' print("Contact: modemuser/sip:modemuser@127.0.0.1 1 Avail")\n'
            'else:\n'
            ' print("Endpoint: modemuser")\n'
            ' print("InAuth: elmodem_modem_1_auth/modemuser")\n'
            ' print("Aor: modemuser")\n'
            ' print("Transport: udp-transport")\n'
            ' print(" context : from-modems")\n'
            ' print(" auth : elmodem_modem_1_auth")\n'
            ' print(" aors : modemuser")\n'
        )
        cli.chmod(0o700)
        with patch.dict(os.environ, {'PATH': str(bin_dir) + os.pathsep + os.environ.get('PATH', '')}):
            report = self.adapter.check(plan, live=True)
        self.assertTrue(report.consistent, report)
        self.assertEqual(report.registrations, (
            ('modem-1', 'registered contact present (Avail)'),
        ))
        self.assertEqual(report.runtime, (('modem-1', 'runtime endpoint matches'),))

    def test_profile_loader_rejects_symlinked_instance_root(self):
        config_dir = self.root / 'user-config'
        config_dir.mkdir()
        config = config_dir / 'config.json'
        config.write_text('{}')
        instances = config_dir / 'instances'
        profile_dir = instances / 'modem-1'
        profile_dir.mkdir(parents=True)
        profile = profile_dir / 'profiles.json'
        profile.write_text('{"profiles":{"0":{"net":{"102":"user"}}}}')
        profile.chmod(0o600)
        self.assertEqual(read_saved_network_profile(config, 'modem-1'), {'102': 'user'})

        outside = self.root / 'outside'
        outside.mkdir()
        (profile_dir / 'profiles.json').unlink()
        (outside / 'modem-1').mkdir()
        (outside / 'modem-1' / 'profiles.json').write_text(
            '{"profiles":{"0":{"net":{"102":"redirected"}}}}')
        instances.rename(config_dir / 'instances-real')
        instances.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'must not be symlinks'):
            read_saved_network_profile(config, 'modem-1')


if __name__ == '__main__':
    unittest.main()
