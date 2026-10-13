import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from modem_dce import AtDce
from modem_settings import set_value,snapshot
from modem_supervisor import Supervisor
from modem_control import Channel

class SettingsTests(unittest.TestCase):
    def test_hayes_roundtrip_and_case_preservation(self):
        d=AtDce()
        self.assertIn(b'OK',d._execute('AT+CFG="sip.user","MixedCase"'))
        self.assertEqual(d.net[102],'MixedCase')
        self.assertIn(b'MixedCase',d._execute('AT+CFG?'))
        self.assertIn(b'ERROR',d._execute('AT+CFG="dsp.eq_span","0"'))
        self.assertIn(b'ERROR',d._execute('AT+CFG="unknown","1"'))
    def test_settings_do_not_touch_online_data(self):
        d=AtDce();d.online=True
        with self.assertRaises(ValueError):set_value(d,'sip.server','example.invalid')
        with self.assertRaises(ValueError):set_value(d,'modem.echo',0)
        self.assertTrue(d.echo)
        d.online=False
        with self.assertRaises(ValueError):set_value(d,'modem.compression',2)
    def test_secret_is_masked(self):
        d=AtDce();set_value(d,'sip.password','secret-for-test')
        self.assertNotIn('secret-for-test',json.dumps(snapshot(d)))
    def test_profile_pending_and_active(self):
        with tempfile.TemporaryDirectory() as root:
            d=AtDce();d.load_profiles(root+'/profile.json')
            set_value(d,'dsp.eq_span',12);d._store_profile(0)
            loaded=AtDce();loaded.load_profiles(root+'/profile.json')
            self.assertEqual(loaded.extended_settings['dsp.eq_span'],'12')
            self.assertEqual(os.stat(root+'/profile.json').st_mode&0o777,0o600)
    def test_save_failure_returns_error(self):
        d=AtDce();d.profile_path='/nonexistent/test'
        with patch('modem_storage.atomic_json',side_effect=OSError('denied')):
            self.assertIn(b'ERROR',d._execute('AT&W'))
            self.assertNotIn(0,d.profiles)
    def test_profile_validator_and_factory_compatibility(self):
        d=AtDce();self.assertEqual(d.ms['max'],14400)
        set_value(d,'modem.max_rate','33600');self.assertEqual(d.ms['max'],33600)
        with self.assertRaises(ValueError):set_value(d,'modem.modulation','bogus')
        with patch.dict(os.environ,{'EL_MODEM_FACTORY_NATIVE':'1'}):
            native=AtDce();self.assertEqual(native.ms['max'],33600);self.assertEqual(native.error_control,3)
    def test_invalid_port_and_bind(self):
        d=AtDce()
        for name,value in [('sip.port','0'),('sip.bind_host','no address'),('sip.bind_port','65536')]:
            with self.assertRaises(ValueError):set_value(d,name,value)

class SupervisorTests(unittest.TestCase):
    def test_instances_isolated_and_retained_ids_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');one=s.add(serial='pty');two=s.add(serial='tcp')
            self.assertNotEqual(one.sip_port,two.sip_port)
            (s.root/one.id).mkdir(parents=True)
            s.remove(one.id);three=s.add(serial='pty');self.assertNotEqual(three.id,one.id)
            self.assertEqual(len(Supervisor(root+'/config.json').specs),2)
    def test_duplicate_and_lock(self):
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.acquire();m=s.add(serial='pty')
            try:
                with self.assertRaises(ValueError):Supervisor(root+'/config.json').acquire()
                s.specs.append(m)
                with self.assertRaises(ValueError):s.validate()
            finally:s.close()
    def test_bounded_ipc(self):
        a,b=socket.socketpair();left,right=Channel(a),Channel(b)
        try:
            left.send(dict(id=1,action='settings'));left.flush()
            self.assertEqual(right.receive()[0]['id'],1)
            with self.assertRaises(BufferError):left.send({'large':'a'*Channel.LIMIT})
        finally:left.close();right.close()

class UITests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_dialog_and_settings_screen(self):
        from modem_tui import ModemApp,SettingsScreen
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.add(serial='pty');app=ModemApp(s)
            async with app.run_test(size=(120,45)) as pilot:
                await pilot.pause();await pilot.click('#configure');await pilot.pause()
                await pilot.click('#cancel');await pilot.pause()
                app.push_screen(SettingsScreen('modem-1'));await pilot.pause()
                app.screen.load_rows(snapshot(AtDce()));await pilot.pause()
                await pilot.click('#back');await pilot.pause()
            s.close()
    def test_ascii_controls_and_bounds(self):
        from modem_tui import AsciiBuffer
        b=AsciiBuffer(limit=2)
        self.assertEqual(b.feed(b'abc\rX\nq\x08Y\x1b'),'Xbc\nY<1B>')
        self.assertEqual(b.feed(b'\nend'),'Y<1B>\nend')

class TerminalUITests(unittest.IsolatedAsyncioTestCase):
    async def test_ascii_keys_do_not_trigger_dashboard_shortcuts(self):
        from modem_tui import ModemApp,TerminalScreen
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.add(serial='pty')
            sent=[]
            def request(ident,action,**fields):
                s.sequence+=1;sent.append((action,fields));s.responses[s.sequence]=dict(ok=True,value='done');return s.sequence
            s.request=request
            app=ModemApp(s)
            async with app.run_test(size=(120,40)) as pilot:
                app.action_terminal();await pilot.pause(.2)
                self.assertIsInstance(app.screen,TerminalScreen)
                await pilot.press('a','t','q','enter','ctrl+c');await pilot.pause(.2)
                data=b''.join(bytes.fromhex(fields['hex']) for action,fields in sent if action=='terminal.write')
                self.assertEqual(data,b'atq\r\x03')
                self.assertIsInstance(app.screen,TerminalScreen)
                await pilot.press('f10');await pilot.pause(.2)
                self.assertFalse(isinstance(app.screen,TerminalScreen))
            s.close()

class BootSettingsTests(unittest.TestCase):
    def test_hayes_boot_overrides_and_instance_edit_are_consistent(self):
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');spec=s.add(serial='pty')
            d=AtDce();d.load_profiles(str(s.root/spec.id/'profiles.json'))
            set_value(d,'sip.bind_port',5200);d._store_profile(0)
            self.assertEqual(s.effective(spec).sip_port,5200)
            s.configure(spec,{'sip_port':5202})
            self.assertEqual(s.effective(s.specs[0]).sip_port,5202)
            other=s.add(serial='pty')
            s.specs[1].sip_port=5202
            with self.assertRaises(ValueError):s.validate()
