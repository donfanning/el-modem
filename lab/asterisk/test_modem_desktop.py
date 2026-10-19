import asyncio
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from modem_dce import AtDce
from modem_settings import set_value,snapshot
from modem_supervisor import Supervisor
from modem_control import Channel

class SettingsTests(unittest.TestCase):
    def test_pipe_backend_hayes_profile_roundtrip(self):
        backend='pipe' if sys.platform=='win32' else 'device'
        endpoint='ElModemSecond' if backend=='pipe' else '/dev/ttyUSB1'
        with tempfile.TemporaryDirectory() as root:
            d=AtDce();d.load_profiles(root+'/profile.json')
            self.assertIn(b'OK',d._execute(f'AT+CFG="serial.backend","{backend}"'))
            self.assertIn(b'OK',d._execute(f'AT+CFG="serial.device","{endpoint}"'))
            self.assertIn(b'OK',d._execute('AT&W'))
            loaded=AtDce();loaded.load_profiles(root+'/profile.json')
            self.assertEqual(loaded.extended_settings['serial.backend'],backend)
            self.assertEqual(loaded.extended_settings['serial.device'],endpoint)
    def test_hayes_roundtrip_and_case_preservation(self):
        d=AtDce()
        self.assertIn(b'OK',d._execute('AT+CFG="sip.user","MixedCase"'))
        self.assertEqual(d.net[102],'MixedCase')
        self.assertIn(b'MixedCase',d._execute('AT+CFG?'))
        self.assertIn(b'ERROR',d._execute('AT+CFG="dsp.eq_span","0"'))
        self.assertIn(b'ERROR',d._execute('AT+CFG="unknown","1"'))
    def test_trunk_choice_is_a_validated_switch(self):
        d=AtDce()
        self.assertEqual(set_value(d,'network.trunk','1'),'idle')
        rows={row['name']:row for row in snapshot(d)}
        self.assertEqual(rows['network.trunk']['value'],'1')
        with self.assertRaisesRegex(ValueError,'expected one of'):
            set_value(d,'network.trunk','2')
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
            if os.name=='posix':self.assertEqual(os.stat(root+'/profile.json').st_mode&0o777,0o600)
            else:self.assertTrue(os.access(root+'/profile.json',os.R_OK|os.W_OK)) # Windows access follows directory ACL, not POSIX mode bits.
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
    @unittest.skipUnless(os.name=='posix','POSIX serial backend only')
    def test_native_device_backend_uses_character_tty(self):
        import pty
        import select
        from modem_serial_device import DeviceSerial, SUPPORTED_BAUDS
        master,slave=pty.openpty();path=os.ttyname(slave)
        try:
            serial=DeviceSerial(path,115200)
            try:
                os.write(master,b'AT\r')
                readable,_,_=select.select(serial.fds(),[],[],.2)
                data,dropped=serial.service(readable,[])
                self.assertEqual(data,b'AT\r');self.assertFalse(dropped)
                serial.write(b'OK\r\n')
                self.assertEqual(os.read(master,5),b'OK\r\n')
                self.assertEqual(serial.name,path)
                self.assertIn(115200,SUPPORTED_BAUDS)
                terminal=serial.open_terminal()
                serial.terminal_write(b'ATI')
                self.assertEqual(serial.service([],[])[0],b'ATI')
                serial.write(b'EL-MODEM')
                self.assertEqual(serial.terminal_read(),b'EL-MODEM')
                serial.release_terminal(terminal)
            finally:serial.close()
        finally:os.close(master);os.close(slave)

    @unittest.skipUnless(os.name=='posix','POSIX serial backend only')
    def test_device_backend_config_is_switchable_and_exclusive(self):
        import pty
        with tempfile.TemporaryDirectory() as root:
            master,slave=pty.openpty();path=os.ttyname(slave)
            try:
                s=Supervisor(root+'/config.json');one=s.add(serial='device',device=path,baud=115200)
                self.assertEqual(one.serial,'device');self.assertEqual(one.serial_device,path)
                with self.assertRaisesRegex(ValueError,'colliding serial devices'):
                    s.add(serial='device',device=path)
                s.configure(one,{'serial':'pty','serial_device':''})
                self.assertEqual(s.specs[0].serial,'pty')
                self.assertEqual(Supervisor(root+'/config.json').specs[0].serial,'pty')
            finally:os.close(master);os.close(slave)

    @unittest.skipUnless(sys.platform=='win32','Windows named-pipe backend only')
    def test_pipe_names_are_exclusive_normalized_and_case_insensitive(self):
        from modem_serial_pipe import DEFAULT_PIPE_NAME
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.add(serial='pipe')
            for name in ('',DEFAULT_PIPE_NAME,'elmodemserial'):
                with self.assertRaisesRegex(ValueError,'colliding serial pipes'):
                    s.add(serial='pipe',device=name)
                self.assertEqual(len(s.specs),1)
            s.add(serial='pipe',device='OtherModem')
            self.assertEqual(len(Supervisor(root+'/config.json').specs),2)
            with self.assertRaisesRegex(ValueError,'colliding serial pipes'):
                s.configure(s.specs[1],{'serial_device':DEFAULT_PIPE_NAME})
    def test_instances_isolated_and_retained_ids_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');one=s.add(serial='pty');two=s.add(serial='tcp')
            self.assertNotEqual(one.sip_port,two.sip_port)
            (s.root/one.id).mkdir(parents=True)
            s.remove(one.id);three=s.add(serial='pty');self.assertNotEqual(three.id,one.id)
            self.assertEqual(len(Supervisor(root+'/config.json').specs),2)
    def test_runtime_port_repick_is_persisted_over_profile_override(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'config.json';s=Supervisor(path)
            spec=s.add(serial='tcp');other=s.add(serial='tcp')
            profile=s.root/spec.id/'profiles.json';profile.parent.mkdir(parents=True)
            profile.write_text(json.dumps({'profiles':{'0':{'extended_settings':{
                'serial.tcp_port':str(spec.serial_port)}}}}))
            old_port=spec.serial_port
            # The configured port became unavailable after add(). The first
            # alternative is reserved by another, not-yet-started modem; skip it.
            with patch.object(s,'_bindable',side_effect=[False,False,True,True]):
                chosen=s._refresh_ports_for_start(spec.id)
            self.assertNotEqual(chosen.serial_port,old_port)
            self.assertNotEqual(chosen.serial_port,other.serial_port)
            self.assertEqual(s.specs[0].serial_port,chosen.serial_port)
            saved=json.loads(path.read_text())['modems'][0]
            self.assertEqual(saved['serial_port'],chosen.serial_port)
            reloaded=Supervisor(path)
            self.assertEqual(reloaded.effective(reloaded.specs[0]).serial_port,chosen.serial_port)
    def test_worker_nonzero_exit_after_ready_is_still_failure(self):
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');spec=s.add(serial='pty');s.started.add(spec.id)
            class Process:
                def poll(self): return 1
            state=s._exit_status(spec.id,Process())
            self.assertEqual(state['state'],'failed')
            self.assertIn('status 1',state['error'])
            class CleanProcess:
                def poll(self): return 0
            self.assertEqual(s._exit_status(spec.id,CleanProcess())['state'],'stopped')
            s.started.clear()
            self.assertEqual(s._exit_status(spec.id,CleanProcess())['state'],'failed')
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
    def test_serial_choices_are_platform_specific(self):
        from modem_tui import serial_backend_choices
        self.assertEqual(serial_backend_choices('win32'),('tcp','pipe','com'))
        self.assertEqual(serial_backend_choices('darwin'),('pty','tcp','device'))
        self.assertEqual(serial_backend_choices('linux'),('driver','pty','tcp','device'))
    async def test_split_pane_focus_and_ascii_status_surface(self):
        from modem_tui import ModemApp
        from textual.widgets import Static
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.add(serial='pty');app=ModemApp(s)
            async with app.run_test(size=(120,45)) as pilot:
                await pilot.pause()
                panel=str(app.query_one('#modems',Static).renderable)
                self.assertIn('[1/1] Modem 1',panel)
                rows=panel.splitlines()
                # One bordered box per modem keeps several modems separate.
                rules=[r.strip() for r in rows if r.strip().startswith('+') and r.strip().endswith('+')]
                self.assertGreaterEqual(len(rules),2,'each modem needs a box above and below')
                stack=app.dashboard_widgets['stack']
                inner=max(37,stack.size.width-6)
                names=('HS','AA','CD','OH','RD','SD','TR','MR','RS','CS','SYN','ARQ')
                if inner>=62:
                    self.assertGreaterEqual(len(rules[0])-2,62,'wide box must fit the 60-column lamp row')
                    labels=[r for r in rows if all(n in r for n in names)]
                    self.assertEqual(len(labels),1,'wide labels must occupy one row')
                    bare=lambda r:r[2:-2][:60]
                    label_row=bare(labels[0])
                    lamp_row=bare(rows[rows.index(labels[0])-1])
                    self.assertEqual(len(label_row),60)
                    self.assertEqual(len(lamp_row),60)
                    for i,name in enumerate(names):
                        self.assertEqual(label_row[i*5:(i+1)*5].strip(),name)
                        self.assertRegex(lamp_row[i*5:(i+1)*5],r'^\s\(\.\)\s$')
                    self.assertTrue({'(*)','( )','(.)'}&set(lamp_row.split()))
                else:
                    columns=6 if inner>=40 else 4
                    lamp_rows=[r for r in rows if '(.)' in r]
                    self.assertEqual(len(lamp_rows),(len(names)+columns-1)//columns)
                    self.assertEqual([name for row in lamp_rows for name in names if name in row],list(names))
                # Tab is the only pane switch; the arrow keys belong to the focused pane.
                # Default level is the system menu; Enter enters modem actions.
                self.assertEqual(app.focus_pane,'right')
                self.assertIn('Add modem',str(app.query_one('#menu',Static).renderable))
                # Enter acts on whatever the > marks; Escape climbs back to the top and
                # is inert once there.
                await pilot.press('tab')
                self.assertEqual(app.focus_pane,'left','Tab reaches the modem stack')
                await pilot.press('enter')
                self.assertEqual(app.menu_level,'modem')
                self.assertIn('Settings',str(app.query_one('#menu',Static).renderable))
                await pilot.press('escape')
                self.assertEqual(app.menu_level,'system')
                self.assertIn('Add modem',str(app.query_one('#menu',Static).renderable))
                before=(app.focus_pane,app.menu_level)
                await pilot.press('escape')
                self.assertEqual((app.focus_pane,app.menu_level),before,'Escape at the top is inert')
                for key in ('left','right'):
                    await pilot.press(key)
                    self.assertEqual(app.focus_pane,'right',f'{key} must not switch panes')
            s.close()

    async def test_dashboard_indicator_cards_fit_80_columns(self):
        from modem_tui import ModemApp
        from textual.widgets import Static
        names=('HS','AA','CD','OH','RD','SD','TR','MR','RS','CS','SYN','ARQ')
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.add(serial='pty');app=ModemApp(s)
            async with app.run_test(size=(80,45)) as pilot:
                await pilot.pause()
                panel=app.query_one('#modems',Static)
                rendered=str(panel.renderable)
                rows=rendered.splitlines()
                stack=app.dashboard_widgets['stack']
                self.assertLess(stack.size.width,70,'80-column terminal should select compact layout')
                self.assertLessEqual(max(map(len,rows)),stack.size.width-4,
                                     f'modem cards must remain inside the left pane: {stack.size.width=} {rows!r}')
                inner=max(37,stack.size.width-6)
                columns=6 if inner>=40 else 4
                lamp_rows=[row for row in rows if '(.)' in row]
                self.assertEqual(len(lamp_rows),(len(names)+columns-1)//columns)
                self.assertEqual([name for row in lamp_rows for name in names if name in row],list(names))
            s.close()

    async def test_instance_dialog_mounts_and_saves(self):
        from modem_tui import ModemApp,serial_backend_choices
        with tempfile.TemporaryDirectory() as root:
            backend=serial_backend_choices()[0]
            s=Supervisor(root+'/config.json');s.add(serial=backend);app=ModemApp(s)
            async with app.run_test(size=(120,55)) as pilot:
                await pilot.pause();app.configure_selected();await pilot.pause()
                from textual.widgets import Select
                self.assertEqual(app.screen.query_one('#serial',Select).value,backend)
                await pilot.click('#save');await pilot.pause()
                self.assertEqual(s.specs[0].serial,backend)
            s.close()

    async def test_instance_dialog_offers_native_serial_device_on_posix(self):
        import sys
        if os.name!='posix':self.skipTest('POSIX only')
        from modem_tui import InstanceDialog,ModemApp,serial_backend_choices
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');spec=s.add(serial='pty');app=ModemApp(s)
            async with app.run_test(size=(120,55)) as pilot:
                app.push_screen(InstanceDialog(spec,lambda changes:None));await pilot.pause()
                self.assertIn('device',serial_backend_choices())
                if sys.platform=='darwin':self.assertNotIn('driver',serial_backend_choices())
                if sys.platform.startswith('linux'):self.assertIn('driver',serial_backend_choices())
            s.close()
    async def test_dashboard_dialog_and_settings_screen(self):
        from modem_tui import ModemApp,SettingsScreen
        from textual.widgets import Select,DataTable
        with tempfile.TemporaryDirectory() as root:
            s=Supervisor(root+'/config.json');s.add(serial='pty');app=ModemApp(s)
            async with app.run_test(size=(120,45)) as pilot:
                await pilot.pause();app.configure_selected();await pilot.pause()
                await pilot.click('#cancel');await pilot.pause()
                self.assertIn('[1/1] Modem 1',app.dashboard.query_one('#modems').renderable)
                app.push_screen(SettingsScreen('modem-1'));await pilot.pause()
                app.screen.load_rows(snapshot(AtDce()));await pilot.pause()
                app.screen.query_one('#category',Select).value='routing';await pilot.pause()
                self.assertEqual(app.screen.query_one(DataTable).row_count,1)
                self.assertEqual(app.screen.query_one(DataTable).get_row_at(0)[0],'network.trunk')
                app.screen.action_back();await pilot.pause()
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
                await pilot.press('a','t','q');await pilot.pause(.1)
                self.assertEqual(app.screen.query_one('#term-input').value,'atq')
                self.assertEqual(str(app.screen.query_one('#term-label').renderable),'DTE >')
                app.screen.data(b'REMOTE TEXT\r\n');await pilot.pause()
                self.assertIn('REMOTE TEXT',str(app.screen.query_one('#terminal-output').renderable))
                await pilot.press('enter','ctrl+c');await pilot.pause(.2)
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
