"""Textual dashboard. UI never runs in a DSP worker."""
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Input, Label, Select, Static

def serial_backend_choices(platform=None):
    import sys
    platform=platform or sys.platform
    if platform=='win32':return ('tcp','pipe','com')
    if platform.startswith('linux'):return ('driver','pty','tcp','device')
    return ('pty','tcp','device')

SETTING_LABELS={
    'network.trunk':'Transport',
    'sip.server':'Server','sip.port':'Port','sip.user':'User','sip.password':'Password',
    'sip.registrar':'Registrar','sip.caller_number':'Caller ID','sip.contact_host':'Contact host',
    'iax.server':'Server','iax.port':'Port','iax.user':'User','iax.password':'Password',
    'iax.caller_number':'Caller ID','iax.register':'Register',
    'sip.bind_host':'Bind address','sip.bind_port':'SIP port',
    'sip.media_host':'Media address','sip.media_port':'RTP port',
    'modem.modulation':'Standard','modem.automatic':'Automode',
    'modem.min_rate':'Minimum rate','modem.max_rate':'Maximum rate',
    'modem.error_control':'Error control','modem.compression':'Compression',
    'recording.enabled':'Recording','recording.seconds':'Limit (seconds)',
    'serial.backend':'Port type','serial.device':'Device','serial.tcp_port':'TCP port',
    'serial.baud':'Baud rate','instance.name':'Name',
}
SIP_NETWORK_FIELDS={'sip.contact_host','sip.bind_host','sip.bind_port','sip.media_host','sip.media_port'}

def setting_label(name):
    if name in SETTING_LABELS:return SETTING_LABELS[name]
    if name.startswith('dsp.') or name.startswith('runtime.'):
        return name.split('.',1)[1].replace('_',' ').title()
    if name.startswith('modem.'):
        return name.split('.',1)[1].replace('_',' ').title()
    if name.startswith('network.'):
        return name.split('.',1)[1].replace('_',' ').title()
    if name.startswith('s') and name[1:].isdigit():return f'S-register {name[1:]}'
    return name.replace('_',' ').replace('.',' / ').title()

class AsciiBuffer:
    def __init__(self,limit=1000): self.lines=[''];self.limit=limit
    def feed(self,data):
        for byte in data:
            if byte==13: self.cursor=0
            elif byte==10: self.lines.append('');self.cursor=0
            elif byte in (8,127):
                self.cursor=max(0,getattr(self,'cursor',len(self.lines[-1]))-1)
                self.lines[-1]=self.lines[-1][:self.cursor]+self.lines[-1][self.cursor+1:]
            else:
                text=(' '* (8-getattr(self,'cursor',len(self.lines[-1]))%8) if byte==9 else
                      chr(byte) if 32<=byte<127 else f'<{byte:02X}>')
                position=getattr(self,'cursor',len(self.lines[-1]))
                self.lines[-1]=self.lines[-1][:position]+text+self.lines[-1][position+len(text):]
                self.cursor=position+len(text)
            if len(self.lines)>self.limit: del self.lines[:-self.limit]
            if len(self.lines[-1])>4096: self.lines[-1]=self.lines[-1][-4096:];self.cursor=len(self.lines[-1])
        return '\n'.join(self.lines)

class Confirm(ModalScreen):
    def __init__(self,text,callback):super().__init__();self.text=text;self.callback=callback
    def compose(self):
        with Vertical(id='confirm-window'):
            yield Static('+-- CONFIRM ----------------------------------------------------------+')
            yield Label(self.text)
            with Horizontal():yield Button('Confirm',id='yes',variant='warning');yield Button('Cancel',id='no')
            yield Static('+-----------------------------------------------------------------------+')
    def on_button_pressed(self,event):
        event.stop()
        self.dismiss()
        if event.button.id=='yes':self.callback()

class InstanceDialog(ModalScreen):
    def __init__(self,spec,callback):super().__init__();self.spec=spec;self.callback=callback
    def compose(self):
        serial_options=serial_backend_choices()
        with Vertical(id='instance-window'):
            yield Static(f'{self.spec.name.upper()}  /  CONNECTION',id='instance-title')
            yield Static('Stop modem to edit endpoints.',id='instance-note')
            labels={'name':'Name','sip_host':'SIP bind','sip_port':'SIP port',
                    'media_host':'Media bind','media_port':'RTP port',
                    'serial_port':'TCP port','serial_device':'Serial device','serial_baud':'Baud'}
            for name in ('name','sip_host','sip_port','media_host','media_port','serial_port','serial_device','serial_baud'):
                with Horizontal(classes='field'):
                    yield Label(labels[name])
                    yield Input(str(getattr(self.spec,name,'')),id=name)
            current=self.spec.serial if self.spec.serial in serial_options else serial_options[0]
            with Horizontal(classes='field'):
                yield Label('serial')
                yield Select([(x,x) for x in serial_options],value=current,id='serial',allow_blank=False)
            yield Static('',id='error')
            with Horizontal():yield Button('Save',id='save');yield Button('Cancel',id='cancel')
    def on_button_pressed(self,event):
        event.stop()
        if event.button.id=='cancel':self.dismiss();return
        try:
            changes={name:self.query_one('#'+name,Input).value for name in ('name','sip_host','sip_port','media_host','media_port','serial_port','serial_device','serial_baud')}
            for name in ('sip_port','media_port','serial_port','serial_baud'):changes[name]=int(changes[name])
            changes['serial']=self.query_one('#serial',Select).value
            self.callback(changes);self.dismiss()
        except (ValueError,OSError) as exc:self.query_one('#error',Static).update(str(exc))

class SettingsScreen(ModalScreen):
    BINDINGS=[('escape','back','Back')]
    def __init__(self,ident,category='modem'):super().__init__();self.ident=ident;self.rows={};self.visible_names=[];self.initial_category=category
    def compose(self):
        with Vertical(id='settings-window'):
            yield Static(f'{self.ident.upper()}  /  SETTINGS',id='settings-title')
            with Horizontal(id='settings-toolbar'):
                yield Static('VIEW',id='category-label')
                yield Select([('Modem','modem'),('SIP','sip'),('IAX2','iax'),
                              ('Route','routing'),('Advanced','advanced')],
                             value=self.initial_category,allow_blank=False,id='category')
            yield DataTable(id='settings')
            with Horizontal(id='field-editor'):
                yield Label('VALUE',id='value-label')
                yield Input(id='value',placeholder='Select a setting')
            yield Static('Choose a field',id='field-status')
            with Horizontal():
                yield Button('Apply',id='apply');yield Button('Save profile',id='save');yield Button('Close',id='back')
    def on_mount(self):
        self.query_one(DataTable).add_columns('SETTING','VALUE');self.query_one(DataTable).cursor_type='row';self.refresh_settings()
    def refresh_settings(self):self.app.command(self.ident,'settings',self.load_rows)
    def load_rows(self,rows):self.rows={x['name']:x for x in rows};self.render_rows()
    def render_rows(self):
        # The settings response can arrive after the dialog has been closed. The
        # callback then runs against a screen that no longer has its table, and
        # the resulting NoMatches escapes the tick loop and takes the dashboard
        # down. A closed screen simply has nothing left to draw.
        from textual.css.query import NoMatches
        try: table=self.query_one(DataTable)
        except NoMatches: return
        selected_before=(self.visible_names[table.cursor_row]
                         if table.row_count and table.cursor_row<len(self.visible_names) else None)
        table.clear()
        self.visible_names=[]
        category=self.query_one('#category',Select).value
        for name,row in self.rows.items():
            if category=='modem' and not (name.startswith('modem.') or name.startswith('s') and name[1:].isdigit()):continue
            if category=='sip' and (not name.startswith('sip.') or name in SIP_NETWORK_FIELDS):continue
            if category=='iax' and not name.startswith('iax.'):continue
            if category=='routing' and name!='network.trunk':continue
            if category=='advanced' and not (name.startswith(('dsp.','runtime.','recording.','serial.','instance.'))
                                             or name in SIP_NETWORK_FIELDS):continue
            shown=str(row['value'])
            if name=='network.trunk':shown={'0':'SIP','1':'IAX2'}.get(shown,shown)
            if row['secret']:shown='SET' if row['value'] else 'NOT SET'
            elif len(shown)>44:shown=shown[:41]+'...'
            state='~ ' if str(row['active'])!=str(row['value']) else ''
            table.add_row(setting_label(name),state+shown,key=name)
            self.visible_names.append(name)
        if table.row_count:
            row=self.visible_names.index(selected_before) if selected_before in self.visible_names else 0
            table.move_cursor(row=row)
        else:
            self.query_one('#value',Input).value=''
            self.query_one('#field-status',Static).update('No settings in this view')
    def on_select_changed(self,event):
        if self.rows:self.render_rows()
    def selected(self):
        table=self.query_one(DataTable)
        if not table.row_count:raise ValueError('select a setting')
        return self.visible_names[table.cursor_row]
    def on_data_table_row_highlighted(self,event):
        name=str(event.row_key.value)
        if name in self.rows:
            row=self.rows[name]
            field=self.query_one('#value',Input);field.password=row['secret']
            field.placeholder='Enter new password' if row['secret'] else 'Enter value'
            field.value='' if row['secret'] else str(row['value'])
            active=str(row['active']);configured=str(row['value'])
            timing={'immediate':'Now','idle':'When idle','restart':'After restart'}.get(row['apply'],row['apply'])
            pending=' | pending' if active!=configured else ''
            self.query_one('#value-label',Label).update(setting_label(name).upper())
            self.query_one('#field-status',Static).update(f'{timing}{pending}')
    def on_button_pressed(self,event):
        event.stop()
        if event.button.id=='back':self.action_back()
        elif event.button.id=='save':self.app.command(self.ident,'save',lambda _:self.app.notify('Profile saved'))
        elif event.button.id=='apply':
            try:self.app.command(self.ident,'set',lambda _:self.refresh_settings(),name=self.selected(),value=self.query_one('#value',Input).value)
            except ValueError as exc:self.app.notify(str(exc),severity='error')
    def action_back(self):self.app.pop_screen()

class TelephonySetupScreen(ModalScreen):
    """Small per-modem Asterisk setup flow; account secrets stay in the profile."""
    BINDINGS=[('escape','back','Back')]
    FIELDS=(('host','PBX host'),('port','SIP port'),('context','Call context'),
            ('transport','Transport'),('asterisk_dir','Asterisk config dir'))
    def __init__(self,ident):super().__init__();self.ident=ident;self.account={}
    def compose(self):
        with Vertical(id='provision-window'):
            yield Static(f'{self.ident.upper()}  /  ASTERISK SETUP',id='provision-title')
            yield Static('Account credentials stay in this modem profile.',id='provision-note')
            yield Static('Checking saved SIP account…',id='provision-account')
            for name,label in self.FIELDS:
                with Horizontal(classes='field'):
                    yield Label(label)
                    yield Input(id=name,placeholder=label)
            with Horizontal():
                yield Button('SIP settings',id='sip')
                yield Button('Refresh',id='refresh')
                yield Button('List contexts',id='contexts')
                yield Button('Save map',id='save-map')
                yield Button('Preview',id='preview')
            with Horizontal():
                yield Button('Apply to Asterisk',id='apply',variant='warning')
                yield Button('Check link',id='check')
                yield Button('Rollback last change',id='rollback')
                yield Button('Close',id='close')
            yield Static('Save SIP user, password and registrar in SIP Settings first.',id='provision-status')
    def on_mount(self):
        self.load_existing_map()
        self.refresh_account()
    def load_existing_map(self):
        import json
        from pathlib import Path
        path=self.app.supervisor.path.parent/'telephony-map.json'
        if not path.is_file():
            defaults={'host':'','port':'5060','context':'','transport':'udp-transport','asterisk_dir':''}
            for name,value in defaults.items():self.query_one('#'+name,Input).value=value
            return
        try:
            saved=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(saved,dict):raise ValueError('saved map must be an object')
            pbx=saved.get('pbx',{})
            if not isinstance(pbx,dict):raise ValueError('saved PBX settings must be an object')
            values={'host':pbx.get('host',''),'port':str(pbx.get('port',5060)),
                    'context':pbx.get('context','from-internal'),
                    'transport':pbx.get('transport','udp-transport'),
                    'asterisk_dir':pbx.get('config_dir','')}
            for name,value in values.items():self.query_one('#'+name,Input).value=str(value)
        except (OSError,ValueError,TypeError):
            self.query_one('#provision-status',Static).update('Saved telephony map is unreadable; repair it before continuing.')
    def refresh_account(self):
        from modem_provisioning import read_saved_network_profile
        profile_error=None
        try:
            net=read_saved_network_profile(self.app.supervisor.path,self.ident)
        except ValueError as exc:
            net={}
            profile_error=str(exc)
        user=str(net.get('102','')).strip()
        registrar=str(net.get('104','')).strip()
        password=str(net.get('103',''))
        if profile_error:
            state=profile_error
        elif user and password and registrar:
            state=f'Saved SIP account {user}  /  registrar {registrar}'
        else:
            state='SIP account incomplete. Set user, password and registrar in SIP Settings, then save profile.'
        self.query_one('#provision-account',Static).update(state)
        if registrar:
            host,sep,port=registrar.rpartition(':')
            self.query_one('#host',Input).value=host if sep and port.isdigit() else registrar
            if sep and port.isdigit():self.query_one('#port',Input).value=port
    def _mapping(self):
        import json
        from pathlib import Path
        host=self.query_one('#host',Input).value.strip()
        try:port=int(self.query_one('#port',Input).value)
        except ValueError as exc:raise ValueError('SIP port must be a number from 1 to 65535') from exc
        if not host or not 1<=port<=65535:raise ValueError('Enter a PBX host and SIP port')
        context=self.query_one('#context',Input).value.strip()
        if not context:raise ValueError('Enter an existing Asterisk dialplan context that you have reviewed')
        transport=self.query_one('#transport',Input).value.strip()
        config_dir=self.query_one('#asterisk_dir',Input).value.strip()
        path=self.app.supervisor.path.parent/'telephony-map.json'
        if path.is_file():
            current=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(current,dict):raise ValueError('Saved telephony map must contain an object')
            if current.get('schema')!=1:raise ValueError('Unsupported telephony map version')
            pbx=current.get('pbx',{})
            bindings=current.get('bindings',[])
            if not isinstance(pbx,dict):raise ValueError('Saved PBX settings are malformed')
            if not isinstance(bindings,list) or any(not isinstance(item,dict) for item in bindings):
                raise ValueError('Saved modem bindings are malformed')
            new_pbx=dict(type='asterisk',host=host,port=port,context=context,transport=transport)
            if config_dir:new_pbx['config_dir']=config_dir
            if bindings and {k:pbx.get(k) for k in ('type','host','port','context','transport')} != {
                    k:new_pbx.get(k) for k in ('type','host','port','context','transport')}:
                raise ValueError('This map already targets another Asterisk setup; keep its binding and use a separate map for this PBX.')
        else:
            current={'schema':1,'pbx':{},'bindings':[]}
            new_pbx=dict(type='asterisk',host=host,port=port,context=context,transport=transport)
            if config_dir:new_pbx['config_dir']=config_dir
            bindings=[]
        bindings=list(current.get('bindings',[]))
        if not any(item.get('modem')==self.ident for item in bindings):bindings.append({'modem':self.ident})
        return path,dict(schema=1,pbx=new_pbx,bindings=bindings)
    def save_map(self):
        import tempfile
        from pathlib import Path
        from modem_storage import atomic_json
        from modem_provision import adapter_for, build_provision_request
        path,mapping=self._mapping()
        # Validate the profile against this PBX target before replacing the
        # durable map; the temporary map contains no secrets.
        with tempfile.TemporaryDirectory(prefix='el-modem-map-',dir=path.parent) as tempdir:
            temporary=Path(tempdir)/'map.json'
            atomic_json(temporary,mapping)
            adapter_for(mapping['pbx']['type'])
            build_provision_request(self.app.supervisor.path,temporary)
        atomic_json(path,mapping)
        self.query_one('#provision-status',Static).update('Map saved privately. Preview or apply to update Asterisk.')
        return mapping
    async def plan_current_map(self):
        import asyncio
        import tempfile
        from pathlib import Path
        from modem_storage import atomic_json
        from modem_provision import adapter_for, build_provision_request
        path,mapping=self._mapping()
        config_dir=self.query_one('#asterisk_dir',Input).value.strip()
        target=Path(config_dir).expanduser().resolve() if config_dir else None
        with tempfile.TemporaryDirectory(prefix='el-modem-map-',dir=path.parent) as tempdir:
            temporary=Path(tempdir)/'map.json'
            atomic_json(temporary,mapping)
            adapter=adapter_for(mapping['pbx']['type'])
            request=build_provision_request(self.app.supervisor.path,temporary,target)
            plan=await asyncio.to_thread(adapter.plan,request)
            if not self.is_mounted:return None
            # Return only the safe summary/target outside the temporary map's
            # lifetime. The secret-bearing plan artifacts are not needed here.
            return mapping,plan.adapter_id,plan.target_id,plan.summary,plan.warnings
    async def list_contexts(self):
        import asyncio
        from pathlib import Path
        from modem_provision import _defined_dialplan_contexts
        try:
            directory_text=self.query_one('#asterisk_dir',Input).value.strip()
            if not directory_text:raise ValueError('Enter the Asterisk config directory first')
            dialplan=Path(directory_text).expanduser().resolve()/'extensions.conf'
            self.query_one('#provision-status',Static).update('Reading statically included dialplan contexts…')
            contexts=await asyncio.to_thread(_defined_dialplan_contexts,dialplan)
            if not self.is_mounted:return
            if not contexts:message='No static contexts found in extensions.conf or its includes.'
            else:message='Review one of these existing contexts, then enter it above: '+', '.join(sorted(contexts))
            self.query_one('#provision-status',Static).update(message)
        except (ValueError,OSError,TypeError) as exc:
            if self.is_mounted:self.query_one('#provision-status',Static).update(str(exc))
    def apply_map(self):
        import json
        from pathlib import Path
        from modem_storage import atomic_json
        from modem_provision import adapter_for, build_provision_request
        try:
            config_dir=self.query_one('#asterisk_dir',Input).value.strip()
            if not config_dir:raise ValueError('Enter the Asterisk config directory before applying')
            directory=Path(config_dir).expanduser().resolve()
            map_path=self.app.supervisor.path.parent/'telephony-map.json'
            if map_path.is_symlink():raise ValueError('Telephony map must not be a symlink')
            previous_map=json.loads(map_path.read_text(encoding='utf-8')) if map_path.is_file() else None
            mapping=self.save_map()
            adapter=adapter_for(mapping['pbx']['type'])
            try:
                plan=adapter.plan(build_provision_request(self.app.supervisor.path,map_path,directory))
                backups=adapter.apply(plan)
            except Exception as apply_error:
                try:
                    if previous_map is None:map_path.unlink(missing_ok=True)
                    else:atomic_json(map_path,previous_map)
                except (OSError,TypeError,ValueError) as restore_error:
                    raise RuntimeError(
                        f'PBX apply failed ({apply_error}); restoring the previous telephony map also failed '
                        f'({restore_error}). Check both the map and Asterisk files before retrying.') from apply_error
                if isinstance(apply_error,PermissionError):
                    self.query_one('#provision-status',Static).update(
                        'Asterisk config is not writable by this user. Previous map restored. Keep the TUI unprivileged; '
                        'run the documented `el-modem provision apply` command with administrator rights.')
                    return
                raise
            backup_text=(' Backups: '+', '.join(path.name for path in backups)) if backups else ''
            self.query_one('#provision-status',Static).update(
                'Asterisk files updated. Review, then reload PJSIP: asterisk -rx "pjsip reload".'+backup_text)
        except (ValueError,OSError,TypeError,RuntimeError) as exc:
            if self.is_mounted:self.query_one('#provision-status',Static).update(str(exc))
    async def check_link(self):
        import asyncio
        import tempfile
        from pathlib import Path
        from modem_storage import atomic_json
        from modem_provision import adapter_for, build_provision_request, load_plan_inputs
        try:
            map_path,mapping=self._mapping()
            directory_text=self.query_one('#asterisk_dir',Input).value.strip()
            if not directory_text:raise ValueError('Enter the Asterisk config directory before checking')
            directory=Path(directory_text).expanduser().resolve()
            adapter=adapter_for(mapping['pbx']['type'])
            self.query_one('#provision-status',Static).update('Checking current form values against PBX; saved map will not change…')
            with tempfile.TemporaryDirectory(prefix='el-modem-check-',dir=map_path.parent) as tempdir:
                temporary=Path(tempdir)/'map.json'
                atomic_json(temporary,mapping)
                request=build_provision_request(self.app.supervisor.path,temporary,directory)
                plan=await asyncio.to_thread(adapter.plan,request)
                if not self.is_mounted:return
                report=await asyncio.to_thread(adapter.check,plan,live=True)
                if not self.is_mounted:return
                details=['mapped PBX resources: '+('consistent' if report.consistent else 'inconsistent')]
                details.extend(report.details)
                details.extend([f'{modem}: {status}' for modem,status in report.registrations])
                details.extend([f'{modem} loaded endpoint: {status}' for modem,status in report.runtime])
                _,_,endpoints=load_plan_inputs(self.app.supervisor.path,temporary)
                account=next(item for item in endpoints if item['modem']==self.ident)
                registrations=dict(report.registrations)
                runtime=dict(report.runtime)
            if self.ident in self.app.supervisor.workers:
                self.query_one('#provision-status',Static).update('Checking live modem account, PBX endpoint and contact…')
                self.app.command(self.ident,'settings',
                    lambda rows:self.finish_live_check(rows,details,registrations,runtime,account))
            else:
                details.append(f'{self.ident} modem worker is stopped; live modem registration unavailable')
                self.query_one('#provision-status',Static).update('  |  '.join(details))
        except (ValueError,OSError,TypeError) as exc:
            if self.is_mounted:self.query_one('#provision-status',Static).update(str(exc))
    def rollback_map(self):
        from pathlib import Path
        from modem_provision import adapter_for
        from modem_provisioning import ProvisionTarget
        try:
            directory_text=self.query_one('#asterisk_dir',Input).value.strip()
            if not directory_text:raise ValueError('Enter the Asterisk config directory before rollback')
            directory=Path(directory_text).expanduser().resolve()
            target=ProvisionTarget('asterisk',str(directory),{'config_dir':str(directory)})
            try:
                restored=adapter_for('asterisk').rollback(target)
            except PermissionError:
                self.query_one('#provision-status',Static).update(
                    'Asterisk config is not writable by this user. Keep the TUI unprivileged; '
                    'run `el-modem provision rollback` with administrator rights.')
                return
            if restored:
                message='Restored '+', '.join(path.name for path in restored)+'. Review, then reload PJSIP.'
            else:
                message='No files needed restoring; the latest El-Modem transaction was already at its prior state.'
            self.query_one('#provision-status',Static).update(message)
        except (ValueError,OSError,TypeError) as exc:
            self.query_one('#provision-status',Static).update(str(exc))
    def finish_live_check(self,rows,details,registrations,runtime,account):
        from textual.css.query import NoMatches
        try:status=self.query_one('#provision-status',Static)
        except NoMatches:return
        active={item['name']:str(item['value']) for item in rows}
        user_match=active.get('sip.user','')==account['username']
        registrar_match=active.get('sip.registrar','').strip().casefold()==account['registrar'].strip().casefold()
        password_set=bool(active.get('sip.password',''))
        modem_registration=self.app.supervisor.states.get(self.ident,{}).get('registration','unknown')
        pbx_registration=registrations.get(self.ident,'not queried')
        pbx_runtime=runtime.get(self.ident,'not queried')
        if user_match and registrar_match and password_set:
            details.append(f'{self.ident} active SIP username/registrar match saved profile; password is set but masked, so value equality is not checked')
        else:
            details.append(f'{self.ident} active SIP username/registrar/password presence DIFFERS from saved profile')
        details.append(f'{self.ident} modem: {modem_registration}; PBX: {pbx_registration}; loaded endpoint: {pbx_runtime}')
        if (user_match and registrar_match and password_set and modem_registration=='registered'
                and pbx_registration.startswith('registered contact present (')
                and pbx_runtime=='runtime endpoint matches'):
            details.append('saved profile, active SIP identity, loaded PBX endpoint, and registration states agree; password values remain masked and are not compared')
        else:
            details.append('end-to-end registration consistency is not confirmed')
        status.update('  |  '.join(details))
    async def preview_plan(self):
        try:
            planned=await self.plan_current_map()
            if planned is None:return
            mapping,adapter_id,target,summary,warnings=planned
            self.query_one('#provision-status',Static).update(
                f'Plan: {adapter_id} at {target or "config directory not set"}: '
                + '  |  '.join(summary)+('  |  '+'  |  '.join(warnings) if warnings else ''))
        except (ValueError,OSError,TypeError) as exc:
            if self.is_mounted:self.query_one('#provision-status',Static).update(str(exc))
    async def on_button_pressed(self,event):
        event.stop()
        if event.button.id=='close':self.action_back()
        elif event.button.id=='refresh':self.refresh_account()
        elif event.button.id=='sip':self.app.push_screen(SettingsScreen(self.ident,'sip'))
        elif event.button.id=='save-map':
            try:self.save_map()
            except (ValueError,OSError,TypeError) as exc:self.query_one('#provision-status',Static).update(str(exc))
        elif event.button.id=='preview':await self.preview_plan()
        elif event.button.id=='contexts':await self.list_contexts()
        elif event.button.id=='apply':
            self.app.push_screen(Confirm('Update only El-Modem managed PJSIP resources?',self.apply_map))
        elif event.button.id=='check':await self.check_link()
        elif event.button.id=='rollback':
            self.app.push_screen(Confirm('Restore the last El-Modem change if PBX files are unchanged?',self.rollback_map))
    def action_back(self):self.app.pop_screen()

class TerminalScreen(Screen):
    BINDINGS=[Binding('f10','close','Release terminal',priority=True),Binding('f9','hangup','Hang up',priority=True)]
    def __init__(self,ident):super().__init__();self.ident=ident;self.buffer=AsciiBuffer();self.typed=''
    def compose(self):
        yield Static('EL-MODEM // ASCII TERMINAL     F9 HANGUP   F10 RELEASE   Ctrl-C BREAK',id='term-head')
        with VerticalScroll(id='terminal'):
            yield Static('',id='terminal-output')
        with Horizontal(id='term-entry'):
            yield Static('DTE >',id='term-label')
            yield Input(placeholder='Type to modem; Enter sends CR',id='term-input')
        yield Footer()
    def data(self,data):
        self.query_one('#terminal-output',Static).update(self.buffer.feed(data))
        self.query_one('#terminal',VerticalScroll).scroll_end(animate=False)
    def on_mount(self):
        # Claim DTE focus as soon as the screen mounts, then reassert it after
        # layout. On slower event loops the first rendered frame can otherwise
        # leave focus on the scroll pane long enough for an initial key to be
        # lost or handled by the dashboard.
        self.focus_input()
        self.call_after_refresh(self.focus_input)
    def focus_input(self):
        """Set DTE focus after the screen is laid out so the first key is not lost."""
        self.query_one('#term-input',Input).focus()
    def action_send_control(self, byte):
        self.app.command(self.ident,'terminal.write',None,hex=bytes([byte]).hex())
    async def on_key(self,event):
        mapping={'tab':9,'escape':27,'ctrl+space':0}
        byte=mapping.get(event.key)
        if event.key.startswith('ctrl+') and len(event.key)==6 and 'a'<=event.key[-1]<='z':byte=ord(event.key[-1])-96
        if byte is not None:
            event.stop();event.prevent_default()
            self.app.command(self.ident,'terminal.write',None,hex=bytes([byte]).hex())
    def on_input_changed(self,event):
        # Printable keys are sent immediately; this line is the local DTE echo
        # and editing surface, fixed below the scrolling receive display.
        value=event.value
        if value.startswith(self.typed):
            added=value[len(self.typed):]
            for char in added:
                if ord(char)<128:self.app.command(self.ident,'terminal.write',None,hex=char.encode('ascii').hex())
        elif self.typed.startswith(value):
            for _ in range(len(self.typed)-len(value)):self.app.command(self.ident,'terminal.write',None,hex=b'\x08'.hex())
        else:
            # An edit inside the line is represented as a backspace-and-retype
            # sequence on the DTE, which is the closest useful serial behavior.
            for _ in range(len(self.typed)):self.app.command(self.ident,'terminal.write',None,hex=b'\x08'.hex())
            for char in value:
                if ord(char)<128:self.app.command(self.ident,'terminal.write',None,hex=char.encode('ascii').hex())
        self.typed=value
    def on_input_submitted(self,event):
        self.app.command(self.ident,'terminal.write',None,hex=b'\r'.hex())
        self.typed=''
        self.query_one('#term-input',Input).value=''
    def action_hangup(self):self.app.destructive(self.ident,lambda:self.app.command(self.ident,'hangup'))
    def action_close(self):self.app.command(self.ident,'terminal.close',lambda _:self.app.pop_screen())

class ModemApp(App):
    TITLE='EL-MODEM'
    CSS='''
    Screen { background: #080d08; color: #c4d7b2; }
    Footer { background: #152315; color: #b9ca9d; }
    #masthead { height: 1; padding: 0 1; color: #f3c969; }
    #workarea { height: 1fr; margin: 0 1; }
    #stack, #context { border: ascii $success; padding: 0 1; }
    #stack { width: 2fr; }
    #context { width: 1fr; margin-left: 1; }
    #stack-title, #context-title { height: 1; color: #f3c969; text-style: bold; }
    #modems { height: 1fr; overflow-y: auto; }
    #menu { height: 1fr; }
    #help { height: 1; padding: 0 1; color: #f3c969; }
    #details { height: 2; color: #ef9a8d; }
    #term-head { height: 3; padding: 0 1; color: #f3c969; border: ascii $success; }
    #terminal { height: 1fr; border: ascii $success; padding: 0 1; color: #c4d7b2; }
    #terminal-output { width: 1fr; height: auto; color: #c4d7b2; }
    #term-entry { height: 3; margin-top: 1; border: ascii $warning; background: #101810; }
    #term-label { width: 7; height: 1; padding: 0 1; color: #f3c969; }
    #term-input { width: 1fr; height: 1; border: none; background: #101810; color: #f3c969; }
    ModalScreen { align: center middle; background: #000000 75%; }
    #settings-window, #instance-window, #confirm-window, #provision-window { width: 86%; max-width: 96; max-height: 90%; padding: 1 2; border: ascii $warning; background: #101810; }
    #settings-title, #instance-title { height: 1; color: #f3c969; text-style: bold; }
    #settings-toolbar { height: 3; margin: 1 0 0 0; }
    #category-label { width: 8; height: 3; content-align: left middle; color: #8fa17e; }
    #category { width: 1fr; height: 3; }
    #settings-window DataTable { height: 1fr; border: none; background: #101810; }
    #settings-window DataTable > .datatable--header { color: #8fa17e; }
    #field-editor { height: 3; margin-top: 1; }
    #value-label { width: 18; height: 3; content-align: left middle; color: #f3c969; }
    #settings-window #value { width: 1fr; }
    #field-status { height: 1; color: #8fa17e; }
    # Textual's Input draws a tall border. At height 1 both rows are border and
    # the text has nowhere to render, so the field looked empty and untypable.
    # Keep the normal height and put the label beside the field instead, which
    # is what brings the dialog back inside its own window.
    #settings-window Input, #instance-window Input { height: 3; }
    #instance-window .field { height: 3; }
    #instance-note { height: 1; color: #8fa17e; }
    #instance-window .field Label { width: 18; content-align: left middle; }
    #instance-window .field Input, #instance-window .field Select { width: 1fr; }
    #instance-window .field Select { height: 3; }
    #instance-window { overflow-y: auto; }
    #provision-title { height: 1; color: #f3c969; text-style: bold; }
    #provision-note, #provision-account { height: 1; color: #8fa17e; }
    #provision-status { height: auto; max-height: 4; color: #8fa17e; }
    #provision-window { overflow-y: auto; }
    #provision-window .field { height: 3; }
    #provision-window .field Label { width: 18; content-align: left middle; color: #f3c969; }
    #provision-window .field Input { width: 1fr; height: 3; }
    #provision-window Button, #settings-window Button, #instance-window Button, #confirm-window Button { margin: 0 1; }
    '''
    BINDINGS=[Binding('ctrl+c','interrupt',show=False,priority=True),('q','quit_checked','Quit'),('n','create','New modem'),('s','settings','Settings'),('t','terminal','Terminal'),('h','hangup','Hang up')]
    def __init__(self,supervisor):super().__init__();self.supervisor=supervisor;self.callbacks={}
    def compose(self):
        yield Static('',id='masthead')
        with Horizontal(id='workarea'):
            with VerticalScroll(id='stack'):
                yield Static(' MODEM STACK',id='stack-title')
                yield Static('',id='modems')
            with Vertical(id='context'):
                yield Static(' SYSTEM ACTIONS',id='context-title')
                yield Static('',id='menu')
        yield Static('',id='details')
        yield Static('[Enter] act on >  [Esc] up a level  [↑/↓] move >  [Tab] switch pane  [N] add  [Q] quit',id='help')
        yield Footer()
    def on_mount(self):
        self.dashboard=self.screen
        self.dashboard_widgets={w.id:w for w in self.dashboard.query('*') if w.id}
        self.dashboard_table=self.dashboard_widgets['modems']
        self.details=self.dashboard_widgets['details']
        # The action menu has two levels. The system menu is the default because a
        # modem is not being acted on until you choose one; Enter enters the
        # modem's own actions, Escape steps back out to the system menu.
        self.menu_level='system';self.focus_pane='right';self.modem_row=0;self.menu_row=0
        self.set_interval(.1,self.tick);self.set_interval(1,self.clock_tick);self.tick();self.clock_tick()
    def on_resize(self,event):
        # The initial mount tick can run before Textual has distributed the
        # horizontal fr widths. Re-render cards after layout so the lamp row
        # is sized for the actual pane, not the transient mount width.
        table=getattr(self,'dashboard_table',None)
        if table is not None and table.is_mounted:
            self.call_after_refresh(self.tick)
    def clock_tick(self):
        from datetime import datetime
        masthead=self.dashboard_widgets.get('masthead')
        if not (masthead and masthead.is_mounted):return
        # Pad between the two halves so the clock sits flush at the right edge
        # of the current terminal width, instead of a fixed column that drifts.
        left='EL-MODEM  |  DSP MODEM POOL'
        right=datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        gap=max(1,self.size.width-2-len(left)-len(right))
        masthead.update(left+' '*gap+right)
    def check_action(self, action, parameters):
        if isinstance(self.screen,TerminalScreen) and action in ('quit_checked','create','settings','terminal','hangup'):
            return False
        return True
    def command(self,ident,action,callback=None,**fields):
        try:
            request=self.supervisor.request(ident,action,**fields);self.callbacks[request]=(callback,__import__('time').monotonic()+5)
        except (ValueError,OSError,BufferError) as exc:self.notify(str(exc),severity='error')
    def tick(self):
        self.supervisor.poll()
        import time
        for request,(callback,deadline) in list(self.callbacks.items()):
            response=self.supervisor.responses.pop(request,None)
            if response is not None:
                del self.callbacks[request]
                if response['ok']:
                    if callback:callback(response.get('value'))
                else:self.notify(response['error'],severity='error')
            elif time.monotonic()>deadline:
                del self.callbacks[request];self.notify('Worker request timed out',severity='error')
        for ident,data in self.supervisor.events:
            if isinstance(self.screen,TerminalScreen) and self.screen.ident==ident:self.screen.data(bytes.fromhex(data))
        self.supervisor.events.clear()
        if not self.dashboard_table.is_mounted:return
        cards=[]; self.visible_ids=[]
        # One bordered box per modem so several modems stay visually separate.
        # At narrow terminal widths, split the 12 lamps into rows that fit;
        # otherwise the fixed 60-column row overflows the two-thirds stack pane.
        # The card's rules and side borders must fit inside the widget's border
        # and padding, hence the six-column allowance below.
        inner=max(37,self.dashboard_widgets['stack'].size.width-6)
        rule='+'+'-'*inner+'+'
        for spec in self.supervisor.specs:
            state=self.supervisor.states.get(spec.id,{})
            lights=state.get('lights',{})
            names=('HS','AA','CD','OH','RD','SD','TR','MR','DC','CS','EC','ARQ')
            # Owner-selected twelve-slot panel: RS -> DC, SYN -> EC.
            # Retain ARQ alongside EC; both report negotiated error control.
            mapped={'HS':lights.get('HS'),'AA':lights.get('AA'),'CD':lights.get('CD'),'OH':lights.get('OH'),'RD':lights.get('RD'),'SD':lights.get('SD'),'TR':lights.get('TR'),'MR':lights.get('MR'),'DC':lights.get('DC'),'CS':lights.get('CTS'),'EC':lights.get('EC'),'ARQ':lights.get('ARQ')}
            # Three states: (*) on, ( ) off, (.) not reported. The third state
            # matters: a modem that failed to start must not look identical
            # to a healthy idle one.
            lamp=lambda n:'(*)' if mapped[n] is True else '( )' if mapped[n] is False else '(.)'
            if inner>=62:
                # Wide layout: fixed 5-column slots, 12 indicators per row.
                lamps=''.join(f' {lamp(n)} ' for n in names)
                labels=''.join(f' {n}  ' for n in names)
                indicator_rows=[lamps,labels]
            else:
                # Compact layout: use six columns when they fit, otherwise
                # three rows of four. ARQ takes one extra column, so the
                # narrowest card must not truncate the final indicator.
                columns=6 if inner>=40 else 4
                indicator_rows=[' '.join(f'{lamp(n)}{n}' for n in names[start:start+columns])
                                for start in range(0,len(names),columns)]
            try: name=self.supervisor.effective(spec).name
            except (ValueError,OSError,TypeError): name=spec.name
            self.visible_ids.append(spec.id)
            idx=len(self.visible_ids);focus='>' if self.focus_pane=='left' and idx-1==self.modem_row else ' '
            call=state.get('call') or {}
            running=spec.id in self.supervisor.workers
            availability=('STARTING' if running and state.get('state')=='starting' else
                          'ONLINE' if running else 'OFFLINE')
            rows=[f"{focus} [{idx}/{len(self.supervisor.specs)}] {name} | {state.get('port') or state.get('serial','')} | "
                  f"Ext {state.get('extension','—')} | {state.get('registration','unknown').upper()} | "
                  f"{availability}",
                  *indicator_rows,
                  f"DSP {state.get('health',{}).get('mean_dsp_ms','—')} ms   Owner {state.get('owner','—')}   "
                  f"{call.get('speed','')}"]
            if inner<62:
                rows[0:1]=[f"{focus} [{idx}/{len(self.supervisor.specs)}] {name}",
                           f"{state.get('port') or state.get('serial','')} Ext {state.get('extension','—')} | "
                           f"{state.get('registration','unknown').upper()} | "
                           f"{'START' if availability=='STARTING' else 'ON' if running else 'OFF'}"]
            rows=[r[:inner-2] for r in rows]
            self.card_stride=len(rows)+3 # top/bottom rules plus separator before next card
            cards.append(rule+''.join('\n| '+r.ljust(inner-2)+' |' for r in rows)+'\n'+rule)
        self.dashboard_table.update('\n'.join(cards) if cards else '  No modem instances. Press N to add one.')
        # 'Settings' edits live modem values; 'Instance' edits name/port/baud and
        # needs the modem stopped. Distinct labels so they are not confusable.
        selected=self.visible_ids[min(self.modem_row,len(self.visible_ids)-1)] if self.visible_ids else None
        start_label='Stop' if selected in self.supervisor.workers else 'Start'
        actions=(['Settings','Terminal','Statistics','Instance','Start / Stop','Hang up','Remove','Telephony setup'],['Add modem','Save config','Load config','Background','Quit'])
        actions=(['Settings','Terminal','Statistics','Instance',f'{start_label} modem','Hang up','Remove','Telephony setup'],actions[1])
        labels=actions[0] if self.in_modem_menu() else actions[1]
        self.dashboard_widgets['context-title'].update(' MODEM ACTIONS' if self.in_modem_menu() else ' SYSTEM ACTIONS')
        # Only mark the pane that actually has the keyboard. Up/Down moves modem_row
        # while the left pane is focused and menu_row while the right pane is,
        # so a marker on the unfocused menu just sat frozen on one entry.
        marker='>' if self.focus_pane=='right' else ' '
        self.dashboard_widgets['menu'].update('\n'.join((marker if i==self.menu_row else ' ')+f' {i+1}. {item}' for i,item in enumerate(labels)))
        self.dashboard_widgets['stack'].border_title='FOCUS' if self.focus_pane=='left' else 'MODEMS'
        self.dashboard_widgets['context'].border_title='FOCUS' if self.focus_pane=='right' else 'ACTIONS'
        self.modem_row=min(self.modem_row,max(0,len(self.visible_ids)-1))
        if self.visible_ids:
            ident=self.visible_ids[self.modem_row]
            self.details.update(self.supervisor.states.get(ident,{}).get('error',''))
    def on_key(self,event):
        if isinstance(self.screen,TerminalScreen):return
        if event.key=='tab':
            # Tab is the only pane switch. Up/Down always moves the marker in
            # whichever pane holds focus, so the arrow keys mean the same thing
            # in both panes instead of only when an "actions" menu is showing.
            event.stop();event.prevent_default();self.focus_pane='right' if self.focus_pane=='left' else 'left';self.tick()
        elif event.key in ('up','down'):
            event.stop();event.prevent_default()
            if self.focus_pane=='left':self.modem_row=max(0,min(len(self.supervisor.specs)-1,self.modem_row+(-1 if event.key=='up' else 1)))
            else:self.menu_row=max(0,min(self.menu_count()-1,self.menu_row+(-1 if event.key=='up' else 1)))
            self.tick()
            if self.focus_pane=='left':self.dashboard_widgets['stack'].scroll_to(y=self.modem_row*getattr(self,'card_stride',7),animate=False)
        elif event.key=='escape':
            # Climb one level: out of the modem's actions, back to the system
            # menu at the top. Escape is already at the top and does nothing,
            # and it never stops a running modem.
            event.stop();event.prevent_default()
            if self.menu_level=='modem':
                self.menu_level='system';self.focus_pane='right';self.menu_row=0
            self.tick()
        elif event.key=='enter':
            # Enter acts on whatever the marker is on: a modem in the stack
            # opens its actions, an entry in the menu runs that action.
            event.stop();event.prevent_default()
            if self.focus_pane=='left':
                self.menu_level='modem';self.focus_pane='right';self.menu_row=0
            else:
                self.activate_menu()
            self.tick()
        elif self.focus_pane=='left' and event.character and event.character in '12345678':
            self.menu_row=int(event.character)-1;self.activate_menu()
    def menu_count(self):
        """How many entries the action menu currently offers."""
        return 8 if self.in_modem_menu() else 5
    def in_modem_menu(self):
        """True once Enter has moved from the system menu into the modem's actions."""
        return self.menu_level=='modem' and bool(self.visible_ids)
    def activate_menu(self):
        if not self.in_modem_menu():
            (self.action_create,self.save_config,self.load_config,self.background,self.action_quit_checked)[self.menu_row]()
        else:
            actions=(self.action_settings,self.action_terminal,self.show_statistics,self.configure_selected,self.toggle_selected,self.action_hangup,self.remove_selected,self.action_telephony_setup)
            if self.visible_ids:actions[self.menu_row]()
    def selected(self):
        if not self.supervisor.specs:raise ValueError('create a modem first')
        self.modem_row=min(self.modem_row,len(self.supervisor.specs)-1)
        return self.supervisor.specs[self.modem_row].id
    def save_config(self):self.supervisor.save();self.notify('Configuration saved')
    def load_config(self):
        import json
        from modem_supervisor import ModemSpec
        if self.supervisor.workers:
            self.notify('Stop all modem workers before reloading configuration',severity='warning');return
        try:
            data=json.loads(self.supervisor.path.read_text())
            if data.get('schema')!=1:raise ValueError('unsupported configuration schema')
            old=self.supervisor.specs
            self.supervisor.specs=[ModemSpec(**item).validate() for item in data['modems']]
            try:self.supervisor.validate()
            except BaseException:self.supervisor.specs=old;raise
            self.notify('Configuration loaded')
        except (OSError,ValueError,TypeError,KeyError) as exc:self.notify(f'Configuration not loaded: {exc}',severity='error')
    def background(self):self.notify('Use Ctrl-Z in the launching terminal to background the process')
    def show_statistics(self):
        state=self.supervisor.states.get(self.selected(),{})
        self.push_screen(Confirm(f"Statistics: {state.get('state','unknown')} | {state.get('call') or 'No active call'} | {state.get('health') or 'No DSP metrics'}",lambda:None))
    def configure_selected(self):
        ident=self.selected()
        if ident in self.supervisor.workers:raise ValueError('stop the modem before changing its instance configuration')
        spec=next(s for s in self.supervisor.specs if s.id==ident)
        self.push_screen(InstanceDialog(self.supervisor.effective(spec),lambda changes:self.configure_spec(spec,changes)))
    def toggle_selected(self):
        ident=self.selected()
        if ident in self.supervisor.workers:self.destructive(ident,lambda:self.supervisor.stop(ident))
        else:self.supervisor.start(ident)
    def remove_selected(self):
        ident=self.selected();self.destructive(ident,lambda:self.supervisor.remove(ident))
    def action_hangup(self):
        ident=self.selected();self.destructive(ident,lambda:self.command(ident,'hangup'))
    def action_settings(self):self.push_screen(SettingsScreen(self.selected()))
    def action_telephony_setup(self):self.push_screen(TelephonySetupScreen(self.selected()))
    def action_terminal(self):
        ident=self.selected();self.command(ident,'terminal.open',lambda _:self.push_screen(TerminalScreen(ident)))
    def action_create(self):
        from modem_supervisor import ModemSpec
        def save(changes):
            spec=self.supervisor.add(name=changes['name'],serial=changes['serial'],
                                     device=changes.get('serial_device',''),
                                     baud=changes.get('serial_baud',115200))
            try:self.configure_spec(spec,changes)
            except BaseException:self.supervisor.specs.remove(spec);self.supervisor.save();raise
        number=1
        while any(s.id==f'modem-{number}' for s in self.supervisor.specs) or (self.supervisor.root/f'modem-{number}').exists(): number+=1
        self.push_screen(InstanceDialog(ModemSpec(f'modem-{number}',f'Modem {number}',sip_port=5060+number*2,media_port=15996+number*4,serial_port=9099+number),save))
    def configure_spec(self,spec,changes):
        self.supervisor.configure(spec,changes)
    def destructive(self,ident,callback):
        state=self.supervisor.states.get(ident,{}).get('state')
        if state in ('connected','dialing','ringing'):self.push_screen(Confirm('This will terminate the active call.',callback))
        else:callback()
    def action_interrupt(self):
        if isinstance(self.screen,TerminalScreen):self.screen.action_send_control(3)
        else:self.action_quit_checked()
    def action_quit_checked(self):
        active=any(x.get('state') in ('connected','dialing','ringing') for x in self.supervisor.states.values())
        if active:self.push_screen(Confirm('Hang up all owned calls and exit?',self.exit))
        else:self.exit()
