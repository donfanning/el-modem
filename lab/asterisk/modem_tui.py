"""Textual dashboard. UI never runs in a DSP worker."""
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static, TextArea

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
        yield Label(self.text)
        with Horizontal():yield Button('Confirm',id='yes',variant='warning');yield Button('Cancel',id='no')
    def on_button_pressed(self,event):
        event.stop()
        self.dismiss()
        if event.button.id=='yes':self.callback()

class InstanceDialog(ModalScreen):
    def __init__(self,spec,callback):super().__init__();self.spec=spec;self.callback=callback
    def compose(self):
        yield Label('Instance / serial / listening addresses (modem must be stopped)')
        for name in ('name','sip_host','sip_port','media_host','media_port','serial_port','serial_device','serial_baud'):
            yield Label(name);yield Input(str(getattr(self.spec,name,'')),id=name)
        yield Select([(x,x) for x in ('driver','pty','tcp','com','pipe')],value=self.spec.serial,id='serial',allow_blank=False)
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

class SettingsScreen(Screen):
    BINDINGS=[('escape','back','Back')]
    def __init__(self,ident):super().__init__();self.ident=ident;self.rows={}
    def compose(self):
        yield Header();yield Label('Modem settings — IAX is stored only; native IAX is unavailable')
        yield Select([(x.title(),x) for x in ('all','modem','sip','iax','advanced')],value='all',allow_blank=False,id='category')
        yield DataTable(id='settings')
        yield Label('Select a setting, edit its value, then Apply. Save profile makes changes survive reset/restart.')
        yield Input(id='value',placeholder='Value (password fields never read back)')
        with Horizontal():
            yield Button('Apply',id='apply');yield Button('Save profile',id='save');yield Button('Refresh',id='refresh');yield Button('Back',id='back')
        yield Footer()
    def on_mount(self):
        self.query_one(DataTable).add_columns('Setting','Value','Active','Apply');self.query_one(DataTable).cursor_type='row';self.refresh_settings()
    def refresh_settings(self):self.app.command(self.ident,'settings',self.load_rows)
    def load_rows(self,rows):self.rows={x['name']:x for x in rows};self.render_rows()
    def render_rows(self):
        table=self.query_one(DataTable);table.clear()
        category=self.query_one('#category',Select).value
        for name,row in self.rows.items():
            if category=='modem' and not (name.startswith('modem.') or name.startswith('s') and name[1:].isdigit()):continue
            if category in ('sip','iax') and not name.startswith(category+'.'):continue
            if category=='advanced' and not name.startswith(('dsp.','runtime.','recording.')):continue
            table.add_row(name,str(row['value']),str(row['active']),row['apply'],key=name)
    def on_select_changed(self,event):
        if self.rows:self.render_rows()
    def selected(self):
        table=self.query_one(DataTable)
        if not table.row_count:raise ValueError('select a setting')
        return str(table.get_row_at(table.cursor_row)[0])
    def on_data_table_row_highlighted(self,event):
        name=str(event.row_key.value)
        if name in self.rows:
            field=self.query_one('#value',Input);field.password=self.rows[name]['secret']
            field.value='' if self.rows[name]['secret'] else str(self.rows[name]['value'])
    def on_button_pressed(self,event):
        event.stop()
        if event.button.id=='back':self.action_back()
        elif event.button.id=='refresh':self.refresh_settings()
        elif event.button.id=='save':self.app.command(self.ident,'save',lambda _:self.app.notify('Profile saved'))
        elif event.button.id=='apply':
            try:self.app.command(self.ident,'set',lambda _:self.refresh_settings(),name=self.selected(),value=self.query_one('#value',Input).value)
            except ValueError as exc:self.app.notify(str(exc),severity='error')
    def action_back(self):self.app.pop_screen()

class TerminalScreen(Screen):
    BINDINGS=[Binding('f10','close','Release terminal',priority=True),Binding('f9','hangup','Hang up',priority=True)] + [
        Binding('ctrl+'+chr(96+n),f'send_control({n})',show=False,priority=True) for n in range(1,27)
    ] + [Binding(key,f'send_control({n})',show=False,priority=True)
         for key,n in [('enter',13),('backspace',8),('tab',9),('escape',27),('ctrl+space',0)]]
    def __init__(self,ident):super().__init__();self.ident=ident;self.buffer=AsciiBuffer()
    def compose(self):
        yield Header();yield Label('ASCII terminal — exclusive DTE ownership; F10 releases port using AT&D behavior')
        yield TextArea(read_only=True,id='terminal');yield Footer()
    def data(self,data):
        area=self.query_one(TextArea);area.load_text(self.buffer.feed(data));area.scroll_end(animate=False)
    def action_send_control(self, byte):
        self.app.command(self.ident,'terminal.write',None,hex=bytes([byte]).hex())
    async def on_key(self,event):
        mapping={'enter':13,'backspace':8,'tab':9,'escape':27,'ctrl+space':0}
        if event.key=='f10':return
        byte=mapping.get(event.key)
        if event.key.startswith('ctrl+') and len(event.key)==6 and 'a'<=event.key[-1]<='z':byte=ord(event.key[-1])-96
        if byte is None and event.character and len(event.character)==1 and ord(event.character)<128:byte=ord(event.character)
        if byte is not None:
            event.stop();event.prevent_default();self.app.command(self.ident,'terminal.write',None,hex=bytes([byte]).hex())
    def action_hangup(self):self.app.destructive(self.ident,lambda:self.app.command(self.ident,'hangup'))
    def action_close(self):self.app.command(self.ident,'terminal.close',lambda _:self.app.pop_screen())

class ModemApp(App):
    TITLE='EL-MODEM'
    CSS='''DataTable {height: 1fr;} Horizontal {height: auto;} Button {margin: 0 1;} ModalScreen {align: center middle; padding: 2 8; background: $surface;} ModalScreen Input {height: 3;} #terminal {height: 1fr;} #actions {grid-size: 4; grid-gutter: 0 1; height: 6;} #actions Button {width: 100%; margin: 0;} ModalScreen {overflow-y: auto;}'''
    BINDINGS=[Binding('ctrl+c','interrupt',show=False,priority=True),('q','quit_checked','Quit'),('n','create','New modem'),('s','settings','Settings'),('t','terminal','Terminal'),('h','hangup','Hang up')]
    def __init__(self,supervisor):super().__init__();self.supervisor=supervisor;self.callbacks={}
    def compose(self):
        yield Header();yield Label('Native DSP modems — user-selected instance count; monitor timing when adding calls')
        yield DataTable(id='modems')
        with Grid(id='actions'):
            for ident,title in [('new','New'),('configure','Instance'),('settings','Settings'),('terminal','Terminal'),('start','Start'),('stop','Stop'),('restart','Restart'),('remove','Remove')]:yield Button(title,id=ident)
        yield Static('',id='details');yield Footer()
    def on_mount(self):
        self.dashboard=self.screen
        self.dashboard_table=self.query_one('#modems',DataTable)
        self.details=self.query_one('#details',Static)
        self.query_one(DataTable).add_columns('Modem','State','Serial','Owner','SIP','Rate','DSP ms','AA CD OH RD SD TR MR RI RTS CTS')
        self.query_one(DataTable).cursor_type='row';self.set_interval(.1,self.tick);self.tick()
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
        table=self.dashboard_table;cursor=table.cursor_row;table.clear()
        for spec in self.supervisor.specs:
            state=self.supervisor.states.get(spec.id,{})
            lights=state.get('lights',{})
            lamps=' '.join('●' if lights.get(name) else '○' if lights.get(name) is not None else '—' for name in ('AA','CD','OH','RD','SD','TR','MR','RI','RTS','CTS'))
            try: name=self.supervisor.effective(spec).name
            except (ValueError,OSError,TypeError): name=spec.name
            table.add_row(name,state.get('state','stopped'),state.get('serial',''),state.get('owner',''),state.get('registration',''),str((state.get('call') or {}).get('speed','')),str(state.get('health',{}).get('mean_dsp_ms','')),lamps,key=spec.id)
        if table.row_count:table.move_cursor(row=min(cursor,table.row_count-1))
        if table.row_count:
            ident=str(table.ordered_rows[table.cursor_row].key.value)
            self.details.update(self.supervisor.states.get(ident,{}).get('error',''))
    def selected(self):
        table=self.dashboard.query_one('#modems',DataTable)
        if not table.row_count:raise ValueError('create a modem first')
        return str(table.ordered_rows[table.cursor_row].key.value)
    def action_hangup(self):
        ident=self.selected();self.destructive(ident,lambda:self.command(ident,'hangup'))
    def action_settings(self):self.push_screen(SettingsScreen(self.selected()))
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
    def on_button_pressed(self,event):
        event.stop()
        try:
            action=event.button.id
            if action=='new':self.action_create();return
            ident=self.selected()
            if action=='settings':self.action_settings()
            elif action=='terminal':self.action_terminal()
            elif action=='configure':
                if ident in self.supervisor.workers:raise ValueError('stop the modem before changing its instance configuration')
                spec=next(s for s in self.supervisor.specs if s.id==ident)
                self.push_screen(InstanceDialog(self.supervisor.effective(spec),lambda changes:self.configure_spec(spec,changes)))
            elif action=='start':self.supervisor.start(ident)
            elif action=='stop':self.destructive(ident,lambda:self.supervisor.stop(ident))
            elif action=='remove':self.destructive(ident,lambda:self.supervisor.remove(ident))
            elif action=='restart':
                def restart():self.supervisor.stop(ident);self.supervisor.start(ident)
                self.destructive(ident,restart)
        except (ValueError,OSError) as exc:self.notify(str(exc),severity='error')
    def action_interrupt(self):
        if isinstance(self.screen,TerminalScreen):self.screen.action_send_control(3)
        else:self.action_quit_checked()
    def action_quit_checked(self):
        active=any(x.get('state') in ('connected','dialing','ringing') for x in self.supervisor.states.values())
        if active:self.push_screen(Confirm('Hang up all owned calls and exit?',self.exit))
        else:self.exit()
