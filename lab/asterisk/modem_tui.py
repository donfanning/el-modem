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
            yield Static('+-- INSTANCE / CONNECTION SETUP -------------------------------------+')
            yield Static('The modem must be stopped before changing these values.')
            for name in ('name','sip_host','sip_port','media_host','media_port','serial_port','serial_device','serial_baud'):
                # Label and field share a row: stacked at height 3 each, the
                # eight fields pushed Save below the dialog's own bottom edge
                # and the button could not be clicked at all.
                with Horizontal(classes='field'):
                    yield Label(name)
                    yield Input(str(getattr(self.spec,name,'')),id=name)
            current=self.spec.serial if self.spec.serial in serial_options else serial_options[0]
            with Horizontal(classes='field'):
                yield Label('serial')
                yield Select([(x,x) for x in serial_options],value=current,id='serial',allow_blank=False)
            yield Static('',id='error')
            with Horizontal():yield Button('Save',id='save');yield Button('Cancel',id='cancel')
            yield Static('+-----------------------------------------------------------------------+')
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
    def __init__(self,ident):super().__init__();self.ident=ident;self.rows={}
    def compose(self):
        with Vertical(id='settings-window'):
            yield Static('+-- MODEM CONFIGURATION ----------------------------------------------+')
            yield Static('Settings are per modem. SIP and IAX2 share the modem DSP and data-link stack.')
            yield Select([(x.title(),x) for x in ('all','modem','sip','iax','routing','advanced')],value='all',allow_blank=False,id='category')
            yield DataTable(id='settings')
            yield Static('Select a row, edit the value below, then Apply. Save profile persists changes.')
            yield Input(id='value',placeholder='Value (password fields never read back)')
            with Horizontal():
                yield Button('Apply',id='apply');yield Button('Save profile',id='save');yield Button('Refresh',id='refresh');yield Button('Close [Esc]',id='back')
            yield Static('+-----------------------------------------------------------------------+')
    def on_mount(self):
        self.query_one(DataTable).add_columns('Setting','Value','Active','Apply');self.query_one(DataTable).cursor_type='row';self.refresh_settings()
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
        table.clear()
        category=self.query_one('#category',Select).value
        for name,row in self.rows.items():
            if category=='modem' and not (name.startswith('modem.') or name.startswith('s') and name[1:].isdigit()):continue
            if category in ('sip','iax') and not name.startswith(category+'.'):continue
            if category=='routing' and name!='network.trunk':continue
            if category=='advanced' and not name.startswith(('dsp.','runtime.','recording.')):continue
            shown=str(row['value'])
            if name=='network.trunk':shown={'0':'0 (SIP)','1':'1 (IAX2)'} .get(shown,shown)
            table.add_row(name,shown,str(row['active']),row['apply'],key=name)
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
    def on_mount(self):self.query_one('#term-input',Input).focus()
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
    #settings-window, #instance-window, #confirm-window { width: 90%; max-width: 110; max-height: 90%; padding: 1 2; border: ascii $warning; background: #101810; }
    #settings-window DataTable { height: 1fr; }
    # Textual's Input draws a tall border. At height 1 both rows are border and
    # the text has nowhere to render, so the field looked empty and untypable.
    # Keep the normal height and put the label beside the field instead, which
    # is what brings the dialog back inside its own window.
    #settings-window Input, #instance-window Input { height: 3; }
    #instance-window .field { height: 3; }
    #instance-window .field Label { width: 16; content-align: left middle; }
    #instance-window .field Input, #instance-window .field Select { width: 1fr; }
    #instance-window .field Select { height: 3; }
    #instance-window { overflow-y: auto; }
    #settings-window Button, #instance-window Button, #confirm-window Button { margin: 0 1; }
    #settings-window Select { width: 1fr; }
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
            names=('HS','AA','CD','OH','RD','SD','TR','MR','RS','CS','SYN','ARQ')
            mapped={'HS':None,'AA':lights.get('AA'),'CD':lights.get('CD'),'OH':lights.get('OH'),'RD':lights.get('RD'),'SD':lights.get('SD'),'TR':None,'MR':lights.get('MR'),'RS':lights.get('RTS'),'CS':lights.get('CTS'),'SYN':None,'ARQ':None}
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
            rows=[f"{focus} [{idx}/{len(self.supervisor.specs)}] {name} | {state.get('serial','')} | "
                  f"Ext {state.get('extension','—')} | {state.get('registration','unknown').upper()} | "
                  f"{availability}",
                  *indicator_rows,
                  f"DSP {state.get('health',{}).get('mean_dsp_ms','—')} ms   Owner {state.get('owner','—')}   "
                  f"{call.get('speed','')}"]
            if inner<62:
                rows[0:1]=[f"{focus} [{idx}/{len(self.supervisor.specs)}] {name}",
                           f"{state.get('serial','')} Ext {state.get('extension','—')} | "
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
        actions=(['Settings','Terminal','Statistics','Instance','Start / Stop','Hang up','Remove'],['Add modem','Save config','Load config','Background','Quit'])
        actions=(['Settings','Terminal','Statistics','Instance',f'{start_label} modem','Hang up','Remove'],actions[1])
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
        elif self.focus_pane=='left' and event.character and event.character in '1234567':
            self.menu_row=int(event.character)-1;self.activate_menu()
    def menu_count(self):
        """How many entries the action menu currently offers."""
        return 7 if self.in_modem_menu() else 5
    def in_modem_menu(self):
        """True once Enter has moved from the system menu into the modem's actions."""
        return self.menu_level=='modem' and bool(self.visible_ids)
    def activate_menu(self):
        if not self.in_modem_menu():
            (self.action_create,self.save_config,self.load_config,self.background,self.action_quit_checked)[self.menu_row]()
        else:
            actions=(self.action_settings,self.action_terminal,self.show_statistics,self.configure_selected,self.toggle_selected,self.action_hangup,self.remove_selected)
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
