"""Local worker/serial integration. No dialing or external services."""
import os
import errno
from pathlib import Path
import select
import tempfile
import time
import unittest
from modem_supervisor import Supervisor

class InstanceTests(unittest.TestCase):
    def test_two_workers_isolate_profiles_and_terminal_ownership(self):
        with tempfile.TemporaryDirectory() as root:
            supervisor=Supervisor(root+'/config.json');supervisor.acquire()
            first=supervisor.add(serial='pty');second=supervisor.add(serial='pty')
            try:
                for spec in (first,second):supervisor.start(spec.id)
                deadline=time.monotonic()+90
                while time.monotonic()<deadline:
                    supervisor.poll()
                    if all(supervisor.states[s.id]['state']!='starting' for s in (first,second)):break
                    time.sleep(.02)
                for spec in (first,second):self.assertEqual(supervisor.states[spec.id]['state'],'idle')
                self.assertNotEqual(supervisor.workers[first.id][0].pid,supervisor.workers[second.id][0].pid)
                def request(ident,action,**fields):
                    key=supervisor.request(ident,action,**fields)
                    deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        supervisor.poll()
                        if key in supervisor.responses:return supervisor.responses.pop(key)
                        time.sleep(.01)
                    self.fail('request timed out')
                self.assertTrue(request(first.id,'set',name='sip.user',value='first-user')['ok'])
                self.assertTrue(request(first.id,'save')['ok'])
                settings=request(second.id,'settings')['value']
                self.assertEqual(next(x['value'] for x in settings if x['name']=='sip.user'),'')
                self.assertTrue(request(first.id,'terminal.open')['ok'])
                with self.assertRaises(OSError) as busy:
                    os.open(supervisor.states[first.id]['serial'],os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY)
                self.assertEqual(busy.exception.errno,errno.EBUSY)
                self.assertTrue(request(first.id,'terminal.write',hex=b'AT+MS?\r'.hex())['ok'])
                deadline=time.monotonic()+3;data=b''
                while time.monotonic()<deadline and b'OK' not in data:
                    supervisor.poll()
                    data+=b''.join(bytes.fromhex(value) for ident,value in supervisor.events if ident==first.id)
                    supervisor.events.clear();time.sleep(.01)
                self.assertIn(b'V34,1,300,33600',data)
                self.assertTrue(request(first.id,'terminal.close')['ok'])
                time.sleep(.2);supervisor.poll()
                endpoint=supervisor.states[first.id]['serial']
                fd=os.open(endpoint,os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY)
                try:
                    time.sleep(.2);supervisor.poll()
                    self.assertFalse(request(first.id,'terminal.open')['ok'])
                    self.assertTrue(request(first.id,'settings')['ok'])
                finally:os.close(fd)
                supervisor.stop(first.id)
                self.assertIsNone(supervisor.workers[second.id][0].poll())
                supervisor.start(first.id)
                deadline=time.monotonic()+90
                while time.monotonic()<deadline:
                    supervisor.poll()
                    if supervisor.states[first.id]['state']!='starting':break
                    time.sleep(.02)
                settings=request(first.id,'settings')['value']
                self.assertEqual(next(x['value'] for x in settings if x['name']=='sip.user'),'first-user')
            finally:supervisor.close()
            self.assertFalse(supervisor.workers)
            self.assertFalse((supervisor.root/first.id/'serial').is_symlink())
