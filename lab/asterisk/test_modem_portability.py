"""No calls: portable worker control, TCP terminal ownership and shutdown."""
import os
import json
import subprocess
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from modem_supervisor import Supervisor
from modem_serial import default_backend

class PortableWorkerTests(unittest.TestCase):
    def test_platform_defaults(self):
        for platform,expected in [('linux','driver'),('win32','tcp'),('darwin','pty')]:
            with patch.object(sys,'platform',platform):self.assertEqual(default_backend(),expected)

    def test_authenticated_tcp_worker_and_ascii_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor=Supervisor(Path(folder)/'modems.json')
            spec=supervisor.add(serial='tcp')
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1',0));spec.serial_port=reserve.getsockname()[1]
            supervisor.save()
            try:
                # Exercise the Windows control transport with a real child even
                # on Linux; native OS builds must additionally run this test.
                original_spawn=subprocess.Popen
                def spawn_after_unauthorized_client(*args,**kwargs):
                    endpoint=json.loads((supervisor.root/spec.id/'worker.json').read_text())['control']
                    with socket.create_connection(('127.0.0.1',endpoint['port'])) as unauthorized:
                        unauthorized.sendall(b'0'*64)
                    return original_spawn(*args,**kwargs)
                with patch.object(sys,'platform','win32'),patch('modem_supervisor.subprocess.Popen',side_effect=spawn_after_unauthorized_client):
                    supervisor.start(spec.id)
                child=supervisor.workers[spec.id][0]
                deadline=time.monotonic()+90
                while time.monotonic()<deadline:
                    supervisor.poll()
                    if supervisor.states[spec.id]['state']!='starting':break
                    time.sleep(.02)
                self.assertEqual(supervisor.states[spec.id]['state'],'idle',(supervisor.states,(supervisor.root/spec.id/'worker.log').read_text()))
                def request(action,**fields):
                    key=supervisor.request(spec.id,action,**fields);deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        supervisor.poll()
                        if key in supervisor.responses:return supervisor.responses.pop(key)
                        time.sleep(.01)
                    self.fail('control response timed out')
                self.assertTrue(request('terminal.open')['ok'])
                self.assertTrue(request('terminal.write',hex=b'ATI3\r'.hex())['ok'])
                data=b'';deadline=time.monotonic()+5
                while time.monotonic()<deadline and b'OK' not in data:
                    supervisor.poll();data+=b''.join(bytes.fromhex(value) for ident,value in supervisor.events)
                    supervisor.events.clear();time.sleep(.02)
                self.assertIn(b'donfanning/el-modem',data)
                self.assertTrue(request('terminal.close')['ok'])
            finally:supervisor.close()
            self.assertFalse(supervisor.workers)
            self.assertEqual(child.returncode,0,'worker must exit cooperatively, not be forcibly terminated')

if __name__=='__main__':unittest.main()
