"""Real prerequisite-driver tests; explicitly opt in after setup."""
import array
import fcntl
import os
import select
import termios
import time
import unittest
from modem_serial import DriverSerial,GET_REMOTE

@unittest.skipUnless(os.environ.get('EL_MODEM_TEST_DRIVER')=='1','requires installed patched serialsim driver')
class DriverTests(unittest.TestCase):
    def setUp(self):self.serial=DriverSerial(None);self.client=self.serial.open_terminal();time.sleep(.05)
    def tearDown(self):os.close(self.client);self.serial.close()
    def read(self,fd,timeout=.3):
        if select.select([fd],[],[],timeout)[0]:return os.read(fd,65536)
        return b''
    def test_signal_mapping_and_exclusive(self):
        self.assertEqual(self.serial.get_open_count(),1)
        with self.assertRaises(OSError):os.open(self.serial.client,os.O_RDWR|os.O_NONBLOCK|os.O_NOCTTY)
        self.serial.update_lines(True,True,True,True)
        time.sleep(.05)
        bits=array.array('I',[0]);fcntl.ioctl(self.client,termios.TIOCMGET,bits,True)
        for mask in (termios.TIOCM_CAR,termios.TIOCM_DSR,termios.TIOCM_RNG,termios.TIOCM_CTS):self.assertTrue(bits[0]&mask)
        fcntl.ioctl(self.client,termios.TIOCMBIC,array.array('I',[termios.TIOCM_DTR]))
        time.sleep(.05);self.assertFalse(self.serial.signals()['TR'])
    def test_cts_stops_client_then_preserves_payload(self):
        attrs=termios.tcgetattr(self.client);attrs[2]|=termios.CRTSCTS
        termios.tcsetattr(self.client,termios.TCSANOW,attrs)
        self.serial.update_lines(False,False,True,True,backlog=50000);time.sleep(.05)
        payload=bytes(range(128))*8;self.assertEqual(os.write(self.client,payload),len(payload))
        self.assertEqual(self.read(self.serial.fd),b'')
        self.serial.update_lines(False,False,True,True,backlog=0)
        received=b'';deadline=time.monotonic()+3
        while len(received)<len(payload) and time.monotonic()<deadline:received+=self.read(self.serial.fd)
        self.assertEqual(received,payload)
    def test_rts_stops_modem_then_preserves_payload(self):
        self.serial.update_lines(False,False,True,True)
        fcntl.ioctl(self.client,termios.TIOCMBIC,array.array('I',[termios.TIOCM_RTS]));time.sleep(.05)
        payload=b'modem-output'*30;self.serial.write(payload)
        self.assertEqual(self.read(self.client),b'')
        fcntl.ioctl(self.client,termios.TIOCMBIS,array.array('I',[termios.TIOCM_RTS]));time.sleep(.05)
        self.serial._flush()
        received=b'';deadline=time.monotonic()+3
        while len(received)<len(payload) and time.monotonic()<deadline:received+=self.read(self.client)
        self.assertEqual(received,payload)
