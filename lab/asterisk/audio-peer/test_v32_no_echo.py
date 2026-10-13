"""Disabling the unused echo reference preserves TX waveform and protocol clock."""
import unittest
from v32_stream import V32Modem


class NoEchoTransmitTests(unittest.TestCase):
    def test_training_e_data_waveform_and_transmit_clock_parity(self):
        for rate in (9600,14400):
            with self.subTest(rate=rate):
                reference=V32Modem(my_rates=(rate,),echo=True,use_v8=False)
                disabled=V32Modem(my_rates=(rate,),echo=False,use_v8=False)
                payload=bytes(range(256))*32
                for modem in (reference,disabled):
                    modem.feed_bytes(payload)
                    modem._tx_started=True;modem.state='tx'
                    modem.tx.start_training_now();modem.tx.request_e(rate)
                for index in range(220):
                    count=(73,160,257)[index%3]
                    self.assertEqual(reference.tx_samples(count),disabled.tx_samples(count))
                    self.assertEqual(reference._txn,disabled._txn)
                    self.assertEqual(reference.tx._n,disabled.tx._n)
                self.assertLess(len(disabled.tx._tx_bytes),len(payload))
                self.assertEqual(disabled.tx.data_rate,rate)
                self.assertGreater(reference._tx_out,0)
                self.assertEqual(disabled._tx_out,0)
                self.assertFalse(disabled.echo_stats['requested'])
