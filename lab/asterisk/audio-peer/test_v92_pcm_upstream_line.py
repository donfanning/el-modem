import unittest

import numpy as np

import v34_g711
import v92_pair_session as session
from v92_pcm_upstream_rx import PcmUpstreamStreamReceiver


def octets(line, law):
    return (v34_g711.encode(np.rint(line).astype(np.int32), law).astype(np.uint8) & 0xFF).tobytes()


class StreamOnADispersiveLineTests(unittest.TestCase):
    def check(self, channel, law, noise, payload):
        # A transmit budget is needed: uncapped, the designed precoder asks for peaks above 50000.
        r = session.run(channel, payload, law=law, noise_rms=noise, max_tx_rms=6000.0)
        self.assertTrue(r["connected"], r.get("reason"))
        line = r["line_samples"][r["line_delay"]:]
        rx = PcmUpstreamStreamReceiver(r["profile"], law)
        data = octets(line, law)
        for i in range(0, len(data), 160):                # RTP-sized chunks
            rx.feed(data[i:i + 160])
        self.assertEqual(rx.state, rx.DATA, rx.failure)
        self.assertEqual(rx.b1_errors, 0)
        self.assertEqual(rx.trellis_violations, 0)
        self.assertEqual(rx.take_bytes()[:len(payload)], payload)
        return r

    def test_designed_precoder_on_a_dispersive_line_under_both_laws(self):
        payload = bytes((i * 37 + 11) & 0xFF for i in range(400))
        for law in ("pcmu", "pcma"):
            for channel in ([1.0, 0.45, -0.2, 0.1], [1.0, -0.3, 0.15], [1.0, 0.4, -0.15], [0.8, 0.3, 0.2, -0.1, 0.05]):
                with self.subTest(law=law, channel=channel):
                    r = self.check(channel, law, 0.0, payload)
                    self.assertGreater(r["rate"], 28000)
                    self.assertLess(r["peak_transmit"], 32768)

    def test_with_line_noise(self):
        payload = bytes(range(200))
        self.check([1.0, 0.4, -0.15], "pcmu", 20.0, payload)


if __name__ == "__main__":
    unittest.main()
