"""Established HDLC quality includes control frames and real corrupt FCS."""
import unittest
import v42_lapm as lapm


class FrameQualityTests(unittest.TestCase):
    def feed(self, link, frame, corrupt=False):
        bits = lapm._stuffed(frame)
        if corrupt:
            # Flip an address bit away from stuffing/flags: HDLC remains framed.
            bits[3] ^= 1
        for bit in lapm.FLAG_BITS + bits + lapm.FLAG_BITS:
            link.rx_bit(bit)

    def test_control_and_data_have_same_fcs_opportunities(self):
        link = lapm.V42Link(True, detect=False)
        link.state = 'lapm'
        self.feed(link, bytes((link.rsp_addr, lapm.RR, 0)))
        self.feed(link, bytes((link.rsp_addr, 0, 0)) + b'hello')
        self.feed(link, bytes((link.rsp_addr, lapm.RR, 0)), corrupt=True)
        self.assertEqual(link.stats['hdlc_good'], 2)
        self.assertEqual(link.stats['hdlc_bad'], 1)
        self.assertEqual(link.stats['fcs_errors'], 1)
        self.assertEqual(link.stats['i_rx'], 1)

    def test_protocol_negotiation_is_excluded(self):
        link = lapm.V42Link(True, detect=False)
        self.feed(link, bytes((link.rsp_addr, lapm.RR, 0)))
        self.feed(link, bytes((link.rsp_addr, lapm.RR, 0)), corrupt=True)
        self.assertEqual(link.stats['hdlc_good'], 0)
        self.assertEqual(link.stats['hdlc_bad'], 0)


if __name__ == '__main__':
    unittest.main()
