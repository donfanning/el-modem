"""Lost poll-final response must not leave an acknowledged sender blocked."""
import unittest

from v42_lapm import RR, REJ, V42Link


class PollProgressTests(unittest.TestCase):
    def waiting_sender(self):
        link = V42Link(True, detect=False, bit_rate=9600)
        link._connected()
        link._queue_i(b'first')
        link._queue_i(b'second')
        link._start_t401()
        link._t401_expired()
        self.assertTrue(link._poll_out)
        return link

    def test_partial_piggyback_ack_keeps_poll_retry_and_recovers(self):
        link = self.waiting_sender()
        deadline = link._t401_at
        # An in-sequence I frame acknowledges the first outbound frame, but
        # is not a supervisory final response to our outstanding poll.
        link._on_frame(bytes((link.rsp_addr, 0, 2)) + b'peer data')
        self.assertEqual(link.read(), b'peer data')
        self.assertEqual(link.stats['wire_acked'], 5)
        self.assertTrue(link._poll_out)
        self.assertEqual(link._t401_at, deadline)
        # Timer checks occur at frame/flag boundaries, not midway through
        # an already queued HDLC frame. Allow one second for that boundary.
        while link.bits_sent <= deadline + link.rate:
            link.tx_bit()
        self.assertGreaterEqual(link.stats['t401_expiries'], 2)
        link._on_frame(bytes((link.cmd_addr, RR, 5)))  # F=1, N(R)=2
        self.assertFalse(link._poll_out)
        self.assertTrue(link._window_open())

    def test_full_nonfinal_ack_still_retries_the_unanswered_poll(self):
        link = self.waiting_sender()
        deadline, tries = link._t401_at, link._tries
        link._on_frame(bytes((link.cmd_addr, RR, 4)))  # F=0, N(R)=2
        self.assertEqual(link._unacked, [])
        self.assertTrue(link._poll_out)
        self.assertEqual(link._t401_at, deadline)
        self.assertEqual(link._tries, tries)
        while link.bits_sent <= deadline + link.rate:
            link.tx_bit()
        self.assertGreaterEqual(link.stats['t401_expiries'], 2)

    def test_nonfinal_rej_waits_for_final_before_retransmission(self):
        link = self.waiting_sender()
        deadline, tries = link._t401_at, link._tries
        link._peer_busy = True
        # REJ/F=0 acknowledges one frame but cannot complete timer recovery.
        link._on_frame(bytes((link.cmd_addr, REJ, 2)))
        self.assertEqual(link.stats['wire_acked'], 5)
        self.assertFalse(link._peer_busy)
        self.assertTrue(link._poll_out)
        self.assertEqual(link._t401_at, deadline)
        self.assertEqual(link._tries, tries)
        self.assertFalse(link._retx)
        self.assertFalse(link._window_open())
        # A subsequent valid final REJ permits retransmission of the remainder.
        link._on_frame(bytes((link.cmd_addr, REJ, 3)))
        self.assertFalse(link._poll_out)
        self.assertEqual(list(link._retx), [(1, b'second')])
        self.assertIsNotNone(link._t401_at)

    def test_polled_rej_command_answers_without_ending_our_poll(self):
        link = self.waiting_sender()
        link._tx.clear()
        deadline = link._t401_at
        # The peer's own P=1 command requires a final response, but is not
        # a response to the poll we sent. The two directions are independent.
        link._on_frame(bytes((link.rsp_addr, REJ, 3)))
        self.assertTrue(link._tx)
        self.assertTrue(link._poll_out)
        self.assertEqual(link._t401_at, deadline)
        self.assertFalse(link._retx)


if __name__ == '__main__':
    unittest.main()
