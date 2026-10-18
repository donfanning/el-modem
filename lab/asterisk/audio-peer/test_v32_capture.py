#!/usr/bin/env python3
"""Replay real PSTN V.32 captures through the receive state machine.

The fixtures are the receive side of calls to NIST ACTS, recorded with
``MODEM_CAPTURE``. Replaying them pins the handshake against a real peer without dialling, which
matters because that peer's training sequence is shorter than our own answerer sends and stops
matching our expected TRN partway through.

Three recordings, because the peer is not deterministic: ``acts_v32`` resumed with a repeated
rate word instead of a second TRN, ``acts_v32_no_second_trn`` is another such call, and
``acts_v32_second_trn`` is a call where it did train again and the call connected.
"""
import unittest
from pathlib import Path

from replay_v32_capture import replay

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = _FIXTURES / "acts_v32.rx.raw"
NO_TRN = _FIXTURES / "acts_v32_no_second_trn.rx.raw"
SECOND_TRN = _FIXTURES / "acts_v32_second_trn.rx.raw"


class ActsCaptureTests(unittest.TestCase):
    def test_the_short_training_sequence_is_detected_and_r1_is_read(self):
        m, _ = replay(FIXTURE, secs=12)
        events = " | ".join(e for _, e in m.events)
        # we locate the answerer's TRN, train on it, and must notice it ends at ~2420 symbols
        # rather than grinding to 7000 against a sequence that is no longer TRN
        self.assertIn("answerer TRN located", events)
        ends = [int(e.split("after ")[1].split(" ")[0])
                for _, e in m.events if "TRN ended after" in e]
        self.assertTrue(ends, events)
        self.assertLess(ends[0], 3000, events)
        # ... and still read its rate word, which is what tells us when to train in turn
        self.assertIn("R1 = ", events)

    def test_the_far_end_never_reaches_a_second_training_sequence(self):
        # recorded run: ACTS resumes with a repeated rate word, not an S/S-bar/TRN we can locate,
        # so this leg is expected to end in NO CARRIER. Guards against silently "passing" by
        # changing the timeline expectations above.
        m, _ = replay(FIXTURE, secs=25)
        self.assertFalse(m.connected)


class SecondTrainingTests(unittest.TestCase):
    def test_a_call_where_the_peer_trains_again_connects_and_carries_data(self):
        m, got = replay(SECOND_TRN, secs=20, caller=True)
        events = " | ".join(e for _, e in m.events)
        self.assertIn("second TRN located", events)
        self.assertIn("R3 = ", events)
        self.assertIn("E received", events)
        self.assertTrue(m.connected, events)
        self.assertEqual(m.bit_rate, 9600)
        self.assertGreater(len(got), 0, "expected payload after CONNECT")

    def test_without_a_second_trn_the_rate_word_is_read_from_where_the_line_came_back_up(self):
        # ACTS sometimes resumes with a repeated rate word and no second TRN. Reading R3 then
        # means decoding with an equaliser trained on the *first* leg, five seconds earlier, so
        # where the reader starts matters enormously: it has to re-anchor on the resumed
        # transmission. Before this was fixed the symbol index was still pointing at the first
        # leg's TRN, and the slicer was handed the silent gap that follows it -- a flat zero
        # every symbol, which no amount of adaptation recovers from.
        #
        # Asserted on what the slicer is actually fed, not on the log line: ``radial`` is the
        # equaliser output's magnitude over the nearest constellation point, so a reader
        # pointed at the silent gap reports ~0.00 and one pointed at the resumed transmission
        # reports ~0.9.
        m, _ = replay(NO_TRN, secs=20, caller=True)
        events = [e for _, e in m.events]
        joined = " | ".join(events)
        self.assertIn("no second TRN", joined)

        start = next(i for i, e in enumerate(events) if "no second TRN" in e)
        radial = [float(e.rsplit("radial ", 1)[1])
                  for e in events[start:] if "radial " in e]
        self.assertTrue(radial, joined)
        best = max(radial)
        self.assertGreater(best, 0.5,
                           f"slicer never reached the constellation (radial {best:.3f}): "
                           f"the reader is not pointed at the resumed transmission")

        # the answerer resumed at 15.66s in this recording, so the reader must start just after
        anchor = [e for e in events if "no second TRN" in e][0]
        at = float(anchor.rsplit("from ", 1)[1].rstrip("s"))
        self.assertGreater(at, 15.5, anchor)
        self.assertLess(at, 16.5, anchor)

    def test_that_leg_gives_up_quickly_instead_of_hanging(self):
        # there is no second TRN to retrain on, so the stale equaliser cannot resolve a 16-bit
        # word (checked against this capture's successful sibling too: forcing that one down the
        # same path also fails to recover its R3). The leg must therefore end promptly and say
        # why, rather than sitting in decision-directed mode until the DCE's carrier timer.
        m, _ = replay(NO_TRN, secs=20, caller=True)
        events = " | ".join(e for _, e in m.events)
        self.assertIn("no second training sequence", events)
        failed = [t for t, e in m.events if e.startswith("FAILED")]
        self.assertTrue(failed, events)
        # the fallback engages at ~16.3s and gets 2.5s of rate word before giving up
        self.assertLess(failed[0], 20.0, events)


if __name__ == "__main__":
    unittest.main()
