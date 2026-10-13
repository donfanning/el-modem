"""An E only counts after an acknowledged MP (lab/V34_LCP1_REPLAY.md: a steady tone gave a false E and a bogus B1 on a real call)."""
import unittest
from types import SimpleNamespace

import v34_call
import v34_mp
import v34_rx

MP = dict(max_c2a=14, max_a2c=14, aux=0, trellis=0, nonlinear=0, shaping=0, mask=16383, asym=1, precoder=[0] * 6)


def receiver(bits):
    """A Phase4Receiver positioned on a given descrambled bit stream (no signal processing)."""
    r = v34_rx.Phase4Receiver.__new__(v34_rx.Phase4Receiver)
    r.bits, r.pos, r.frames, r.mp_spans = list(bits), 0, [], []
    r.c16, r.k_trn, r.trn_end, r.b1_symbol, r.e_span = False, 100, 1500, None, None
    r.eq = SimpleNamespace(k=0)
    r._parse(None)
    return r


class ParserTests(unittest.TestCase):
    def test_twenty_ones_alone_are_not_an_e(self):
        r = receiver([0, 1, 0, 0, 1] * 5 + [1] * 20 + [0, 1, 1, 0] * 10)
        self.assertEqual(r.frames, [])
        self.assertIsNone(r.b1_symbol)
        self.assertIsNone(r.e_span)

    def test_twenty_ones_after_an_unacknowledged_mp_are_not_an_e(self):
        r = receiver(v34_mp.build_mp(1, ack=0, **MP) * 3 + [1] * 20 + [0] * 40)
        self.assertEqual(len(r.frames), 3)
        self.assertNotIn("E", r.frames)
        self.assertIsNone(r.b1_symbol)

    def test_e_after_an_acknowledged_mp_is_found_and_places_b1(self):
        r = receiver(v34_mp.build_mp(1, ack=0, **MP) + v34_mp.build_mp(1, ack=1, **MP) + [1] * 20 + [0] * 40)
        self.assertEqual(r.frames[-1], "E")
        self.assertIsNotNone(r.b1_symbol)
        self.assertIsNotNone(r.e_span)

    def test_a_late_ack_makes_a_later_e_valid(self):
        bits = v34_mp.build_mp(1, ack=0, **MP) * 2 + [1] * 20 + [0] * 8 + v34_mp.build_mp(1, ack=1, **MP) + [1] * 20 + [0] * 8
        r = receiver(bits)
        self.assertEqual([f if f == "E" else f["ack"] for f in r.frames], [0, 0, 1, "E"])


class CallerTests(unittest.TestCase):
    def mp(self, ack):
        return dict(type=1, ack=ack)

    def test_acknowledgement_rules(self):
        self.assertFalse(v34_call.mp_acknowledged([]))
        self.assertFalse(v34_call.mp_acknowledged([self.mp(0)] * 154))
        self.assertFalse(v34_call.mp_acknowledged([self.mp(0)] * 154 + ["E"]))      # the lcp-1 case
        self.assertFalse(v34_call.mp_acknowledged(["E"]))
        self.assertTrue(v34_call.mp_acknowledged([self.mp(0), self.mp(1)]))
        self.assertTrue(v34_call.mp_acknowledged([self.mp(0), self.mp(1), "E"]))


if __name__ == "__main__":
    unittest.main()
