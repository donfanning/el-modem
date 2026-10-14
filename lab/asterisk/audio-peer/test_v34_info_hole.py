"""Opt-in INFO1a recovery across a network media dropout: accepted only when the evidence says so."""
import unittest
import numpy as np
import v34_info as vi

FS = 8000
SPB = FS/600


def frame_signal(lead=0.25, tail=0.25, drop=None, corrupt_payload_bit=None, **fields):
    args = dict(hi_carrier=1, max_rate=12, sym_ans=5, sym_call=5, freq_offset=9)
    args.update(fields)
    bits = vi.build_info1a(**args)
    if corrupt_payload_bit is not None:
        bits = list(bits);bits[corrupt_payload_bit] ^= 1
    sig = vi.InfoModulator('answer').modulate(bits)
    carrier = lambda n: 2700*np.cos(2*np.pi*2400*np.arange(n)/FS)       # the answerer's tone before and after
    x = np.r_[carrier(int(lead*FS)), sig, carrier(int(tail*FS))]
    if drop:
        first_bit, ms = drop
        a = int(lead*FS+(first_bit+1)*SPB)
        x[a:a+int(ms*FS/1000)] = 0.
    return x


class HoleRecoveryTests(unittest.TestCase):
    def test_clean_frame_is_left_to_the_normal_path(self):
        x = frame_signal()
        self.assertTrue(vi.InfoDemodulator('answer', phases=8).decode(x, 'info1a'))
        self.assertIsNone(vi.recover_info1a(x))

    def test_dropout_over_the_crc_is_recovered_and_marked_unverified(self):
        x = frame_signal(drop=(58, 60))
        self.assertEqual(vi.InfoDemodulator('answer', phases=8).decode(x, 'info1a'), [])     # normal path fails
        r = vi.recover_info1a(x)
        self.assertIsNotNone(r)
        self.assertTrue(r['_crc_unverified'])
        self.assertEqual((r['hi_carrier'], r['max_rate'], r['sym_ans'], r['sym_call'], r['freq_offset']), (1, 12, 5, 5, 9))
        self.assertGreaterEqual(r['_dropout_ms'], 40)
        self.assertGreaterEqual(r['_support_phases'], 6)

    def test_refuses_without_a_dropout(self):
        x = frame_signal(corrupt_payload_bit=20)
        self.assertEqual(vi.InfoDemodulator('answer', phases=8).decode(x, 'info1a'), [])     # the CRC really fails
        self.assertIsNone(vi.recover_info1a(x))                                               # but nothing explains it

    def test_refuses_a_short_dropout_and_only_the_floor_does_so(self):
        x = frame_signal(drop=(58, 6))                       # 6 ms: breaks the CRC but is below the 8 ms floor
        self.assertEqual(vi.InfoDemodulator('answer', phases=8).decode(x, 'info1a'), [])
        self.assertIsNone(vi.recover_info1a(x))
        self.assertIsNotNone(vi.recover_info1a(x, min_dropout_s=0.003))                       # so the floor was the reason

    def test_refuses_when_the_payload_is_inside_the_dropout(self):
        self.assertIsNone(vi.recover_info1a(frame_signal(drop=(28, 60))))
        self.assertIsNone(vi.recover_info1a(frame_signal(drop=(12, 40))))

    def test_refuses_implausible_fields(self):
        self.assertIsNone(vi.recover_info1a(frame_signal(drop=(58, 60), sym_ans=6)))          # index 6 is V.90, not V.34
        self.assertIsNone(vi.recover_info1a(frame_signal(drop=(58, 60), max_rate=0)))

    def test_noise_and_silence_give_nothing(self):
        self.assertIsNone(vi.recover_info1a(np.random.default_rng(3).normal(0, 1500, 16000)))
        self.assertIsNone(vi.recover_info1a(np.zeros(16000)))

    def test_parse_info1a_is_unchanged(self):
        bits = vi.build_info1a(hi_carrier=1, max_rate=12, sym_ans=5, sym_call=5, freq_offset=-3)
        self.assertEqual(vi.parse_info1a(bits)['freq_offset'], -3)
        bad = list(bits);bad[55] ^= 1
        self.assertIsNone(vi.parse_info1a(bad))


def concealed_signal(first_bit=52, samples=160, **fields):
    """A frame whose tail was replaced by the DCE's concealment: the previous `samples` samples repeated."""
    x = frame_signal(**fields)
    a = int(0.25*FS+(first_bit+1)*SPB)
    x[a:a+samples] = x[a-samples:a]
    return x, (a, samples)


class ConcealmentRecoveryTests(unittest.TestCase):
    def test_tail_concealment_breaks_the_crc_but_leaves_a_frame_the_evidence_can_recover(self):
        x, span = concealed_signal()
        self.assertEqual(vi.InfoDemodulator('answer', phases=8).decode(x, 'info1a'), [])        # CRC really fails
        self.assertIsNone(vi.recover_info1a(x))                                                  # no silence to see
        r = vi.recover_info1a_concealed(x, [span])
        self.assertIsNotNone(r)
        self.assertTrue(r['_crc_unverified'])
        self.assertEqual((r['hi_carrier'], r['max_rate'], r['sym_ans'], r['sym_call'], r['freq_offset']), (1, 12, 5, 5, 9))
        self.assertEqual(r['_concealed_samples'], 160)

    def test_refuses_without_concealment_evidence(self):
        x, span = concealed_signal()
        self.assertIsNone(vi.recover_info1a_concealed(x, []))
        self.assertIsNone(vi.recover_info1a_concealed(x, [(span[0]+int(0.2*FS), 160)]))         # concealment after the frame

    def test_refuses_when_the_concealment_touches_the_payload(self):
        x, span = concealed_signal(first_bit=30, samples=320)
        self.assertIsNone(vi.recover_info1a_concealed(x, [span]))
        x2, span2 = concealed_signal()
        self.assertIsNone(vi.recover_info1a_concealed(x2, [span2, (span2[0]-int(20*SPB), 80)]))  # an extra span over the payload

    def test_refuses_implausible_fields_and_unsupported_frames(self):
        x, span = concealed_signal(sym_ans=6)
        self.assertIsNone(vi.recover_info1a_concealed(x, [span]))
        x2, span2 = concealed_signal(max_rate=0)
        self.assertIsNone(vi.recover_info1a_concealed(x2, [span2]))
        noise = np.random.default_rng(5).normal(0, 1500, 16000)
        self.assertIsNone(vi.recover_info1a_concealed(noise, [(8000, 160)]))

    def test_clean_frame_is_left_to_the_normal_path(self):
        x = frame_signal()
        self.assertIsNone(vi.recover_info1a_concealed(x, [(int(0.25*FS+60*SPB), 160)]))


if __name__ == '__main__':
    unittest.main()
