import random
import unittest

import numpy as np

import v34_g711
import v34_tx
import v92_pcm_upstream as pu
from v92_pcm_upstream_rx import B1_FRAMES, PcmUpstreamStreamReceiver


def profile(law):
    codes = np.arange(0, 120, 3, dtype=np.uint8)
    levels = tuple(sorted({abs(float(v)) for v in v34_g711.decode(codes ^ 0x00, law)} - {0.0}))[:48]
    return pu.PcmUpstreamProfile(constellations=(levels,), interval_set=(0,) * 12,
                                 moduli=(30, 30, 30, 15) * 3, trellis=1)


def wire(prof, law, payload, *, b1_ones=B1_FRAMES, lead=0):
    tx = pu.PcmUpstreamTransmitter(prof)
    k = prof.bits_per_frame
    scrambler = v34_tx.Scrambler("answer")
    data = [(byte >> i) & 1 for byte in payload for i in range(8)]
    data += [0] * (-len(data) % k)
    stream = [1] * (b1_ones * k) + data
    scrambled = [scrambler.step(b) for b in stream]
    samples = []
    for i in range(0, len(scrambled), k):
        samples.extend(tx.encode_frame(scrambled[i:i + k]))
    codes = v34_g711.encode(np.rint(samples), law).astype(np.uint8) & 0xFF
    return bytes([0xFF if law == "pcmu" else 0xD5] * lead) + bytes(codes)


class StreamReceiverTests(unittest.TestCase):
    def test_lost_frame_limit_applies_to_consecutive_bad_frames(self):
        rx = PcmUpstreamStreamReceiver(
            profile("pcmu"), "pcmu", max_lost_frames=1, max_data_lost_frames=1)
        rx.state = rx.DATA
        calls = iter((False, True, False, False))

        def decode_intervals(_samples):
            if next(calls):
                return [0] * pu.INTERVALS
            raise ValueError("outside constellation")

        rx._rx.decode_intervals = decode_intervals
        rx._frame([0] * pu.INTERVALS)
        self.assertEqual((rx.state, rx.lost_frames, rx.consecutive_lost_frames),
                         (rx.DATA, 1, 1))
        self.assertEqual(rx.max_consecutive_lost_frames, 1)
        rx._frame([0] * pu.INTERVALS)  # valid frame resets the burst counter
        self.assertEqual((rx.state, rx.lost_frames, rx.consecutive_lost_frames),
                         (rx.DATA, 1, 0))
        rx._frame([0] * pu.INTERVALS)
        self.assertEqual((rx.state, rx.lost_frames, rx.consecutive_lost_frames),
                         (rx.DATA, 2, 1))
        rx._frame([0] * pu.INTERVALS)  # two consecutive losses exceed max=1
        self.assertEqual(rx.state, rx.FAILED)
        self.assertEqual(rx.lost_frames, 3)
        self.assertEqual(rx.max_consecutive_lost_frames, 2)

    def test_data_loss_emits_exact_k_bits_and_best_effort_advances_sign(self):
        levels = (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236)
        prof = pu.PcmUpstreamProfile(
            constellations=(levels,), interval_set=(0,) * 12,
            moduli=(10, 10, 10, 5) * 3, trellis=1, bits=36)
        rx = PcmUpstreamStreamReceiver(prof, "pcmu")
        rx.state = rx.DATA
        invalid_ks = [9, 9, 9, 4] * 3
        rx._rx.decode_intervals = lambda _samples: invalid_ks
        expected_rx = PcmUpstreamStreamReceiver(prof, "pcmu")
        expected_bits = expected_rx._descramble([1] * prof.bits_per_frame)
        rx._frame([0] * pu.INTERVALS)
        self.assertEqual(rx.state, rx.DATA)
        self.assertEqual(rx.lost_frames, 1)
        self.assertEqual(len(rx._data_bits), 36)
        self.assertEqual(rx._data_bits, expected_bits)
        self.assertEqual(rx._rx._sign, 1)

    def test_data_loss_limit_is_relaxed_but_b1_limit_is_unchanged(self):
        prof = profile("pcmu")
        rx = PcmUpstreamStreamReceiver(prof, "pcmu")
        rx.state = rx.DATA
        for _ in range(4):
            rx._lose_frame()
        self.assertEqual(rx.state, rx.DATA)
        self.assertEqual(rx.lost_frames, 4)
        self.assertEqual(len(rx._data_bits), 4 * prof.bits_per_frame)

        b1 = PcmUpstreamStreamReceiver(prof, "pcmu", max_lost_frames=3)
        b1.state = b1.B1
        for _ in range(4):
            b1._lose_frame()
        self.assertEqual(b1.state, b1.FAILED)
        self.assertEqual(b1.frames, 3)

    def test_payload_comes_back_exact_in_any_chunking_under_both_laws(self):
        payload = random.Random(2).randbytes(300)
        for law in ("pcmu", "pcma"):
            prof = profile(law)
            octets = wire(prof, law, payload)
            for chunk in (1, 5, 12, 160, 10_000):
                with self.subTest(law=law, chunk=chunk):
                    rx = PcmUpstreamStreamReceiver(prof, law)
                    for i in range(0, len(octets), chunk):
                        rx.feed(octets[i:i + chunk])
                    self.assertEqual(rx.state, rx.DATA, rx.failure)
                    self.assertEqual(rx.b1_errors, 0)
                    self.assertEqual(rx.take_bytes()[:len(payload)], payload)

    def test_leading_units_are_skipped(self):
        prof = profile("pcmu")
        payload = bytes(range(100))
        octets = wire(prof, "pcmu", payload, lead=777)
        rx = PcmUpstreamStreamReceiver(prof, "pcmu", start_skip=777)
        rx.feed(octets[:500])
        self.assertEqual(rx.state, rx.IDLE)
        rx.feed(octets[500:])
        self.assertEqual(rx.state, rx.DATA)
        self.assertEqual(rx.take_bytes()[:100], payload)

    def test_a_b1u_that_is_not_ones_fails_the_connection(self):
        prof = profile("pcmu")
        octets = bytearray(wire(prof, "pcmu", bytes(40)))
        for i in range(12 * 10, 12 * 12):                      # wreck two B1u frames
            octets[i] = octets[i] ^ 0x05
        rx = PcmUpstreamStreamReceiver(prof, "pcmu")
        rx.feed(bytes(octets))
        self.assertEqual(rx.state, rx.FAILED)
        self.assertTrue(rx.failure)

    def test_expanded_constellation_decodes_both_equivalence_representatives(self):
        levels = (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236)
        prof = pu.PcmUpstreamProfile(
            constellations=(levels,), interval_set=(0,) * 12,
            moduli=(10, 10, 10, 5) * 3, trellis=1, bits=36)

        # For every interval and every modulus class, send each signed point in
        # that equivalence class in turn. The fourth point is selected only after
        # the trellis parity from its preceding three points is known.
        for target in range(pu.INTERVALS):
            modulus = prof.moduli[target]
            for target_k in range(modulus):
                # Generate candidate points in the target class, with the slot-3
                # parity supplied by the trellis state when that slot is reached.
                conv = pu.convolutional_encoder(prof.trellis)
                prefix_etas = []
                candidates = None
                for i in range(pu.INTERVALS):
                    if i % 4 < 3:
                        k = target_k if i == target else 0
                        m = prof.moduli[i]
                        valid = [eta for eta in range(-len(levels), len(levels))
                                 if eta % m == k]
                        eta = (valid[0] if i == target else 0)
                    else:
                        parity = (sum(prefix_etas) + conv.y0) & 1
                        k = target_k if i == target else 0
                        m = prof.moduli[i]
                        valid = [eta for eta in range(-len(levels), len(levels))
                                 if (eta & 1) == parity and ((eta - parity) // 2) % m == k]
                        eta = valid[0] if i != target else valid[0]
                    if i == target:
                        candidates = valid
                        break
                    prefix_etas.append(eta)
                    if i % 4 == 3:
                        pu._trellis_step(conv, prefix_etas)
                        prefix_etas = []

                for candidate in candidates:
                    with self.subTest(interval=target, k=target_k, eta=candidate):
                        conv = pu.convolutional_encoder(prof.trellis)
                        etas = []
                        samples = []
                        for i in range(pu.INTERVALS):
                            if i % 4 < 3:
                                k = target_k if i == target else 0
                                m = prof.moduli[i]
                                valid = [eta for eta in range(-len(levels), len(levels))
                                         if eta % m == k]
                                eta = candidate if i == target else 0
                            else:
                                parity = (sum(etas) + conv.y0) & 1
                                k = target_k if i == target else 0
                                m = prof.moduli[i]
                                valid = [point for point in range(-len(levels), len(levels))
                                         if (point & 1) == parity
                                         and ((point - parity) // 2) % m == k]
                                eta = candidate if i == target else valid[0]
                            samples.append(prof.level(i, eta))
                            etas.append(eta)
                            if i % 4 == 3:
                                pu._trellis_step(conv, etas)
                                etas = []
                        rx = pu.PcmUpstreamReceiver(prof)
                        decoded = rx.decode_intervals(samples)
                        expected = [0] * pu.INTERVALS
                        expected[target] = target_k
                        self.assertEqual(decoded, expected)
                        self.assertEqual(rx.trellis_violations, 0)

    def test_viterbi32_corrects_a_marginal_b1_representative_and_decodes_data(self):
        levels = (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236)
        prof = pu.PcmUpstreamProfile(
            constellations=(levels,), interval_set=(0,) * 12,
            moduli=(10, 10, 10, 5) * 3, trellis=1, bits=36)
        payload = random.Random(92).randbytes(64)
        payload_bits = [(byte >> bit) & 1 for byte in payload for bit in range(8)]
        data_bits = payload_bits + [0] * (24 * prof.bits_per_frame - len(payload_bits))
        scrambler = v34_tx.Scrambler("answer")
        source = [1] * (B1_FRAMES * prof.bits_per_frame) + data_bits
        scrambled = [scrambler.step(bit) for bit in source]
        tx = pu.PcmUpstreamTransmitter(prof, record=True)
        samples = []
        for start in range(0, len(scrambled), prof.bits_per_frame):
            samples.extend(tx.encode_frame(scrambled[start:start + prof.bits_per_frame]))

        # In this sequence B1 frame 7, slot 3 sends eta=1 (U=292); place it
        # just below the eta=0/1 midpoint so a nearest-point slicer picks eta=0.
        at = 7 * pu.INTERVALS + 3
        self.assertEqual(tx.levels[at], 292)
        samples[at] = 190.0

        hard = PcmUpstreamStreamReceiver(prof, "pcmu")
        hard.feed_samples(samples[:B1_FRAMES * pu.INTERVALS])
        self.assertNotEqual(hard.state, hard.DATA)
        self.assertGreater(hard.trellis_violations, 0)

        soft = PcmUpstreamStreamReceiver(prof, "pcmu", decoder_mode="viterbi32",
                                         viterbi_candidates=2, viterbi_traceback_groups=8)
        soft.feed_samples(samples)
        self.assertEqual(soft.state, soft.DATA)
        self.assertEqual(soft.b1_errors, 0)
        self.assertEqual(soft.trellis_violations, 0)
        self.assertGreater(soft.soft_representative_corrections, 0)
        received = soft.take_bits()
        self.assertEqual(received[:len(payload_bits)], payload_bits)

    def test_viterbi32_decodes_a_constellation_of_more_than_ten_levels(self):
        levels = tuple(range(150, 150 + 12 * 180, 180))
        prof = pu.PcmUpstreamProfile(
            constellations=(levels,), interval_set=(0,) * 12,
            moduli=(12, 12, 12, 6) * 3, trellis=1, bits=40)
        payload = random.Random(12).randbytes(64)
        payload_bits = [(byte >> bit) & 1 for byte in payload for bit in range(8)]
        data_bits = payload_bits + [0] * (24 * prof.bits_per_frame - len(payload_bits))
        scrambler = v34_tx.Scrambler("answer")
        scrambled = [scrambler.step(bit) for bit in [1] * (B1_FRAMES * prof.bits_per_frame) + data_bits]
        tx = pu.PcmUpstreamTransmitter(prof)
        samples = []
        for start in range(0, len(scrambled), prof.bits_per_frame):
            samples.extend(tx.encode_frame(scrambled[start:start + prof.bits_per_frame]))
        soft = PcmUpstreamStreamReceiver(prof, "pcmu", decoder_mode="viterbi32",
                                         viterbi_traceback_groups=8)
        soft.feed_samples(samples)
        self.assertEqual(soft.state, soft.DATA)
        self.assertEqual(soft.trellis_violations, 0)
        self.assertEqual(soft.take_bits()[:len(payload_bits)], payload_bits)

    def test_inputs_are_validated(self):
        prof = profile("pcmu")
        with self.assertRaises(ValueError):
            PcmUpstreamStreamReceiver(prof, "alaw")
        with self.assertRaises(ValueError):
            PcmUpstreamStreamReceiver(prof, "pcmu", gain=0)
        with self.assertRaises(ValueError):
            PcmUpstreamStreamReceiver(prof, "pcmu", start_skip=-1)
        for bad in (True, -1, 1.5):
            with self.subTest(max_lost_frames=bad), self.assertRaisesRegex(
                    ValueError, "max_lost_frames"):
                PcmUpstreamStreamReceiver(prof, "pcmu", max_lost_frames=bad)
        for bad in (True, -1, 1.5):
            with self.subTest(max_data_lost_frames=bad), self.assertRaisesRegex(
                    ValueError, "max_data_lost_frames"):
                PcmUpstreamStreamReceiver(prof, "pcmu", max_data_lost_frames=bad)
        unlimited = PcmUpstreamStreamReceiver(prof, "pcmu", max_lost_frames=10 ** 9)
        self.assertEqual(unlimited._max_lost, 10 ** 9)


if __name__ == "__main__":
    unittest.main()
