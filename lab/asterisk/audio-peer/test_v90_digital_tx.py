"""Tests for the digital modem's exact V.90 Phase 3 codeword signals."""

import unittest

import v34_info
import v90_digital_tx as v90tx
from v90_dil import g711_ucode_codewords


def independent_crc(bits):
    reg = 0xffff
    for bit in bits:
        feedback = ((reg >> 15) & 1) ^ int(bit)
        reg = (reg << 1) & 0xffff
        if feedback:
            reg ^= 0x1021
    return sum(((reg >> bit) & 1) << (15 - bit) for bit in range(16))


def independent_gpc(input_bits, state=None):
    history = [0] * 23 if state is None else list(state)
    output = []
    for input_bit in input_bits:
        bit = int(input_bit) ^ history[17] ^ history[22]
        history = [bit, *history[:22]]
        output.append(bit)
    return output, history


class V90DigitalPhase3Tests(unittest.TestCase):
    def test_jd_rejects_reserved_rates_and_encodes_crc_independently(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            v90tx.build_jd_bits(1 << 22)
        mask = (1 << 0) | (1 << 16) | (1 << 21)
        bits = v90tx.build_jd_bits(mask, phase3_16point=True,
                                   renegotiation_16point=True, max_lookahead=3)
        self.assertEqual(bits[18:34], [(mask >> n) & 1 for n in range(16)])
        self.assertEqual(bits[35:47], [(mask >> (16 + n)) & 1 for n in range(12)])
        self.assertEqual(bits[47:51], [1, 1, 1, 1])
        expected = independent_crc(bits[18:34] + bits[35:51])
        actual = sum(bits[52 + n] << n for n in range(16))
        self.assertEqual(actual, expected)
        self.assertEqual(actual, v34_info.crc16(bits[18:34] + bits[35:51]))
        self.assertEqual(bits[68:], [0] * 4)

    def test_sd_and_sd_prime_are_exact_g711_patterns(self):
        uinfo, law = 90, "pcmu"
        table = g711_ucode_codewords(law)
        w = uinfo + 16
        expected_sd = bytes((table[w], table[0], table[w], table[w] ^ 0x80,
                             table[0] ^ 0x80, table[w] ^ 0x80)) * 64
        expected_prime = bytes((table[w] ^ 0x80, table[0] ^ 0x80,
                                table[w] ^ 0x80, table[w], table[0], table[w])) * 8
        self.assertEqual(v90tx.sd_codewords(uinfo, law), expected_sd)
        self.assertEqual(v90tx.sd_prime_codewords(uinfo, law), expected_prime)

    def test_trn1d_jd_and_jprime_preserve_scrambler_and_differential_state(self):
        tx = v90tx.V90DigitalPhase3Transmitter(90, "pcma")
        table = g711_ucode_codewords("pcma")
        trn_bits, history = independent_gpc([1] * 2040)
        expected_trn = bytes(int(table[90]) if bit else int(table[90]) ^ 0x80
                             for bit in trn_bits)
        trn = tx.trn1d()
        self.assertEqual(trn, expected_trn)
        self.assertEqual(len(trn), 2040)

        frame = v90tx.build_jd_bits(1 << 21, max_lookahead=2)
        scrambled, history = independent_gpc(frame, history)
        # Jd starts differential state at the final TRN1d sign.
        differential = trn_bits[-1]
        signs = []
        for bit in scrambled:
            differential ^= bit
            signs.append(differential)
        expected_jd = bytes(int(table[90]) if bit else int(table[90]) ^ 0x80
                            for bit in signs)
        jd = tx.jd_frame(1 << 21, max_lookahead=2)
        self.assertEqual(jd, expected_jd)
        self.assertEqual(len(jd), 72)

        zero_bits, _history = independent_gpc([0] * 12, history)
        signs = []
        for bit in zero_bits:
            differential ^= bit
            signs.append(differential)
        expected_prime = bytes(int(table[90]) if bit else int(table[90]) ^ 0x80
                               for bit in signs)
        prime = tx.jprime_d()
        self.assertEqual(prime, expected_prime)
        self.assertEqual(len(prime), 12)
        self.assertTrue(tx.jprime_sent)

    def test_phase_order_and_trn_length_are_guarded(self):
        tx = v90tx.V90DigitalPhase3Transmitter(90, "pcmu")
        with self.assertRaisesRegex(ValueError, "2040"):
            tx.trn1d(2039)
        with self.assertRaisesRegex(RuntimeError, "TRN1d"):
            tx.jd_frame(1)
        tx.trn1d()
        with self.assertRaisesRegex(RuntimeError, "Jd"):
            tx.jprime_d()


if __name__ == "__main__":
    unittest.main()
