import math
import unittest

import numpy as np

import v8
import v92_anspcm
import v92_quick_connect as qc
import v92_short_phase1 as short_phase1
from autochain import V8Chain
from g711 import linear_to_ulaw, ulaw_to_linear
from v92_qc_answerer import (V92QuickConnectAnswerer, QCA_SAMPLES,
                             SILENCE_SAMPLES, TIMEOUT_AFTER_QCA)
from v8 import V8_MOD_V34, V8_PCM_DIGITAL, V8_PROTOCOL_LAPM_V42


def _qc1a_pcm(*, lapm=True, uqts=0b0100):
    src = v8._Fsk(v8.V21_L, 2500.0)
    src.bits = qc.build("QC1a", lapm=lapm, code=uqts)
    silence = np.zeros(400, dtype=float)
    # Give the receiver enough trailing samples to make a centre-of-bit
    # decision for the final stop bit at every one of its eight phases.
    return np.concatenate((silence, src.read(1600), silence))


def _pcm_bytes(samples):
    values = np.asarray(samples, dtype=np.int16)
    return values.astype("<i2", copy=False).tobytes()


class _Engine:
    def __init__(self, *, role, negotiation, media_at, media_origin):
        self.role = role
        self.negotiation = negotiation
        self.media_at = media_at
        self.media_origin = media_origin
        self.events = []

    def tx_samples(self, count): return [0] * count
    def feed_pcm(self, pcm): return b""
    def feed_bytes(self, data): pass
    def tx_g711(self, count, codec): return bytes([0xFF if codec == "pcmu" else 0xD5]) * count
    def feed_g711(self, payload, codec, context): return b""
    def feed_gap(self, samples, codec, context): return b""
    def close(self): pass


class V92QuickConnectAnswererTests(unittest.TestCase):
    def test_v8_recognizes_qc1a_only_when_the_profile_enables_it(self):
        for enabled, expected in ((False, None), (True, "QC1a")):
            with self.subTest(enabled=enabled):
                answer = v8.V8Negotiator(
                    False, modulations=V8_MOD_V34,
                    pcm_availability=V8_PCM_DIGITAL,
                    v92_quick_connect=enabled)
                for _ in range(11):
                    answer.tx(160)                 # start ANSam
                signal = _qc1a_pcm()
                for start in range(0, len(signal), 160):
                    answer.rx(signal[start:start + 160])
                got = answer.quick_connect_request
                self.assertEqual(None if got is None else got["kind"], expected)
                self.assertEqual(answer.quick_connect_attempted, enabled)

    def test_after_qc_timeout_a_second_qc1a_is_ignored(self):
        answer = v8.V8Negotiator(
            False, modulations=V8_MOD_V34,
            pcm_availability=V8_PCM_DIGITAL,
            v92_quick_connect=True)
        for _ in range(11):
            answer.tx(160)
        signal = _qc1a_pcm()
        for start in range(0, len(signal), 160):
            answer.rx(signal[start:start + 160])
        self.assertIsNotNone(answer.quick_connect_request)
        answer.resume_after_quick_connect_timeout()
        for start in range(0, len(signal), 160):
            answer.rx(signal[start:start + 160])
        self.assertTrue(answer.quick_connect_attempted)
        self.assertIsNone(answer.quick_connect_request)
        self.assertEqual(answer._stage, "ansam")

    def test_qca_qts_and_anspcm_are_sent_as_the_selected_g711_codewords(self):
        peer = qc.parse(qc.build("QC1a", lapm=True, code=0b0100))
        responder = V92QuickConnectAnswerer(
            peer=peer, law="pcmu", lapm_offer=True, level_code=0b01)
        # QCA1d uses V.21 high channel and is repeated in the exact QC format.
        wire = responder.tx_g711(QCA_SAMPLES, "pcmu")
        self.assertEqual(len(wire), QCA_SAMPLES)
        self.assertTrue(all(0 <= value <= 255 for value in wire))
        self.assertEqual(responder.stage, "post-qca-silence")

        self.assertEqual(responder.tx_g711(SILENCE_SAMPLES, "pcmu"),
                         bytes([0xFF]) * SILENCE_SAMPLES)
        qts = short_phase1.qts_codewords(peer["ucode"], "pcmu")
        self.assertEqual(responder.tx_g711(len(qts), "pcmu"), qts)
        self.assertEqual(responder.stage, "anspcm")
        self.assertEqual(responder.tx_g711(301, "pcmu"),
                         v92_anspcm.codewords(-12.0, "pcmu"))

    def test_toneq_requires_fifty_ms_of_a_dominant_980_hz_tone(self):
        peer = qc.parse(qc.build("QC1a", lapm=False, code=0b0100))
        responder = V92QuickConnectAnswerer(peer=peer, law="pcmu", lapm_offer=False)
        responder.tx_g711(QCA_SAMPLES + SILENCE_SAMPLES + 816, "pcmu")
        t = np.arange(800) / 8000.0
        tone = (3500.0 * np.sin(2 * math.pi * 980.0 * t)).astype(np.int16)
        for start in range(0, 320, 160):
            responder.feed_pcm(tone[start:start + 160])
        self.assertFalse(responder.complete)      # 40 ms is too short
        responder.feed_pcm(tone[320:480])
        self.assertFalse(responder.complete)  # TONEq is accepted; 75 ms silence remains
        self.assertEqual(responder.stage, "post-toneq-silence")
        self.assertTrue(any(name == "toneq" for name, _ in responder.events))
        self.assertEqual(responder.tx_g711(SILENCE_SAMPLES - 1, "pcmu"),
                         bytes([0xFF]) * (SILENCE_SAMPLES - 1))
        self.assertFalse(responder.complete)
        self.assertEqual(responder.tx_g711(1, "pcmu"), bytes([0xFF]))
        self.assertTrue(responder.complete)

    def test_qca1d_transmit_decodes_as_its_declared_v21_high_fields(self):
        peer = qc.parse(qc.build("QC1a", lapm=True, code=0b0100))
        responder = V92QuickConnectAnswerer(peer=peer, law="pcmu", lapm_offer=True,
                                            level_code=0b10)
        wire = responder.tx_g711(QCA_SAMPLES, "pcmu")
        pcm = np.asarray([ulaw_to_linear(v) for v in wire], dtype=float)
        rx = v8._FskRx(v8.V21_H)
        rx.feed(np.concatenate((np.zeros(400), pcm, np.zeros(400))))
        decoded = rx.quick_connect_sequences()
        qca = next((item for item in decoded if item["kind"] == "QCA1d"), None)
        self.assertIsNotNone(qca)
        self.assertEqual(qca["level_code"], 0b10)
        self.assertEqual(qca["level_dbm0"], -15.0)
        self.assertTrue(qca["lapm"])

    def test_toneq_below_level_threshold_is_not_accepted(self):
        peer = qc.parse(qc.build("QC1a", lapm=False, code=0b0100))
        responder = V92QuickConnectAnswerer(peer=peer, law="pcmu", lapm_offer=False)
        responder.tx_g711(QCA_SAMPLES + SILENCE_SAMPLES + 816, "pcmu")
        t = np.arange(160) / 8000.0
        low = (100.0 * np.sin(2 * math.pi * 980.0 * t)).astype(np.int16)
        for _ in range(5):
            responder.feed_pcm(low)
        self.assertFalse(responder._toneq_seen)

    def test_toneq_rejects_another_tone_and_timeout_expires(self):
        peer = qc.parse(qc.build("QC1a", lapm=True, code=0b0100))
        responder = V92QuickConnectAnswerer(peer=peer, law="pcmu", lapm_offer=True)
        responder.tx_g711(QCA_SAMPLES + SILENCE_SAMPLES + 816, "pcmu")
        t = np.arange(160) / 8000.0
        other = (3500.0 * np.sin(2 * math.pi * 2100.0 * t)).astype(np.int16)
        responder.feed_pcm(other)
        self.assertFalse(responder.complete)
        responder.feed_gap(TIMEOUT_AFTER_QCA - len(other) - 1)
        self.assertFalse(responder.expired)
        responder.feed_gap(1)
        self.assertTrue(responder.expired)

    def test_chain_enters_digital_phase2_after_qc_and_toneq(self):
        created = []

        def factory(**kwargs):
            engine = _Engine(**kwargs)
            created.append(engine)
            return engine

        chain = V8Chain(
            False, allow=V8_MOD_V34,
            protocols=V8_PROTOCOL_LAPM_V42,
            pcm_availability=V8_PCM_DIGITAL,
            v90_engine_factory=factory,
            v92_quick_connect=True,
            v92_quick_connect_law="pcmu")
        try:
            for _ in range(11):
                chain.tx_samples(160)
            signal = _qc1a_pcm(lapm=True)
            for start in range(0, len(signal), 160):
                chain.feed_pcm(_pcm_bytes(signal[start:start + 160]))
            self.assertEqual(chain.kind, "v92-qc")
            self.assertTrue(callable(chain.tx_g711))
            self.assertTrue(callable(chain.feed_g711))

            # Drive QCA/QTS through the G.711 path, then present TONEq.
            chain.tx_g711(QCA_SAMPLES + SILENCE_SAMPLES + 816, "pcmu")
            tone = (3500.0 * np.sin(2 * math.pi * 980.0
                                    * np.arange(160) / 8000.0)).astype(np.int16)
            for _ in range(3):
                chain._feed_qc_g711(
                    bytes([0xFF]) * 160, "pcmu", {})   # interspersed silence
            for start in range(0, 800, 160):
                chain._feed_qc_g711(
                    bytes(linear_to_ulaw(int(x)) for x in tone), "pcmu", {})
                if chain.kind == "v90":
                    break
            self.assertEqual(chain.kind, "v92-qc")
            self.assertFalse(created)
            self.assertEqual(chain.tx_g711(SILENCE_SAMPLES, "pcmu"),
                             bytes([0xFF]) * SILENCE_SAMPLES)
            chain._feed_qc_g711(bytes([0xFF]) * 160, "pcmu", {})
            self.assertEqual(chain.kind, "v90")
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].negotiation["quick_connect"])
            self.assertEqual(created[0].negotiation["pcm_negotiated_bits"],
                             V8_PCM_DIGITAL)
            self.assertTrue(created[0].negotiation["protocols"]
                            & V8_PROTOCOL_LAPM_V42)
        finally:
            chain.close()


if __name__ == "__main__":
    unittest.main()
