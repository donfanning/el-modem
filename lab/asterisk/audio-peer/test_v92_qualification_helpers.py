import unittest
from types import SimpleNamespace

from v8 import V8_PROTOCOL_LAPM_V42
from v92_qualification_helpers import lapm_detect_required, lapm_payload_drained


class V92QualificationHelperTests(unittest.TestCase):
    def test_v8_lapm_agreement_selects_direct_startup(self):
        self.assertFalse(lapm_detect_required(V8_PROTOCOL_LAPM_V42))
        self.assertTrue(lapm_detect_required(0))
        self.assertTrue(lapm_detect_required(
            V8_PROTOCOL_LAPM_V42, force_detect=True))

    def test_payload_drain_ignores_control_ack_queue_but_requires_all_data_acked(self):
        drained = SimpleNamespace(
            stats={"wire_out": 2048, "wire_acked": 2048},
            _unacked=[], pending=lambda: 0)
        self.assertTrue(lapm_payload_drained(drained))
        drained.stats["wire_acked"] = 2047
        self.assertFalse(lapm_payload_drained(drained))
        drained.stats["wire_acked"] = 2048
        drained._unacked = [object()]
        self.assertFalse(lapm_payload_drained(drained))
        self.assertFalse(lapm_payload_drained(None))


if __name__ == "__main__":
    unittest.main()
