"""The spec-derived scrambler, encoders and tables reproduce the earlier revision's output exactly.

The digest was taken from the previous implementation before it was rewritten from the
Recommendation (lab/V32_PROVENANCE_REWRITE_2026_10_13.md)."""
import hashlib
import json
import random
import unittest

import v32_caller as m

GOLDEN = "98a8e982db0fa44e78b261afd9edd973d17a4c1aa76d650961825425a674242e"


class RewriteEquivalence(unittest.TestCase):
    def test_output_matches_earlier_revision(self):
        r = random.Random(5)
        bits = [r.getrandbits(1) for _ in range(5000)]
        out = {}
        for tap in (18, 5):
            s = m.Scrambler(tap)
            out[f'fwd{tap}'] = [s.fwd(b) for b in bits]
            s = m.Scrambler(tap)
            out[f'rev{tap}'] = [s.rev(b) for b in bits]
        e = m.Trellis7200Encoder()
        out['enc'] = [(lambda z: [z.real, z.imag])(e.encode(r.getrandbits(3))) for _ in range(3000)]
        out['tables'] = [repr(getattr(m, n)) for n in
                         ('POINTS', 'NR_DIFF', 'RATE_BITS', 'RATE_WORD_BASE', 'ZTAB4', 'TFSM', 'TE_DIFF')]
        out['tcm'] = repr(sorted((k, (v.real, v.imag)) for k, v in m.V32_TCM_9600.items()))
        self.assertEqual(hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest(), GOLDEN)


if __name__ == '__main__':
    unittest.main()
