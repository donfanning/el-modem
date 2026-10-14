"""Check V.32bis numeric tables independently against the 02/1991 standard.

Evaluates the first-party pure-Python V.32 module and compares its tables with values
re-derived here from the Recommendation.
This is standards conformance evidence, not a determination of copyright rights.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def literal(node):
    # ast.literal_eval rejects complex subtraction expressions such as 3-3j.
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = literal(node.left), literal(node.right)
        return left + right if isinstance(node.op, ast.Add) else left - right
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(item) for item in node.elts]
    return ast.literal_eval(node)


def expected():
    # Table 1: previous differential output p, incoming dibit q.
    trellis_diff = []
    for p in range(4):
        p1, p2 = p >> 1, p & 1
        for q in range(4):
            q1, q2 = q >> 1, q & 1
            trellis_diff.append(((p1 ^ q1) << 1) | (p2 ^ q2 ^ (p1 & q1)))

    # Table 2: A,B,D,C indices are Gray labels of quadrant angles.
    gray_to_quadrant = [0, 1, 3, 2]
    phase_steps = [1, 0, 2, 3]
    uncoded_diff = [gray_to_quadrant[(gray_to_quadrant[p] + phase_steps[q]) % 4]
                    for p in range(4) for q in range(4)]

    # Figure 1: state bits are the three delays, left to right (MSB to LSB).
    transitions = []
    for state in range(8):
        left, middle, right = state >> 2, (state >> 1) & 1, state & 1
        row = []
        for dibit in range(4):
            y1, y2 = dibit >> 1, dibit & 1
            branch = middle ^ y2
            next_middle = left ^ y1 ^ y2 ^ (branch & right)
            next_right = branch ^ (y1 & right)
            row.append((right << 2) | (next_middle << 1) | next_right)
        transitions.append(row)

    # Figure 2-4: label Y0,Y1,Y2,Q3 at each lattice position.
    labels = ['0010 1000 0111 1110', '1011 0001 1101 0100',
              '0110 1111 0011 1001', '1100 0101 1010 0000']
    points = [None] * 16
    for y, row in zip((3, 1, -1, -3), labels):
        for x, label in zip((-3, -1, 1, 3), row.split()):
            points[int(label, 2)] = complex(x, y)

    # Table 5 numbers bits in transmission order, B0 (MSB) through B15.
    rate_positions = {4800: 5, 7200: 9, 9600: 6, 12000: 10, 14400: 12}
    return dict(TE_DIFF=trellis_diff, NR_DIFF=uncoded_diff, TFSM=transitions,
                ZTAB4=points, POINTS=[complex(-3, -1), complex(1, -3),
                                     complex(-1, 3), complex(3, 1)],
                RATE_BITS={rate: 1 << (15 - bit) for rate, bit in rate_positions.items()},
                RATE_WORD_BASE=sum(1 << (15 - bit) for bit in (4, 7, 8, 11, 15)))


def audit(source):
    tables = expected()
    # The tables are now generated from the Recommendation's formulas, so read the evaluated values.
    # The module is first-party pure Python; import it with its own directory on the path.
    import importlib.util
    import sys
    sys.path.insert(0, str(source.parent))
    try:
        spec = importlib.util.spec_from_file_location('_v32_audited', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    found = {name: getattr(module, name, None) for name in tables}
    results = {name: found.get(name) == value for name, value in tables.items()}
    return dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                standard='ITU-T V.32bis (02/1991)', checks=results,
                all_pass=all(results.values()),
                scope='Numeric conformance only; does not certify historical source licensing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    print(json.dumps(result, indent=2))
    if not result['all_pass']:
        raise SystemExit(1)
