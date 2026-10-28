"""Compare index separation proofs with concrete integer assignments."""

import itertools

from kirin import ir, types

from bloqade.analysis.indices import Affine, same


def test_affine_proofs_agree_with_concrete_integer_indices():
    block = ir.Block(argtypes=(types.Int, types.Int))
    i, j = block.args
    expressions = [
        Affine.make(offset, {i: a, j: b})
        for offset, a, b in itertools.product(range(-2, 3), repeat=3)
    ]
    assignments = list(itertools.product(range(-3, 4), repeat=2))
    for left, right in itertools.combinations(expressions, 2):
        if same(left, right) is False:
            for x, y in assignments:
                values = {i: x, j: y}
                lhs = left.const + sum(
                    coefficient * values[symbol] for symbol, coefficient in left.terms
                )
                rhs = right.const + sum(
                    coefficient * values[symbol] for symbol, coefficient in right.terms
                )
                assert lhs != rhs
