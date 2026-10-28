"""Compare affine alias proofs with concrete integer assignments."""

import itertools

from kirin import ir, types

from bloqade.analysis.reference import (
    Slot,
    Affine,
    Unknown,
    Register,
    disjoint,
    must_alias,
)


def test_affine_proofs_agree_with_concrete_integer_indices():
    block = ir.Block(argtypes=(types.Any, types.Int, types.Int))
    root, i, j = block.args
    expressions = [
        Affine.make(offset, {i: a, j: b})
        for offset, a, b in itertools.product(range(-2, 3), repeat=3)
    ]
    assignments = list(itertools.product(range(-3, 4), repeat=2))
    for left, right in itertools.combinations(expressions, 2):
        a, b = Slot(root, left), Slot(root, right)
        if disjoint(a, b):
            for x, y in assignments:
                values = {i: x, j: y}
                lhs = left.const + sum(
                    coefficient * values[symbol] for symbol, coefficient in left.terms
                )
                rhs = right.const + sum(
                    coefficient * values[symbol] for symbol, coefficient in right.terms
                )
                assert lhs != rhs


def test_unknown_references_prove_neither_identity_nor_separation():
    root = ir.TestValue(types.Any)
    unknown = Unknown("A runtime selection has no unique root.")
    for other in (unknown, Register(root), Slot(root, 0)):
        assert not disjoint(unknown, other)
        assert not must_alias(unknown, other)
