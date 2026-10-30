"""Check the ordering laws and call identities of wire position validation."""

from itertools import product

from kirin.dialects import py

from bloqade.analysis.indices import Affine
from bloqade.squin.analysis.validation.wire.origins import (
    Origin,
    QubitHandle,
    RegisterHandle,
    element_of,
)


def test_origin_join_and_meet_obey_lattice_laws():
    a, b = Origin(QubitHandle("a")), Origin(QubitHandle("b"))
    top, bottom = Origin.top(), Origin.bottom()
    values = (
        top,
        bottom,
        a,
        b,
        Origin((a, b)),
        Origin((top, b)),
        Origin((a, bottom)),
        Origin(()),
    )
    for left, right in product(values, repeat=2):
        assert left.join(right) == right.join(left)
        assert left.meet(right) == right.meet(left)
        assert left.join(left.meet(right)) == left
        assert left.meet(left.join(right)) == left
        assert left.is_subseteq(left.join(right))
        assert left.meet(right).is_subseteq(left)
    for a, b, c in product(values, repeat=3):
        assert a.join(b).join(c) == a.join(b.join(c))
        assert a.meet(b).meet(c) == a.meet(b.meet(c))


def test_tuple_joins_keep_the_fields_that_agree():
    a, b = Origin(QubitHandle("a")), Origin(QubitHandle("b"))
    assert Origin((a, b)).join(Origin((a, a))) == Origin((a, Origin.top()))


def test_call_results_keep_borrowed_roots_and_distinguish_fresh_roots():
    borrowed = Origin(QubitHandle("input"))
    fresh = Origin(QubitHandle("local"))
    outputs = Origin((borrowed, fresh))
    first = outputs.at_call(py.Constant(0), frozenset({"input"}))
    second = outputs.at_call(py.Constant(0), frozenset({"input"}))
    assert isinstance(first.data, tuple) and isinstance(second.data, tuple)
    assert first.data[0] == second.data[0] == borrowed
    assert first.data[1].roots().isdisjoint(second.data[1].roots())


def test_register_origins_resolve_negative_indices_and_reject_invalid_elements():
    register = Origin(RegisterHandle("register", Affine(3)))
    assert element_of(register, Affine(-1)) == element_of(register, Affine(2))
    assert element_of(register, Affine(3)) == Origin.top()
    assert element_of(register, Affine(-4)) == Origin.top()
    assert element_of(Origin(QubitHandle("wire")), Affine(0)) == Origin.top()
