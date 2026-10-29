"""Check the ordering laws and call identities of wire position validation."""

from itertools import product

from kirin.dialects import py

from bloqade.analysis.indices import Affine
from bloqade.squin.wire.types import WireType, RegisterType
from bloqade.squin.analysis.validation.wire.identities import Handle, Origin


def test_origin_join_and_meet_obey_lattice_laws():
    a, b = Origin(Handle("a", WireType)), Origin(Handle("b", WireType))
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
    a, b = Origin(Handle("a", WireType)), Origin(Handle("b", WireType))
    assert Origin((a, b)).join(Origin((a, a))) == Origin((a, Origin.top()))


def test_call_results_keep_borrowed_roots_and_distinguish_fresh_roots():
    borrowed = Origin(Handle("input", WireType))
    fresh = Origin(Handle("local", WireType))
    outputs = Origin((borrowed, fresh))
    first = outputs.at_call(py.Constant(0), frozenset({"input"}))
    second = outputs.at_call(py.Constant(0), frozenset({"input"}))
    assert isinstance(first.data, tuple) and isinstance(second.data, tuple)
    assert first.data[0] == second.data[0] == borrowed
    assert first.data[1].roots().isdisjoint(second.data[1].roots())


def test_register_origins_resolve_negative_indices_and_reject_invalid_elements():
    register = Handle("register", RegisterType, Affine(3))
    assert register.pick(Affine(-1)) == register.pick(Affine(2))
    assert register.pick(Affine(3)) == Origin.top()
    assert register.pick(Affine(-4)) == Origin.top()
    assert Handle("wire", WireType).pick(Affine(0)) == Origin.top()
