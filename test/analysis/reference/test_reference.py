"""Test reference tracking through allocations, calls, indexing, and loops."""

import sys
import math
from typing import Any

import pytest
from kirin import types
from kirin.dialects import scf, func, ilist

from bloqade import squin
from bloqade.types import Qubit, QubitType, MeasurementResult
from bloqade.analysis.reference.lattice import (
    CLASSICAL,
    Slot,
    Whole,
    Unknown,
    Returned,
    Positions,
    QubitList,
    overlap,
    is_allocator,
    static_length,
)
from bloqade.analysis.reference.analysis import analyze, as_positions

QUBIT_LIST = ilist.IListType[QubitType, types.Any]


def refs_of(kernel):
    mt = kernel.similar()
    return mt, analyze(mt)


def statements(mt, kind):
    return [n for n in mt.callable_region.walk() if isinstance(n, kind)]


def calls(mt, name):
    return [n for n in statements(mt, func.Invoke) if n.callee.sym_name == name]


# -- kernels -------------------------------------------------------------------


@squin.kernel
def prep(q: Qubit) -> tuple[Qubit, MeasurementResult]:
    squin.h(q)
    r = q
    return r, squin.qubit.measure(r)


@squin.kernel
def maker() -> tuple[Qubit, MeasurementResult]:
    a = squin.qubit.new()
    return a, squin.qubit.measure(a)


@squin.kernel
def through(q: Qubit) -> Qubit:
    return prep(q)[0]


@squin.kernel
def twice() -> tuple[Qubit, Qubit]:
    a = squin.qubit.new()
    return a, a


@squin.kernel
def elements(qs: ilist.IList[Qubit, Any], i: int):
    q = squin.qubit.new()
    made = maker()
    squin.cx(qs[1], qs[-1])
    squin.cx(qs[i], q)
    squin.cx(made[0], q)
    prep(q)
    squin.h(q)
    for _ in range(2):
        squin.h(qs[0])
    return made[1]


@squin.kernel
def sized():
    qs = squin.qalloc(3)
    squin.cx(qs[-1], qs[0])
    squin.broadcast.x([qs[0], qs[2]])
    return squin.qubit.measure(qs[2])


@squin.kernel
def bad_reads():
    qs = squin.qalloc(3)
    squin.broadcast.x(qs[0:2])
    squin.h(qs[5])


@squin.kernel
def swaps_in_loop():
    q = squin.qubit.new()
    p = squin.qubit.new()
    for _ in range(2):
        squin.h(q)
        q = p
    return squin.qubit.measure(q)


@squin.kernel
def rotation(theta: float):
    q = squin.qubit.new()
    squin.rx(theta, q)
    squin.rz(math.pi / 2, q)
    return squin.qubit.measure(q)


# -- the rules -----------------------------------------------------------------


def test_parameters_allocations_and_elements():
    mt, refs = refs_of(elements)
    _, qs, i = mt.callable_region.blocks[0].args
    assert refs[qs] == Whole(qs) and refs[i] == CLASSICAL
    cx = calls(mt, "cx")
    assert [refs[v] for v in cx[0].inputs] == [Slot(qs, 1), Slot(qs, -1)]
    (new,) = calls(mt, "new")
    fresh = Whole(Returned(new, 0))
    assert [refs[v] for v in cx[1].inputs] == [Slot(qs, i), fresh]
    (made,) = calls(mt, "maker")
    assert [refs[v] for v in cx[2].inputs] == [Whole(Returned(made, 0)), fresh]
    assert refs[made.result] == Positions((Whole(Returned(made, 0)), CLASSICAL))
    h_prepped, h_loop = calls(mt, "h")
    assert refs[h_prepped.inputs[0]] == fresh  # the qubit lent to prep stays q
    assert refs[h_loop.inputs[0]] == Slot(qs, 0)
    (ret,) = statements(mt, func.Return)
    assert refs[ret.value] == CLASSICAL


def test_static_length_normalizes_constant_indices():
    mt, refs = refs_of(sized)
    (qs,) = (n.result for n in statements(mt, func.Invoke) if is_allocator(n))
    (cx,) = calls(mt, "cx")
    assert [refs[v] for v in cx.inputs] == [Slot(qs, 2), Slot(qs, 0)]
    (operands,) = statements(mt, ilist.New)  # the inlined broadcast's operand
    assert refs[operands.result] == QubitList((Slot(qs, 0), Slot(qs, 2)))


def reason(ref) -> str | None:
    return ref.reason if isinstance(ref, Unknown) else None


def test_unknowns_carry_their_reason():
    mt, refs = refs_of(bad_reads)
    (x,) = calls(mt, "x")
    assert reason(refs[x.inputs[0]]) == "a slice of a register"
    (h,) = calls(mt, "h")
    assert reason(refs[h.inputs[0]]) == "a constant index out of range"


def test_a_qubit_swapped_by_a_loop_is_unknown():
    """A loop-carried reference becomes unknown when the body changes its root."""
    mt, refs = refs_of(swaps_in_loop)
    (loop,) = statements(mt, scf.For)
    carried = [a for a in loop.body.blocks[0].args[1:] if a.type.is_subseteq(QubitType)]
    assert "a qubit carried by a loop or branch" in [reason(refs[a]) for a in carried]
    assert all(isinstance(refs[a], (Whole, Unknown)) for a in carried)


# -- what calls hand back ----------------------------------------------------------


def test_a_lent_qubit_handed_back_is_the_callers():
    """Nested calls preserve the root of a returned argument."""
    mt, refs = refs_of(through)
    (q,) = mt.callable_region.blocks[0].args[1:]
    (ret,) = statements(mt, func.Return)
    assert refs[ret.value] == Whole(q)
    (call,) = calls(mt, "prep")
    assert refs[call.result] == Positions((Whole(q), CLASSICAL))


def test_an_allocation_made_by_a_call_is_rooted_at_the_call():
    mt, refs = refs_of(elements)
    (made,) = calls(mt, "maker")
    assert refs[made.result] == Positions((Whole(Returned(made, 0)), CLASSICAL))


@squin.kernel
def reads_twice():
    pair = twice()
    squin.cx(pair[0], pair[1])


def test_an_allocation_at_two_positions_is_known_once():
    mt, refs = refs_of(reads_twice)
    (call,) = calls(mt, "twice")
    (cx,) = calls(mt, "cx")
    first, second = (refs[v] for v in cx.inputs)
    assert first == Whole(Returned(call, 0))
    assert reason(second) == "an allocation returned at two positions"


@squin.kernel
def make_pair() -> ilist.IList[Qubit, Any]:
    return squin.qalloc(2)


@squin.kernel
def reads_made_pair():
    qs = make_pair()
    squin.cx(qs[1], qs[-1])


def test_a_returned_register_keeps_its_static_length():
    """Returned allocation lengths allow normalization of negative indices."""
    mt, refs = refs_of(reads_made_pair)
    (cx,) = calls(mt, "cx")
    first, second = (refs[v] for v in cx.inputs)
    assert isinstance(first, Slot) and isinstance(second, Slot)
    assert first.root == second.root and static_length(first.root) == 2
    assert first.index == second.index == 1


def test_unknown_return_keeps_its_reason_at_every_position():
    reason = Unknown("a qubit carried by a loop or branch")
    assert as_positions(reason, 2) == (reason, reason)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="kirin's constant propagation overflows the C stack defining a recursive kernel",
)
def test_recursive_qubit_return_is_unknown():
    @squin.kernel
    def rec(q: Qubit, n: int) -> Qubit:
        squin.h(q)
        if n > 0:
            r = rec(q, n - 1)
        else:
            r = q
        return r

    mt, refs = refs_of(rec)
    (call,) = calls(mt, "rec")
    assert reason(refs[call.result]) == "the result of a recursive call"


# -- certainty, not proof -------------------------------------------------------


@squin.kernel
def pairs(n: int, i: int):
    qs = squin.qalloc(n)
    squin.cx(qs[i], qs[i])
    squin.cx(qs[0], qs[-1])
    squin.cx(qs[i], qs[i + 1])


def test_overlap_is_certainty_not_proof():
    """Overlap detects definite aliases and may miss runtime aliases."""
    mt, refs = refs_of(pairs)
    (a, b), (c, d), (e, f) = (
        (refs[call.inputs[0]], refs[call.inputs[1]]) for call in calls(mt, "cx")
    )
    assert overlap(a, b)  # qs[i], qs[i]
    assert not overlap(c, d)  # qs[0], qs[-1]: coincide only if n == 1
    assert not overlap(e, f)  # qs[i], qs[i + 1]
    assert overlap(Whole(a.root), a)  # a register and its element


def test_unknown_is_one_top_element():
    """Unknown reasons do not affect equality; joins preserve the first reason."""
    a, b = Unknown("a"), Unknown("b")
    assert a == b and a.is_subseteq(b) and b.is_subseteq(a)
    assert a.join(b) is a and b.join(a) is b
    assert CLASSICAL.join(a) is a and a.join(CLASSICAL) is a
