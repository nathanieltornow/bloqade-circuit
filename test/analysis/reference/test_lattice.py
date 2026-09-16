"""Test reference lattice operations, formatting, and queries."""

from typing import Any, Literal

from kirin import types
from kirin.dialects import func, ilist

from bloqade import squin
from bloqade.types import Qubit, QubitType, MeasurementResult
from bloqade.analysis.reference.lattice import (
    BOTTOM,
    CARRIED,
    CLASSICAL,
    Slot,
    Whole,
    Unknown,
    Returned,
    Positions,
    QubitList,
    roots,
    length_key,
    qubit_refs,
    same_length,
    is_allocator,
    typed_length,
    static_length,
    mentions_quantum,
    mentions_unknown,
)
from bloqade.analysis.reference.analysis import analyze


def analyzed(kernel):
    mt = kernel.similar()
    return mt, analyze(mt)


def allocations(mt):
    return [n.result for n in mt.callable_region.walk() if is_allocator(n)]


def calls(mt, name):
    return [
        n
        for n in mt.callable_region.walk()
        if isinstance(n, func.Invoke) and n.callee.sym_name == name
    ]


# -- kernels -------------------------------------------------------------------


@squin.kernel
def two_registers(n: int, qs: ilist.IList[Qubit, Any]):
    a = squin.qalloc(n)
    b = squin.qalloc(n)
    c = squin.qalloc(2)
    d = squin.qalloc(2)
    e = squin.qalloc(-1)
    squin.cx(a[0], b[0])
    squin.cx(c[0], d[0])
    squin.cx(e[0], qs[0])


@squin.kernel
def maker() -> tuple[Qubit, MeasurementResult]:
    a = squin.qubit.new()
    return a, squin.qubit.measure(a)


@squin.kernel
def returns_list() -> ilist.IList[Qubit, Any]:
    a = squin.qubit.new()
    b = squin.qubit.new()
    return [a, b]


@squin.kernel
def returns_element() -> Qubit:
    qs = squin.qalloc(2)
    return qs[0]


@squin.kernel(fold=False, typeinfer=False)
def returns_one_for_two() -> tuple[Qubit, Qubit]:
    return squin.qubit.new()


@squin.kernel
def declares_three() -> ilist.IList[Qubit, Literal[3]]:
    return squin.qalloc(2)


@squin.kernel
def reads_declared_three():
    qs = declares_three()
    squin.h(qs[0])


@squin.kernel
def reads_returned_list():
    qs = returns_list()
    squin.h(qs[0])


@squin.kernel
def reads_returned_element():
    squin.h(returns_element())


@squin.kernel(fold=False, typeinfer=False)
def reads_one_for_two():
    pair = returns_one_for_two()
    squin.cx(pair[0], pair[1])


@squin.kernel
def reads_pair_at_runtime(i: int):
    made = maker()
    squin.h(made[0])
    a = squin.qubit.new()
    b = squin.qubit.new()
    squin.h([a, b][i])
    squin.h([a, b][2])


@squin.kernel(fold=False, typeinfer=False)
def takes_tuple(pair: tuple[Qubit, Qubit]):
    squin.h(pair[0])


# -- order and printing ----------------------------------------------------------


def test_order():
    mt, refs = analyzed(two_registers)
    a, b = allocations(mt)[:2]
    whole, slot = Whole(a), Slot(a, 0)
    assert BOTTOM.join(whole) is whole and whole.join(BOTTOM) is whole
    assert whole.join(slot) == Unknown(CARRIED)
    assert isinstance(whole.join(Whole(b)), Unknown)
    assert isinstance(CLASSICAL.join(whole), Unknown)
    assert whole.meet(Unknown("x")) is whole and Unknown("x").meet(whole) is whole
    assert whole.meet(slot) is BOTTOM and whole.meet(whole) is whole
    assert BOTTOM.is_subseteq(whole) and whole.is_subseteq(Unknown("x"))
    assert not whole.is_subseteq(slot)


def test_printed_forms():
    mt, refs = analyzed(two_registers)
    a = allocations(mt)[0]
    n = mt.callable_region.blocks[0].args[1]
    assert repr(Whole(a)) == "Whole(%a)"
    assert repr(Slot(a, 0)) == "Slot(%a, 0)"
    assert repr(Slot(a, n)) == "Slot(%a, %n)"
    assert repr(QubitList((Whole(a), Slot(a, 1)))) == "[Whole(%a), Slot(%a, 1)]"
    assert repr(Positions((CLASSICAL, Unknown("why")))) == "(Classical, Unknown: why)"
    assert repr(BOTTOM) == "Bottom"
    mt, refs = analyzed(reads_pair_at_runtime)
    (made,) = calls(mt, "maker")
    assert repr(Whole(Returned(made, 0))) == "Whole(maker()[0])"


# -- lengths ---------------------------------------------------------------------


def test_lengths_of_allocations_and_parameters():
    mt, refs = analyzed(two_registers)
    a, b, c, d, e = allocations(mt)
    qs = mt.callable_region.blocks[0].args[2]
    assert static_length(a) is None and static_length(c) == 2
    assert static_length(e) == 0  # a negative size allocates nothing
    assert length_key(a) is length_key(b) and same_length(a, b)
    assert same_length(c, d) and not same_length(a, c)
    assert length_key(qs) is None and not same_length(a, qs)
    assert typed_length(ilist.IListType[QubitType, types.Literal(3)]) == 3
    assert typed_length(ilist.IListType[QubitType, types.Any]) is None


# -- questions about references ----------------------------------------------------


def test_qubit_refs_expand_registers_of_static_length():
    mt, refs = analyzed(two_registers)
    a, _, c, *_ = allocations(mt)
    assert qubit_refs(Whole(c)) == [Slot(c, 0), Slot(c, 1)]
    assert qubit_refs(Whole(a)) is None  # a runtime length has no slots
    assert qubit_refs(Slot(a, 1)) == [Slot(a, 1)]
    assert qubit_refs(QubitList((Slot(a, 0), Slot(c, 1)))) == [Slot(a, 0), Slot(c, 1)]
    assert qubit_refs(QubitList((Whole(c),))) is None
    assert qubit_refs(Positions((Slot(a, 0),))) is None
    assert qubit_refs(Unknown("x")) is None and qubit_refs(CLASSICAL) is None


def test_roots_and_mentions():
    mt, refs = analyzed(two_registers)
    a, _, c, *_ = allocations(mt)
    mixed = Positions((Slot(a, 0), CLASSICAL, QubitList((Whole(c),))))
    assert roots(mixed) == [a, c]
    assert roots(CLASSICAL) == [] and roots(Unknown("x")) == []
    assert mentions_quantum(mixed) and not mentions_quantum(Positions((CLASSICAL,)))
    assert mentions_quantum(Unknown("x")) and not mentions_quantum(CLASSICAL)
    assert mentions_unknown(Positions((CLASSICAL, Unknown("x"))))
    assert not mentions_unknown(mixed)


# -- reasons -----------------------------------------------------------------------


def reason(ref) -> str | None:
    return ref.reason if isinstance(ref, Unknown) else None


def test_returns_the_lattice_cannot_root():
    mt, refs = analyzed(reads_returned_list)
    (h,) = calls(mt, "h")
    assert reason(refs[h.inputs[0]]) == "a list holding an allocation of the call"
    mt, refs = analyzed(reads_returned_element)
    (h,) = calls(mt, "h")
    assert reason(refs[h.inputs[0]]) == "an element of a register the call allocates"
    mt, refs = analyzed(reads_one_for_two)
    (cx,) = calls(mt, "cx")
    assert {reason(refs[v]) for v in cx.inputs} == {
        "a return that does not match the declared output positions"
    }


def test_the_allocated_length_is_what_a_returned_register_bears():
    mt, refs = analyzed(reads_declared_three)
    (h,) = calls(mt, "h")
    (call,) = calls(mt, "declares_three")
    ref = refs[h.inputs[0]]
    assert ref == Slot(Returned(call, 0), 0) and static_length(ref.root) == 2


def test_reads_of_tuples_and_lists():
    mt, refs = analyzed(reads_pair_at_runtime)
    from_tuple, runtime_list, out_of_range = calls(mt, "h")
    (made,) = calls(mt, "maker")
    assert refs[from_tuple.inputs[0]] == Whole(Returned(made, 0))
    assert (
        reason(refs[runtime_list.inputs[0]])
        == "a qubit list or tuple read at a runtime index"
    )
    assert reason(refs[out_of_range.inputs[0]]) == "a constant index out of range"


def test_a_tuple_parameter_has_no_convention():
    mt, refs = analyzed(takes_tuple)
    pair = mt.callable_region.blocks[0].args[1]
    assert reason(refs[pair]) == "a member of a tuple parameter"
