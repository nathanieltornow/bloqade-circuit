"""Test reference lattice operations, formatting, and queries."""

from typing import Any, Literal

from kirin.dialects import func, ilist

from bloqade import squin
from bloqade.types import Qubit, MeasurementResult
from bloqade.jeff.analysis.reference import (
    CARRIED,
    UNTRACKED,
    Slot,
    Items,
    Whole,
    Bottom,
    Unknown,
    Returned,
    Positions,
    QubitReferenceAnalysis,
    is_allocator,
)


def analyzed(kernel):
    mt = kernel.similar()
    frame, _ = QubitReferenceAnalysis(mt.dialects).run(mt)
    return mt, frame.entries


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
    assert Bottom().join(whole) is whole and whole.join(Bottom()) is whole
    assert whole.join(slot) == Unknown(CARRIED)
    assert isinstance(whole.join(Whole(b)), Unknown)
    assert isinstance(UNTRACKED.join(whole), Unknown)
    assert whole.meet(Unknown("x")) is whole and Unknown("x").meet(whole) is whole
    assert whole.meet(slot) is Bottom() and whole.meet(whole) is whole
    assert Bottom().is_subseteq(whole) and whole.is_subseteq(Unknown("x"))
    assert not whole.is_subseteq(slot)


def test_printed_forms():
    mt, refs = analyzed(two_registers)
    a = allocations(mt)[0]
    n = mt.callable_region.blocks[0].args[1]
    assert repr(Whole(a)) == "Whole(%a)"
    assert repr(Slot(a, 0)) == "Slot(%a, 0)"
    assert repr(Slot(a, n)) == "Slot(%a, %n)"
    assert repr(Items((Whole(a), Slot(a, 1)))) == "[Whole(%a), Slot(%a, 1)]"
    assert repr(Positions((UNTRACKED, Unknown("why")))) == "(Untracked, Unknown: why)"
    assert repr(Bottom()) == "Bottom"
    mt, refs = analyzed(reads_pair_at_runtime)
    (made,) = calls(mt, "maker")
    assert repr(Whole(Returned(made, 0))) == "Whole(maker()[0])"


# -- lengths ---------------------------------------------------------------------


def test_lengths_of_allocations_and_parameters():
    mt, refs = analyzed(two_registers)
    a, b, c, d, e = allocations(mt)
    qs = mt.callable_region.blocks[0].args[2]
    static_length = QubitReferenceAnalysis(mt.dialects).register_length
    assert static_length(a) is None and static_length(c) == 2
    assert static_length(e) == 0  # a negative size allocates nothing
    assert static_length(qs) is None


def reason(ref) -> str | None:
    return ref.reason if isinstance(ref, Unknown) else None


def test_returns_the_lattice_cannot_root():
    mt, refs = analyzed(reads_returned_list)
    (h,) = calls(mt, "h")
    assert reason(refs[h.inputs[0]]) == "a list that holds an allocation of the call"
    mt, refs = analyzed(reads_returned_element)
    (h,) = calls(mt, "h")
    assert reason(refs[h.inputs[0]]) == "an item of a register that the call allocates"
    mt, refs = analyzed(reads_one_for_two)
    (cx,) = calls(mt, "cx")
    assert {reason(refs[v]) for v in cx.inputs} == {
        "a return that does not match the declared output positions"
    }


def test_the_allocated_length_is_what_a_returned_register_bears():
    mt = reads_declared_three.similar()
    analysis = QubitReferenceAnalysis(mt.dialects)
    frame, _ = analysis.run(mt)
    (h,) = calls(mt, "h")
    (call,) = calls(mt, "declares_three")
    ref = frame.entries[h.inputs[0]]
    assert ref == Slot(Returned(call, 0), 0)
    assert analysis.register_length(analysis.origin(ref.root)) == 2


def test_reads_of_tuples_and_lists():
    mt, refs = analyzed(reads_pair_at_runtime)
    from_tuple, runtime_list, out_of_range = calls(mt, "h")
    (made,) = calls(mt, "maker")
    assert refs[from_tuple.inputs[0]] == Whole(Returned(made, 0))
    assert (
        reason(refs[runtime_list.inputs[0]])
        == "a list or tuple read at a runtime index"
    )
    assert reason(refs[out_of_range.inputs[0]]) == "a constant index out of range"


def test_a_tuple_parameter_has_no_convention():
    mt, refs = analyzed(takes_tuple)
    pair = mt.callable_region.blocks[0].args[1]
    assert reason(refs[pair]) == "a member of a tuple parameter"
