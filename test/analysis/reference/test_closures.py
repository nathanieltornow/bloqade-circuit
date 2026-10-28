"""Test closures, `ilist.map` and `ilist.for_each` in the reference analysis."""

from typing import Literal

from kirin.dialects import func, ilist

from bloqade import squin
from bloqade.types import Qubit
from bloqade.analysis.reference import (
    Iter,
    Part,
    Slot,
    Slice,
    Affine,
    Closure,
    Unknown,
    Register,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis


def refs_of(kernel):
    mt = kernel.similar()
    frame, _ = QubitReferenceAnalysis(mt.dialects).run(mt)
    return mt, frame.entries


def statements(mt, kind):
    return [n for n in mt.callable_region.walk() if isinstance(n, kind)]


def operands(mt, name):
    """Return the first operand of each call of the gate kernel `name`, in order."""
    return [
        node.inputs[0]
        for node in statements(mt, func.Invoke)
        if node.callee.sym_name == name
    ]


def reason(ref) -> str | None:
    return ref.reason if isinstance(ref, Unknown) else None


@squin.kernel
def ident(q: Qubit) -> Qubit:
    return q


@squin.kernel
def identity_map(qs: ilist.IList[Qubit, Literal[3]]):
    ys = ilist.map(ident, qs)
    squin.x(ys[0])
    squin.x(qs[0])


def test_an_identity_map_aliases_its_register():
    """`ys[0]` and `qs[0]` are one qubit, so they get one place."""
    mt, refs = refs_of(identity_map)
    qs = mt.callable_region.blocks[0].args[1]
    (mapped,) = statements(mt, ilist.Map)
    assert refs[mapped.result] == Register(qs)
    first, second = (refs[v] for v in operands(mt, "x"))
    assert first == second == Slot(qs, 0)


@squin.kernel
def shifted_map(qs: ilist.IList[Qubit, Literal[7]]):
    def next_one(k: int) -> Qubit:
        return qs[k + 1]

    def even(k: int) -> Qubit:
        return qs[2 * k]

    nexts = ilist.map(next_one, ilist.range(3))
    evens = ilist.map(even, ilist.range(4))
    squin.x(nexts[1])
    squin.x(evens[3])


def test_a_map_of_an_affine_index_gives_a_slice():
    mt, refs = refs_of(shifted_map)
    qs = mt.callable_region.blocks[0].args[1]
    nexts, evens = statements(mt, ilist.Map)
    assert refs[nexts.result] == Part(qs, (Slice(Affine(1), 1, Affine(3)),))
    assert refs[evens.result] == Part(qs, (Slice(Affine(0), 2, Affine(4)),))
    assert [refs[v] for v in operands(mt, "x")] == [Slot(qs, 2), Slot(qs, 6)]
    lambdas = statements(mt, func.Lambda)
    assert all(isinstance(refs[f.result], Closure) for f in lambdas)


@squin.kernel
def allocating_map():
    def fresh(k: int) -> Qubit:
        return squin.qubit.new()

    made = ilist.map(fresh, ilist.range(3))
    squin.x(made[2])


def test_a_map_that_allocates_gives_a_new_root():
    mt, refs = refs_of(allocating_map)
    (mapped,) = statements(mt, ilist.Map)
    assert refs[mapped.result] == Register(mapped.result)
    (x,) = operands(mt, "x")
    assert refs[x] == Slot(mapped.result, 2)


@squin.kernel
def constant_map(qs: ilist.IList[Qubit, Literal[3]]):
    def first(k: int) -> Qubit:
        return qs[0]

    same = ilist.map(first, ilist.range(3))
    squin.x(same[1])


def test_a_map_whose_items_are_one_qubit_is_unknown():
    mt, refs = refs_of(constant_map)
    (mapped,) = statements(mt, ilist.Map)
    assert reason(refs[mapped.result]) == "a map whose items are all one place"


@squin.kernel
def each(qs: ilist.IList[Qubit, Literal[3]]):
    def flip(q: Qubit):
        squin.x(q)
        squin.y(qs[0])

    ilist.for_each(flip, qs)


def test_for_each_runs_the_body_on_the_item_at_its_symbol():
    mt, refs = refs_of(each)
    qs = mt.callable_region.blocks[0].args[1]
    (loop,) = statements(mt, ilist.ForEach)
    k = Affine.of(Iter(loop, range(3)))
    assert [refs[v] for v in operands(mt, "x")] == [Slot(qs, k)]
    assert [refs[v] for v in operands(mt, "y")] == [Slot(qs, 0)]
