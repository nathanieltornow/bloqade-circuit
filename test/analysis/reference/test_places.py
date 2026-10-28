"""Test places with nested registers, slices, affine loop indices and scope exit."""

from typing import Literal

from kirin.dialects import scf, func, ilist

from bloqade import squin
from bloqade.types import Qubit
from bloqade.analysis.reference import (
    OUT_OF_SCOPE,
    Item,
    Iter,
    Part,
    Slot,
    Slice,
    Affine,
    Unknown,
    same,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis, qubit_items


def refs_of(kernel):
    mt = kernel.similar()
    analysis = QubitReferenceAnalysis(mt.dialects)
    frame, _ = analysis.run(mt)
    return mt, frame.entries, analysis


def gate_operands(mt, name):
    """Return the operands of each call of the gate kernel `name`, in order."""
    return [
        node.inputs
        for node in mt.callable_region.walk()
        if isinstance(node, func.Invoke) and node.callee.sym_name == name
    ]


def reason(ref) -> str | None:
    return ref.reason if isinstance(ref, Unknown) else None


@squin.kernel
def nested(qss: ilist.IList[ilist.IList[Qubit, Literal[2]], Literal[3]]):
    row = qss[2]
    squin.x(row[1])
    squin.broadcast.h(qss[0])


def test_a_nested_register_gives_rows_and_items():
    mt, refs, _ = refs_of(nested)
    qss = mt.callable_region.blocks[0].args[1]
    ((item,),) = gate_operands(mt, "x")
    assert refs[item] == Item(qss, (Affine(2), Affine(1)))
    assert repr(refs[item]) == "Place(%qss, (2, 1))"
    ((row,),) = gate_operands(mt, "h")
    assert refs[row] == Part(qss, (Affine(0),))
    assert qubit_items(refs[row]) == (
        Item(qss, (Affine(0), Affine(0))),
        Item(qss, (Affine(0), Affine(1))),
    )


@squin.kernel
def slices(qs: ilist.IList[Qubit, Literal[6]]):
    ys = qs[1:5]
    squin.x(ys[0])
    squin.x(ys[1:3][1])
    squin.x(qs[::2][2])
    squin.x(ys[-1])
    squin.broadcast.h(ys[::3])


def test_an_index_into_a_slice_reads_the_register():
    mt, refs, _ = refs_of(slices)
    qs = mt.callable_region.blocks[0].args[1]
    first, inner, strided, last = (refs[ops[0]] for ops in gate_operands(mt, "x"))
    assert first == Slot(qs, 1)
    assert inner == Slot(qs, 3)
    assert strided == Slot(qs, 4)
    assert last == Slot(qs, 4)
    ((picked,),) = gate_operands(mt, "h")
    assert refs[picked] == Part(qs, (Slice(Affine(1), 3, Affine(2)),))
    assert qubit_items(refs[picked]) == (Slot(qs, 1), Slot(qs, 4))


@squin.kernel
def ladder(qs: ilist.IList[Qubit, Literal[6]]):
    for i in range(5):
        squin.cx(qs[i], qs[i + 1])
    for j in range(3):
        squin.cz(qs[2 * j], qs[2 * j + 1])


def test_loop_indices_are_affine_in_the_loop_variable():
    mt, refs, _ = refs_of(ladder)
    first, second = (s for s in mt.callable_region.walk() if isinstance(s, scf.For))
    i, j = Affine.of(Iter(first, range(5))), Affine.of(Iter(second, range(3)))
    qs = mt.callable_region.blocks[0].args[1]
    ((a, b),) = gate_operands(mt, "cx")
    assert (refs[a], refs[b]) == (Slot(qs, i), Slot(qs, i + 1))
    assert same(refs[a].index, refs[b].index) is False
    ((c, d),) = gate_operands(mt, "cz")
    assert (refs[c], refs[d]) == (Slot(qs, j.scale(2)), Slot(qs, j.scale(2) + 1))
    assert same(refs[c].index, refs[d].index) is False


@squin.kernel
def backwards(qs: ilist.IList[Qubit, Literal[3]]):
    for i in range(-1, 2):
        squin.x(qs[i])
    for k in range(3):
        squin.x(qs[k + 1])


def test_a_loop_index_that_leaves_the_register_is_unknown():
    """A negative loop index counts from the end, so the analysis cannot name it."""
    mt, refs, _ = refs_of(backwards)
    negative, past = (refs[ops[0]] for ops in gate_operands(mt, "x"))
    assert reason(negative) == "an index that can be negative"
    assert reason(past) == "an index that can lie past the end of the register"


@squin.kernel
def square(qs: ilist.IList[Qubit, Literal[9]], k: int) -> Qubit:
    return qs[k * k]


@squin.kernel
def shifted(qs: ilist.IList[Qubit, Literal[9]], k: int) -> Qubit:
    return qs[2 * k + 1]


@squin.kernel
def calls_with_indices(n: int):
    qs = squin.qalloc(9)
    squin.x(shifted(qs, 3))
    squin.x(shifted(qs, n))
    squin.x(square(qs, 2))
    for i in range(4):
        squin.x(shifted(qs, i))


def test_a_call_substitutes_the_arguments_into_the_index():
    mt, refs, _ = refs_of(calls_with_indices)
    n = mt.callable_region.blocks[0].args[1]
    (loop,) = (s for s in mt.callable_region.walk() if isinstance(s, scf.For))
    constant, symbolic, computed, looped = (
        refs[ops[0]] for ops in gate_operands(mt, "x")
    )
    assert isinstance(constant, Slot) and constant.index == Affine(7)
    assert isinstance(symbolic, Slot) and symbolic.index == Affine.of(n).scale(2) + 1
    assert reason(computed) == "an item at an index that the callee computes"
    i = Affine.of(Iter(loop, range(4)))
    assert isinstance(looped, Slot) and looped.index == i.scale(2) + 1


def test_a_place_that_leaves_its_loop_is_unknown():
    mt, refs, analysis = refs_of(ladder)
    first = next(s for s in mt.callable_region.walk() if isinstance(s, scf.For))
    qs = mt.callable_region.blocks[0].args[1]
    inside = Slot(qs, Affine.of(Iter(first, range(5))) + 1)
    assert reason(analysis.leave(inside, first)) == OUT_OF_SCOPE
    outside = Slot(qs, 2)
    assert analysis.leave(outside, first) == outside


@squin.kernel
def nested_slices(qs: ilist.IList[Qubit, Literal[4]]):
    for i in range(1, 3):
        squin.broadcast.h(qs[0:i][1:])


def test_slice_bounds_with_the_bottom_type_keep_their_values():
    # Type inference leaves the bounds in this loop body with the type `Bottom`.
    mt, refs, _ = refs_of(nested_slices)
    loop = next(node for node in mt.callable_region.walk() if isinstance(node, scf.For))
    i = Affine.of(Iter(loop, range(1, 3)))
    (operand,) = gate_operands(mt, "h")[0]
    assert refs[operand] == Part(operand_root(mt), (Slice(Affine(1), 1, i - 1),))


def operand_root(mt):
    return mt.callable_region.blocks[0].args[1]


def test_flat_gives_a_whole_register_all_of_its_qubits():
    from bloqade.analysis.reference import Register
    from bloqade.analysis.reference.queries import flat

    mt, _, _ = refs_of(nested_slices)
    assert flat(Register(operand_root(mt))) == Slice(Affine(0), 1, Affine(4))
