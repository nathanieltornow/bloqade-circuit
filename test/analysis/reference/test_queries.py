"""Test the place queries on squin kernels."""

from typing import Literal

from kirin.dialects import scf, func, ilist

from bloqade import squin
from bloqade.types import Qubit
from bloqade.analysis.reference import (
    Iter,
    Part,
    Slot,
    Slice,
    Whole,
    Affine,
    Unknown,
    Register,
    family,
    leaves,
    disjoint,
    must_alias,
    disjoint_across_iterations,
)
from bloqade.squin.analysis.reference import (
    QubitReferenceAnalysis,
    qubit_register_length,
)


@squin.kernel
def two_lists(
    qs: ilist.IList[Qubit, Literal[6]], ps: ilist.IList[Qubit, Literal[2]]
) -> None:
    for i in range(3):
        squin.cx(qs[2 * i], qs[2 * i + 1])
    for _ in range(2):
        t = squin.qubit.new()
        squin.x(t)


def parts():
    mt = two_lists.similar()
    frame, _ = QubitReferenceAnalysis(mt.dialects).run(mt)
    qs, ps = mt.callable_region.blocks[0].args[1:]
    first, _ = (s for s in mt.callable_region.walk() if isinstance(s, scf.For))
    i = Affine.of(Iter(first, range(3)))
    (new,) = [
        s
        for s in mt.callable_region.walk()
        if isinstance(s, func.Invoke) and s.callee.sym_name == "new"
    ]
    return frame.entries, qs, ps, i, frame.entries[new.result].root


def test_disjoint_proves_that_places_share_no_qubit():
    _, qs, ps, i, fresh = parts()
    assert disjoint(Slot(qs, i.scale(2)), Slot(qs, i.scale(2) + 1))
    assert disjoint(Slot(qs, i), Slot(qs, i + 1))
    assert not disjoint(Slot(qs, i), Slot(qs, 1))
    middle = Part(qs, (Slice(Affine(1), 1, Affine(2)),))
    assert disjoint(Slot(qs, 0), middle) and disjoint(Slot(qs, i + 3), middle)
    assert not disjoint(Slot(qs, 2), middle)
    assert not disjoint(Register(qs), Slot(qs, 0))
    assert not disjoint(Register(qs), Register(ps))
    assert disjoint(Whole(fresh), Register(qs))


def test_must_alias_needs_one_root_and_equal_paths():
    _, qs, _, i, fresh = parts()
    assert must_alias(Slot(qs, i + 1), Slot(qs, i + 1))
    assert not must_alias(Slot(qs, i), Slot(qs, i + 1))
    assert not must_alias(Register(qs), Slot(qs, 0))
    middle = Part(qs, (Slice(Affine(1), 1, Affine(2)),))
    assert must_alias(middle, Part(qs, (Slice(Affine(1), 1, Affine(2)),)))
    assert family(fresh) and not family(qs)
    assert not must_alias(Whole(fresh), Whole(fresh))


@squin.kernel
def chained(qs: ilist.IList[Qubit, Literal[10]], n: int):
    for i in range(3):
        squin.cx(qs[i], qs[i + 1])
        squin.cx(qs[2 * i], qs[2 * i + 1])
        squin.cx(qs[n], qs[n + 1])
        k = i * i
        squin.cx(qs[k], qs[k + 1])
    squin.cx(qs[0], qs[1])
    squin.cx(qs[0], qs[0])


def operand_pairs():
    """Return the loop of `chained` and the places of the operands of each `cx`."""
    mt = chained.similar()
    frame, _ = QubitReferenceAnalysis(mt.dialects).run(mt)
    (loop,) = (s for s in mt.callable_region.walk() if isinstance(s, scf.For))
    pairs = [
        (frame.entries[s.args[0]], frame.entries[s.args[1]])
        for s in mt.callable_region.walk()
        if isinstance(s, func.Invoke) and s.callee.sym_name == "cx"
    ]
    return loop, pairs


def test_disjoint_across_iterations_compares_two_iterations():
    loop, (shifted, even_odd, outer, squared, fixed, twice) = operand_pairs()
    # `qs[i + 1]` of iteration 0 is `qs[i]` of iteration 1.
    assert disjoint(*shifted)
    assert not disjoint_across_iterations(shifted[0], shifted[1], loop)
    assert disjoint_across_iterations(even_odd[0], even_odd[1], loop)
    # `n` comes from outside the loop, so it has one value in all iterations.
    assert disjoint_across_iterations(outer[0], outer[1], loop)
    # `k` is a value of the loop body, so it changes with the iteration.
    assert disjoint(*squared)
    assert not disjoint_across_iterations(squared[0], squared[1], loop)


def test_disjoint_across_iterations_without_loop_symbols_is_disjoint():
    loop, (*_, fixed, twice) = operand_pairs()
    assert disjoint(*fixed)
    assert disjoint_across_iterations(fixed[0], fixed[1], loop)
    assert not disjoint(*twice)
    assert not disjoint_across_iterations(twice[0], twice[1], loop)
    _, qs, ps, _, fresh = parts()
    middle = Part(qs, (Slice(Affine(1), 1, Affine(2)),))
    for a, b in [
        (Slot(qs, 0), middle),
        (Slot(qs, 2), middle),
        (Whole(fresh), Register(qs)),
        (Register(qs), Register(ps)),
    ]:
        assert disjoint_across_iterations(a, b, loop) == disjoint(a, b)


def test_leaves_lists_the_qubits_of_a_reference():
    _, qs, _, _, _ = parts()
    middle = Part(qs, (Slice(Affine(1), 2, Affine(2)),))
    assert leaves(middle, qubit_register_length) == (Slot(qs, 1), Slot(qs, 3))
    assert leaves(Register(qs), qubit_register_length) == tuple(
        Slot(qs, k) for k in range(6)
    )


@squin.kernel
def new_and_passed():
    qs = squin.qalloc(2)

    def pair(q: Qubit) -> ilist.IList[Qubit, Literal[2]]:
        return [q, squin.qubit.new()]

    rs = ilist.map(pair, qs)
    squin.cx(rs[0][0], qs[0])


def test_a_map_that_returns_new_and_passed_qubits_is_unknown():
    mt = new_and_passed.similar()
    frame, _ = QubitReferenceAnalysis(mt.dialects).run(mt)
    ((control, target),) = [
        node.inputs
        for node in mt.callable_region.walk()
        if isinstance(node, func.Invoke) and node.callee.sym_name == "cx"
    ]
    assert isinstance(frame.entries[control], Unknown)
    assert not disjoint(frame.entries[control], frame.entries[target])
