"""Test the place queries and the per-qubit analyses on squin kernels."""

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
)
from bloqade.squin.analysis.per_qubit import (
    Qubit as QubitKey,
    MeasurementCount,
    circuit_depth,
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


def test_leaves_lists_the_qubits_of_a_reference():
    _, qs, _, _, _ = parts()
    middle = Part(qs, (Slice(Affine(1), 2, Affine(2)),))
    assert leaves(middle, qubit_register_length) == (Slot(qs, 1), Slot(qs, 3))
    assert leaves(Register(qs), qubit_register_length) == tuple(
        Slot(qs, k) for k in range(6)
    )


@squin.kernel
def measured_twice() -> None:
    qs = squin.qalloc(3)
    for _ in range(2):
        squin.measure(qs[0])
    squin.broadcast.measure(qs[1:3])
    for _ in range(2):
        t = squin.qubit.new()
        squin.measure(t)


def test_measurement_count_names_each_qubit_and_each_run():
    count = MeasurementCount()
    state = count.run(measured_twice)
    assert count.failure is None
    by_name = {repr(q): n for q, n in state.items()}
    assert by_name["Slot(%qs = qalloc()[0], 0)"] == 2
    assert by_name["Slot(%qs = qalloc()[0], 1)"] == 1
    assert by_name["Slot(%qs = qalloc()[0], 2)"] == 1
    runs = sorted(q.run for q in state if q.path == ())
    assert runs == [0, 1]
    assert all(isinstance(q, QubitKey) for q in state)


@squin.kernel
def repeated():
    q = squin.qubit.new()
    p = squin.qubit.new()
    for _ in range(3):
        squin.h(q)
        squin.cx(q, p)
    return squin.measure(p)


@squin.kernel
def at_runtime(n: int):
    q = squin.qubit.new()
    for _ in range(n):
        squin.h(q)


def test_depth_unrolls_constant_loops_and_stops_on_others():
    assert circuit_depth(repeated) == 7
    stopped = circuit_depth(at_runtime)
    assert isinstance(stopped, Unknown)
    assert stopped.reason == "a loop over a range that is not constant"


def test_a_kernel_without_operations_has_depth_zero():
    @squin.kernel
    def empty() -> int:
        return 1

    assert circuit_depth(empty) == 0
