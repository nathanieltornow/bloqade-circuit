"""Test that squin kernels survive the trip to wire squin and back."""

import math
from typing import Any

import pytest
from kirin import ir, types, interp
from kirin.dialects import scf, func, ilist

from bloqade import squin
from bloqade.types import Qubit, MeasurementResult
from bloqade.pyqrack import DynamicMemorySimulator
from bloqade.squin.wire import (
    WireType,
    SquinToWire,
    WireToSquin,
    WireValidation,
    stmts as wire_stmts,
)
from bloqade.squin.wire.rewrite import Simplify, ReuseQubits


def plain(value):
    """Return `value` as plain Python, with each measurement result as an integer."""
    if isinstance(value, (ilist.IList, list, tuple)):
        items = value.data if isinstance(value, ilist.IList) else value
        return [plain(v) for v in items]
    if isinstance(value, float):
        return value
    return int(value)


def simulate(kernel: ir.Method, *args):
    """Run `kernel` once on the pyqrack simulator and return its result as plain Python."""
    return plain(DynamicMemorySimulator().run(kernel, args=args))


def trip(kernel: ir.Method, *, optimize: bool) -> ir.Method:
    """Return the squin kernel that comes back from wire squin.

    If `optimize` is True, the wire passes run before the trip back.
    """
    wired = SquinToWire().emit(kernel)
    if optimize:
        Simplify(wired.dialects)(wired)
        ReuseQubits(wired.dialects)(wired)
        WireValidation().check(wired)
    return WireToSquin().emit(wired)


@squin.kernel
def flip(q: Qubit) -> Qubit:
    squin.x(q)
    return q


@squin.kernel
def ladder() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(4)
    squin.x(qs[0])
    for i in range(3):
        squin.cx(qs[i], qs[i + 1])
    return squin.broadcast.measure(qs)


@squin.kernel
def measured_branch() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(3)
    squin.x(qs[0])
    if squin.measure(qs[0]):
        squin.x(qs[2])
    return squin.broadcast.measure(qs)


@squin.kernel
def born_inside(n: int) -> int:
    q = squin.qubit.new()
    total = 0
    for _ in range(n):
        t = squin.qubit.new()
        squin.x(t)
        squin.cx(t, q)
        total = total + 1
    if squin.measure(q):
        total = total + 10
    return total


@squin.kernel
def nested(n: int, flag: bool) -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(3)
    q = squin.qubit.new()
    for _ in range(n):
        for j in range(2):
            if flag:
                squin.x(qs[j])
            else:
                squin.x(q)
        squin.x(qs[2])
    return squin.broadcast.measure([qs[0], qs[1], qs[2], q])


@squin.kernel
def through_calls() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(4)
    flip(qs[1])
    squin.broadcast.x(qs[2:4])
    flip(qs[2:4][1])
    return squin.broadcast.measure(qs)


@squin.kernel
def cancelling() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(2)
    squin.h(qs[0])
    squin.x(qs[1])
    squin.h(qs[0])
    squin.rx(0.25 * math.pi, qs[1])
    squin.rx(0.75 * math.pi, qs[1])
    squin.s(qs[0])
    squin.s_adj(qs[0])
    return squin.broadcast.measure(qs)


@squin.kernel
def two_ancillas() -> ilist.IList[MeasurementResult, Any]:
    a = squin.qubit.new()
    squin.x(a)
    first = squin.measure(a)
    b = squin.qubit.new()
    second = squin.measure(b)
    return [first, second]


@squin.kernel
def loop_over_qubits() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(3)
    for q in qs:
        squin.x(q)
    return squin.broadcast.measure(qs)


@squin.kernel
def picked_in_a_branch(flag: bool) -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(2)
    q = qs[0]
    if flag:
        q = qs[1]
    squin.x(q)
    return squin.broadcast.measure(qs)


@squin.kernel
def carried_by_a_loop(n: int) -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(2)
    q = qs[0]
    for _ in range(n):
        squin.x(q)
        q = qs[1]
    return squin.broadcast.measure(qs)


@squin.kernel
def concatenated() -> ilist.IList[MeasurementResult, Any]:
    a = squin.qalloc(2)
    b = squin.qalloc(2)
    squin.broadcast.x(a + b)
    return squin.broadcast.measure([a[0], b[1]])


@squin.kernel
def for_each_qubit() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(2)

    def flip_one(q: Qubit):
        squin.x(q)

    ilist.for_each(flip_one, qs)
    return squin.broadcast.measure(qs)


@squin.kernel
def sliced_in_a_loop() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(4)
    for i in range(1, 3):
        squin.x(qs[0:3][i])
    return squin.broadcast.measure(qs)


@squin.kernel
def growing_slice() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(4)
    for i in range(1, 4):
        squin.x(qs[0:i][i - 1])
    return squin.broadcast.measure(qs)


CASES = [
    (ladder, (), [1, 1, 1, 1]),
    (measured_branch, (), [1, 0, 1]),
    (born_inside, (2,), 2),
    (born_inside, (3,), 13),
    (nested, (1, True), [1, 1, 1, 0]),
    (nested, (3, False), [0, 0, 1, 0]),
    (through_calls, (), [0, 1, 1, 0]),
    (cancelling, (), [0, 0]),
    (two_ancillas, (), [1, 0]),
    (loop_over_qubits, (), [1, 1, 1]),
    (picked_in_a_branch, (True,), [0, 1]),
    (picked_in_a_branch, (False,), [1, 0]),
    (carried_by_a_loop, (1,), [1, 0]),
    (carried_by_a_loop, (3,), [1, 0]),
    (concatenated, (), [1, 1]),
    (for_each_qubit, (), [1, 1]),
    (sliced_in_a_loop, (), [0, 1, 1, 0]),
    (growing_slice, (), [1, 1, 1, 0]),
]


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_kernel_gives_the_same_result_after_the_trip(
    kernel, args, expected, optimize
):
    assert simulate(kernel, *args) == expected
    assert simulate(trip(kernel, optimize=optimize), *args) == expected


def test_the_conversion_leaves_the_kernel_as_it_is():
    before = ladder.code.print_str()
    SquinToWire().emit(ladder)
    assert ladder.code.print_str() == before


def test_the_reuse_pass_turns_a_second_allocation_into_a_reset():
    wired = SquinToWire().emit(two_ancillas)
    ReuseQubits(wired.dialects)(wired)
    allocs = [
        s for s in wired.callable_region.walk() if isinstance(s, wire_stmts.Alloc)
    ]
    assert len(allocs) == 1


@squin.kernel
def opaque_partner(qs: ilist.IList[Qubit, Any], j: int):
    squin.cx(qs[0], qs[j])


@squin.kernel
def maybe_the_same_qubit(j: int) -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(3)
    squin.x(qs[0])
    opaque_partner(qs, j)
    return squin.broadcast.measure(qs)


def test_a_gate_on_two_places_that_may_alias_is_refused():
    with pytest.raises(interp.InterpreterError, match="may name one qubit"):
        SquinToWire().emit(maybe_the_same_qubit)


def test_assume_disjoint_trusts_two_places_that_may_alias():
    wired = SquinToWire(assume_disjoint=True).emit(maybe_the_same_qubit)
    assert simulate(WireToSquin().emit(wired), 2) == [1, 0, 1]


@squin.kernel
def kick(q: Qubit, k: int) -> int:
    for _ in range(k):
        squin.x(q)
    return k + 1


@squin.kernel
def fresh_pair() -> tuple[Qubit, Qubit]:
    a = squin.qubit.new()
    b = squin.qubit.new()
    squin.x(b)
    return a, b


@squin.kernel
def flip_all(qs: ilist.IList[Qubit, Any]):
    for i in range(len(qs)):
        squin.x(qs[i])


@squin.kernel
def calls() -> tuple[ilist.IList[MeasurementResult, Any], int]:
    qs = squin.qalloc(3)
    pair = fresh_pair()
    a = pair[0]
    b = pair[1]
    total = kick(qs[1], 3)
    flip_all(qs)
    squin.cx(b, a)
    return squin.broadcast.measure([qs[0], qs[1], qs[2], a, b]), total


def test_a_call_with_qubit_arguments_stays_a_borrowing_call():
    wired = SquinToWire().emit(calls)
    callees = [
        node.callee
        for node in wired.callable_region.walk()
        if isinstance(node, func.Invoke)
    ]
    assert sorted(c.sym_name for c in callees) == ["flip_all", "fresh_pair", "kick"]
    kick_signature = next(c for c in callees if c.sym_name == "kick").code.signature
    assert kick_signature.inputs == (WireType, types.Int)
    assert kick_signature.output == types.Generic(tuple, types.Int, WireType)


def test_a_borrowing_call_gives_the_same_result_after_the_trip():
    expected = [[1, 0, 1, 1, 1], 4]
    assert simulate(calls) == expected
    assert simulate(trip(calls, optimize=False)) == expected
    assert simulate(trip(calls, optimize=True)) == expected


@squin.kernel
def mapped() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(2)

    def measure_one(q: Qubit) -> MeasurementResult:
        return squin.measure(q)

    return ilist.map(measure_one, qs)


def test_a_map_over_qubits_becomes_a_loop():
    wired = SquinToWire().emit(mapped)
    assert not any(isinstance(n, ilist.Map) for n in wired.callable_region.walk())
    assert any(isinstance(n, scf.For) for n in wired.callable_region.walk())
    assert simulate(WireToSquin().emit(wired)) == [0, 0]
