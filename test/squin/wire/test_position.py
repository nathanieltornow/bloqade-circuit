"""Test positional contracts that preserve linear use counts when they fail."""

import pytest
from kirin import ir
from kirin.dialects import py, scf, func
from test_validation import first, errors, ladder
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.types import Qubit
from bloqade.squin.wire import stmts, is_linear
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.analysis.validation.wire import (
    WireValidation,
    BorrowValidation,
    PositionValidation,
    LinearityValidation,
    StructureValidation,
)


@squin.kernel
def loop_pair(a: Qubit, n: int, b: Qubit) -> int:
    for _ in range(n):
        squin.x(a)
        squin.z(b)
    return n


@squin.kernel
def branch_pair(a: Qubit, flag: bool, b: Qubit):
    if flag:
        squin.x(a)
        squin.z(b)
    else:
        squin.z(a)
        squin.x(b)


@squin.kernel
def call_pair(n: int):
    qs = squin.qalloc(2)
    loop_pair(qs[0], n, qs[1])


@squin.kernel
def swap_pair(a: Qubit, b: Qubit):
    squin.swap(a, b)


def swap_yield(end: scf.Yield):
    """Exchange the first two linear values in a yield."""
    values = list(end.values)
    a, b = [k for k, value in enumerate(values) if is_linear(value.type)][:2]
    values[a], values[b] = values[b], values[a]
    replacement = scf.Yield(*values)
    replacement.insert_before(end)
    end.delete()


def swap_suffix(method: ir.Method):
    """Exchange the borrowed wires at the end of a function's return tuple."""
    returned = first(method, func.Return)
    packed = returned.value.owner
    assert isinstance(packed, py.tuple.New)
    values = list(packed.args)
    values[-2], values[-1] = values[-1], values[-2]
    packed.args = tuple(values)


def assert_only_positions_fail(method: ir.Method, match: str):
    """Require a positional error while structure, use counts and borrows remain valid."""
    for validation in (
        StructureValidation(),
        LinearityValidation(),
        BorrowValidation(),
    ):
        assert errors(validation, method) == []
    assert any(match in error for error in errors(PositionValidation(), method))
    with pytest.raises(ValidationErrorGroup, match="positions"):
        WireValidation().check(method)


@pytest.mark.parametrize("kernel", [loop_pair, branch_pair, call_pair, swap_pair])
def test_gates_and_control_flow_preserve_qubit_positions(kernel):
    WireValidation().check(SquinToWire(assume_disjoint=True).emit(kernel))


def test_a_loop_cannot_exchange_its_carried_qubits():
    wired = SquinToWire(assume_disjoint=True).emit(loop_pair)
    loop = first(wired, scf.For)
    swap_yield(loop.body.blocks[0].last_stmt)
    assert_only_positions_fail(wired, "loop input position")


@pytest.mark.parametrize("body", ["then_body", "else_body"])
def test_both_branches_must_continue_the_same_qubits(body):
    wired = SquinToWire(assume_disjoint=True).emit(branch_pair)
    branch = first(wired, scf.IfElse)
    swap_yield(getattr(branch, body).blocks[0].last_stmt)
    assert_only_positions_fail(wired, "both branches")


def test_a_function_cannot_exchange_its_borrowed_parameters():
    wired = SquinToWire(assume_disjoint=True).emit(loop_pair)
    swap_suffix(wired)
    assert_only_positions_fail(wired, "return suffix position")


def test_a_call_checks_the_return_suffix_inside_its_callee():
    wired = SquinToWire(assume_disjoint=True).emit(call_pair)
    call = first(wired, func.Invoke)
    swap_suffix(call.callee)
    assert_only_positions_fail(wired, "return suffix position")


def test_a_release_cannot_exchange_the_qubits_of_two_slots():
    wired = SquinToWire(assume_disjoint=True).emit(ladder)
    releases = [
        node for node in wired.callable_region.walk() if isinstance(node, stmts.Release)
    ]
    first_release, second_release = releases[:2]
    first_release.wire, second_release.wire = second_release.wire, first_release.wire
    assert_only_positions_fail(wired, "register slot")


def test_a_function_cannot_replace_a_borrowed_wire_with_a_fresh_wire():
    wired = SquinToWire(assume_disjoint=True).emit(swap_pair)
    returned = first(wired, func.Return)
    packed = returned.value.owner
    old = packed.args[-1]
    fresh = stmts.Alloc()
    fresh.insert_before(packed)
    stmts.Free(old).insert_before(packed)
    packed.args = (*packed.args[:-1], fresh.result)
    assert_only_positions_fail(wired, "return suffix position")


def test_a_function_must_return_every_borrowed_parameter():
    wired = SquinToWire(assume_disjoint=True).emit(swap_pair)
    returned = first(wired, func.Return)
    packed = returned.value.owner
    returned.value = packed.args[0]
    assert any(
        "every borrowed parameter" in error
        for error in errors(PositionValidation(), wired)
    )
