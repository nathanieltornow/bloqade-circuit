"""Test the rewrite rules and the linearity validation of wire squin."""

from typing import Any

import pytest
from kirin import ir
from kirin.dialects import scf, ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.types import MeasurementResult
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.wire import (
    TWINS,
    SquinToWire,
    LinearityValidation,
    stmts as wire_stmts,
)
from bloqade.squin.noise import stmts as noise_stmts
from bloqade.squin.wire.rewrite import Simplify


def count(mt: ir.Method, kind: type[ir.Statement]) -> int:
    """Return the count of statements of the class `kind` in `mt`, bodies included."""
    return sum(isinstance(node, kind) for node in mt.callable_region.walk())


def simplified(kernel: ir.Method) -> ir.Method:
    """Return the wire squin method of `kernel` after the pass `Simplify`."""
    wired = SquinToWire().emit(kernel)
    Simplify(wired.dialects)(wired)
    LinearityValidation().check(wired)
    return wired


@squin.kernel
def h_twice() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(2)
    squin.h(qs[0])
    squin.x(qs[1])
    squin.h(qs[0])
    return squin.broadcast.measure(qs)


def test_two_hadamards_on_one_register_qubit_cancel():
    wired = simplified(h_twice)
    assert count(wired, TWINS[gate_stmts.H]) == 0
    assert count(wired, TWINS[gate_stmts.X]) == 1


def test_the_register_round_trips_between_gates_disappear():
    wired = simplified(h_twice)
    # One borrow and one release per qubit remain, around the measurements.
    assert count(wired, wire_stmts.Borrow) == 2
    assert count(wired, wire_stmts.Release) == 2


@squin.kernel
def rotations() -> MeasurementResult:
    q = squin.qubit.new()
    squin.rz(0.25, q)
    squin.rz(0.5, q)
    squin.t(q)
    squin.t_adj(q)
    return squin.measure(q)


def test_two_rotations_merge_and_an_adjoint_pair_cancels():
    wired = simplified(rotations)
    assert count(wired, TWINS[gate_stmts.Rz]) == 1
    assert count(wired, TWINS[gate_stmts.T]) == 0


@squin.kernel
def noisy() -> MeasurementResult:
    q = squin.qubit.new()
    squin.h(q)
    squin.depolarize(0.1, q)
    squin.h(q)
    return squin.measure(q)


def test_a_noise_channel_keeps_two_gates_apart():
    wired = simplified(noisy)
    assert count(wired, TWINS[gate_stmts.H]) == 2
    assert count(wired, TWINS[noise_stmts.Depolarize]) == 1


@squin.kernel
def h_in_a_branch(flag: bool) -> MeasurementResult:
    q = squin.qubit.new()
    squin.h(q)
    if flag:
        squin.h(q)
    return squin.measure(q)


def test_a_gate_in_a_branch_keeps_its_partner_outside():
    wired = simplified(h_in_a_branch)
    assert count(wired, TWINS[gate_stmts.H]) == 2


@squin.kernel
def ladder() -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(3)
    for i in range(2):
        squin.cx(qs[i], qs[i + 1])
    return squin.broadcast.measure(qs)


def test_a_wire_with_two_uses_breaks_linearity():
    wired = SquinToWire().emit(ladder)
    alloc = next(
        node
        for node in wired.callable_region.walk()
        if isinstance(node, wire_stmts.AllocRegister)
    )
    wire_stmts.FreeRegister(alloc.result).insert_after(alloc)
    with pytest.raises(ValidationErrorGroup, match="linearity"):
        LinearityValidation().check(wired)


def test_a_loop_body_that_reads_an_outer_wire_breaks_linearity():
    wired = SquinToWire().emit(ladder)
    loop = next(n for n in wired.callable_region.walk() if isinstance(n, scf.For))
    carried = loop.body.blocks[0].args[1]
    carried.replace_by(loop.initializers[0])
    _, errors = LinearityValidation().run(wired)
    assert any("crosses into the body" in str(error) for error in errors)
