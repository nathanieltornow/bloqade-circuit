"""Test the rewrite rules of wire squin."""

from typing import Any

from kirin import ir
from kirin.dialects import ilist

from bloqade import squin
from bloqade.types import MeasurementResult
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.wire import TWINS, stmts as wire_stmts
from bloqade.squin.noise import stmts as noise_stmts
from bloqade.squin.passes.wire import Simplify
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.analysis.validation.wire import WireValidation


def count(mt: ir.Method, kind: type[ir.Statement]) -> int:
    """Return the count of statements of the class `kind` in `mt`, bodies included."""
    return sum(isinstance(node, kind) for node in mt.callable_region.walk())


def simplified(kernel: ir.Method) -> ir.Method:
    """Return the wire squin method of `kernel` after the pass `Simplify`."""
    wired = SquinToWire().emit(kernel)
    Simplify(wired.dialects)(wired)
    WireValidation().check(wired)
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


def test_a_register_with_constant_indices_splits_into_wires():
    wired = simplified(h_twice)
    assert count(wired, wire_stmts.AllocRegister) == 0
    assert count(wired, wire_stmts.Alloc) == 2
    assert count(wired, wire_stmts.Borrow) == 0


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
