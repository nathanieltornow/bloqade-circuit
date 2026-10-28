"""Test `Unwrap`, `Wrap` and `Reference`, their way back, and the reference check."""

from kirin import ir
from kirin.rewrite import Walk, Fixpoint
from kirin.dialects import ilist

from bloqade import squin
from bloqade.types import QubitType
from bloqade.pyqrack import DynamicMemorySimulator
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.wire import TWINS, stmts as wire_stmts, kernel as wire_kernel
from bloqade.squin.passes.squin_to_wire import InlineCalls, is_library
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import ReferenceValidation


@squin.kernel
def flipped():
    q = squin.qubit.new()
    squin.x(q)
    return squin.measure(q)


def lowered(kernel: ir.Method) -> tuple[ir.Method, ir.Statement]:
    """Return a copy of `kernel` whose `x` gate reads its qubit through `Unwrap`.

    The second result is the `Wrap` that puts the qubit back.
    """
    mt = kernel.similar()
    Fixpoint(Walk(InlineCalls(lambda call: is_library(call.callee)))).rewrite(mt.code)
    mt.dialects = wire_kernel
    gate = next(s for s in mt.callable_region.walk() if isinstance(s, gate_stmts.X))
    (qubit,) = gate.qubits.owner.values
    take = wire_stmts.Unwrap(qubit)
    take.insert_before(gate)
    flip = TWINS[gate_stmts.X](qubit=take.wire)
    flip.insert_before(gate)
    put = wire_stmts.Wrap(qubit, flip.qubit_out)
    put.insert_before(gate)
    gate.delete()
    return mt, put


def errors(mt: ir.Method) -> list[str]:
    """Return the messages of the reference check on `mt`."""
    return [str(error) for error in ReferenceValidation().run(mt)[1]]


def test_a_balanced_unwrap_and_wrap_pass_the_check():
    mt, _ = lowered(flipped)
    assert errors(mt) == []


def test_unwrap_and_wrap_go_back_to_the_squin_kernel():
    mt, _ = lowered(flipped)
    back = WireToSquin().emit(mt)
    assert int(DynamicMemorySimulator().run(back)) == 1


def test_a_second_unwrap_of_a_qubit_that_is_out_is_an_error():
    mt, put = lowered(flipped)
    take = put.wire.owner.qubit.owner
    wire_stmts.Unwrap(put.qubit).insert_after(take)
    assert any("already out" in message for message in errors(mt))


def test_a_wrap_of_a_qubit_that_is_not_out_is_an_error():
    mt, put = lowered(flipped)
    second = wire_stmts.Wrap(put.qubit, put.wire)
    second.insert_after(put)
    assert any("not out" in message for message in errors(mt))


def test_a_squin_gate_on_a_qubit_that_is_out_is_an_error():
    mt, put = lowered(flipped)
    listed = ilist.New(values=(put.qubit,), elem_type=QubitType)
    listed.insert_before(put)
    gate_stmts.H(qubits=listed.result).insert_before(put)
    assert any("touches a qubit that is out" in message for message in errors(mt))


def moved_into_a_reference(mt: ir.Method, put: ir.Statement) -> ir.Statement:
    """Measure a new reference to a fresh qubit after `put`, and return its `Wrap`.

    A fresh qubit fills the new reference, as the result of a call does. The
    final measurement then reads the new reference.
    """
    take = wire_stmts.Alloc()
    take.insert_after(put)
    fresh = wire_stmts.Reference()
    fresh.insert_after(take)
    fill = wire_stmts.Wrap(fresh.result, take.result)
    fill.insert_after(fresh)
    measured = next(
        s for s in mt.callable_region.walk() if s.name == "measure" and s is not put
    )
    listed = ilist.New(values=(fresh.result,), elem_type=QubitType)
    listed.insert_before(measured)
    measured.qubits = listed.result
    return fill


def test_a_reference_takes_over_the_qubit_of_its_wrap():
    mt, put = lowered(flipped)
    moved_into_a_reference(mt, put)
    assert errors(mt) == []
    back = WireToSquin().emit(mt)
    assert int(DynamicMemorySimulator().run(back)) == 0


def test_a_reference_without_a_wrap_right_after_it_is_an_error():
    mt, put = lowered(flipped)
    fill = moved_into_a_reference(mt, put)
    fresh = fill.qubit.owner
    fresh.detach()
    fresh.insert_before(fill.wire.owner)
    # The reference now comes before the `Alloc`, so no `Wrap` follows it.
    assert any("right after it" in message for message in errors(mt))
