"""Check conversion boundaries, failure isolation and bounded rewrite execution."""

import pytest
from kirin import ir
from kirin.dialects import py
from test_roundtrip import calls, ladder, simulate, maybe_the_same_qubit
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade import squin
from bloqade.types import Qubit
from bloqade.squin.wire import WireConversionError
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin


class AlwaysChanges(RewriteRule):
    """Report a rewrite on every visit to exercise the convergence limit."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Keep the driver running without changing the test statement."""
        return RewriteResult(has_done_something=True)


def test_rewrite_exhaustion_is_an_error():
    with pytest.raises(WireConversionError, match="within 2 iterations"):
        CheckedFixpoint(AlwaysChanges(), max_iter=2).rewrite(py.Constant(0))


def test_preparation_exhaustion_preserves_the_source():
    before = ladder.code.print_str()
    with pytest.raises(WireConversionError, match="did not converge"):
        SquinToWire(max_iterations=1).emit(ladder)
    assert ladder.code.print_str() == before


def test_one_converter_recovers_after_failure_and_produces_deterministic_output():
    converter = SquinToWire()
    before = maybe_the_same_qubit.code.print_str()
    with pytest.raises(WireConversionError, match="may name one qubit"):
        converter.emit(maybe_the_same_qubit)
    assert maybe_the_same_qubit.code.print_str() == before
    first = converter.emit(calls)
    second = converter.emit(calls)
    assert first.code.print_str() == second.code.print_str()
    assert simulate(WireToSquin().emit(second)) == [[1, 0, 1, 1, 1], 4]


def test_normalization_reaches_a_stable_form():
    method = calls.similar()
    SquinToWire.normalize(method)
    before = method.code.print_str()
    SquinToWire.normalize(method)
    assert method.code.print_str() == before
    method.verify()
    method.verify_type()


@squin.kernel
def recursive(q: Qubit, n: int):
    squin.x(q)
    if n > 0:
        recursive(q, n - 1)


def test_recursive_calls_fail_before_conversion():
    before = recursive.code.print_str()
    with pytest.raises(WireConversionError, match="recursive calls"):
        SquinToWire().emit(recursive)
    assert recursive.code.print_str() == before


@squin.kernel
def repeated_return():
    q = squin.qubit.new()
    return q, q


@squin.kernel
def repeated_register_return():
    qs = squin.qalloc(2)
    return qs, qs


@pytest.mark.parametrize("kernel", [repeated_return, repeated_register_return])
def test_duplicate_quantum_returns_fail_before_materialization(kernel):
    before = kernel.code.print_str()
    with pytest.raises(WireConversionError, match="cannot return one qubit twice"):
        SquinToWire().emit(kernel)
    assert kernel.code.print_str() == before


def test_missing_handles_cannot_authorize_a_gate():
    from kirin.dialects import ilist

    from bloqade.qubit.stmts import New
    from bloqade.squin.gate.stmts import CX
    from bloqade.squin.rewrite.wire.lower import LocalLowering
    from bloqade.squin.rewrite.wire.handles import Handles

    a, b = New(), New()
    controls, targets = ilist.New((a.result,)), ilist.New((b.result,))
    gate = CX(controls.result, targets.result)
    with pytest.raises(WireConversionError, match="has no place"):
        LocalLowering(gate, Handles()).gate()
