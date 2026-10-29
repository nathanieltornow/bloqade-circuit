"""Check complete conversion before independent wire ownership validation."""

import pytest
from test_rejections import opaque_partner
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.squin.emit.wire import EmitWire
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.passes.squin_to_wire import SquinToWire, remaining
from bloqade.squin.analysis.validation.wire import WireValidation, BorrowValidation


@squin.kernel
def uncertain_operands(index: int):
    qs = squin.qalloc(2)
    squin.cx(qs[0], qs[index])


@squin.kernel
def identical_operands():
    qs = squin.qalloc(2)
    squin.cx(qs[0], qs[0])


@pytest.mark.parametrize(
    "kernel, reason",
    [(uncertain_operands, "provably differ"), (identical_operands, "already borrowed")],
)
def test_conversion_builds_wire_operations_and_validation_rejects_invalid_borrows(
    kernel, reason
):
    before = kernel.code.print_str()
    method = kernel.similar()
    method = EmitWire(method.dialects, method, {}).emit_method().method
    assert not remaining(method)
    method.verify()
    method.verify_type()
    assert any(reason in str(error) for error in BorrowValidation().run(method)[1])
    with pytest.raises(ValidationErrorGroup, match=reason):
        WireValidation().check(method)
    with pytest.raises(ValidationErrorGroup, match=reason):
        SquinToWire().emit(kernel)
    assert kernel.code.print_str() == before


@squin.kernel
def constant_call_operand():
    qs = squin.qalloc(2)
    squin.x(qs[0])
    opaque_partner(qs, 1)
    return squin.broadcast.measure(qs)


def test_a_callee_failure_propagates_even_when_the_caller_supplies_a_constant():
    converter = SquinToWire()
    before = constant_call_operand.code.print_str()
    with pytest.raises(ValidationErrorGroup, match="provably differ"):
        converter.emit(constant_call_operand)
    assert not converter.contracts
    assert constant_call_operand.code.print_str() == before


@squin.kernel
def call_identical_operands():
    identical_operands()


@pytest.mark.parametrize(
    "failure",
    [
        WireConversionError("unsupported callee operation"),
        RuntimeError("compiler failure"),
        ValueError("invalid compiler state"),
    ],
)
def test_unexpected_callee_failures_do_not_request_inlining(monkeypatch, failure):
    from bloqade.squin.emit.wire import EmitWire

    run = EmitWire.emit_method
    attempts = []

    def fail_callee(emitter):
        attempts.append(emitter.source.sym_name)
        if emitter.source.sym_name == identical_operands.sym_name:
            raise failure
        return run(emitter)

    monkeypatch.setattr(EmitWire, "emit_method", fail_callee)
    with pytest.raises(type(failure)) as caught:
        SquinToWire().emit(call_identical_operands)
    assert caught.value is failure
    assert attempts == [identical_operands.sym_name]


def test_definite_callee_ownership_failure_stops_emission(monkeypatch):
    run = EmitWire.emit_method
    attempts = []

    def record(emitter):
        attempts.append(emitter.source.sym_name)
        return run(emitter)

    monkeypatch.setattr(EmitWire, "emit_method", record)
    with pytest.raises(ValidationErrorGroup, match="already borrowed"):
        SquinToWire().emit(call_identical_operands)
    assert attempts == [identical_operands.sym_name]


def test_callee_failure_keeps_the_original_condition_and_location():
    from test_rejections import maybe_the_same_qubit

    from bloqade.squin.wire.errors import UnresolvedBorrowSeparation

    with pytest.raises(ValidationErrorGroup) as caught:
        SquinToWire().emit(maybe_the_same_qubit)
    assert caught.value.__cause__ is None
    assert caught.value.errors
    assert all(
        isinstance(error, UnresolvedBorrowSeparation) for error in caught.value.errors
    )
    for error in caught.value.errors:
        assert error.node.source is not None
        assert error.source is not None
        assert error.source.lineno > 0
        assert "provably differ" in str(error)


def test_borrow_failure_points_to_the_source_call_and_preserves_source_metadata():
    import inspect
    from pathlib import Path

    source = [(node, repr(node.source)) for node in identical_operands.code.walk()]
    lines, start = inspect.getsourcelines(identical_operands.py_func)
    expected_line = start + next(
        i for i, line in enumerate(lines) if "squin.cx" in line
    )
    for _ in range(2):
        with pytest.raises(ValidationErrorGroup) as caught:
            SquinToWire().emit(identical_operands)
        errors = [
            error for error in caught.value.errors if "already borrowed" in str(error)
        ]
        assert errors
        for error in errors:
            assert error.source.file == str(Path(__file__).resolve())
            assert error.source.lineno + error.source.lineno_begin == expected_line
        assert [(node, repr(node.source)) for node, _ in source] == source
