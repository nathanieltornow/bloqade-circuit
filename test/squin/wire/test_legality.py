"""Check that full conversion rejects every surviving reference operation."""

import pytest
from kirin import ir, types
from test_partial import opaque_partner
from kirin.dialects import py, func, ilist
from test_roundtrip import simulate

from bloqade import squin
from bloqade.types import QubitType
from bloqade.squin.wire import WireConversionError
from bloqade.squin.noise import stmts as noise_stmts
from bloqade.squin.wire.twins import TWINS, H
from bloqade.squin.passes.wire_target import ConversionTarget
from bloqade.squin.passes.squin_to_wire import SquinToWire, remaining
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import StructureValidation


@squin.kernel
def correlated():
    qs = [squin.qubit.new(), squin.qubit.new()]
    squin.broadcast.x(qs)
    squin.broadcast.correlated_qubit_loss(0.0, [qs])
    return squin.broadcast.measure(qs)


@squin.kernel
def calls_correlated():
    return correlated()


@squin.kernel
def conditional_correlated(flag: bool):
    if flag:
        correlated()


@pytest.mark.parametrize(
    "kernel", [correlated, calls_correlated, conditional_correlated]
)
def test_full_conversion_rejects_an_operation_without_a_lowering(kernel):
    with pytest.raises(WireConversionError, match="correlated_qubit_loss"):
        SquinToWire().emit(kernel)


def test_partial_conversion_reports_and_preserves_an_unsupported_operation():
    converter = SquinToWire(partial=True)
    wired = converter.emit(correlated)
    assert any("correlated_qubit_loss" in line for line in converter.report)
    assert "correlated_qubit_loss" in remaining(wired)
    assert any(
        isinstance(node, noise_stmts.CorrelatedQubitLoss)
        for node in wired.callable_region.walk()
    )
    assert simulate(WireToSquin().emit(wired)) == simulate(correlated) == [1, 1]


def test_partial_conversion_still_checks_wire_operand_types():
    wired = SquinToWire(partial=True).emit(opaque_partner)
    assert any(type(node) in TWINS for node in wired.callable_region.walk())
    assert StructureValidation().run(wired)[1] == []
    block = wired.callable_region.blocks[0]
    head = block.first_stmt
    number = py.Constant(1)
    number.insert_before(head)
    H(qubit=number.result).insert_before(head)
    assert StructureValidation().run(wired)[1]


@squin.kernel
def unsupported_only():
    qs = [squin.qubit.new(), squin.qubit.new()]
    squin.broadcast.correlated_qubit_loss(0.0, [qs])


def test_an_unsupported_operation_requires_no_bridge_to_fail_conversion():
    with pytest.raises(WireConversionError, match="correlated_qubit_loss"):
        SquinToWire().emit(unsupported_only)


@pytest.mark.parametrize(
    "kind",
    [
        QubitType,
        ilist.IListType[QubitType, types.Any],
        types.Generic(tuple, types.Int, QubitType),
        types.Generic(tuple, types.Vararg(QubitType)),
        types.Union(QubitType, types.NoneType),
        types.TypeVar("Q", bound=types.Union(QubitType, types.NoneType)),
    ],
)
def test_the_target_checks_unused_parameters_that_can_hold_references(kind):
    block = ir.Block(argtypes=(types.Any, kind))
    none = py.Constant(None)
    block.stmts.extend([none, func.Return(none.result)])
    code = func.Function(
        sym_name="unused",
        body=ir.Region(block),
        signature=func.Signature(inputs=(kind,), output=types.NoneType),
    )
    method = ir.Method(dialects=squin.kernel, code=code)
    assert "block argument" in ConversionTarget.check(method)
