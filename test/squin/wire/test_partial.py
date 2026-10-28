"""Test partial promotion of squin kernels to wire squin.

A place is what the reference analysis names for a qubit value, such as item 2 of
a register. If the analysis can't name the place of a qubit, or can't prove two
places of one gate apart, the gate keeps reference semantics. Partial promotion
turns the rest of the kernel into wires.
"""

from typing import Any

import pytest
from kirin import ir
from kirin.dialects import ilist
from test_roundtrip import simulate

from bloqade import squin
from bloqade.types import Qubit, MeasurementResult
from bloqade.squin.wire import WireConversionError, stmts as wire_stmts
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import WireValidation


@squin.kernel
def opaque_partner(qs: ilist.IList[Qubit, Any], j: int):
    squin.cx(qs[0], qs[j])


@squin.kernel
def maybe_the_same_qubit(j: int) -> ilist.IList[MeasurementResult, Any]:
    qs = squin.qalloc(3)
    squin.x(qs[0])
    opaque_partner(qs, j)
    return squin.broadcast.measure(qs)


@squin.kernel
def two_registers(flag: bool) -> ilist.IList[MeasurementResult, Any]:
    a = squin.qalloc(1)
    b = squin.qalloc(1)
    q = a[0]
    if flag:
        q = b[0]
    squin.x(q)
    return squin.broadcast.measure([a[0], b[0]])


CASES = [
    (maybe_the_same_qubit, (2,), [1, 0, 1], "may name one qubit"),
    (two_registers, (True,), [0, 1], "has no place"),
    (two_registers, (False,), [1, 0], "has no place"),
]


def references(mt: ir.Method) -> int:
    """Return the count of reference accesses in `mt`, outside its callees."""
    kinds = (
        wire_stmts.Unwrap,
        wire_stmts.Wrap,
        wire_stmts.UnwrapRegister,
        wire_stmts.WrapRegister,
    )
    return sum(isinstance(node, kinds) for node in mt.callable_region.walk())


@pytest.mark.parametrize("kernel, args, expected, reason", CASES)
def test_the_strict_converter_refuses_the_kernel(kernel, args, expected, reason):
    with pytest.raises(WireConversionError, match=reason):
        SquinToWire().emit(kernel)


@pytest.mark.parametrize("kernel, args, expected, reason", CASES)
def test_partial_promotion_reports_the_reason(kernel, args, expected, reason):
    converter = SquinToWire(partial=True)
    converter.emit(kernel)
    assert any(reason in line for line in converter.report)


@pytest.mark.parametrize("kernel, args, expected, reason", CASES)
def test_a_partially_promoted_kernel_passes_the_validation_suite(
    kernel, args, expected, reason
):
    WireValidation().check(SquinToWire(partial=True).emit(kernel))


@pytest.mark.parametrize("kernel, args, expected, reason", CASES)
def test_a_partially_promoted_kernel_gives_the_same_result(
    kernel, args, expected, reason
):
    wired = SquinToWire(partial=True).emit(kernel)
    assert simulate(WireToSquin().emit(wired), *args) == expected


def test_the_promotable_rest_of_a_kernel_still_becomes_wires():
    wired = SquinToWire(partial=True).emit(maybe_the_same_qubit)
    names = {node.name for node in wired.callable_region.walk()}
    assert "alloc_register" in names and "x" in names
    assert references(wired) == 0
