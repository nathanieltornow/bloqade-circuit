"""Check that unsupported references cause conversion to fail without changing the source."""

from typing import Any

import pytest
from kirin.dialects import ilist

from bloqade import squin
from bloqade.types import Qubit, MeasurementResult
from bloqade.squin.wire import WireConversionError
from bloqade.squin.passes.squin_to_wire import SquinToWire


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


@pytest.mark.parametrize(
    "kernel, reason",
    [(maybe_the_same_qubit, "may name one qubit"), (two_registers, "has no place")],
)
def test_unresolved_references_fail_conversion_and_preserve_the_source(kernel, reason):
    before = kernel.code.print_str()
    helper_before = opaque_partner.code.print_str()
    with pytest.raises(WireConversionError, match=reason):
        SquinToWire().emit(kernel)
    assert kernel.code.print_str() == before
    assert opaque_partner.code.print_str() == helper_before
