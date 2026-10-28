"""Test that the lowering rules keep the meaning of a squin kernel."""

import pytest
from kirin import ir
from test_roundtrip import CASES, simulate

from bloqade.squin.wire import stmts as wire_stmts
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import ReferenceValidation


def lowered(kernel: ir.Method) -> ir.Method:
    """Return the prepared copy of `kernel` after the lowering rules."""
    return SquinToWire().lower(kernel)


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_lowered_kernel_passes_the_reference_check(kernel, args, expected):
    assert ReferenceValidation().run(lowered(kernel))[1] == []


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_lowered_kernel_gives_the_same_result_after_the_way_back(
    kernel, args, expected
):
    back = WireToSquin().emit(lowered(kernel))
    assert simulate(back, *args) == expected


def test_lowering_leaves_no_squin_gate():
    copy = lowered(CASES[0][0])
    names = {node.name for node in copy.callable_region.walk()}
    assert "cx" in names and "x" in names
    assert any(isinstance(n, wire_stmts.Unwrap) for n in copy.callable_region.walk())
    assert not any(
        type(n).__module__.startswith("bloqade.squin.gate")
        for n in copy.callable_region.walk()
    )
