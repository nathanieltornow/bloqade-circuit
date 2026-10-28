"""Test that promotion turns lowered kernels into wire squin with the same meaning."""

import pytest
from kirin import ir
from test_roundtrip import CASES, simulate

from bloqade.qubit import stmts as qubit_stmts
from bloqade.squin.wire import (
    WireToSquin,
    WireValidation,
    stmts as wire_stmts,
)
from bloqade.squin.wire.lower import lower, layout_of
from bloqade.squin.wire.promote import promote
from bloqade.squin.wire.pipeline import SquinToWire


def promoted(kernel: ir.Method) -> ir.Method:
    """Return the wire squin method of `kernel` from lowering and promotion."""
    converter = SquinToWire()
    converter.prepared, converter.refs, converter.effects = {}, {}, {}
    converter.prepare(kernel)
    copies = list(converter.prepared.values())
    layouts = {copy.code: layout_of(copy.code, converter.refs) for copy in copies}
    for copy in copies:
        lower(copy, converter.refs, converter.prepared)
    for copy in copies:
        promote(copy, layouts[copy.code])
    return converter.prepared[kernel]


def survivors(mt: ir.Method) -> int:
    """Return the count of `Unwrap` and `Wrap` statements left in `mt`."""
    kinds = (wire_stmts.Unwrap, wire_stmts.Wrap)
    return sum(isinstance(node, kinds) for node in mt.callable_region.walk())


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_promoted_kernel_passes_the_validation_suite(kernel, args, expected):
    WireValidation().check(promoted(kernel))


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_promoted_kernel_gives_the_same_result_after_the_way_back(
    kernel, args, expected
):
    assert simulate(WireToSquin().emit(promoted(kernel)), *args) == expected


FULLY_PROMOTED = [
    case
    for case in CASES
    if case[0].sym_name
    in {"measured_branch", "born_inside", "through_calls", "cancelling", "two_ancillas"}
]


@pytest.mark.parametrize("kernel, args, expected", FULLY_PROMOTED)
def test_a_kernel_of_single_qubits_promotes_completely(kernel, args, expected):
    mt = promoted(kernel)
    assert survivors(mt) == 0
    assert not any(isinstance(n, qubit_stmts.New) for n in mt.callable_region.walk())
