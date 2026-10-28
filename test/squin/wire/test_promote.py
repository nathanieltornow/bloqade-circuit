"""Test that promotion turns lowered kernels into wire squin with the same meaning."""

import pytest
from kirin import ir
from kirin.dialects import func
from test_roundtrip import CASES, calls, simulate, cancelling

from bloqade.qubit import stmts as qubit_stmts
from bloqade.squin.wire import stmts as wire_stmts, kernel as wire_kernel
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import WireValidation


def promoted(kernel: ir.Method) -> ir.Method:
    """Return the wire squin method of `kernel` from lowering and promotion."""
    return SquinToWire().emit(kernel)


def survivors(mt: ir.Method) -> int:
    """Return the count of qubit reference accesses left in `mt` and its callees."""
    kinds = (
        wire_stmts.Unwrap,
        wire_stmts.Wrap,
        wire_stmts.UnwrapRegister,
        wire_stmts.WrapRegister,
    )
    count, seen, todo = 0, set(), [mt]
    while todo:
        method = todo.pop()
        if method in seen:
            continue
        seen.add(method)
        for node in method.callable_region.walk():
            count += isinstance(node, kinds)
            if isinstance(node, func.Invoke) and node.callee.dialects is wire_kernel:
                todo.append(node.callee)
    return count


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_promoted_kernel_passes_the_validation_suite(kernel, args, expected):
    WireValidation().check(promoted(kernel))


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_a_promoted_kernel_gives_the_same_result_after_the_way_back(
    kernel, args, expected
):
    assert simulate(WireToSquin().emit(promoted(kernel)), *args) == expected


@pytest.mark.parametrize("kernel, args, expected", [*CASES, (calls, (), None)])
def test_a_kernel_promotes_completely(kernel, args, expected):
    mt = promoted(kernel)
    assert survivors(mt) == 0
    assert not any(isinstance(n, qubit_stmts.New) for n in mt.callable_region.walk())


def test_a_borrowing_call_gives_the_same_result_after_promotion():
    assert simulate(WireToSquin().emit(promoted(calls))) == [[1, 0, 1, 1, 1], 4]


def test_a_register_with_constant_indices_stays_whole():
    wired = promoted(cancelling)
    statements = tuple(wired.callable_region.walk())
    assert sum(isinstance(node, wire_stmts.AllocRegister) for node in statements) == 1
    assert not any(isinstance(node, wire_stmts.Alloc) for node in statements)
    assert any(isinstance(node, wire_stmts.Borrow) for node in statements)
    assert sum(isinstance(node, wire_stmts.FreeRegister) for node in statements) == 1
