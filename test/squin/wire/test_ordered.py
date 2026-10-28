"""Check ordered conversion, independent validation and partial reference lifetimes."""

from typing import Literal

import pytest
from test_partial import two_registers
from test_position import assert_only_positions_fail
from kirin.dialects import func, ilist
from test_roundtrip import CASES, calls, simulate

from bloqade import squin
from bloqade.jeff import WireToJeff
from bloqade.types import Qubit
from bloqade.squin.wire import WireConversionError, stmts
from bloqade.analysis.reference import ReferenceAnalysis
from bloqade.squin.passes.squin_to_wire import SquinToWire, remaining
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import WireValidation


@pytest.fixture
def forbid_reference_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Wire conversion must resolve definitions during rewriting."
        )

    monkeypatch.setattr(ReferenceAnalysis, "run", forbidden)


@pytest.mark.parametrize("kernel, args, expected", CASES)
def test_conversion_and_validation_need_no_reference_analysis(
    forbid_reference_analysis, kernel, args, expected
):
    wired = SquinToWire().emit(kernel)
    assert not remaining(wired)
    WireValidation().check(wired)
    assert simulate(WireToSquin().emit(wired), *args) == expected


def test_calls_and_direct_jeff_emission_need_no_reference_analysis(
    forbid_reference_analysis,
):
    converter = SquinToWire()
    wired = converter.emit(calls)
    assert len(converter.contracts) > 1
    assert simulate(WireToSquin().emit(wired)) == [[1, 0, 1, 1, 1], 4]
    emitted = WireToJeff().emit(wired)
    emitted.verify()
    emitted.verify_type()


@pytest.mark.parametrize("flag", [False, True])
def test_partial_conversion_needs_no_reference_analysis(
    forbid_reference_analysis, flag
):
    converter = SquinToWire(partial=True)
    wired = converter.emit(two_registers)
    assert converter.report
    assert simulate(WireToSquin().emit(wired), flag) == simulate(two_registers, flag)


@squin.kernel
def replaced_in_loop(n: int):
    q = squin.qubit.new()
    for _ in range(n):
        q = squin.qubit.new()
        squin.x(q)
    return squin.measure(q)


@squin.kernel
def mixed_gate(flag: bool):
    a = squin.qubit.new()
    b = squin.qubit.new()
    c = squin.qubit.new()
    q = a
    if flag:
        q = b
    squin.x(q)
    squin.cx(a, c)
    return squin.broadcast.measure([a, b, c])


@pytest.mark.parametrize(
    "kernel, args, expected",
    [
        (replaced_in_loop, (0,), 0),
        (replaced_in_loop, (1,), 1),
        (replaced_in_loop, (3,), 1),
        (mixed_gate, (False,), [1, 0, 1]),
        (mixed_gate, (True,), [0, 1, 0]),
    ],
)
def test_partial_conversion_preserves_every_possible_allocation(
    forbid_reference_analysis, kernel, args, expected
):
    before = kernel.code.print_str()
    with pytest.raises(WireConversionError):
        SquinToWire().emit(kernel)
    assert kernel.code.print_str() == before
    wired = SquinToWire(partial=True).emit(kernel)
    assert simulate(kernel, *args) == expected
    assert simulate(WireToSquin().emit(wired), *args) == expected


@squin.kernel
def fresh_register():
    return squin.qalloc(1)


@squin.kernel
def two_fresh_calls():
    a = fresh_register()
    b = fresh_register()
    squin.cx(a[0], b[0])
    return squin.broadcast.measure([a[0], b[0]])


def test_each_call_has_distinct_fresh_allocations():
    wired = SquinToWire().emit(two_fresh_calls)
    calls = [node for node in wired.code.walk() if isinstance(node, func.Invoke)]
    assert len(calls) == 2
    assert calls[0].callee is calls[1].callee
    assert simulate(WireToSquin().emit(wired)) == [0, 0]
    releases = [node for node in wired.code.walk() if isinstance(node, stmts.Release)]
    first, second = releases[:2]
    first.wire, second.wire = second.wire, first.wire
    assert_only_positions_fail(wired, "register slot")


@squin.kernel
def sliced_rows(rows: ilist.IList[ilist.IList[Qubit, Literal[2]], Literal[3]]):
    squin.x(rows[::-1][0][1])
    squin.x(rows[::2][1][0])


def test_slicing_nested_register_parameters_preserves_row_boundaries():
    back = WireToSquin().emit(SquinToWire().emit(sliced_rows))

    @squin.kernel
    def run_flat():
        qs = squin.qalloc(6)
        back(qs)
        return squin.broadcast.measure(qs)

    assert simulate(run_flat) == [0, 0, 0, 0, 1, 1]
