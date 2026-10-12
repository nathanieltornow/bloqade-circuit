"""Test the exact circuit depth analysis on jeff programs."""

import pytest
from kirin import types

from bloqade import jeff, squin
from bloqade.jeff.types import qureg
from bloqade.jeff.dialects import stmts
from bloqade.jeff.analysis.depth import Unknown, circuit_depth

from .build import add, entry, method, switch, for_loop, while_loop

WIRE = jeff.WireType
TWO_WIRES = types.Generic(tuple, WIRE, WIRE)


def constants(block, *values):
    return [add(block, stmts.ConstInt(value=v)).result for v in values]


def gate(block, name, *wires):
    return add(block, stmts.Gate(tuple(wires), (), (), gate_name=name)).results


def cx(block, control, target):
    result = add(block, stmts.Gate((target,), (control,), (), gate_name="x")).results
    return result[1], result[0]


def two_wire_program(build):
    """Return a program over two wire parameters that returns both wires."""
    block, (a, b) = entry(WIRE, WIRE)
    out = build(block, a, b)
    return method(block, out, TWO_WIRES, inputs=(WIRE, WIRE))


def counted(block, n, state, body):
    lo, hi, one = constants(block, 0, n, 1)
    return for_loop(block, lo, hi, one, state, body).results


def swap_body(b, i, x, y):
    (x,) = gate(b, "h", x)
    return (y, x)


def ladder_body(b, i, x, y):
    (x,) = gate(b, "h", x)
    x, y = cx(b, x, y)
    (y,) = gate(b, "h", y)
    return (y, x)


def unrolled(body, n):
    """Return the program that runs `body` `n` times without a loop."""

    def build(block, a, b):
        for _ in range(n):
            a, b = body(block, None, a, b)
        return (a, b)

    return two_wire_program(build)


def looped(body, n):
    """Return the program that runs `body` in a `For` loop of `n` iterations."""
    return two_wire_program(lambda block, a, b: counted(block, n, (a, b), body))


def test_gates_on_disjoint_wires_share_one_layer():
    def build(block, a, b):
        (a,) = gate(block, "h", a)
        (b,) = gate(block, "x", b)
        return (a, b)

    assert circuit_depth(two_wire_program(build)) == 1


def test_a_gate_after_a_cx_adds_a_layer():
    def build(block, a, b):
        a, b = cx(block, a, b)
        (b,) = gate(block, "h", b)
        return (a, b)

    assert circuit_depth(two_wire_program(build)) == 2


def test_a_swap_loop_counts_a_gate_on_each_wire_every_second_iteration():
    # One iteration has depth 1, so a multiple of it would give 5.
    assert circuit_depth(looped(swap_body, 5)) == 3


@pytest.mark.parametrize("body", [swap_body, ladder_body])
@pytest.mark.parametrize("n", range(9))
def test_a_loop_has_the_depth_of_its_unrolled_program(body, n):
    assert circuit_depth(looped(body, n)) == circuit_depth(unrolled(body, n))


def test_a_loop_of_a_million_iterations_finishes():
    assert circuit_depth(looped(swap_body, 10**6)) == 500_000


def test_a_loop_keeps_the_deepest_measurement_of_any_iteration():
    def body(b, i, w):
        (w,) = gate(b, "h", w)
        add(b, stmts.Measure(w))
        return (add(b, stmts.Alloc()).result,)

    block, (w,) = entry(WIRE)
    for _ in range(5):
        (w,) = gate(block, "x", w)
    (w,) = counted(block, 3, (w,), body)
    mt = method(block, w, WIRE, inputs=(WIRE,))
    # The first iteration measures at 5 + 1 + 1. Later iterations start at 0.
    assert circuit_depth(mt) == 7


def test_nested_loops_multiply_their_trip_counts():
    def inner(b, j, w):
        return gate(b, "h", w)

    def outer(b, i, w):
        return counted(b, 4, (w,), inner)

    block, (w,) = entry(WIRE)
    (w,) = counted(block, 3, (w,), outer)
    assert circuit_depth(method(block, w, WIRE, inputs=(WIRE,))) == 12


def test_a_cx_ladder_over_a_register_in_a_loop():
    def body(b, i, reg):
        zero, one, two = constants(b, 0, 1, 2)
        e0 = add(b, stmts.Extract(reg, zero))
        e1 = add(b, stmts.Extract(e0.result_reg, one))
        e2 = add(b, stmts.Extract(e1.result_reg, two))
        q0, q1 = cx(b, e0.wire, e1.wire)
        q1, q2 = cx(b, q1, e2.wire)
        reg = add(b, stmts.Insert(e2.result_reg, zero, q0)).result
        reg = add(b, stmts.Insert(reg, one, q1)).result
        return (add(b, stmts.Insert(reg, two, q2)).result,)

    block, _ = entry()
    (three,) = constants(block, 3)
    reg = add(block, stmts.RegAlloc(three)).result
    (reg,) = counted(block, 2, (reg,), body)
    add(block, stmts.RegFree(reg))
    # Iteration 1 gives slots (1, 2, 2). Iteration 2 gives slots (3, 4, 4).
    assert circuit_depth(method(block, None, types.NoneType)) == 4


def test_a_constant_selector_runs_only_its_branch():
    block, (a, b) = entry(WIRE, WIRE)
    (selector,) = constants(block, 1)
    out = switch(
        block,
        selector,
        (a, b),
        [lambda r, x, y: (x, y), lambda r, x, y: (*gate(r, "h", x), y)],
        lambda r, x, y: cx(r, gate(r, "h", x)[0], y),
    ).results
    assert circuit_depth(method(block, tuple(out), TWO_WIRES, inputs=(WIRE, WIRE))) == 1


def test_a_dynamic_selector_takes_the_deepest_branch():
    block, (a, b, flag) = entry(WIRE, WIRE, types.Int)
    out = switch(
        block,
        flag,
        (a, b),
        [lambda r, x, y: (x, y)],
        lambda r, x, y: cx(r, gate(r, "h", x)[0], y),
    ).results
    mt = method(block, tuple(out), TWO_WIRES, inputs=(WIRE, WIRE, types.Int))
    assert circuit_depth(mt) == 2


def test_a_call_applies_the_callee_with_the_caller_depths():
    callee_block, (w,) = entry(WIRE)
    (w,) = gate(callee_block, "h", w)
    callee = method(callee_block, w, WIRE, inputs=(WIRE,), name="flip")

    block, (w,) = entry(WIRE)
    for _ in range(3):
        (w,) = add(block, stmts.Call(callee, (w,), (WIRE,))).results
    assert circuit_depth(method(block, w, WIRE, inputs=(WIRE,))) == 3


def unknown_reason(mt) -> str:
    result = circuit_depth(mt)
    assert isinstance(result, Unknown)
    return result.reason


def test_a_while_loop_has_no_static_depth():
    block, (w,) = entry(WIRE)
    (w,) = while_loop(
        block,
        (w,),
        lambda r, x: (*constants(r, 1), x),
        lambda r, x: gate(r, "h", x),
    ).results
    mt = method(block, w, WIRE, inputs=(WIRE,))
    assert "while loop" in unknown_reason(mt)


def test_a_loop_with_a_parameter_bound_has_no_static_depth():
    block, (w, n) = entry(WIRE, types.Int)
    lo, one = constants(block, 0, 1)
    (w,) = for_loop(block, lo, n, one, (w,), lambda b, i, x: gate(b, "h", x)).results
    mt = method(block, w, WIRE, inputs=(WIRE, types.Int))
    assert "constant bounds" in unknown_reason(mt)


def test_an_extract_at_the_loop_index_has_no_static_depth():
    def body(b, i, reg):
        extracted = add(b, stmts.Extract(reg, i))
        (w,) = gate(b, "h", extracted.wire)
        return (add(b, stmts.Insert(extracted.result_reg, i, w)).result,)

    block, (reg,) = entry(qureg(3))
    (reg,) = counted(block, 3, (reg,), body)
    mt = method(block, reg, qureg(3), inputs=(qureg(3),))
    assert "not a constant" in unknown_reason(mt)


@squin.kernel
def repeated_ladder():
    qs = squin.qalloc(2)
    for _ in range(4):
        squin.h(qs[0])
        squin.cx(qs[0], qs[1])


def test_a_squin_kernel_keeps_its_depth_through_the_jeff_conversion():
    mt = jeff.SquinToJeff().emit(repeated_ladder)
    # Each iteration adds the h and the cx to the first qubit.
    assert circuit_depth(mt) == 8
