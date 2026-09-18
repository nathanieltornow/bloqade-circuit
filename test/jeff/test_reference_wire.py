"""Test the reference analysis on jeff wires, registers, loops, switches and calls."""

from kirin import types

from bloqade import jeff
from bloqade.jeff.types import qureg
from bloqade.jeff.dialects import stmts
from bloqade.jeff.analysis.reference import (
    UNTRACKED,
    Slot,
    Whole,
    Unknown,
    Returned,
    Positions,
    WireReferenceAnalysis,
)

from .build import add, entry, method, switch, for_loop


def returned(mt):
    """Return the references of the values that `mt` returns."""
    _, result = WireReferenceAnalysis(jeff.kernel).run(mt)
    assert isinstance(result, Positions)
    return result.refs


def reason(ref) -> str | None:
    return ref.reason if isinstance(ref, Unknown) else None


def constants(block, *values):
    return [add(block, stmts.ConstInt(value=v)).result for v in values]


def gate(block, name, *wires):
    return add(block, stmts.Gate(tuple(wires), (), (), gate_name=name)).results


def test_a_gate_a_measurement_and_a_loop_hand_their_wires_on():
    block, (q0, q1) = entry(jeff.WireType, jeff.WireType)
    (gated,) = gate(block, "h", q0)
    measured = add(block, stmts.MeasureNd(gated))
    lo, hi, one = constants(block, 0, 2, 1)
    loop = for_loop(block, lo, hi, one, (q1,), lambda b, i, s: gate(b, "x", s))
    mt = method(
        block,
        (measured.bit, measured.result_wire, loop.results[0]),
        types.Generic(tuple, types.Bool, jeff.WireType, jeff.WireType),
        inputs=(jeff.WireType, jeff.WireType),
    )
    assert returned(mt) == (UNTRACKED, Whole(q0), Whole(q1))


def test_a_loop_that_swaps_its_wires_loses_both():
    block, (q0, q1) = entry(jeff.WireType, jeff.WireType)
    lo, hi, one = constants(block, 0, 2, 1)
    loop = for_loop(block, lo, hi, one, (q0, q1), lambda b, i, a, c: (c, a))
    mt = method(
        block,
        tuple(loop.results),
        types.Generic(tuple, jeff.WireType, jeff.WireType),
        inputs=(jeff.WireType, jeff.WireType),
    )
    assert [reason(ref) for ref in returned(mt)] == [
        "a value carried by a loop or branch"
    ] * 2


def flip_second_if():
    """Build a function that measures its first wire and may flip its second."""
    block, (w0, w1, c) = entry(jeff.WireType, jeff.WireType, types.Int)
    picked = switch(block, c, (w1,), [lambda b, s: (s,)], lambda b, s: gate(b, "x", s))
    bit = add(block, stmts.Measure(w0)).bit
    return method(
        block,
        (bit, picked.results[0]),
        types.Generic(tuple, types.Bool, jeff.WireType),
        inputs=(jeff.WireType, jeff.WireType, types.Int),
        name="flip_second_if",
    )


def test_a_switch_hands_on_a_wire_that_every_branch_yields():
    callee = flip_second_if()
    w1 = callee.callable_region.blocks[0].args[2]
    assert returned(callee) == (UNTRACKED, Whole(w1))


def test_a_call_hands_back_the_wire_of_the_caller():
    callee = flip_second_if()
    block, (a, b) = entry(jeff.WireType, jeff.WireType)
    (c,) = constants(block, 0)
    call = add(block, stmts.Call(callee, (a, b, c), (types.Bool, jeff.WireType)))
    mt = method(
        block,
        (call.results[1],),
        jeff.WireType,
        inputs=(jeff.WireType, jeff.WireType),
    )
    assert returned(mt) == (Whole(b),)


def test_a_wire_that_a_callee_allocates_is_rooted_at_the_call():
    inner, _ = entry()
    fresh = add(inner, stmts.Alloc()).result
    callee = method(inner, (fresh,), jeff.WireType, name="make")
    block, _ = entry()
    call = add(block, stmts.Call(callee, (), (jeff.WireType,)))
    mt = method(block, (call.results[0],), jeff.WireType)
    (ref,) = returned(mt)
    assert ref == Whole(Returned(call, 0))
    assert repr(ref) == "Whole(make()[0])"


def test_a_wire_inserted_into_its_own_slot_hands_the_register_on():
    block, (reg,) = entry(qureg(2))
    zero, one = constants(block, 0, 1)
    extracted = add(block, stmts.Extract(reg, zero))
    (flipped,) = gate(block, "x", extracted.wire)
    back = add(block, stmts.Insert(extracted.result_reg, zero, flipped))
    moved = add(block, stmts.Extract(back.result, zero))
    elsewhere = add(block, stmts.Insert(moved.result_reg, one, moved.wire))
    mt = method(
        block,
        (back.result, elsewhere.result),
        types.Generic(tuple, jeff.QuregType, jeff.QuregType),
        inputs=(qureg(2),),
    )
    frame, _ = WireReferenceAnalysis(jeff.kernel).run(mt)
    assert frame.entries[extracted.wire] == Slot(reg, 0)
    assert frame.entries[back.result] == Whole(reg)
    assert reason(frame.entries[elsewhere.result]) == (
        "a register that holds a wire from another slot"
    )


def test_a_recursive_call_returns_unknown_wires():
    block, (q,) = entry(jeff.WireType)
    mt = method(block, (q,), jeff.WireType, inputs=(jeff.WireType,), name="rec")
    ret = block.last_stmt
    call = stmts.Call(mt, (q,), (jeff.WireType,))
    call.insert_before(ret)
    ret.replace_by(stmts.Return(call.results[0]))
    (ref,) = returned(mt)
    assert reason(ref) == "the result of a recursive call"


def test_a_returned_register_keeps_its_allocated_length():
    inner, _ = entry()
    (three,) = constants(inner, 3)
    made = add(inner, stmts.RegAlloc(three)).result
    measured = add(inner, stmts.RegLength(made))
    callee = method(inner, (measured.result_reg,), jeff.QuregType, name="make")
    block, (q,) = entry(jeff.WireType)
    reset = add(block, stmts.Reset(q)).result
    call = add(block, stmts.Call(callee, (), (jeff.QuregType,)))
    (last,) = constants(block, -1)
    extracted = add(block, stmts.Extract(call.results[0], last))
    mt = method(
        block,
        (reset, extracted.result_reg, extracted.wire),
        types.Generic(tuple, jeff.WireType, jeff.QuregType, jeff.WireType),
        inputs=(jeff.WireType,),
    )
    root = Returned(call, 0)
    assert returned(mt) == (Whole(q), Whole(root), Slot(root, 2))


def test_the_analysis_keeps_every_function_in_its_own_terms():
    callee = flip_second_if()
    block, (a, b) = entry(jeff.WireType, jeff.WireType)
    (c,) = constants(block, 0)
    call = add(block, stmts.Call(callee, (a, b, c), (types.Bool, jeff.WireType)))
    mt = method(
        block,
        (call.results[1],),
        jeff.WireType,
        inputs=(jeff.WireType, jeff.WireType),
    )
    analysis = WireReferenceAnalysis(jeff.kernel)
    frame, _ = analysis.run(mt)
    assert list(analysis.results) == [callee, mt]
    w1 = callee.callable_region.blocks[0].args[2]
    assert analysis.results[callee] == Positions((UNTRACKED, Whole(w1)))
    assert frame.entries[w1] == Whole(w1)
    assert frame.entries[call.results[1]] == Whole(b)
