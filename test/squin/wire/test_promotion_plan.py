"""Check promotion planning, allocation lifetimes and structured value connections."""

import pytest
from test_lower import lowered
from kirin.dialects import py, scf, func
from test_roundtrip import ladder, simulate

from bloqade import squin
from bloqade.squin.wire import WireConversionError, stmts, twins
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.rewrite.wire.promote import apply_promotion
from bloqade.squin.analysis.wire.promotion import plan_promotion
from bloqade.squin.analysis.validation.wire import WireValidation


def snapshot(method):
    """Record the program and every SSA use without changing either."""
    values = {
        value for node in method.code.walk() for value in (*node.args, *node.results)
    }
    return method.code.print_str(), {value: frozenset(value.uses) for value in values}


def test_planning_keeps_the_program_and_its_use_lists_unchanged():
    method = lowered(ladder)
    before = snapshot(method)
    plan = plan_promotion(method)
    assert plan.before and plan.regions and plan.replacements
    assert snapshot(method) == before
    apply_promotion(method, plan)
    WireValidation().check(method)
    assert simulate(WireToSquin().emit(method)) == [1, 1, 1, 1]


def test_a_plan_can_only_apply_once_to_its_source_method():
    method = lowered(ladder)
    plan = plan_promotion(method)
    other = method.similar()
    before = snapshot(other)
    with pytest.raises(WireConversionError, match="once to the method"):
        apply_promotion(other, plan)
    assert snapshot(other) == before
    apply_promotion(method, plan)
    before = snapshot(method)
    with pytest.raises(WireConversionError, match="once to the method"):
        apply_promotion(method, plan)
    assert snapshot(method) == before


def test_an_invalid_transfer_fails_during_planning_without_mutation():
    method = lowered(ladder)
    first = next(node for node in method.code.walk() if isinstance(node, stmts.Unwrap))
    duplicate = stmts.Unwrap(first.qubit)
    duplicate.insert_after(first)
    before = snapshot(method)
    with pytest.raises(
        WireConversionError, match="distinct simultaneous register borrows"
    ):
        plan_promotion(method)
    assert snapshot(method) == before


@squin.kernel
def allocation_order():
    a = squin.qubit.new()
    b = squin.qubit.new()
    squin.x(b)
    squin.z(a)


def test_allocations_keep_source_order_and_live_until_the_block_exit():
    method = SquinToWire().emit(allocation_order)
    nodes = list(method.callable_region.blocks[0].stmts)
    first, second = [node for node in nodes if isinstance(node, stmts.Alloc)]
    x = next(node for node in nodes if isinstance(node, twins.X))
    z = next(node for node in nodes if isinstance(node, twins.Z))
    assert x.qubit is second.result
    assert z.qubit is first.result
    assert nodes.index(first) < nodes.index(second) < nodes.index(x) < nodes.index(z)
    frees = [node for node in nodes if isinstance(node, stmts.Free)]
    assert len(frees) == 2
    assert all(nodes.index(free) > nodes.index(z) for free in frees)
    assert isinstance(nodes[-1], func.Return)


@squin.kernel
def unused_allocation():
    squin.qalloc(3)


def test_promotion_preserves_an_unused_register_allocation():
    method = SquinToWire().emit(unused_allocation)
    nodes = list(method.callable_region.blocks[0].stmts)
    allocation = next(node for node in nodes if isinstance(node, stmts.AllocRegister))
    release = next(node for node in nodes if isinstance(node, stmts.FreeRegister))
    assert allocation.size.owner.value.unwrap() == 3
    assert release.register is allocation.result


@squin.kernel
def local_in_loop(n: int):
    for _ in range(n):
        q = squin.qubit.new()
        squin.x(q)


def test_each_iteration_owns_its_local_allocation():
    method = SquinToWire().emit(local_in_loop)
    loop = next(node for node in method.code.walk() if isinstance(node, scf.For))
    allocation = next(
        node for node in loop.body.walk() if isinstance(node, stmts.Alloc)
    )
    release = next(node for node in loop.body.walk() if isinstance(node, stmts.Free))
    assert allocation.parent_block is release.parent_block is loop.body.blocks[0]
    assert not loop.initializers


@squin.kernel
def blocked_loop(n: int, i: int):
    a = squin.qalloc(2)
    b = squin.qalloc(1)
    for _ in range(n):
        squin.cx(a[0], a[i])
        squin.x(b[0])
    return squin.broadcast.measure(a), squin.broadcast.measure(b)


def test_one_blocked_allocation_causes_the_whole_conversion_to_fail():
    before = snapshot(blocked_loop)
    with pytest.raises(WireConversionError, match="may name one qubit"):
        SquinToWire().emit(blocked_loop)
    assert snapshot(blocked_loop) == before


def test_a_pure_observer_of_a_reference_blocks_its_promotion():
    method = lowered(allocation_order)
    transfers = [node for node in method.code.walk() if isinstance(node, stmts.Unwrap)]
    observed = transfers[0].qubit
    compare = py.cmp.Eq(observed, observed)
    compare.insert_before(transfers[0])
    plan = plan_promotion(method)
    assert all(slot.allocation is not observed.owner for slot in plan.slots.values())
