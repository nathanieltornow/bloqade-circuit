"""Check allocation lifetimes and current wire values across structured regions."""

import pytest
from test_lower import lowered
from kirin.dialects import py, scf, func
from test_roundtrip import ladder, simulate

from bloqade import squin
from bloqade.qubit import qalloc
from bloqade.constants import constant_int
from bloqade.squin.gate import stmts as gates
from bloqade.squin.wire import WireConversionError, stmts, twins
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.rewrite.wire.prepare import Conversion
from bloqade.squin.analysis.validation.wire import WireValidation


def snapshot(method):
    """Record the program and every SSA use without changing either."""
    values = {
        value for node in method.code.walk() for value in (*node.args, *node.results)
    }
    return method.code.print_str(), {value: frozenset(value.uses) for value in values}


def test_conversion_preserves_the_program_and_its_use_lists():
    before = snapshot(ladder)
    method = SquinToWire().emit(ladder)
    assert snapshot(ladder) == before
    WireValidation().check(method)
    assert simulate(WireToSquin().emit(method)) == [1, 1, 1, 1]


def test_promotion_rejects_a_duplicate_transfer():
    method = lowered(ladder)
    first = next(node for node in method.code.walk() if isinstance(node, stmts.Unwrap))
    duplicate = stmts.Unwrap(first.qubit)
    duplicate.insert_after(first)
    with pytest.raises(
        WireConversionError, match="distinct simultaneous register borrows"
    ):
        Conversion(method, SquinToWire().convert).run()


@squin.kernel
def branch_then_loop(flag: bool, n: int):
    qs = squin.qalloc(2)
    if flag:
        squin.x(qs[0])
    else:
        squin.x(qs[1])
    for _ in range(n):
        squin.x(qs[0])
    return squin.broadcast.measure(qs)


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_branches_start_from_the_same_values_and_loops_continue_the_selected_branch(
    flag, n
):
    wired = SquinToWire().emit(branch_then_loop)
    assert simulate(WireToSquin().emit(wired), flag, n) == [
        (int(flag) + n) % 2,
        int(not flag),
    ]


@squin.kernel
def index_from_branch(flag: bool):
    qs = squin.qalloc(3)
    if flag:
        squin.x(qs[0])
        index = 1
    else:
        squin.x(qs[0])
        index = 2
    squin.x(qs[index])
    return squin.broadcast.measure(qs)


@pytest.mark.parametrize("flag", [False, True])
def test_index_materialization_uses_the_rebuilt_branch_result(flag):
    wired = SquinToWire().emit(index_from_branch)
    assert simulate(WireToSquin().emit(wired), flag) == [1, int(flag), int(not flag)]


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
    x = next(node for node in nodes if isinstance(node, twins.TWINS[gates.X]))
    z = next(node for node in nodes if isinstance(node, twins.TWINS[gates.Z]))
    assert x.args[0] is second.result
    assert z.args[0] is first.result
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
    assert constant_int(allocation.size) == 3
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
def partially_blocked_loop(n: int, i: int):
    a = squin.qalloc(2)
    b = squin.qalloc(1)
    for _ in range(n):
        squin.cx(a[0], a[i])
        squin.x(b[0])
    return squin.broadcast.measure(a), squin.broadcast.measure(b)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_a_pinned_allocation_keeps_an_independent_allocation_promotable(n):
    method = SquinToWire(partial=True).emit(partially_blocked_loop)
    allocations = [
        node for node in method.code.walk() if isinstance(node, stmts.AllocRegister)
    ]
    references = [
        node
        for node in method.code.walk()
        if isinstance(node, func.Invoke) and node.callee is qalloc
    ]
    assert len(allocations) == len(references) == 1
    assert constant_int(allocations[0].size) == 1
    loop = next(node for node in method.code.walk() if isinstance(node, scf.For))
    assert allocations[0].result in loop.initializers
    assert simulate(WireToSquin().emit(method), n, 1) == [[0, 0], [n % 2]]


def test_a_pure_observer_of_a_reference_blocks_its_promotion():
    method = lowered(allocation_order)
    transfers = [node for node in method.code.walk() if isinstance(node, stmts.Unwrap)]
    observed = transfers[0].qubit
    compare = py.cmp.Eq(observed, observed)
    compare.insert_before(transfers[0])
    with pytest.raises(WireConversionError, match="Unsupported reference operation"):
        Conversion(method, SquinToWire().convert).run()


def test_promotion_visits_each_prepared_statement_once(monkeypatch):
    """Check the driver's order while boundary rules move scheduled body statements."""
    from test_roundtrip import nested

    from bloqade.squin.rewrite.wire import prepare, promote

    expected = []
    visited = []
    run = prepare.promote
    rewrite = promote.Promote.rewrite_Statement

    def record_input(method, locations, allocations):
        expected.extend(method.code.walk())
        return run(method, locations, allocations)

    def record_visit(rule, node):
        assert node is expected[len(visited)]
        visited.append(node)
        children = [child for region in node.regions for child in region.walk()]
        result = rewrite(rule, node)
        assert all(child.parent_block is not None for child in children)
        return result

    monkeypatch.setattr(prepare, "promote", record_input)
    monkeypatch.setattr(promote.Promote, "rewrite_Statement", record_visit)
    method = SquinToWire().emit(nested)
    assert visited == expected
    assert len(set(visited)) == len(visited)
    WireValidation().check(method)


@squin.kernel
def one_captured_register(n: int):
    active = squin.qalloc(2)
    outside = squin.qalloc(1)
    for _ in range(n):
        squin.x(active[0])
        local = squin.qubit.new()
        squin.x(local)
    squin.x(outside[0])
    return squin.broadcast.measure(active), squin.broadcast.measure(outside)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_a_loop_carries_only_the_allocations_that_its_body_uses(n):
    """Keep unused outer allocations outside the loop and free local allocations inside it."""
    method = SquinToWire().emit(one_captured_register)
    loop = next(node for node in method.code.walk() if isinstance(node, scf.For))
    assert len(loop.initializers) == 1
    assert isinstance(loop.initializers[0].owner, stmts.AllocRegister)
    assert constant_int(loop.initializers[0].owner.size) == 2
    assert len([node for node in loop.body.walk() if isinstance(node, stmts.Free)]) == 1
    assert simulate(WireToSquin().emit(method), n) == [[n % 2, 0], [1]]


def test_a_boundary_rejects_an_outstanding_register_borrow():
    """Require the register's complete state before passing it into a loop or branch."""
    method = lowered(branch_then_loop)
    wrap = next(node for node in method.code.walk() if isinstance(node, stmts.Wrap))
    wrap.delete()
    with pytest.raises(WireConversionError, match="every borrowed element"):
        Conversion(method, SquinToWire().convert).run()
