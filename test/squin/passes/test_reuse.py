"""Test the qubit lifetimes and the pass `ReuseQubits` on squin kernels."""

from kirin import ir
from kirin.dialects import scf, func

from bloqade import squin
from bloqade.types import Qubit
from bloqade.pyqrack import DynamicMemorySimulator
from bloqade.squin.passes.reuse import ReuseQubits
from bloqade.squin.analysis.lifetime import QubitEffects, block_lifetimes


def reused(kernel: ir.Method) -> tuple[ir.Method, int]:
    """Return a copy of `kernel` after the pass, and the count of reused qubits."""
    mt = kernel.similar()
    reuse = ReuseQubits(mt.dialects)
    reuse(mt)
    return mt, reuse.reused


def allocations(mt: ir.Method) -> int:
    """Return the count of calls of `squin.qubit.new` in `mt`, bodies included."""
    return sum(
        isinstance(node, func.Invoke) and node.callee is squin.qubit.new
        for node in mt.callable_region.walk()
    )


def peak(mt: ir.Method) -> int:
    """Return the peak count of live single-qubit allocations in the body of `mt`."""
    effects = QubitEffects(mt.dialects)
    effects.run(mt)
    return block_lifetimes(mt.callable_region.blocks[0], effects.effects).peak()


def outcomes(mt: ir.Method) -> list[int]:
    """Return the measured bits of a deterministic kernel as integers."""
    return [int(bit) for bit in DynamicMemorySimulator().run(mt)]


@squin.kernel
def sequential():
    data = squin.qubit.new()
    squin.x(data)
    a = squin.qubit.new()
    squin.cx(data, a)
    first = squin.measure(a)
    b = squin.qubit.new()
    squin.cx(data, b)
    second = squin.measure(b)
    c = squin.qubit.new()
    squin.x(c)
    return [first, second, squin.measure(c), squin.measure(data)]


def test_sequential_ancillas_share_one_qubit():
    # The ancillas `b` and `c` both reuse the qubit of `a`.
    mt, count = reused(sequential)
    assert count == 2
    assert allocations(mt) == allocations(sequential) - 2
    assert peak(mt) == peak(sequential) == 2
    assert outcomes(mt) == outcomes(sequential) == [1, 1, 1, 1]


@squin.kernel
def dead_data():
    a = squin.qubit.new()
    squin.x(a)
    first = squin.measure(a)
    b = squin.qubit.new()
    squin.h(b)
    squin.h(b)
    return [first, squin.measure(b)]


def test_a_qubit_after_its_last_use_goes_to_the_next_allocation():
    mt, count = reused(dead_data)
    assert count == 1
    assert peak(dead_data) == 1
    # The pass resets `a`, so `b` starts in the state 0.
    assert outcomes(mt) == outcomes(dead_data) == [1, 0]


@squin.kernel
def overlapping():
    a = squin.qubit.new()
    b = squin.qubit.new()
    squin.cx(a, b)
    return [squin.measure(a), squin.measure(b)]


def test_qubits_that_live_at_once_keep_their_own_allocations():
    _, count = reused(overlapping)
    assert count == 0


@squin.kernel
def returned() -> Qubit:
    a = squin.qubit.new()
    squin.x(a)
    b = squin.qubit.new()
    squin.measure(b)
    return a


def test_a_returned_qubit_stays_alive():
    _, count = reused(returned)
    assert count == 0


@squin.kernel
def picked_in_a_branch():
    a = squin.qubit.new()
    squin.x(a)
    b = squin.qubit.new()
    q = a
    if squin.measure(b):
        q = b
    c = squin.qubit.new()
    squin.h(q)
    return squin.measure(c)


def test_an_operand_without_a_place_keeps_every_earlier_qubit_alive():
    # The analysis cannot name `q`, so `h(q)` can touch `a` or `b` after `c` exists.
    _, count = reused(picked_in_a_branch)
    assert count == 0


@squin.kernel
def loop_body():
    for _ in range(3):
        a = squin.qubit.new()
        squin.x(a)
        squin.measure(a)
        b = squin.qubit.new()
        squin.measure(b)


def test_the_pass_reuses_qubits_inside_a_loop_body():
    mt, count = reused(loop_body)
    assert count == 1
    assert allocations(mt) == 1


@squin.kernel
def ancilla() -> Qubit:
    q = squin.qubit.new()
    squin.x(q)
    return q


@squin.kernel
def from_a_call():
    a = ancilla()
    first = squin.measure(a)
    b = squin.qubit.new()
    return [first, squin.measure(b)]


def test_a_qubit_that_a_call_returns_can_be_reused():
    mt, count = reused(from_a_call)
    assert count == 1
    assert outcomes(mt) == outcomes(from_a_call) == [1, 0]


@squin.kernel
def noise_after():
    a = squin.qubit.new()
    first = squin.measure(a)
    b = squin.qubit.new()
    squin.depolarize(0.1, a)
    return [first, squin.measure(b)]


def test_a_statement_without_a_rule_still_uses_its_qubits():
    # `depolarize` has no rule in `QubitEffects`, so the fallback records its use
    # of `a`. That use comes after `b` exists, so `b` keeps its own qubit.
    _, count = reused(noise_after)
    assert count == 0


@squin.kernel
def loop_uses(data: Qubit):
    for _ in range(3):
        squin.h(data)


def test_a_loop_carries_the_uses_of_its_body():
    mt = loop_uses.similar()
    effects = QubitEffects(mt.dialects)
    effects.run(mt)
    (loop,) = [node for node in mt.callable_region.walk() if isinstance(node, scf.For)]
    data = mt.callable_region.blocks[0].args[1]
    assert effects.effects[loop].uses == {data}
