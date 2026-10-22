"""This module computes the lifetime of each qubit root in one block of a squin kernel.

The lifetime of a root runs from the statement that allocates it to the last
statement that touches it. The analysis reads the places of `QubitReferenceAnalysis`.
So a use through an alias, such as `q = qs[0]` or a qubit inside a list, counts as
a use of the root.

The analysis works at the level of one block. A statement with a body, such as a
loop or a branch, touches every root that any statement inside the body touches.
An operand that the reference analysis cannot name touches every root that the
block has allocated so far.
"""

from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.dialects import func, ilist

from bloqade import squin
from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import QubitType
from bloqade.analysis.reference import (
    Ref,
    Root,
    Whole,
    Unknown,
    Returned,
    Untracked,
    roots,
)

ESCAPES = float("inf")
"""Give the end of a root that leaves the block, such as a returned qubit."""


@dataclass
class Lifetime:
    """Hold the first and the last statement of a block that touch one root.

    `start` is the index of the allocation in the block. A root that the block
    receives from outside starts at -1. `end` is the index of the last statement
    that touches the root, or `ESCAPES` if the root leaves the block.
    """

    root: Root
    start: int
    end: float


@dataclass
class BlockLifetimes:
    """Hold the lifetime of each root that one block allocates or touches.

    `allocations` maps the index of each single-qubit allocation in the block to
    the root that it allocates.
    """

    block: ir.Block
    lifetimes: dict[Root, Lifetime] = field(default_factory=dict)
    allocations: dict[int, Root] = field(default_factory=dict)

    def peak(self) -> int:
        """Return the largest count of single-qubit allocations that live at once."""
        events: list[tuple[float, int]] = []
        for index in self.allocations:
            life = self.lifetimes[self.allocations[index]]
            events += [(life.start, 1), (life.end + 0.5, -1)]
        live = best = 0
        for _, change in sorted(events):
            live += change
            best = max(best, live)
        return best


def allocated_root(stmt: ir.Statement, refs: Mapping[ir.SSAValue, Ref]) -> Root | None:
    """Return the root of the single qubit that `stmt` allocates, or None.

    `stmt` must be a `qubit.new` statement or a call of the library kernel
    `squin.qubit.new`.
    """
    is_new = isinstance(stmt, qubit_stmts.New) or (
        isinstance(stmt, func.Invoke) and stmt.callee is squin.qubit.new
    )
    if not is_new:
        return None
    ref = refs.get(stmt.results[0])
    return ref.root if isinstance(ref, Whole) else None


def may_hide_qubits(kind: types.TypeAttribute) -> bool:
    """Return True if a value of type `kind` can hold a qubit that the analysis misses.

    The reference analysis reads the type to decide whether a value holds qubits.
    A value of the type `Bottom`, `Any` or a type variable can hold a qubit at
    runtime, and the analysis gives it `Untracked`.
    """
    if isinstance(kind, (types.BottomType, types.AnyType, types.TypeVar)):
        return True
    if isinstance(kind, types.Generic):
        # The second variable of an `IList` is its length, which holds no qubit.
        members = kind.vars[:1] if kind.is_subseteq(ilist.IListType) else kind.vars
        return any(may_hide_qubits(member) for member in members)
    return False


def touched(
    stmt: ir.Statement, refs: Mapping[ir.SSAValue, Ref]
) -> tuple[set[Root], bool, bool]:
    """Return the roots that `stmt` touches, and two flags about its operands.

    The first flag is True if an operand can name any qubit, so the statement
    touches every live root. The second flag is True if the statement returns a
    root out of the function. The statement and every statement in its body count.
    """
    found: set[Root] = set()
    opaque = escapes = False
    for node in stmt.walk():
        for value in node.args:
            ref = refs.get(value)
            if isinstance(ref, Unknown):
                opaque = True
                continue
            if ref is None or isinstance(ref, Untracked):
                opaque = opaque or may_hide_qubits(value.type)
                opaque = opaque or (ref is None and _quantum(value.type))
                continue
            found.update(roots(ref))
            if isinstance(node, func.Return) and roots(ref):
                escapes = True
    return found, opaque, escapes


def _quantum(kind: types.TypeAttribute) -> bool:
    """Return True if the type `kind` holds qubits."""
    if kind.is_subseteq(QubitType):
        return True
    if isinstance(kind, types.Generic):
        members = kind.vars[:1] if kind.is_subseteq(ilist.IListType) else kind.vars
        return any(_quantum(member) for member in members)
    return False


def block_lifetimes(block: ir.Block, refs: Mapping[ir.SSAValue, Ref]) -> BlockLifetimes:
    """Return the lifetime of each root in `block`, from the places `refs`."""
    result = BlockLifetimes(block)
    lifetimes = result.lifetimes
    for index, stmt in enumerate(block.stmts):
        if (root := allocated_root(stmt, refs)) is not None:
            result.allocations[index] = root
            lifetimes[root] = Lifetime(root, index, index)
            continue
        found, opaque, escapes = touched(stmt, refs)
        if opaque:
            found |= set(lifetimes)
        for root in found:
            life = lifetimes.setdefault(root, Lifetime(root, -1, index))
            life.end = ESCAPES if escapes else max(life.end, index)
    return result


def value_of(root: Root, block: ir.Block) -> ir.SSAValue | None:
    """Return the SSA value of `block` that holds the single qubit `root`, or None.

    An allocation in `block` holds its root in its result. A call that returns one
    qubit holds the root that the callee allocates in its only result.
    """
    if isinstance(root, Returned):
        if root.path != (0,) or len(root.call.results) != 1:
            return None
        value = root.call.results[0]
    else:
        value = root
    owner = value.owner
    if isinstance(owner, ir.Statement) and owner.parent_block is block:
        return value
    return None
