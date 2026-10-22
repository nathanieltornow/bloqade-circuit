"""This module holds `QubitEffects`, which records what each statement does to qubits.

`QubitEffects` is a kirin forward analysis over `EmptyLattice`. It computes no
value. Its method tables record one `Effect` for each statement that they visit:
the roots that the statement allocates, uses or returns. The roots come from the
places of `QubitReferenceAnalysis`, so a use through an alias or a list counts as a
use of the root.

A statement inside a body also adds its effect to each statement around it. So a
loop or a branch carries the effect of its whole body.

`block_lifetimes` reads the effects of one block and gives the lifetime of each
root, from its allocation to its last use.
"""

from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.lattice import EmptyLattice
from kirin.analysis import Forward
from kirin.dialects import ilist
from kirin.analysis.forward import ForwardFrame

from bloqade.types import QubitType
from bloqade.analysis.reference import (
    Ref,
    Root,
    Unknown,
    Returned,
    Untracked,
    roots,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

KEY = "squin.qubit_effects"
"""Name the registry key of the method tables of `QubitEffects`."""

ESCAPES = float("inf")
"""Give the end of a root that leaves the function, such as a returned qubit."""


@dataclass
class Effect:
    """Hold what one statement does to qubit roots.

    `allocates` is the root of the single qubit that the statement allocates, or
    None. `uses` holds the roots that the statement touches. If `opaque` is True,
    the statement can touch any qubit. If `escapes` is True, the statement hands
    the roots in `uses` out of the function.
    """

    allocates: Root | None = None
    uses: set[Root] = field(default_factory=set)
    opaque: bool = False
    escapes: bool = False

    def absorb(self, inner: "Effect") -> None:
        """Add the effect `inner` of a statement in the body of this statement.

        An allocation inside the body counts as a use, because a value of the body
        can carry the qubit out of it.
        """
        if inner.allocates is not None:
            self.uses.add(inner.allocates)
        self.uses |= inner.uses
        self.opaque = self.opaque or inner.opaque
        self.escapes = self.escapes or inner.escapes


def may_hide_qubits(kind: types.TypeAttribute) -> bool:
    """Return True if a value of type `kind` can hold a qubit that the places miss.

    The reference analysis reads the type to decide whether a value holds qubits.
    A value of the type `Bottom`, `Any` or a type variable can hold a qubit at
    runtime, and the reference analysis gives it `Untracked`.
    """
    if isinstance(kind, (types.BottomType, types.AnyType, types.TypeVar)):
        return True
    if isinstance(kind, types.Generic):
        # The second variable of an `IList` is its length, which holds no qubit.
        members = kind.vars[:1] if kind.is_subseteq(ilist.IListType) else kind.vars
        return any(may_hide_qubits(member) for member in members)
    return False


def holds_qubits(kind: types.TypeAttribute) -> bool:
    """Return True if the type `kind` holds qubits."""
    if isinstance(kind, types.BottomType):
        return False
    if kind.is_subseteq(QubitType):
        return True
    if isinstance(kind, types.Generic):
        members = kind.vars[:1] if kind.is_subseteq(ilist.IListType) else kind.vars
        return any(holds_qubits(member) for member in members)
    return False


@dataclass
class QubitEffects(Forward[EmptyLattice]):
    """Record the `Effect` of each statement of a squin kernel.

    `refs` holds the place of each value. If `refs` is empty, `run` computes it
    with `QubitReferenceAnalysis`. After `run`, `effects` maps each statement
    that the analysis reached to its effect. The analysis records the effects of
    the analyzed method. It treats a call as one statement and does not enter the
    callee.
    """

    keys = (KEY, "absint")
    lattice = EmptyLattice
    refs: Mapping[ir.SSAValue, Ref] = field(default_factory=dict)
    effects: dict[ir.Statement, Effect] = field(default_factory=dict, init=False)

    def initialize(self):
        """Clear the effects of an earlier run."""
        super().initialize()
        self.effects = {}
        return self

    def run(self, method: ir.Method, *args: EmptyLattice, **kwargs: EmptyLattice):
        """Compute the places of `method` if `refs` is empty, then record effects."""
        if not self.refs:
            frame, _ = QubitReferenceAnalysis(method.dialects).run(method)
            self.refs = frame.entries
        return super().run(method, *args, **kwargs)

    def method_self(self, method: ir.Method) -> EmptyLattice:
        """Return the bottom element for the method object."""
        return EmptyLattice.bottom()

    def results(self, stmt: ir.Statement) -> tuple[EmptyLattice, ...]:
        """Return the top element for each result of `stmt`."""
        return tuple(EmptyLattice.top() for _ in stmt.results)

    def eval_fallback(
        self, frame: ForwardFrame[EmptyLattice], node: ir.Statement
    ) -> tuple[EmptyLattice, ...]:
        """Record that a statement without a rule uses the roots of its operands.

        A statement with a body that has no rule is opaque. The analysis does not
        run that body, so a use inside it would stay hidden.
        """
        effect = self.operand_uses(node)
        effect.opaque = effect.opaque or bool(node.regions)
        self.record(node, effect)
        return self.results(node)

    def operand_uses(self, stmt: ir.Statement) -> Effect:
        """Return the effect of `stmt` that its operands give.

        An operand without a place, or with a type that can hide a qubit, makes the
        effect opaque.
        """
        effect = Effect()
        for value in stmt.args:
            ref = self.refs.get(value)
            if isinstance(ref, Unknown):
                effect.opaque = True
            elif ref is None or isinstance(ref, Untracked):
                hidden = may_hide_qubits(value.type)
                effect.opaque |= hidden or (ref is None and holds_qubits(value.type))
            else:
                effect.uses.update(roots(ref))
        return effect

    def record(self, stmt: ir.Statement, effect: Effect) -> None:
        """Store the effect of `stmt` and add it to each statement around `stmt`.

        A loop runs its body more than once, so the effect of a statement joins
        the effect of each earlier run.
        """
        self.effects.setdefault(stmt, Effect()).absorb(effect)
        if effect.allocates is not None:
            self.effects[stmt].allocates = effect.allocates
            self.effects[stmt].uses.discard(effect.allocates)
        parent = stmt.parent_stmt
        while parent is not None and parent.parent_stmt is not None:
            self.effects.setdefault(parent, Effect()).absorb(effect)
            parent = parent.parent_stmt


@dataclass
class Lifetime:
    """Hold the first and the last statement of a block that touch one root.

    `start` is the index of the allocation in the block. A root that the block
    receives from outside starts at -1. `end` is the index of the last statement
    that touches the root, or `ESCAPES` if the root leaves the function.
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
        for root in self.allocations.values():
            life = self.lifetimes[root]
            events += [(life.start, 1), (life.end + 0.5, -1)]
        live = best = 0
        for _, change in sorted(events):
            live += change
            best = max(best, live)
        return best


def block_lifetimes(
    block: ir.Block, effects: Mapping[ir.Statement, Effect]
) -> BlockLifetimes:
    """Return the lifetime of each root in `block`, from the recorded `effects`.

    An opaque statement touches every root that the block has met so far. A
    statement without an effect is a statement that the analysis never reached.
    """
    result = BlockLifetimes(block)
    lifetimes = result.lifetimes
    for index, stmt in enumerate(block.stmts):
        effect = effects.get(stmt)
        if effect is None:
            continue
        if effect.allocates is not None:
            result.allocations[index] = effect.allocates
            lifetimes[effect.allocates] = Lifetime(effect.allocates, index, index)
        used = set(effect.uses)
        if effect.opaque:
            used |= set(lifetimes)
        for root in used:
            life = lifetimes.setdefault(root, Lifetime(root, -1, index))
            life.end = ESCAPES if effect.escapes else max(life.end, index)
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
