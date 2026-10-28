"""Record what each statement of a squin kernel does to qubits with `QubitEffects`.

`QubitEffects` is a kirin forward analysis over `EmptyLattice`. It computes no
value. Its method tables record one `Effect` for each statement that they visit:
the roots that the statement allocates or uses. The roots come from the
places of `QubitReferenceAnalysis`, so a use through an alias or a list counts as a
use of the root.

A statement inside a body also adds its effect to each statement around it. So a
loop or a branch carries the effect of its whole body.
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
    Place,
    Members,
    Unknown,
    Untracked,
    roots,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

KEY = "squin.qubit_effects"
"""Name the registry key of the method tables of `QubitEffects`."""


@dataclass
class Effect:
    """Hold what one statement does to qubit roots.

    `allocates` is the root of the single qubit that the statement allocates, or
    None. `uses` holds the roots that the statement touches, and `places` holds the
    places of its qubit operands, such as `Slot(%qs, 0)`. If `opaque` is True, the
    statement can touch any qubit.
    """

    allocates: Root | None = None
    uses: set[Root] = field(default_factory=set)
    places: set[Place] = field(default_factory=set)
    opaque: bool = False

    def absorb(self, inner: "Effect") -> None:
        """Add the effect `inner` of a statement in the body of this statement.

        An allocation inside the body counts as a use, because a value of the body
        can carry the qubit out of it.
        """
        if inner.allocates is not None:
            self.uses.add(inner.allocates)
        self.uses |= inner.uses
        self.places |= inner.places
        self.opaque = self.opaque or inner.opaque


def places_in(ref: Ref) -> list[Place]:
    """Return the places that `ref` holds, including the members of lists and tuples."""
    if isinstance(ref, Place):
        return [ref]
    if isinstance(ref, Members):
        return [place for member in ref.refs for place in places_in(member)]
    return []


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
    accessed: dict[Root, set[Place]] = field(default_factory=dict, init=False)
    """Map each root to the places that statements of the method touch in it."""

    def initialize(self):
        """Clear the effects of an earlier run."""
        super().initialize()
        self.effects = {}
        self.accessed = {}
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

        A pure statement only builds values, such as the reference `qs[i]` or a
        slice, and touches no qubit. So it has no effect, even if an operand has a
        type that could hide a qubit. The reference analysis still tracks the
        references that it builds. A statement with a body that has no rule is
        opaque. The analysis does not run that body, so a use inside it would stay
        hidden.
        """
        if node.has_trait(ir.Pure):
            # The empty effect still gives each statement around `node` an entry.
            self.record(node, Effect())
            return self.results(node)
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
                effect.places.update(places_in(ref))
        return effect

    def record(self, stmt: ir.Statement, effect: Effect) -> None:
        """Store the effect of `stmt` and add it to each statement around `stmt`.

        A loop runs its body more than once, so the effect of a statement joins
        the effect of each earlier run.
        """
        self.effects.setdefault(stmt, Effect()).absorb(effect)
        for place in effect.places:
            self.accessed.setdefault(place.root, set()).add(place)
        if effect.allocates is not None:
            self.effects[stmt].allocates = effect.allocates
            self.effects[stmt].uses.discard(effect.allocates)
        parent = stmt.parent_stmt
        while parent is not None and parent.parent_stmt is not None:
            self.effects.setdefault(parent, Effect()).absorb(effect)
            parent = parent.parent_stmt
