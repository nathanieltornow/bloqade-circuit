"""Select allocations for promotion and retain their reference facts.

The analysis identifies allocation roots and the accesses inside each structured region.
Unsupported accesses exclude their whole allocation from promotion.
"""

from functools import cached_property
from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.rewrite import Walk
from kirin.dialects import scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import WireType, RegisterType
from bloqade.squin.wire.traits import ReferenceAccess
from bloqade.analysis.reference import Ref, Root, Place, Affine, Returned, places_in
from bloqade.squin.analysis.wire.slots import SlotAnalysis
from bloqade.analysis.reference.queries import flat


@dataclass(frozen=True)
class Slot:
    """Identify one whole allocation and its linear value type."""

    root: Root
    allocation: ir.Statement
    type: types.TypeAttribute


@dataclass
class PromotionInfo:
    """Hold reference facts for one method before promotion changes its statements."""

    slots: dict[Root, Slot]
    accesses: dict[ir.Statement, Place]
    touched: dict[ir.Statement, set[Place]]

    @cached_property
    def allocations(self) -> dict[ir.Statement, Slot]:
        """Index selected allocations by their source statement."""
        return {slot.allocation: slot for slot in self.slots.values()}


def allocation(root: Root) -> ir.Statement | None:
    """Return the local allocation or reference initializer for a root."""
    if isinstance(root, Returned):
        return (
            root.call
            if isinstance(root.call, func.Invoke) and root.call.callee is qalloc
            else None
        )
    owner = root.owner
    if isinstance(owner, (qubit_stmts.New, stmts.Reference, stmts.ReferenceRegister)):
        return owner
    return None


@dataclass
class Eligibility(RewriteRule):
    """Select whole allocations whose accesses promotion can replace."""

    refs: Mapping[ir.SSAValue, Ref]
    blocked: set[Root]
    allocations: dict[ir.Statement, Slot]
    ordered: dict[Root, Slot] = field(default_factory=dict)
    accesses: dict[ir.Statement, Place] = field(default_factory=dict)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Record allocations in program order and exclude unsupported reference accesses."""
        if slot := self.allocations.get(node):
            self.ordered[slot.root] = slot
        if access := node.get_trait(ReferenceAccess):
            ref = self.refs.get(access.reference(node))
            if isinstance(ref, Place):
                self.accesses[node] = ref
                if ref.path and not (ref.one_item and isinstance(flat(ref), Affine)):
                    self.blocked.add(ref.root)
                scope = node.parent_stmt
                while scope is not None and not isinstance(scope, func.Function):
                    if not isinstance(scope, (scf.For, scf.IfElse)):
                        self.blocked.add(ref.root)
                    scope = scope.parent_stmt
        return RewriteResult()


def analyze_promotion(method: ir.Method) -> PromotionInfo:
    """Identify eligible allocations and their accesses without changing the method."""
    analysis = SlotAnalysis(method.dialects)
    frame, _ = analysis.run(method)
    if analysis.pin_all:
        return PromotionInfo({}, {}, {})
    allocations = {}
    roots = {place.root for ref in frame.entries.values() for place in places_in(ref)}
    for root in roots:
        if (node := allocation(root)) is not None:
            kind = (
                WireType
                if isinstance(node, (qubit_stmts.New, stmts.Reference))
                else RegisterType
            )
            allocations[node] = Slot(root, node, kind)
    eligibility = Eligibility(frame.entries, set(analysis.pinned), allocations)
    Walk(eligibility, region_first=True).rewrite(method.code)
    slots = {
        root: slot
        for root, slot in eligibility.ordered.items()
        if root not in eligibility.blocked
    }
    return PromotionInfo(slots, eligibility.accesses, analysis.touched)
