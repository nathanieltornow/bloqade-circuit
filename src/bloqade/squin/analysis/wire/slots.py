"""Identify reference accesses and allocations that promotion must preserve.

The reference analysis names each accessed allocation and register position.
This extension records the allocations that each structured region accesses.
An unsupported reference use pins its allocation, which prevents promotion of that allocation.
An unknown reference pins every allocation because the analysis cannot bound its aliases.
"""

from dataclasses import field, dataclass

from kirin import ir
from kirin.dialects import func
from kirin.analysis.forward import ForwardFrame

from bloqade.squin.wire.traits import ReferenceAccess
from bloqade.analysis.reference import Ref, Root, Place, Members, Unknown, places_in
from bloqade.squin.analysis.reference import QubitReferenceAnalysis, holds_qubits


def names_unknown(ref: Ref) -> bool:
    """Return True if `ref`, or a member of its lists and tuples, is `Unknown`."""
    if isinstance(ref, Unknown):
        return True
    if isinstance(ref, Members):
        return any(names_unknown(member) for member in ref.refs)
    return False


@dataclass
class SlotAnalysis(QubitReferenceAnalysis):
    """Record accessed places, region effects and allocation-wide promotion barriers.

    `touched` maps each enclosing region operation to the places that its body accesses.
    `accessed` groups those places by allocation root.
    `pinned` holds roots that an unsupported operation accesses.
    `pin_all` records an access whose possible roots remain unknown.
    """

    touched: dict[ir.Statement, set[Place]] = field(default_factory=dict, init=False)
    accessed: dict[Root, set[Place]] = field(default_factory=dict, init=False)
    pinned: set[Root] = field(default_factory=set, init=False)
    pin_all: bool = field(default=False, init=False)

    def initialize(self):
        """Clear the records of an earlier run."""
        self.touched, self.accessed = {}, {}
        self.pinned, self.pin_all = set(), False
        return super().initialize()

    def eval_fallback(
        self, frame: ForwardFrame[Ref], node: ir.Statement
    ) -> tuple[Ref, ...]:
        """Record the places that `node` touches, then apply the fallback rule.

        A reference transfer records its access.
        A pure operation that returns references constructs aliases, such as `qs[i]`.
        Every other operation with a reference operand pins the allocation that it can access.
        """
        access = node.get_trait(ReferenceAccess)
        if access is not None:
            refs = [frame.get(access.reference(node))]
        elif node.has_trait(ir.Pure) and any(
            holds_qubits(value.type) for value in node.results
        ):
            refs = []
        else:
            refs = [frame.get(value) for value in node.args]
        places = [place for ref in refs for place in places_in(ref)]
        unknown = any(names_unknown(ref) for ref in refs)
        blocking = unknown or (access is None and bool(places))
        if places or unknown:
            self.pin_all = self.pin_all or unknown
            self.touch(node, places, blocking)
        return super().eval_fallback(frame, node)

    def touch(self, node: ir.Statement, places: list[Place], blocking: bool) -> None:
        """Record that `node` touches `places`, for each loop and branch around it."""
        for place in places:
            self.accessed.setdefault(place.root, set()).add(place)
            if blocking:
                self.pinned.add(place.root)
        scope = node.parent_stmt
        while scope is not None and not isinstance(scope, func.Function):
            self.touched.setdefault(scope, set()).update(places)
            scope = scope.parent_stmt
