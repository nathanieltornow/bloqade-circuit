"""Find the places that each loop and branch of a lowered kernel touches, with `SlotAnalysis`.

After lowering, only the statements with the `ReferenceAccess` trait touch a
qubit reference, as only loads and stores touch memory for MLIR's `mem2reg`.
`SlotAnalysis` is the reference analysis of squin with one addition: it records
the place of each access for every loop and branch around it. A statement that
touches a reference without the trait blocks the promotion of its loops and
branches, like a blocking use in `mem2reg`.
"""

from dataclasses import field, dataclass

from kirin import ir
from kirin.dialects import func
from kirin.analysis.forward import ForwardFrame

from bloqade.analysis.reference import (
    Ref,
    Root,
    Place,
    Unknown,
    places_in,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

from .traits import ReferenceAccess


@dataclass
class SlotAnalysis(QubitReferenceAnalysis):
    """Compute the places of a lowered kernel and the places that its statements touch.

    After `run`, `touched` maps each loop and branch to the places that its body
    accesses, and `accessed` maps each root to the places that the kernel
    accesses in it. `blocked` holds each loop and branch whose body touches a
    reference without the `ReferenceAccess` trait.
    """

    touched: dict[ir.Statement, set[Place]] = field(default_factory=dict, init=False)
    accessed: dict[Root, set[Place]] = field(default_factory=dict, init=False)
    blocked: set[ir.Statement] = field(default_factory=set, init=False)

    def initialize(self):
        """Clear the records of an earlier run."""
        self.touched, self.accessed, self.blocked = {}, {}, set()
        return super().initialize()

    def eval_fallback(
        self, frame: ForwardFrame[Ref], node: ir.Statement
    ) -> tuple[Ref, ...]:
        """Record the places that `node` touches, then apply the fallback rule.

        An access records the place of its reference. A pure statement only builds
        values, such as `qs[i]`, and touches nothing. Any other statement with a
        reference operand blocks the loops and branches around it.
        """
        access = node.get_trait(ReferenceAccess)
        if access is not None:
            ref = frame.get(access.reference(node))
            self.touch(node, places_in(ref), blocking=not isinstance(ref, Place))
        elif not node.has_trait(ir.Pure):
            refs = [frame.get(value) for value in node.args]
            places = [place for ref in refs for place in places_in(ref)]
            if places or any(isinstance(ref, Unknown) for ref in refs):
                self.touch(node, places, blocking=True)
        return super().eval_fallback(frame, node)

    def touch(self, node: ir.Statement, places: list[Place], blocking: bool) -> None:
        """Record that `node` touches `places`, for each loop and branch around it."""
        for place in places:
            self.accessed.setdefault(place.root, set()).add(place)
        scope = node.parent_stmt
        while scope is not None and not isinstance(scope, func.Function):
            self.touched.setdefault(scope, set()).update(places)
            if blocking:
                self.blocked.add(scope)
            scope = scope.parent_stmt
