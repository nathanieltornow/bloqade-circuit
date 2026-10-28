"""Find the places that each loop and branch of a lowered kernel touches, with `SlotAnalysis`.

After lowering, only the statements with the `ReferenceAccess` trait touch a
qubit reference, as only loads and stores touch memory for MLIR's `mem2reg`.
`SlotAnalysis` is the reference analysis of squin with one addition: it records
the place of each access for every loop and branch around it. A statement that
touches a reference without the trait is a blocking use, as in `mem2reg`. It
blocks the promotion of its loops and branches, and of every slot of the roots
that it touches.
"""

from dataclasses import field, dataclass

from kirin import ir
from kirin.dialects import func
from kirin.analysis.forward import ForwardFrame

from bloqade.analysis.reference import (
    Ref,
    Root,
    Place,
    Members,
    Unknown,
    places_in,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

from .traits import ReferenceAccess


def names_unknown(ref: Ref) -> bool:
    """Return True if `ref`, or a member of its lists and tuples, is `Unknown`."""
    if isinstance(ref, Unknown):
        return True
    if isinstance(ref, Members):
        return any(names_unknown(member) for member in ref.refs)
    return False


@dataclass
class SlotAnalysis(QubitReferenceAnalysis):
    """Compute the places of a lowered kernel and the places that its statements touch.

    After `run`, `touched` maps each loop and branch to the places that its body
    accesses, and `accessed` maps each root to the places that the kernel
    accesses in it. `blocked` holds each loop and branch whose body touches a
    reference without the `ReferenceAccess` trait, and `pinned` holds each root
    that such a statement touches. If such a statement reads a reference that no
    place names, the reference may name a qubit of any root, so `pin_all` is True.
    """

    touched: dict[ir.Statement, set[Place]] = field(default_factory=dict, init=False)
    accessed: dict[Root, set[Place]] = field(default_factory=dict, init=False)
    blocked: set[ir.Statement] = field(default_factory=set, init=False)
    pinned: set[Root] = field(default_factory=set, init=False)
    pin_all: bool = field(default=False, init=False)

    def initialize(self):
        """Clear the records of an earlier run."""
        self.touched, self.accessed, self.blocked = {}, {}, set()
        self.pinned, self.pin_all = set(), False
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
            refs = [frame.get(access.reference(node))]
        elif not node.has_trait(ir.Pure):
            refs = [frame.get(value) for value in node.args]
        else:
            refs = []
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
            if blocking:
                self.blocked.add(scope)
            scope = scope.parent_stmt
