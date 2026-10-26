"""Run the wire squin rewrite rules as kirin passes."""

from dataclasses import dataclass

from kirin import ir
from kirin.passes import Pass
from kirin.rewrite import (
    Walk,
    Chain,
    Fixpoint,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)
from kirin.rewrite.abc import RewriteResult

from .gates import CancelInverses, MergeRotations
from .reuse import HoistFree, ReuseFreed
from .registers import BorrowFirst, DropRoundTrip, ForwardRelease


@dataclass
class Simplify(Pass):
    """Join register round trips, then cancel and merge neighboring gates.

    The rules run together until none of them changes the method. A cancellation
    can expose a new round trip, and a joined round trip can expose a new pair of
    neighboring gates.
    """

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        """Rewrite `mt` in place."""
        rules = Chain(
            ForwardRelease(),
            BorrowFirst(),
            DropRoundTrip(),
            CancelInverses(),
            MergeRotations(),
        )
        result = Fixpoint(Walk(rules)).rewrite(mt.code)
        result = (
            Fixpoint(Walk(CommonSubexpressionElimination()))
            .rewrite(mt.code)
            .join(result)
        )
        return Fixpoint(Walk(DeadCodeElimination())).rewrite(mt.code).join(result)


@dataclass
class ReuseQubits(Pass):
    """Free each qubit right after its last use, and reuse it for a later `Alloc`."""

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        """Rewrite `mt` in place."""
        result = Fixpoint(Walk(HoistFree())).rewrite(mt.code)
        return Fixpoint(Walk(ReuseFreed())).rewrite(mt.code).join(result)
