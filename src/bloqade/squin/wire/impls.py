"""Define the rules of the reference analysis and of `QubitEffects` for wire statements.

A `Reference` or a `ReferenceRegister` is a new root, like `qubit.new`. Every
other wire statement holds no reference, so the fallback rules of both analyses
cover it.
"""

from kirin import interp
from kirin.lattice import EmptyLattice
from kirin.analysis.forward import ForwardFrame

from bloqade.analysis.reference import KEY, Ref, Whole, Register, ReferenceAnalysis
from bloqade.squin.analysis.effects import (
    KEY as EFFECTS_KEY,
    Effect,
    QubitEffects,
)

from . import stmts
from ._dialect import dialect


@dialect.register(key=KEY)
class _References(interp.MethodTable):
    """Make each new reference a root."""

    @interp.impl(stmts.Reference)
    def reference(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Reference,
    ) -> tuple[Ref, ...]:
        """Return the new reference as a whole root."""
        return (Whole(stmt.result),)

    @interp.impl(stmts.ReferenceRegister)
    def reference_register(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.ReferenceRegister,
    ) -> tuple[Ref, ...]:
        """Return the new reference as a register root."""
        return (Register(stmt.result),)


@dialect.register(key=EFFECTS_KEY)
class _Effects(interp.MethodTable):
    """Record that a new reference allocates its root."""

    @interp.impl(stmts.Reference)
    def reference(
        self,
        analysis: QubitEffects,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.Reference,
    ) -> tuple[EmptyLattice, ...]:
        """Record the root of the new reference as an allocation."""
        analysis.record(stmt, Effect(allocates=stmt.result))
        return analysis.results(stmt)
