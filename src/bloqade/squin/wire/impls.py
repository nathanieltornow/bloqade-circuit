"""Define the rules of the reference analysis for wire statements.

A `Reference` or a `ReferenceRegister` is a new root, like `qubit.new`. Every
other wire statement holds no reference, so the fallback rule covers it.
"""

from kirin import interp
from kirin.analysis.forward import ForwardFrame

from bloqade.analysis.reference import KEY, Ref, Whole, Register, ReferenceAnalysis

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
