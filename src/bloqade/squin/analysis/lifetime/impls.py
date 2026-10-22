"""Define the method tables of `QubitEffects` for the squin dialects.

Every statement without a rule uses the roots of its operands, through
`QubitEffects.eval_fallback`. So a new gate or noise statement needs no rule here.
Write a rule for any statement that allocates a qubit, returns a qubit, or runs a
body.
"""

from kirin import interp
from kirin.lattice import EmptyLattice
from kirin.dialects import scf, func
from kirin.analysis.forward import ForwardFrame

from bloqade import squin
from bloqade.qubit import stmts as qubit_stmts
from bloqade.analysis.reference import Whole

from .analysis import KEY, Effect, QubitEffects


def _allocation(analysis: QubitEffects, result) -> Effect:
    """Return the effect of an allocation whose result is `result`."""
    ref = analysis.refs.get(result)
    if isinstance(ref, Whole):
        return Effect(allocates=ref.root)
    # The places cannot name the new qubit, so any later use can be this qubit.
    return Effect(opaque=True)


@qubit_stmts.dialect.register(key=KEY)
class _Qubit(interp.MethodTable):
    """Record the qubit that each `qubit.new` statement allocates."""

    @interp.impl(qubit_stmts.New)
    def new(
        self,
        analysis: QubitEffects,
        frame: ForwardFrame[EmptyLattice],
        stmt: qubit_stmts.New,
    ) -> tuple[EmptyLattice, ...]:
        """Record the root of the new qubit."""
        analysis.record(stmt, _allocation(analysis, stmt.result))
        return analysis.results(stmt)


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """Record the allocations, the calls and the returns of the `func` dialect."""

    @interp.impl(func.Invoke)
    def invoke(
        self,
        analysis: QubitEffects,
        frame: ForwardFrame[EmptyLattice],
        stmt: func.Invoke,
    ) -> tuple[EmptyLattice, ...]:
        """Record a call of `squin.qubit.new` as an allocation, and any other as uses.

        The analysis does not enter the callee. A callee can only touch the qubits
        that the caller passes, so the uses of the operands cover the call.
        """
        if stmt.callee is squin.qubit.new:
            analysis.record(stmt, _allocation(analysis, stmt.result))
        else:
            analysis.record(stmt, analysis.operand_uses(stmt))
        return analysis.results(stmt)

    @interp.impl(func.Return)
    def return_(
        self,
        analysis: QubitEffects,
        frame: ForwardFrame[EmptyLattice],
        stmt: func.Return,
    ) -> interp.ReturnValue[EmptyLattice]:
        """Record that the returned roots leave the function."""
        effect = analysis.operand_uses(stmt)
        effect.escapes = True
        analysis.record(stmt, effect)
        return interp.ReturnValue(EmptyLattice.top())


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """Run the body of each loop once to record its effects.

    The effects of a body do not depend on the iteration, so one run records
    them all. `scf.IfElse` uses kirin's rule. That rule runs both branches, or only
    the taken branch if the condition is a constant.
    """

    @interp.impl(scf.For)
    def for_(
        self,
        analysis: QubitEffects,
        frame: ForwardFrame[EmptyLattice],
        stmt: scf.For,
    ) -> tuple[EmptyLattice, ...]:
        """Record the uses of the loop operands, then run the body once."""
        analysis.record(stmt, analysis.operand_uses(stmt))
        carried = tuple(EmptyLattice.top() for _ in stmt.initializers)
        with analysis.new_frame(stmt, has_parent_access=True) as body:
            analysis.frame_call_region(
                body, stmt, stmt.body, EmptyLattice.top(), *carried
            )
        return analysis.results(stmt)
