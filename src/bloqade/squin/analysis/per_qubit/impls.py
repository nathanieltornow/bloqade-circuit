"""This module registers the rules of the per-qubit interpreter for each dialect.

Each method table has the key `per_qubit`. A rule that meets only known values
hands the statement to its `main` rule with `run_main`.
"""

# The squin gate statements derive from base classes that a bare `@statement`
# decorates, which pyright reads as functions. A rule that stacks `interp.impl`
# for several of them then fails the argument check.
# pyright: reportArgumentType=false

from kirin import ir, interp
from kirin.dialects import scf, func, ilist

from bloqade.qubit import stmts as qubit_stmts, qalloc, dialect as qubit_dialect
from bloqade.types import MeasurementResultType
from bloqade.jeff.forms import GATES, QUBIT_KERNELS
from bloqade.squin.gate import stmts as gate_stmts, dialect as gate_dialect

from .analysis import (
    KEY,
    OPAQUE,
    Stop,
    Opaque,
    PerQubitInterpreter,
    same,
    touches,
    holds_qubits,
)


@scf.dialect.register(key=KEY)
class ScfRules(interp.MethodTable):
    """Run loops and branches, and both branches of an `if` on an opaque condition."""

    @interp.impl(scf.IfElse)
    def if_else(
        self, interp_: PerQubitInterpreter, frame: interp.Frame, stmt: scf.IfElse
    ):
        """Run the taken branch, or both branches if the condition is opaque.

        Each branch runs on its own copy of the state, and `join` merges the two
        states. A result is the value that both branches give, or `OPAQUE` if the
        branches give different values.
        """
        cond = frame.get(stmt.cond)
        if not isinstance(cond, Opaque):
            return interp_.run_main(frame, stmt)
        before = interp_.qubit_state
        interp_.qubit_state = dict(before)
        then = interp_.frame_call_region(frame, stmt, stmt.then_body, cond)
        then_state, interp_.qubit_state = interp_.qubit_state, dict(before)
        other = interp_.frame_call_region(frame, stmt, stmt.else_body, cond)
        interp_.qubit_state = interp_.join_states(then_state, interp_.qubit_state)
        if isinstance(then, interp.ReturnValue) and isinstance(
            other, interp.ReturnValue
        ):
            return interp.ReturnValue(same(then.value, other.value))
        if isinstance(then, interp.ReturnValue) or isinstance(
            other, interp.ReturnValue
        ):
            raise Stop("a return in only one branch of an opaque condition")
        return tuple(same(a, b) for a, b in zip(then or (), other or ()))

    @interp.impl(scf.For)
    def for_loop(
        self, interp_: PerQubitInterpreter, frame: interp.Frame, stmt: scf.For
    ):
        """Run the body once for each item, or skip a classical loop over an opaque range.

        A loop over an opaque range gives opaque results. If its body can act on a
        qubit, the interpreter stops.
        """
        if not isinstance(frame.get(stmt.iterable), Opaque):
            return interp_.run_main(frame, stmt)
        if touches(stmt):
            raise Stop("a loop over a range that is not constant")
        return tuple(OPAQUE for _ in stmt.results)


@func.dialect.register(key=KEY)
class FuncRules(interp.MethodTable):
    """Allocate qubits for the calls of `qalloc` and `new`, and run other calls."""

    @interp.impl(func.Invoke)
    def invoke(
        self, interp_: PerQubitInterpreter, frame: interp.Frame, stmt: func.Invoke
    ):
        """Return fresh qubits named by the call result, or run the callee.

        The call result is the root of the fresh qubits. So the qubits of
        `qs = squin.qalloc(3)` print as `Slot(%qs, 0)` to `Slot(%qs, 2)`.
        """
        if stmt.callee is qalloc:
            return (interp_.allocate(stmt.result, frame.get(stmt.inputs[0])),)
        if QUBIT_KERNELS.get(stmt.callee) is qubit_stmts.New:
            return (interp_.allocate(stmt.result, None),)
        return interp_.run_main(frame, stmt)


@ilist.dialect.register(key=KEY)
class IListRules(interp.MethodTable):
    """Run a map or a for-each over a list, or skip one over an opaque list."""

    @interp.impl(ilist.Map)
    @interp.impl(ilist.ForEach)
    def apply(
        self,
        interp_: PerQubitInterpreter,
        frame: interp.Frame,
        stmt: ilist.Map | ilist.ForEach,
    ):
        """Call the function on each item, or give an opaque result for an opaque list.

        If the list is opaque and the function can act on a qubit, the
        interpreter stops.
        """
        collection = frame.get(stmt.collection)
        if isinstance(collection, ilist.IList):
            return interp_.run_main(frame, stmt)
        if isinstance(collection, Opaque) and not holds_qubits([frame.get(stmt.fn)]):
            return tuple(OPAQUE for _ in stmt.results)
        raise Stop(
            f"a '{stmt.name}' statement over items that the interpreter cannot list"
        )


@qubit_dialect.register(key=KEY)
class QubitRules(interp.MethodTable):
    """Allocate, measure and reset qubits."""

    @interp.impl(qubit_stmts.New)
    def new(
        self, interp_: PerQubitInterpreter, frame: interp.Frame, stmt: qubit_stmts.New
    ):
        """Return a fresh qubit named by the result of `stmt`."""
        return (interp_.allocate(stmt.result, None),)

    @interp.impl(qubit_stmts.Measure)
    def measure(
        self,
        interp_: PerQubitInterpreter,
        frame: interp.Frame,
        stmt: qubit_stmts.Measure,
    ):
        """Call `update` for each qubit and return one opaque result per qubit."""
        qubits = interp_.qubits(frame.get(stmt.qubits))
        for qubit in qubits:
            interp_.apply(stmt, [qubit])
        return (ilist.IList([OPAQUE] * len(qubits), elem=MeasurementResultType),)

    @interp.impl(qubit_stmts.Reset)
    def reset(
        self, interp_: PerQubitInterpreter, frame: interp.Frame, stmt: qubit_stmts.Reset
    ):
        """Call `update` for each qubit."""
        for qubit in interp_.qubits(frame.get(stmt.qubits)):
            interp_.apply(stmt, [qubit])
        return ()


@gate_dialect.register(key=KEY)
class GateRules(interp.MethodTable):
    """Call `update` for each group of qubits of a gate statement."""

    @interp.impl(gate_stmts.X)
    @interp.impl(gate_stmts.Y)
    @interp.impl(gate_stmts.Z)
    @interp.impl(gate_stmts.H)
    @interp.impl(gate_stmts.S)
    @interp.impl(gate_stmts.T)
    @interp.impl(gate_stmts.SqrtX)
    @interp.impl(gate_stmts.SqrtY)
    @interp.impl(gate_stmts.Rx)
    @interp.impl(gate_stmts.Ry)
    @interp.impl(gate_stmts.Rz)
    @interp.impl(gate_stmts.U3)
    @interp.impl(gate_stmts.CX)
    @interp.impl(gate_stmts.CY)
    @interp.impl(gate_stmts.CZ)
    @interp.impl(gate_stmts.Swap)
    @interp.impl(gate_stmts.CCZ)
    def gate(
        self, interp_: PerQubitInterpreter, frame: interp.Frame, stmt: ir.Statement
    ) -> interp.StatementResult:
        """Call `update` once for each group of qubits of the gate `stmt`.

        The qubit operands follow the angle operands. Group `k` holds the item `k`
        of each list of qubits, so `CX` gives the pairs of a control and a target.
        """
        operands = stmt.args[GATES[type(stmt)].angles :]
        interp_.apply(stmt, frame.get_values(operands))
        return ()
