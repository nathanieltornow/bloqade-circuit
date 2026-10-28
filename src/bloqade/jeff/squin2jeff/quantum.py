"""Emit Jeff quantum operations directly from squin wire operands."""

from kirin import ir, interp

from bloqade.qubit import stmts as qubit_stmts
from bloqade.jeff.forms import GATES
from bloqade.squin.gate import stmts as source_gates
from bloqade.squin.wire import stmts as wire
from bloqade.jeff.dialects import stmts
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.traits import Twin

from .emitter import KEY, Frame, WireEmitter


@wire.dialect.register(key=KEY)
class _Wire(interp.MethodTable):
    """Translate explicit wire operands and results into Jeff operations."""

    @interp.impl(wire.Alloc)
    def alloc(self, emit: WireEmitter, frame: Frame, stmt: wire.Alloc):
        """Allocate a Jeff wire."""
        return (frame.push(stmts.Alloc()).result,)

    @interp.impl(wire.AllocRegister)
    def register(self, emit: WireEmitter, frame: Frame, stmt: wire.AllocRegister):
        """Allocate a Jeff register of the requested size."""
        return (frame.push(stmts.RegAlloc(frame.scalar(stmt.size))).result,)

    @interp.impl(wire.Free)
    def free(self, emit: WireEmitter, frame: Frame, stmt: wire.Free):
        """Consume the emitted qubit at its explicit lifetime end."""
        frame.push(stmts.Free(frame.scalar(stmt.wire)))
        return ()

    @interp.impl(wire.FreeRegister)
    def free_register(self, emit: WireEmitter, frame: Frame, stmt: wire.FreeRegister):
        """Consume the emitted register at its explicit lifetime end."""
        frame.push(stmts.RegFree(frame.scalar(stmt.register)))
        return ()

    @interp.impl(wire.Borrow)
    def borrow(self, emit: WireEmitter, frame: Frame, stmt: wire.Borrow):
        """Extract one qubit and return the register continuation first."""
        made = frame.push(
            stmts.Extract(frame.scalar(stmt.register), frame.scalar(stmt.index))
        )
        return made.result_reg, made.wire

    @interp.impl(wire.Release)
    def release(self, emit: WireEmitter, frame: Frame, stmt: wire.Release):
        """Insert the borrowed qubit into its register position."""
        return (
            frame.push(
                stmts.Insert(
                    frame.scalar(stmt.register),
                    frame.scalar(stmt.index),
                    frame.scalar(stmt.wire),
                )
            ).result,
        )

    @interp.impl(wire.Length)
    def length(self, emit: WireEmitter, frame: Frame, stmt: wire.Length):
        """Read the register length and preserve its continuation."""
        made = frame.push(stmts.RegLength(frame.scalar(stmt.register)))
        return made.result_reg, made.length

    @interp.impl(TWINS[qubit_stmts.Measure])
    def measure(self, emit: WireEmitter, frame: Frame, stmt):
        """Measure one qubit and return its continuation followed by its bit."""
        made = frame.push(stmts.MeasureNd(frame.scalar(stmt.qubit)))
        return made.result_wire, made.bit

    @interp.impl(TWINS[qubit_stmts.Reset])
    def reset(self, emit: WireEmitter, frame: Frame, stmt):
        """Reset one qubit and return its continuation."""
        return (frame.push(stmts.Reset(frame.scalar(stmt.qubit))).result,)

    @interp.impl(TWINS[source_gates.X])
    @interp.impl(TWINS[source_gates.Y])
    @interp.impl(TWINS[source_gates.Z])
    @interp.impl(TWINS[source_gates.H])
    @interp.impl(TWINS[source_gates.S])
    @interp.impl(TWINS[source_gates.T])
    @interp.impl(TWINS[source_gates.SqrtX])
    @interp.impl(TWINS[source_gates.SqrtY])
    @interp.impl(TWINS[source_gates.Rx])
    @interp.impl(TWINS[source_gates.Ry])
    @interp.impl(TWINS[source_gates.Rz])
    @interp.impl(TWINS[source_gates.U3])
    @interp.impl(TWINS[source_gates.CX])
    @interp.impl(TWINS[source_gates.CY])
    @interp.impl(TWINS[source_gates.CZ])
    @interp.impl(TWINS[source_gates.CCZ])
    @interp.impl(TWINS[source_gates.Swap])
    def gate(self, emit: WireEmitter, frame: Frame, stmt: ir.Statement):
        """Convert turns to radians and map Jeff's target-first result order back to wire order."""
        form = GATES[stmt.get_present_trait(Twin).source]
        angles = tuple(emit.radians(frame, angle) for angle in stmt.args[: form.angles])
        values = tuple(frame.scalar(value) for value in stmt.args[form.angles :])
        controls, targets = values[: form.controls], values[form.controls :]
        made = frame.push(
            stmts.Gate(
                targets,
                controls,
                angles,
                gate_name=form.name,
                adjoint=bool(getattr(stmt, "adjoint", False)),
            )
        )
        count = len(targets)
        return (*made.results[count:], *made.results[:count])

    @interp.impl(wire.Take)
    @interp.impl(wire.Put)
    @interp.impl(wire.TakeSlice)
    @interp.impl(wire.PutSlice)
    @interp.impl(wire.SplitRegister)
    @interp.impl(wire.JoinRegister)
    @interp.impl(wire.DiscardRegister)
    def owned_register(self, emit: WireEmitter, frame: Frame, stmt: ir.Statement):
        """Preserve the operands and results of an explicit register ownership operation."""
        table = {
            wire.Take: stmts.Extract,
            wire.Put: stmts.Insert,
            wire.TakeSlice: stmts.ExtractSlice,
            wire.PutSlice: stmts.InsertSlice,
            wire.SplitRegister: stmts.RegSplit,
            wire.JoinRegister: stmts.RegJoin,
            wire.DiscardRegister: stmts.RegFree,
        }
        made = frame.push(
            table[type(stmt)](*(frame.scalar(value) for value in stmt.args))
        )
        return tuple(made.results)

    @interp.impl(wire.PackRegister)
    def pack(self, emit: WireEmitter, frame: Frame, stmt: wire.PackRegister):
        """Create a Jeff register from its owned qubits in operand order."""
        return (
            frame.push(
                stmts.RegCreate(tuple(frame.scalar(value) for value in stmt.wires))
            ).result,
        )


@qubit_stmts.dialect.register(key=KEY)
class _Predicates(interp.MethodTable):
    """Translate classical predicates on measurement results."""

    @interp.impl(qubit_stmts.IsOne)
    def one(self, emit: WireEmitter, frame: Frame, stmt):
        """Use the measurement bits as their own tests for one."""
        return (frame.value(stmt.measurements),)

    @interp.impl(qubit_stmts.IsZero)
    def zero(self, emit: WireEmitter, frame: Frame, stmt):
        """Negate each measurement bit."""
        return (emit.negate(frame, stmt.measurements),)
