"""Emit Jeff quantum operations directly from squin wire operands.

A Jeff register value carries its static length in its type, such as `qureg(3)`,
whenever the emitter knows that length. An operation that continues a register
keeps the type of its input. A Jeff loop gives its carried values the types of its
initial values, so the static length also holds inside loop bodies.
"""

from typing import Any, cast
from collections.abc import Callable

from kirin import ir, types, interp

from bloqade.qubit import stmts as qubit_stmts
from bloqade.constants import constant_int
from bloqade.jeff.forms import GATES
from bloqade.jeff.types import IntArrayType, qureg, qureg_length
from bloqade.squin.gate import stmts as source_gates
from bloqade.squin.wire import stmts as wire
from bloqade.jeff.dialects import stmts
from bloqade.squin.wire.twins import TWINS, REGISTER_TWINS
from bloqade.squin.wire.traits import Twin, RegisterTwin

from .emitter import KEY, Frame, Value, WireEmitter, bitwidth, elements


def register_twin_rule(rule: Any) -> Any:
    """Register the method `rule` for every register twin."""
    for twin in REGISTER_TWINS.values():
        rule = interp.impl(twin)(rule)
    return rule


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
        made = frame.push(stmts.RegAlloc(frame.scalar(stmt.size)))
        made.result.type = qureg(constant_int(stmt.size))
        return (made.result,)

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
        register = frame.scalar(stmt.register)
        made = frame.push(stmts.Extract(register, frame.scalar(stmt.index)))
        made.result_reg.type = register.type
        return made.result_reg, made.wire

    @interp.impl(wire.Release)
    def release(self, emit: WireEmitter, frame: Frame, stmt: wire.Release):
        """Insert the borrowed qubit into its register position."""
        register = frame.scalar(stmt.register)
        made = frame.push(
            stmts.Insert(register, frame.scalar(stmt.index), frame.scalar(stmt.wire))
        )
        made.result.type = register.type
        return (made.result,)

    @interp.impl(wire.Length)
    def length(self, emit: WireEmitter, frame: Frame, stmt: wire.Length):
        """Read the register length and preserve its continuation."""
        register = frame.scalar(stmt.register)
        made = frame.push(stmts.RegLength(register))
        made.result_reg.type = register.type
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
        # The first result of a take or a put continues the register operand.
        if isinstance(stmt, (wire.Take, wire.Put, wire.TakeSlice, wire.PutSlice)):
            made.results[0].type = frame.scalar(stmt.register).type
        if isinstance(stmt, wire.TakeSlice):
            made.results[1].type = qureg(constant_int(stmt.length))
        return tuple(made.results)

    @interp.impl(wire.PackRegister)
    def pack(self, emit: WireEmitter, frame: Frame, stmt: wire.PackRegister):
        """Create a Jeff register from its owned qubits in operand order."""
        return (
            frame.push(
                stmts.RegCreate(tuple(frame.scalar(value) for value in stmt.wires))
            ).result,
        )

    @register_twin_rule
    def broadcast(self, emit: WireEmitter, frame: Frame, stmt: ir.Statement):
        """Apply the scalar twin of a register twin at each register position.

        Jeff gates act on single qubits. If the registers have a static length, the
        rule emits the scalar twin once per position and builds each classical
        result array from its items. Otherwise it emits a Jeff loop that fills
        arrays of runtime length.
        """
        trait = stmt.get_present_trait(RegisterTwin)
        probe = scalar_probe(stmt)
        rule = emit.lookup_registry(frame, probe)
        if rule is None:
            raise interp.InterpreterError(
                f"Jeff has no form for '{trait.scalar.name}'."
            )
        registers = [frame.scalar(getattr(stmt, field)) for _, field in trait.wires]
        lengths = [qureg_length(register.type) for register in registers]
        if trait.truncate:
            size = None if None in lengths else min(n for n in lengths if n is not None)
        else:
            size = next((n for n in lengths if n is not None), None)
        if size is None:
            return looped(emit, frame, stmt, probe, rule, registers)
        return unrolled(emit, frame, stmt, probe, rule, registers, size)


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


def scalar_probe(stmt: ir.Statement) -> ir.Statement:
    """Return the scalar twin of the register twin `stmt` on placeholder operands.

    The placeholders let the Jeff rule of the scalar twin run on Jeff values that
    `apply_scalar` supplies.
    """
    scalar = stmt.get_present_trait(RegisterTwin).scalar
    fields = getattr(scalar, "__kirin_stmt_fields")
    return scalar(
        **{name: ir.TestValue(field.type) for name, field in fields.std_args.items()},
        **{name: getattr(stmt, name) for name in fields.attributes},
    )


def apply_scalar(
    emit: WireEmitter,
    frame: Frame,
    stmt: ir.Statement,
    probe: ir.Statement,
    rule: Callable[..., interp.StatementResult[Value]],
    qubits: dict[str, ir.SSAValue],
) -> tuple[ir.SSAValue, ...]:
    """Emit the scalar twin `probe` through `rule` and return its Jeff results.

    `qubits` maps each qubit-list field of the source statement to one Jeff wire.
    The classical operands come from the register twin `stmt`.
    """
    trait = stmt.get_present_trait(RegisterTwin)
    fields = getattr(trait.scalar, "__kirin_stmt_fields")
    register_of = {
        wire_field: source
        for source, wire_field in trait.scalar.get_present_trait(Twin).wires
    }
    for name, arg in zip(fields.std_args, probe.args, strict=True):
        frame.entries[arg] = (
            qubits[register_of[name]]
            if name in register_of
            else frame.value(getattr(stmt, name))
        )
    results = rule(emit, frame, probe)
    assert isinstance(results, tuple)
    return tuple(cast(ir.SSAValue, value) for value in results)


def unrolled(
    emit: WireEmitter,
    frame: Frame,
    stmt: ir.Statement,
    probe: ir.Statement,
    rule: Callable[..., interp.StatementResult[Value]],
    registers: list[ir.SSAValue],
    size: int,
) -> tuple[ir.SSAValue, ...]:
    """Emit the scalar twin at each of the `size` positions of the registers."""
    trait = stmt.get_present_trait(RegisterTwin)
    collected: list[list[ir.SSAValue]] = [[] for _ in trait.results]
    for position in range(size):
        at = frame.push(stmts.ConstInt(value=position)).result
        qubits: dict[str, ir.SSAValue] = {}
        for k, (source, _) in enumerate(trait.wires):
            made = frame.push(stmts.Extract(registers[k], at))
            made.result_reg.type = registers[k].type
            registers[k] = made.result_reg
            qubits[source] = made.wire
        outputs = apply_scalar(emit, frame, stmt, probe, rule, qubits)
        for k, register in enumerate(registers):
            inserted = frame.push(stmts.Insert(register, at, outputs[k]))
            inserted.result.type = register.type
            registers[k] = inserted.result
        for k, items in enumerate(collected):
            items.append(outputs[len(registers) + k])
    arrays = [
        frame.push(
            stmts.IntArrayCreate(
                tuple(items), bitwidth=bitwidth(elements(getattr(stmt, result).type))
            )
        ).result
        for (_, result), items in zip(trait.results, collected, strict=True)
    ]
    return (*registers, *arrays)


def looped(
    emit: WireEmitter,
    frame: Frame,
    stmt: ir.Statement,
    probe: ir.Statement,
    rule: Callable[..., interp.StatementResult[Value]],
    registers: list[ir.SSAValue],
) -> tuple[ir.SSAValue, ...]:
    """Emit a Jeff loop that applies the scalar twin at each register position.

    Each classical result goes into an array of runtime length.
    """
    trait = stmt.get_present_trait(RegisterTwin)
    # Without truncation, every register has the length of the first one.
    sizes: list[ir.SSAValue] = []
    for k in range(len(registers) if trait.truncate else 1):
        made = frame.push(stmts.RegLength(registers[k]))
        made.result_reg.type = registers[k].type
        registers[k] = made.result_reg
        sizes.append(made.length)
    size = sizes[0]
    for other in sizes[1:]:
        size = frame.push(stmts.IntMinS(size, other)).result
    arrays = [
        frame.push(
            stmts.IntArrayZero(
                size, bitwidth=bitwidth(elements(getattr(stmt, result).type))
            )
        ).result
        for _, result in trait.results
    ]
    with emit.new_frame(stmt) as inner:
        index = inner.body.args.append_from(types.Int, "index")
        carried: list[ir.SSAValue] = [
            inner.body.args.append_from(r.type) for r in registers
        ]
        bits: list[ir.SSAValue] = [
            inner.body.args.append_from(IntArrayType) for _ in arrays
        ]
        qubits: dict[str, ir.SSAValue] = {}
        for k, (source, _) in enumerate(trait.wires):
            made = inner.push(stmts.Extract(carried[k], index))
            made.result_reg.type = carried[k].type
            carried[k] = made.result_reg
            qubits[source] = made.wire
        outputs = apply_scalar(emit, inner, stmt, probe, rule, qubits)
        for k, register in enumerate(carried):
            inserted = inner.push(stmts.Insert(register, index, outputs[k]))
            inserted.result.type = register.type
            carried[k] = inserted.result
        for k, array in enumerate(bits):
            bits[k] = inner.push(
                stmts.IntArraySet(array, index, outputs[len(carried) + k])
            ).result
        body = emit.leave(inner, [*carried, *bits])
    start = frame.push(stmts.ConstInt(value=0)).result
    step = frame.push(stmts.ConstInt(value=1)).result
    loop = frame.push(stmts.For(start, size, step, (*registers, *arrays), body))
    return tuple(loop.results)
