"""This module holds the statements of the squin wire dialect.

`Alloc` and `Free` start and end the life of one qubit. `AllocRegister` and
`FreeRegister` do the same for a list of qubits. `Borrow` and `Release` take one
qubit out of a register and put it back. `Length` reads the size of a register.

`Gate`, `Noise`, `Measure` and `Reset` act on single wires. Their register forms
act at every position of whole registers. A gate or a noise channel names its
operation with a kind from `kinds.py`.
"""

from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.types import MeasurementResultType

from .kinds import GATE_FACTS, NOISE_FACTS, Facts, GateKind, NoiseKind
from .types import WireType, RegisterType
from .traits import ThreadsPairs, ThreadsInOrder
from ._dialect import dialect


@statement(dialect=dialect)
class Alloc(ir.Statement):
    """Allocate one qubit in the zero state and return its wire."""

    name = "alloc"
    result: ir.ResultValue = info.result(WireType)


@statement(dialect=dialect)
class Free(ir.Statement):
    """End the life of the qubit that `wire` carries."""

    name = "free"
    wire: ir.SSAValue = info.argument(WireType)


@statement(dialect=dialect)
class AllocRegister(ir.Statement):
    """Allocate `size` qubits in the zero state and return them as one register."""

    name = "alloc_register"
    size: ir.SSAValue = info.argument(types.Int)
    result: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class FreeRegister(ir.Statement):
    """End the life of every qubit of `register`."""

    name = "free_register"
    register: ir.SSAValue = info.argument(RegisterType)


@statement(dialect=dialect)
class Borrow(ir.Statement):
    """Take the qubit at `index` out of `register` as a wire.

    The slot stays empty until a `Release` at the same index fills it again.
    """

    name = "borrow"
    traits = frozenset({ThreadsPairs((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    register_out: ir.ResultValue = info.result(RegisterType)
    wire: ir.ResultValue = info.result(WireType)


@statement(dialect=dialect)
class Release(ir.Statement):
    """Put the qubit that `wire` carries back into `register` at `index`."""

    name = "release"
    traits = frozenset({ThreadsPairs((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    wire: ir.SSAValue = info.argument(WireType)
    register_out: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class Length(ir.Statement):
    """Return `register` and the count of qubits that it holds."""

    name = "length"
    traits = frozenset({ThreadsPairs((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    register_out: ir.ResultValue = info.result(RegisterType)
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class Take(ir.Statement):
    """Move one qubit out of a register and leave its element empty."""

    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    register_out: ir.ResultValue = info.result(RegisterType)
    wire: ir.ResultValue = info.result(WireType)


@statement(dialect=dialect)
class Put(ir.Statement):
    """Fill an empty register element with an owned qubit."""

    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    wire: ir.SSAValue = info.argument(WireType)
    register_out: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class PackRegister(ir.Statement):
    """Consume qubits and form a register in operand order."""

    wires: tuple[ir.SSAValue, ...] = info.argument(WireType)
    result: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class SplitRegister(ir.Statement):
    """Divide a complete register at an index into two owned registers."""

    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    before: ir.ResultValue = info.result(RegisterType)
    after: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class JoinRegister(ir.Statement):
    """Consume two complete registers and concatenate their qubits."""

    first: ir.SSAValue = info.argument(RegisterType)
    second: ir.SSAValue = info.argument(RegisterType)
    result: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class TakeSlice(ir.Statement):
    """Move a contiguous range into an owned register and leave its source elements empty."""

    register: ir.SSAValue = info.argument(RegisterType)
    start: ir.SSAValue = info.argument(types.Int)
    length: ir.SSAValue = info.argument(types.Int)
    register_out: ir.ResultValue = info.result(RegisterType)
    slice_register: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class PutSlice(ir.Statement):
    """Fill an empty contiguous range with the qubits of an owned register."""

    register: ir.SSAValue = info.argument(RegisterType)
    start: ir.SSAValue = info.argument(types.Int)
    slice_register: ir.SSAValue = info.argument(RegisterType)
    register_out: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class DiscardRegister(ir.Statement):
    """Consume a register and free its remaining qubits, including registers with moved-out elements."""

    register: ir.SSAValue = info.argument(RegisterType)


class Operation(ir.Statement):
    """Give the gates, noise channels, measurements and resets one shared view.

    `linear` holds the wires or registers that the operation consumes, and
    `outputs` holds the results that continue them in the same order.
    `parameters` holds the classical operands, and `classical` holds the classical
    results, such as measurement bits.
    """

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the wires or registers that the operation consumes."""
        raise NotImplementedError

    @property
    def parameters(self) -> tuple[ir.SSAValue, ...]:
        """Return the classical operands."""
        return ()

    @property
    def outputs(self) -> tuple[ir.ResultValue, ...]:
        """Return the results that continue `linear` in order."""
        return tuple(self.results[: len(self.linear)])

    @property
    def classical(self) -> tuple[ir.ResultValue, ...]:
        """Return the classical results."""
        return tuple(self.results[len(self.linear) :])

    @property
    def label(self) -> str:
        """Return the kind of a gate or a noise channel, or else the statement name."""
        kind = getattr(self, "kind", None)
        return self.name if kind is None else kind.value


def check_shape(stmt: ir.Statement, facts: Facts, wires: int, params: int) -> None:
    """Raise a validation error if `stmt` has another shape than its kind."""
    if (wires, params) != (facts.wires, facts.params):
        raise ir.ValidationError(
            stmt,
            f"'{stmt.name}' needs {facts.wires} qubits and {facts.params}"
            f" parameters, but it has {wires} and {params}.",
        )


@statement(dialect=dialect, init=False)
class Gate(Operation):
    """Apply the unitary gate `kind` to one wire per qubit of the gate.

    `params` holds the angles in turns. Result `i` continues wire `i`.
    """

    name = "gate"
    traits = frozenset({ThreadsInOrder("wires")})
    wires: tuple[ir.SSAValue, ...] = info.argument(WireType)
    params: tuple[ir.SSAValue, ...] = info.argument()
    kind: GateKind = info.attribute()
    adjoint: bool = info.attribute(default=False)

    def __init__(
        self,
        kind: GateKind,
        wires: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...] = (),
        *,
        adjoint: bool = False,
    ) -> None:
        """Build the gate `kind` on `wires` with the angles `params`."""
        super().__init__(
            args=(*wires, *params),
            result_types=tuple(WireType for _ in wires),
            args_slice={
                "wires": slice(0, len(wires)),
                "params": slice(len(wires), None),
            },
            attributes={"kind": ir.PyAttr(kind), "adjoint": ir.PyAttr(adjoint)},
        )

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the wires of the gate."""
        return tuple(self.wires)

    @property
    def parameters(self) -> tuple[ir.SSAValue, ...]:
        """Return the angles of the gate."""
        return tuple(self.params)

    def verify(self) -> None:
        """Check the number of wires and angles against the kind."""
        super().verify()
        check_shape(self, GATE_FACTS[self.kind], len(self.wires), len(self.params))


@statement(dialect=dialect, init=False)
class Noise(Operation):
    """Apply the noise channel `kind` to one wire per qubit of the channel.

    `params` holds the probabilities. Result `i` continues wire `i`.
    """

    name = "noise"
    traits = frozenset({ThreadsInOrder("wires")})
    wires: tuple[ir.SSAValue, ...] = info.argument(WireType)
    params: tuple[ir.SSAValue, ...] = info.argument()
    kind: NoiseKind = info.attribute()

    def __init__(
        self,
        kind: NoiseKind,
        wires: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...] = (),
    ) -> None:
        """Build the noise channel `kind` on `wires` with the probabilities `params`."""
        super().__init__(
            args=(*wires, *params),
            result_types=tuple(WireType for _ in wires),
            args_slice={
                "wires": slice(0, len(wires)),
                "params": slice(len(wires), None),
            },
            attributes={"kind": ir.PyAttr(kind)},
        )

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the wires of the channel."""
        return tuple(self.wires)

    @property
    def parameters(self) -> tuple[ir.SSAValue, ...]:
        """Return the probabilities of the channel."""
        return tuple(self.params)

    def verify(self) -> None:
        """Check the number of wires and probabilities against the kind."""
        super().verify()
        check_shape(self, NOISE_FACTS[self.kind], len(self.wires), len(self.params))


@statement(dialect=dialect)
class Measure(Operation):
    """Measure the qubit of `wire`, keep it, and return its measurement result."""

    name = "measure"
    traits = frozenset({ThreadsPairs((("wire", "wire_out"),))})
    wire: ir.SSAValue = info.argument(WireType)
    wire_out: ir.ResultValue = info.result(WireType)
    bit: ir.ResultValue = info.result(MeasurementResultType)

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the measured wire."""
        return (self.wire,)


@statement(dialect=dialect)
class Reset(Operation):
    """Reset the qubit of `wire` to the zero state."""

    name = "reset"
    traits = frozenset({ThreadsPairs((("wire", "wire_out"),))})
    wire: ir.SSAValue = info.argument(WireType)
    wire_out: ir.ResultValue = info.result(WireType)

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the reset wire."""
        return (self.wire,)


class RegisterOperation(Operation):
    """Apply a single-wire operation at every position of whole registers.

    Position `i` of each register forms one group of wires. Unless the kind
    truncates, the program must give every register the same length.
    """

    def scalar(self, wires: tuple[ir.SSAValue, ...]) -> Operation:
        """Return the single-wire operation that this statement applies to `wires`."""
        raise NotImplementedError


@statement(dialect=dialect, init=False)
class RegisterGate(RegisterOperation):
    """Apply the gate `kind` at every position of one register per qubit of the gate."""

    name = "register_gate"
    traits = frozenset({ThreadsInOrder("registers")})
    registers: tuple[ir.SSAValue, ...] = info.argument(RegisterType)
    params: tuple[ir.SSAValue, ...] = info.argument()
    kind: GateKind = info.attribute()
    adjoint: bool = info.attribute(default=False)

    def __init__(
        self,
        kind: GateKind,
        registers: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...] = (),
        *,
        adjoint: bool = False,
    ) -> None:
        """Build the gate `kind` on `registers` with the angles `params`."""
        super().__init__(
            args=(*registers, *params),
            result_types=tuple(RegisterType for _ in registers),
            args_slice={
                "registers": slice(0, len(registers)),
                "params": slice(len(registers), None),
            },
            attributes={"kind": ir.PyAttr(kind), "adjoint": ir.PyAttr(adjoint)},
        )

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the registers of the gate."""
        return tuple(self.registers)

    @property
    def parameters(self) -> tuple[ir.SSAValue, ...]:
        """Return the angles of the gate."""
        return tuple(self.params)

    def scalar(self, wires: tuple[ir.SSAValue, ...]) -> Operation:
        """Return the gate on one wire per register."""
        return Gate(self.kind, wires, self.parameters, adjoint=self.adjoint)

    def verify(self) -> None:
        """Check the number of registers and angles against the kind."""
        super().verify()
        facts = GATE_FACTS[self.kind]
        check_shape(self, facts, len(self.registers), len(self.params))


@statement(dialect=dialect, init=False)
class RegisterNoise(RegisterOperation):
    """Apply the noise channel `kind` at every position of whole registers."""

    name = "register_noise"
    traits = frozenset({ThreadsInOrder("registers")})
    registers: tuple[ir.SSAValue, ...] = info.argument(RegisterType)
    params: tuple[ir.SSAValue, ...] = info.argument()
    kind: NoiseKind = info.attribute()

    def __init__(
        self,
        kind: NoiseKind,
        registers: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...] = (),
    ) -> None:
        """Build the noise channel `kind` on `registers` with `params`."""
        super().__init__(
            args=(*registers, *params),
            result_types=tuple(RegisterType for _ in registers),
            args_slice={
                "registers": slice(0, len(registers)),
                "params": slice(len(registers), None),
            },
            attributes={"kind": ir.PyAttr(kind)},
        )

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the registers of the channel."""
        return tuple(self.registers)

    @property
    def parameters(self) -> tuple[ir.SSAValue, ...]:
        """Return the probabilities of the channel."""
        return tuple(self.params)

    def scalar(self, wires: tuple[ir.SSAValue, ...]) -> Operation:
        """Return the noise channel on one wire per register."""
        return Noise(self.kind, wires, self.parameters)

    def verify(self) -> None:
        """Check the number of registers and probabilities against the kind."""
        super().verify()
        facts = NOISE_FACTS[self.kind]
        check_shape(self, facts, len(self.registers), len(self.params))


@statement(dialect=dialect)
class RegisterMeasure(RegisterOperation):
    """Measure every qubit of `register` and return the results in order."""

    name = "register_measure"
    traits = frozenset({ThreadsPairs((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    register_out: ir.ResultValue = info.result(RegisterType)
    bits: ir.ResultValue = info.result(
        ilist.IListType[MeasurementResultType, types.Any]
    )

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the measured register."""
        return (self.register,)

    def scalar(self, wires: tuple[ir.SSAValue, ...]) -> Operation:
        """Return the measurement of one wire."""
        (wire,) = wires
        return Measure(wire)


@statement(dialect=dialect)
class RegisterReset(RegisterOperation):
    """Reset every qubit of `register` to the zero state."""

    name = "register_reset"
    traits = frozenset({ThreadsPairs((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    register_out: ir.ResultValue = info.result(RegisterType)

    @property
    def linear(self) -> tuple[ir.SSAValue, ...]:
        """Return the reset register."""
        return (self.register,)

    def scalar(self, wires: tuple[ir.SSAValue, ...]) -> Operation:
        """Return the reset of one wire."""
        (wire,) = wires
        return Reset(wire)


def truncates(stmt: Operation) -> bool:
    """Check whether the register form of `stmt` stops at the shortest register."""
    if isinstance(stmt, (Gate, RegisterGate)):
        return GATE_FACTS[stmt.kind].truncate
    if isinstance(stmt, (Noise, RegisterNoise)):
        return NOISE_FACTS[stmt.kind].truncate
    return False
