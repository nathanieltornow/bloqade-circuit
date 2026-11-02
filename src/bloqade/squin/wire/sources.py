"""Map each squin statement on qubits to the wire statements that the conversions use.

Only the conversions between squin and wire read this table. The wire dialect names
its operations with the kinds of `kinds.py` and imports nothing from squin. An entry
names the qubit-list fields, the classical fields and the classical results of its
squin statement in operand order. `check` compares each entry with the squin schema
when this module loads.
"""

from __future__ import annotations

from typing import TypeAlias, cast
from dataclasses import dataclass
from collections.abc import Callable

from kirin import ir, types
from kirin.dialects import ilist

from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import QubitType, holds_qubits
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.noise import stmts as noise_stmts
from bloqade.native.dialects.gate import stmts as native_stmts

from .kinds import GATE_FACTS, NOISE_FACTS, Facts, GateKind, NoiseKind
from .stmts import (
    Gate,
    Noise,
    Reset,
    Measure,
    Operation,
    RegisterGate,
    RegisterNoise,
    RegisterReset,
    RegisterMeasure,
    RegisterOperation,
)

ONE = ("qubits",)
PAIR = ("controls", "targets")


@dataclass(frozen=True)
class Entry:
    """Name the fields of one squin statement on qubits, in operand order."""

    source: type[ir.Statement]
    """The squin statement."""
    qubits: tuple[str, ...]
    """The qubit-list fields of the squin statement."""
    params: tuple[str, ...] = ()
    """The classical fields of the squin statement."""
    results: tuple[str, ...] = ()
    """The classical list results of the squin statement."""

    def qubit_operands(self, stmt: ir.Statement) -> tuple[ir.SSAValue, ...]:
        """Return the qubit-list operands of the squin statement `stmt`."""
        return tuple(getattr(stmt, name) for name in self.qubits)

    def param_operands(self, stmt: ir.Statement) -> tuple[ir.SSAValue, ...]:
        """Return the classical operands of the squin statement `stmt`."""
        return tuple(getattr(stmt, name) for name in self.params)

    def result_values(self, stmt: ir.Statement) -> tuple[ir.ResultValue, ...]:
        """Return the classical results of the squin statement `stmt`."""
        return tuple(getattr(stmt, name) for name in self.results)

    @property
    def has_adjoint(self) -> bool:
        """Check whether the squin statement has the attribute `adjoint`."""
        return "adjoint" in getattr(self.source, "__kirin_stmt_fields").attributes

    def adjoint(self, stmt: ir.Statement) -> bool:
        """Return the value of the attribute `adjoint` of the squin statement `stmt`."""
        return self.has_adjoint and getattr(stmt, "adjoint")

    def build(
        self,
        qubits: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool = False,
    ) -> ir.Statement:
        """Build the squin statement on the qubit lists `qubits`."""
        fields: dict[str, ir.SSAValue | bool] = {
            **dict(zip(self.qubits, qubits)),
            **dict(zip(self.params, params)),
        }
        if self.has_adjoint:
            fields["adjoint"] = adjoint
        return cast(Callable[..., ir.Statement], self.source)(**fields)


@dataclass(frozen=True)
class GateEntry(Entry):
    """Map a squin gate to the wire gate of the kind `kind`."""

    kind: GateKind = GateKind.X

    @property
    def facts(self) -> Facts:
        """Return the shape of the gate."""
        return GATE_FACTS[self.kind]

    def scalar(
        self,
        wires: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> Operation:
        """Return the gate on one wire per qubit list."""
        return Gate(self.kind, wires, params, adjoint=adjoint)

    def register(
        self,
        registers: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> RegisterOperation:
        """Return the gate on one register per qubit list."""
        return RegisterGate(self.kind, registers, params, adjoint=adjoint)


@dataclass(frozen=True)
class NoiseEntry(Entry):
    """Map a squin noise channel to the wire channel of the kind `kind`."""

    kind: NoiseKind = NoiseKind.DEPOLARIZE

    @property
    def facts(self) -> Facts:
        """Return the shape of the channel."""
        return NOISE_FACTS[self.kind]

    def scalar(
        self,
        wires: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> Operation:
        """Return the channel on one wire per qubit list."""
        return Noise(self.kind, wires, params)

    def register(
        self,
        registers: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> RegisterOperation:
        """Return the channel on one register per qubit list."""
        return RegisterNoise(self.kind, registers, params)


@dataclass(frozen=True)
class MeasureEntry(Entry):
    """Map the squin measurement to `Measure` and `RegisterMeasure`."""

    @property
    def facts(self) -> Facts:
        """Return the shape of a measurement."""
        return Facts(1)

    def scalar(
        self,
        wires: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> Operation:
        """Return the measurement of one wire."""
        (wire,) = wires
        return Measure(wire)

    def register(
        self,
        registers: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> RegisterOperation:
        """Return the measurement of one register."""
        (register,) = registers
        return RegisterMeasure(register)


@dataclass(frozen=True)
class ResetEntry(Entry):
    """Map the squin reset to `Reset` and `RegisterReset`."""

    @property
    def facts(self) -> Facts:
        """Return the shape of a reset."""
        return Facts(1)

    def scalar(
        self,
        wires: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> Operation:
        """Return the reset of one wire."""
        (wire,) = wires
        return Reset(wire)

    def register(
        self,
        registers: tuple[ir.SSAValue, ...],
        params: tuple[ir.SSAValue, ...],
        adjoint: bool,
    ) -> RegisterOperation:
        """Return the reset of one register."""
        (register,) = registers
        return RegisterReset(register)


Source: TypeAlias = GateEntry | NoiseEntry | MeasureEntry | ResetEntry
"""Map one squin statement on qubits to its wire statements."""

GATES = (
    GateEntry(gate_stmts.X, ONE, kind=GateKind.X),
    GateEntry(gate_stmts.Y, ONE, kind=GateKind.Y),
    GateEntry(gate_stmts.Z, ONE, kind=GateKind.Z),
    GateEntry(gate_stmts.H, ONE, kind=GateKind.H),
    GateEntry(gate_stmts.T, ONE, kind=GateKind.T),
    GateEntry(gate_stmts.S, ONE, kind=GateKind.S),
    GateEntry(gate_stmts.SqrtX, ONE, kind=GateKind.SQRT_X),
    GateEntry(gate_stmts.SqrtY, ONE, kind=GateKind.SQRT_Y),
    GateEntry(gate_stmts.Rx, ONE, ("angle",), kind=GateKind.RX),
    GateEntry(gate_stmts.Ry, ONE, ("angle",), kind=GateKind.RY),
    GateEntry(gate_stmts.Rz, ONE, ("angle",), kind=GateKind.RZ),
    GateEntry(gate_stmts.CX, PAIR, kind=GateKind.CX),
    GateEntry(gate_stmts.CY, PAIR, kind=GateKind.CY),
    GateEntry(gate_stmts.CZ, PAIR, kind=GateKind.CZ),
    GateEntry(gate_stmts.CCZ, ("controls1", "controls2", "targets"), kind=GateKind.CCZ),
    GateEntry(gate_stmts.Swap, ("qubits1", "qubits2"), kind=GateKind.SWAP),
    GateEntry(gate_stmts.U3, ONE, ("theta", "phi", "lam"), kind=GateKind.U3),
    GateEntry(
        gate_stmts.PhasedXZ,
        ONE,
        ("x_exponent", "z_exponent", "axis_phase_exponent"),
        kind=GateKind.PHASED_XZ,
    ),
    GateEntry(native_stmts.CZ, PAIR, kind=GateKind.NATIVE_CZ),
    GateEntry(
        native_stmts.R, ONE, ("axis_angle", "rotation_angle"), kind=GateKind.NATIVE_R
    ),
    GateEntry(native_stmts.Rz, ONE, ("rotation_angle",), kind=GateKind.NATIVE_RZ),
)
"""Map each squin and native gate to its wire gate."""

NOISES = (
    NoiseEntry(
        noise_stmts.SingleQubitPauliChannel,
        ONE,
        ("px", "py", "pz"),
        kind=NoiseKind.SINGLE_QUBIT_PAULI_CHANNEL,
    ),
    NoiseEntry(
        noise_stmts.TwoQubitPauliChannel,
        PAIR,
        ("probabilities",),
        kind=NoiseKind.TWO_QUBIT_PAULI_CHANNEL,
    ),
    NoiseEntry(noise_stmts.Depolarize, ONE, ("p",), kind=NoiseKind.DEPOLARIZE),
    NoiseEntry(noise_stmts.Depolarize2, PAIR, ("p",), kind=NoiseKind.DEPOLARIZE2),
    NoiseEntry(noise_stmts.QubitLoss, ONE, ("p",), kind=NoiseKind.QUBIT_LOSS),
    NoiseEntry(
        noise_stmts.QubitLeakage, ONE, ("p0", "p1"), kind=NoiseKind.QUBIT_LEAKAGE
    ),
)
"""Map each squin noise channel on flat qubit lists to its wire channel.

`CorrelatedQubitLoss` takes nested qubit lists and has no entry.
"""

MEASURE = MeasureEntry(qubit_stmts.Measure, ONE, results=("result",))
RESET = ResetEntry(qubit_stmts.Reset, ONE)


def flat_qubits(kind: types.TypeAttribute) -> bool:
    """Check whether `kind` is a flat list of qubits, such as `IList[Qubit, N]`."""
    return (
        isinstance(kind, types.Generic)
        and kind.is_subseteq(ilist.IListType)
        and kind.vars[0].is_subseteq(QubitType)
        and not isinstance(kind.vars[0], types.BottomType)
    )


def check(entry: Source) -> Source:
    """Return `entry` after a check against the schema of its squin statement.

    The qubit-list fields, the classical fields and the list results must appear in
    the squin statement in the listed order, and their counts must match the kind.
    Each qubit-list field must hold a flat list of qubits.
    """
    fields = getattr(entry.source, "__kirin_stmt_fields")
    arguments = {**fields.std_args, **fields.kw_args}
    quantum = tuple(name for name, f in arguments.items() if holds_qubits(f.type))
    classical = tuple(name for name, f in arguments.items() if name not in quantum)
    name = entry.source.__name__
    if quantum != entry.qubits or classical != entry.params:
        raise ValueError(f"The wire entry of {name} must list its fields in order.")
    if not all(flat_qubits(arguments[field].type) for field in quantum):
        raise ValueError(f"The wire entry of {name} must take flat qubit lists.")
    if tuple(fields.results) != entry.results:
        raise ValueError(f"The wire entry of {name} must list its results in order.")
    if (len(entry.qubits), len(entry.params)) != (
        entry.facts.wires,
        entry.facts.params,
    ):
        raise ValueError(f"The wire entry of {name} must match the shape of its kind.")
    return entry


SOURCES: dict[type[ir.Statement], Source] = {
    entry.source: check(entry) for entry in (*GATES, *NOISES, MEASURE, RESET)
}
"""Map each squin statement class to its entry."""

GATE_SOURCES: dict[GateKind, GateEntry] = {entry.kind: entry for entry in GATES}
"""Map each gate kind back to the squin statement that it copies."""

NOISE_SOURCES: dict[NoiseKind, NoiseEntry] = {entry.kind: entry for entry in NOISES}
"""Map each noise kind back to the squin statement that it copies."""


def source_of(stmt: Operation) -> Source:
    """Return the entry of the squin statement that the wire statement `stmt` copies."""
    if isinstance(stmt, (Gate, RegisterGate)):
        return GATE_SOURCES[stmt.kind]
    if isinstance(stmt, (Noise, RegisterNoise)):
        return NOISE_SOURCES[stmt.kind]
    if isinstance(stmt, (Measure, RegisterMeasure)):
        return MEASURE
    return RESET
