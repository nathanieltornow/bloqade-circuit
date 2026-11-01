"""Execute small wire programs with dense matrices for independent semantic tests.

The interpreter keeps allocated qubits in allocation order, including freed qubits.
Tests supply measurement outcomes and multiply their exact conditional probabilities.
Noise channels act directly on the density matrix and need no random samples.
"""

from dataclasses import field, dataclass

import numpy as np
from kirin import ir, interp
from kirin.dialects import ilist

from bloqade.squin.wire import stmts, twins
from bloqade.squin.wire.traits import Twin, RegisterTwin

IDENTITY = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


class ImpossibleOutcome(BaseException):
    """Stop a replay whose measurement outcome has zero probability."""


@dataclass
class DensityInterpreter(interp.Interpreter):
    """Evaluate wire operations without reconstructing source quantum operations."""

    keys = ("wire.test.density", "main")
    rho: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=complex))
    outcomes: tuple[int, ...] = ()
    measurements: int = 0
    probability: float = 1.0
    qubits: int = 0

    def allocate(self):
        qubit = self.qubits
        self.qubits += 1
        self.rho = np.kron(self.rho, np.diag([1, 0]))
        return qubit

    def embed(self, matrix, operands):
        """Place a local matrix on the given qubits using allocation order."""
        size = 1 << self.qubits
        full = np.zeros((size, size), dtype=complex)
        masks = [1 << (self.qubits - qubit - 1) for qubit in operands]
        for column in range(size):
            local = 0
            for mask in masks:
                local = (local << 1) | bool(column & mask)
            for row in range(1 << len(operands)):
                target = column
                for bit, mask in enumerate(masks):
                    target = (
                        target | mask
                        if row & (1 << (len(masks) - bit - 1))
                        else target & ~mask
                    )
                full[target, column] = matrix[row, local]
        return full

    def apply(self, matrix, operands):
        full = self.embed(matrix, operands)
        self.rho = full @ self.rho @ full.conj().T

    def channel(self, matrices, operands):
        before = self.rho.copy()
        result = np.zeros_like(before)
        for matrix in matrices:
            full = self.embed(matrix, operands)
            result += full @ before @ full.conj().T
        self.rho = result


@dataclass(frozen=True)
class Hole:
    """Mark the register slot that lent `qubit` to an outstanding borrow."""

    qubit: int


@stmts.dialect.register(key="wire.test.density")
class WireSemantics(interp.MethodTable):
    """Implement the tested wire operations with independent matrix definitions."""

    @interp.impl(stmts.Alloc)
    def alloc(self, sim, frame, stmt):
        return (sim.allocate(),)

    @interp.impl(stmts.AllocRegister)
    def register(self, sim, frame, stmt):
        return (tuple(sim.allocate() for _ in range(frame.get(stmt.size))),)

    @interp.impl(stmts.Free)
    @interp.impl(stmts.FreeRegister)
    def free(self, sim, frame, stmt):
        return ()

    @interp.impl(stmts.Borrow)
    def borrow(self, sim, frame, stmt):
        register, index = frame.get_values(stmt.args)
        assert (
            0 <= index < len(register)
        ), f"borrow index {index} is outside the register"
        qubit = register[index]
        assert not isinstance(qubit, Hole), f"slot {index} is already borrowed"
        return register[:index] + (Hole(qubit),) + register[index + 1 :], qubit

    @interp.impl(stmts.Release)
    def release(self, sim, frame, stmt):
        register, index, qubit = frame.get_values(stmt.args)
        assert (
            0 <= index < len(register)
        ), f"release index {index} is outside the register"
        assert register[index] == Hole(qubit), f"slot {index} did not lend this qubit"
        return (register[:index] + (qubit,) + register[index + 1 :],)

    @interp.impl(stmts.Length)
    def length(self, sim, frame, stmt):
        register = frame.get(stmt.register)
        return register, len(register)

    @interp.impl(stmts.TakeSlice)
    def take_slice(self, sim, frame, stmt):
        register, start, size = frame.get_values(stmt.args)
        part = register[start : start + size]
        assert len(part) == size, "the slice runs past the end of the register"
        assert not any(isinstance(q, Hole) for q in part), "the slice holds a hole"
        holes = tuple(Hole(qubit) for qubit in part)
        return register[:start] + holes + register[start + size :], part

    @interp.impl(stmts.PutSlice)
    def put_slice(self, sim, frame, stmt):
        register, start, part = frame.get_values(stmt.args)
        holes = register[start : start + len(part)]
        assert holes == tuple(Hole(q) for q in part), "the slice did not come from here"
        return (register[:start] + part + register[start + len(part) :],)

    @interp.impl(twins.X)
    @interp.impl(twins.Y)
    @interp.impl(twins.Z)
    @interp.impl(twins.H)
    @interp.impl(twins.S)
    @interp.impl(twins.T)
    @interp.impl(twins.SqrtX)
    @interp.impl(twins.SqrtY)
    def single(self, sim, frame, stmt):
        matrices = {
            twins.X: X,
            twins.Y: Y,
            twins.Z: Z,
            twins.H: H,
            twins.S: np.diag([1, 1j]),
            twins.T: np.diag([1, np.exp(1j * np.pi / 4)]),
            twins.SqrtX: (IDENTITY - 1j * X) / np.sqrt(2),
            twins.SqrtY: (IDENTITY - 1j * Y) / np.sqrt(2),
        }
        matrix = matrices[type(stmt)]
        if getattr(stmt, "adjoint", False):
            matrix = matrix.conj().T
        qubit = frame.get(stmt.qubit)
        sim.apply(matrix, (qubit,))
        return (qubit,)

    @interp.impl(twins.Rx)
    @interp.impl(twins.Ry)
    @interp.impl(twins.Rz)
    def rotation(self, sim, frame, stmt):
        angle, qubit = frame.get_values(stmt.args)
        pauli = {twins.Rx: X, twins.Ry: Y, twins.Rz: Z}[type(stmt)]
        matrix = np.cos(np.pi * angle) * IDENTITY - 1j * np.sin(np.pi * angle) * pauli
        sim.apply(matrix, (qubit,))
        return (qubit,)

    @interp.impl(twins.CX)
    @interp.impl(twins.CY)
    @interp.impl(twins.CZ)
    @interp.impl(twins.Swap)
    def pair(self, sim, frame, stmt):
        operands = frame.get_values(stmt.args)
        if isinstance(stmt, twins.Swap):
            matrix = np.eye(4)[[0, 2, 1, 3]]
        else:
            matrix = np.eye(4, dtype=complex)
            matrix[2:, 2:] = {twins.CX: X, twins.CY: Y, twins.CZ: Z}[type(stmt)]
        sim.apply(matrix, operands)
        return operands

    @interp.impl(twins.Measure)
    def measure(self, sim, frame, stmt):
        qubit = frame.get(stmt.qubit)
        outcome = sim.outcomes[sim.measurements]
        sim.measurements += 1
        sim.apply(np.diag([1 - outcome, outcome]), (qubit,))
        weight = float(np.trace(sim.rho).real)
        if weight < 1e-14:
            raise ImpossibleOutcome()
        sim.probability *= weight
        sim.rho /= weight
        return qubit, outcome

    @interp.impl(twins.Reset)
    def reset(self, sim, frame, stmt):
        qubit = frame.get(stmt.qubit)
        sim.channel((np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])), (qubit,))
        return (qubit,)

    @interp.impl(twins.Depolarize)
    def depolarize(self, sim, frame, stmt):
        probability, qubit = frame.get_values(stmt.args)
        sim.channel(
            (
                np.sqrt(1 - probability) * IDENTITY,
                *(np.sqrt(probability / 3) * p for p in (X, Y, Z)),
            ),
            (qubit,),
        )
        return (qubit,)

    @interp.impl(twins.SingleQubitPauliChannel)
    def pauli(self, sim, frame, stmt):
        px, py, pz, qubit = frame.get_values(stmt.args)
        sim.channel(
            (
                np.sqrt(1 - px - py - pz) * IDENTITY,
                np.sqrt(px) * X,
                np.sqrt(py) * Y,
                np.sqrt(pz) * Z,
            ),
            (qubit,),
        )
        return (qubit,)

    def broadcast(self, sim, frame, stmt):
        """Apply the scalar twin of a register twin to each register position."""
        trait = stmt.get_present_trait(RegisterTwin)
        scalar = trait.scalar
        registers = {field: frame.get(getattr(stmt, field)) for _, field in trait.wires}
        lengths = {len(register) for register in registers.values()}
        assert trait.truncate or len(lengths) == 1, "the registers differ in length"
        count = min(lengths)
        fields = getattr(scalar, "__kirin_stmt_fields")
        register_of = {
            wire: source for source, wire in scalar.get_present_trait(Twin).wires
        }
        names = list(fields.std_args)
        probe = scalar(
            **{name: ir.TestValue(fields.std_args[name].type) for name in names},
            **{name: getattr(stmt, name) for name in fields.attributes},
        )
        collected = [[] for _ in trait.results]
        for i in range(count):
            values = [
                (
                    registers[register_of[name]][i]
                    if name in register_of
                    else frame.get(getattr(stmt, name))
                )
                for name in names
            ]
            inner = sim.initialize_frame(probe)
            inner.set_values(probe.args, values)
            outputs = sim.lookup_registry(inner, probe)(sim, inner, probe)
            for results, value in zip(collected, outputs[len(registers) :]):
                results.append(value)
        return (*registers.values(), *(ilist.IList(values) for values in collected))

    for _register_twin in twins.REGISTER_TWINS.values():
        broadcast = interp.impl(_register_twin)(broadcast)
    del _register_twin
