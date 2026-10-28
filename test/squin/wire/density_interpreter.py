"""Execute small wire programs with dense matrices for independent semantic tests.

The interpreter keeps allocated qubits in allocation order, including freed qubits.
Tests supply measurement outcomes and multiply their exact conditional probabilities.
Noise channels act directly on the density matrix and need no random samples.
"""

from dataclasses import field, dataclass

import numpy as np
from kirin import interp

from bloqade.squin.wire import stmts, twins

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
        return register, register[index]

    @interp.impl(stmts.Release)
    def release(self, sim, frame, stmt):
        register, index, qubit = frame.get_values(stmt.args)
        assert register[index] == qubit
        return (register,)

    @interp.impl(stmts.Length)
    def length(self, sim, frame, stmt):
        register = frame.get(stmt.register)
        return register, len(register)

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
