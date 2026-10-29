"""This module holds the traits of the squin wire dialect.

`Threads` tells which result of a statement continues which linear operand. A
rewrite rule reads it to follow one qubit from statement to statement.

`Twin` marks a statement that the twin generator builds from a squin statement.
The kind traits `Unitary`, `Noise`, `Measurement` and `Preparation` tell rewrite
rules what the statement does to its qubits.
"""

from dataclasses import dataclass

from kirin import ir


@dataclass(frozen=True)
class Threads(ir.StmtTrait):
    """Name the pairs of a linear operand and the result that continues it.

    Each pair holds the name of an argument field and the name of a result
    field. For example, `Release` has the pair `("register", "register_out")`.
    """

    pairs: tuple[tuple[str, str], ...]

    def threads(self, stmt: ir.Statement) -> list[tuple[ir.SSAValue, ir.ResultValue]]:
        """Return each linear operand of `stmt` with the result that continues it."""
        return [(getattr(stmt, a), getattr(stmt, r)) for a, r in self.pairs]

    def successor(self, stmt: ir.Statement, operand: ir.SSAValue) -> ir.ResultValue:
        """Return the result of `stmt` that continues the linear operand `operand`."""
        for value, result in self.threads(stmt):
            if value is operand:
                return result
        raise ValueError(f"{operand} is no linear operand of {stmt.name}")


@dataclass(frozen=True)
class Twin(ir.StmtTrait):
    """Link a wire statement to the squin statement that it copies.

    `source` is the squin statement class. `wires` maps each qubit-list field of
    `source` to the wire field of the twin, such as `controls` to `control`.
    `results` maps each list result of `source` to the scalar result of the twin,
    such as `result` of `qubit.Measure` to `result` of its twin.
    If `truncate` is True, the source broadcast stops at the shortest qubit list.
    """

    source: type[ir.Statement]
    wires: tuple[tuple[str, str], ...]
    results: tuple[tuple[str, str], ...] = ()
    truncate: bool = False


@dataclass(frozen=True)
class Unitary(ir.StmtTrait):
    """Mark a twin that applies a unitary gate to its wires."""


@dataclass(frozen=True)
class Noise(ir.StmtTrait):
    """Mark a twin that applies a noise channel to its wires.

    A rewrite rule must not move a unitary gate across a noise channel.
    """


@dataclass(frozen=True)
class Measurement(ir.StmtTrait):
    """Mark a twin that measures its wires and keeps the qubits."""


@dataclass(frozen=True)
class Preparation(ir.StmtTrait):
    """Mark a twin that resets its wires to the zero state."""
