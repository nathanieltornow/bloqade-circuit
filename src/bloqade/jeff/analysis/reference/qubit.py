"""This module holds the reference analysis for squin qubits."""

from typing import TypeGuard
from dataclasses import dataclass

from kirin import ir, types, interp
from kirin.dialects import func, ilist
from kirin.analysis.forward import ForwardFrame

from bloqade import squin
from bloqade.qubit import stmts as qubit_stmts, dialect as qubit_dialect
from bloqade.types import QubitType
from bloqade.jeff.constants import const_int

from .lattice import Ref, Whole
from .analysis import KEY, ReferenceAnalysis


def is_allocator(stmt: ir.Statement | None) -> TypeGuard[func.Invoke]:
    """Return True if `stmt` calls `squin.qalloc`, which allocates a register."""
    return isinstance(stmt, func.Invoke) and stmt.callee is squin.qalloc


QUBIT_KEY = "qubit.reference"
"""The registry key of the rules for squin qubit statements."""


@dataclass
class QubitReferenceAnalysis(ReferenceAnalysis):
    """A reference analysis whose roots are squin qubits and registers.

    `squin.qalloc` allocates a register root, and `qubit.new` allocates a qubit root.
    """

    keys = (QUBIT_KEY, KEY)

    def is_tracked_type(self, type_: types.TypeAttribute) -> bool:
        """Return True for a qubit and for a list of qubits."""
        return type_.is_subseteq(QubitType) or type_.is_subseteq(
            ilist.IListType[QubitType, types.Any]
        )

    def is_allocating_call(self, call: func.Invoke) -> bool:
        """Return True if `call` calls `squin.qalloc`."""
        return is_allocator(call)

    def register_length(self, value: ir.SSAValue) -> int | None:
        """Return the static length of the register root `value`, or None.

        The length comes from the constant size of a `squin.qalloc` call or from the
        length in the type of a parameter. A negative size allocates no qubit.
        """
        owner = value.owner if isinstance(value, ir.ResultValue) else None
        if is_allocator(owner):
            size = const_int(owner.inputs[0])
            return None if size is None else max(0, size)
        kind = value.type
        length = kind.vars[1] if isinstance(kind, types.Generic) else None
        if isinstance(length, types.Literal) and isinstance(length.data, int):
            return length.data
        return None


@qubit_dialect.register(key=QUBIT_KEY)
class _Qubit(interp.MethodTable):
    """A method table that makes each new qubit a root."""

    @interp.impl(qubit_stmts.New)
    def new(
        self,
        analysis: QubitReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: qubit_stmts.New,
    ) -> tuple[Ref, ...]:
        """Return a reference to the new qubit as a whole root."""
        return (Whole(stmt.result),)
