"""This module holds the reference analysis for squin kernels."""

from dataclasses import dataclass

from kirin import ir, types
from kirin.dialects import ilist

from bloqade import squin
from bloqade.types import QubitType
from bloqade.constants import constant_int
from bloqade.analysis.reference import (
    KEY,
    Ref,
    Returned,
    ReferenceAnalysis,
    items_of,
    argument_of,
)


@dataclass
class QubitReferenceAnalysis(ReferenceAnalysis):
    """A reference analysis whose roots are squin qubits and registers.

    `qubit.new` allocates a qubit root, and `squin.qalloc` allocates a register root.
    """

    keys = (KEY, "absint")

    def shape(self, type_: types.TypeAttribute) -> int | None:
        """Return 0 for a qubit, 1 more for each list level around it, else None.

        So a list of qubits has the shape 1, and a list of lists of qubits has 2.
        """
        if isinstance(type_, types.BottomType):
            return None
        if type_.is_subseteq(QubitType):
            return 0
        if type_.is_subseteq(ilist.IListType) and isinstance(type_, types.Generic):
            inner = self.shape(type_.vars[0])
            return None if inner is None else inner + 1
        return None

    def register_length(
        self, root: ir.SSAValue, calls: tuple[Returned, ...]
    ) -> int | None:
        """Return the static length of the register root `root`, or None."""
        return qubit_register_length(root, calls)


def qubit_register_length(root: ir.SSAValue, calls: tuple[Returned, ...]) -> int | None:
    """Return the static length of the squin register root `root`, or None.

    A register that `squin.qalloc` returns has the constant size of the call,
    and a negative size allocates no qubit. If the size is a parameter of a
    callee, the size is the constant that the caller passes. A parameter has the
    length in its type.
    """
    if calls and calls[0].callee is squin.qalloc:
        size = constant_int(argument_of(calls[0].call.args[0], calls[1:]))
        return None if size is None else max(0, size)
    kind = root.type
    length = kind.vars[1] if isinstance(kind, types.Generic) else None
    if isinstance(length, types.Literal) and isinstance(length.data, int):
        return length.data
    return None


def qubit_items(ref: Ref) -> tuple[Ref, ...] | None:
    """Return the qubits that the squin reference `ref` holds, or None if unknown.

    One qubit is itself. A literal list holds its members. A register of static
    length holds one slot per index.
    """
    return items_of(ref, qubit_register_length)
