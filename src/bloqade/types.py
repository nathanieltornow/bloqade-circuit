"""Bloqade types.

This module defines the basic types used in Bloqade eDSLs.
"""

from abc import ABC
from typing import Any

from kirin import types
from kirin.dialects.ilist import IList, IListType

from bloqade.decoders.dialects.annotate.types import (
    MeasurementResult as MeasurementResult,
    MeasurementResultType as MeasurementResultType,
)


class Qubit(ABC):
    """Runtime representation of a qubit.

    Note:
        This is the base class of more specific qubit types, such as
        a reference to a piece of quantum register in some quantum register
        dialects.
    """

    pass


Register = IList[Qubit, Any]
"""Runtime representation of a qubit register."""

QubitType = types.PyClass(Qubit)
"""Kirin type for a qubit."""


def is_subtype(typ: types.TypeAttribute, expected: types.TypeAttribute) -> bool:
    """Return True if `typ` is a subtype of `expected`.

    Kirin's `is_subseteq` also holds for `Bottom`, the type of a value that type
    inference could not reach. This check treats `Bottom` as a subtype of nothing,
    so a failed type inference fails the check.
    """
    return not isinstance(typ, types.BottomType) and typ.is_subseteq(expected)


def item_type(typ: types.TypeAttribute) -> types.TypeAttribute:
    """Return the item type of the list type `typ`, or `Any` for any other type."""
    if isinstance(typ, types.Generic) and is_subtype(typ, IListType):
        return typ.vars[0]
    return types.Any


def holds_qubits(typ: types.TypeAttribute) -> bool:
    """Return True if the type `typ` holds qubits, such as `IList[Qubit, 3]`."""
    if isinstance(typ, types.BottomType):
        return False
    if is_subtype(typ, QubitType):
        return True
    if isinstance(typ, types.Union):
        return any(holds_qubits(member) for member in typ.types)
    if isinstance(typ, types.TypeVar):
        return holds_qubits(typ.bound)
    if isinstance(typ, types.Generic):
        # The second variable of an `IList` is its length, which holds no qubit.
        members = typ.vars[:1] if is_subtype(typ, IListType) else typ.vars
        return any(holds_qubits(member) for member in members) or (
            typ.vararg is not None and holds_qubits(typ.vararg.typ)
        )
    return False


def output_types(output: types.TypeAttribute) -> tuple[types.TypeAttribute, ...]:
    """Return the type of each value that a function of the output type `output` returns.

    For example, `tuple[Wire, bool]` gives `(Wire, bool)` and `None` gives `()`. A
    literal tuple gives one literal type per item. Any other type, such as a tuple of
    any length, gives itself.
    """
    if is_subtype(output, types.NoneType):
        return ()
    if (
        isinstance(output, types.Generic)
        and is_subtype(output, types.Tuple)
        and output.vararg is None
    ):
        return tuple(output.vars)
    if isinstance(output, types.Literal) and isinstance(output.data, tuple):
        return tuple(types.Literal(item) for item in output.data)
    return (output,)
