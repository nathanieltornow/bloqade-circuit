"""This module defines the linear types of the squin wire dialect.

A `Wire` stands for one qubit at one point in the program. A statement consumes
its input wires and returns new wires, so each wire has exactly one use.

A `Register` stands for a list of qubits. `Borrow` takes one qubit out of a
register as a wire, and `Release` puts the wire back. A register type can carry a
static number of qubits, such as `Register[3]`.
"""

import math

from kirin import types
from kirin.dialects import ilist

from bloqade.types import QubitType, is_subtype


class Wire:
    """A marker class that names the linear type of one qubit."""


class Register:
    """A marker class that names the linear type of a list of qubits."""


WireType = types.PyClass(Wire)
RegisterType = types.PyClass(Register)


def register_type(length: int | None) -> types.TypeAttribute:
    """Return the type of a register of `length` qubits.

    The result is `RegisterType` if the length is None.
    """
    if length is None:
        return RegisterType
    return types.Generic(Register, types.Literal(length))


def register_length(typ: types.TypeAttribute) -> int | None:
    """Return the static number of qubits of a register type, or None."""
    if isinstance(typ, types.Generic) and is_subtype(typ, RegisterType) and typ.vars:
        length = typ.vars[0]
        if isinstance(length, types.Literal) and isinstance(length.data, int):
            return length.data
    return None


def is_linear(typ: types.TypeAttribute) -> bool:
    """Return True if a value of the type `typ` must have exactly one use."""
    if isinstance(typ, types.BottomType):
        return False
    return is_subtype(typ, WireType) or is_subtype(typ, RegisterType)


def contains_linear(typ: types.TypeAttribute) -> bool:
    """Return whether the type contains a wire or register at any depth."""
    if is_linear(typ):
        return True
    if isinstance(typ, types.Union):
        return any(contains_linear(member) for member in typ.types)
    if isinstance(typ, types.TypeVar):
        return contains_linear(typ.bound)
    if isinstance(typ, types.Generic):
        return any(contains_linear(member) for member in typ.vars) or (
            typ.vararg is not None and contains_linear(typ.vararg.typ)
        )
    return False


def linear_paths(typ: types.TypeAttribute) -> tuple[tuple[int, ...], ...]:
    """Return the tuple paths of every wire and register in a supported type.

    Quantum containers must have fixed tuple fields so each field has an owner.
    """
    if is_linear(typ):
        return ((),)
    if not contains_linear(typ):
        return ()
    if (
        isinstance(typ, types.Generic)
        and is_subtype(typ, types.Tuple)
        and typ.vararg is None
    ):
        return tuple(
            (index, *path)
            for index, member in enumerate(typ.vars)
            for path in linear_paths(member)
        )
    raise ValueError("Quantum containers must use fixed tuple fields.")


def quantum_type(typ: types.TypeAttribute) -> types.TypeAttribute | None:
    """Return the linear type of a scalar qubit or a nested qubit register."""
    if isinstance(typ, types.BottomType):
        return None
    if is_subtype(typ, QubitType) or is_subtype(typ, WireType):
        return WireType
    if is_subtype(typ, RegisterType):
        return RegisterType
    if isinstance(typ, types.Generic) and is_subtype(typ, ilist.IListType):
        if quantum_type(typ.vars[0]) is not None:
            return RegisterType
    return None


def typed_length(typ: types.TypeAttribute) -> int | None:
    """Return a list's declared constant length, if its type supplies one."""
    if isinstance(typ, types.Generic) and is_subtype(typ, ilist.IListType):
        length = typ.vars[1]
        if isinstance(length, types.Literal) and isinstance(length.data, int):
            return length.data
    return None


def register_shape(typ: types.TypeAttribute) -> tuple[int | None, ...]:
    """Return the source register dimensions in outer-to-inner order."""
    shape = []
    while isinstance(typ, types.Generic) and is_subtype(typ, ilist.IListType):
        shape.append(typed_length(typ))
        typ = typ.vars[0]
    return tuple(shape)


def parameter_type(typ: types.TypeAttribute) -> types.TypeAttribute:
    """Return the wire type of a function parameter of the source type `typ`.

    A register keeps its number of qubits if the source type gives every
    dimension. For example, `IList[IList[Qubit, 2], 3]` becomes `Register[6]`.
    """
    linear = quantum_type(typ)
    if linear is None:
        return typ
    shape = register_shape(typ)
    if linear == RegisterType and shape and all(size is not None for size in shape):
        return register_type(math.prod(size for size in shape if size is not None))
    return linear
