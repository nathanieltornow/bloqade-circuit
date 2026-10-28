"""This module defines the linear types of the squin wire dialect.

A `Wire` stands for one qubit at one point in the program. A statement consumes
its input wires and returns new wires, so each wire has exactly one use.

A `Register` stands for a list of qubits. `Borrow` takes one qubit out of a
register as a wire, and `Release` puts the wire back.
"""

from kirin import types


class Wire:
    """A marker class that names the linear type of one qubit."""


class Register:
    """A marker class that names the linear type of a list of qubits."""


WireType = types.PyClass(Wire)
RegisterType = types.PyClass(Register)


def is_linear(kind: types.TypeAttribute) -> bool:
    """Return True if a value of the type `kind` must have exactly one use."""
    if isinstance(kind, types.BottomType):
        return False
    return kind.is_subseteq(WireType) or kind.is_subseteq(RegisterType)


def contains_linear(kind: types.TypeAttribute) -> bool:
    """Return whether the type contains a wire or register at any depth."""
    if is_linear(kind):
        return True
    if isinstance(kind, types.Union):
        return any(contains_linear(member) for member in kind.types)
    if isinstance(kind, types.TypeVar):
        return contains_linear(kind.bound)
    if isinstance(kind, types.Generic):
        return any(contains_linear(member) for member in kind.vars) or (
            kind.vararg is not None and contains_linear(kind.vararg.typ)
        )
    return False


def linear_paths(kind: types.TypeAttribute) -> tuple[tuple[int, ...], ...]:
    """Return the tuple paths of every wire and register in a supported type.

    Quantum containers must have fixed tuple fields so each field has an owner.
    """
    if is_linear(kind):
        return ((),)
    if not contains_linear(kind):
        return ()
    if (
        isinstance(kind, types.Generic)
        and kind.is_subseteq(types.Tuple)
        and kind.vararg is None
    ):
        return tuple(
            (index, *path)
            for index, member in enumerate(kind.vars)
            for path in linear_paths(member)
        )
    raise ValueError("Quantum containers must use fixed tuple fields.")
