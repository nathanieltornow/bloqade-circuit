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
