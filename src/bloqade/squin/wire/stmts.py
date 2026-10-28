"""This module holds the statements of the squin wire dialect that have no squin twin.

`Alloc` and `Free` start and end the life of one qubit. `AllocRegister` and
`FreeRegister` do the same for a list of qubits. `Borrow` and `Release` take one
qubit out of a register and put it back. `Length` reads the size of a register.

`Unwrap` and `Wrap` connect wires to the qubit references of reference squin, as
a load and a store connect values to memory. `Unwrap` takes the qubit of a
reference out as a wire, and `Wrap` puts a wire back. `UnwrapRegister` and
`WrapRegister` do the same for a whole register. `Reference` and
`ReferenceRegister` create a reference without a qubit, which the next `Wrap`
fills. The promotion passes remove these statements wherever the reference
analysis can name the place of each reference.
"""

from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.types import QubitType

from .types import WireType, RegisterType
from .traits import Threads, ReferenceAccess
from ._dialect import dialect


@statement(dialect=dialect)
class Alloc(ir.Statement):
    """Allocate one qubit in the zero state and return its wire."""

    name = "alloc"
    result: ir.ResultValue = info.result(WireType)


@statement(dialect=dialect)
class Free(ir.Statement):
    """End the life of the qubit that `wire` carries."""

    name = "free"
    wire: ir.SSAValue = info.argument(WireType)


@statement(dialect=dialect)
class AllocRegister(ir.Statement):
    """Allocate `size` qubits in the zero state and return them as one register."""

    name = "alloc_register"
    size: ir.SSAValue = info.argument(types.Int)
    result: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class FreeRegister(ir.Statement):
    """End the life of every qubit of `register`."""

    name = "free_register"
    register: ir.SSAValue = info.argument(RegisterType)


@statement(dialect=dialect)
class Borrow(ir.Statement):
    """Take the qubit at `index` out of `register` as a wire.

    The slot stays empty until a `Release` at the same index fills it again.
    """

    name = "borrow"
    traits = frozenset({Threads((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    register_out: ir.ResultValue = info.result(RegisterType)
    wire: ir.ResultValue = info.result(WireType)


@statement(dialect=dialect)
class Release(ir.Statement):
    """Put the qubit that `wire` carries back into `register` at `index`."""

    name = "release"
    traits = frozenset({Threads((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    index: ir.SSAValue = info.argument(types.Int)
    wire: ir.SSAValue = info.argument(WireType)
    register_out: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class Length(ir.Statement):
    """Return `register` and the count of qubits that it holds."""

    name = "length"
    traits = frozenset({Threads((("register", "register_out"),))})
    register: ir.SSAValue = info.argument(RegisterType)
    register_out: ir.ResultValue = info.result(RegisterType)
    result: ir.ResultValue = info.result(types.Int)


QubitListType = ilist.IListType[QubitType, types.Any]
"""Type a reference squin list of qubits of any length."""


@statement(dialect=dialect)
class Unwrap(ir.Statement):
    """Take the qubit that the reference `qubit` names out as a wire.

    The reference holds no qubit until a `Wrap` puts a wire back.
    """

    name = "unwrap"
    traits = frozenset({ReferenceAccess(takes=True)})
    qubit: ir.SSAValue = info.argument(QubitType)
    wire: ir.ResultValue = info.result(WireType)


@statement(dialect=dialect)
class Wrap(ir.Statement):
    """Put the qubit that `wire` carries back into the reference `qubit`."""

    name = "wrap"
    traits = frozenset({ReferenceAccess(takes=False)})
    qubit: ir.SSAValue = info.argument(QubitType)
    wire: ir.SSAValue = info.argument(WireType)


@statement(dialect=dialect)
class UnwrapRegister(ir.Statement):
    """Take every qubit of the register that `qubits` names out as one register."""

    name = "unwrap_register"
    traits = frozenset({ReferenceAccess(takes=True)})
    qubits: ir.SSAValue = info.argument(QubitListType)
    register: ir.ResultValue = info.result(RegisterType)


@statement(dialect=dialect)
class WrapRegister(ir.Statement):
    """Put the qubits of `register` back into the reference list `qubits`."""

    name = "wrap_register"
    traits = frozenset({ReferenceAccess(takes=False)})
    qubits: ir.SSAValue = info.argument(QubitListType)
    register: ir.SSAValue = info.argument(RegisterType)


@statement(dialect=dialect)
class Reference(ir.Statement):
    """Create a reference to one qubit that holds no qubit yet.

    A call that returns a qubit gives a wire, and the caller needs a reference to
    keep using the qubit. The next `Wrap` puts the wire into this reference.
    """

    name = "reference"
    result: ir.ResultValue = info.result(QubitType)


@statement(dialect=dialect)
class ReferenceRegister(ir.Statement):
    """Create a reference to a register that holds no qubits yet.

    The result type `IList[Qubit, Literal(n)]` gives the length of the register
    when the length is static. The next `WrapRegister` fills the reference.
    """

    name = "reference_register"
    result: ir.ResultValue = info.result(QubitListType)
