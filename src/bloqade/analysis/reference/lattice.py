"""Define qubit references and their lattice operations.

References describe whole qubits or registers, register elements, lists,
and tuples. Roots identify allocations, parameters, or returned allocations.
Classical, unknown, and unreached values have separate lattice elements.
"""

from dataclasses import field, dataclass

from kirin import ir, types
from kirin.lattice import BoundedLattice
from kirin.dialects import func, ilist

from bloqade import squin
from bloqade.types import QubitType

from ._ir import value_in, const_int, function_of, statement_in, declared_outputs


@dataclass(frozen=True, repr=False)
class Returned:
    """An allocation returned at a call's output position.

    The call and position identify the root. The optional length records the
    allocation size and does not affect equality.
    """

    call: func.Invoke
    position: int
    length: int | None = field(default=None, compare=False, hash=False)

    def __repr__(self) -> str:
        """Format the root for IR annotations."""
        return f"{self.call.callee.sym_name}()[{self.position}]"


Root = ir.SSAValue | Returned
"""An allocation, parameter, or returned allocation that identifies quantum state."""


CARRIED = "a qubit carried by a loop or branch"


class Ref(BoundedLattice["Ref"]):
    """A reference lattice element; distinct known references join to Unknown."""

    @classmethod
    def top(cls) -> "Ref":
        """Return an unknown reference with the default control-flow reason."""
        return Unknown(CARRIED)

    @classmethod
    def bottom(cls) -> "Ref":
        """Return the element representing an unreached value."""
        return BOTTOM

    def is_subseteq(self, other: "Ref") -> bool:
        """Compare references in the lattice order."""
        return self is BOTTOM or isinstance(other, Unknown) or self == other

    def join(self, other: "Ref") -> "Ref":
        """Return the least upper bound, preserving the first unknown reason."""
        if self is BOTTOM:
            return other
        if other is BOTTOM or isinstance(self, Unknown) or self == other:
            return self
        if isinstance(other, Unknown):
            return other
        return self.top()

    def meet(self, other: "Ref") -> "Ref":
        """Return the greatest lower bound."""
        if self == other or isinstance(other, Unknown):
            return self
        if isinstance(self, Unknown):
            return other
        return BOTTOM

    def __repr__(self) -> str:
        """Format the reference for IR annotations."""
        match self:
            case Whole(root):
                return f"Whole({_owner(root)})"
            case Slot(root, index):
                at = index if isinstance(index, int) else _name(index)
                return f"Slot({_owner(root)}, {at})"
            case QubitList(refs):
                return "[" + ", ".join(map(repr, refs)) + "]"
            case Positions(refs):
                return "(" + ", ".join(map(repr, refs)) + ")"
            case Unknown(reason):
                return f"Unknown: {reason}"
            case Classical():
                return "Classical"
        return "Bottom"


def _name(value: ir.SSAValue) -> str:
    return f"%{value.name}" if value.name else "%?"


def _owner(root: Root) -> str:
    return repr(root) if isinstance(root, Returned) else _name(root)


@dataclass(frozen=True, repr=False)
class Whole(Ref):
    """A reference to an entire qubit or register."""

    root: Root


@dataclass(frozen=True, repr=False)
class Slot(Ref):
    """A reference to one register element.

    The analysis normalizes constant indices when the register length is known.
    """

    root: Root
    index: int | ir.SSAValue


@dataclass(frozen=True, repr=False)
class Members(Ref):
    """An ordered collection of references.

    Distinct collections join to Unknown, even when some members match.
    """

    refs: tuple[Ref, ...]


@dataclass(frozen=True, repr=False)
class QubitList(Members):
    """References to the elements of an explicit qubit list."""


@dataclass(frozen=True, repr=False)
class Positions(Members):
    """References to tuple elements in order."""


@dataclass(frozen=True, repr=False)
class Classical(Ref):
    """A value with no quantum references."""


@dataclass(frozen=True, repr=False)
class Unknown(Ref):
    """An unresolved reference, the top element of the lattice.

    The reason describes the missing information and does not affect equality.
    """

    reason: str = field(compare=False, hash=False)


@dataclass(frozen=True, repr=False)
class _Bottom(Ref):
    """An unreached value."""


CLASSICAL = Classical()
BOTTOM = _Bottom()
QubitRef = Whole | Slot
"""A whole qubit or register, or one register element."""


# -- types and roots ----------------------------------------------------------


def is_allocator(node: ir.Statement) -> bool:
    """Check whether a statement calls squin.qalloc."""
    return isinstance(node, func.Invoke) and node.callee is squin.qalloc


def static_size(size: ir.SSAValue) -> int | None:
    """Return a constant allocation size, clamped to zero, or None if unknown."""
    constant = const_int(size)
    return None if constant is None else max(0, constant)


def is_quantum_type(declared: types.TypeAttribute) -> bool:
    """Check whether a type describes a qubit or a list of qubits."""
    return declared.is_subseteq(QubitType) or declared.is_subseteq(
        ilist.IListType[QubitType, types.Any]
    )


def typed_length(list_type: types.TypeAttribute) -> int | None:
    """Return the integer length from a list type, or None if unknown."""
    length = _list_length(list_type)
    if isinstance(length, types.Literal) and isinstance(length.data, int):
        return length.data
    return None


def _list_length(list_type: types.TypeAttribute) -> types.TypeAttribute | None:
    """Return the length parameter of a generic list type."""
    return list_type.vars[1] if isinstance(list_type, types.Generic) else None


def root_type(root: Root) -> types.TypeAttribute:
    """Return the SSA value type or the declared type of a call output."""
    if isinstance(root, Returned):
        outputs = declared_outputs(function_of(root.call.callee).signature)
        return outputs[root.position] if root.position < len(outputs) else types.Bottom
    return root.type


def is_register_root(root: Root) -> bool:
    """Check whether a root represents a register."""
    return root_type(root).is_subseteq(ilist.IListType[QubitType, types.Any])


def _allocated_size(root: Root) -> ir.SSAValue | None:
    """Return the size argument of a squin.qalloc result, or None."""
    if not isinstance(root, ir.SSAValue):
        return None
    owner = root.owner
    if isinstance(owner, func.Invoke) and is_allocator(owner):
        return owner.inputs[0]
    return None


def static_length(root: Root) -> int | None:
    """Return the known register length, or None.

    Use the allocation size for local and returned allocations, and the type's
    length for parameters.
    """
    if isinstance(root, Returned):
        return root.length
    if (size := _allocated_size(root)) is not None:
        return static_size(size)
    return typed_length(root_type(root))


LengthKey = ir.SSAValue | types.TypeVar
"""An SSA size argument or a list-length type variable."""


def length_key(root: Root) -> LengthKey | None:
    """Return the SSA size argument or type variable identifying a register's length.

    Matching keys imply equal lengths. None means the length has no known key.
    """
    if not isinstance(root, ir.SSAValue):
        return None
    if (size := _allocated_size(root)) is not None:
        return size
    length = _list_length(root.type)
    return length if isinstance(length, types.TypeVar) else None


def same_length(a: Root, b: Root) -> bool:
    """Check for equal static lengths or a shared runtime length key."""
    a_size, b_size = static_length(a), static_length(b)
    if a_size is not None and b_size is not None:
        return a_size == b_size
    key = length_key(a)
    return key is not None and key is length_key(b)


def root_in(root: Root, region: ir.Region) -> bool:
    """Check whether a region contains the root definition."""
    if isinstance(root, Returned):
        return statement_in(root.call, region)
    return value_in(root, region)


# -- questions about references -------------------------------------------------


def qubit_refs(ref: Ref) -> list[QubitRef] | None:
    """Expand a reference into individual qubit references in order.

    Expand registers with known lengths and explicit lists of single qubits.
    Return None when expansion is unavailable.
    """
    match ref:
        case Whole(root) if is_register_root(root):
            size = static_length(root)
            return None if size is None else [Slot(root, i) for i in range(size)]
        case Whole() | Slot():
            return [ref]
        case QubitList(members):
            found: list[QubitRef] = []
            for member in members:
                if not isinstance(member, (Whole, Slot)) or (
                    isinstance(member, Whole) and is_register_root(member.root)
                ):
                    return None
                found.append(member)
            return found
    return None


def overlap(a: QubitRef, b: QubitRef) -> bool:
    """Check whether two references definitely share a qubit.

    A register overlaps each of its elements. Element references overlap when
    they have the same root and equal constant indices or the same SSA index.
    False does not prove that the references are disjoint.
    """
    if a.root != b.root:
        return False
    if isinstance(a, Whole) or isinstance(b, Whole):
        return True
    if isinstance(a.index, int) and isinstance(b.index, int):
        return a.index == b.index
    return a.index is b.index


def roots(ref: Ref) -> list[Root]:
    """Collect roots in member order, including duplicates."""
    match ref:
        case Whole(root) | Slot(root, _):
            return [root]
        case Members(members):
            return [root for member in members for root in roots(member)]
    return []


def mentions_quantum(ref: Ref) -> bool:
    """Check for a quantum or unknown reference, including nested members."""
    match ref:
        case Whole() | Slot() | Unknown():
            return True
        case Members(members):
            return any(mentions_quantum(m) for m in members)
    return False


def mentions_unknown(ref: Ref) -> bool:
    """Check for an unknown reference, including nested members."""
    match ref:
        case Unknown():
            return True
        case Members(members):
            return any(mentions_unknown(m) for m in members)
    return False
