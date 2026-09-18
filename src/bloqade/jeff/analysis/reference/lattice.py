"""This module holds the lattice of references that the reference analysis computes."""

from dataclasses import field, dataclass

from kirin import ir
from kirin.lattice import (
    SingletonMeta,
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
)
from kirin.dialects import func

from bloqade.jeff.dialects import stmts

Call = func.Invoke | stmts.Call
"""A statement that calls a function: a kirin invoke or a jeff call."""


@dataclass(frozen=True, repr=False)
class Returned:
    """A root that a call allocates and returns at one position of its result."""

    call: Call
    position: int

    def __repr__(self) -> str:
        """Return the callee name and the position, such as `make()[0]`."""
        return f"{self.call.callee.sym_name}()[{self.position}]"


Root = ir.SSAValue | Returned
"""An owner of tracked state: an allocation, a parameter, or a `Returned` root."""

CARRIED = "a value carried by a loop or branch"
"""The reason of an `Unknown` that a join of two different references produces."""


class Ref(SimpleJoinMixin["Ref"], SimpleMeetMixin["Ref"], BoundedLattice["Ref"]):
    """A lattice element that states what a value refers to.

    The join of two different references is `Unknown`.
    """

    @classmethod
    def top(cls) -> "Ref":
        """Return `Unknown` with the reason `CARRIED`."""
        return Unknown(CARRIED)

    @classmethod
    def bottom(cls) -> "Ref":
        """Return the element of a value that the analysis never reached."""
        return Bottom()

    def is_subseteq(self, other: "Ref") -> bool:
        """Return True if `other` is `Unknown` or equal to `self`."""
        return isinstance(other, Unknown) or self == other


def _name(value: ir.SSAValue) -> str:
    """Return the printed name of an SSA value, such as `%qs`."""
    return f"%{value.name}" if value.name else "%?"


def _owner(root: Root) -> str:
    """Return the printed name of a root."""
    return repr(root) if isinstance(root, Returned) else _name(root)


@dataclass(frozen=True, repr=False)
class Whole(Ref):
    """A reference to a whole root, which is one item or one register."""

    root: Root

    def __repr__(self) -> str:
        """Return the short form, such as `Whole(%q)`."""
        return f"Whole({_owner(self.root)})"


@dataclass(frozen=True, repr=False)
class Slot(Ref):
    """A reference to one item of a register root.

    If the register has a static length, a constant index lies in the range 0 to the
    length minus 1.
    """

    root: Root
    index: int | ir.SSAValue

    def __repr__(self) -> str:
        """Return the short form, such as `Slot(%qs, 2)`."""
        at = self.index if isinstance(self.index, int) else _name(self.index)
        return f"Slot({_owner(self.root)}, {at})"


@dataclass(frozen=True, repr=False)
class Members(Ref):
    """A value that holds other values, with one reference for each member.

    The join of two different member tuples is `Unknown`, even if some members are
    equal.
    """

    refs: tuple[Ref, ...]


@dataclass(frozen=True, repr=False)
class Items(Members):
    """The references of the items of a literal list."""

    def __repr__(self) -> str:
        """Return the short form, such as `[Whole(%a), Whole(%b)]`."""
        return "[" + ", ".join(map(repr, self.refs)) + "]"


@dataclass(frozen=True, repr=False)
class Positions(Members):
    """The references of the positions of a tuple."""

    def __repr__(self) -> str:
        """Return the short form, such as `(Whole(%a), Untracked)`."""
        return "(" + ", ".join(map(repr, self.refs)) + ")"


@dataclass(frozen=True, repr=False)
class Untracked(Ref, metaclass=SingletonMeta):
    """A value that refers to no tracked state."""

    def __repr__(self) -> str:
        """Return `Untracked`."""
        return "Untracked"


@dataclass(frozen=True, repr=False)
class Unknown(Ref):
    """A value that may refer to tracked state that the analysis cannot name.

    `Unknown` is the top element. The reason explains the lost information, and it
    does not take part in equality.
    """

    reason: str = field(compare=False, hash=False)

    def __repr__(self) -> str:
        """Return the reason, such as `Unknown: a slice of a register`."""
        return f"Unknown: {self.reason}"


@dataclass(frozen=True, repr=False)
class Bottom(Ref, metaclass=SingletonMeta):
    """A value that the analysis never reached, the bottom element."""

    def is_subseteq(self, other: Ref) -> bool:
        """Return True, because bottom lies below every element."""
        return True

    def __repr__(self) -> str:
        """Return `Bottom`."""
        return "Bottom"


UNTRACKED = Untracked()


def root_in(root: Root, region: ir.Region) -> bool:
    """Return True if `region` defines `root` at any depth."""
    if isinstance(root, Returned):
        return region.is_ancestor(root.call)
    if isinstance(root, ir.BlockArgument):
        return region.is_ancestor(root.block)
    return isinstance(root.owner, ir.Statement) and region.is_ancestor(root.owner)
