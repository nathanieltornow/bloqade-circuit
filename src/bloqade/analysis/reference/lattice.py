"""This module holds the lattice of references that the reference analysis computes."""

from typing import ClassVar
from dataclasses import field, dataclass

from kirin import ir, types
from kirin.lattice import (
    SingletonMeta,
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
)

from .index import Slice, Affine

Step = Affine | Slice
"""One step of an access path: an index into a register, or a slice of it."""


@dataclass(frozen=True, repr=False)
class Returned:
    """A root that a call allocates and returns at one position of its result.

    `call` is a statement with kirin's `StaticCall` trait. `inner` is the root
    inside the callee that the call returns at `position`.
    """

    call: ir.Statement
    position: int
    inner: "Root"

    @property
    def callee(self) -> ir.Method:
        """Return the method that the call invokes, through kirin's `StaticCall`."""
        return self.call.get_present_trait(ir.StaticCall).get_callee(self.call)

    def __repr__(self) -> str:
        """Return the call result that holds the root and its origin in the callee.

        The form is `%x = make()[0]`. A result that packs several outputs in a tuple
        prints as `%pair[1] = make()[1]`. A result without a name prints as
        `make()[0]`.
        """
        origin = f"{self.callee.sym_name}()[{self.position}]"
        results = self.call.results
        packed = len(results) == 1
        value = results[0] if packed else results[self.position]
        if not value.name:
            return origin
        if packed and self.callee.return_type.is_subseteq(types.Tuple):
            return f"%{value.name}[{self.position}] = {origin}"
        return f"%{value.name} = {origin}"


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


class Place(Ref):
    """A reference to tracked state at an access path below a root.

    `root` owns the state. `path` holds the steps from the root to the state. An
    empty path names the whole root. The path `(i,)` names item `i` of a register.
    For example, `Slot(%qs, 2)` is the place of `qs[2]`, with the path `(2,)`.

    Each subclass fixes whether the place names one item or a register, so a
    `match` on the subclass tells the two apart.
    """

    __match_args__ = ("root", "path")
    root: Root
    one_item: ClassVar[bool]
    """True if the place names one item of tracked state, such as one qubit."""

    @property
    def path(self) -> tuple[Step, ...]:
        """Return the steps from the root to the state."""
        raise NotImplementedError


def _index(index: "Affine | int | ir.SSAValue") -> Affine:
    """Return `index` as an affine index.

    An integer becomes a constant index. An SSA value becomes the index that is
    that value alone.
    """
    if isinstance(index, Affine):
        return index
    if isinstance(index, int):
        return Affine(index)
    return Affine.of(index)


@dataclass(frozen=True, repr=False)
class Whole(Place):
    """A reference to one item of tracked state that is a root, such as one qubit."""

    root: Root
    one_item: ClassVar[bool] = True

    @property
    def path(self) -> tuple[Step, ...]:
        """Return the empty path, which names the whole root."""
        return ()

    def __repr__(self) -> str:
        """Return the short form, such as `Whole(%q)`."""
        return f"Whole({_owner(self.root)})"


@dataclass(frozen=True, repr=False)
class Register(Place):
    """A reference to a whole register, which holds items of tracked state."""

    root: Root
    one_item: ClassVar[bool] = False

    @property
    def path(self) -> tuple[Step, ...]:
        """Return the empty path, which names the whole root."""
        return ()

    def __repr__(self) -> str:
        """Return the short form, such as `Register(%qs)`."""
        return f"Register({_owner(self.root)})"


@dataclass(frozen=True, repr=False)
class Slot(Place):
    """A reference to one item of a register root.

    The constructor accepts an integer or an SSA value as `index` and stores it as
    an `Affine` index. If the register has a static length, a constant index lies in
    the range 0 to the length minus 1.
    """

    root: Root
    index: Affine
    one_item: ClassVar[bool] = True

    def __init__(self, root: Root, index: "Affine | int | ir.SSAValue") -> None:
        """Store `root` and `index`, with `index` converted to an `Affine` index."""
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "index", _index(index))

    @property
    def path(self) -> tuple[Step, ...]:
        """Return the path that holds the index alone."""
        return (self.index,)

    def __repr__(self) -> str:
        """Return the short form, such as `Slot(%qs, 2)` or `Slot(%qs, %i + 1)`."""
        return f"Slot({_owner(self.root)}, {self.index!r})"


@dataclass(frozen=True, repr=False)
class Closure(Ref):
    """A function value that a kernel builds, with the references that it captures.

    `code` is the statement that defines the body, such as a `func.Lambda`.
    `captured` holds the reference of each captured value, in order.
    """

    code: ir.Statement
    captured: tuple[Ref, ...]

    def __repr__(self) -> str:
        """Return the short form, such as `Closure(f, [Whole(%q)])`."""
        name = getattr(self.code, "sym_name", self.code.name)
        return f"Closure({name}, [{', '.join(map(repr, self.captured))}])"


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


def roots(ref: Ref) -> list[Root]:
    """Return the roots that `ref` names."""
    match ref:
        case Place(root):
            return [root]
        case Members(members):
            return [root for member in members for root in roots(member)]
        case Closure(_, captured):
            return [root for member in captured for root in roots(member)]
    return []


def positions(ref: Ref, count: int) -> tuple[Ref, ...]:
    """Return the reference at each of the `count` positions of a returned value.

    A `Positions` reference names each position. Any other reference, such as
    the one of a constant tuple, holds for every position.
    """
    return ref.refs if isinstance(ref, Positions) else (ref,) * count
