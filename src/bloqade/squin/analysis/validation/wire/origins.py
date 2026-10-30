"""Represent exact wire origins and tuple fields in a bounded validation lattice.

A handle identifies one allocation, or one element of a register allocation, by its
root. A root is an SSA value for a parameter or an allocation. For a fresh result
of a call, it is a pair of the call and the callee root.
"""

from __future__ import annotations

from typing import Literal, TypeAlias
from dataclasses import replace, dataclass
from collections.abc import Hashable

from kirin import ir
from kirin.lattice import BoundedLattice

from bloqade.analysis.indices import Affine
from bloqade.squin.wire.types import WireType


@dataclass(frozen=True)
class QubitHandle:
    """Identify one standalone qubit allocation."""

    root: Hashable
    """The identity of the allocation."""


@dataclass(frozen=True)
class RegisterHandle:
    """Identify one whole register allocation."""

    root: Hashable
    """The identity of the allocation."""
    length: Affine | None = None
    """The number of elements of the register, or None if it is unknown."""


@dataclass(frozen=True)
class ElementHandle:
    """Identify one element of a register allocation."""

    root: Hashable
    """The identity of the register allocation."""
    index: Affine
    """The index of the element in the register."""


Handle: TypeAlias = QubitHandle | RegisterHandle | ElementHandle
"""Identify an allocation or one element of a register allocation."""


def fresh_origin(value: ir.SSAValue, length: Affine | None = None) -> Origin:
    """Return the origin of the new allocation that `value` names."""
    if value.type == WireType:
        return Origin(QubitHandle(value))
    return Origin(RegisterHandle(value, length))


def element_of(origin: Origin, index: Affine) -> Origin:
    """Return the origin of element `index` of a whole register origin.

    The result is unknown for any other origin and for an index outside a known
    length.
    """
    register = origin.data
    if not isinstance(register, RegisterHandle):
        return Origin.top()
    position = index.constant
    if position is not None and position < 0:
        if register.length is None:
            return Origin.top()
        index = register.length + index
        position = index.constant
    size = None if register.length is None else register.length.constant
    if size is not None and position is not None and not 0 <= position < size:
        return Origin.top()
    return Origin(ElementHandle(register.root, index))


@dataclass(frozen=True)
class Origin(BoundedLattice["Origin"]):
    """Track an exact allocation identity or a tuple of identities.

    Unknown is the top element. Unreachable is the bottom element.
    Joining different identities gives unknown. Tuple joins compare corresponding fields.
    """

    data: Handle | tuple[Origin, ...] | Literal["unknown", "unreachable"] = "unknown"

    @classmethod
    def top(cls) -> Origin:
        """Return an unknown origin."""
        return cls("unknown")

    @classmethod
    def bottom(cls) -> Origin:
        """Return the value for unreachable execution."""
        return cls("unreachable")

    def is_subseteq(self, other: Origin) -> bool:
        """Check whether this value carries at least the origin facts that the other value carries."""
        if self.data == "unreachable" or other.data == "unknown" or self == other:
            return True
        return (
            isinstance(self.data, tuple)
            and isinstance(other.data, tuple)
            and len(self.data) == len(other.data)
            and all(a.is_subseteq(b) for a, b in zip(self.data, other.data))
        )

    def join(self, other: Origin) -> Origin:
        """Keep origin facts that hold for both incoming values."""
        if self.is_subseteq(other):
            return other
        if other.is_subseteq(self):
            return self
        if (
            isinstance(self.data, tuple)
            and isinstance(other.data, tuple)
            and len(self.data) == len(other.data)
        ):
            return Origin(tuple(a.join(b) for a, b in zip(self.data, other.data)))
        return self.top()

    def meet(self, other: Origin) -> Origin:
        """Combine compatible facts or return unreachable for conflicting identities."""
        if self.is_subseteq(other):
            return self
        if other.is_subseteq(self):
            return other
        if (
            isinstance(self.data, tuple)
            and isinstance(other.data, tuple)
            and len(self.data) == len(other.data)
        ):
            return Origin(tuple(a.meet(b) for a, b in zip(self.data, other.data)))
        return self.bottom()

    def roots(self) -> frozenset[Hashable]:
        """Return the allocation roots that this value identifies exactly."""
        if isinstance(self.data, Handle):
            return frozenset({self.data.root})
        if isinstance(self.data, tuple):
            return frozenset(root for item in self.data for root in item.roots())
        return frozenset()

    def at_call(self, call: ir.Statement, inputs: frozenset[Hashable]) -> Origin:
        """Give each fresh result a call-specific identity while preserving borrowed input identities."""
        value = self.data
        if isinstance(value, Handle) and value.root not in inputs:
            return Origin(replace(value, root=(call, value.root)))
        if isinstance(value, tuple):
            return Origin(tuple(item.at_call(call, inputs) for item in value))
        return self
