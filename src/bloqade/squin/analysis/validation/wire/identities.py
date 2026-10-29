"""Represent exact wire origins and tuple fields in a bounded validation lattice."""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass
from collections.abc import Hashable

from kirin import ir, types
from kirin.lattice import BoundedLattice

from bloqade.analysis.indices import Affine
from bloqade.squin.wire.types import WireType


@dataclass(frozen=True)
class Handle:
    """Identify a wire allocation or one element of a register."""

    root: Hashable
    kind: types.TypeAttribute
    length: Affine | None = None
    selection: Affine | None = None

    def pick(self, index: Affine) -> Origin:
        """Identify an element of a whole register when its index permits a valid selection."""
        if self.kind == WireType or self.selection is not None:
            return Origin.top()
        if index.constant is not None and index.const < 0:
            if self.length is None:
                return Origin.top()
            index = self.length + index
        if (
            self.length is not None
            and self.length.constant is not None
            and index.constant is not None
        ):
            if not 0 <= index.const < self.length.const:
                return Origin.top()
        return Origin(Handle(self.root, self.kind, self.length, index))


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
            return Origin(
                Handle((call, value.root), value.kind, value.length, value.selection)
            )
        if isinstance(value, tuple):
            return Origin(tuple(item.at_call(call, inputs) for item in value))
        return self
