"""This module holds the lattice of depths that the depth analysis computes."""

from dataclasses import dataclass

from kirin.lattice import BoundedLattice, SimpleMeetMixin

from .maxplus import Form

ZERO = Form.constant(0)
"""The depth of a fresh wire."""

NONE = Form()
"""The max-plus zero. A sink with this form has consumed no wire yet."""


class Depth(SimpleMeetMixin["Depth"], BoundedLattice["Depth"]):
    """A lattice element that states the depth of one value."""

    @classmethod
    def top(cls) -> "Depth":
        """Return `Unknown` with a generic reason."""
        return Unknown("a value without a static depth")

    @classmethod
    def bottom(cls) -> "Depth":
        """Return the element of a value that the analysis never reached."""
        return Bottom()

    def is_subseteq(self, other: "Depth") -> bool:
        """Return True if `self` is `Bottom`, `other` is `Unknown`, or both are equal."""
        return isinstance(self, Bottom) or isinstance(other, Unknown) or self == other

    def join(self, other: "Depth") -> "Depth":
        """Return the worst case of `self` and `other`.

        The join of two wire depths is their max. The join of two registers is the
        max per slot. A mix of different kinds of value gives `Unknown`.
        """
        if isinstance(self, Bottom) or self == other:
            return other
        if isinstance(other, Bottom):
            return self
        if isinstance(self, Unknown):
            return self
        if isinstance(other, Unknown):
            return other
        if isinstance(self, WireDepth) and isinstance(other, WireDepth):
            return WireDepth(self.form.join(other.form))
        if isinstance(self, RegDepth) and isinstance(other, RegDepth):
            return self.join_slots(other)
        if isinstance(self, Classical) and isinstance(other, Classical):
            return self
        return Unknown("a join of two different kinds of value")


@dataclass(frozen=True)
class Bottom(Depth):
    """The element of a value that the analysis never reached."""


@dataclass(frozen=True)
class Unknown(Depth):
    """The element of a value whose depth the analysis cannot compute exactly.

    `reason` names the construct that stopped the analysis.
    """

    reason: str


@dataclass(frozen=True)
class Classical(Depth):
    """The element of an integer, a float or a measured bit.

    The circuit depth ignores classical values.
    """


@dataclass(frozen=True)
class WireDepth(Depth):
    """The depth of a wire, as a max-plus form over the depths that enter the region."""

    form: Form


@dataclass(frozen=True)
class RegDepth(Depth):
    """The depths of the qubits in a register of static length.

    `slots` holds one form per slot. A slot holds None while an `Extract` has
    taken its wire out.
    """

    slots: tuple[Form | None, ...]

    def join_slots(self, other: "RegDepth") -> Depth:
        """Return the max per slot, or `Unknown` if the slot patterns differ."""
        if len(self.slots) != len(other.slots):
            return Unknown("a join of two registers of different length")
        joined: list[Form | None] = []
        for mine, theirs in zip(self.slots, other.slots):
            if mine is None and theirs is None:
                joined.append(None)
            elif mine is None or theirs is None:
                return Unknown("a join of a filled slot and an empty slot")
            else:
                joined.append(mine.join(theirs))
        return RegDepth(tuple(joined))


@dataclass(frozen=True)
class Results(Depth):
    """The depths of the values that a function returns, in order."""

    values: tuple[Depth, ...]
