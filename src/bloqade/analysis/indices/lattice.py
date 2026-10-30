"""Hold what the index analysis knows about one value."""

from __future__ import annotations

from dataclasses import dataclass

from kirin.lattice import BoundedLattice

from .affine import Affine


@dataclass(frozen=True)
class Index(BoundedLattice["Index"]):
    """Hold the affine index fact of one value.

    `UnknownIndex` is the top, `UnreachableIndex` is the bottom, and `KnownIndex`
    holds an affine index.
    """

    @classmethod
    def top(cls) -> Index:
        """Return the fact of a value without an affine index."""
        return UnknownIndex()

    @classmethod
    def bottom(cls) -> Index:
        """Return the fact of a value that no execution reaches."""
        return UnreachableIndex()

    def is_subseteq(self, other: Index) -> bool:
        """Return True if `other` holds no more than this fact."""
        return (
            isinstance(self, UnreachableIndex)
            or isinstance(other, UnknownIndex)
            or self == other
        )

    def join(self, other: Index) -> Index:
        """Return the fact that holds for both values."""
        if self.is_subseteq(other):
            return other
        if other.is_subseteq(self):
            return self
        return UnknownIndex()

    def meet(self, other: Index) -> Index:
        """Return the fact that both facts imply."""
        if self.is_subseteq(other):
            return self
        if other.is_subseteq(self):
            return other
        return UnreachableIndex()


@dataclass(frozen=True)
class UnknownIndex(Index):
    """Name the fact of a value without an affine index."""


@dataclass(frozen=True)
class UnreachableIndex(Index):
    """Name the fact of a value that no execution reaches."""


@dataclass(frozen=True)
class KnownIndex(Index):
    """Hold the affine index of a value."""

    affine: Affine
