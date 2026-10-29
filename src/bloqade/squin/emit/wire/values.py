"""Describe source references with immutable locations in emitted allocations."""

from __future__ import annotations

from typing import TypeAlias
from dataclasses import dataclass

from kirin import ir

from bloqade.analysis.indices import Slice, Affine, python_slice
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError


@dataclass(frozen=True)
class Location:
    """Name a whole allocation, one element of a register, or a view of a register.

    For example, after `qs = squin.qalloc(4)` the source value `qs[1:3]` has the
    location `Location(r, Affine(4), Slice(Affine(1), 1, Affine(2)))`. Here `r` is
    the emitted register.
    """

    root: ir.SSAValue
    """The emitted allocation, which is an allocation result, a call result or a parameter."""
    root_length: Affine | None = None
    """The number of qubits in the whole root allocation, or None if it is unknown."""
    selection: Affine | Slice | None = None
    """The selected index or slice of the root. None selects the whole root."""
    shape: tuple[int | None, ...] = ()
    """The dimensions of a nested register view, outermost first.

    The root of a nested register is one flat register that holds the rows in order.
    """

    @property
    def scalar(self) -> bool:
        """Check whether this location names one qubit."""
        return self.root.type == WireType or isinstance(self.selection, Affine)

    @property
    def length(self) -> Affine | None:
        """Return the number of items in this view, or None if it is unknown.

        An item of a nested view is a row. An item of a flat view is a qubit.
        """
        if len(self.shape) > 1:
            return None if self.shape[0] is None else Affine(self.shape[0])
        if isinstance(self.selection, Slice):
            return self.selection.length
        return self.root_length

    def select(self, index: Affine, at: ir.Statement) -> Location:
        """Resolve one element through the dimensions and selection of this location."""
        if self.scalar:
            raise WireConversionError("A scalar qubit has no indexed elements.", at)
        length = self.length
        count = None if length is None else length.constant
        position = index.constant
        if position is not None and position < 0:
            if count is None:
                raise WireConversionError(
                    "Cannot resolve a negative index into a register of unknown length.",
                    at,
                )
            position += count
            index = Affine(position)
        if count is not None and position is not None and not 0 <= position < count:
            raise WireConversionError(
                "The register index has no place in this allocation.", at
            )
        if len(self.shape) > 1:
            width = 1
            for dimension in self.shape[1:]:
                if dimension is None:
                    raise WireConversionError(
                        "Nested register indexing requires known inner dimensions.", at
                    )
                width *= dimension
            start = index.scale(width)
            if isinstance(self.selection, Slice):
                start = self.selection.item(start)
            row = Slice(start, 1, Affine(width))
            return Location(self.root, self.root_length, row, self.shape[1:])
        if isinstance(self.selection, Slice):
            index = self.selection.item(index)
        return Location(self.root, self.root_length, index)

    def slice(
        self, start: Affine | None, stop: Affine | None, step: int, at: ir.Statement
    ) -> Value:
        """Compose a Python slice with this register selection."""
        length = self.length
        count = None if length is None else length.constant
        if len(self.shape) > 1:
            if count is None or any(
                x is not None and x.constant is None for x in (start, stop)
            ):
                raise WireConversionError(
                    "A nested register slice requires constant bounds and an outer length.",
                    at,
                )
            selected = range(count)[
                slice(
                    None if start is None else start.constant,
                    None if stop is None else stop.constant,
                    step,
                )
            ]
            return tuple(self.select(Affine(i), at) for i in selected)
        view = python_slice(start, stop, step, count)
        if view is not None and isinstance(self.selection, Slice):
            view = self.selection.then(view)
        if view is None:
            raise WireConversionError(
                "The slice has no place in a known allocation.", at
            )
        return Location(self.root, self.root_length, view)


Value: TypeAlias = ir.SSAValue | Location | tuple["Value", ...] | None
"""Represent an emitted classical value, a quantum location, or a source container."""


def contains_location(value: Value) -> bool:
    """Identify source containers that still contain quantum locations."""
    return (
        isinstance(value, Location)
        or isinstance(value, tuple)
        and any(contains_location(item) for item in value)
    )
