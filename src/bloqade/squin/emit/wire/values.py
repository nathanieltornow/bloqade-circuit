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
    """Name a whole allocation, one element, or a view of an emitted register."""

    root: ir.SSAValue
    length: Affine | None = None
    selection: Affine | Slice | None = None
    shape: tuple[int | None, ...] = ()

    @property
    def scalar(self) -> bool:
        """Identify a standalone qubit or one selected register element."""
        return self.root.type == WireType or isinstance(self.selection, Affine)

    @property
    def size(self) -> Affine | None:
        """Return the logical length of this register or view."""
        if len(self.shape) > 1:
            return None if self.shape[0] is None else Affine(self.shape[0])
        return (
            self.selection.length if isinstance(self.selection, Slice) else self.length
        )

    def select(self, index: Affine, at: ir.Statement) -> Location:
        """Resolve one element through the dimensions and selection of this location."""
        if self.scalar:
            raise WireConversionError("A scalar qubit has no indexed elements.", at)
        size = self.size
        if index.constant is not None and index.const < 0:
            if size is None or size.constant is None:
                raise WireConversionError(
                    "Cannot resolve a negative index into a register of unknown length.",
                    at,
                )
            index = size + index
        if (
            size is not None
            and size.constant is not None
            and index.constant is not None
        ):
            if not 0 <= index.const < size.const:
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
            return Location(
                self.root, self.length, Slice(start, 1, Affine(width)), self.shape[1:]
            )
        if isinstance(self.selection, Slice):
            index = self.selection.item(index)
        return Location(self.root, self.length, index)

    def slice(
        self, start: Affine | None, stop: Affine | None, step: int, at: ir.Statement
    ) -> Value:
        """Compose a Python slice with this register selection."""
        size = self.size
        if len(self.shape) > 1:
            if (
                size is None
                or size.constant is None
                or any(x is not None and x.constant is None for x in (start, stop))
            ):
                raise WireConversionError(
                    "A nested register slice requires constant bounds and an outer length.",
                    at,
                )
            selected = range(size.const)[
                slice(
                    None if start is None else start.const,
                    None if stop is None else stop.const,
                    step,
                )
            ]
            return tuple(self.select(Affine(i), at) for i in selected)
        view = python_slice(start, stop, step, None if size is None else size.constant)
        if view is not None and isinstance(self.selection, Slice):
            view = self.selection.then(view)
        if view is None:
            raise WireConversionError(
                "The slice has no place in a known allocation.", at
            )
        return Location(self.root, self.length, view)


Value: TypeAlias = ir.SSAValue | Location | tuple["Value", ...] | None
"""Represent an emitted classical value, a quantum location, or a source container."""


def contains_location(value: Value) -> bool:
    """Identify source containers that still contain quantum locations."""
    return (
        isinstance(value, Location)
        or isinstance(value, tuple)
        and any(contains_location(item) for item in value)
    )
