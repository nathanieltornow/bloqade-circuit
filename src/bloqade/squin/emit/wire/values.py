"""Describe source qubit references as immutable locations in emitted allocations.

A location has one of three forms. For example, after `qs = squin.qalloc(4)` and
`q = squin.qubit.new()`:

- `q` and `qs` are `Whole(q_root)` and `Whole(qs_root, Affine(4))`.
- `qs[1]` is `Element(qs_root, Affine(1))`.
- `qs[1:3]` is `View(qs_root, Slice(Affine(1), 1, Affine(2)))`.

A nested register has one flat root register that holds its rows in order. A row
of a nested register is a `View` that keeps the inner dimensions in `shape`.
"""

from __future__ import annotations

from typing import TypeAlias
from dataclasses import dataclass

from kirin import ir

from bloqade.analysis.indices import Slice, Affine, python_slice
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError


@dataclass(frozen=True)
class Whole:
    """Name one whole allocation, which is a standalone qubit or a whole register."""

    root: ir.SSAValue
    """The emitted allocation, which is an allocation result, a call result or a parameter."""
    size: Affine | None = None
    """The number of qubits in a register, or None for a qubit or an unknown size."""
    shape: tuple[int | None, ...] = ()
    """The dimensions of a nested register, outermost first."""

    @property
    def length(self) -> Affine | None:
        """Return the number of rows of a nested register, or else its number of qubits."""
        if len(self.shape) > 1:
            return None if self.shape[0] is None else Affine(self.shape[0])
        return self.size


@dataclass(frozen=True)
class Element:
    """Name one qubit of a register."""

    root: ir.SSAValue
    """The emitted register."""
    index: Affine
    """The index of the qubit in the flat root register."""


@dataclass(frozen=True)
class View:
    """Name a contiguous part of a register, which is a slice or a nested row."""

    root: ir.SSAValue
    """The emitted register."""
    span: Slice
    """The elements of the flat root register that the view holds."""
    shape: tuple[int | None, ...] = ()
    """The inner dimensions of a nested row, outermost first, or empty for a slice."""

    @property
    def length(self) -> Affine | None:
        """Return the number of rows of a nested row, or else its number of qubits."""
        if len(self.shape) > 1:
            return None if self.shape[0] is None else Affine(self.shape[0])
        return self.span.length


Location: TypeAlias = Whole | Element | View
"""Name the allocation, the qubit or the register part that a source reference holds."""

Value: TypeAlias = ir.SSAValue | Location | tuple["Value", ...] | None
"""Represent an emitted classical value, a quantum location, or a source container."""


def is_qubit(value: Value) -> bool:
    """Check whether `value` names exactly one qubit."""
    if isinstance(value, Element):
        return True
    return isinstance(value, Whole) and value.root.type == WireType


def contains_location(value: Value) -> bool:
    """Identify source containers that still contain quantum locations."""
    return isinstance(value, (Whole, Element, View)) or (
        isinstance(value, tuple) and any(contains_location(item) for item in value)
    )


def register_part(location: Location, at: ir.Statement) -> Whole | View:
    """Return `location` if it names more than one qubit, or raise an error."""
    if isinstance(location, Element) or is_qubit(location):
        raise WireConversionError("A scalar qubit has no indexed elements.", at)
    assert isinstance(location, (Whole, View))
    return location


def select(location: Location, index: Affine, at: ir.Statement) -> Location:
    """Return the location of item `index` of a register or a register part.

    An item of a nested register is a row. An item of a flat register is one qubit.
    """
    part = register_part(location, at)
    index = checked_index(part, index, at)
    if len(part.shape) > 1:
        return row(part, index, at)
    return Element(part.root, root_index(part, index))


def checked_index(part: Whole | View, index: Affine, at: ir.Statement) -> Affine:
    """Count a negative constant index from the end and check a constant index.

    The bounds check runs only if both the index and the length are known.
    """
    count = None if part.length is None else part.length.constant
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
    return index


def row(part: Whole | View, index: Affine, at: ir.Statement) -> View:
    """Return the location of row `index` of a nested register."""
    width = 1
    for dimension in part.shape[1:]:
        if dimension is None:
            raise WireConversionError(
                "Nested register indexing requires known inner dimensions.", at
            )
        width *= dimension
    start = root_index(part, index.scale(width))
    return View(part.root, Slice(start, 1, Affine(width)), part.shape[1:])


def root_index(part: Whole | View, index: Affine) -> Affine:
    """Return the index in the flat root register of item `index` of `part`."""
    return part.span.item(index) if isinstance(part, View) else index


def select_slice(
    location: Location,
    start: Affine | None,
    stop: Affine | None,
    step: int,
    at: ir.Statement,
) -> Value:
    """Return the meaning of a register or a register part sliced with `start:stop:step`.

    A flat register gives one `View`. A nested register gives a tuple with the
    location of each selected row.
    """
    part = register_part(location, at)
    count = None if part.length is None else part.length.constant
    if len(part.shape) > 1:
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
        return tuple(select(part, Affine(i), at) for i in selected)
    span = python_slice(start, stop, step, count)
    if span is not None and isinstance(part, View):
        span = part.span.then(span)
    if span is None:
        raise WireConversionError("The slice has no place in a known allocation.", at)
    return View(part.root, span)
