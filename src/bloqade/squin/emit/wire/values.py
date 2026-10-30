"""Describe source qubit references as immutable locations in emitted allocations.

A location has one of four forms. For example, after `q = squin.qubit.new()` and
`qs = squin.qalloc(4)`:

- `q` is `Qubit(q_root)`.
- `qs` is `Register(qs_root, Affine(4))`.
- `qs[1]` is `Element(qs_root, Affine(1))`.
- `qs[1:3]` is `View(qs_root, Slice(Affine(1), 1, Affine(2)))`.

A nested register has one flat root register that holds its rows in order. The
items of a nested register or of a nested row are rows, and `inner` holds the
dimensions of one row. For a `grid` of 3 rows of 2 qubits, `grid` is
`Register(grid_root, Affine(3), (2,))`, and `grid[1]` is the flat view
`View(grid_root, Slice(Affine(2), 1, Affine(2)))`.
"""

from __future__ import annotations

import math
from typing import TypeAlias
from dataclasses import dataclass

from kirin import ir

from bloqade.analysis.indices import Slice, Affine, python_slice
from bloqade.squin.wire.errors import WireConversionError


@dataclass(frozen=True)
class Qubit:
    """Name one standalone qubit allocation."""

    root: ir.SSAValue
    """The emitted qubit, which is an allocation result, a call result or a parameter."""


@dataclass(frozen=True)
class Register:
    """Name one whole register allocation."""

    root: ir.SSAValue
    """The emitted register, which is an allocation result, a call result or a parameter."""
    length: Affine | None = None
    """The number of items, or None if it is unknown."""
    inner: tuple[int | None, ...] = ()
    """The dimensions of one item, or empty if each item is one qubit."""


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
    """The flat root index where each item starts, with one entry per item."""
    inner: tuple[int, ...] = ()
    """The dimensions of one item, or empty if each item is one qubit."""

    @property
    def length(self) -> Affine | None:
        """Return the number of items, or None if it is unknown."""
        return self.span.length


Location: TypeAlias = Qubit | Register | Element | View
"""Name the allocation, the qubit or the register part that a source reference holds."""

Value: TypeAlias = ir.SSAValue | Location | tuple["Value", ...] | None
"""Represent an emitted classical value, a quantum location, or a source container."""


def is_qubit(value: Value) -> bool:
    """Check whether `value` names exactly one qubit."""
    return isinstance(value, (Qubit, Element))


def contains_location(value: Value) -> bool:
    """Identify source containers that still contain quantum locations."""
    return isinstance(value, (Qubit, Register, Element, View)) or (
        isinstance(value, tuple) and any(contains_location(item) for item in value)
    )


def register_part(location: Location, at: ir.Statement) -> Register | View:
    """Return `location` if it holds items, or raise an error for one qubit."""
    if not isinstance(location, (Register, View)):
        raise WireConversionError("A scalar qubit has no indexed elements.", at)
    return location


def select(location: Location, index: Affine, at: ir.Statement) -> Location:
    """Return the location of item `index` of a register or a register part.

    An item of a nested register or row is a row. Any other item is one qubit.
    """
    part = register_part(location, at)
    index = checked_index(part, index, at)
    start = item_start(part, index, at)
    if not part.inner:
        return Element(part.root, start)
    rows = known_dimensions(part, at)
    return View(part.root, Slice(start, math.prod(rows[1:]), Affine(rows[0])), rows[1:])


def checked_index(part: Register | View, index: Affine, at: ir.Statement) -> Affine:
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


def known_dimensions(part: Register | View, at: ir.Statement) -> tuple[int, ...]:
    """Return the dimensions of one item of `part`, which must all be known."""
    dimensions = tuple(size for size in part.inner if size is not None)
    if len(dimensions) != len(part.inner):
        raise WireConversionError(
            "Nested register indexing requires known inner dimensions.", at
        )
    return dimensions


def item_start(part: Register | View, index: Affine, at: ir.Statement) -> Affine:
    """Return the flat root index where item `index` of `part` starts."""
    if isinstance(part, View):
        return part.span.item(index)
    return index.scale(math.prod(known_dimensions(part, at)))


def select_slice(
    location: Location,
    start: Affine | None,
    stop: Affine | None,
    step: int,
    at: ir.Statement,
) -> Value:
    """Return the meaning of a register or a register part sliced with `start:stop:step`.

    A slice of qubits gives one `View`. A slice of rows gives a tuple with the
    location of each selected row.
    """
    part = register_part(location, at)
    count = None if part.length is None else part.length.constant
    if part.inner:
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
