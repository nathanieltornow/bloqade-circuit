"""Resolve reference expressions as an ordered converter visits their definitions.

Handles identify allocations and register positions independently of current wire values.
The converter records each handle when it visits the value's definition.
"""

from typing import TypeAlias
from dataclasses import field, dataclass
from collections.abc import Hashable

from kirin import ir, types
from kirin.dialects import py, ilist
from kirin.dialects.py import slice as py_slice

from bloqade.types import QubitType
from bloqade.constants import constant_int
from bloqade.squin.wire.types import WireType, RegisterType, contains_linear
from bloqade.analysis.reference.index import (
    Slice,
    Affine,
    IndexAnalysis,
    same,
    python_slice,
)
from bloqade.squin.analysis.reference import holds_qubits


def quantum_type(kind: types.TypeAttribute) -> types.TypeAttribute | None:
    """Return the linear type of a scalar qubit or a nested qubit register."""
    if isinstance(kind, types.BottomType):
        return None
    if kind.is_subseteq(QubitType) or kind.is_subseteq(WireType):
        return WireType
    if kind.is_subseteq(RegisterType):
        return RegisterType
    if isinstance(kind, types.Generic) and kind.is_subseteq(ilist.IListType):
        if quantum_type(kind.vars[0]) is not None:
            return RegisterType
    return None


def typed_length(kind: types.TypeAttribute) -> int | None:
    """Return a list's declared constant length, if its type supplies one."""
    if isinstance(kind, types.Generic) and kind.is_subseteq(ilist.IListType):
        length = kind.vars[1]
        if isinstance(length, types.Literal) and isinstance(length.data, int):
            return length.data
    return None


@dataclass(frozen=True)
class Handle:
    """Identify a whole allocation, one register element, or a register view."""

    root: Hashable
    kind: types.TypeAttribute
    length: Affine | None = None
    selection: Affine | Slice | None = None
    shape: tuple[int | None, ...] = ()

    @property
    def scalar(self) -> bool:
        """Return whether this handle names one qubit."""
        return self.kind == WireType or isinstance(self.selection, Affine)

    @property
    def size(self) -> Affine | None:
        """Return the register view's length when it has a known expression."""
        if len(self.shape) > 1:
            return None if self.shape[0] is None else Affine(self.shape[0])
        return (
            self.selection.length if isinstance(self.selection, Slice) else self.length
        )

    def pick(self, index: Affine) -> "Descriptor":
        """Resolve an element index through this register or view."""
        if self.scalar:
            return Unknown(frozenset({self.root}))
        size = self.size
        if index.constant is not None and index.const < 0:
            if size is None:
                return Unknown(
                    frozenset({self.root}),
                    "a negative index into a register of unknown length",
                )
            index = size + index
        if (
            size is not None
            and size.constant is not None
            and index.constant is not None
        ):
            if not 0 <= index.const < size.const:
                return Unknown(frozenset({self.root}))
        if len(self.shape) > 1:
            width = 1
            for dimension in self.shape[1:]:
                if dimension is None:
                    return Unknown(frozenset({self.root}))
                width *= dimension
            start = index.scale(width)
            if isinstance(self.selection, Slice):
                start = self.selection.item(start)
            return Handle(
                self.root,
                self.kind,
                self.length,
                Slice(start, 1, Affine(width)),
                self.shape[1:],
            )
        if isinstance(self.selection, Slice):
            index = self.selection.item(index)
        return Handle(self.root, self.kind, self.length, index)


@dataclass(frozen=True)
class Unknown:
    """Retain possible allocation roots when a reference expression cannot resolve."""

    roots: frozenset[Hashable] | None = None
    reason: str = "an unresolved reference"


Descriptor: TypeAlias = Handle | Unknown | tuple["Descriptor", ...] | None


def roots(value: Descriptor) -> set[Hashable] | None:
    """Return possible roots, or None when an expression can name any allocation."""
    if isinstance(value, Handle):
        return {value.root}
    if isinstance(value, Unknown):
        return None if value.roots is None else set(value.roots)
    found: set[Hashable] = set()
    for item in value or ():
        members = roots(item)
        if members is None:
            return None
        found.update(members)
    return found


def merge(left: Descriptor, right: Descriptor) -> Descriptor:
    """Keep matching handles and retain possible roots for differing branch results."""
    if left == right:
        return left
    a, b = roots(left), roots(right)
    return Unknown(
        None if a is None or b is None else frozenset(a | b),
        "a value carried by a loop or branch",
    )


def distinct(left: Handle, right: Handle) -> bool:
    """Prove separation using allocation identities or normalized element indices."""
    if left.root != right.root:
        return True
    if isinstance(left.selection, Affine) and isinstance(right.selection, Affine):
        return same(left.selection, right.selection) is False
    return False


@dataclass
class Handles:
    """Record handles when their defining statements receive a conversion visit."""

    values: dict[Hashable, Descriptor] = field(default_factory=dict)
    indices: IndexAnalysis = field(default_factory=IndexAnalysis)
    extents: dict[Hashable, Affine] = field(default_factory=dict)

    def get(self, value: Hashable) -> Descriptor:
        """Read a recorded handle or conservatively classify an unresolved value."""
        if value in self.values:
            return self.values[value]
        if not isinstance(value, ir.SSAValue):
            return Unknown()
        return (
            Unknown()
            if holds_qubits(value.type) or contains_linear(value.type)
            else None
        )

    def size(self, value: ir.SSAValue) -> Affine | None:
        """Return a container's length from its handle or declared type."""
        handle = self.get(value)
        if isinstance(handle, tuple):
            return Affine(len(handle))
        if isinstance(handle, Handle) and not handle.scalar:
            return handle.size
        length = typed_length(value.type)
        return None if length is None else Affine(length)

    def items(self, value: ir.SSAValue) -> tuple[Descriptor, ...] | None:
        """Resolve the elements of a statically sized quantum container."""
        handle = self.get(value)
        if isinstance(handle, tuple):
            return handle
        size = self.size(value)
        if (
            isinstance(handle, Handle)
            and size is not None
            and size.constant is not None
        ):
            return tuple(handle.pick(Affine(i)) for i in range(max(0, size.const)))
        return None

    def record(self, node: ir.Statement) -> bool:
        """Resolve aliases and container expressions from previously visited operands."""
        if isinstance(node, py.assign.Alias):
            value = self.get(node.value)
        elif isinstance(node, (ilist.New, py.tuple.New)):
            value = tuple(self.get(arg) for arg in node.args)
        elif isinstance(node, py.Add):
            left, right = self.items(node.lhs), self.items(node.rhs)
            if left is None or right is None:
                return False
            value = left + right
        elif isinstance(node, py.GetItem):
            obj = self.get(node.obj)
            if node.index.type.is_subseteq(types.Slice):
                value = self.slice(obj, node.index)
            else:
                index = self.indices.of(node.index)
                if isinstance(obj, Handle):
                    value = obj.pick(index)
                elif isinstance(obj, tuple) and index.constant is not None:
                    value = (
                        obj[index.const]
                        if -len(obj) <= index.const < len(obj)
                        else Unknown()
                    )
                elif obj is None:
                    value = None
                else:
                    possible = roots(obj)
                    value = Unknown(None if possible is None else frozenset(possible))
        else:
            return False
        self.values[node.results[0]] = value
        return True

    def slice(self, obj: Descriptor, index: ir.SSAValue) -> Descriptor:
        """Compose a Python slice with a register view when its bounds resolve."""
        owner = index.owner
        possible = roots(obj)
        unknown = Unknown(None if possible is None else frozenset(possible))

        def bound(value):
            if value.type.is_subseteq(types.NoneType):
                return None
            return self.indices.of(value)

        if isinstance(owner, py_slice.Slice):
            start, stop = bound(owner.start), bound(owner.stop)
            step = (
                1
                if owner.step.type.is_subseteq(types.NoneType)
                else constant_int(owner.step)
            )
        elif isinstance(owner, py.Constant) and isinstance(
            value := owner.value.unwrap(), slice
        ):
            start = None if value.start is None else Affine(value.start)
            stop = None if value.stop is None else Affine(value.stop)
            step = 1 if value.step is None else value.step
        else:
            return unknown
        if step is None or step == 0:
            return unknown
        if isinstance(obj, tuple):
            if all(item is None or item.constant is not None for item in (start, stop)):
                return obj[
                    slice(
                        None if start is None else start.const,
                        None if stop is None else stop.const,
                        step,
                    )
                ]
            return unknown
        if not isinstance(obj, Handle) or obj.scalar:
            return unknown
        size = obj.size
        if len(obj.shape) > 1:
            if (
                size is None
                or size.constant is None
                or any(
                    item is not None and item.constant is None for item in (start, stop)
                )
            ):
                return unknown
            selected = range(size.const)[
                slice(
                    None if start is None else start.const,
                    None if stop is None else stop.const,
                    step,
                )
            ]
            return tuple(obj.pick(Affine(i)) for i in selected)
        view = python_slice(start, stop, step, None if size is None else size.constant)
        if view is None:
            return unknown
        if isinstance(obj.selection, Slice):
            view = obj.selection.then(view)
        return unknown if view is None else Handle(obj.root, obj.kind, obj.length, view)


def register_shape(kind: types.TypeAttribute) -> tuple[int | None, ...]:
    """Return the source register dimensions in outer-to-inner order."""
    shape = []
    while isinstance(kind, types.Generic) and kind.is_subseteq(ilist.IListType):
        shape.append(typed_length(kind))
        kind = kind.vars[0]
    return tuple(shape)
