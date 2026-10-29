"""Resolve source qubit references and copy classical operations."""

from kirin import ir, types, interp
from kirin.dialects import py, ilist
from kirin.dialects.py import len as py_len, slice as py_slice

from bloqade import qubit
from bloqade.types import holds_qubits
from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Affine
from bloqade.squin.wire.errors import WireConversionError

from .base import EmitWire, WireFrame, index_of
from .values import Location, contains_location


@qubit.dialect.register(key="emit.squin_wire")
class Qubits(interp.MethodTable):
    """Emit standalone qubit allocations."""

    @interp.impl(qubit.stmts.New)
    def new(self, emit: EmitWire, frame: WireFrame, stmt):
        """Create a qubit and record its lifetime in the current region."""
        root = frame.push(stmts.Alloc()).results[0]
        frame.latest[root] = root
        frame.local_roots.append(root)
        return (Location(root),)


@py.assign.dialect.register(key="emit.squin_wire")
class Aliases(interp.MethodTable):
    """Resolve source aliases from their previously emitted operands."""

    @interp.impl(py.assign.Alias)
    def alias(self, emit: EmitWire, frame: WireFrame, stmt):
        """Retain the source value's emitted value or quantum location."""
        return (frame.get(stmt.value),)


@ilist.dialect.register(key="emit.squin_wire")
class Lists(interp.MethodTable):
    """Retain quantum lists as location sequences and emit classical lists."""

    @interp.impl(ilist.New)
    def new(self, emit: EmitWire, frame: WireFrame, stmt):
        """Build a source container or copy a classical list into the output."""
        values = frame.get_values(stmt.values)
        if holds_qubits(stmt.result.type) or any(
            contains_location(value) for value in values
        ):
            return (values,)
        return emit.eval_fallback(frame, stmt)


@py.tuple.dialect.register(key="emit.squin_wire")
class Tuples(interp.MethodTable):
    """Keep source tuple fields available for indexing and function returns."""

    @interp.impl(py.tuple.New)
    def new(self, emit: EmitWire, frame: WireFrame, stmt):
        """Record tuple fields without emitting a redundant source container."""
        return (frame.get_values(stmt.args),)


@py.binop.dialect.register(key="emit.squin_wire")
class Arithmetic(interp.MethodTable):
    """Resolve quantum list concatenation and emit ordinary arithmetic."""

    @interp.impl(py.Add)
    def add(self, emit: EmitWire, frame: WireFrame, stmt):
        """Concatenate known quantum sequences or copy the classical addition."""
        left, right = frame.get_values(stmt.args)
        if contains_location(left) or contains_location(right):
            a, b = emit.known_items(left, stmt), emit.known_items(right, stmt)
            if a is None or b is None:
                raise WireConversionError(
                    "Quantum concatenation requires known source lengths.", stmt
                )
            return (a + b,)
        return emit.eval_fallback(frame, stmt)


@py_len.dialect.register(key="emit.squin_wire")
class Lengths(interp.MethodTable):
    """Emit the length of a source sequence."""

    @interp.impl(py_len.Len)
    def length(self, emit: EmitWire, frame: WireFrame, stmt):
        """Read the source sequence's known or runtime length."""
        return (emit.length_value(frame, frame.get(stmt.value)),)


@py.indexing.dialect.register(key="emit.squin_wire")
class Indexing(interp.MethodTable):
    """Select source quantum locations and emit ordinary classical indexing."""

    @interp.impl(py.GetItem)
    def getitem(self, emit: EmitWire, frame: WireFrame, stmt):
        """Resolve one source element or slice using already emitted operands."""
        obj, at = frame.get_values(stmt.args)
        index = emit.classical_value(frame, at)
        if isinstance(obj, ir.SSAValue):
            return emit.eval_fallback(frame, stmt)
        if index.type.is_subseteq(types.Slice):
            start, stop, step = slice_bounds(index, stmt)
            if isinstance(obj, Location):
                return (obj.slice(start, stop, step, stmt),)
            if isinstance(obj, tuple) and all(
                bound is None or bound.constant is not None for bound in (start, stop)
            ):
                return (
                    obj[
                        slice(
                            None if start is None else start.constant,
                            None if stop is None else stop.constant,
                            step,
                        )
                    ],
                )
        else:
            position = index_of(index)
            if isinstance(obj, Location):
                return (obj.select(position, stmt),)
            if (
                isinstance(obj, tuple)
                and (item := position.constant) is not None
                and -len(obj) <= item < len(obj)
            ):
                return (obj[item],)
            if isinstance(obj, tuple) and not contains_location(obj):
                return (
                    frame.push(
                        py.GetItem(emit.classical_value(frame, obj), index)
                    ).results[0],
                )
        raise WireConversionError(
            "A quantum container requires a tuple read at a constant index or a resolved register element.",
            stmt,
        )


def slice_bounds(value: ir.SSAValue, at: ir.Statement):
    """Read supported slice bounds from an emitted slice operation."""
    node = value.owner
    if isinstance(node, py_slice.Slice):
        start = (
            None
            if node.start.type.is_subseteq(types.NoneType)
            else index_of(node.start)
        )
        stop = (
            None if node.stop.type.is_subseteq(types.NoneType) else index_of(node.stop)
        )
        step = (
            1 if node.step.type.is_subseteq(types.NoneType) else constant_int(node.step)
        )
    elif isinstance(node, py.Constant) and isinstance(
        sliced := node.value.unwrap(), slice
    ):
        start = None if sliced.start is None else Affine(sliced.start)
        stop = None if sliced.stop is None else Affine(sliced.stop)
        step = 1 if sliced.step is None else sliced.step
    else:
        raise WireConversionError("The slice bounds have no supported source form.", at)
    if step is None or step == 0:
        raise WireConversionError(
            "A quantum slice requires a nonzero constant step.", at
        )
    return start, stop, step
