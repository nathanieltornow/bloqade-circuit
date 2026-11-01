"""Emit fresh wire IR while Kirin visits unchanged source operations."""

from typing import TypeVar
from dataclasses import field, replace, dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.emit import EmitABC, EmitFrame
from kirin.dialects import py, func
from kirin.emit.abc import EmitTable
from kirin.worklist import WorkList
from kirin.dialects.py.len import Len

from bloqade.types import holds_qubits
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Sym, Iter, Emitted, Indices, IndexExpr
from bloqade.squin.wire.group import reference_kernel
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError

from .values import (
    Grid,
    Rows,
    View,
    Qubit,
    Value,
    Element,
    Location,
    Register,
    select,
    row_width,
    contains_location,
)

StatementT = TypeVar("StatementT", bound=ir.Statement)


@dataclass
class WireFrame(EmitFrame[Value]):
    """Hold the emission state of one source region.

    The inherited `entries` map each source value to its meaning: an emitted
    classical value, a quantum `Location` or a tuple of meanings.
    """

    output: ir.Block = field(default_factory=ir.Block)
    """The wire block that this frame emits."""
    latest: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict)
    """The latest wire or register value of each allocation root.

    An operation that consumes a qubit replaces the entry of its root. A root has
    no entry while an operation holds its whole value.
    """
    local_roots: list[ir.SSAValue] = field(default_factory=list)
    """The allocation roots that this region allocated and must free at its end."""
    site: func.Invoke | None = None
    """The user call that runs the library wrapper of this frame, or None in user code.

    Each operation that a library wrapper emits takes its source location from this
    call, so that diagnostics point at user code.
    """

    def push(self, node: StatementT) -> StatementT:
        """Append an operation and give it the source location of the current operation."""
        source = self.site or self.current_stmt
        if source is not None and source.source is not None:
            node.source = replace(source.source)
        self.output.stmts.append(node)
        return node

    def free_local_allocations(self) -> None:
        """Free each local allocation that the region still holds at its end."""
        for root in self.local_roots:
            if (value := self.latest.pop(root, None)) is not None:
                self.push(
                    stmts.Free(value)
                    if value.type == WireType
                    else stmts.FreeRegister(value)
                )


@dataclass
class EmitWire(EmitABC[WireFrame, Value]):
    """Translate source operations with method tables and explicit per-region wire state."""

    keys = ("emit.squin_wire",)
    void = None
    source: ir.Method
    converted: Mapping[ir.Method, ir.Method]
    """The wire method of each user callee, which the driver converts first."""
    result: ir.Method | None = field(default=None, init=False)
    """The emitted wire method, which the function handler sets at its end."""
    indices: Indices = field(init=False)
    """The index expression of each source integer, whose symbols are source values."""

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> WireFrame:
        """Create a region frame whose source values can access its enclosing frame."""
        return WireFrame(node, has_parent_access=has_parent_access)

    def reset(self) -> None:
        """Reset Kirin's callable queue and the result of this emission."""
        self.callables = EmitTable(prefix="")
        self.callable_to_emit = WorkList()
        self.result = None
        self.indices = Indices(reference_kernel)

    def emit_method(self) -> ir.Method:
        """Emit one source method using the wire methods of its callees.

        Kirin's `run` evaluates `source.code`, which dispatches to the function
        handler in `functions.py`. That handler sets `result`.
        """
        self.run(self.source)
        assert self.result is not None
        return self.result

    def classical_value(self, frame: WireFrame, value: Value) -> ir.SSAValue:
        """Return the emitted classical value of a meaning.

        A tuple of meanings becomes an emitted tuple.
        """
        if isinstance(value, ir.SSAValue):
            return value
        if value is None:
            raise WireConversionError(
                "A library wrapper cannot use its own function object.",
                frame.current_stmt,
            )
        if isinstance(value, tuple):
            return frame.push(
                py.tuple.New(tuple(self.classical_value(frame, x) for x in value))
            ).results[0]
        raise WireConversionError(
            "A classical operation cannot consume a quantum reference.",
            frame.current_stmt,
        )

    def emit_index(self, frame: WireFrame, expression: IndexExpr) -> ir.SSAValue:
        """Emit the integer operations that compute an index expression."""
        if (value := expression.constant) is not None:
            return frame.push(py.Constant(value)).results[0]
        total = None
        for symbol, coefficient in expression.terms:
            term = self.emitted_symbol(frame, symbol)
            if coefficient != 1:
                constant = frame.push(py.Constant(coefficient)).results[0]
                term = self._int_op(frame, py.Mult, term, constant)
            total = term if total is None else self._int_op(frame, py.Add, total, term)
        assert total is not None
        if expression.offset:
            constant = frame.push(py.Constant(expression.offset)).results[0]
            total = self._int_op(frame, py.Add, total, constant)
        return total

    def index(self, value: ir.SSAValue) -> IndexExpr:
        """Return the index expression of the source integer `value`.

        The symbols of the index are source values. `emit_index` maps them to
        emitted values through the frame.
        """
        return self.indices.of(value)

    def emitted_symbol(self, frame: WireFrame, symbol: Sym) -> ir.SSAValue:
        """Return the emitted integer of a source index symbol.

        An `Iter` symbol maps to the emitted loop index, and an `Emitted` symbol is
        already emitted.
        """
        if isinstance(symbol, Emitted):
            return symbol.value
        if isinstance(symbol, Iter):
            source: ir.SSAValue = symbol.stmt.regions[0].blocks[0].args[0]
        else:
            source = symbol
        value = frame.get(source)
        if not isinstance(value, ir.SSAValue):
            raise WireConversionError(
                "An emitted index requires output SSA symbols.", frame.current_stmt
            )
        return value

    def _int_op(
        self,
        frame: WireFrame,
        kind: type[py.Add] | type[py.Mult] | type[py.FloorDiv],
        left: ir.SSAValue,
        right: ir.SSAValue,
    ) -> ir.SSAValue:
        """Emit one integer operation and give its result the type `int`."""
        node = frame.push(kind(left, right))
        node.results[0].type = types.Int
        return node.results[0]

    def acquire(self, frame: WireFrame, location: Location) -> ir.SSAValue:
        """Return the wire value of a location for an operation that consumes it.

        A whole allocation leaves `latest` until `restore`. An element emits a `Borrow`,
        and a slice with step 1 emits a `TakeSlice`. `restore` emits the matching
        `Release` or `PutSlice`.
        """
        value = frame.latest.get(location.root)
        if value is None:
            raise WireConversionError(
                "An operation cannot use one qubit or register twice.",
                frame.current_stmt,
            )
        match location:
            case Element(root, index):
                node = frame.push(stmts.Borrow(value, self.emit_index(frame, index)))
                frame.latest[root] = node.results[0]
                return node.results[1]
            case View(root, span) if span.step == 1:
                start = self.emit_index(frame, span.start)
                size = self.length_value(frame, location)
                node = frame.push(stmts.TakeSlice(value, start, size))
                frame.latest[root] = node.register_out
                return node.slice_register
            case View() | Rows():
                raise WireConversionError(
                    "A whole-register operation requires a whole allocation.",
                    frame.current_stmt,
                )
            case Qubit(root) | Register(root) | Grid(root):
                del frame.latest[root]
                return value

    def restore(self, frame: WireFrame, location: Location, value: ir.SSAValue) -> None:
        """Record the next value of a location after the operation that consumed it."""
        register = frame.latest.get(location.root)
        match location:
            case Element(index=index):
                assert register is not None
                at = self.emit_index(frame, index)
                value = frame.push(stmts.Release(register, at, value)).register_out
            case View(span=span):
                assert register is not None
                at = self.emit_index(frame, span.start)
                value = frame.push(stmts.PutSlice(register, at, value)).register_out
        frame.latest[location.root] = value

    def length_value(self, frame: WireFrame, value: Value) -> ir.SSAValue:
        """Return the length of a source container as an emitted integer.

        A constant or slice length becomes an index expression. Any other register
        length emits `Length`, because a symbolic length such as `n` in `qalloc(n)` can
        be negative.
        """
        if isinstance(value, tuple):
            return self.emit_index(frame, IndexExpr(len(value)))
        match value:
            case View(length=length) | Rows(length=length) if length is not None:
                return self.emit_index(frame, length)
            case View() | Rows() | Element() | Qubit():
                raise WireConversionError(
                    "The register view requires a known length.", frame.current_stmt
                )
            case Register(length=length) | Grid(length=length) if (
                length is not None and length.constant is not None
            ):
                return self.emit_index(frame, length)
            case Register(root):
                return self.register_length(frame, root)
            case Grid(root, _, row_shape):
                # `Length` counts qubits, and each row holds `row_width` of them.
                qubits = self.register_length(frame, root)
                width = frame.push(
                    py.Constant(row_width(row_shape, frame.current_stmt))
                ).results[0]
                return self._int_op(frame, py.FloorDiv, qubits, width)
        return frame.push(Len(self.classical_value(frame, value))).results[0]

    def register_length(self, frame: WireFrame, root: ir.SSAValue) -> ir.SSAValue:
        """Emit `Length` on the latest value of `root` and return the qubit count."""
        node = frame.push(stmts.Length(frame.latest[root]))
        frame.latest[root] = node.results[0]
        return node.results[1]

    def known_items(self, value: Value, at: ir.Statement) -> tuple[Value, ...] | None:
        """Return the item meanings of a container whose length is known, or None."""
        if isinstance(value, tuple):
            return value
        if isinstance(value, (Register, View, Grid, Rows)) and value.length is not None:
            if (count := value.length.constant) is not None:
                return tuple(
                    select(value, IndexExpr(i), at) for i in range(max(0, count))
                )
        return None

    def eval_fallback(self, frame: WireFrame, node: ir.Statement) -> tuple[Value, ...]:
        """Emit declared quantum twins or copy supported classical operations with mapped operands."""
        if type(node) in TWINS:
            # `gates` imports this module, so this import must stay local.
            from .gates import emit_gate

            return emit_gate(self, frame, node)
        values = frame.get_values(node.args)
        if (
            node.regions
            or any(contains_location(value) for value in values)
            or any(holds_qubits(result.type) for result in node.results)
        ):
            raise WireConversionError(
                f"Unsupported reference operation '{node.name}'.", node
            )
        made = node.from_stmt(
            node, args=tuple(self.classical_value(frame, value) for value in values)
        )
        frame.push(made)
        return tuple(made.results)
