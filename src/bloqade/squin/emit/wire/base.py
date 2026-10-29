"""Emit fresh wire IR while Kirin visits unchanged source operations."""

from typing import TypeVar
from dataclasses import field, replace, dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.emit import EmitABC, EmitFrame
from kirin.dialects import py, scf, func
from kirin.emit.abc import EmitTable
from kirin.worklist import WorkList

from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Iter, Slice, Affine, IndexAnalysis
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.signature import Contract

from .values import Value, Location

StatementT = TypeVar("StatementT", bound=ir.Statement)


@dataclass
class WireFrame(EmitFrame[Value]):
    """Store source meanings, current wires and allocations local to one output block."""

    output: ir.Block = field(default_factory=ir.Block)
    current: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict)
    local_roots: list[ir.SSAValue] = field(default_factory=list)

    def push(self, node: StatementT) -> StatementT:
        """Append an operation and retain the current source operation's diagnostic location."""
        source = self.code if isinstance(self.code, func.Invoke) else self.current_stmt
        if source is not None and source.source is not None:
            node.source = replace(source.source)
        self.output.stmts.append(node)
        return node

    def finish(self) -> None:
        """Free local allocations whose current wires remain in this block."""
        for root in self.local_roots:
            if (value := self.current.pop(root, None)) is not None:
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
    contracts: Mapping[ir.Method, Contract]
    result: Contract | None = field(default=None, init=False)

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

    def emit_method(self) -> Contract:
        """Emit one source method using the supplied checked callee contracts."""
        self.run(self.source)
        assert self.result is not None
        return self.result

    def classical(self, frame: WireFrame, value: Value) -> ir.SSAValue:
        """Require an emitted classical operand or construct its classical tuple."""
        if isinstance(value, ir.SSAValue):
            return value
        if value is None:
            return frame.push(func.ConstantNone()).results[0]
        if isinstance(value, tuple):
            return frame.push(
                py.tuple.New(tuple(self.classical(frame, x) for x in value))
            ).results[0]
        raise WireConversionError(
            "A classical operation cannot consume a quantum reference.",
            frame.current_stmt,
        )

    def integer(self, frame: WireFrame, expression: Affine) -> ir.SSAValue:
        """Materialize an integer expression whose symbols already belong to the output IR."""
        if expression.constant is not None:
            return frame.push(py.Constant(expression.const)).results[0]
        total = None
        for symbol, coefficient in expression.terms:
            if isinstance(symbol, Iter) and isinstance(symbol.stmt, scf.For):
                symbol = symbol.stmt.body.blocks[0].args[0]
            if not isinstance(symbol, ir.SSAValue):
                raise WireConversionError(
                    "An emitted index requires output SSA symbols.", frame.current_stmt
                )
            term = symbol
            if coefficient != 1:
                constant = frame.push(py.Constant(coefficient)).results[0]
                term = self.binary(frame, py.Mult, term, constant)
            total = term if total is None else self.binary(frame, py.Add, total, term)
        assert total is not None
        if expression.const:
            constant = frame.push(py.Constant(expression.const)).results[0]
            total = self.binary(frame, py.Add, total, constant)
        return total

    def binary(
        self, frame: WireFrame, kind, left: ir.SSAValue, right: ir.SSAValue
    ) -> ir.SSAValue:
        """Emit integer arithmetic with the result type required by wire indices."""
        node = frame.push(kind(left, right))
        node.results[0].type = types.Int
        return node.results[0]

    def take(self, frame: WireFrame, location: Location) -> ir.SSAValue:
        """Consume a whole current value or borrow one selected register element."""
        value = frame.current.get(location.root)
        if value is None:
            raise WireConversionError(
                "Emission requires an available wire at this operation.",
                frame.current_stmt,
            )
        if isinstance(location.selection, Affine):
            node = frame.push(
                stmts.Borrow(value, self.integer(frame, location.selection))
            )
            frame.current[location.root] = node.results[0]
            return node.results[1]
        if location.selection is not None:
            raise WireConversionError(
                "A whole-register operation requires a whole allocation.",
                frame.current_stmt,
            )
        del frame.current[location.root]
        return value

    def restore(self, frame: WireFrame, location: Location, value: ir.SSAValue) -> None:
        """Record an updated whole value or restore an element to its register."""
        if isinstance(location.selection, Affine):
            register = frame.current[location.root]
            value = frame.push(
                stmts.Release(register, self.integer(frame, location.selection), value)
            ).results[0]
        frame.current[location.root] = value

    def length(self, frame: WireFrame, value: Value) -> ir.SSAValue:
        """Read a source container length or the current wire register's runtime length."""
        if isinstance(value, tuple):
            return self.integer(frame, Affine(len(value)))
        if isinstance(value, Location):
            if value.size is not None and (
                value.size.constant is not None or isinstance(value.selection, Slice)
            ):
                return self.integer(frame, value.size)
            if value.selection is not None:
                raise WireConversionError(
                    "The register view requires a known length.", frame.current_stmt
                )
            node = frame.push(stmts.Length(frame.current[value.root]))
            frame.current[value.root] = node.results[0]
            return node.results[1]
        from kirin.dialects.py.len import Len

        return frame.push(Len(self.classical(frame, value))).results[0]

    def items(self, value: Value, at: ir.Statement) -> tuple[Value, ...] | None:
        """Expand source containers whose logical lengths are known during compilation."""
        if isinstance(value, tuple):
            return value
        if (
            isinstance(value, Location)
            and value.size is not None
            and value.size.constant is not None
        ):
            return tuple(
                value.select(Affine(i), at) for i in range(max(0, value.size.const))
            )
        return None

    def eval_fallback(self, frame: WireFrame, node: ir.Statement):
        """Emit declared quantum twins or copy supported classical operations with mapped operands."""
        if type(node) in TWINS:
            from .gates import emit_gate

            return emit_gate(self, frame, node)
        from bloqade.types import holds_qubits

        from .values import contains_location

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
            node, args=tuple(self.classical(frame, value) for value in values)
        )
        frame.push(made)
        return tuple(made.results)


def index_of(value: ir.SSAValue) -> Affine:
    """Read integer arithmetic from emitted SSA definitions."""
    return IndexAnalysis().of(value)
