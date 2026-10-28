"""Emit Jeff directly from explicit squin wire values through Kirin method tables."""

import math
from typing import TypeVar, TypeAlias
from dataclasses import field, dataclass
from collections.abc import Sequence

from kirin import ir, types, interp
from kirin.emit import EmitABC, EmitFrame
from kirin.dialects import func, ilist, ssacfg
from kirin.emit.abc import EmitTable
from kirin.worklist import WorkList

from bloqade.types import MeasurementResultType
from bloqade.jeff.types import WireType, QuregType, IntArrayType, FloatArrayType, is_bit
from bloqade.jeff.dialects import stmts, kernel as jeff_kernel
from bloqade.squin.wire.types import WireType as SquinWireType, RegisterType
from bloqade.jeff.dialects.stmts.call import declared_outputs

KEY = "emit.wire2jeff"
Value: TypeAlias = "ir.SSAValue | tuple[Value, ...] | None"
_S = TypeVar("_S", bound=ir.Statement)


def jeff_kind(kind: types.TypeAttribute) -> types.TypeAttribute:
    """Return the jeff type of a value of the squin type `kind`."""
    if isinstance(kind, types.Literal):
        match kind.data:
            case bool():
                return types.Bool
            case int():
                return types.Int
            case float():
                return types.Float
            case ilist.IList(data=list(items)) if any(
                isinstance(item, float) for item in items
            ):
                return FloatArrayType
            case ilist.IList():
                return IntArrayType
    if kind.is_subseteq(SquinWireType):
        return WireType
    if kind.is_subseteq(MeasurementResultType):
        # Jeff measures to one bit, so the conversion assumes that no qubit is lost.
        return types.Bool
    if kind.is_subseteq(RegisterType):
        return QuregType
    if kind.is_subseteq(ilist.IListType[types.Float, types.Any]):
        return FloatArrayType
    if kind.is_subseteq(ilist.IListType):
        return IntArrayType
    return kind


def elements(kind: types.TypeAttribute) -> types.TypeAttribute:
    """Return the element type of the list type `kind`, or `Any`."""
    if isinstance(kind, types.Generic) and kind.is_subseteq(ilist.IListType):
        return kind.vars[0]
    return types.Any


def bitwidth(elements: types.TypeAttribute) -> int:
    """Return the jeff bitwidth of an integer array with the element type `elements`.

    A measurement result or a boolean is one bit, any other integer is 32 bits.
    """
    if elements.is_subseteq(MeasurementResultType) or elements.is_subseteq(types.Bool):
        return 1
    return 32


@dataclass
class Frame(EmitFrame[Value]):
    """Hold the jeff block and the value mapping for one source region."""

    body: ir.Block = field(default_factory=ir.Block, kw_only=True)
    """The jeff block under construction."""

    def push(self, stmt: _S) -> _S:
        """Append a jeff statement to the block and return it.

        An integer operation on bits gives a bit, which the statement types `Bool`.
        """
        if isinstance(stmt, (stmts.IntUnary, stmts.IntBinary)) and is_bit(stmt.args[0]):
            stmt.results[0].type = types.Bool
        self.body.stmts.append(stmt)
        return stmt

    def value(self, value: ir.SSAValue) -> Value:
        """Return the emitted value, including values captured from an enclosing region.

        A region reads the values of its parent frame directly, and
        `IsolateRegions` later makes them inputs of the region.
        """
        if value in self.entries:
            return self.entries[value]
        if not isinstance(self.parent, Frame):
            raise interp.InterpreterError(f"{value} has no jeff value")
        return self.parent.value(value)

    def values(self, values: Sequence[ir.SSAValue]) -> tuple[Value, ...]:
        """Return the jeff value of each squin value in `values`."""
        return tuple(self.value(v) for v in values)

    def scalar(self, value: ir.SSAValue) -> ir.SSAValue:
        """Return the jeff value of the classical squin value `value`."""
        jeff = self.value(value)
        if not isinstance(jeff, ir.SSAValue):
            raise interp.InterpreterError(f"{value} has no jeff value")
        return jeff


@dataclass
class WireEmitter(EmitABC[Frame, Value]):
    """Map each wire value to one Jeff value without reconstructing reference identities."""

    keys = (KEY,)
    void = None
    dialects: ir.DialectGroup
    functions: dict[ir.Statement, ir.Method] = field(default_factory=dict, init=False)

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> Frame:
        """Create a frame for an emitted function or region."""
        return Frame(node, has_parent_access=has_parent_access)

    def reset(self) -> None:
        """Clear declarations and pending functions for a new emission."""
        self.callables = EmitTable(prefix="")
        self.callable_to_emit = WorkList()
        self.functions = {}

    def declare(self, code: func.Function) -> ir.Method:
        """Declare the Jeff signature in the wire function's input and output order."""
        if code in self.functions:
            return self.functions[code]
        inputs = tuple(jeff_kind(kind) for kind in code.signature.inputs)
        outputs = tuple(
            jeff_kind(kind) for kind in declared_outputs(code.signature.output)
        )
        output = (
            types.NoneType
            if not outputs
            else outputs[0] if len(outputs) == 1 else types.Generic(tuple, *outputs)
        )
        block = ir.Block(argtypes=(types.MethodType[list(inputs), output], *inputs))
        made = func.Function(
            sym_name=code.sym_name,
            body=ir.Region(block),
            signature=func.Signature(inputs=inputs, output=output),
        )
        method = ir.Method(dialects=jeff_kernel, code=made, sym_name=code.sym_name)
        self.functions[code] = method
        return method

    def eval_fallback(self, frame: Frame, node: ir.Statement):
        """Reject every operation without an explicit Jeff emission rule."""
        raise interp.InterpreterError(f"Jeff has no form for '{node.name}'.")

    def leave(self, inner: Frame, outs: Sequence[ir.SSAValue]) -> ir.Region:
        """Terminate an emitted region with its explicit result values."""
        inner.push(stmts.Yield(*outs))
        return ir.Region(inner.body)

    def fitted(
        self, frame: Frame, value: ir.SSAValue, kind: types.TypeAttribute
    ) -> ir.SSAValue:
        """Return the jeff `value` as the squin type `kind` expects it.

        Squin joins a bit and an integer to an integer. Jeff has no conversion,
        so a bit in an integer position becomes `int_select(bit, 1, 0)`.
        """
        if is_bit(value) and jeff_kind(kind) == types.Int:
            one = frame.push(stmts.ConstInt(value=1)).result
            zero = frame.push(stmts.ConstInt(value=0)).result
            return frame.push(stmts.IntSelect(value, one, zero)).result
        return value

    def negate(self, frame: Frame, value: ir.SSAValue) -> ir.SSAValue:
        """Return the negation of the bit `value`, or of each bit of the bit array."""
        bits = frame.scalar(value)
        if not value.type.is_subseteq(ilist.IListType):
            return frame.push(stmts.IntNot(bits)).result
        size = value.type.vars[1] if isinstance(value.type, types.Generic) else None
        if not isinstance(size, types.Literal):
            raise interp.InterpreterError(f"{value} is a bit array of unknown length")
        negated: list[ir.SSAValue] = []
        for position in range(size.data):
            at = frame.push(stmts.ConstInt(value=position)).result
            bit = frame.push(stmts.IntArrayGet(bits, at, bitwidth=1)).result
            negated.append(frame.push(stmts.IntNot(bit)).result)
        return frame.push(stmts.IntArrayCreate(tuple(negated), bitwidth=1)).result

    def radians(self, frame: Frame, turns: ir.SSAValue) -> ir.SSAValue:
        """Return the angle in radians of the squin gate statement angle `turns`."""
        full = frame.push(stmts.ConstFloat(value=2 * math.pi)).result
        return frame.push(stmts.FloatMul(frame.scalar(turns), full)).result


@ssacfg.dialect.register(key=KEY)
class _SsaCfg(ssacfg.Concrete):
    """Run the statements of an SSA control-flow region with kirin's concrete rule."""
