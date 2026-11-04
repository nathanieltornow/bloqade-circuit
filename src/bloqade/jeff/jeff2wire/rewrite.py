"""Replace Jeff quantum operations and calls with explicit squin wire operations."""

import math
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from kirin import ir, types
from kirin.dialects import py, func, ilist
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.jeff.gates import canonical
from bloqade.jeff.types import (
    WireType as JeffWireType,
    QuregType,
    IntArrayType,
    FloatArrayType,
    is_subtype,
    qureg_length,
)
from bloqade.squin.gate import stmts as gates
from bloqade.squin.wire import stmts as wire
from bloqade.jeff.dialects import stmts
from bloqade.squin.wire.types import WireType, register_type
from bloqade.squin.rewrite.wire.driver import put, pack, replace
from bloqade.squin.rewrite.wire.sources import ENTRIES
from bloqade.squin.rewrite.wire.signature import packed_type
from bloqade.jeff.analysis.validation.to_squin.gates import (
    SINGLE,
    ROTATION,
    CONTROLLED,
    NON_HERMITIAN,
    gate_supported,
)


def wire_type(typ: types.TypeAttribute) -> types.TypeAttribute:
    """Translate quantum and array types while retaining classical scalar types."""
    if is_subtype(typ, JeffWireType):
        return WireType
    if is_subtype(typ, QuregType):
        return register_type(qureg_length(typ))
    if is_subtype(typ, IntArrayType):
        return ilist.IListType[types.Int, types.Any]
    if is_subtype(typ, FloatArrayType):
        return ilist.IListType[types.Float, types.Any]
    if isinstance(typ, types.Generic) and is_subtype(typ, types.Tuple):
        return types.Generic(tuple, *(wire_type(member) for member in typ.vars))
    return typ


DIRECT: dict[type[ir.Statement], Callable[..., ir.Statement]] = {
    stmts.Alloc: wire.Alloc,
    stmts.Free: wire.Free,
    stmts.FreeZero: wire.Free,
    stmts.RegAlloc: wire.AllocRegister,
    stmts.RegFree: wire.DiscardRegister,
    stmts.RegFreeZero: wire.DiscardRegister,
    stmts.Extract: wire.Take,
    stmts.Insert: wire.Put,
    stmts.ExtractSlice: wire.TakeSlice,
    stmts.InsertSlice: wire.PutSlice,
    stmts.RegSplit: wire.SplitRegister,
    stmts.RegJoin: wire.JoinRegister,
    stmts.RegLength: wire.Length,
    stmts.Reset: wire.Reset,
}
"""Map each Jeff statement to the wire statement with the same operands and results."""


@dataclass
class Quantum(RewriteRule):
    """Preserve explicit quantum ownership while replacing operations locally."""

    methods: Mapping[ir.Method, ir.Method]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Replace a quantum operation, call or function return."""
        if build := DIRECT.get(type(node)):
            made = put(build(*node.args), node)
            replace(node, *made.results)
        elif isinstance(node, stmts.RegCreate):
            replace(node, put(wire.PackRegister(node.wires), node).result)
        elif isinstance(node, (stmts.Measure, stmts.MeasureNd)):
            self.measure(node)
        elif isinstance(node, stmts.Gate):
            return self.gate(node)
        elif isinstance(node, stmts.Call):
            self.call(node)
        elif isinstance(node, stmts.Return):
            self.returned(node)
        else:
            return RewriteResult()
        return RewriteResult(has_done_something=True)

    def measure(self, node: stmts.Measure | stmts.MeasureNd) -> None:
        """Replace a Jeff measurement with a wire measurement.

        `MeasureNd` keeps its qubit. `Measure` frees it after the measurement.
        """
        measured = put(wire.Measure(node.wire), node)
        if isinstance(node, stmts.MeasureNd):
            replace(node, *measured.results)
            return
        put(wire.Free(measured.results[0]), node)
        replace(node, measured.results[1])

    def call(self, node: stmts.Call) -> None:
        """Replace a Jeff call with a call of the wire copy of its callee.

        A wire method returns several outputs as one tuple, so each result reads its
        position of that tuple.
        """
        callee = self.methods[node.callee]
        made = put(func.Invoke(node.inputs, callee=callee), node)
        outputs = tuple(wire_type(value.type) for value in node.results)
        made.result.type = packed_type(outputs)
        if len(outputs) == 1:
            replace(node, made.result)
            return
        values = []
        for i, typ in enumerate(outputs):
            index = put(py.Constant(i), node).result
            item = put(py.GetItem(made.result, index), node)
            item.result.type = typ
            values.append(item.result)
        replace(node, *values)

    def returned(self, node: stmts.Return) -> None:
        """Replace a Jeff return with a return of None, one value or a tuple."""
        node.replace_by(func.Return(pack(node.values, node)))

    def gate(self, node: stmts.Gate) -> RewriteResult:
        """Apply each gate power and map output wires to their declared operand positions."""
        shape = canonical(node)
        if not gate_supported(shape):
            raise ValueError(f"Gate '{shape.name}' has no squin wire form.")
        current = {value: value for value in (*node.targets, *node.controls)}

        def apply(source, operands, angles=(), adjoint=False):
            wires = tuple(current[value] for value in operands)
            made = put(ENTRIES[source].scalar(wires, angles, adjoint), node)
            current.update(zip(operands, made.outputs, strict=True))

        def turns(value):
            scale = put(py.Constant(1 / (2 * math.pi)), node).result
            return put(py.Mult(value, scale), node).result

        def negative(value):
            return put(py.USub(value), node).result

        for _ in range(shape.power):
            name = shape.name
            operands = (*shape.controls, *shape.targets)
            if len(shape.controls) == 1:
                apply(CONTROLLED[name], operands)
            elif len(shape.controls) == 2:
                apply(gates.CCZ, operands)
            elif name == "u":
                theta, phi, lam = (turns(value) for value in shape.params)
                angles = (
                    (negative(theta), negative(lam), negative(phi))
                    if shape.adjoint
                    else (theta, phi, lam)
                )
                apply(gates.U3, operands, angles)
            elif name == "swap":
                apply(gates.Swap, operands)
            elif name in SINGLE:
                apply(SINGLE[name], operands)
            elif name in NON_HERMITIAN:
                apply(NON_HERMITIAN[name], operands, adjoint=shape.adjoint)
            elif name in ROTATION:
                angle = turns(shape.params[0])
                apply(
                    ROTATION[name],
                    operands,
                    (negative(angle) if shape.adjoint else angle,),
                )
        replace(node, *(current[value] for value in (*node.targets, *node.controls)))
        return RewriteResult(has_done_something=True)
