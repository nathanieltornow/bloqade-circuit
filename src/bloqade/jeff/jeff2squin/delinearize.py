"""This module holds the pass that turns jeff wires and registers into squin qubits."""

import math
from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.passes import Pass
from kirin.rewrite import Walk, Chain
from kirin.dialects import py, func, ilist, ssacfg
from kirin.dialects.py import len as py_len, slice as py_slice
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade import squin
from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import QubitType
from bloqade.jeff.gates import canonical
from bloqade.jeff.types import WireType, QuregType, is_linear, is_subtype, qureg_length
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.jeff.dialects import stmts, kernel as jeff_kernel
from bloqade.jeff.analysis.reference import Ref, Whole, WireReferenceAnalysis
from bloqade.jeff.analysis.validation.envelope import (
    SINGLE,
    ROTATION,
    CONTROLLED,
    NON_HERMITIAN,
    gate_supported,
)
from bloqade.jeff.analysis.validation.to_squin import handed_back

_DONE = RewriteResult(has_done_something=True)


@dataclass
class Delinearize(Pass):
    """A pass that rewrites the jeff quantum statements of a method for squin."""

    kernels: Mapping[ir.Method, ir.Method] = field(kw_only=True)
    """The squin kernel of each jeff function."""

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        """Rewrite `mt` in place and give it the output type `Any` for typeinfer.

        The pass runs the wire reference analysis on `mt` first. It reads from the
        analysis which wires an insert returns to their slot and which outputs a
        function hands back.
        """
        references = WireReferenceAnalysis(mt.dialects)
        frame, _ = references.run(mt)
        own = handed_back(mt, references.results[mt])
        returned = _Return(tuple(p for p in own if p is not None))
        calls = _Calls(self.kernels, references.results)
        registers = _Registers(frame.entries)
        rules = Chain(_Qubits(), _Gates(), registers, returned, calls, _Retype())
        result = Walk(rules).rewrite(mt.code)
        ir.HasSignature.set_signature(
            mt.code, func.Signature(inputs=tuple(mt.arg_types), output=types.Any)
        )
        return result


def _retype(value: ir.SSAValue) -> bool:
    """Retype a wire as a qubit and a register as a list of qubits.

    The function returns True if it changed the type of `value`.
    """
    if is_subtype(value.type, WireType):
        value.type = QubitType
        return True
    if is_subtype(value.type, QuregType):
        length = qureg_length(value.type)
        value.type = ilist.IListType[
            QubitType, types.Any if length is None else types.Literal(length)
        ]
        return True
    return False


def _wrap(ref: ir.SSAValue, before: ir.Statement) -> ir.SSAValue:
    """Return a one-element list of `ref`, which squin statements take."""
    (wrapped := ilist.New([ref])).insert_before(before)
    return wrapped.result


def _const(value: int | float, before: ir.Statement) -> ir.SSAValue:
    """Return a constant of `value`."""
    (constant := py.Constant(value)).insert_before(before)
    return constant.result


def _negate(value: ir.SSAValue, before: ir.Statement) -> ir.SSAValue:
    """Return the negated value."""
    (negated := py.unary.USub(value)).insert_before(before)
    return negated.result


def _replace(stmt: ir.Statement, *values: ir.SSAValue) -> RewriteResult:
    """Replace the results of `stmt` by `values` in order, then delete `stmt`."""
    for result, value in zip(stmt.results, values, strict=True):
        result.replace_by(value)
    stmt.delete()
    return _DONE


class _Qubits(RewriteRule):
    """A rewrite rule that rewrites qubit allocation, freeing, reset and measurement."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it allocates, frees, resets or measures a qubit."""
        match node:
            case stmts.Alloc():
                (qubit := qubit_stmts.New()).insert_before(node)
                return _replace(node, qubit.result)
            case (
                stmts.Free() | stmts.FreeZero() | stmts.RegFree() | stmts.RegFreeZero()
            ):
                node.delete()
                return _DONE
            case stmts.Reset():
                qubit_stmts.Reset(_wrap(node.wire, node)).insert_before(node)
                return _replace(node, node.wire)
            case stmts.Measure() | stmts.MeasureNd():
                (measured := qubit_stmts.Measure(_wrap(node.wire, node))).insert_before(
                    node
                )
                index = _const(0, node)
                (bit := py.indexing.GetItem(measured.result, index)).insert_before(node)
                if isinstance(node, stmts.MeasureNd):
                    return _replace(node, node.wire, bit.result)
                return _replace(node, bit.result)
        return RewriteResult()


class _Gates(RewriteRule):
    """A rewrite rule that applies a jeff gate as squin gate statements."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Apply `node` once per unit of its power and map each output to its input."""
        if not isinstance(node, stmts.Gate):
            return RewriteResult()
        shape = canonical(node)
        if not gate_supported(shape):
            return RewriteResult()
        name, adjoint = shape.name, shape.adjoint
        for _ in range(shape.power):
            targets = [_wrap(w, node) for w in shape.targets]
            controls = [_wrap(w, node) for w in shape.controls]
            if len(controls) == 1:
                CONTROLLED[name](controls[0], targets[0]).insert_before(node)
            elif len(controls) == 2:
                gate_stmts.CCZ(controls[0], controls[1], targets[0]).insert_before(node)
            elif name in SINGLE:
                SINGLE[name](targets[0]).insert_before(node)
            elif name in NON_HERMITIAN:
                NON_HERMITIAN[name](targets[0], adjoint=adjoint).insert_before(node)
            elif name in ROTATION:
                angle = _turns(shape.params[0], node)
                if adjoint:
                    angle = _negate(angle, node)
                ROTATION[name](angle, targets[0]).insert_before(node)
            elif name == "u":
                theta, phi, lam = (_turns(p, node) for p in shape.params)
                if adjoint:
                    theta, phi, lam = (_negate(a, node) for a in (theta, lam, phi))
                gate_stmts.U3(theta, phi, lam, targets[0]).insert_before(node)
            elif name == "swap":
                gate_stmts.Swap(targets[0], targets[1]).insert_before(node)
        return _replace(node, *node.targets, *node.controls)


def _turns(radians: ir.SSAValue, before: ir.Statement) -> ir.SSAValue:
    """Convert an angle from radians to squin's turns."""
    (turns := py.binop.Div(radians, _const(2 * math.pi, before))).insert_before(before)
    return turns.result


@dataclass
class _Registers(RewriteRule):
    """A rewrite rule that rewrites register statements as list operations.

    An insert that returns a wire to the slot that it came from keeps the list.
    """

    references: Mapping[ir.SSAValue, Ref]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a register statement."""
        match node:
            case stmts.RegAlloc():
                (
                    invoke := func.Invoke((node.size,), callee=squin.qalloc)
                ).insert_before(node)
                return _replace(node, invoke.result)
            case stmts.RegCreate():
                (created := ilist.New(list(node.wires))).insert_before(node)
                return _replace(node, created.result)
            case stmts.Extract():
                (get := py.indexing.GetItem(node.reg, node.index)).insert_before(node)
                return _replace(node, node.reg, get.result)
            case stmts.Insert():
                if isinstance(self.references.get(node.result), Whole):
                    return _replace(node, node.reg)
                one = _const(1, node)
                (stop := py.binop.Add(node.index, one)).insert_before(node)
                element = _wrap(node.wire, node)
                spliced = _splice(node.reg, node.index, stop.result, element, node)
                return _replace(node, spliced)
            case stmts.ExtractSlice():
                (stop := py.binop.Add(node.start, node.length)).insert_before(node)
                one = _const(1, node)
                (sliced := py_slice.Slice(node.start, stop.result, one)).insert_before(
                    node
                )
                (part := py.indexing.GetItem(node.reg, sliced.result)).insert_before(
                    node
                )
                return _replace(node, node.reg, part.result)
            case stmts.InsertSlice():
                (length := py_len.Len(node.slice_reg)).insert_before(node)
                (stop := py.binop.Add(node.start, length.result)).insert_before(node)
                spliced = _splice(
                    node.reg, node.start, stop.result, node.slice_reg, node
                )
                return _replace(node, spliced)
            case stmts.RegSplit():
                (length := py_len.Len(node.reg)).insert_before(node)
                zero, one = _const(0, node), _const(1, node)
                (head := py_slice.Slice(zero, node.index, one)).insert_before(node)
                (front := py.indexing.GetItem(node.reg, head.result)).insert_before(
                    node
                )
                one = _const(1, node)
                tail = py_slice.Slice(node.index, length.result, one)
                tail.insert_before(node)
                (back := py.indexing.GetItem(node.reg, tail.result)).insert_before(node)
                return _replace(node, front.result, back.result)
            case stmts.RegJoin():
                (joined := py.binop.Add(node.first, node.second)).insert_before(node)
                return _replace(node, joined.result)
            case stmts.RegLength():
                (length := py_len.Len(node.reg)).insert_before(node)
                return _replace(node, node.reg, length.result)
        return RewriteResult()


def _splice(
    reg: ir.SSAValue,
    start: ir.SSAValue,
    stop: ir.SSAValue,
    middle: ir.SSAValue,
    before: ir.Statement,
) -> ir.SSAValue:
    """Return `reg` with `middle` in place of the slots from `start` to `stop`."""
    zero, one = _const(0, before), _const(1, before)
    (length := py_len.Len(reg)).insert_before(before)
    (head_slice := py_slice.Slice(zero, start, one)).insert_before(before)
    (head := py.indexing.GetItem(reg, head_slice.result)).insert_before(before)
    (tail_slice := py_slice.Slice(stop, length.result, one)).insert_before(before)
    (tail := py.indexing.GetItem(reg, tail_slice.result)).insert_before(before)
    (front := py.binop.Add(head.result, middle)).insert_before(before)
    (whole := py.binop.Add(front.result, tail.result)).insert_before(before)
    return whole.result


@dataclass
class _Return(RewriteRule):
    """A rewrite rule that returns the outputs that the function does not hand back."""

    handed_back: tuple[int, ...]
    """The output positions that hand back a wire parameter."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a jeff return."""
        if not isinstance(node, stmts.Return):
            return RewriteResult()
        values = [v for i, v in enumerate(node.values) if i not in self.handed_back]
        if not values:
            (none := func.ConstantNone()).insert_before(node)
            value = none.result
        elif len(values) == 1:
            value = values[0]
        else:
            (packed := py.tuple.New(tuple(values))).insert_before(node)
            value = packed.result
        func.Return(value).insert_before(node)
        node.delete()
        return _DONE


@dataclass
class _Calls(RewriteRule):
    """A rewrite rule that invokes the squin kernel of each jeff callee."""

    kernels: Mapping[ir.Method, ir.Method]
    results: Mapping[ir.Method, Ref]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a jeff call."""
        if not isinstance(node, stmts.Call):
            return RewriteResult()
        kinds = node.callee.arg_types
        quantum = [v for v, kind in zip(node.inputs, kinds) if is_linear(kind)]
        result = self.results.get(node.callee)
        kernel = self.kernels.get(node.callee)
        if kernel is None or result is None:
            return RewriteResult()
        positions = handed_back(node.callee, result)
        if len(positions) != len(quantum):
            return RewriteResult()
        continued = {
            position: value
            for value, position in zip(quantum, positions)
            if position is not None
        }
        rest = len(node.results) - len(continued)
        if rest < 0 or any(position >= len(node.results) for position in continued):
            return RewriteResult()
        (invoke := func.Invoke(tuple(node.inputs), callee=kernel)).insert_before(node)
        reads: list[ir.SSAValue] = [invoke.result]
        if rest != 1:
            reads = []
            for i in range(rest):
                read = py.indexing.GetItem(invoke.result, _const(i, node))
                read.insert_before(node)
                reads.append(read.result)
        remaining = iter(reads)
        outputs = [
            continued[position] if position in continued else next(remaining)
            for position in range(len(node.results))
        ]
        return _replace(node, *outputs)


class _Retype(RewriteRule):
    """A rewrite rule that retypes the wires and registers that other rules left."""

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        """Retype the wire and register arguments of a block."""
        changed = [_retype(arg) for arg in node.args]
        return RewriteResult(has_done_something=any(changed))

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Retype the results of `node` if an earlier rule left it in its block."""
        jeff = node.dialect in jeff_kernel and node.dialect not in (
            func.dialect,
            ssacfg.dialect,
        )
        if node.parent is None or jeff:
            return RewriteResult()
        changed = [_retype(result) for result in node.results]
        return RewriteResult(has_done_something=any(changed))
