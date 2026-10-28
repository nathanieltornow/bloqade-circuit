"""Lower each squin statement on qubits onto `Unwrap`, its twin and `Wrap`.

Each rule rewrites one statement and keeps reference semantics. The gate
`squin.gate.cx(controls=[a], targets=[b])` becomes:

    %wa = Unwrap(a)
    %wb = Unwrap(b)
    %va, %vb = wire.cx(%wa, %wb)
    Wrap(a, %va)
    Wrap(b, %vb)

A gate on a list of several qubits becomes one such group per item. A call of a
kernel takes each qubit argument out, calls the wire function, and puts the
returned wires back. The promotion passes in `promote` later join each `Wrap`
with the next `Unwrap` of the same place.

The rules read the places of `QubitReferenceAnalysis` only to count the items of
a list. A list whose length the places don't give keeps its squin statement, and
the reference check then reports it.
"""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from kirin import ir, types
from kirin.rewrite import Walk, Chain
from kirin.dialects import py, func, ilist
from kirin.dialects.py import len as py_len
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.types import QubitType
from bloqade.analysis.reference import (
    Ref,
    Place,
    Untracked,
    positions,
    typed_length,
)
from bloqade.squin.analysis.reference import qubit_items

from . import stmts
from .group import kernel
from .twins import TWINS
from .types import WireType, RegisterType
from .traits import Twin


def _put(stmt: ir.Statement, before: ir.Statement) -> ir.Statement:
    """Insert `stmt` before the statement `before` and return `stmt`."""
    stmt.insert_before(before)
    return stmt


def output_types(output: types.TypeAttribute) -> tuple[types.TypeAttribute, ...]:
    """Return the type of each output of a function whose output type is `output`.

    `None` gives no output. A tuple gives one output per member, and any other type
    gives one output.
    """
    if output.is_subseteq(types.NoneType):
        return ()
    if isinstance(output, types.Generic) and output.is_subseteq(types.Tuple):
        return tuple(output.vars)
    return (output,)


def packed_type(outputs: Sequence[types.TypeAttribute]) -> types.TypeAttribute:
    """Return the output type of a function with the output types `outputs`."""
    if not outputs:
        return types.NoneType
    if len(outputs) == 1:
        return outputs[0]
    return types.Generic(tuple, *outputs)


def linear_type(ref: Ref | None) -> types.TypeAttribute | None:
    """Return the wire squin type of a value with the place `ref`, or None.

    One qubit becomes a `Wire`, and a whole register becomes a `Register`. A value
    without qubits gives None.
    """
    if ref is None or isinstance(ref, Untracked):
        return None
    if isinstance(ref, Place) and ref.one_item:
        return WireType
    if isinstance(ref, Place):
        return RegisterType
    return None


def take(value: ir.SSAValue, ref: Ref, before: ir.Statement) -> ir.SSAValue:
    """Insert the statement that takes the qubits of `value` out, and return them."""
    if isinstance(ref, Place) and ref.one_item:
        return _put(stmts.Unwrap(value), before).wire
    return _put(stmts.UnwrapRegister(value), before).register


def give(value: ir.SSAValue, wire: ir.SSAValue, before: ir.Statement) -> None:
    """Insert the statement that puts `wire` back into the reference `value`."""
    if wire.type.is_subseteq(RegisterType):
        _put(stmts.WrapRegister(value, wire), before)
    else:
        _put(stmts.Wrap(value, wire), before)


@dataclass(frozen=True)
class Layout:
    """Describe the outputs of the wire function of one kernel.

    `positions` counts the ordinary outputs, and `packed` is True if the kernel
    returns a tuple. `borrowed` holds the qubit parameters. Their wires follow the
    ordinary outputs in parameter order, as in Guppy and in MQT's QCO dialect.
    `outputs` holds the wire squin type of every output.
    """

    positions: int
    packed: bool
    borrowed: tuple[ir.BlockArgument, ...]
    outputs: tuple[types.TypeAttribute, ...]

    @property
    def output(self) -> types.TypeAttribute:
        """Return the output type of the wire function."""
        return packed_type(self.outputs)


def layout_of(code: ir.Statement, refs: Mapping[ir.SSAValue, Ref]) -> Layout:
    """Return the layout of the wire function of the kernel code `code`."""
    assert isinstance(code, func.Function)
    entry = code.body.blocks[0]
    borrowed = tuple(p for p in entry.args[1:] if linear_type(refs.get(p)) is not None)
    originals = output_types(code.signature.output)
    returned = entry.last_stmt
    assert isinstance(returned, func.Return)
    places = positions(refs.get(returned.value, Untracked()), len(originals))
    outputs = [linear_type(r) or t for r, t in zip(places, originals, strict=True)]
    outputs += [linear_type(refs.get(p)) or p.type for p in borrowed]
    packed = code.signature.output.is_subseteq(types.Tuple) and bool(originals)
    return Layout(len(originals), packed, borrowed, tuple(outputs))


@dataclass
class LowerTwins(RewriteRule):
    """Lower a squin statement that has a twin onto `Unwrap`, twin and `Wrap`.

    `refs` holds the places of the kernel before the lowering.
    """

    refs: Mapping[ir.SSAValue, Ref]

    def items(
        self, value: ir.SSAValue, before: ir.Statement
    ) -> list[ir.SSAValue] | None:
        """Return one reference for each qubit of the list `value`, or None.

        A literal list gives its values. Any other list of known length gives the
        reads `value[k]`. The places give the length, or the type gives it for a
        reference that the lowering created, such as the register that a call
        returns.
        """
        if isinstance(made := value.owner, ilist.New):
            return list(made.values)
        ref = self.refs.get(value)
        counted = None if ref is None else qubit_items(ref)
        size = typed_length(value.type, 0) if counted is None else len(counted)
        if size is None:
            return None
        items: list[ir.SSAValue] = []
        for k in range(size):
            at = _put(py.Constant(k), before)
            item = _put(py.GetItem(value, at.result), before)
            item.result.type = QubitType
            items.append(item.result)
        return items

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Lower `node` if it is a squin statement with a twin."""
        twin = TWINS.get(type(node))
        if twin is None:
            return RewriteResult()
        trait = twin.get_present_trait(Twin)
        fields = getattr(twin, "__kirin_stmt_fields")
        wires = dict(trait.wires)
        lists = [self.items(getattr(node, source), node) for source in wires]
        if any(items is None for items in lists):
            return RewriteResult()
        classical = {
            name: getattr(node, name)
            for name in fields.std_args
            if name not in wires.values()
        }
        attributes = {name: getattr(node, name) for name in fields.attributes}
        scalars: dict[str, list[ir.SSAValue]] = {name: [] for name, _ in trait.results}
        for group in zip(*lists, strict=True):  # type: ignore[misc]
            taken = {
                wire: _put(stmts.Unwrap(qubit), node).wire
                for wire, qubit in zip(wires.values(), group, strict=True)
            }
            made = _put(twin(**classical, **taken, **attributes), node)
            for wire, qubit in zip(wires.values(), group, strict=True):
                _put(stmts.Wrap(qubit, getattr(made, f"{wire}_out")), node)
            for source, result in trait.results:
                scalars[source].append(getattr(made, result))
        for name in getattr(type(node), "__kirin_stmt_fields").results:
            old = getattr(node, name)
            element = old.type.vars[0] if isinstance(old.type, types.Generic) else None
            listed = _put(
                ilist.New(values=tuple(scalars[name]), elem_type=element), node
            )
            old.replace_by(listed.result)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class LowerCalls(RewriteRule):
    """Lower a call of a kernel onto a call of its wire function.

    `kernels` maps each kernel to its prepared copy, which becomes the wire
    function. `refs` holds the places of every prepared copy.
    """

    refs: Mapping[ir.SSAValue, Ref]
    kernels: Mapping[ir.Method, ir.Method]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Lower `node` if it calls a prepared kernel."""
        if not isinstance(node, func.Invoke) or node.callee not in self.kernels:
            return RewriteResult()
        callee = self.kernels[node.callee]
        layout = layout_of(callee.code, self.refs)
        passed = [self.refs.get(arg, Untracked()) for arg in node.inputs]
        inputs = [
            arg if linear_type(ref) is None else take(arg, ref, node)
            for arg, ref in zip(node.inputs, passed, strict=True)
        ]
        call = _put(func.Invoke(tuple(inputs), callee=callee), node)
        call.result.type = layout.output
        outs = self.unpack(call.result, layout.outputs, node)
        given = [arg for arg, ref in zip(node.inputs, passed) if linear_type(ref)]
        for arg, wire in zip(given, outs[layout.positions :], strict=True):
            give(arg, wire, node)
        places = positions(self.refs.get(node.result, Untracked()), layout.positions)
        values = [
            self.reference(out, place, node)
            for out, place in zip(outs[: layout.positions], places, strict=True)
        ]
        if not layout.positions:
            node.result.replace_by(_put(func.ConstantNone(), node).result)
        elif not layout.packed:
            node.result.replace_by(values[0])
        else:
            node.result.replace_by(_put(py.tuple.New(tuple(values)), node).result)
        node.delete()
        return RewriteResult(has_done_something=True)

    @staticmethod
    def unpack(
        value: ir.SSAValue, kinds: Sequence[types.TypeAttribute], before: ir.Statement
    ) -> list[ir.SSAValue]:
        """Return one value for each output of the call result `value`."""
        if len(kinds) <= 1:
            return [value][: len(kinds)]
        items: list[ir.SSAValue] = []
        for k, kind in enumerate(kinds):
            at = _put(py.Constant(k), before)
            item = _put(py.GetItem(value, at.result), before)
            item.result.type = kind
            items.append(item.result)
        return items

    @staticmethod
    def reference(value: ir.SSAValue, place: Ref, before: ir.Statement) -> ir.SSAValue:
        """Return a reference for a returned wire or register, or `value` itself.

        A returned qubit gets a new `Reference` that the wire fills, so the rest of
        the caller can keep using it as a reference. A returned register gets a
        `ReferenceRegister` whose type holds the length that `place` gives.
        """
        if value.type.is_subseteq(WireType):
            fresh = _put(stmts.Reference(), before)
            _put(stmts.Wrap(fresh.result, value), before)
            return fresh.result
        if value.type.is_subseteq(RegisterType):
            fresh = _put(stmts.ReferenceRegister(), before)
            counted = qubit_items(place)
            size = types.Any if counted is None else types.Literal(len(counted))
            fresh.result.type = ilist.IListType[QubitType, size]
            _put(stmts.WrapRegister(fresh.result, value), before)
            return fresh.result
        return value


@dataclass
class LowerReturn(RewriteRule):
    """Lower the return of a kernel onto the outputs of its wire function.

    The kernel returns its ordinary outputs, then the wire of each qubit parameter.
    A returned qubit leaves as a wire. Run the rule once with `Walk`, because the
    new return already returns wires.
    """

    refs: Mapping[ir.SSAValue, Ref]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Lower `node` if it is the return of a function that holds qubits."""
        if not isinstance(node, func.Return):
            return RewriteResult()
        code = node.parent_stmt
        if not isinstance(code, func.Function):
            return RewriteResult()
        layout = layout_of(code, self.refs)
        if layout.outputs == output_types(code.signature.output):
            return RewriteResult()
        if layout.positions == 0:
            values: list[ir.SSAValue] = []
        elif not layout.packed:
            values = [node.value]
        elif isinstance(made := node.value.owner, py.tuple.New):
            values = list(made.args)
        else:
            return RewriteResult()
        outs = [
            (
                value
                if linear_type(ref := self.refs.get(value)) is None
                else take(value, ref, node)
            )  # type: ignore[arg-type]
            for value in values
        ]
        outs += [take(p, self.refs[p], node) for p in layout.borrowed]
        if not outs:
            packed = _put(func.ConstantNone(), node).result
        elif len(outs) == 1:
            packed = outs[0]
        else:
            packed = _put(py.tuple.New(tuple(outs)), node).result
        _put(func.Return(packed), node)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class LowerLength(RewriteRule):
    """Lower `len` of a list of qubits.

    A list of known length gives a constant. A whole register of unknown length
    gives `Length` on the register.
    """

    refs: Mapping[ir.SSAValue, Ref]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Lower `node` if it reads the length of a list of qubits."""
        if not isinstance(node, py_len.Len):
            return RewriteResult()
        ref = self.refs.get(node.value)
        if ref is None or isinstance(ref, Untracked):
            return RewriteResult()
        if (items := qubit_items(ref)) is not None:
            node.result.replace_by(_put(py.Constant(len(items)), node).result)
        elif isinstance(ref, Place) and not ref.path:
            register = _put(stmts.UnwrapRegister(node.value), node).register
            length = _put(stmts.Length(register), node)
            _put(stmts.WrapRegister(node.value, length.register_out), node)
            node.result.replace_by(length.result)
        else:
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)


def lower(
    method: ir.Method,
    refs: Mapping[ir.SSAValue, Ref],
    kernels: Mapping[ir.Method, ir.Method],
) -> RewriteResult:
    """Lower every qubit statement of `method` in place, with the places `refs`.

    The method moves `method` to the dialect group of wire squin first, because
    the lowered statements belong to it.
    """
    method.dialects = kernel
    rules = Chain(LowerTwins(refs), LowerCalls(refs, kernels), LowerLength(refs))
    result = Walk(rules).rewrite(method.code)
    return Walk(LowerReturn(refs)).rewrite(method.code).join(result)
