"""Turn wire squin back into reference squin with the kirin rewrite rules of `FromWire`.

Every wire statement hands each wire on to a result. So a qubit keeps one
reference squin value along its whole chain of wires. The rule for each statement
replaces the results of the statement with its inputs. Once the rule for `Alloc`
replaces the first wire of a chain with a `qubit.new` result, every later
statement on the chain reads that qubit. The order of the rewrites therefore
makes no difference.

The rules for loops and branches run after the other rules. They remove the
carried wires from the initializers, the block arguments and the yields.

A borrowing call returns the wires of its qubit arguments after its ordinary
outputs. The call rule reads each returned wire as the argument that went in. The
signature rule turns each wire parameter back into a qubit parameter and drops the
returned wires from the return statement.
"""

from typing import Any, TypeVar
from dataclasses import field, dataclass

from kirin import ir, types
from kirin.rewrite import Walk, Chain, DeadCodeElimination
from kirin.dialects import py, scf, func, ilist
from kirin.dialects.py import len as py_len, slice as py_slice
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.types import QubitType
from bloqade.squin.wire import stmts
from bloqade.squin.wire.group import kernel as wire_kernel, reference_kernel
from bloqade.squin.wire.types import RegisterType, is_linear
from bloqade.squin.wire.traits import Twin
from bloqade.squin.wire.ownership import KEY as OWNED, owns_inputs
from bloqade.squin.wire.signature import packed_type, output_types
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint


def squin_type(kind: types.TypeAttribute) -> types.TypeAttribute:
    """Return the reference squin type of the wire squin type `kind`."""
    if kind.is_subseteq(RegisterType):
        return ilist.IListType[QubitType, types.Any]
    if is_linear(kind):
        return QubitType
    if isinstance(kind, types.Generic) and kind.is_subseteq(types.Tuple):
        return types.Generic(tuple, *(squin_type(member) for member in kind.vars))
    return kind


def _kept_outputs(
    signature: func.Signature, owned: bool = False
) -> tuple[types.TypeAttribute, ...]:
    """Return the ordinary output types of a wire function with `signature`.

    The wire function returns one wire for each linear parameter after them.
    """
    borrowed = 0 if owned else sum(is_linear(kind) for kind in signature.inputs)
    outputs = output_types(signature.output)
    return tuple(squin_type(kind) for kind in outputs[: len(outputs) - borrowed])


_S = TypeVar("_S", bound=ir.Statement)


def _put(stmt: _S, before: ir.Statement) -> _S:
    """Insert `stmt` before the statement `before` and return `stmt`."""
    stmt.insert_before(before)
    return stmt


class FromWireStatements(RewriteRule):
    """Replace each wire statement outside a loop or branch with reference squin."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it belongs to the wire dialect."""
        if (twin := node.get_trait(Twin)) is not None:
            self.twin(node, twin)
        elif isinstance(node, stmts.Alloc):
            node.result.replace_by(_put(qubit_stmts.New(), node).result)
        elif isinstance(node, stmts.AllocRegister):
            call = _put(func.Invoke((node.size,), callee=qalloc), node)
            call.result.type = qalloc.return_type
            node.result.replace_by(call.result)
        elif isinstance(node, (stmts.Borrow, stmts.Take)):
            item = _put(py.GetItem(node.register, node.index), node)
            item.result.type = QubitType
            node.wire.replace_by(item.result)
            node.register_out.replace_by(node.register)
        elif isinstance(node, stmts.Put):
            one = _put(py.Constant(1), node).result
            stop = _put(py.Add(node.index, one), node).result
            middle = _put(ilist.New((node.wire,), elem_type=QubitType), node).result
            node.register_out.replace_by(
                _splice(node.register, node.index, stop, middle, node)
            )
        elif isinstance(node, stmts.PackRegister):
            node.result.replace_by(
                _put(ilist.New(node.wires, elem_type=QubitType), node).result
            )
        elif isinstance(node, stmts.JoinRegister):
            node.result.replace_by(_put(py.Add(node.first, node.second), node).result)
        elif isinstance(node, stmts.SplitRegister):
            zero = _put(py.Constant(0), node).result
            one = _put(py.Constant(1), node).result
            length = _put(py_len.Len(node.register), node).result
            head = _put(py_slice.Slice(zero, node.index, one), node).result
            tail = _put(py_slice.Slice(node.index, length, one), node).result
            node.before.replace_by(_put(py.GetItem(node.register, head), node).result)
            node.after.replace_by(_put(py.GetItem(node.register, tail), node).result)
        elif isinstance(node, stmts.TakeSlice):
            stop = _put(py.Add(node.start, node.length), node).result
            one = _put(py.Constant(1), node).result
            part = _put(py_slice.Slice(node.start, stop, one), node).result
            node.slice_register.replace_by(
                _put(py.GetItem(node.register, part), node).result
            )
            node.register_out.replace_by(node.register)
        elif isinstance(node, stmts.PutSlice):
            length = _put(py_len.Len(node.slice_register), node).result
            stop = _put(py.Add(node.start, length), node).result
            node.register_out.replace_by(
                _splice(node.register, node.start, stop, node.slice_register, node)
            )
        elif isinstance(node, stmts.Release):
            node.register_out.replace_by(node.register)
        elif isinstance(node, stmts.Length):
            node.result.replace_by(_put(py_len.Len(node.register), node).result)
            node.register_out.replace_by(node.register)
        elif not isinstance(
            node, (stmts.Free, stmts.FreeRegister, stmts.DiscardRegister)
        ):
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)

    def twin(self, node: ir.Statement, twin: Twin) -> None:
        """Insert the squin statement of the twin `node` before it.

        Each wire becomes a list of one qubit. A scalar result, such as a
        measurement result, becomes item 0 of the list result of the squin
        statement.
        """
        fields = getattr(twin.source, "__kirin_stmt_fields")
        wires = dict(twin.wires)
        kwargs: dict[str, Any] = {}
        for name in fields.std_args:
            if name in wires:
                qubit = getattr(node, wires[name])
                listed = _put(ilist.New(values=(qubit,), elem_type=QubitType), node)
                kwargs[name] = listed.result
            else:
                kwargs[name] = getattr(node, name)
        for name in fields.attributes:
            kwargs[name] = getattr(node, name)
        made = _put(twin.source(**kwargs), node)
        for source, result in twin.results:
            zero = _put(py.Constant(0), node)
            item = _put(py.GetItem(getattr(made, source), zero.result), node)
            item.result.type = getattr(node, result).type
            getattr(node, result).replace_by(item.result)
        for wire in wires.values():
            getattr(node, f"{wire}_out").replace_by(getattr(node, wire))


def _without(values: tuple[ir.SSAValue, ...], dropped: list[int]) -> list[ir.SSAValue]:
    """Return `values` without the positions in `dropped`."""
    return [v for k, v in enumerate(values) if k not in dropped]


def _strip_yield(block: ir.Block, dropped: list[int]) -> None:
    """Replace the yield of `block` with a yield that lacks the positions `dropped`."""
    old = block.last_stmt
    assert isinstance(old, scf.Yield)
    scf.Yield(*_without(old.values, dropped)).insert_before(old)
    old.delete()


class FromWireRegions(RewriteRule):
    """Remove the carried wires of each loop and branch.

    The rule waits until `FromWireStatements` has rewritten the bodies, so that
    every carried value is a qubit of the enclosing block.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a loop or a branch with carried wires."""
        if isinstance(node, scf.For):
            return self.loop(node)
        if isinstance(node, scf.IfElse):
            return self.branch(node)
        return RewriteResult()

    def loop(self, node: scf.For) -> RewriteResult:
        """Read each carried qubit from its initializer and drop the carried wire."""
        block = node.body.blocks[0]
        dropped = [k for k, arg in enumerate(block.args[1:]) if is_linear(arg.type)]
        if not dropped:
            return RewriteResult()
        for k in dropped:
            block.args[k + 1].replace_by(node.initializers[k])
            node.results[k].replace_by(node.initializers[k])
        _strip_yield(block, dropped)
        for k in reversed(dropped):
            block.args[k + 1].delete()
        region = node.body
        region.detach()
        kept = _without(tuple(node.results), dropped)
        made = scf.For(node.iterable, region, *_without(node.initializers, dropped))
        made.insert_before(node)
        for old, new in zip(kept, made.results, strict=True):
            old.replace_by(new)
        node.delete()
        return RewriteResult(has_done_something=True)

    def branch(self, node: scf.IfElse) -> RewriteResult:
        """Replace each carried wire result with the qubit that the branch yields."""
        dropped = [k for k, r in enumerate(node.results) if is_linear(r.type)]
        if not dropped:
            return RewriteResult()
        then_yield = node.then_body.blocks[0].last_stmt
        assert isinstance(then_yield, scf.Yield)
        if any(is_linear(then_yield.values[k].type) for k in dropped):
            return RewriteResult()
        for k in dropped:
            node.results[k].replace_by(then_yield.values[k])
        for body in (node.then_body, node.else_body):
            _strip_yield(body.blocks[0], dropped)
        kept = _without(tuple(node.results), dropped)
        then_body, else_body = node.then_body, node.else_body
        then_body.detach()
        else_body.detach()
        made = scf.IfElse(node.cond, then_body, else_body)
        made.insert_before(node)
        for old, new in zip(kept, made.results, strict=True):
            old.replace_by(new)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class FromWireCalls(RewriteRule):
    """Replace each call of a wire function with a call of its squin kernel.

    `convert` returns the squin kernel of a wire method. A returned wire of a
    borrowed argument becomes the argument itself.
    """

    convert: "FromWire"

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it calls a wire squin method."""
        if not isinstance(node, func.Invoke) or node.callee.dialects is not wire_kernel:
            return RewriteResult()
        assert isinstance(node.callee.code, func.Function)
        signature = node.callee.code.signature
        owned = owns_inputs(node.callee)
        squin_kernel = self.convert.kernel_of(node.callee)
        call = _put(func.Invoke(node.inputs, callee=squin_kernel), node)
        kept = _kept_outputs(signature, owned)
        call.result.type = packed_type(kept)
        borrowed = (
            []
            if owned
            else [a for a, k in zip(node.inputs, signature.inputs) if is_linear(k)]
        )
        total = len(kept) + len(borrowed)

        def output(k: int, before: ir.Statement) -> ir.SSAValue:
            if k >= len(kept):
                return borrowed[k - len(kept)]
            if len(kept) == 1:
                return call.result
            at = _put(py.Constant(k), before)
            item = _put(py.GetItem(call.result, at.result), before)
            item.result.type = kept[k]
            return item.result

        if total == 1:
            node.result.replace_by(output(0, node))
        elif total > 1:
            for use in list(node.result.uses):
                item = use.stmt
                assert isinstance(item, py.GetItem)
                index = item.index.owner
                assert isinstance(index, py.Constant)
                item.result.replace_by(output(index.value.unwrap(), item))
                item.delete()
        node.delete()
        return RewriteResult(has_done_something=True)


class FromWireSignature(RewriteRule):
    """Give a wire function the signature of its squin kernel.

    Each wire or register parameter becomes a qubit or a list of qubits. The return
    statement drops the wires that the function hands back.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a function with linear parameters or outputs."""
        if not isinstance(node, func.Function):
            return RewriteResult()
        signature = node.signature
        outputs = output_types(signature.output)
        if not any(is_linear(k) for k in (*signature.inputs, *outputs)):
            return RewriteResult()
        kept = _kept_outputs(signature, node.attributes.get(OWNED) == ir.PyAttr(True))
        block = node.body.blocks[0]
        returned = block.last_stmt
        assert isinstance(returned, func.Return)
        if len(outputs) == 1:
            values = [returned.value]
        elif isinstance(made := returned.value.owner, py.tuple.New):
            values = list(made.args)
        else:
            values = []
        values = values[: len(kept)]
        if not values:
            value = _put(func.ConstantNone(), returned).result
        elif len(values) == 1:
            value = values[0]
        else:
            value = _put(py.tuple.New(tuple(values)), returned).result
        _put(func.Return(value), returned)
        returned.delete()
        inputs = tuple(squin_type(k) for k in signature.inputs)
        for arg in block.args[1:]:
            arg.type = squin_type(arg.type)
        output = packed_type(kept)
        block.args[0].type = types.MethodType[list(inputs), output]
        node.signature = func.Signature(inputs=inputs, output=output)
        return RewriteResult(has_done_something=True)


@dataclass
class FromWire:
    """Convert wire squin methods into squin kernels, one copy per method."""

    kernels: dict[ir.Method, ir.Method] = field(default_factory=dict)
    """Map each wire method to its squin kernel."""
    pending: list[ir.Method] = field(default_factory=list)
    """Hold the wire methods whose squin kernels still need the rewrite."""

    def kernel_of(self, method: ir.Method) -> ir.Method:
        """Return the squin kernel of the wire method `method`, made on first request.

        The kernel starts as a copy, and `convert` rewrites it later. So a call in
        a recursive kernel can name the kernel before its rewrite.
        """
        if method not in self.kernels:
            self.kernels[method] = method.similar(reference_kernel)
            self.pending.append(method)
        return self.kernels[method]

    def convert(self, method: ir.Method) -> ir.Method:
        """Return the squin kernel of `method` and of every wire method that it calls."""
        result = self.kernel_of(method)
        while self.pending:
            self.rewrite(self.kernels[self.pending.pop()])
        return result

    def rewrite(self, method: ir.Method) -> RewriteResult:
        """Run the statement rules, then the region and signature rules, then DCE."""
        statements = Walk(Chain(FromWireStatements(), FromWireCalls(self)))
        result = CheckedFixpoint(statements).rewrite(method.code)
        if owns_inputs(method):
            result = Walk(FromWireSignature()).rewrite(method.code).join(result)
            result = Walk(ReferenceTypes()).rewrite(method.code).join(result)
            method.code.attributes.pop(OWNED)
            return result
        regions = Walk(Chain(FromWireRegions(), FromWireSignature()))
        result = CheckedFixpoint(regions).rewrite(method.code).join(result)
        return (
            CheckedFixpoint(Walk(DeadCodeElimination()))
            .rewrite(method.code)
            .join(result)
        )


def _splice(
    reg: ir.SSAValue,
    start: ir.SSAValue,
    stop: ir.SSAValue,
    middle: ir.SSAValue,
    before: ir.Statement,
) -> ir.SSAValue:
    """Return `reg` with `middle` in place of the slots from `start` to `stop`."""
    zero, one = _put(py.Constant(0), before).result, _put(py.Constant(1), before).result
    (length := py_len.Len(reg)).insert_before(before)
    (head_slice := py_slice.Slice(zero, start, one)).insert_before(before)
    (head := py.indexing.GetItem(reg, head_slice.result)).insert_before(before)
    (tail_slice := py_slice.Slice(stop, length.result, one)).insert_before(before)
    (tail := py.indexing.GetItem(reg, tail_slice.result)).insert_before(before)
    (front := py.binop.Add(head.result, middle)).insert_before(before)
    (whole := py.binop.Add(front.result, tail.result)).insert_before(before)
    return whole.result


class ReferenceTypes(RewriteRule):
    """Retype explicit owned state after quantum operations become reference operations."""

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        """Convert each region parameter to its corresponding reference type."""
        for value in node.args:
            value.type = squin_type(value.type)
        return RewriteResult()

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Convert each remaining result type before reference type inference."""
        for value in node.results:
            value.type = squin_type(value.type)
        return RewriteResult()


def bounded_list_lengths(kind: types.TypeAttribute) -> types.TypeAttribute:
    """Constrain unknown list lengths to integers without assuming a concrete size."""
    if isinstance(kind, types.Generic) and kind.is_subseteq(ilist.IListType):
        element, length = kind.vars
        return ilist.IListType[
            bounded_list_lengths(element), types.Int if length == types.Any else length
        ]
    if (
        isinstance(kind, types.Generic)
        and kind.is_subseteq(types.Tuple)
        and kind.vararg is None
    ):
        return types.Generic(
            tuple, *(bounded_list_lengths(member) for member in kind.vars)
        )
    return kind


class BoundListLengths(RewriteRule):
    """Give list lengths integer bounds before checking operation types."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Refine result types and the declared function signature."""
        changed = False
        for value in node.results:
            kind = bounded_list_lengths(value.type)
            changed |= value.type != kind
            value.type = kind
        if isinstance(node, func.Function):
            signature = node.signature
            inputs = tuple(bounded_list_lengths(kind) for kind in signature.inputs)
            output = bounded_list_lengths(signature.output)
            changed |= inputs != signature.inputs or output != signature.output
            node.signature = func.Signature(inputs=inputs, output=output)
            node.body.blocks[0].args[0].type = types.MethodType[list(inputs), output]
        return RewriteResult(has_done_something=changed)

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        """Refine list lengths in block arguments."""
        changed = False
        for value in node.args:
            kind = bounded_list_lengths(value.type)
            changed |= value.type != kind
            value.type = kind
        return RewriteResult(has_done_something=changed)
