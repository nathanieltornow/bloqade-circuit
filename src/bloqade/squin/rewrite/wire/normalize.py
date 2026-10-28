"""Rewrite reference squin into forms whose qubits `QubitReferenceAnalysis` can name.

Each rule is a kirin rewrite rule, and each keeps the meaning of the kernel.

- `QubitIteration` turns `for q in qs` into a loop over `range(len(qs))` that
  reads `qs[i]`.
- `MapToLoop` turns `ilist.map` and `ilist.for_each` over qubits into such a loop.
  The loop calls the function on each qubit, and a map appends each result to
  a list that the loop carries.
- `IndexInBranch` and `IndexInLoop` handle a qubit that a branch or a loop picks
  from one register. The branch or the loop carries the index, and a read of the
  register after it gives the qubit.
- `Concatenate` turns `a + b` of registers of known length into a literal list
  of their qubits.
- `BroadcastToLoop` turns a broadcast of unknown length into a loop of
  singleton broadcasts.

Guppy asks the user to write these forms by hand. The rules write them for the
user, so a squin kernel can keep its aliases.
"""

from typing import TypeVar
from dataclasses import dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.dialects import py, scf, func, ilist
from kirin.dialects.py import cmp, len as py_len, assertion
from kirin.rewrite.abc import RewriteRule, RewriteResult
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.types import QubitType
from bloqade.constants import constant_int
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.traits import Twin
from bloqade.analysis.reference import Ref
from bloqade.squin.analysis.reference import qubit_items, holds_qubits
from bloqade.analysis.reference.analysis import typed_length

_S = TypeVar("_S", bound=ir.Statement)


def _put(stmt: _S, before: ir.Statement) -> _S:
    """Insert `stmt` before the statement `before` and return `stmt`."""
    stmt.insert_before(before)
    return stmt


def _read_at(value: ir.SSAValue, register: ir.SSAValue, before: ir.Statement) -> None:
    """Replace each use of the index `value` outside the new read with `register[value]`.

    The method inserts the read before `before`. The read becomes the new owner of
    every former use of `value`.
    """
    item = _put(py.GetItem(register, value), before)
    item.result.type = QubitType
    value.replace_by(item.result)
    item.index = value


def _invariant(value: ir.SSAValue) -> ir.SSAValue:
    """Return the value outside every loop that `value` passes through unchanged.

    Kirin's lowering carries each variable of a loop body as a block argument,
    even if the body never assigns it. Such an argument holds its initializer.
    """
    while isinstance(value, ir.BlockArgument) and value.index > 0:
        loop = value.owner.parent_stmt
        if not isinstance(loop, scf.For):
            break
        returned = loop.body.blocks[0].last_stmt
        if not isinstance(returned, scf.Yield):
            break
        if returned.values[value.index - 1] is not value:
            break
        value = loop.initializers[value.index - 1]
    return value


def _picked(value: ir.SSAValue) -> tuple[ir.SSAValue, ir.SSAValue] | None:
    """Return the register and the index of the read `register[index]` behind `value`."""
    owner = value.owner
    if not isinstance(owner, py.GetItem) or not value.type.is_subseteq(QubitType):
        return None
    if not owner.index.type.is_subseteq(types.Int):
        return None
    return _invariant(owner.obj), owner.index


def _range_over(collection: ir.SSAValue, before: ir.Statement) -> ir.SSAValue:
    """Insert `range(len(collection))` before `before` and return the range."""
    size = _put(py_len.Len(collection), before)
    zero = _put(py.Constant(0), before)
    one = _put(py.Constant(1), before)
    made = _put(ilist_stmts.Range(zero.result, size.result, one.result), before)
    return made.result


def _set_yield(block: ir.Block, position: int, value: ir.SSAValue) -> None:
    """Replace the yielded value at `position` of the yield that ends `block`."""
    returned = block.last_stmt
    assert isinstance(returned, scf.Yield)
    values = list(returned.values)
    values[position] = value
    _put(scf.Yield(*values), returned)
    returned.delete()


class QubitIteration(RewriteRule):
    """Turn a loop over the qubits of a list into a loop over their indices."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a loop over a list of qubits."""
        if not isinstance(node, scf.For) or not holds_qubits(node.iterable.type):
            return RewriteResult()
        collection = node.iterable
        node.iterable = _range_over(collection, node)
        block = node.body.blocks[0]
        index = block.args[0]
        index.type = types.Int
        first = block.first_stmt
        assert first is not None
        _read_at(index, collection, first)
        return RewriteResult(has_done_something=True)


class MapToLoop(RewriteRule):
    """Turn `ilist.map` and `ilist.for_each` over qubits into a loop of calls."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it maps a function over a list of qubits."""
        if not isinstance(node, (ilist.Map, ilist.ForEach)):
            return RewriteResult()
        if not holds_qubits(node.collection.type):
            return RewriteResult()
        block = ir.Block()
        index = block.args.append_from(types.Int, "i")
        item = py.GetItem(node.collection, index)
        item.result.type = QubitType
        block.stmts.append(item)
        call = self.call(node.fn, item.result)
        if call is None:
            block.delete()
            return RewriteResult()
        block.stmts.append(call)
        iterable = _range_over(node.collection, node)
        if isinstance(node, ilist.ForEach):
            block.stmts.append(scf.Yield())
            _put(scf.For(iterable, ir.Region(block)), node)
            node.delete()
            return RewriteResult(has_done_something=True)
        kind = node.result.type
        element = kind.vars[0] if isinstance(kind, types.Generic) else types.Any
        listed = ilist.IListType[element, types.Any]
        collected = block.args.append_from(listed, "collected")
        # Kirin's `ilist.push` declares its value as a list, so `verify_type`
        # rejects a push of one element. `collected + [value]` appends it.
        single = ilist.New(values=(call.results[0],), elem_type=element)
        block.stmts.append(single)
        appended = py.Add(collected, single.result)
        appended.result.type = listed
        block.stmts.append(appended)
        block.stmts.append(scf.Yield(appended.result))
        empty = _put(ilist.New(values=(), elem_type=element), node)
        loop = _put(scf.For(iterable, ir.Region(block), empty.result), node)
        node.result.replace_by(loop.results[0])
        node.delete()
        return RewriteResult(has_done_something=True)

    def call(self, fn: ir.SSAValue, item: ir.SSAValue) -> ir.Statement | None:
        """Return a call of the function value `fn` on `item`, or None.

        A constant method becomes a `func.Invoke`, and a closure becomes a
        `func.Call`.
        """
        owner = fn.owner
        if isinstance(owner, py.Constant) and isinstance(
            method := owner.value.unwrap(), ir.Method
        ):
            invoke = func.Invoke((item,), callee=method)
            invoke.result.type = method.return_type
            return invoke
        if isinstance(owner, func.Lambda):
            return func.Call(fn, (item,), kwargs=())
        return None


class IndexInBranch(RewriteRule):
    """Carry the index of a qubit that each branch reads from one register."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite the first qubit result of `node` that both branches pick."""
        if not isinstance(node, scf.IfElse) or not node.else_body.blocks:
            return RewriteResult()
        bodies = (node.then_body.blocks[0], node.else_body.blocks[0])
        for k, result in enumerate(node.results):
            if not result.type.is_subseteq(QubitType):
                continue
            picks = []
            for block in bodies:
                returned = block.last_stmt
                assert isinstance(returned, scf.Yield)
                picks.append(_picked(returned.values[k]))
            if None in picks or picks[0][0] is not picks[1][0]:  # type: ignore[index]
                continue
            register = picks[0][0]  # type: ignore[index]
            for block, (_, index) in zip(bodies, picks):  # type: ignore[misc]
                _set_yield(block, k, index)
            result.type = types.Int
            after = node.next_stmt
            assert after is not None
            _read_at(result, register, after)
            return RewriteResult(has_done_something=True)
        return RewriteResult()


class IndexInLoop(RewriteRule):
    """Carry the index of a qubit that a loop reads from one register."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite the first qubit that `node` carries and picks from one register."""
        if not isinstance(node, scf.For):
            return RewriteResult()
        block = node.body.blocks[0]
        returned = block.last_stmt
        if not isinstance(returned, scf.Yield):
            return RewriteResult()
        for k, start in enumerate(node.initializers):
            if not start.type.is_subseteq(QubitType):
                continue
            first, again = _picked(start), _picked(returned.values[k])
            if first is None or again is None or first[0] is not again[0]:
                continue
            register = first[0]
            initializers = list(node.initializers)
            initializers[k] = first[1]
            node.initializers = tuple(initializers)
            _set_yield(block, k, again[1])
            carried = block.args[k + 1]
            carried.type = types.Int
            head = block.first_stmt
            assert head is not None
            _read_at(carried, register, head)
            result = node.results[k]
            result.type = types.Int
            after = node.next_stmt
            assert after is not None
            _read_at(result, register, after)
            return RewriteResult(has_done_something=True)
        return RewriteResult()


@dataclass
class Concatenate(RewriteRule):
    """Turn `a + b` of qubit lists of known length into one literal list.

    `refs` holds the places of the kernel, which give the length of each list.
    """

    refs: Mapping[ir.SSAValue, Ref]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it adds two qubit lists whose lengths the places give."""
        if not isinstance(node, py.Add) or not holds_qubits(node.result.type):
            return RewriteResult()
        reads: list[ir.SSAValue] = []
        for side in (node.lhs, node.rhs):
            ref = self.refs.get(side)
            items = None if ref is None else qubit_items(ref)
            if items is None:
                return RewriteResult()
            for k in range(len(items)):
                at = _put(py.Constant(k), node)
                item = _put(py.GetItem(side, at.result), node)
                item.result.type = QubitType
                reads.append(item.result)
        joined = _put(ilist.New(values=tuple(reads), elem_type=QubitType), node)
        node.result.replace_by(joined.result)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class BroadcastToLoop(RewriteRule):
    """Expand a broadcast whose qubit count depends on runtime values.

    The body broadcasts over singleton lists. Lowering can then replace each
    singleton broadcast with its wire twin. A measurement collects its results
    in a list that the loop carries. The twin declares how the source handles
    unequal lengths.
    """

    refs: Mapping[ir.SSAValue, Ref]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Replace an unknown-length broadcast with an indexed loop."""
        twin = TWINS.get(type(node))
        if twin is None:
            return RewriteResult()
        trait = twin.get_present_trait(Twin)
        operands = [getattr(node, name) for name, _ in trait.wires]
        if all(self.length(value) is not None for value in operands):
            return RewriteResult()
        block = ir.Block()
        index = block.args.append_from(types.Int, "i")
        fields = getattr(type(node), "__kirin_stmt_fields")
        kwargs = {name: getattr(node, name) for name in fields.std_args}
        kwargs.update({name: getattr(node, name) for name in fields.attributes})
        for name, _ in trait.wires:
            item = py.GetItem(getattr(node, name), index)
            item.result.type = QubitType
            block.stmts.append(item)
            single = ilist.New(values=(item.result,), elem_type=QubitType)
            block.stmts.append(single)
            kwargs[name] = single.result
        broadcast = type(node)(**kwargs)
        block.stmts.append(broadcast)
        initializers = []
        yielded = []
        for source, _ in trait.results:
            old = getattr(node, source)
            element = old.type.vars[0]
            empty = _put(ilist.New(values=(), elem_type=element), node)
            collected = block.args.append_from(
                ilist.IListType[element, types.Any], source
            )
            appended = py.Add(collected, getattr(broadcast, source))
            block.stmts.append(appended)
            initializers.append(empty.result)
            yielded.append(appended.result)
        block.stmts.append(scf.Yield(*yielded))
        iterable = self.iterable(node, operands, trait.truncate)
        loop = _put(scf.For(iterable, ir.Region(block), *initializers), node)
        for (source, _), result in zip(trait.results, loop.results, strict=True):
            getattr(node, source).replace_by(result)
        node.delete()
        return RewriteResult(has_done_something=True)

    def length(self, value: ir.SSAValue) -> int | None:
        """Read the list length from its reference or its type."""
        if value in self.refs and (items := qubit_items(self.refs[value])) is not None:
            return len(items)
        return typed_length(value.type, 0)

    def iterable(
        self, node: ir.Statement, operands: list[ir.SSAValue], truncate: bool
    ) -> ir.SSAValue:
        """Build the index range with the source's rule for unequal list lengths."""
        size = _put(py_len.Len(operands[0]), node).result
        for operand in operands[1:]:
            other = _put(py_len.Len(operand), node).result
            if truncate:
                shorter = _put(cmp.Lt(size, other), node).result
                left, right = ir.Block(), ir.Block()
                left.args.append_from(types.Bool)
                right.args.append_from(types.Bool)
                left.stmts.append(scf.Yield(size))
                right.stmts.append(scf.Yield(other))
                size = _put(scf.IfElse(shorter, left, right), node).results[0]
                continue
            equal = _put(cmp.Eq(size, other), node).result
            message = _put(
                py.Constant(f"'{node.name}' requires equal qubit-list lengths."), node
            ).result
            _put(assertion.Assert(equal, message), node)
        zero = _put(py.Constant(0), node).result
        one = _put(py.Constant(1), node).result
        return _put(ilist_stmts.Range(zero, size, one), node).result


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


@dataclass
class TupleProjection(RewriteRule):
    """Resolve a constant tuple projection to the value that constructs its field."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Replace a direct tuple field read with its constructing operand."""
        if not isinstance(node, py.GetItem) or not isinstance(
            node.obj.owner, py.tuple.New
        ):
            return RewriteResult()
        index = constant_int(node.index)
        values = node.obj.owner.args
        if index is None or not -len(values) <= index < len(values):
            return RewriteResult()
        node.result.replace_by(values[index])
        node.delete()
        return RewriteResult(has_done_something=True)
