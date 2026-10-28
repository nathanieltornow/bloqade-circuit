"""Convert one normalized function in source order using handles and current wires.

Local rules introduce reference transfers around gates and calls.
The same traversal resolves each transfer and threads whole allocations through structured regions.
Partial conversion can retry a private copy after discovering an allocation that must keep references.
"""

from typing import TypeVar
from dataclasses import field, dataclass
from collections.abc import Callable, Hashable

from kirin import ir, types
from kirin.rewrite import (
    Walk,
    Chain,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)
from kirin.dialects import py, scf, func
from kirin.dialects.py import len as py_len
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.squin.wire.group import kernel as wire_kernel
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.types import WireType, RegisterType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import Twin, ReferenceAccess
from bloqade.analysis.reference.index import Affine, same
from bloqade.squin.analysis.reference import holds_qubits
from bloqade.squin.rewrite.wire.index import index_value
from bloqade.squin.rewrite.wire.lower import (
    Contract,
    LocalLowering,
    packed_type,
    output_types,
)
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint
from bloqade.squin.rewrite.wire.handles import (
    Handle,
    Handles,
    Descriptor,
    merge,
    roots,
    distinct,
    quantum_type,
    typed_length,
    register_shape,
)
from bloqade.squin.rewrite.wire.normalize import TupleProjection, BoundListLengths

_S = TypeVar("_S", bound=ir.Statement)


def _put(stmt: _S, before: ir.Statement) -> _S:
    """Insert a statement at the source access or region boundary."""
    stmt.insert_before(before)
    return stmt


def encloses(scope: ir.Statement, owner: ir.Statement | ir.Block) -> bool:
    """Return whether an owner belongs to a nested region of the statement."""
    node = owner.parent_stmt
    while node is not None:
        if node is scope:
            return True
        node = node.parent_stmt
    return False


class InlineRequired(WireConversionError):
    """Request another normalized copy with this callee inlined."""

    def __init__(self, callee: ir.Method):
        """Record the callee that prevents this call boundary from converting."""
        self.callee = callee
        super().__init__(f"Call '{callee.sym_name}' requires inlining.")


class ReferencesRequired(WireConversionError):
    """Request another private copy that preserves the selected source allocations."""

    def __init__(self, allocations: set[int] | None, reason: str):
        """Record the allocation numbers and the source operation that requires references."""
        self.allocations = allocations
        self.reason = reason
        super().__init__("Partial conversion requires source reference allocations.")


@dataclass(frozen=True)
class Contents:
    """Hold the latest linear value and the indices of outstanding borrows."""

    value: ir.SSAValue | None
    borrowed: frozenset[Affine] = frozenset()


@dataclass
class Conversion:
    """Share source handles, call contracts and allocation identities across region rewrites.

    Allocation numbers follow source visitation order.
    Partial retries reuse those numbers because each private copy contains new SSA objects.
    """

    method: ir.Method
    callee: Callable[[ir.Method], Contract]
    partial: bool = False
    promote: bool = True
    blocked: set[int] = field(default_factory=set)
    block_all: bool = False
    handles: Handles = field(default_factory=Handles)
    allocations: dict[Hashable, int] = field(default_factory=dict)
    removed: set[ir.Statement] = field(default_factory=set)
    borrowed: list[ir.SSAValue] = field(default_factory=list)
    report: list[str] = field(default_factory=list)
    contract: Contract | None = None
    original_output: types.TypeAttribute = field(init=False)
    input_types: tuple[types.TypeAttribute, ...] = field(init=False)
    borrowed_positions: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        """Record the source signature before entry parameters acquire linear types."""
        assert isinstance(self.method.code, func.Function)
        self.original_output = self.method.code.signature.output
        if isinstance(self.original_output, types.BottomType):
            raise WireConversionError(
                "The function return has the impossible type Bottom."
            )
        inputs = tuple(self.method.code.signature.inputs)
        self.borrowed_positions = tuple(
            i for i, kind in enumerate(inputs) if quantum_type(kind) is not None
        )
        self.input_types = tuple(quantum_type(kind) or kind for kind in inputs)

    def preserves(self, root: Hashable) -> bool:
        """Return whether partial conversion must preserve this allocation's reference operations."""
        return self.block_all or self.allocations[root] in self.blocked

    def require_references(self, values: list[ir.SSAValue], reason: str) -> None:
        """Fail complete conversion or retry partial conversion with all possible roots preserved."""
        if not self.partial:
            raise WireConversionError(reason)
        selected: set[int] = set()
        for value in values:
            possible = roots(self.handles.get(value))
            if possible is None:
                if not self.block_all:
                    raise ReferencesRequired(None, reason)
                break
            selected.update(
                self.allocations[root] for root in possible if root in self.allocations
            )
        if not self.block_all and not selected <= self.blocked:
            raise ReferencesRequired(selected, reason)
        self.report.append(reason)

    def run(self) -> ir.Method:
        """Lower entry parameters, visit the function, then remove dead source references."""
        method = self.method
        assert isinstance(method.code, func.Function)
        method.dialects = wire_kernel
        block = method.callable_region.blocks[0]
        first = block.first_stmt
        assert first is not None
        for i in self.borrowed_positions:
            param = block.args[i + 1]
            kind = self.input_types[i]
            fresh = _put(
                stmts.Reference() if kind == WireType else stmts.ReferenceRegister(),
                first,
            )
            fresh.result.type = param.type
            param.replace_by(fresh.result)
            param.type = kind
            _put(
                (
                    stmts.Wrap(fresh.result, param)
                    if kind == WireType
                    else stmts.WrapRegister(fresh.result, param)
                ),
                first,
            )
            self.borrowed.append(fresh.result)
        OrderedPromotion(self).walk(method.callable_region)
        assert self.contract is not None
        cleanup = Chain(
            TupleProjection(),
            DeadCodeElimination(),
            DropInvariantReferences(),
            RemovePromotedAllocations(self.removed),
        )
        CheckedFixpoint(Walk(cleanup)).rewrite(method.code)
        if any(node.parent_block is not None for node in self.removed):
            raise WireConversionError(
                "Promotion left a live reference to a converted allocation."
            )
        CheckedFixpoint(Walk(CommonSubexpressionElimination())).rewrite(method.code)
        CheckedFixpoint(Walk(DeadCodeElimination())).rewrite(method.code)
        signature = func.Signature(
            inputs=self.input_types, output=packed_type(self.contract.outputs)
        )
        method.code.signature = signature
        block.args[0].type = types.MethodType[list(signature.inputs), signature.output]
        Walk(BoundListLengths()).rewrite(method.code)
        return method


@dataclass
class OrderedPromotion(RewriteRule):
    """Resolve source definitions and rewrite transfers once within one structured region."""

    conversion: Conversion
    current: dict[Hashable, Contents] = field(default_factory=dict)
    local: list[Hashable] = field(default_factory=list)

    @property
    def handles(self) -> Handles:
        """Return the definition map shared by all nested region rewrites."""
        return self.conversion.handles

    def walk(self, region: ir.Region) -> RewriteResult:
        """Visit one block and let structured rules own the traversal of nested bodies."""
        if len(region.blocks) != 1:
            raise WireConversionError(
                "Wire conversion requires single-block structured regions."
            )
        return Walk(
            self, skip=lambda node: isinstance(node, ir.Region) and node is not region
        ).rewrite(region)

    def generated(self, nodes: list[ir.Statement]) -> RewriteResult:
        """Convert the statements that a local lowering constructed in their construction order."""
        result = RewriteResult(has_done_something=bool(nodes))
        for node in nodes:
            result = self.rewrite(node).join(result)
        return result

    def value(self, root: Hashable, *, restored=True) -> ir.SSAValue:
        """Require an available allocation value and optionally require every borrowed element to be present."""
        state = self.current.get(root)
        if state is None or state.value is None:
            raise WireConversionError(
                "Promotion requires an available value before an unwrap."
            )
        if restored and state.borrowed:
            raise WireConversionError(
                "Promotion requires every borrowed element at a region boundary."
            )
        return state.value

    def finish(self, before: ir.Statement) -> None:
        """Free local allocations whose values still belong to this region."""
        for root in self.local:
            if self.current[root].value is None:
                continue
            handle = self.handles.get(root)
            assert isinstance(handle, Handle)
            kind = stmts.Free if handle.kind == WireType else stmts.FreeRegister
            _put(kind(self.value(root)), before)
            self.current[root] = Contents(None)

    def allocation(self, node: ir.Statement) -> RewriteResult:
        """Create a stable handle and initialize its current value at the source allocation."""
        conv = self.conversion
        root = node.results[0]
        kind = (
            WireType
            if isinstance(node, (qubit_stmts.New, stmts.Reference))
            else RegisterType
        )
        length = typed_length(root.type)
        if isinstance(node, func.Invoke):
            size = constant_int(node.inputs[0])
            length = None if size is None else max(0, size)
            self.handles.extents[root] = self.handles.indices.of(node.inputs[0])
        shape = register_shape(root.type)
        if len(shape) > 1:
            length = 1
            for dimension in shape:
                if dimension is None:
                    length = None
                    break
                length *= dimension
        self.handles.values[root] = Handle(
            root, kind, None if length is None else Affine(length), shape=shape
        )

        conv.allocations[root] = len(conv.allocations)
        if not conv.promote or conv.preserves(root):
            return RewriteResult()
        if isinstance(node, (stmts.Reference, stmts.ReferenceRegister)):
            value = None
        elif kind == WireType:
            value = _put(stmts.Alloc(), node).result
        else:
            assert isinstance(node, func.Invoke)
            value = _put(stmts.AllocRegister(node.inputs[0]), node).result
        self.current[root] = Contents(value)
        self.local.append(root)
        conv.removed.add(node)
        return RewriteResult(has_done_something=value is not None)

    def transfer(self, node: ir.Statement) -> RewriteResult:
        """Resolve one transfer from its handle and the current allocation value."""
        access = node.get_present_trait(ReferenceAccess)
        handle = self.handles.get(access.reference(node))
        if not isinstance(handle, Handle):
            self.conversion.require_references(
                [access.reference(node)],
                f"'{node.name}' has no place in a known allocation.",
            )
            return RewriteResult()
        root = handle.root
        if root not in self.current:
            return RewriteResult()
        if handle.selection is not None:
            if not isinstance(handle.selection, Affine):
                raise WireConversionError(
                    "A whole-register transfer requires a whole allocation."
                )
            index = handle.selection
            value = self.value(root, restored=False)
            state = self.current[root]
            if access.takes:
                if any(same(index, held) is not False for held in state.borrowed):
                    raise WireConversionError(
                        "Promotion requires distinct simultaneous register borrows."
                    )
                at = index_value(index, node)
                made = _put(stmts.Borrow(value, at), node)
                access.value(node).replace_by(made.wire)
                self.current[root] = Contents(
                    made.register_out, state.borrowed | {index}
                )
            else:
                if index not in state.borrowed:
                    raise WireConversionError(
                        "Promotion requires a matching borrow before a register release."
                    )
                at = index_value(index, node)
                made = _put(stmts.Release(value, at, access.value(node)), node)
                self.current[root] = Contents(
                    made.register_out, state.borrowed - {index}
                )
        elif access.takes:
            access.value(node).replace_by(self.value(root))
            self.current[root] = Contents(None)
        else:
            if self.current[root].value is not None:
                raise WireConversionError(
                    "Promotion requires an empty reference before a wrap."
                )
            self.current[root] = Contents(access.value(node))
        node.delete()
        return RewriteResult(has_done_something=True)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Dispatch source definitions, local lowerings and structured boundaries in source order."""
        if isinstance(node, scf.For):
            return self.loop(node)
        if isinstance(node, scf.IfElse):
            return self.branch(node)
        if isinstance(node, scf.Yield):
            self.finish(node)
            return RewriteResult()
        if isinstance(node, func.Return):
            return self.return_(node)
        if (
            isinstance(
                node, (qubit_stmts.New, stmts.Reference, stmts.ReferenceRegister)
            )
            or isinstance(node, func.Invoke)
            and node.callee is qalloc
        ):
            return self.allocation(node)
        if node.has_trait(ReferenceAccess):
            return self.transfer(node)
        if isinstance(node, func.Invoke) and node.callee.dialects is not wire_kernel:
            return self.call(node)
        if self.handles.record(node):
            return RewriteResult()
        if isinstance(node, py_len.Len) and holds_qubits(node.value.type):
            return self.length(node)
        if type(node) in TWINS:
            operands = [
                getattr(node, name)
                for name, _ in TWINS[type(node)].get_present_trait(Twin).wires
            ]
            possible = [roots(self.handles.get(value)) for value in operands]
            if self.conversion.promote and any(
                group and any(self.conversion.preserves(root) for root in group)
                for group in possible
                if group is not None
            ):
                self.conversion.require_references(
                    operands,
                    "A gate uses an allocation that requires reference semantics.",
                )
                return RewriteResult()
            try:
                made = LocalLowering(node, self.handles).gate()
            except WireConversionError as error:
                self.conversion.require_references(operands, str(error))
                return RewriteResult()
            return self.generated(made)
        values = [value for value in node.args if holds_qubits(value.type)]
        if values:
            self.conversion.require_references(
                values, f"Unsupported reference operation '{node.name}'."
            )
        return RewriteResult()

    def length(self, node: py_len.Len) -> RewriteResult:
        """Lower a register length using its descriptor or an explicit wire length operation."""
        size = self.handles.size(node.value)
        if size is not None:
            node.result.replace_by(index_value(size, node))
            node.delete()
            return RewriteResult(has_done_something=True)
        handle = self.handles.get(node.value)
        if not isinstance(handle, Handle) or handle.selection is not None:
            self.conversion.require_references(
                [node.value], "The register length has no place in a known allocation."
            )
            return RewriteResult()
        local = LocalLowering(node, self.handles)
        register = local.take(node.value)
        length = local.put(stmts.Length(register))
        local.give(node.value, length.register_out)
        node.result.replace_by(length.result)
        node.delete()
        return self.generated(local.created)

    def call(self, node: func.Invoke) -> RewriteResult:
        """Require separated whole arguments and use the callee's checked wire contract."""
        if any(value.type.is_subseteq(types.Tuple) for value in node.inputs):
            raise InlineRequired(node.callee)
        quantum = [
            self.handles.get(value) for value in node.inputs if holds_qubits(value.type)
        ]
        for i, handle in enumerate(quantum):
            if not isinstance(handle, Handle) or not (
                handle.scalar or handle.selection is None
            ):
                raise InlineRequired(node.callee)
            if any(
                not isinstance(other, Handle) or not distinct(handle, other)
                for other in quantum[i + 1 :]
            ):
                raise InlineRequired(node.callee)
        try:
            contract = self.conversion.callee(node.callee)
        except WireConversionError:
            raise InlineRequired(node.callee) from None
        return self.generated(LocalLowering(node, self.handles).call(contract))

    def return_(self, node: func.Return) -> RewriteResult:
        """Check owned results, return borrowed parameters, and finish local lifetimes."""
        conv = self.conversion
        if node.parent_stmt is not conv.method.code:
            raise WireConversionError(
                "Wire conversion requires a function-level return."
            )
        original = output_types(conv.original_output)
        packed = conv.original_output.is_subseteq(types.Tuple) and bool(original)
        local = LocalLowering(node, self.handles)
        if not original:
            values = []
        elif not packed:
            values = [node.value]
        elif isinstance(node.value.owner, py.tuple.New):
            values = list(node.value.owner.args)
        else:
            values = []
            for i, kind in enumerate(original):
                at = local.put(py.Constant(i)).result
                item = local.put(py.GetItem(node.value, at))
                item.result.type = kind
                values.append(item.result)
            self.generated(local.created)
            local.created.clear()
        borrowed_roots = {
            handle.root
            for value in conv.borrowed
            if isinstance(handle := self.handles.get(value), Handle)
        }
        seen = set()
        ordinary, lengths, returned = [], [], []
        for value in values:
            handle = self.handles.get(value)
            if holds_qubits(value.type):
                if not isinstance(handle, Handle) or handle.selection is not None:
                    raise WireConversionError(
                        "A quantum return must own one whole allocation."
                    )
                if handle.root in borrowed_roots:
                    raise WireConversionError(
                        "A function cannot return a borrowed quantum parameter as an ordinary output."
                    )
                if handle.root in seen:
                    raise WireConversionError(
                        "A function cannot return one qubit twice."
                    )
                seen.add(handle.root)
                ordinary.append(handle.kind)
                lengths.append(handle.length or self.handles.extents.get(handle.root))
                returned.append(local.take(value))
            else:
                ordinary.append(value.type)
                lengths.append(None)
                returned.append(value)
        returned.extend(local.take(value) for value in conv.borrowed)
        if not returned:
            result = _put(func.ConstantNone(), node).result
        elif len(returned) == 1:
            result = returned[0]
        else:
            result = _put(py.tuple.New(tuple(returned)), node).result
        returned_node = _put(func.Return(result), node)
        self.generated(local.created)
        self.finish(returned_node)
        node.delete()
        conv.contract = Contract(
            conv.method,
            conv.input_types,
            tuple(ordinary),
            conv.borrowed_positions,
            tuple(lengths),
            packed,
        )
        return RewriteResult(has_done_something=True)

    def body(
        self, region: ir.Region, current: dict[Hashable, Contents]
    ) -> "OrderedPromotion":
        """Visit one body with separate current values and local lifetimes."""
        if len(region.blocks) != 1 or not isinstance(
            region.blocks[0].last_stmt, scf.Yield
        ):
            raise WireConversionError(
                "Wire conversion requires structured bodies with yield terminators."
            )
        inner = OrderedPromotion(self.conversion, dict(current))
        inner.walk(region)
        return inner

    def append_yield(
        self, region: ir.Region, inner: "OrderedPromotion", roots_: tuple[Hashable, ...]
    ) -> list[Descriptor]:
        """Preserve original result descriptors and append the current outer allocations."""
        end = region.blocks[0].last_stmt
        assert isinstance(end, scf.Yield)
        descriptors = [self.handles.get(value) for value in end.values]
        if roots_:
            _put(scf.Yield(*end.values, *(inner.value(root) for root in roots_)), end)
            end.delete()
        return descriptors

    def replace_region(
        self,
        node: ir.Statement,
        made: ir.Statement,
        descriptors: list[Descriptor],
        roots_: tuple[Hashable, ...],
    ) -> RewriteResult:
        """Map source results and replace the enclosing current values with region results."""
        _put(made, node)
        count = len(node.results)
        for old, new, descriptor in zip(
            node.results, made.results[:count], descriptors, strict=True
        ):
            self.handles.values[new] = descriptor
            old.replace_by(new)
        self.current.update(
            (root, Contents(value))
            for root, value in zip(roots_, made.results[count:], strict=True)
        )
        node.delete()
        return RewriteResult(has_done_something=True)

    def loop(self, node: scf.For) -> RewriteResult:
        """Carry every available outer allocation through the loop and check reference invariants."""
        roots_ = tuple(
            root for root, state in self.current.items() if state.value is not None
        )
        inputs = tuple(self.value(root) for root in roots_)
        initial = [self.handles.get(value) for value in node.initializers]
        for argument, descriptor in zip(node.body.blocks[0].args[1:], initial):
            self.handles.values[argument] = descriptor
        current = dict(self.current)
        for root, value in zip(roots_, inputs):
            current[root] = Contents(node.body.blocks[0].args.append_from(value.type))
        inner = self.body(node.body, current)
        descriptors = self.append_yield(node.body, inner, roots_)
        end = node.body.blocks[0].last_stmt
        assert isinstance(end, scf.Yield)
        for i, (initial_value, yielded, before, after) in enumerate(
            zip(node.initializers, end.values, initial, descriptors)
        ):
            if before != after and holds_qubits(initial_value.type):
                self.conversion.require_references(
                    [initial_value, yielded],
                    "Cannot resolve a value carried by a loop or branch.",
                )
                descriptors[i] = merge(before, after)
        body = node.body
        body.detach()
        return self.replace_region(
            node,
            scf.For(node.iterable, body, *node.initializers, *inputs),
            descriptors,
            roots_,
        )

    def branch(self, node: scf.IfElse) -> RewriteResult:
        """Start both branches with the same values and join their matching result positions."""
        roots_ = tuple(
            root for root, state in self.current.items() if state.value is not None
        )
        for root in roots_:
            self.value(root)
        left = self.body(node.then_body, self.current)
        a = self.append_yield(node.then_body, left, roots_)
        right = self.body(node.else_body, self.current)
        b = self.append_yield(node.else_body, right, roots_)
        descriptors = [merge(x, y) for x, y in zip(a, b, strict=True)]
        then_body, else_body = node.then_body, node.else_body
        then_body.detach()
        else_body.detach()
        return self.replace_region(
            node, scf.IfElse(node.cond, then_body, else_body), descriptors, roots_
        )


class DropInvariantReferences(RewriteRule):
    """Remove references that control flow returns unchanged.

    Kirin's lowering carries every variable of a loop body, so a loop over `x(q)`
    carries the reference `q`. After promotion, the qubit itself travels as a
    wire, and the carried reference keeps only its allocation alive.
    If both branches yield the same reference, the branch result names that reference.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Remove the invariant qubit references that a loop or branch carries."""
        if isinstance(node, scf.IfElse) and node.parent_block is not None:
            return self.branch(node)
        if not isinstance(node, scf.For) or node.parent_block is None:
            return RewriteResult()
        block = node.body.blocks[0]
        returned = block.last_stmt
        if not isinstance(returned, scf.Yield):
            return RewriteResult()
        # A body yields an invariant reference back as its block argument, or as
        # the initializer once an earlier drop has replaced the argument.
        dropped = [
            k
            for k, arg in enumerate(block.args[1:])
            if holds_qubits(arg.type)
            and returned.values[k] in (arg, node.initializers[k])
        ]
        if not dropped:
            return RewriteResult()
        for k in dropped:
            block.args[k + 1].replace_by(node.initializers[k])
            node.results[k].replace_by(node.initializers[k])
        kept_values = [v for k, v in enumerate(returned.values) if k not in dropped]
        _put(scf.Yield(*kept_values), returned)
        returned.delete()
        for k in reversed(dropped):
            block.args[k + 1].delete()
        region = node.body
        region.detach()
        inits = [v for k, v in enumerate(node.initializers) if k not in dropped]
        made = _put(scf.For(node.iterable, region, *inits), node)
        kept = [r for k, r in enumerate(node.results) if k not in dropped]
        for old, new in zip(kept, made.results, strict=True):
            old.replace_by(new)
        node.delete()
        return RewriteResult(has_done_something=True)

    def branch(self, node: scf.IfElse) -> RewriteResult:
        """Replace a result with the reference that both branches yield unchanged."""
        if not node.then_body.blocks or not node.else_body.blocks:
            return RewriteResult()
        left = node.then_body.blocks[0].last_stmt
        right = node.else_body.blocks[0].last_stmt
        if not isinstance(left, scf.Yield) or not isinstance(right, scf.Yield):
            return RewriteResult()
        dropped = {
            k
            for k, (a, b) in enumerate(zip(left.values, right.values, strict=True))
            if holds_qubits(a.type) and a is b and not encloses(node, a.owner)
        }
        if not dropped:
            return RewriteResult()
        for k in dropped:
            node.results[k].replace_by(left.values[k])
        for returned in (left, right):
            _put(
                scf.Yield(
                    *(v for k, v in enumerate(returned.values) if k not in dropped)
                ),
                returned,
            )
            returned.delete()
        then_body, else_body = node.then_body, node.else_body
        then_body.detach()
        else_body.detach()
        made = _put(scf.IfElse(node.cond, then_body, else_body), node)
        kept = [result for k, result in enumerate(node.results) if k not in dropped]
        for old, new in zip(kept, made.results, strict=True):
            old.replace_by(new)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class RemovePromotedAllocations(RewriteRule):
    """Remove reference allocations after wire allocations replace every use."""

    allocations: set[ir.Statement]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Delete a selected allocation when its reference expressions have no remaining uses."""
        if node not in self.allocations or any(value.uses for value in node.results):
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)
