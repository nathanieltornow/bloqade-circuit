"""Prepare explicit transfers and source locations before ordered wire promotion.

The preparation walk resolves source definitions and checks reference identities at region exits.
Local lowering creates transfers around gates, calls and function returns.
Partial conversion can preserve allocations with unsupported reference uses.
"""

from typing import TypeVar
from dataclasses import field, dataclass
from collections.abc import Callable, Hashable

from kirin import ir, types
from kirin.rewrite import (
    Walk,
    Chain,
    DeadCodeElimination,
)
from kirin.dialects import py, scf, func
from kirin.dialects.py import len as py_len
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.types import holds_qubits
from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Affine
from bloqade.squin.wire.group import kernel as wire_kernel
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.types import WireType, RegisterType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import Twin, ReferenceAccess
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
    known,
    merge,
    roots,
    distinct,
    quantum_type,
    typed_length,
    register_shape,
)
from bloqade.squin.rewrite.wire.promote import promote
from bloqade.squin.rewrite.wire.normalize import TupleProjection, BoundListLengths

_S = TypeVar("_S", bound=ir.Statement)


def _put(stmt: _S, before: ir.Statement) -> _S:
    """Insert a statement at the source access or region boundary."""
    stmt.insert_before(before)
    return stmt


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


@dataclass
class Conversion:
    """Prepare source locations, call contracts and allocation identities for one function.

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
        Walk(LowerTransfers(self), region_first=True).rewrite(method.code)
        if self.promote:
            self.removed = {
                root.owner
                for root in self.allocations
                if isinstance(root, ir.ResultValue) and not self.preserves(root)
            }
            promote(method, self.handles, self.removed)
        assert self.contract is not None
        cleanup = Chain(
            TupleProjection(),
            DeadCodeElimination(),
            RemovePromotedAllocations(self.removed),
        )
        CheckedFixpoint(Walk(cleanup)).rewrite(method.code)
        if any(node.parent_block is not None for node in self.removed):
            raise WireConversionError(
                "Promotion left a live reference to a converted allocation."
            )
        CheckedFixpoint(Walk(DeadCodeElimination())).rewrite(method.code)
        signature = func.Signature(
            inputs=self.input_types, output=packed_type(self.contract.outputs)
        )
        method.code.signature = signature
        block.args[0].type = types.MethodType[list(signature.inputs), signature.output]
        Walk(BoundListLengths()).rewrite(method.code)
        return method


@dataclass
class LowerTransfers(RewriteRule):
    """Resolve source locations and lower operations during one source-order walk."""

    conversion: Conversion

    @property
    def handles(self) -> Handles:
        """Return the source locations shared by the preparation rules."""
        return self.conversion.handles

    def rewrite_Region(self, node: ir.Region) -> RewriteResult:
        """Require one block so source order determines each region's reference definitions."""
        if len(node.blocks) != 1:
            raise WireConversionError(
                "Wire conversion requires single-block structured regions."
            )
        if isinstance(node.parent_node, (scf.For, scf.IfElse)) and not isinstance(
            node.blocks[0].last_stmt, scf.Yield
        ):
            raise WireConversionError(
                "Wire conversion requires structured bodies with yield terminators."
            )
        return RewriteResult()

    def generated(self, nodes: list[ir.Statement]) -> RewriteResult:
        """Convert the statements that a local lowering constructed in their construction order."""
        result = RewriteResult(has_done_something=bool(nodes))
        for node in nodes:
            result = Walk(self, region_first=True).rewrite(node).join(result)
        return result

    def allocation(self, node: ir.Statement) -> RewriteResult:
        """Record the allocation identity, shape and length before wire promotion."""
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
        return RewriteResult()

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Dispatch source definitions, local lowerings and structured boundaries in source order."""
        if isinstance(node, scf.For):
            for argument, value in zip(node.body.blocks[0].args[1:], node.initializers):
                self.handles.values[argument] = self.handles.get(value)
            return RewriteResult()
        if isinstance(node, scf.IfElse):
            return RewriteResult()
        if isinstance(node, scf.Yield):
            return self.yield_(node)
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
            reference = node.get_present_trait(ReferenceAccess).reference(node)
            if not isinstance(self.handles.get(reference), Handle):
                self.conversion.require_references(
                    [reference], f"'{node.name}' has no place in a known allocation."
                )
            return RewriteResult()
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
        """Check owned results and create transfers for the borrowed parameter suffix."""
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
        _put(func.Return(result), node)
        self.generated(local.created)
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

    def yield_(self, node: scf.Yield) -> RewriteResult:
        """Resolve boundary results after the walk has visited each body's source definitions."""
        owner = node.parent_stmt
        descriptors = [self.handles.get(value) for value in node.values]
        if isinstance(owner, scf.For):
            for i, (initial, yielded, after) in enumerate(
                zip(owner.initializers, node.values, descriptors, strict=True)
            ):
                before = self.handles.get(initial)
                if holds_qubits(initial.type) and before != after:
                    self.conversion.require_references(
                        [initial, yielded],
                        "Cannot resolve a value carried by a loop or branch.",
                    )
                    descriptors[i] = merge(before, after)
                elif holds_qubits(initial.type) and known(before):
                    owner.body.blocks[0].args[i + 1].replace_by(initial)
                    owner.results[i].replace_by(initial)
        elif isinstance(owner, scf.IfElse):
            if node.parent_block is owner.then_body.blocks[0]:
                return RewriteResult()
            left = owner.then_body.blocks[0].last_stmt
            assert isinstance(left, scf.Yield)
            descriptors = [
                merge(self.handles.get(a), b)
                for a, b in zip(left.values, descriptors, strict=True)
            ]
            for result, a, b in zip(
                owner.results, left.values, node.values, strict=True
            ):
                if holds_qubits(a.type) and a is b:
                    result.replace_by(a)
        else:
            raise WireConversionError(
                "Wire conversion requires a loop or branch around each yield."
            )
        for result, descriptor in zip(owner.results, descriptors, strict=True):
            self.handles.values[result] = descriptor
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
