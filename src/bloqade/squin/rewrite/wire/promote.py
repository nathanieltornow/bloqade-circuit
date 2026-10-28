"""Promote prepared transfers through separate rules under one ordered Kirin walk.

Preparation supplies whole allocations, element indices and invariant reference results.
Each region owns its current wires. Headers seed body arguments before the walk visits
those bodies. Terminators yield the final wires and close local allocation lifetimes.
"""

from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir
from kirin.rewrite import Walk, Chain
from kirin.dialects import scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts
from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import ReferenceAccess
from bloqade.squin.rewrite.wire.index import compare_indices


@dataclass(frozen=True)
class Whole:
    """Identify one whole source allocation."""

    root: ir.ResultValue


@dataclass(frozen=True)
class Element:
    """Identify one element with an integer SSA value available at the transfer."""

    root: ir.ResultValue
    index: ir.SSAValue


@dataclass(frozen=True)
class PromotionInput:
    """Supply concrete transfer locations and reference results that retain their source identity.

    Each location belongs to one transfer because its index must dominate that use.
    Partial conversion omits transfers whose allocations retain reference semantics.
    """

    locations: Mapping[ir.Statement, Whole | Element]
    allocations: frozenset[ir.Statement]
    invariants: Mapping[ir.Statement, Mapping[int, ir.SSAValue]]


@dataclass(frozen=True)
class Contents:
    """Hold the current wire and the register indices that still require release."""

    value: ir.SSAValue | None
    borrowed: tuple[ir.SSAValue, ...] = ()


@dataclass
class RegionState:
    """Track available wires and allocations whose lifetimes end in this region."""

    current: dict[ir.ResultValue, Contents] = field(default_factory=dict)
    local: list[ir.ResultValue] = field(default_factory=list)

    def value(self, root: ir.ResultValue, *, restored: bool = True) -> ir.SSAValue:
        """Require an available wire and, at boundaries, a complete register."""
        content = self.current.get(root)
        if content is None or content.value is None:
            raise WireConversionError(
                "Promotion requires an available value before an unwrap."
            )
        if restored and content.borrowed:
            raise WireConversionError(
                "Promotion requires every borrowed element at a region boundary."
            )
        return content.value

    def finish(self, before: ir.Statement) -> None:
        """Free local allocations that the region still owns at its terminator."""
        for root in self.local:
            if self.current[root].value is None:
                continue
            value = self.value(root)
            kind = stmts.Free if value.type == WireType else stmts.FreeRegister
            kind(value).insert_before(before)
            self.current[root] = Contents(None)


@dataclass
class PromotionState:
    """Share prepared input and mutable region maps across the promotion rules."""

    inputs: PromotionInput
    regions: dict[ir.Region, RegionState] = field(default_factory=dict)
    boundaries: dict[ir.Statement, tuple[ir.ResultValue, ...]] = field(
        default_factory=dict
    )
    replacements: dict[ir.SSAValue, ir.SSAValue] = field(default_factory=dict)

    def frame(self, node: ir.Statement) -> RegionState:
        """Return the map for the statement's current enclosing region."""
        assert node.parent_block is not None and node.parent_block.parent is not None
        return self.regions[node.parent_block.parent]

    def index(self, location: Element) -> ir.SSAValue:
        """Resolve an index whose defining region header has acquired new results."""
        return self.replacements.get(location.index, location.index)


@dataclass
class PromoteEntry(RewriteRule):
    """Open the function's current-wire map before visiting its body."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Initialize region state for the prepared function entry."""
        if isinstance(node, func.Function):
            self.state.regions[node.body] = RegionState()
        return RewriteResult()


@dataclass
class PromoteAllocation(RewriteRule):
    """Create initial wires and record their local lifetimes."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Promote a selected allocation or initialize an empty reference for an incoming wire."""
        if node not in self.state.inputs.allocations:
            return RewriteResult()
        frame = self.state.frame(node)
        root = node.results[0]
        if isinstance(node, (stmts.Reference, stmts.ReferenceRegister)):
            value = None
        else:
            if isinstance(node, qubit_stmts.New):
                made = stmts.Alloc()
            else:
                assert isinstance(node, func.Invoke)
                made = stmts.AllocRegister(node.inputs[0])
            made.insert_before(node)
            value = made.results[0]
        frame.current[root] = Contents(value)
        frame.local.append(root)
        return RewriteResult(has_done_something=value is not None)


@dataclass
class PromoteUnwrap(RewriteRule):
    """Replace an unwrap with the available wire or an explicit register borrow."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Consume the current value at a prepared transfer location."""
        access = node.get_trait(ReferenceAccess)
        if access is None or not access.takes:
            return RewriteResult()
        location = self.state.inputs.locations.get(node)
        if location is None:
            return RewriteResult()
        frame = self.state.frame(node)
        root = location.root
        value = frame.value(root, restored=isinstance(location, Whole))
        if isinstance(location, Element):
            index = self.state.index(location)
            borrowed = frame.current[root].borrowed
            if any(compare_indices(index, held) is not False for held in borrowed):
                raise WireConversionError(
                    "Promotion requires distinct simultaneous register borrows."
                )
            made = stmts.Borrow(value, index)
            made.insert_before(node)
            access.value(node).replace_by(made.wire)
            frame.current[root] = Contents(made.register_out, (*borrowed, index))
        else:
            access.value(node).replace_by(value)
            frame.current[root] = Contents(None)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class PromoteWrap(RewriteRule):
    """Restore a whole allocation or release a borrowed register element."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Record the replacement wire at a prepared transfer location."""
        access = node.get_trait(ReferenceAccess)
        if access is None or access.takes:
            return RewriteResult()
        location = self.state.inputs.locations.get(node)
        if location is None:
            return RewriteResult()
        frame = self.state.frame(node)
        root = location.root
        if isinstance(location, Element):
            register = frame.value(root, restored=False)
            index = self.state.index(location)
            borrowed = frame.current[root].borrowed
            position = next(
                (
                    i
                    for i, held in enumerate(borrowed)
                    if compare_indices(index, held) is True
                ),
                None,
            )
            if position is None:
                raise WireConversionError(
                    "Promotion requires a matching borrow before a register release."
                )
            made = stmts.Release(register, index, access.value(node))
            made.insert_before(node)
            frame.current[root] = Contents(
                made.register_out, borrowed[:position] + borrowed[position + 1 :]
            )
        else:
            content = frame.current.get(root)
            if content is None or content.value is not None:
                raise WireConversionError(
                    "Promotion requires an empty reference before a wrap."
                )
            frame.current[root] = Contents(access.value(node))
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class PromoteBoundary(RewriteRule):
    """Carry available allocations through loop and branch boundaries."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Create boundary results and seed bodies while preserving scheduled statements."""
        if not isinstance(node, (scf.For, scf.IfElse)):
            return RewriteResult()
        outer = self.state.frame(node)
        roots = tuple(
            root for root, content in outer.current.items() if content.value is not None
        )
        inputs = tuple(outer.value(root) for root in roots)
        dropped = self.state.inputs.invariants.get(node, {})
        original = tuple(node.results)
        kept = [i for i in range(len(original)) if i not in dropped]

        if isinstance(node, scf.For):
            body = node.body
            block = body.blocks[0]
            for i, value in dropped.items():
                block.args[i + 1].replace_by(value)
            for i in sorted(dropped, reverse=True):
                block.args[i + 1].delete()
            arguments = tuple(block.args.append_from(value.type) for value in inputs)
            self.seed(body, roots, arguments, dropped)
            initializers = tuple(
                value for i, value in enumerate(node.initializers) if i not in dropped
            )
            body.detach()
            made = scf.For(node.iterable, body, *initializers, *inputs)
        else:
            left, right = node.then_body, node.else_body
            self.seed(left, roots, inputs, dropped)
            self.seed(right, roots, inputs, dropped)
            left.detach()
            right.detach()
            made = scf.IfElse(node.cond, left, right)

        made.attributes.update(node.attributes)
        made.source = node.source
        made.insert_before(node)
        self.state.boundaries[made] = roots
        for i, value in dropped.items():
            original[i].replace_by(value)
        for i, value in zip(kept, made.results[: len(kept)], strict=True):
            self.state.replacements[original[i]] = value
            original[i].replace_by(value)
        for root, result in zip(roots, made.results[len(kept) :], strict=True):
            outer.current[root] = Contents(result)
        node.delete()
        return RewriteResult(has_done_something=True)

    def seed(
        self,
        region: ir.Region,
        roots: tuple[ir.ResultValue, ...],
        inputs: tuple[ir.SSAValue, ...],
        dropped: Mapping[int, ir.SSAValue],
    ) -> None:
        """Seed a body and reserve its wire yield positions while retaining the original terminator."""
        self.state.regions[region] = RegionState(
            {root: Contents(value) for root, value in zip(roots, inputs, strict=True)}
        )
        end = region.blocks[0].last_stmt
        assert isinstance(end, scf.Yield)
        end.args = (
            *[value for i, value in enumerate(end.values) if i not in dropped],
            *inputs,
        )


@dataclass
class PromoteYield(RewriteRule):
    """Yield current wires and close the body's local allocation lifetimes."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Close a prepared loop or branch body at its original terminator."""
        if not isinstance(node, scf.Yield):
            return RewriteResult()
        owner = node.parent_stmt
        assert owner is not None
        roots = self.state.boundaries[owner]
        frame = self.state.frame(node)
        frame.finish(node)
        if roots:
            count = len(node.values) - len(roots)
            node.args = (*node.values[:count], *(frame.value(root) for root in roots))
        assert node.parent_block is not None and node.parent_block.parent is not None
        del self.state.regions[node.parent_block.parent]
        return RewriteResult(has_done_something=True)


@dataclass
class PromoteReturn(RewriteRule):
    """Close local allocation lifetimes before the prepared function return."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Free remaining local wires after the return transfers consume the output wires."""
        if not isinstance(node, func.Return):
            return RewriteResult()
        self.state.frame(node).finish(node)
        return RewriteResult(has_done_something=True)


def promote(method: ir.Method, inputs: PromotionInput) -> None:
    """Apply the promotion rules once to the caller's prepared private method copy."""
    state = PromotionState(inputs)
    rules = Chain(
        PromoteEntry(state),
        PromoteAllocation(state),
        PromoteBoundary(state),
        PromoteUnwrap(state),
        PromoteWrap(state),
        PromoteYield(state),
        PromoteReturn(state),
    )
    Walk(rules, region_first=True).rewrite(method.code)
