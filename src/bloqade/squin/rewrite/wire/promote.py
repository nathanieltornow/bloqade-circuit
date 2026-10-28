"""Promote explicit reference transfers with one source-order Kirin walk.

Each region owns a map of its current wires. Boundary rules create wire arguments
and results before the walk reaches their bodies. Yield rules close those boundaries.
Structural rules preserve the source statements that Kirin has already scheduled.
"""

from dataclasses import field, dataclass
from collections.abc import Hashable

from kirin import ir
from kirin.rewrite import Walk
from kirin.dialects import scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import holds_qubits
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Sym, Affine, same
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import ReferenceAccess
from bloqade.squin.rewrite.wire.index import index_value
from bloqade.squin.rewrite.wire.handles import Handle, Handles, known


@dataclass(frozen=True)
class Contents:
    """Hold the current wire and the register indices that still require release."""

    value: ir.SSAValue | None
    borrowed: frozenset[Affine] = frozenset()


@dataclass
class RegionState:
    """Track available wires and allocations whose lifetimes end in this region."""

    current: dict[Hashable, Contents] = field(default_factory=dict)
    local: list[Hashable] = field(default_factory=list)

    def value(self, root: Hashable, *, restored: bool = True) -> ir.SSAValue:
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
    """Share prepared locations and region maps across the ordered rewrite rules."""

    locations: Handles
    allocations: set[ir.Statement]
    regions: dict[ir.Region, RegionState] = field(default_factory=dict)
    boundaries: dict[ir.Statement, dict[Hashable, None]] = field(default_factory=dict)
    symbols: dict[Sym, ir.SSAValue] = field(default_factory=dict)

    def frame(self, node: ir.Statement) -> RegionState:
        """Return the map for the statement's current enclosing region."""
        assert node.parent_block is not None and node.parent_block.parent is not None
        return self.regions[node.parent_block.parent]


@dataclass
class CollectCaptures(RewriteRule):
    """Record allocations that transfers use across enclosing region boundaries."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Collect captures in source order while preserving allocation scope."""
        if isinstance(node, (scf.For, scf.IfElse)):
            self.state.boundaries.setdefault(node, {})
        if access := node.get_trait(ReferenceAccess):
            location = self.state.locations.get(access.reference(node))
            if isinstance(location, Handle) and isinstance(location.root, ir.SSAValue):
                allocation = location.root.owner
                if (
                    isinstance(allocation, ir.Statement)
                    and allocation in self.state.allocations
                ):
                    scope = allocation.parent_block
                    assert scope is not None
                    block = node.parent_block
                    while block is not scope:
                        assert block is not None
                        owner = block.parent_stmt
                        if not isinstance(owner, (scf.For, scf.IfElse)):
                            raise WireConversionError(
                                "A quantum capture requires a supported loop or branch."
                            )
                        self.state.boundaries.setdefault(owner, {})[
                            location.root
                        ] = None
                        block = owner.parent_block
        return RewriteResult()


@dataclass
class Promote(RewriteRule):
    """Apply each promotion rule once under Kirin's parent-first walk."""

    state: PromotionState

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Dispatch allocations, transfers, region headers and terminators."""
        if node in self.state.allocations:
            self.allocation(node)
        elif node.has_trait(ReferenceAccess):
            if not self.transfer(node):
                return RewriteResult()
        elif isinstance(node, (scf.For, scf.IfElse)):
            self.boundary(node)
        elif isinstance(node, scf.Yield):
            self.yield_(node)
        elif isinstance(node, func.Return):
            self.state.frame(node).finish(node)
        else:
            return RewriteResult()
        return RewriteResult(has_done_something=True)

    def allocation(self, node: ir.Statement) -> None:
        """Create the initial wire and register the allocation's local lifetime."""
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

    def transfer(self, node: ir.Statement) -> bool:
        """Replace a transfer with its current wire or an explicit register borrow."""
        access = node.get_present_trait(ReferenceAccess)
        location = self.state.locations.get(access.reference(node))
        if not isinstance(location, Handle):
            return False
        root = location.root
        frame = self.state.frame(node)
        if root not in frame.current:
            if isinstance(root, ir.SSAValue) and root.owner in self.state.allocations:
                raise WireConversionError(
                    "A promoted allocation has no wire in the current region."
                )
            return False
        if location.selection is not None:
            if not isinstance(location.selection, Affine):
                raise WireConversionError(
                    "A whole-register transfer requires a whole allocation."
                )
            index = location.selection
            register = frame.value(root, restored=False)
            content = frame.current[root]
            if access.takes:
                if any(same(index, held) is not False for held in content.borrowed):
                    raise WireConversionError(
                        "Promotion requires distinct simultaneous register borrows."
                    )
                at = index_value(index, node, self.state.symbols)
                made = stmts.Borrow(register, at)
                made.insert_before(node)
                access.value(node).replace_by(made.wire)
                frame.current[root] = Contents(
                    made.register_out, content.borrowed | {index}
                )
            else:
                if index not in content.borrowed:
                    raise WireConversionError(
                        "Promotion requires a matching borrow before a register release."
                    )
                at = index_value(index, node, self.state.symbols)
                made = stmts.Release(register, at, access.value(node))
                made.insert_before(node)
                frame.current[root] = Contents(
                    made.register_out, content.borrowed - {index}
                )
        elif access.takes:
            access.value(node).replace_by(frame.value(root))
            frame.current[root] = Contents(None)
        else:
            if frame.current[root].value is not None:
                raise WireConversionError(
                    "Promotion requires an empty reference before a wrap."
                )
            frame.current[root] = Contents(access.value(node))
        node.delete()
        return True

    def boundary(self, node: scf.For | scf.IfElse) -> None:
        """Create boundary results and seed body maps before visiting the original body statements."""
        roots = self.state.boundaries.pop(node)
        outer = self.state.frame(node)
        inputs = tuple(outer.value(root) for root in roots)
        dropped = self.invariant_references(node)
        original = tuple(node.results)
        kept = [i for i in range(len(original)) if i not in dropped]

        if isinstance(node, scf.For):
            body = node.body
            block = body.blocks[0]
            for symbol, _ in self.state.locations.indices.of(block.args[0]).terms:
                self.state.symbols[symbol] = block.args[0]
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
            self.state.symbols[original[i]] = value
            self.state.locations.values[value] = self.state.locations.get(original[i])
            original[i].replace_by(value)
        for root, result in zip(roots, made.results[len(kept) :], strict=True):
            outer.current[root] = Contents(result)
        node.delete()

    def seed(
        self,
        region: ir.Region,
        roots: dict[Hashable, None],
        inputs: tuple[ir.SSAValue, ...],
        dropped: dict[int, ir.SSAValue],
    ) -> None:
        """Seed a body and reserve its wire yield positions without replacing scheduled statements."""
        self.state.regions[region] = RegionState(
            {root: Contents(value) for root, value in zip(roots, inputs, strict=True)}
        )
        end = region.blocks[0].last_stmt
        assert isinstance(end, scf.Yield)
        end.args = (
            *[value for i, value in enumerate(end.values) if i not in dropped],
            *inputs,
        )

    def invariant_references(
        self, node: scf.For | scf.IfElse
    ) -> dict[int, ir.SSAValue]:
        """Find reference results that retain the same source location across the boundary."""
        if isinstance(node, scf.For):
            end = node.body.blocks[0].last_stmt
            assert isinstance(end, scf.Yield)
            return {
                i: value
                for i, (value, yielded) in enumerate(
                    zip(node.initializers, end.values, strict=True)
                )
                if holds_qubits(value.type)
                and known(self.state.locations.get(value))
                and self.state.locations.get(value) == self.state.locations.get(yielded)
            }
        left = node.then_body.blocks[0].last_stmt
        right = node.else_body.blocks[0].last_stmt
        assert isinstance(left, scf.Yield) and isinstance(right, scf.Yield)
        return {
            i: a
            for i, (a, b) in enumerate(zip(left.values, right.values, strict=True))
            if holds_qubits(a.type) and a is b
        }

    def yield_(self, node: scf.Yield) -> None:
        """Close local lifetimes and yield each captured allocation's final wire."""
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


def promote(
    method: ir.Method, locations: Handles, allocations: set[ir.Statement]
) -> None:
    """Collect captures and promote each prepared statement once in source order."""
    state = PromotionState(locations, allocations)
    Walk(CollectCaptures(state), region_first=True).rewrite(method.code)
    state.regions[method.callable_region] = RegionState()
    Walk(Promote(state), region_first=True).rewrite(method.code)
