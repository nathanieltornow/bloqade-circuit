"""Thread current wire values through reference accesses and structured regions.

Each region owns a rewrite state with its current values and outstanding register borrows.
The rewrite visits statements in source order and creates actual wire values immediately.
Loop and branch rules visit their bodies with separate states, then connect their final values.
"""

from typing import TypeVar
from dataclasses import field, dataclass

from kirin import ir, types
from kirin.rewrite import Walk, Chain, DeadCodeElimination
from kirin.dialects import scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import WireType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import ReferenceAccess
from bloqade.analysis.reference import Root, Place, Affine
from bloqade.analysis.reference.index import Sym, same
from bloqade.squin.analysis.reference import holds_qubits
from bloqade.squin.rewrite.wire.index import index_value
from bloqade.squin.rewrite.wire.lower import Layout
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint
from bloqade.analysis.reference.queries import flat
from bloqade.analysis.reference.analysis import encloses
from bloqade.squin.analysis.wire.promotion import PromotionInfo, analyze_promotion

_S = TypeVar("_S", bound=ir.Statement)


def _put(stmt: _S, before: ir.Statement) -> _S:
    """Insert the statement before the access or boundary that requires it."""
    stmt.insert_before(before)
    return stmt


@dataclass(frozen=True)
class Contents:
    """Hold a current wire value and the indices of outstanding register borrows.

    A missing value means that an unwrap transferred the whole allocation to an explicit wire.
    A later wrap must restore the value before another reference access.
    """

    value: ir.SSAValue | None
    borrowed: frozenset[Affine] = frozenset()


@dataclass
class PromoteReferences(RewriteRule):
    """Replace reference transfers in source order within one structured region.

    Nested regions get separate copies of the incoming state.
    The shared index substitutions preserve reference facts when region results change.
    """

    info: PromotionInfo
    contents: dict[Root, Contents] = field(default_factory=dict)
    local: list[Root] = field(default_factory=list)
    substitutions: dict[Sym, Affine] = field(default_factory=dict)

    def walk(self, region: ir.Region) -> RewriteResult:
        """Visit this region once and let each structured operation visit its own bodies."""
        if len(region.blocks) != 1:
            raise WireConversionError(
                "Promotion requires a single block in each structured region."
            )
        return Walk(
            self, skip=lambda node: isinstance(node, ir.Region) and node is not region
        ).rewrite(region)

    def require(self, root: Root, *, restored=False) -> Contents:
        """Require an available value and optionally require every borrowed element to be present."""
        state = self.contents.get(root)
        if state is None or state.value is None:
            raise WireConversionError(
                f"Promotion requires an available value for {root!r}."
            )
        if restored and state.borrowed:
            raise WireConversionError(
                f"Promotion requires every borrowed element of {root!r} at the region boundary."
            )
        return state

    def value(self, root: Root) -> ir.SSAValue:
        """Return a fully restored value for a region boundary."""
        state = self.require(root, restored=True)
        assert state.value is not None
        return state.value

    def finish(self, node: ir.Statement) -> RewriteResult:
        """Free the allocations that this region still owns at its exit."""
        changed = False
        for root in self.local:
            if self.contents[root].value is None:
                continue
            kind = (
                stmts.Free
                if self.info.slots[root].type == WireType
                else stmts.FreeRegister
            )
            _put(kind(self.value(root)), node)
            self.contents[root] = Contents(None)
            changed = True
        return RewriteResult(has_done_something=changed)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Allocate current values, replace transfers, and thread values through region boundaries."""
        if isinstance(node, scf.For):
            return self.loop(node)
        if isinstance(node, scf.IfElse):
            return self.branch(node)
        if isinstance(node, (func.Return, scf.Yield)):
            return self.finish(node)
        if (
            ref := self.info.accesses.get(node)
        ) is not None and ref.root in self.info.slots:
            return self.transfer(node, ref)
        if slot := self.info.allocations.get(node):
            if isinstance(node, (stmts.Reference, stmts.ReferenceRegister)):
                value = None
            elif slot.type == WireType:
                value = _put(stmts.Alloc(), node).result
            else:
                assert isinstance(node, func.Invoke)
                value = _put(stmts.AllocRegister(*node.inputs), node).result
            self.contents[slot.root] = Contents(value)
            self.local.append(slot.root)
            return RewriteResult(has_done_something=value is not None)
        return RewriteResult()

    def transfer(self, node: ir.Statement, ref: Place) -> RewriteResult:
        """Replace one transfer using the allocation's current value and borrow state."""
        access = node.get_trait(ReferenceAccess)
        assert access is not None
        root = ref.root
        if ref.path:
            index = flat(ref)
            assert isinstance(index, Affine)
            state = self.require(root)
            assert state.value is not None
            if access.takes:
                if any(same(index, held) is not False for held in state.borrowed):
                    raise WireConversionError(
                        "Promotion requires distinct simultaneous register borrows."
                    )
                position = index_value(index.substitute(self.substitutions), node)
                made = _put(stmts.Borrow(state.value, position), node)
                access.value(node).replace_by(made.wire)
                self.contents[root] = Contents(
                    made.register_out, state.borrowed | {index}
                )
            else:
                if index not in state.borrowed:
                    raise WireConversionError(
                        "Promotion requires a matching borrow before a register release."
                    )
                position = index_value(index.substitute(self.substitutions), node)
                released = _put(
                    stmts.Release(state.value, position, access.value(node)), node
                )
                self.contents[root] = Contents(
                    released.register_out, state.borrowed - {index}
                )
        elif access.takes:
            access.value(node).replace_by(self.value(root))
            self.contents[root] = Contents(None)
        else:
            state = self.contents.get(root)
            if state is None or state.value is not None:
                raise WireConversionError(
                    "Promotion requires an empty reference before a wrap."
                )
            self.contents[root] = Contents(access.value(node))
        node.delete()
        return RewriteResult(has_done_something=True)

    def crossing(self, node: ir.Statement) -> tuple[Root, ...]:
        """Select accessed allocations that already exist at this region boundary."""
        touched = {place.root for place in self.info.touched.get(node, ())}
        return tuple(
            root
            for root in self.info.slots
            if root in touched and root in self.contents
        )

    def body(
        self, region: ir.Region, contents: dict[Root, Contents], roots: tuple[Root, ...]
    ) -> RewriteResult:
        """Rewrite one body and append its final outer values to the yield."""
        if len(region.blocks) != 1 or not isinstance(
            region.blocks[0].last_stmt, scf.Yield
        ):
            raise WireConversionError(
                "Promotion requires structured bodies to end with a yield."
            )
        inner = PromoteReferences(
            self.info, dict(contents), substitutions=self.substitutions
        )
        result = inner.walk(region)
        returned = region.blocks[0].last_stmt
        assert isinstance(returned, scf.Yield)
        if roots:
            _put(
                scf.Yield(*returned.values, *(inner.value(root) for root in roots)),
                returned,
            )
            returned.delete()
        return result.join(RewriteResult(has_done_something=bool(roots)))

    def replace_region(
        self, node: ir.Statement, made: ir.Statement, roots: tuple[Root, ...]
    ) -> RewriteResult:
        """Connect original results and retain the final values of each outer allocation."""
        _put(made, node)
        count = len(node.results)
        for old, new in zip(node.results, made.results[:count], strict=True):
            self.substitutions[old] = Affine.of(new)
            old.replace_by(new)
        self.contents.update(
            (root, Contents(value))
            for root, value in zip(roots, made.results[count:], strict=True)
        )
        node.delete()
        return RewriteResult(has_done_something=True)

    def loop(self, node: scf.For) -> RewriteResult:
        """Create iteration arguments before visiting the body, then connect its yield and results."""
        roots = self.crossing(node)
        inputs = tuple(self.value(root) for root in roots)
        contents = dict(self.contents)
        for root in roots:
            argument = node.body.blocks[0].args.append_from(self.info.slots[root].type)
            contents[root] = Contents(argument)
        result = self.body(node.body, contents, roots)
        if not roots:
            return result
        body = node.body
        body.detach()
        made = scf.For(node.iterable, body, *node.initializers, *inputs)
        return self.replace_region(node, made, roots)

    def branch(self, node: scf.IfElse) -> RewriteResult:
        """Visit both branches from the same incoming state and connect matching result positions."""
        roots = self.crossing(node)
        for root in roots:
            self.value(root)
        left = self.body(node.then_body, self.contents, roots)
        right = self.body(node.else_body, self.contents, roots)
        if not roots:
            return left.join(right)
        then_body, else_body = node.then_body, node.else_body
        then_body.detach()
        else_body.detach()
        return self.replace_region(
            node, scf.IfElse(node.cond, then_body, else_body), roots
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


def promote(method: ir.Method, layout: Layout | None = None) -> RewriteResult:
    """Analyze references once, thread current values, and remove unused reference expressions.

    The rewrite changes the method in place and can fail after a partial rewrite.
    SquinToWire applies this rewrite to copies so failures preserve the source methods.
    """
    info = analyze_promotion(method)
    result = (
        PromoteReferences(info).walk(method.callable_region)
        if info.slots
        else RewriteResult()
    )
    allocations = {slot.allocation for slot in info.slots.values()}
    cleanup = Chain(
        DeadCodeElimination(),
        DropInvariantReferences(),
        RemovePromotedAllocations(allocations),
    )
    result = CheckedFixpoint(Walk(cleanup)).rewrite(method.code).join(result)
    if any(node.parent_block is not None for node in allocations):
        raise WireConversionError(
            "Promotion left a live reference to a selected allocation."
        )
    if layout is not None:
        retype(method, layout)
    return result


def retype(method: ir.Method, layout: Layout) -> None:
    """Give the function of `method` the signature of its wire function."""
    code = method.code
    assert isinstance(code, func.Function)
    block = code.body.blocks[0]
    inputs = tuple(arg.type for arg in block.args[1:])
    block.args[0].type = types.MethodType[list(inputs), layout.output]
    code.signature = func.Signature(inputs=inputs, output=layout.output)
