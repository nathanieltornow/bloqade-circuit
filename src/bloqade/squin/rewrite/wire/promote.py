"""Apply a completed reference-promotion plan with Kirin rewrite rules.

Analysis decides every allocation, transfer and control-flow connection before this module mutates the program.
The rewrite first creates loop arguments, then applies planned operations in region order.
Cleanup removes reference expressions whose quantum accesses the plan replaced.
"""

from typing import TypeVar
from dataclasses import field, dataclass

from kirin import ir, types
from kirin.rewrite import Walk, Chain, DeadCodeElimination
from kirin.dialects import scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.analysis.reference import holds_qubits
from bloqade.squin.rewrite.wire.index import index_value
from bloqade.squin.rewrite.wire.lower import Layout
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint
from bloqade.analysis.reference.analysis import encloses
from bloqade.squin.analysis.wire.promotion import Value, PromotionPlan, plan_promotion

_S = TypeVar("_S", bound=ir.Statement)


def _put(stmt: _S, before: ir.Statement) -> _S:
    """Insert the statement at its planned program position."""
    stmt.insert_before(before)
    return stmt


@dataclass
class ApplyPromotion(RewriteRule):
    """Resolve symbolic definitions and apply the precomputed promotion decisions."""

    plan: PromotionPlan
    values: dict[Value, ir.SSAValue] = field(default_factory=dict)

    def resolve(self, value: Value) -> ir.SSAValue:
        """Resolve a planned definition or a replaced source value to its current SSA value."""
        if value in self.values:
            return self.values[value]
        if isinstance(value, ir.SSAValue) and value in self.plan.replacements:
            return self.resolve(self.plan.replacements[value])
        if isinstance(value, ir.SSAValue):
            return value
        raise ValueError(
            "The promotion plan uses a definition before its construction."
        )

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Insert planned operations, connect region outputs, and remove reference transfers."""
        changed = False
        for operation in self.plan.before.get(node, ()):
            args = tuple(self.resolve(value) for value in operation.operands)
            if operation.index is not None:
                index = index_value(operation.index, node)
                args = (args[0], index, *args[1:])
            made = _put(operation.kind(*args), node)
            self.values.update(zip(operation.results, made.results, strict=True))
            changed = True
        if isinstance(node, scf.Yield) and node in self.plan.yields:
            values = tuple(self.resolve(value) for value in self.plan.yields[node])
            _put(scf.Yield(*node.values, *values), node)
            node.delete()
            return RewriteResult(has_done_something=True)
        if region := self.plan.regions.get(node):
            count = len(node.results)
            if isinstance(node, scf.For):
                body = node.body
                body.detach()
                made = _put(
                    scf.For(
                        node.iterable,
                        body,
                        *node.initializers,
                        *(self.resolve(value) for value in region.inputs),
                    ),
                    node,
                )
            else:
                assert isinstance(node, scf.IfElse)
                then_body, else_body = node.then_body, node.else_body
                then_body.detach()
                else_body.detach()
                made = _put(scf.IfElse(node.cond, then_body, else_body), node)
            for old, new in zip(node.results, made.results[:count], strict=True):
                self.values[old] = new
                old.replace_by(new)
            self.values.update(zip(region.results, made.results[count:], strict=True))
            node.delete()
            return RewriteResult(has_done_something=True)
        if node in self.plan.removed:
            for value in node.results:
                replacement = self.resolve(value)
                self.values[value] = replacement
                value.replace_by(replacement)
            node.delete()
            changed = True
        return RewriteResult(has_done_something=changed)


@dataclass
class LoopArguments(RewriteRule):
    """Create the planned loop arguments before rewriting statements that use them."""

    application: ApplyPromotion

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Bind each symbolic iteration value to an appended block argument."""
        region = self.application.plan.regions.get(node)
        if region is None or not isinstance(node, scf.For):
            return RewriteResult()
        for value in region.arguments:
            self.application.values[value] = node.body.blocks[0].args.append_from(
                value.type
            )
        return RewriteResult(has_done_something=bool(region.arguments))


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
    """Remove reference allocations after their planned wire allocations replace every use."""

    allocations: set[ir.Statement]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Delete a selected allocation when its reference expressions have no remaining uses."""
        if node not in self.allocations or any(value.uses for value in node.results):
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)


def apply_promotion(method: ir.Method, plan: PromotionPlan) -> RewriteResult:
    """Apply a plan to its unchanged source method and remove dead reference expressions."""
    if plan.code is not method.code or plan.applied:
        raise WireConversionError(
            "Apply a promotion plan once to the method that supplied it."
        )
    plan.applied = True
    rewrite = ApplyPromotion(plan)
    result = Walk(LoopArguments(rewrite)).rewrite(method.code)
    result = Walk(rewrite).rewrite(method.code).join(result)
    allocations = {slot.allocation for slot in plan.slots.values()}
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
    return result


def promote(method: ir.Method, layout: Layout | None = None) -> RewriteResult:
    """Plan promotion without mutation, then apply the complete plan once."""
    plan = plan_promotion(method)
    result = apply_promotion(method, plan)
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
