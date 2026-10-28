"""Run bounded rewrites and reject unsupported recursive call graphs."""

from dataclasses import field, dataclass
from collections.abc import Callable

from kirin import ir
from kirin.rewrite import Walk, Fixpoint
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.squin.wire.errors import WireConversionError


@dataclass
class CheckedFixpoint(Fixpoint):
    """Report exhaustion of Kirin's rewrite limit as a conversion failure."""

    def rewrite(self, node: ir.IRNode) -> RewriteResult:
        """Run the rules until convergence or report the configured iteration limit."""
        result = super().rewrite(node)
        if result.exceeded_max_iter:
            raise WireConversionError(
                f"Wire rewriting did not converge within {self.max_iter} iterations."
            )
        return result


@dataclass
class Calls(RewriteRule):
    """Collect static callees through Kirin's walker."""

    callees: dict[ir.Method, None] = field(default_factory=dict)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Record each direct call without changing its statement."""
        if isinstance(node, func.Invoke):
            self.callees[node.callee] = None
        return RewriteResult()


def reachable(
    method: ir.Method, skip: Callable[[ir.Method], bool] = lambda _: False
) -> list[ir.Method]:
    """Return reachable methods and reject call cycles before conversion or validation."""
    done: dict[ir.Method, None] = {}
    active: set[ir.Method] = set()

    def visit(current: ir.Method) -> None:
        if current in active:
            raise WireConversionError(
                f"Wire conversion does not support recursive calls to '{current.sym_name}'."
            )
        if current in done or skip(current):
            return
        active.add(current)
        calls = Calls()
        Walk(calls).rewrite(current.code)
        for callee in calls.callees:
            visit(callee)
        active.remove(current)
        done[current] = None

    visit(method)
    return list(done)
