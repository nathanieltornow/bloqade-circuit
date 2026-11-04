"""Run bounded rewrites, insert replacement statements, and reject recursion.

`put` and `replace` are the 2 steps that every rewrite rule of the wire
conversions uses to swap a statement for its replacement.
"""

from typing import TypeVar
from dataclasses import field, dataclass
from collections.abc import Callable

from kirin import ir
from kirin.rewrite import Walk, Fixpoint
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.squin.wire.errors import WireConversionError

StatementT = TypeVar("StatementT", bound=ir.Statement)


def put(stmt: StatementT, before: ir.Statement) -> StatementT:
    """Insert `stmt` before the statement `before` and return `stmt`."""
    stmt.insert_before(before)
    return stmt


def replace(node: ir.Statement, *values: ir.SSAValue) -> None:
    """Replace each result of `node` with the value at its position and delete `node`."""
    for result, value in zip(node.results, values, strict=True):
        result.replace_by(value)
    node.delete()


@dataclass
class CheckedFixpoint(Fixpoint):
    """Report exhaustion of Kirin's rewrite limit as a conversion failure."""

    def rewrite(self, node: ir.IRNode) -> RewriteResult:
        """Run the rules until convergence or report the configured iteration limit."""
        result = super().rewrite(node)
        if result.exceeded_max_iter:
            raise WireConversionError(
                f"Wire rewriting did not converge within {self.max_iter} iterations.",
                node,
            )
        return result


def invoked(node: ir.Statement) -> ir.Method | None:
    """Return the callee of a `func.Invoke`, or None for any other statement."""
    return node.callee if isinstance(node, func.Invoke) else None


@dataclass
class Calls(RewriteRule):
    """Collect static callees through Kirin's walker."""

    callee_of: Callable[[ir.Statement], ir.Method | None] = invoked
    """Return the static callee of a call statement, or None for any other one."""
    callees: dict[ir.Method, None] = field(default_factory=dict)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Record each direct call without changing its statement."""
        if (callee := self.callee_of(node)) is not None:
            self.callees[callee] = None
        return RewriteResult()


def reachable(
    method: ir.Method,
    skip: Callable[[ir.Method], bool] = lambda _: False,
    callee_of: Callable[[ir.Statement], ir.Method | None] = invoked,
) -> list[ir.Method]:
    """Return the reachable methods, callees first, and reject call cycles.

    `callee_of` names the callee of a call statement, such as a Jeff `Call`.
    """
    done: dict[ir.Method, None] = {}
    active: set[ir.Method] = set()

    def visit(current: ir.Method) -> None:
        if current in active:
            raise WireConversionError(
                f"Wire conversion does not support recursive calls to '{current.sym_name}'.",
                current.code,
            )
        if current in done or skip(current):
            return
        active.add(current)
        calls = Calls(callee_of)
        Walk(calls).rewrite(current.code)
        for callee in calls.callees:
            visit(callee)
        active.remove(current)
        done[current] = None

    visit(method)
    return list(done)
