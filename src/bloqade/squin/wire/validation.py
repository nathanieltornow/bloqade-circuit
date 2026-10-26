"""Check the linearity rules of wire squin with `LinearityValidation`.

A wire or a register must have exactly one use. The use must sit in the region
that defines the value, with one exception. A branch of kirin's `scf.if` has no
carried block arguments, so each branch reads the outer wire itself. Each branch
must then use the outer wire exactly once. A loop body must get every wire as a
block argument, because the body runs more than once.

`LinearityCheck` is a kirin forward analysis. Its fallback rule checks the
results of every statement, and its `scf` rules also check the block arguments
of each body.
"""

from dataclasses import dataclass
from collections.abc import Iterable

from kirin import ir, interp
from kirin.lattice import EmptyLattice
from kirin.analysis import Forward
from kirin.dialects import scf
from kirin.validation import ValidationPass
from kirin.ir.exception import ValidationErrorGroup
from kirin.analysis.forward import ForwardFrame

from .types import is_linear

KEY = "squin.wire.linearity"
"""Name the registry key of the method tables of `LinearityCheck`."""


def _anchor(stmt: ir.Statement, region: ir.Region) -> ir.Statement | None:
    """Return the statement of `region` that holds `stmt`, or None if none holds it."""
    node: ir.Statement | None = stmt
    while node is not None and node.parent_region is not region:
        node = node.parent_stmt
    return node


def misuse(users: list[ir.Statement], region: ir.Region) -> str | None:
    """Return why the statements `users` break the use rule in `region`, or None.

    `users` holds one entry for each use of a linear value that `region` defines.
    """
    if not users:
        return "has no use"
    groups: dict[ir.Statement, list[ir.Statement]] = {}
    for user in users:
        anchor = _anchor(user, region)
        if anchor is None:
            return "has a use outside the region that defines it"
        groups.setdefault(anchor, []).append(user)
    for anchor, inside in groups.items():
        nested = any(user is not anchor for user in inside)
        if nested and not isinstance(anchor, scf.IfElse):
            return (
                f"crosses into the body of '{anchor.name}' outside its block arguments"
            )
    if len(groups) > 1:
        return f"has {len(users)} uses"
    ((anchor, inside),) = groups.items()
    if inside == [anchor]:
        return None
    if anchor in inside:
        return f"has {len(users)} uses"
    # The loop above returns for every nested use outside a branch.
    assert isinstance(anchor, scf.IfElse)
    for body in (anchor.then_body, anchor.else_body):
        branch = [user for user in inside if _anchor(user, body) is not None]
        if (reason := misuse(branch, body)) is not None:
            return f"{reason} in a branch"
    return None


def _region_of(value: ir.SSAValue) -> ir.Region | None:
    """Return the region that defines `value`."""
    if isinstance(value, ir.BlockArgument):
        return value.owner.parent
    owner = value.owner
    return owner.parent_region if isinstance(owner, ir.Statement) else None


@dataclass
class LinearityCheck(Forward[EmptyLattice]):
    """Record a validation error for each linear value that breaks the use rule."""

    keys = (KEY, "absint")
    lattice = EmptyLattice

    def method_self(self, method: ir.Method) -> EmptyLattice:
        """Return the top element for the method object."""
        return EmptyLattice.top()

    def check(self, values: Iterable[ir.SSAValue], node: ir.IRNode) -> None:
        """Record an error at `node` for each linear value of `values` that breaks the rule."""
        for value in values:
            if not is_linear(value.type):
                continue
            region = _region_of(value)
            if region is None:
                continue
            reason = misuse([use.stmt for use in value.uses], region)
            if reason is not None:
                name = value.name or repr(value)
                message = f"the linear value '{name}' {reason}"
                self.add_validation_error(node, ir.ValidationError(node, message))

    def eval_fallback(
        self, frame: ForwardFrame[EmptyLattice], node: ir.Statement
    ) -> tuple[EmptyLattice, ...]:
        """Check the results of a statement without a rule."""
        self.check(node.results, node)
        return tuple(EmptyLattice.top() for _ in node.results)


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """Check the results and the block arguments of loops and branches."""

    @interp.impl(scf.For)
    @interp.impl(scf.IfElse)
    def structured(
        self,
        check: LinearityCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: ir.Statement,
    ) -> tuple[EmptyLattice, ...]:
        """Check the statement, then run each body once."""
        check.check(stmt.results, stmt)
        for region in stmt.regions:
            if not region.blocks:
                continue
            block = region.blocks[0]
            check.check(block.args, stmt)
            args = [EmptyLattice.top() for _ in block.args]
            with check.new_frame(stmt, has_parent_access=True) as inner:
                check.frame_call_region(inner, stmt, region, *args)
        return tuple(EmptyLattice.top() for _ in stmt.results)

    @interp.impl(scf.Yield)
    def yield_(
        self, check: LinearityCheck, frame: ForwardFrame[EmptyLattice], stmt: scf.Yield
    ) -> interp.YieldValue[EmptyLattice]:
        """End a body."""
        return interp.YieldValue(tuple(EmptyLattice.top() for _ in stmt.values))


class LinearityValidation(ValidationPass[ForwardFrame[EmptyLattice]]):
    """Check that a wire squin method uses each wire and register exactly once."""

    def name(self) -> str:
        """Return the pass name that validation reports show."""
        return "squin wire linearity"

    def run(
        self, method: ir.Method
    ) -> tuple[ForwardFrame[EmptyLattice], list[ir.ValidationError]]:
        """Run `LinearityCheck` and return its frame and errors."""
        analysis = LinearityCheck(method.dialects)
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()

    def check(self, method: ir.Method) -> None:
        """Raise a `ValidationErrorGroup` if `method` breaks a linearity rule."""
        _, errors = self.run(method)
        if errors:
            raise ValidationErrorGroup(
                f"'{method.sym_name}' breaks {len(errors)} linearity rules.", errors
            )
