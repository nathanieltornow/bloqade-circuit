"""Check wire squin with the kirin validation suite `WireValidation`.

The suite runs four kirin validation passes. `StructureValidation` runs kirin's
own `verify` and `verify_type`. `LinearityValidation` checks the use rule below.
`BorrowValidation` checks that each `Borrow` has a matching `Release`.
`ReferenceValidation`, in `reference_check`, checks how wires leave and enter
qubit references.

The use rule:

A wire or a register must have exactly one use. The use must sit in the region
that defines the value, with one exception. A branch of kirin's `scf.if` has no
carried block arguments, so each branch reads the outer wire itself. Each branch
must then use the outer wire exactly once. A loop body must get every wire as a
block argument, because the body runs more than once.

`LinearityCheck` is a kirin forward analysis. Its fallback rule checks the
results of every statement, and its `scf` rules also check the block arguments
of each body. Kirin's `absint` rule for calls runs each callee, so the check also
covers every wire function that the method calls.

The borrow rule: a register value holds a set of borrowed slots. `Borrow` adds
its index, and `Release` removes it. A slot must not be borrowed twice. A register
must hold no borrowed slot when a statement frees it, yields it, returns it or
passes it to a call. `BorrowCheck` is a kirin forward analysis over the lattice
`Slots`.
"""

from typing import Hashable
from dataclasses import dataclass
from collections.abc import Iterable

from kirin import ir, interp
from kirin.lattice import EmptyLattice, BoundedLattice
from kirin.analysis import Forward
from kirin.dialects import scf, func
from kirin.validation import ValidationPass, ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from kirin.analysis.forward import ForwardFrame

from bloqade.constants import constant_int

from . import stmts
from .twins import TWINS
from .types import RegisterType, is_linear
from .reference_check import ReferenceValidation

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

    def frame_call(
        self,
        frame: ForwardFrame[EmptyLattice],
        node: ir.Statement,
        *args: EmptyLattice,
        **kwargs: EmptyLattice,
    ) -> EmptyLattice:
        """Check the parameters of the function `node`, then run its body."""
        trait = node.get_trait(ir.CallableStmtInterface)
        if trait is not None:
            self.check(trait.get_callable_region(node).blocks[0].args[1:], node)
        return super().frame_call(frame, node, *args, **kwargs)

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


BORROW_KEY = "squin.wire.borrows"
"""Name the registry key of the method tables of `BorrowCheck`."""


@dataclass(frozen=True)
class Slots(BoundedLattice["Slots"]):
    """Hold the borrowed slots of one register value.

    `held` holds one key for each borrowed index. A constant index has its integer
    as the key, and any other index has its SSA value. If `held` is None, the
    analysis cannot name the slots. That element is the top of the lattice.
    """

    held: frozenset[Hashable] | None

    @classmethod
    def top(cls) -> "Slots":
        """Return the element whose slots the analysis cannot name."""
        return cls(None)

    @classmethod
    def bottom(cls) -> "Slots":
        """Return the element of a register without a borrowed slot."""
        return cls(frozenset())

    def is_subseteq(self, other: "Slots") -> bool:
        """Return True if `other` holds every slot of `self`."""
        if other.held is None:
            return True
        return self.held is not None and self.held <= other.held

    def join(self, other: "Slots") -> "Slots":
        """Return the slots that either element holds."""
        if self.held is None or other.held is None:
            return Slots(None)
        return Slots(self.held | other.held)

    def meet(self, other: "Slots") -> "Slots":
        """Return the slots that both elements hold."""
        if self.held is None:
            return other
        if other.held is None:
            return self
        return Slots(self.held & other.held)


EMPTY = Slots.bottom()


def slot_key(index: ir.SSAValue) -> Hashable:
    """Return the key of the slot at `index`: its constant, or the SSA value itself."""
    constant = constant_int(index)
    return index if constant is None else constant


@dataclass
class BorrowCheck(Forward[Slots]):
    """Record a validation error for each register that breaks the borrow rule."""

    keys = (BORROW_KEY, "absint")
    lattice = Slots

    def method_self(self, method: ir.Method) -> Slots:
        """Return the top element for the method object."""
        return Slots.top()

    def error(self, node: ir.Statement, message: str) -> None:
        """Record the validation error `message` at `node`."""
        self.add_validation_error(node, ir.ValidationError(node, message))

    def expect_empty(
        self,
        frame: ForwardFrame[Slots],
        node: ir.Statement,
        values: Iterable[ir.SSAValue],
    ) -> None:
        """Record an error at `node` for each register of `values` with a borrowed slot."""
        for value in values:
            if not value.type.is_subseteq(RegisterType):
                continue
            held = frame.get(value).held
            if held:
                self.error(node, f"'{node.name}' takes a register with borrowed slots")

    def results(self, node: ir.Statement) -> tuple[Slots, ...]:
        """Return no borrowed slot for each register result, and top for the others."""
        return tuple(
            EMPTY if r.type.is_subseteq(RegisterType) else Slots.top()
            for r in node.results
        )

    def eval_fallback(
        self, frame: ForwardFrame[Slots], node: ir.Statement
    ) -> tuple[Slots, ...]:
        """Check that each register operand holds no borrowed slot."""
        self.expect_empty(frame, node, node.args)
        return self.results(node)


@stmts.dialect.register(key=BORROW_KEY)
class _Borrows(interp.MethodTable):
    """Track the borrowed slots through `Borrow` and `Release`."""

    @interp.impl(stmts.Borrow)
    def borrow(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: stmts.Borrow
    ) -> tuple[Slots, ...]:
        """Add the index to the borrowed slots of the register."""
        held = frame.get(stmt.register).held
        key = slot_key(stmt.index)
        if held is None:
            return (Slots.top(), Slots.top())
        if key in held:
            check.error(stmt, "'borrow' takes a slot that is already borrowed")
        return (Slots(held | {key}), Slots.top())

    @interp.impl(stmts.Release)
    def release(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: stmts.Release
    ) -> tuple[Slots, ...]:
        """Remove the index from the borrowed slots of the register.

        If the index matches no slot, the method reports an error when every slot
        and the index are constants. Otherwise the slots become unknown.
        """
        held = frame.get(stmt.register).held
        key = slot_key(stmt.index)
        if held is None:
            return (Slots.top(),)
        if key in held:
            return (Slots(held - {key}),)
        if isinstance(key, int) and all(isinstance(k, int) for k in held):
            check.error(stmt, "'release' fills a slot that holds no borrow")
            return (Slots(held),)
        return (Slots.top(),)


@scf.dialect.register(key=BORROW_KEY)
class _BorrowScf(interp.MethodTable):
    """Check that loops and branches carry registers without borrowed slots."""

    @interp.impl(scf.For)
    @interp.impl(scf.IfElse)
    def structured(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: ir.Statement
    ) -> tuple[Slots, ...]:
        """Check the operands, then run each body once with empty registers."""
        check.expect_empty(frame, stmt, stmt.args)
        for region in stmt.regions:
            if not region.blocks:
                continue
            args = [
                EMPTY if a.type.is_subseteq(RegisterType) else Slots.top()
                for a in region.blocks[0].args
            ]
            with check.new_frame(stmt, has_parent_access=True) as inner:
                check.frame_call_region(inner, stmt, region, *args)
        return check.results(stmt)

    @interp.impl(scf.Yield)
    def yield_(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: scf.Yield
    ) -> interp.YieldValue[Slots]:
        """Check that each yielded register holds no borrowed slot."""
        check.expect_empty(frame, stmt, stmt.values)
        return interp.YieldValue(tuple(Slots.top() for _ in stmt.values))


@func.dialect.register(key=BORROW_KEY)
class _BorrowFunc(interp.MethodTable):
    """Check that returns and calls take registers without borrowed slots."""

    @interp.impl(func.Return)
    def return_(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: func.Return
    ) -> interp.ReturnValue[Slots]:
        """Check the returned value."""
        check.expect_empty(frame, stmt, stmt.args)
        return interp.ReturnValue(Slots.top())

    @interp.impl(func.Invoke)
    def invoke(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: func.Invoke
    ) -> tuple[Slots, ...]:
        """Check the arguments, then run the callee with empty registers."""
        check.expect_empty(frame, stmt, stmt.args)
        params = stmt.callee.callable_region.blocks[0].args[1:]
        args = [
            EMPTY if p.type.is_subseteq(RegisterType) else Slots.top() for p in params
        ]
        check.call(stmt.callee.code, Slots.top(), *args)
        return check.results(stmt)


class BorrowValidation(ValidationPass[ForwardFrame[Slots]]):
    """Check that every borrowed register slot comes back before the register leaves."""

    def name(self) -> str:
        """Return the pass name that validation reports show."""
        return "squin wire borrows"

    def run(
        self, method: ir.Method
    ) -> tuple[ForwardFrame[Slots], list[ir.ValidationError]]:
        """Run `BorrowCheck` with empty register parameters."""
        analysis = BorrowCheck(method.dialects)
        params = method.callable_region.blocks[0].args[1:]
        args = [
            EMPTY if p.type.is_subseteq(RegisterType) else Slots.top() for p in params
        ]
        frame, _ = analysis.run(method, *args)
        return frame, analysis.get_validation_errors()


class StructureValidation(ValidationPass[None]):
    """Run kirin's own structure and type checks on a wire squin method."""

    def name(self) -> str:
        """Return the pass name that validation reports show."""
        return "squin wire structure"

    def run(self, method: ir.Method) -> tuple[None, list[ir.ValidationError]]:
        """Return the error of `verify` or `verify_type`, if one of them fails.

        A squin statement on qubits that keeps reference semantics comes from an
        inlined library kernel. Kirin's type check rejects such a statement on a
        list of unknown length, although the squin kernel runs. So a method with
        such statements gets `verify` only.
        """
        kept = any(type(node) in TWINS for node in method.callable_region.walk())
        try:
            method.verify()
            if not kept:
                method.verify_type()
        except ir.ValidationError as error:
            return None, [error]
        return None, []


@dataclass
class WireValidation:
    """Run the structure, linearity, borrow and reference checks as one kirin suite."""

    def check(self, method: ir.Method) -> None:
        """Raise a `ValidationErrorGroup` that lists every broken rule of `method`."""
        suite = ValidationSuite(
            [
                StructureValidation,
                LinearityValidation,
                BorrowValidation,
                ReferenceValidation,
            ]
        )
        suite.validate(method).raise_if_invalid()
