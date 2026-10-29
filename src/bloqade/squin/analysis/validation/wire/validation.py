"""Check wire squin with the kirin validation suite `WireValidation`.

`StructureValidation` runs kirin's
own `verify` and `verify_type`. `LinearityValidation` checks the use rule below.
`BorrowValidation` checks that each `Borrow` has a matching `Release`.
`PositionValidation` checks the order of wires across control flow and calls.

The use rule:

A wire or a register must have exactly one use. The use must sit in the region
that defines the value, with one exception. A branch of kirin's `scf.if` has no
carried block arguments, so each branch reads the outer wire itself. Each branch
must then use the outer wire exactly once. A loop body must get every wire as a
block argument, because the body runs more than once.

`LinearityCheck` is a kirin forward analysis.
Its dispatch hook checks the results of every statement.
Its method tables check tuple fields, calls, returns and control-flow boundaries.
The call rule runs each callee through Kirin's interpreter machinery.

The borrow rule: a register value holds a set of borrowed slots. `Borrow` adds
its index, and `Release` removes it. A slot must not be borrowed twice. A register
must hold no borrowed slot when a statement frees it, yields it, returns it or
passes it to a call. `BorrowCheck` is a kirin forward analysis over the lattice
`Slots`.
"""

from dataclasses import field, replace, dataclass
from collections.abc import Iterable

from kirin import ir, types, interp
from kirin.lattice import EmptyLattice, BoundedLattice
from kirin.analysis import Forward
from kirin.dialects import py, scf, func
from kirin.validation import ValidationPass, ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from kirin.analysis.forward import ForwardFrame

from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Slice, Affine, IndexAnalysis, same
from bloqade.squin.wire.types import (
    RegisterType,
    is_linear,
    linear_paths,
    contains_linear,
)
from bloqade.squin.wire.errors import UnresolvedBorrowSeparation
from bloqade.squin.analysis.validation.wire.position import PositionValidation

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

    def error(self, node: ir.IRNode, message: str) -> None:
        """Record an ownership error at the operation that defines the value."""
        self.add_validation_error(node, ir.ValidationError(node, message))

    def check(self, values: Iterable[ir.SSAValue], node: ir.IRNode) -> None:
        """Check one consuming use per quantum tuple field along each execution path."""
        for value in values:
            try:
                paths = linear_paths(value.type)
            except ValueError as error:
                self.error(node, str(error))
                continue
            region = _region_of(value)
            if region is None:
                continue
            for path in paths:
                users = []
                for use in value.uses:
                    user = use.stmt
                    if path and isinstance(user, py.GetItem) and use.index == 0:
                        index = constant_int(user.index)
                        assert isinstance(value.type, types.Generic)
                        size = len(value.type.vars)
                        if index is None or not -size <= index < size:
                            self.error(
                                user,
                                "A quantum tuple requires a constant index within its bounds.",
                            )
                            continue
                        if index % size != path[0]:
                            continue
                        if user.result.type != value.type.vars[index % size]:
                            self.error(
                                user,
                                "Tuple extraction must preserve the quantum field type.",
                            )
                    elif not (
                        user.dialect is stmts.dialect
                        or isinstance(
                            user,
                            (
                                py.tuple.New,
                                func.Invoke,
                                func.Return,
                                scf.For,
                                scf.Yield,
                            ),
                        )
                    ):
                        self.error(
                            user, f"'{user.name}' cannot consume a quantum value."
                        )
                    users.append(user)
                if (reason := misuse(users, region)) is not None:
                    suffix = "" if not path else f" at tuple field {path}"
                    self.error(
                        node,
                        f"the linear value '{value.name or repr(value)}'{suffix} {reason}",
                    )

    def preserve(self, node, actual, declared):
        """Require a boundary type to retain every quantum ownership path."""
        try:
            if linear_paths(actual) != linear_paths(declared):
                self.error(node, "The boundary type must preserve every quantum field.")
        except ValueError as error:
            self.error(node, str(error))

    def frame_eval(self, frame, node):
        """Check results before dispatch so every operation checks ownership."""
        if node.dialect is stmts.dialect and any(
            isinstance(value.type, types.BottomType)
            for value in (*node.args, *node.results)
        ):
            self.error(node, "A wire operation cannot use the impossible type Bottom.")
        self.check(node.results, node)
        return super().frame_eval(frame, node)

    def flat(self, node, values):
        """Require control-flow state to carry each quantum value separately."""
        if any(
            contains_linear(value.type) and not is_linear(value.type)
            for value in values
        ):
            self.error(node, "Control flow requires separate wire and register values.")

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
            params = trait.get_callable_region(node).blocks[0].args[1:]
            for value in params:
                if contains_linear(value.type) and not is_linear(value.type):
                    self.error(
                        node,
                        "Quantum function parameters must be separate wires or registers.",
                    )
            self.check(params, node)
            if isinstance(node, func.Function):
                output = node.signature.output
                if isinstance(output, types.Generic) and output.is_subseteq(
                    types.Tuple
                ):
                    if any(
                        contains_linear(kind) and not is_linear(kind)
                        for kind in output.vars
                    ):
                        self.error(
                            node,
                            "Quantum function outputs require a flat tuple of wires and registers.",
                        )
        return super().frame_call(frame, node, *args, **kwargs)

    def eval_fallback(
        self, frame: ForwardFrame[EmptyLattice], node: ir.Statement
    ) -> tuple[EmptyLattice, ...]:
        """Check the results of a statement without a rule."""
        return tuple(EmptyLattice.top() for _ in node.results)


@py.tuple.dialect.register(key=KEY)
class _TupleLinearity(interp.MethodTable):
    """Check ownership types when an operation constructs a tuple."""

    @interp.impl(py.tuple.New)
    def new(self, check: LinearityCheck, frame, stmt: py.tuple.New):
        """Require each tuple field to retain its constructing operand's type."""
        expected = types.Generic(tuple, *(value.type for value in stmt.args))
        if contains_linear(expected) and stmt.result.type != expected:
            check.error(
                stmt, "Tuple construction must preserve every quantum field type."
            )
        return (EmptyLattice.top(),)


@func.dialect.register(key=KEY)
class _FunctionLinearity(interp.MethodTable):
    """Preserve quantum ownership through declared call and return types."""

    @interp.impl(func.Invoke)
    def invoke(self, check: LinearityCheck, frame, stmt: func.Invoke):
        """Check call types and validate the callee with Kirin's call machinery."""
        parameters = stmt.callee.callable_region.blocks[0].args[1:]
        for argument, parameter in zip(stmt.inputs, parameters):
            check.preserve(stmt, argument.type, parameter.type)
        expected = stmt.callee.return_type
        if contains_linear(expected) and stmt.result.type != expected:
            check.error(
                stmt, "A call result must preserve its declared quantum field types."
            )
        check.call(
            stmt.callee.code,
            EmptyLattice.top(),
            *(EmptyLattice.top() for _ in parameters),
        )
        return (EmptyLattice.top(),)

    @interp.impl(func.Return)
    def return_(self, check: LinearityCheck, frame, stmt: func.Return):
        """Require the function result type to retain its returned quantum fields."""
        owner = stmt.parent_stmt
        while owner is not None and not isinstance(owner, func.Function):
            owner = owner.parent_stmt
        if isinstance(owner, func.Function):
            check.preserve(stmt, stmt.value.type, owner.signature.output)
        return interp.ReturnValue(EmptyLattice.top())


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
        """Check the state types, then run each body once."""
        check.flat(stmt, stmt.results)
        if isinstance(stmt, scf.For):
            check.flat(stmt, stmt.initializers)
            for value, argument in zip(stmt.initializers, stmt.body.blocks[0].args[1:]):
                check.preserve(stmt, value.type, argument.type)
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
        """Check yielded ownership against each declared region result."""
        check.flat(stmt, stmt.values)
        if stmt.parent_stmt is not None:
            for value, result in zip(stmt.values, stmt.parent_stmt.results):
                check.preserve(stmt, value.type, result.type)
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
class Hole:
    """Identify empty elements whose qubits have transferred ownership out of the register."""

    selection: Affine | Slice


@dataclass(frozen=True)
class Slots(BoundedLattice["Slots"]):
    """Hold the borrowed slots of one register value.

    `held` holds one affine index for each borrowed slot. Equal expressions, such
    as `i` and `i + 0`, have the same key. If `held` is None, the
    analysis cannot name the slots. That element is the top of the lattice.
    """

    held: frozenset[Affine | Slice | Hole] | None
    length: Affine | None = None
    unreachable: bool = False

    @classmethod
    def top(cls) -> "Slots":
        """Return the element whose slots the analysis cannot name."""
        return cls(None)

    @classmethod
    def bottom(cls) -> "Slots":
        """Return the unreachable state that precedes every concrete register state."""
        return cls(frozenset(), unreachable=True)

    def is_subseteq(self, other: "Slots") -> bool:
        """Return True if `other` holds every slot of `self`."""
        if self.unreachable:
            return True
        if other.unreachable:
            return False
        if other.held is None:
            return True
        return (
            self.held is not None
            and self.held <= other.held
            and (other.length is None or self.length == other.length)
        )

    def join(self, other: "Slots") -> "Slots":
        """Return the slots that either element holds."""
        if self.unreachable:
            return other
        if other.unreachable:
            return self
        if self.held is None or other.held is None:
            return Slots(None)
        return Slots(
            self.held | other.held, self.length if self.length == other.length else None
        )

    def meet(self, other: "Slots") -> "Slots":
        """Intersect hole information and retain compatible length constraints."""
        if self.unreachable or other.unreachable:
            return Slots.bottom()
        if self.held is None:
            return other
        if other.held is None:
            return self
        if (
            self.length is not None
            and other.length is not None
            and self.length != other.length
        ):
            return Slots.bottom()
        return Slots(self.held & other.held, self.length or other.length)


EMPTY = Slots(frozenset())


@dataclass
class BorrowCheck(Forward[Slots]):
    """Record a validation error for each register that breaks the borrow rule."""

    keys = (BORROW_KEY, "absint")
    lattice = Slots
    indices: IndexAnalysis = field(default_factory=IndexAnalysis, init=False)

    def initialize(self) -> "BorrowCheck":
        """Clear index facts before checking a method that can have changed."""
        self.indices = IndexAnalysis()
        return super().initialize()

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
            if held is None:
                self.error(
                    node, f"'{node.name}' takes a register with unknown borrowed slots"
                )
            elif held:
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

    @interp.impl(stmts.Take)
    @interp.impl(stmts.Borrow)
    def borrow(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: stmts.Borrow
    ) -> tuple[Slots, ...]:
        """Add the index to the borrowed slots of the register."""
        held = frame.get(stmt.register).held
        key = (
            Hole(check.indices.of(stmt.index))
            if isinstance(stmt, stmts.Take)
            else check.indices.of(stmt.index)
        )
        if held is None:
            return (Slots.top(), Slots.top())
        if key in held:
            check.error(stmt, "'borrow' takes a slot that is already borrowed")
        elif any(not separated(key, previous) for previous in held):
            check.add_validation_error(
                stmt,
                UnresolvedBorrowSeparation(
                    stmt,
                    "'borrow' requires indices that provably differ from every outstanding borrow",
                ),
            )
        return (Slots(held | {key}, frame.get(stmt.register).length), Slots.top())

    @interp.impl(stmts.Put)
    @interp.impl(stmts.Release)
    def release(
        self, check: BorrowCheck, frame: ForwardFrame[Slots], stmt: stmts.Release
    ) -> tuple[Slots, ...]:
        """Remove a slot only when its index matches a recorded borrow."""
        held = frame.get(stmt.register).held
        key = (
            Hole(check.indices.of(stmt.index))
            if isinstance(stmt, stmts.Put)
            else check.indices.of(stmt.index)
        )
        if held is not None and key in held:
            return (Slots(held - {key}, frame.get(stmt.register).length),)
        check.error(stmt, "'release' requires a matching borrowed index")
        return (Slots.top(),)

    @interp.impl(stmts.AllocRegister)
    def alloc(self, check: BorrowCheck, frame, stmt: stmts.AllocRegister):
        """Record the length of a newly allocated complete register."""
        return (Slots(frozenset(), check.indices.of(stmt.size)),)

    @interp.impl(stmts.PackRegister)
    def pack(self, check: BorrowCheck, frame, stmt: stmts.PackRegister):
        """Record the number of consumed qubits in the new register."""
        return (Slots(frozenset(), Affine(len(stmt.wires))),)

    @interp.impl(stmts.Length)
    def length(self, check: BorrowCheck, frame, stmt: stmts.Length):
        """Preserve holes and record a known length for later slice checks."""
        state = frame.get(stmt.register)
        if state.length is not None:
            check.indices.cache[stmt.result] = state.length
        return state, Slots.top()

    @interp.impl(stmts.SplitRegister)
    def split(self, check: BorrowCheck, frame, stmt: stmts.SplitRegister):
        """Require a complete register and compute the two result lengths."""
        check.expect_empty(frame, stmt, stmt.args)
        at = check.indices.of(stmt.index)
        length = frame.get(stmt.register).length
        return Slots(frozenset(), at), Slots(
            frozenset(), None if length is None else length - at
        )

    @interp.impl(stmts.JoinRegister)
    def join(self, check: BorrowCheck, frame, stmt: stmts.JoinRegister):
        """Require complete registers and add their known lengths."""
        check.expect_empty(frame, stmt, stmt.args)
        a, b = frame.get(stmt.first).length, frame.get(stmt.second).length
        return (Slots(frozenset(), None if a is None or b is None else a + b),)

    @interp.impl(stmts.TakeSlice)
    def take_slice(self, check: BorrowCheck, frame, stmt: stmts.TakeSlice):
        """Record a contiguous hole and the extracted register's length."""
        state = frame.get(stmt.register)
        size = check.indices.of(stmt.length)
        key = Hole(Slice(check.indices.of(stmt.start), 1, size))
        if state.held is None:
            return Slots.top(), Slots(frozenset(), size)
        if any(not separated(key, held) for held in state.held):
            check.error(
                stmt, "A slice extraction overlaps an outstanding register hole."
            )
        return Slots(state.held | {key}, state.length), Slots(frozenset(), size)

    @interp.impl(stmts.PutSlice)
    def put_slice(self, check: BorrowCheck, frame, stmt: stmts.PutSlice):
        """Require a complete slice that exactly fills one recorded hole."""
        check.expect_empty(frame, stmt, (stmt.slice_register,))
        state = frame.get(stmt.register)
        size = frame.get(stmt.slice_register).length
        key = Hole(Slice(check.indices.of(stmt.start), 1, size))
        if state.held is not None and size is not None and key in state.held:
            return (Slots(state.held - {key}, state.length),)
        check.error(stmt, "A slice insertion must match an empty range and its length.")
        return (Slots.top(),)

    @interp.impl(stmts.DiscardRegister)
    def discard(self, check: BorrowCheck, frame, stmt: stmts.DiscardRegister):
        """Permit moved-out elements and require every outstanding borrow to return."""
        held = frame.get(stmt.register).held
        if held is None or any(not isinstance(hole, Hole) for hole in held):
            check.error(
                stmt, "A discarded register must restore every borrowed element."
            )
        return ()


def separated(a: Affine | Slice | Hole, b: Affine | Slice | Hole) -> bool:
    """Prove that two element or contiguous slice holes do not overlap."""
    a = a.selection if isinstance(a, Hole) else a
    b = b.selection if isinstance(b, Hole) else b
    if isinstance(a, Affine) and isinstance(b, Affine):
        return same(a, b) is False
    left = a if isinstance(a, Slice) else Slice(a, 1, Affine(1))
    right = b if isinstance(b, Slice) else Slice(b, 1, Affine(1))
    if left.length is None or right.length is None:
        return False
    gap_a = (left.start + left.length - right.start).constant
    gap_b = (right.start + right.length - left.start).constant
    return (gap_a is not None and gap_a <= 0) or (gap_b is not None and gap_b <= 0)


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
        """Return the error of `verify` or `verify_type`, if either check fails."""
        try:
            method.code.verify()
            method.code.verify_type()
        except ir.ValidationError as error:
            error.source = None if error.source is None else replace(error.source)
            return None, [error]
        return None, []


@dataclass
class WireValidation:
    """Run the wire structure, linearity, borrow and position checks."""

    def check(self, method: ir.Method) -> None:
        """Raise a `ValidationErrorGroup` that lists every broken rule of `method`."""
        suite = ValidationSuite(
            [
                StructureValidation,
                LinearityValidation,
                BorrowValidation,
                PositionValidation,
            ]
        )
        suite.validate(method).raise_if_invalid()
