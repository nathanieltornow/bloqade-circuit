"""Check that every borrowed register element returns to its register.

A register value holds a set of unavailable slots. `Borrow` adds its index, and
`Release` removes it. `Take` and `TakeSlice` add a moved element or range, and
`Put` and `PutSlice` remove it. A slot must not be unavailable twice. A register
must hold no unavailable slot when a statement frees it, yields it, returns it or
passes it to a call. `DiscardRegister` permits moved slots and rejects borrowed
slots. `BorrowCheck` is a kirin forward analysis over the lattice `Slots`.
"""

from dataclasses import field, dataclass
from collections.abc import Iterable

from kirin import ir, interp
from kirin.lattice import BoundedLattice
from kirin.dialects import scf, func
from kirin.validation import ValidationPass
from kirin.analysis.forward import ForwardFrame

from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Slice, Affine, IndexAnalysis, same
from bloqade.squin.wire.types import (
    RegisterType,
)
from bloqade.squin.wire.errors import UnresolvedBorrowSeparation
from bloqade.squin.analysis.validation.wire.calls import CachedCalls

BORROW_KEY = "squin.wire.borrows"
"""Name the registry key of the method tables of `BorrowCheck`."""


@dataclass(frozen=True)
class Moved:
    """Name register elements whose qubits `Take` or `TakeSlice` moved out for good."""

    selection: Affine | Slice
    """The moved element index or the moved contiguous range."""


@dataclass(frozen=True)
class Slots(BoundedLattice["Slots"]):
    """Hold the unavailable slots of one register value.

    `held` holds the affine index of each borrowed slot and a `Moved` entry for each
    moved element or range. Equal expressions, such as `i` and `i + 0`, have the
    same key. If `held` is None, the analysis cannot name the slots. That element
    is the top of the lattice.
    """

    held: frozenset[Affine | Moved] | None
    length: Affine | None = None
    """The number of elements of the register, or None if it is unknown."""
    unreachable: bool = False
    """Whether no execution reaches this register value. Only the bottom sets it."""

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
        """Return the slots that both elements hold, with a compatible length."""
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
class BorrowCheck(CachedCalls[Slots]):
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
            Moved(check.indices.of(stmt.index))
            if isinstance(stmt, stmts.Take)
            else check.indices.of(stmt.index)
        )
        if held is None:
            return (Slots.top(), Slots.top())
        if key in held:
            check.error(stmt, "'borrow' takes a slot that is already borrowed")
        elif any(not disjoint(key, previous) for previous in held):
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
            Moved(check.indices.of(stmt.index))
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
        """Keep the unavailable slots and record a known length for slice checks."""
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
        """Record a moved contiguous range and the extracted register's length."""
        state = frame.get(stmt.register)
        size = check.indices.of(stmt.length)
        key = Moved(Slice(check.indices.of(stmt.start), 1, size))
        if state.held is None:
            return Slots.top(), Slots(frozenset(), size)
        if any(not disjoint(key, held) for held in state.held):
            check.error(
                stmt, "A slice extraction overlaps an unavailable register element."
            )
        return Slots(state.held | {key}, state.length), Slots(frozenset(), size)

    @interp.impl(stmts.PutSlice)
    def put_slice(self, check: BorrowCheck, frame, stmt: stmts.PutSlice):
        """Require a complete slice that exactly fills one moved range."""
        check.expect_empty(frame, stmt, (stmt.slice_register,))
        state = frame.get(stmt.register)
        size = frame.get(stmt.slice_register).length
        key = Moved(Slice(check.indices.of(stmt.start), 1, size))
        if state.held is not None and size is not None and key in state.held:
            return (Slots(state.held - {key}, state.length),)
        check.error(stmt, "A slice insertion must match an empty range and its length.")
        return (Slots.top(),)

    @interp.impl(stmts.DiscardRegister)
    def discard(self, check: BorrowCheck, frame, stmt: stmts.DiscardRegister):
        """Permit moved-out elements and require every outstanding borrow to return."""
        held = frame.get(stmt.register).held
        if held is None or any(not isinstance(slot, Moved) for slot in held):
            check.error(
                stmt, "A discarded register must restore every borrowed element."
            )
        return ()


def disjoint(a: Affine | Moved, b: Affine | Moved) -> bool:
    """Prove that two unavailable slots share no register element."""
    first = a.selection if isinstance(a, Moved) else a
    second = b.selection if isinstance(b, Moved) else b
    if isinstance(first, Affine) and isinstance(second, Affine):
        return same(first, second) is False
    left = first if isinstance(first, Slice) else Slice(first, 1, Affine(1))
    right = second if isinstance(second, Slice) else Slice(second, 1, Affine(1))
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
