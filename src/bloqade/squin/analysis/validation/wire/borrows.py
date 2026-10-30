"""Check that every borrowed register element returns to its register.

A register value holds a set of unavailable slots. `Borrow` adds its index, and
`Release` removes it. `Take` and `TakeSlice` add a moved element or range, and
`Put` and `PutSlice` remove it. A slot must not be unavailable twice. A register
must hold no unavailable slot when a statement frees it, yields it, returns it or
passes it to a call. `DiscardRegister` permits moved slots and rejects borrowed
slots. `BorrowCheck` is a kirin forward analysis over the lattice `Slots`.
"""

from __future__ import annotations

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
    """Hold what the borrow check knows about the unavailable slots of one register.

    `UnknownSlots` is the top, where the check cannot name the slots.
    `UnreachableSlots` is the bottom, for a register that no execution reaches.
    `KnownSlots` lists the slots.
    """

    @classmethod
    def top(cls) -> Slots:
        """Return the state whose slots the check cannot name."""
        return UnknownSlots()

    @classmethod
    def bottom(cls) -> Slots:
        """Return the state of a register that no execution reaches."""
        return UnreachableSlots()

    def is_subseteq(self, other: Slots) -> bool:
        """Return True if `other` holds every slot of `self`."""
        if isinstance(self, UnreachableSlots) or isinstance(other, UnknownSlots):
            return True
        if not (isinstance(self, KnownSlots) and isinstance(other, KnownSlots)):
            return False
        return self.unavailable <= other.unavailable and (
            other.length is None or self.length == other.length
        )

    def join(self, other: Slots) -> Slots:
        """Return the slots that either state holds."""
        if isinstance(self, UnreachableSlots):
            return other
        if isinstance(other, UnreachableSlots):
            return self
        if not (isinstance(self, KnownSlots) and isinstance(other, KnownSlots)):
            return UnknownSlots()
        length = self.length if self.length == other.length else None
        return KnownSlots(self.unavailable | other.unavailable, length)

    def meet(self, other: Slots) -> Slots:
        """Return the slots that both states hold, with a compatible length."""
        if isinstance(self, UnreachableSlots) or isinstance(other, UnreachableSlots):
            return UnreachableSlots()
        if not isinstance(self, KnownSlots):
            return other
        if not isinstance(other, KnownSlots):
            return self
        if (
            self.length is not None
            and other.length is not None
            and self.length != other.length
        ):
            return UnreachableSlots()
        length = self.length if self.length is not None else other.length
        return KnownSlots(self.unavailable & other.unavailable, length)


@dataclass(frozen=True)
class UnknownSlots(Slots):
    """Name the state of a register whose unavailable slots the check cannot name."""


@dataclass(frozen=True)
class UnreachableSlots(Slots):
    """Name the state of a register that no execution reaches."""


@dataclass(frozen=True)
class KnownSlots(Slots):
    """List the unavailable slots of a register."""

    unavailable: frozenset[Affine | Moved] = frozenset()
    """The affine index of each borrowed slot and a `Moved` entry for each moved part.

    Equal expressions, such as `i` and `i + 0`, have the same key.
    """
    length: Affine | None = None
    """The number of elements of the register, or None if it is unknown."""


EMPTY = KnownSlots()
"""The state of a complete register."""


def known_length(state: Slots) -> Affine | None:
    """Return the register length that `state` records, or None."""
    return state.length if isinstance(state, KnownSlots) else None


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
        """Record an error at `node` for each register of `values` with an unavailable slot."""
        for value in values:
            if not value.type.is_subseteq(RegisterType):
                continue
            state = frame.get(value)
            if isinstance(state, UnknownSlots):
                self.error(
                    node, f"'{node.name}' takes a register with unknown borrowed slots"
                )
            elif isinstance(state, KnownSlots) and state.unavailable:
                self.error(node, f"'{node.name}' takes a register with borrowed slots")

    def results(self, node: ir.Statement) -> tuple[Slots, ...]:
        """Return no unavailable slot for each register result, and top for the others."""
        return tuple(
            EMPTY if r.type.is_subseteq(RegisterType) else Slots.top()
            for r in node.results
        )

    def eval_fallback(
        self, frame: ForwardFrame[Slots], node: ir.Statement
    ) -> tuple[Slots, ...]:
        """Check that each register operand holds no unavailable slot."""
        self.expect_empty(frame, node, node.args)
        return self.results(node)


@stmts.dialect.register(key=BORROW_KEY)
class _Borrows(interp.MethodTable):
    """Track the unavailable slots through borrows, moves and register operations."""

    @interp.impl(stmts.Take)
    @interp.impl(stmts.Borrow)
    def borrow(
        self,
        check: BorrowCheck,
        frame: ForwardFrame[Slots],
        stmt: stmts.Borrow | stmts.Take,
    ) -> tuple[Slots, ...]:
        """Add the borrowed index or the moved element to the unavailable slots."""
        state = frame.get(stmt.register)
        key = (
            Moved(check.indices.of(stmt.index))
            if isinstance(stmt, stmts.Take)
            else check.indices.of(stmt.index)
        )
        if not isinstance(state, KnownSlots):
            return (state, state)
        if key in state.unavailable:
            check.error(stmt, "'borrow' takes a slot that is already borrowed")
        elif any(not disjoint(key, previous) for previous in state.unavailable):
            check.add_validation_error(
                stmt,
                UnresolvedBorrowSeparation(
                    stmt,
                    "'borrow' requires indices that provably differ from every outstanding borrow",
                ),
            )
        return (
            KnownSlots(state.unavailable | {key}, state.length),
            Slots.top(),
        )

    @interp.impl(stmts.Put)
    @interp.impl(stmts.Release)
    def release(
        self,
        check: BorrowCheck,
        frame: ForwardFrame[Slots],
        stmt: stmts.Release | stmts.Put,
    ) -> tuple[Slots, ...]:
        """Remove the slot that matches the released index or the filled element."""
        state = frame.get(stmt.register)
        key = (
            Moved(check.indices.of(stmt.index))
            if isinstance(stmt, stmts.Put)
            else check.indices.of(stmt.index)
        )
        if isinstance(state, UnreachableSlots):
            return (state,)
        if isinstance(state, KnownSlots) and key in state.unavailable:
            return (KnownSlots(state.unavailable - {key}, state.length),)
        check.error(stmt, "'release' requires a matching borrowed index")
        return (Slots.top(),)

    @interp.impl(stmts.AllocRegister)
    def alloc(self, check: BorrowCheck, frame, stmt: stmts.AllocRegister):
        """Record the length of a newly allocated complete register."""
        return (KnownSlots(length=check.indices.of(stmt.size)),)

    @interp.impl(stmts.PackRegister)
    def pack(self, check: BorrowCheck, frame, stmt: stmts.PackRegister):
        """Record the number of consumed qubits in the new register."""
        return (KnownSlots(length=Affine(len(stmt.wires))),)

    @interp.impl(stmts.Length)
    def length(self, check: BorrowCheck, frame, stmt: stmts.Length):
        """Keep the unavailable slots and record a known length for slice checks."""
        state = frame.get(stmt.register)
        if (length := known_length(state)) is not None:
            check.indices.cache[stmt.result] = length
        return state, Slots.top()

    @interp.impl(stmts.SplitRegister)
    def split(self, check: BorrowCheck, frame, stmt: stmts.SplitRegister):
        """Require a complete register and compute the two result lengths."""
        check.expect_empty(frame, stmt, stmt.args)
        at = check.indices.of(stmt.index)
        length = known_length(frame.get(stmt.register))
        rest = None if length is None else length - at
        return KnownSlots(length=at), KnownSlots(length=rest)

    @interp.impl(stmts.JoinRegister)
    def join(self, check: BorrowCheck, frame, stmt: stmts.JoinRegister):
        """Require complete registers and add their known lengths."""
        check.expect_empty(frame, stmt, stmt.args)
        a = known_length(frame.get(stmt.first))
        b = known_length(frame.get(stmt.second))
        return (KnownSlots(length=None if a is None or b is None else a + b),)

    @interp.impl(stmts.TakeSlice)
    def take_slice(self, check: BorrowCheck, frame, stmt: stmts.TakeSlice):
        """Record a moved contiguous range and the extracted register's length."""
        state = frame.get(stmt.register)
        size = check.indices.of(stmt.length)
        key = Moved(Slice(check.indices.of(stmt.start), 1, size))
        if isinstance(state, UnreachableSlots):
            return state, state
        if not isinstance(state, KnownSlots):
            return Slots.top(), KnownSlots(length=size)
        if any(not disjoint(key, slot) for slot in state.unavailable):
            check.error(
                stmt, "A slice extraction overlaps an outstanding register hole."
            )
        unavailable = state.unavailable | {key}
        return KnownSlots(unavailable, state.length), KnownSlots(length=size)

    @interp.impl(stmts.PutSlice)
    def put_slice(self, check: BorrowCheck, frame, stmt: stmts.PutSlice):
        """Require a complete slice that exactly fills one moved range."""
        check.expect_empty(frame, stmt, (stmt.slice_register,))
        state = frame.get(stmt.register)
        size = known_length(frame.get(stmt.slice_register))
        key = Moved(Slice(check.indices.of(stmt.start), 1, size))
        if isinstance(state, UnreachableSlots):
            return (state,)
        if (
            isinstance(state, KnownSlots)
            and size is not None
            and key in state.unavailable
        ):
            return (KnownSlots(state.unavailable - {key}, state.length),)
        check.error(stmt, "A slice insertion must match an empty range and its length.")
        return (Slots.top(),)

    @interp.impl(stmts.DiscardRegister)
    def discard(self, check: BorrowCheck, frame, stmt: stmts.DiscardRegister):
        """Permit moved-out elements and require every outstanding borrow to return."""
        state = frame.get(stmt.register)
        if isinstance(state, UnknownSlots) or (
            isinstance(state, KnownSlots)
            and any(not isinstance(slot, Moved) for slot in state.unavailable)
        ):
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
        """Check that each yielded register holds no unavailable slot."""
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
