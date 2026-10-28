"""Check that control flow and borrowing calls preserve the order of their qubits.

`Threads` maps each linear operand to the result that continues it. The analysis
follows these pairs through gates, tuples and register accesses. Each loop must
yield its carried qubits in input order. Both branches must yield the same qubit
at each linear result position. Each function must return its borrowed parameters
as a suffix in parameter order.
"""

from dataclasses import dataclass

from kirin import ir, types, interp
from kirin.dialects import scf, func
from kirin.validation import ValidationPass
from kirin.analysis.forward import ForwardFrame

from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import WireType, is_linear
from bloqade.squin.wire.traits import Threads
from bloqade.analysis.reference import (
    KEY as REFERENCE_KEY,
    UNTRACKED,
    Ref,
    Place,
    Whole,
    Unknown,
    Register,
    Returned,
    Positions,
    argument_of,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis
from bloqade.analysis.reference.queries import same_place

KEY = "squin.wire.positions"


@dataclass
class PositionCheck(QubitReferenceAnalysis):
    """Follow qubit origins and record each broken positional contract."""

    keys = (KEY, REFERENCE_KEY, "absint")

    def shape(self, type_: types.TypeAttribute) -> int | None:
        """Track wires and registers together with their temporary references."""
        if is_linear(type_):
            return 0 if type_.is_subseteq(WireType) else 1
        return super().shape(type_)

    def register_length(
        self, root: ir.SSAValue, calls: tuple[Returned, ...]
    ) -> int | None:
        """Read a wire allocation's size or use the reference register's length."""
        if isinstance(root.owner, stmts.AllocRegister):
            size = constant_int(argument_of(root.owner.size, calls))
            return None if size is None else max(0, size)
        return super().register_length(root, calls)

    def error(self, node: ir.Statement, message: str) -> None:
        """Record the validation error at its statement."""
        self.add_validation_error(node, ir.ValidationError(node, message))

    def expect(
        self, node: ir.Statement, actual: Ref, expected: Ref, message: str
    ) -> None:
        """Require both origins to name the same qubit or register."""
        if not (
            isinstance(actual, Place)
            and isinstance(expected, Place)
            and same_place(actual, expected)
        ):
            self.error(node, message)

    def eval_fallback(
        self, frame: ForwardFrame[Ref], node: ir.Statement
    ) -> tuple[Ref, ...]:
        """Continue each linear operand through the statement's `Threads` trait."""
        results = list(super().eval_fallback(frame, node))
        if threads := node.get_trait(Threads):
            for operand, result in threads.threads(node):
                results[result.index] = frame.get(operand)
        return tuple(results)


@stmts.dialect.register(key=KEY)
class _Wire(interp.MethodTable):
    """Follow allocations, reference accesses and register slots."""

    @interp.impl(stmts.Alloc)
    def alloc(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: stmts.Alloc):
        """Start a new qubit origin at the allocation."""
        return (Whole(stmt.result),)

    @interp.impl(stmts.AllocRegister)
    def register(
        self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: stmts.AllocRegister
    ):
        """Start a new register origin at the allocation."""
        return (Register(stmt.result),)

    @interp.impl(stmts.Reference)
    @interp.impl(stmts.ReferenceRegister)
    def reference(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt):
        """Give a new reference the origin of the wire that initializes it."""
        fill = stmt.next_stmt
        if isinstance(fill, (stmts.Wrap, stmts.WrapRegister)):
            if fill.args[0] is stmt.results[0]:
                return (frame.get(fill.args[1]),)
        return (Unknown("a reference without an initializing wrap"),)

    @interp.impl(stmts.Unwrap)
    @interp.impl(stmts.UnwrapRegister)
    def unwrap(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt):
        """Continue the qubit origin of the reference as a wire."""
        return (frame.get(stmt.args[0]),)

    @interp.impl(stmts.Wrap)
    @interp.impl(stmts.WrapRegister)
    def wrap(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt):
        """Require the wire to continue the qubit origin of the reference."""
        check.expect(
            stmt,
            frame.get(stmt.args[1]),
            frame.get(stmt.args[0]),
            f"'{stmt.name}' must restore the qubits of its reference",
        )
        return ()

    @interp.impl(stmts.Borrow)
    def borrow(
        self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: stmts.Borrow
    ):
        """Continue the register and identify the qubit at the borrowed index."""
        register = frame.get(stmt.register)
        return register, check.index(register, stmt.index)

    @interp.impl(stmts.Release)
    def release(
        self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: stmts.Release
    ):
        """Require the wire to come from the register slot that it fills."""
        register = frame.get(stmt.register)
        check.expect(
            stmt,
            frame.get(stmt.wire),
            check.index(register, stmt.index),
            "'release' must restore the qubit from its register slot",
        )
        return (register,)


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """Check every loop body and both branches for positional continuity."""

    @interp.impl(scf.For)
    def for_(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: scf.For):
        """Run the body until its carried references stop changing."""
        return check.run_loop(
            frame, stmt, stmt.body, frame.get_values(stmt.initializers)
        )

    @interp.impl(scf.Yield)
    def yield_(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: scf.Yield):
        """Check each loop yield against the corresponding body argument."""
        yielded = frame.get_values(stmt.values)
        if isinstance(stmt.parent_stmt, scf.For):
            carried = stmt.parent_stmt.body.blocks[0].args[1:]
            for k, value in enumerate(carried):
                if is_linear(value.type):
                    actual = (
                        yielded[k] if k < len(yielded) else Unknown("a missing yield")
                    )
                    check.expect(
                        stmt,
                        actual,
                        frame.get(value),
                        f"yield position {k} must continue loop input position {k}",
                    )
        return interp.YieldValue(yielded)

    @interp.impl(scf.IfElse)
    def if_else(self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: scf.IfElse):
        """Require both branches to continue the same qubit at each result position."""
        branches = []
        for body in (stmt.then_body, stmt.else_body):
            with check.new_frame(stmt, has_parent_access=True) as inner:
                yielded = check.frame_call_region(inner, stmt, body, UNTRACKED)
            frame.entries.update(inner.entries)
            branches.append(yielded)
        left, right = branches
        if isinstance(left, tuple) and isinstance(right, tuple):
            for k, result in enumerate(stmt.results):
                if is_linear(result.type) and k < len(left) and k < len(right):
                    check.expect(
                        stmt,
                        right[k],
                        left[k],
                        f"both branches must continue the same qubit at yield position {k}",
                    )
            return check.join_results(left, right)
        if isinstance(left, interp.ReturnValue) and isinstance(
            right, interp.ReturnValue
        ):
            return interp.ReturnValue(left.value.join(right.value))
        if isinstance(left, interp.ReturnValue):
            return left
        return right


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """Check the borrowed-parameter suffix of each function return."""

    @interp.impl(func.Return)
    def return_(
        self, check: PositionCheck, frame: ForwardFrame[Ref], stmt: func.Return
    ):
        """Require the return suffix to continue the linear parameters in order."""
        result = frame.get(stmt.value)
        code = frame.code
        while code.parent_stmt is not None and not isinstance(code, func.Function):
            code = code.parent_stmt
        if isinstance(code, func.Function):
            borrowed = [p for p in code.body.blocks[0].args[1:] if is_linear(p.type)]
            outputs = result.refs if isinstance(result, Positions) else (result,)
            if len(outputs) < len(borrowed):
                check.error(
                    stmt, "the return suffix must include every borrowed parameter"
                )
            elif borrowed:
                suffix = outputs[-len(borrowed) :]
                for k, (actual, param) in enumerate(zip(suffix, borrowed, strict=True)):
                    check.expect(
                        stmt,
                        actual,
                        frame.get(param),
                        f"return suffix position {k} must continue borrowed parameter {k}",
                    )
        return interp.ReturnValue(result)


class PositionValidation(ValidationPass[ForwardFrame[Ref]]):
    """Check the qubit order of yields, calls and register releases."""

    def name(self) -> str:
        """Return the name that the validation report shows."""
        return "squin wire positions"

    def run(
        self, method: ir.Method
    ) -> tuple[ForwardFrame[Ref], list[ir.ValidationError]]:
        """Follow qubit origins and return the frame and positional errors."""
        analysis = PositionCheck(method.dialects)
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()
