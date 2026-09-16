"""This module holds the validation pass for jeff code that squin cannot express."""

from dataclasses import dataclass
from collections.abc import Mapping

from kirin import ir, interp
from kirin.lattice import EmptyLattice
from kirin.validation import ValidationPass
from kirin.analysis.forward import ForwardFrame

from bloqade.jeff.gates import canonical
from bloqade.jeff.types import is_bit, is_linear
from bloqade.jeff.errors import JeffToSquinError
from bloqade.jeff.dialects import stmts, kernel as jeff_kernel
from bloqade.jeff.constants import const_int
from bloqade.jeff.signature import declared_outputs
from bloqade.jeff.analysis.reference import (
    Ref,
    Whole,
    WireReferenceAnalysis,
    as_positions,
)

from .base import SHARED_KEY, Check
from .envelope import gate_supported

KEY = "jeff.to_squin"
"""The registry key of the rules that refuse what squin cannot express."""


def handed_back(function: ir.Method, result: Ref) -> tuple[int | None, ...]:
    """Return the output position that hands back each wire parameter of `function`.

    `result` is the reference result of `function`. A parameter has a position if
    exactly one output refers to it as a whole.
    """
    count = len(declared_outputs(function.return_type))
    outputs = as_positions(result, count)
    positions: list[int | None] = []
    for param in function.callable_region.blocks[0].args[1:]:
        if is_linear(param.type):
            found = [p for p, ref in enumerate(outputs) if ref == Whole(param)]
            positions.append(found[0] if len(found) == 1 else None)
    return tuple(positions)


def is_fill(node: stmts.IntArraySet | stmts.FloatArraySet) -> bool:
    """Return True if `node` fills a known array that nothing else reads."""
    if const_int(node.index) is None or len(node.array.uses) != 1:
        return False
    owner = node.array.owner
    if isinstance(owner, (stmts.IntArrayConst, stmts.FloatArrayConst)):
        return True
    if isinstance(owner, (stmts.IntArrayZero, stmts.FloatArrayZero)):
        return const_int(owner.size) is not None
    if isinstance(owner, (stmts.IntArraySet, stmts.FloatArraySet)):
        return is_fill(owner)
    return False


def fill_length(array: ir.SSAValue) -> int | None:
    """Return the length of a known array, or None if the length is unknown."""
    owner = array.owner
    if isinstance(owner, (stmts.IntArrayConst, stmts.FloatArrayConst)):
        return len(owner.values)
    if isinstance(owner, (stmts.IntArrayZero, stmts.FloatArrayZero)):
        return const_int(owner.size)
    if isinstance(owner, (stmts.IntArraySet, stmts.FloatArraySet)):
        return fill_length(owner.array)
    return None


@dataclass
class JeffToSquinAnalysis(Check[EmptyLattice]):
    """An analysis that reports each jeff construct that squin cannot express."""

    keys = (KEY, SHARED_KEY)
    lattice = EmptyLattice

    def refuse(self, node: ir.Statement, message: str) -> None:
        """Record that squin cannot express `node`."""
        self.add_validation_error(node, JeffToSquinError(node, message))


@stmts.gate.dialect.register(key=KEY)
class _Gates(interp.MethodTable):
    """A method table that refuses the gates that squin has no statement for."""

    @interp.impl(stmts.Ppr)
    def ppr(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.Ppr,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse a Pauli-product rotation."""
        check.refuse(stmt, "squin has no Pauli-product rotation statement")
        return check.eval_fallback(frame, stmt)

    @interp.impl(stmts.Gate)
    def gate(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.Gate,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse a gate whose shape squin has no statement for."""
        if not gate_supported(shape := canonical(stmt)):
            check.refuse(
                stmt,
                f"gate '{stmt.gate_name}' has no squin statement. It reads as "
                f"{shape.name} with {len(shape.controls)} controls and power "
                f"{shape.power}.",
            )
        return check.eval_fallback(frame, stmt)


@stmts.classical.dialect.register(key=KEY)
class _Classical(interp.MethodTable):
    """A method table that refuses classical statements that squin cannot express."""

    @interp.impl(stmts.IntDivU)
    @interp.impl(stmts.IntRemU)
    @interp.impl(stmts.IntShr)
    @interp.impl(stmts.IntLtU)
    @interp.impl(stmts.IntLteU)
    @interp.impl(stmts.IntMinU)
    @interp.impl(stmts.IntMaxU)
    def unsigned(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: ir.Statement,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse an unsigned integer operation."""
        check.refuse(
            stmt,
            f"squin integers have no unsigned form for '{stmt.name}'. Squin reads "
            "every integer as a signed value.",
        )
        return check.eval_fallback(frame, stmt)

    @interp.impl(stmts.IntAdd)
    @interp.impl(stmts.IntSub)
    @interp.impl(stmts.IntMul)
    @interp.impl(stmts.IntDivS)
    @interp.impl(stmts.IntPow)
    @interp.impl(stmts.IntMinS)
    @interp.impl(stmts.IntMaxS)
    @interp.impl(stmts.IntRemS)
    @interp.impl(stmts.IntShl)
    def arithmetic(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.IntBinary,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse arithmetic on a bit, which squin types as a measurement result."""
        if is_bit(stmt.lhs) or is_bit(stmt.rhs):
            check.refuse(
                stmt,
                f"'{stmt.name}' on a bit has no squin form. Squin types a "
                "measurement result as a bit.",
            )
        return check.eval_fallback(frame, stmt)

    @interp.impl(stmts.IntArraySet)
    @interp.impl(stmts.FloatArraySet)
    def array_set(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.IntArraySet | stmts.FloatArraySet,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse a set that is not a fill of a known array at a constant index."""
        index, length = const_int(stmt.index), fill_length(stmt.array)
        if not is_fill(stmt):
            check.refuse(
                stmt,
                f"squin has no mutable array for '{stmt.name}'. Squin can express "
                "only a fill of a known array at constant indices.",
            )
        elif index is not None and length is not None and not -length <= index < length:
            check.refuse(
                stmt, f"index {index} lies outside the {length} slots of the array"
            )
        return check.eval_fallback(frame, stmt)

    @interp.impl(stmts.IntArrayZero)
    @interp.impl(stmts.FloatArrayZero)
    def array_zero(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.IntArrayZero | stmts.FloatArrayZero,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse an array of runtime length."""
        if const_int(stmt.size) is None:
            check.refuse(
                stmt, f"squin has no array of runtime length for '{stmt.name}'"
            )
        return check.eval_fallback(frame, stmt)

    @interp.impl(stmts.FloatAcosh)
    def acosh(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.FloatAcosh,
    ) -> interp.StatementResult[EmptyLattice]:
        """Refuse acosh, which squin's math dialect lacks."""
        check.refuse(stmt, "squin's math dialect has no acosh")
        return check.eval_fallback(frame, stmt)


@stmts.scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """A method table that refuses while loops."""

    @interp.impl(stmts.While)
    def while_(
        self,
        check: JeffToSquinAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: stmts.While,
    ) -> tuple[EmptyLattice, ...]:
        """Refuse the loop, then check its regions."""
        check.refuse(stmt, "squin cannot express while loops")
        return check.run_regions(frame, stmt)


@dataclass
class JeffToSquinValidation(ValidationPass[ForwardFrame[EmptyLattice]]):
    """A validation pass that reports every jeff construct that squin cannot express."""

    results: Mapping[ir.Method, Ref] | None = None
    """The wire reference result of each jeff function, or None to compute them."""

    def name(self) -> str:
        """Return the pass name that refusals show."""
        return "JeffToSquin"

    def run(
        self, method: ir.Method
    ) -> tuple[ForwardFrame[EmptyLattice], list[ir.ValidationError]]:
        """Run the conversion analysis on `method` and every function that it calls."""
        results = self.results
        if results is None:
            references = WireReferenceAnalysis(jeff_kernel)
            references.run(method)
            results = references.results
        analysis = JeffToSquinAnalysis(jeff_kernel)
        frame, _ = analysis.run(method)
        for function, result in results.items():
            handed = [p for p in handed_back(function, result) if p is not None]
            if handed != sorted(handed):
                analysis.refuse(
                    function.code,
                    "a function that hands its inputs back in another order has no "
                    "squin form",
                )
        return frame, analysis.get_validation_errors()
