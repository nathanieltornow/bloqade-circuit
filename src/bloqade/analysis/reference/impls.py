"""Track references through qubits, containers, calls, and control flow."""

from kirin import interp
from kirin.dialects import py, scf, func, ilist
from kirin.analysis.forward import ForwardFrame

from bloqade.qubit import stmts as qubit_stmts, dialect as qubit_dialect
from bloqade.types import QubitType

from ._ir import const_int, declared_outputs
from .lattice import (
    CARRIED,
    CLASSICAL,
    Ref,
    Slot,
    Whole,
    Members,
    Unknown,
    Positions,
    QubitList,
    is_allocator,
    static_length,
    is_quantum_type,
    is_register_root,
)
from .analysis import ReferenceAnalysis, rebased, as_positions, unknown_results


@qubit_dialect.register(key="qubit.reference")
class _Qubit(interp.MethodTable):
    @interp.impl(qubit_stmts.New)
    def new(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: qubit_stmts.New,
    ) -> tuple[Ref, ...]:
        return (Whole(stmt.result),)


@py.assign.dialect.register(key="qubit.reference")
class _Assign(interp.MethodTable):
    @interp.impl(py.assign.Alias)
    def alias(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: py.assign.Alias,
    ) -> tuple[Ref, ...]:
        return (frame.get(stmt.value),)


@ilist.dialect.register(key="qubit.reference")
class _IList(interp.MethodTable):
    @interp.impl(ilist.New)
    def new(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: ilist.New
    ) -> tuple[Ref, ...]:
        members = tuple(frame.get(v) for v in stmt.values)
        quantum = is_quantum_type(stmt.result.type) or any(
            m != CLASSICAL for m in members
        )
        return (QubitList(members) if quantum else CLASSICAL,)


@py.binop.dialect.register(key="qubit.reference")
class _BinOp(interp.MethodTable):
    @interp.impl(py.binop.Add)
    def add(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: py.binop.Add
    ) -> tuple[Ref, ...]:
        """Concatenate the references in two explicit qubit lists."""
        left, right = frame.get(stmt.lhs), frame.get(stmt.rhs)
        if isinstance(left, QubitList) and isinstance(right, QubitList):
            return (QubitList(left.refs + right.refs),)
        if not is_quantum_type(stmt.result.type):
            return (CLASSICAL,)
        return unknown_results(stmt, "a concatenation of registers")


@py.tuple.dialect.register(key="qubit.reference")
class _Tuple(interp.MethodTable):
    @interp.impl(py.tuple.New)
    def new(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: py.tuple.New
    ) -> tuple[Ref, ...]:
        return (Positions(tuple(frame.get(v) for v in stmt.args)),)


@py.indexing.dialect.register(key="qubit.reference")
class _Indexing(interp.MethodTable):
    @interp.impl(py.indexing.GetItem)
    def getitem(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: py.indexing.GetItem,
    ) -> tuple[Ref, ...]:
        obj = frame.get(stmt.obj)
        index = const_int(stmt.index)
        result = stmt.result
        match obj:
            case Whole(root) if is_register_root(root):
                if index is None:
                    if is_quantum_type(result.type) and not result.type.is_subseteq(
                        QubitType
                    ):
                        return (Unknown("a slice of a register"),)
                    return (Slot(root, stmt.index),)
                size = static_length(root)
                if size is None:
                    return (Slot(root, index),)
                if not -size <= index < size:
                    return (Unknown("a constant index out of range"),)
                return (Slot(root, index % size),)
            case Members(members):
                if index is None:
                    if isinstance(obj, Positions) and not is_quantum_type(result.type):
                        return (CLASSICAL,)
                    return (Unknown("a qubit list or tuple read at a runtime index"),)
                if not -len(members) <= index < len(members):
                    return (Unknown("a constant index out of range"),)
                return (members[index],)
            case Unknown():
                return (obj,)
        if is_quantum_type(result.type):
            return (
                Unknown(
                    "an index into a value that is not a register or a literal list"
                ),
            )
        return (CLASSICAL,)


@func.dialect.register(key="qubit.reference")
class _Func(interp.MethodTable):
    @interp.impl(func.Return)
    def return_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Return
    ) -> interp.ReturnValue:
        return interp.ReturnValue(frame.get(stmt.value))

    @interp.impl(func.Invoke)
    def invoke(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Invoke
    ) -> tuple[Ref, ...]:
        if is_allocator(stmt):
            return (Whole(stmt.result),)
        callee = stmt.callee
        if not isinstance(callee.code, func.Function):
            # a nested `def` is a lambda, which has no signature to call by
            return unknown_results(stmt, "the result of a call to a nested function")
        outputs = declared_outputs(callee.code.signature)
        args = tuple(frame.get(v) for v in stmt.inputs)
        if callee.code in analysis.active:
            returned: tuple[Ref, ...] = tuple(
                (
                    Unknown("the result of a recursive call")
                    if is_quantum_type(t)
                    else CLASSICAL
                )
                for t in outputs
            )
        elif (key := (callee.code, args)) in analysis.memo:
            returned = analysis.memo[key]
        else:
            analysis.active.add(callee.code)
            try:
                _, result = analysis.call(callee.code, CLASSICAL, *args)
            finally:
                analysis.active.discard(callee.code)
            returned = analysis.memo[key] = as_positions(result, len(outputs))
        resolved = rebased(returned, stmt, callee.callable_region)
        if not resolved:
            return (CLASSICAL,)
        if len(resolved) == 1:
            return (resolved[0],)
        return (Positions(resolved),)

    @interp.impl(func.Call)
    def call(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Call
    ) -> tuple[Ref, ...]:
        return unknown_results(stmt, "the result of a dynamic call")


def _yielded(result: object) -> tuple[Ref, ...]:
    """Extract yielded references from a YieldValue or tuple."""
    if isinstance(result, interp.YieldValue):
        return tuple(result.values)
    if isinstance(result, tuple):
        return result
    return ()


@scf.dialect.register(key="qubit.reference")
class _Scf(interp.MethodTable):
    """Analyze branches and loops in the enclosing frame.

    Join references across branches and loop iterations. Conflicting references
    become unknown.
    """

    @interp.impl(scf.Yield)
    def yield_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: scf.Yield
    ) -> interp.YieldValue:
        return interp.YieldValue(frame.get_values(stmt.values))

    @interp.impl(scf.For)
    def for_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: scf.For
    ) -> tuple[Ref, ...]:
        carried = tuple(frame.get(v) for v in stmt.initializers)
        while True:  # each pass that changes a slot moves it up a finite lattice
            result = analysis.frame_call_region(
                frame, stmt, stmt.body, CLASSICAL, *carried
            )
            yielded = _yielded(result)
            if len(yielded) != len(carried):
                return unknown_results(stmt, CARRIED)
            joined = tuple(c.join(y) for c, y in zip(carried, yielded, strict=True))
            if joined == carried:
                break
            carried = joined
        frame.set_values(
            stmt.body.blocks[0].args[1:], carried
        )  # the solver skips a seen block
        return carried

    @interp.impl(scf.IfElse)
    def if_else(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: scf.IfElse
    ) -> tuple[Ref, ...]:
        yields = []
        for body in (stmt.then_body, stmt.else_body):
            if body.blocks:
                result = analysis.frame_call_region(frame, stmt, body, CLASSICAL)
                yields.append(_yielded(result))
        if len(yields) == 2 and all(len(y) == len(stmt.results) for y in yields):
            return tuple(a.join(b) for a, b in zip(*yields, strict=True))
        return unknown_results(stmt, CARRIED)
