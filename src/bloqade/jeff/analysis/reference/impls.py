"""This module holds the reference rules for kirin's own dialects."""

from kirin import types, interp
from kirin.dialects import py, scf, func, ilist
from kirin.analysis.forward import ForwardFrame

from bloqade.jeff.constants import const_int
from bloqade.jeff.signature import declared_outputs

from .lattice import (
    UNTRACKED,
    Ref,
    Items,
    Whole,
    Members,
    Unknown,
    Positions,
)
from .analysis import KEY, ReferenceAnalysis, as_positions

LIST = ilist.IListType[types.Any, types.Any]
"""The type of any list, which a register root has."""


@py.assign.dialect.register(key=KEY)
class _Assign(interp.MethodTable):
    """A method table that passes a reference through an alias."""

    @interp.impl(py.assign.Alias)
    def alias(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: py.assign.Alias,
    ) -> tuple[Ref, ...]:
        """Return the reference of the aliased value."""
        return (frame.get(stmt.value),)


@ilist.dialect.register(key=KEY)
class _IList(interp.MethodTable):
    """A method table that builds the references of literal lists."""

    @interp.impl(ilist.New)
    def new(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: ilist.New
    ) -> tuple[Ref, ...]:
        """Return `Items` if the list holds tracked state, and `Untracked` otherwise."""
        members = frame.get_values(stmt.values)
        if analysis.is_tracked_type(stmt.result.type) or any(
            m != UNTRACKED for m in members
        ):
            return (Items(members),)
        return (UNTRACKED,)


@py.binop.dialect.register(key=KEY)
class _BinOp(interp.MethodTable):
    """A method table that concatenates literal lists."""

    @interp.impl(py.binop.Add)
    def add(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: py.binop.Add
    ) -> tuple[Ref, ...]:
        """Join the items of two literal lists, and make other sums `Unknown`."""
        left, right = frame.get(stmt.lhs), frame.get(stmt.rhs)
        if isinstance(left, Items) and isinstance(right, Items):
            return (Items(left.refs + right.refs),)
        return analysis.unknown_results(stmt, "a concatenation of registers")


@py.tuple.dialect.register(key=KEY)
class _Tuple(interp.MethodTable):
    """A method table that builds the references of tuples."""

    @interp.impl(py.tuple.New)
    def new(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: py.tuple.New
    ) -> tuple[Ref, ...]:
        """Return the references of the tuple members as `Positions`."""
        return (Positions(frame.get_values(stmt.args)),)


@py.indexing.dialect.register(key=KEY)
class _Indexing(interp.MethodTable):
    """A method table that reads items of registers, lists and tuples."""

    @interp.impl(py.indexing.GetItem)
    def getitem(
        self,
        analysis: ReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: py.indexing.GetItem,
    ) -> tuple[Ref, ...]:
        """Return the reference of the item at the index."""
        obj = frame.get(stmt.obj)
        index = const_int(stmt.index)
        result = stmt.result.type
        match obj:
            case Whole(root) if analysis.origin(root).type.is_subseteq(LIST):
                if (
                    index is None
                    and analysis.is_tracked_type(result)
                    and result.is_subseteq(LIST)
                ):
                    return (Unknown("a slice of a register"),)
                return (analysis.slot(root, stmt.index),)
            case Members(members):
                if (
                    index is None
                    and isinstance(obj, Positions)
                    and not analysis.is_tracked_type(result)
                ):
                    return (UNTRACKED,)
                return (analysis.item(members, index),)
            case Unknown():
                return (obj,)
        return analysis.unknown_results(
            stmt, "an index into a value that is not a register or a literal list"
        )


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """A method table that translates the result of the callee at each call."""

    @interp.impl(func.Return)
    def return_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Return
    ) -> interp.ReturnValue[Ref]:
        """Return the reference of the returned value."""
        return interp.ReturnValue(frame.get(stmt.value))

    @interp.impl(func.Invoke)
    def invoke(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Invoke
    ) -> tuple[Ref, ...]:
        """Return the reference of the result, which packs the outputs in a tuple."""
        if analysis.is_allocating_call(stmt):
            return (Whole(stmt.result),)
        callee = stmt.callee
        if not isinstance(callee.code, func.Function):
            return analysis.unknown_results(
                stmt, "the result of a call to a nested function"
            )
        result = analysis.result_of(callee, frame)
        count = len(declared_outputs(callee.return_type))
        args = frame.get_values(stmt.inputs)
        resolved = analysis.translate(stmt, args, as_positions(result, count))
        if not resolved:
            return (UNTRACKED,)
        if len(resolved) == 1:
            return (resolved[0],)
        return (Positions(resolved),)

    @interp.impl(func.Call)
    def call(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Call
    ) -> tuple[Ref, ...]:
        """Give each tracked result of a dynamic call `Unknown`."""
        return analysis.unknown_results(stmt, "the result of a dynamic call")


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """A method table that joins the references that loops and branches carry.

    A carried value keeps its reference if every path hands back the same one.
    """

    @interp.impl(scf.Yield)
    def yield_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: scf.Yield
    ) -> interp.YieldValue[Ref]:
        """End the region with the yielded references."""
        return interp.YieldValue(frame.get_values(stmt.values))

    @interp.impl(scf.For)
    def for_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: scf.For
    ) -> tuple[Ref, ...]:
        """Run the body until the carried references stop changing."""
        carried = frame.get_values(stmt.initializers)
        return analysis.run_loop(frame, stmt, stmt.body, carried)

    @interp.impl(scf.IfElse)
    def if_else(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: scf.IfElse
    ) -> tuple[Ref, ...]:
        """Join the references that the two branches yield."""
        regions = (stmt.then_body, stmt.else_body)
        return analysis.run_branches(frame, stmt, regions, (UNTRACKED,))
