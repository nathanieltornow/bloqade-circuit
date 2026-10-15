"""This module holds the reference rules for kirin's own dialects."""

from kirin import types, interp
from kirin.dialects import py, scf, func, ilist
from kirin.analysis.forward import ForwardFrame
from kirin.dialects.scf.absint import Methods as ScfAbsint

from .lattice import (
    UNTRACKED,
    Ref,
    Items,
    Members,
    Positions,
    place,
)
from .analysis import KEY, ReferenceAnalysis


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
    """A method table that builds the references of literal lists and maps."""

    @interp.impl(ilist.New)
    def new(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: ilist.New
    ) -> tuple[Ref, ...]:
        """Return `Items` if the list holds tracked state, and `Untracked` otherwise."""
        members = frame.get_values(stmt.values)
        if analysis.shape(stmt.result.type) is not None or any(
            m != UNTRACKED for m in members
        ):
            return (Items(members),)
        return (UNTRACKED,)

    @interp.impl(ilist.Map)
    def map_(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: ilist.Map
    ) -> tuple[Ref, ...]:
        """Return the result as a register root if the map produces tracked values."""
        if (depth := analysis.shape(stmt.result.type)) is not None and depth > 0:
            return (place(stmt.result, (), depth),)
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
        """Return the reference of the item or the slice at the index."""
        obj = frame.get(stmt.obj)
        shape = analysis.shape(stmt.result.type)
        if stmt.index.type.is_subseteq(types.Slice):
            if shape is None and not isinstance(obj, Members):
                return (UNTRACKED,)
            return (analysis.slice(obj, stmt.index),)
        index = analysis.affine(stmt.index)
        # A tuple read at a runtime index is fine if the item is untracked.
        if index.constant is None and isinstance(obj, Positions) and shape is None:
            return (UNTRACKED,)
        # An untracked item of a list or tuple is looked up. Any other untracked
        # read, such as a classical list, needs no reference.
        if shape is None and not isinstance(obj, Members):
            return (UNTRACKED,)
        return (analysis.index(obj, index),)


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """A method table that runs the callee at each call with the caller's references."""

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
        """Return the reference of the result in the terms of the caller."""
        return (analysis.call_result(frame, stmt),)

    @interp.impl(func.Call)
    def call(
        self, analysis: ReferenceAnalysis, frame: ForwardFrame[Ref], stmt: func.Call
    ) -> tuple[Ref, ...]:
        """Give each tracked result of a dynamic call `Unknown`."""
        return analysis.unknown_results(stmt, "the result of a dynamic call")


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """A method table that joins the references that an `scf.for` loop carries.

    A carried value keeps its reference if the body hands back the same one.
    """

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
    ):
        """Join the branches with kirin's rule, then leave the scope of the branches.

        Kirin runs only one branch if the condition is a constant. That branch can
        yield a place whose index the branch computes, so the result must leave the
        scope of the statement.
        """
        results = ScfAbsint().if_else(analysis, frame, stmt)
        if isinstance(results, tuple):
            return tuple(analysis.leave(r, stmt) for r in results)
        return results
