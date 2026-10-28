"""This module holds the rules that emit jeff loops and switches for kirin control flow."""

from kirin import ir, types, interp
from kirin.dialects import scf
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.jeff.forms import loop_range
from bloqade.jeff.dialects import stmts

from .emitter import KEY, Frame, Value, WireEmitter


def _outs(
    emit: WireEmitter,
    inner: Frame,
    stmt: ir.Statement,
    yielded: object,
) -> list[ir.SSAValue]:
    """Fit each yielded value to the corresponding wire or classical result type."""
    if not isinstance(yielded, tuple):
        raise interp.InterpreterError(f"a region of {stmt.name} does not yield")
    outs: list[ir.SSAValue] = []
    if len(yielded) != len(stmt.results):
        raise interp.InterpreterError(
            "The region yield must match its declared result count."
        )
    for k in range(len(stmt.results)):
        if not isinstance(yielded[k], ir.SSAValue):
            raise interp.InterpreterError(
                f"a region of {stmt.name} requires separate scalar or array results"
            )
        outs.append(emit.fitted(inner, yielded[k], stmt.results[k].type))
    return outs


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """A method table that emits jeff loops and switches for kirin control flow."""

    @interp.impl(scf.Yield)
    def yield_(
        self, emit: WireEmitter, frame: Frame, stmt: scf.Yield
    ) -> interp.YieldValue[Value]:
        """End a region with the values that it yields."""
        return interp.YieldValue(frame.values(stmt.values))

    @interp.impl(scf.For)
    def for_(
        self, emit: WireEmitter, frame: Frame, stmt: scf.For
    ) -> interp.StatementResult[Value]:
        """Emit a Jeff loop that carries every explicit wire and classical value."""
        with emit.new_frame(stmt) as inner:
            index = inner.body.args.append_from(types.Int, "index")
            carried = [
                inner.body.args.append_from(frame.scalar(value).type)
                for value in stmt.initializers
            ]
            args = [index, *carried]
            yielded = emit.frame_call_region(inner, stmt, stmt.body, *args)
            body = emit.leave(inner, _outs(emit, inner, stmt, yielded))
        bounds: tuple[ir.SSAValue, ir.SSAValue, ir.SSAValue]
        match loop_range(stmt.iterable):
            case ilist_stmts.Range(start=start, stop=stop, step=step):
                bounds = (frame.scalar(start), frame.scalar(stop), frame.scalar(step))
            case range(start=start, stop=stop, step=step):
                bounds = (
                    frame.push(stmts.ConstInt(value=start)).result,
                    frame.push(stmts.ConstInt(value=stop)).result,
                    frame.push(stmts.ConstInt(value=step)).result,
                )
            case _:
                raise interp.InterpreterError(f"{stmt.iterable} is not a range")
        state = tuple(frame.scalar(value) for value in stmt.initializers)
        loop = frame.push(stmts.For(*bounds, state, body))
        return tuple(loop.results)

    @interp.impl(scf.IfElse)
    def if_else(
        self, emit: WireEmitter, frame: Frame, stmt: scf.IfElse
    ) -> interp.StatementResult[Value]:
        """Emit a jeff switch on the condition with the else branch as case zero."""
        regions: list[ir.Region] = []
        for body in (stmt.else_body, stmt.then_body):
            with emit.new_frame(stmt) as inner:
                (arg,) = body.blocks[0].args
                cond = inner.scalar(stmt.cond) if arg.uses else None
                yielded = emit.frame_call_region(inner, stmt, body, cond)
                outs = _outs(emit, inner, stmt, yielded)
                regions.append(emit.leave(inner, outs))
        switch = stmts.Switch(frame.scalar(stmt.cond), (), regions[:1], regions[1])
        return tuple(frame.push(switch).results)
