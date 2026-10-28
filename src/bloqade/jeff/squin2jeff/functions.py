"""Emit wire function calls and returns in their declared value order."""

from kirin import ir, types, interp
from kirin.dialects import func

from bloqade.jeff.dialects import stmts
from bloqade.jeff.dialects.stmts.call import declared_outputs

from .emitter import KEY, Frame, WireEmitter


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """Translate explicit wire signatures and calls to Jeff signatures and calls."""

    @interp.impl(func.Function)
    def function(self, emit: WireEmitter, frame: Frame, stmt: func.Function):
        """Fill the declared function using its explicit input values."""
        method = emit.declare(stmt)
        frame.body = block = method.callable_region.blocks[0]
        emit.frame_call(frame, stmt, *block.args)
        return ()

    @interp.impl(func.ConstantNone)
    def none(self, emit: WireEmitter, frame: Frame, stmt: func.ConstantNone):
        """Represent the absence of a return value."""
        return (None,)

    @interp.impl(func.Invoke)
    def invoke(self, emit: WireEmitter, frame: Frame, stmt: func.Invoke):
        """Call the declared wire callee and preserve its result tuple order."""
        if stmt.callee.code not in emit.functions:
            emit.callable_to_emit.append(stmt.callee.code)
        code = stmt.callee.code
        if not isinstance(code, func.Function):
            raise interp.InterpreterError("Jeff calls require a declared function.")
        callee = emit.declare(code)
        call = frame.push(
            stmts.Call(
                callee,
                tuple(frame.scalar(arg) for arg in stmt.inputs),
                declared_outputs(callee.return_type),
            )
        )
        if stmt.result.type.is_subseteq(types.NoneType):
            return (None,)
        if stmt.result.type.is_subseteq(types.Tuple):
            return (tuple(call.results),)
        return (call.results[0],)

    @interp.impl(func.Return)
    def return_(self, emit: WireEmitter, frame: Frame, stmt: func.Return):
        """Return the wire function's ordinary outputs followed by its borrowed inputs."""
        value = frame.value(stmt.value)
        values = (
            () if value is None else value if isinstance(value, tuple) else (value,)
        )
        owner = stmt.parent_stmt
        if not isinstance(owner, func.Function):
            raise interp.InterpreterError("Jeff requires returns at function scope.")
        kinds = declared_outputs(owner.signature.output)
        if len(values) != len(kinds):
            raise interp.InterpreterError(
                "The return values must match the declared function outputs."
            )
        outputs = []
        for value, kind in zip(values, kinds, strict=True):
            if not isinstance(value, ir.SSAValue):
                raise interp.InterpreterError(
                    "Jeff returns require flat scalar or array values."
                )
            outputs.append(emit.fitted(frame, value, kind))
        frame.push(stmts.Return(*outputs))
        return interp.ReturnValue(None)
