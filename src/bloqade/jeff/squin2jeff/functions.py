"""This module holds the rules that emit jeff calls and returns."""

from kirin import ir, interp
from kirin.rewrite import Walk, Fixpoint, DeadCodeElimination
from kirin.dialects import func

from bloqade import squin
from bloqade.qubit import stmts as qubit_stmts
from bloqade.jeff.forms import GATE_KERNELS, QUBIT_KERNELS
from bloqade.jeff.dialects import stmts
from bloqade.analysis.reference import (
    Whole,
    Register,
    Positions,
    Untracked,
    roots,
)
from bloqade.jeff.dialects.stmts.call import declared_outputs

from .linearize import KEY, Frame, Value, Linearize, jeff_type


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """A method table that emits calls and returns."""

    @interp.impl(func.Function)
    def function(
        self, emit: Linearize, frame: Frame, stmt: func.Function
    ) -> interp.StatementResult[Value]:
        """Fill the body of the jeff method of the kernel copy `stmt`."""
        method = emit.declare(stmt)
        frame.body = block = method.callable_region.blocks[0]
        args: list[Value] = []
        for param, arg in zip(stmt.body.blocks[0].args, block.args, strict=True):
            match emit.refs[param]:
                case Whole(root) | Register(root):
                    frame.wires[root] = arg
                    args.append(None)
                case Untracked():
                    args.append(arg)
                case _:
                    args.append(None)
        emit.frame_call(frame, stmt, *args)
        Fixpoint(Walk(DeadCodeElimination())).rewrite(method.code)
        return ()

    @interp.impl(func.ConstantNone)
    def none(
        self, emit: Linearize, frame: Frame, stmt: func.ConstantNone
    ) -> interp.StatementResult[Value]:
        """Give `None` no jeff value."""
        return (None,)

    @interp.impl(func.Invoke)
    def invoke(
        self, emit: Linearize, frame: Frame, stmt: func.Invoke
    ) -> interp.StatementResult[Value]:
        """Emit a library kernel as its jeff statements, and any other as a call.

        A gate kernel applies its gate to each qubit, or group of qubits, of its
        operands. `qalloc` becomes one register allocation. For any other callee,
        a qubit argument goes in as its wire and comes back as one of the first
        outputs, in argument order, and the extra outputs follow. An extra output
        that is a qubit becomes the wire of its reference.
        """
        if (form := GATE_KERNELS.get(stmt.callee)) is not None:
            angles = [frame.scalar(a) for a in stmt.inputs[: form.angles]]
            emit.gate(frame, form, stmt.inputs[form.angles :], angles, form.adjoint)
            return (None,)
        match QUBIT_KERNELS.get(stmt.callee):
            case qubit_stmts.New:
                (root,) = roots(emit.refs[stmt.result])
                frame.wires[root] = frame.push(stmts.Alloc()).result
                return (None,)
            case qubit_stmts.Reset:
                emit.reset_qubits(frame, stmt.inputs[0])
                return (None,)
            case qubit_stmts.Measure:
                return (emit.measure(frame, stmt.inputs[0]),)
            case qubit_stmts.IsOne:
                return (frame.value(stmt.inputs[0]),)
            case qubit_stmts.IsZero:
                return (emit.negate(frame, stmt.inputs[0]),)
        if stmt.callee is squin.qalloc:
            register = frame.push(stmts.RegAlloc(frame.scalar(stmt.inputs[0])))
            register.result.type = jeff_type(stmt.result.type)
            (root,) = roots(emit.refs[stmt.result])
            frame.wires[root] = register.result
            return (None,)
        code = stmt.callee.code
        if not isinstance(code, func.Function):
            raise interp.InterpreterError(f"{stmt.callee.sym_name} is not a function")
        if code not in emit.functions:
            emit.callable_to_emit.append(code)
        callee = emit.declare(code)
        passed = [emit.refs[arg] for arg in stmt.inputs]
        inputs = tuple(
            frame.scalar(arg) if isinstance(ref, Untracked) else emit.take(frame, ref)
            for arg, ref in zip(stmt.inputs, passed)
        )
        qubits = [ref for ref in passed if not isinstance(ref, Untracked)]
        call = frame.push(
            stmts.Call(callee, inputs, declared_outputs(callee.return_type))
        )
        for ref, wire in zip(qubits, call.results[: len(qubits)], strict=True):
            emit.give(frame, ref, wire)
        # An output whose reference is that of a qubit argument was handed back
        # above. The other outputs follow in the order of their positions.
        result = emit.refs[stmt.result]
        if not declared_outputs(stmt.callee.return_type):
            return (None,)
        positions = result.refs if isinstance(result, Positions) else (result,)
        extras = iter(call.results[len(qubits) :])
        values: list[Value] = []
        for ref in positions:
            if ref in qubits:
                values.append(None)
            elif isinstance(ref, Untracked):
                values.append(next(extras))
            else:
                emit.give(frame, ref, next(extras))
                values.append(None)
        return (tuple(values) if isinstance(result, Positions) else values[0],)

    @interp.impl(func.Return)
    def return_(
        self, emit: Linearize, frame: Frame, stmt: func.Return
    ) -> interp.ReturnValue[Value]:
        """End the function with the wires of the qubit parameters, then the rest.

        A returned value whose reference is that of a qubit parameter is handed
        back with the parameters and appears once.
        """
        params = (
            [
                p
                for p in stmt.parent_block.args[1:]
                if not isinstance(emit.refs[p], Untracked)
            ]
            if stmt.parent_block is not None
            else []
        )
        handed = [frame.wires[p] for p in params]
        result = emit.refs[stmt.value]
        value = frame.value(stmt.value)
        values = value if isinstance(value, tuple) else (value,)
        signature = frame.code.get_present_trait(ir.HasSignature).get_signature(
            frame.code
        )
        declared = declared_outputs(signature.output)
        if not declared:
            values = ()
        refs = result.refs if isinstance(result, Positions) else (result,) * len(values)
        rest: list[ir.SSAValue] = []
        for ref, given, kind in zip(refs, values, declared, strict=True):
            handed_back = isinstance(ref, (Whole, Register)) and ref.root in params
            if isinstance(given, ir.SSAValue):
                rest.append(emit.fitted(frame, given, kind))
            elif not handed_back and not isinstance(ref, Untracked):
                rest.append(emit.take(frame, ref))
        emit.free(frame, [*params, *roots(result)])
        frame.push(stmts.Return(*handed, *rest))
        return interp.ReturnValue(None)
