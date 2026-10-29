"""Emit functions and resolve source reference expressions through Kirin method tables."""

from typing import cast
from dataclasses import replace

from kirin import ir, types, interp
from kirin.dialects import py, func, ilist, ssacfg
from kirin.dialects.py import len as py_len, slice as py_slice

from bloqade import qubit
from bloqade.types import holds_qubits
from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Sym, Affine, same
from bloqade.squin.wire.group import kernel
from bloqade.squin.wire.types import (
    WireType,
    RegisterType,
    quantum_type,
    register_shape,
)
from bloqade.squin.wire.errors import (
    WireConversionError,
    UnsupportedCallContract,
    UnresolvedArgumentSeparation,
)
from bloqade.squin.wire.signature import Contract, packed_type, output_types

from .base import EmitWire, WireFrame, index_of
from .values import Value, Location, contains_location


def is_library(method: ir.Method) -> bool:
    """Recognize library kernels whose bodies define source quantum operations."""
    module = method.mod.__name__ if method.mod is not None else ""
    return method is not qubit.qalloc and module.startswith(
        ("bloqade.squin.stdlib", "bloqade.qubit.stdlib", "bloqade.native.stdlib")
    )


@ssacfg.dialect.register(key="emit.squin_wire")
class Regions(ssacfg.Concrete):
    """Use Kirin's region interpreter to visit each source statement in order."""


@func.dialect.register(key="emit.squin_wire")
class Functions(interp.MethodTable):
    """Emit function boundaries and calls with explicit borrowed-parameter results."""

    @interp.impl(func.Function)
    def function(self, emit: EmitWire, frame: WireFrame, stmt: func.Function):
        """Create output arguments, interpret the source body and construct its output method."""
        if isinstance(stmt.signature.output, types.BottomType):
            raise WireConversionError(
                "The function return has the impossible type Bottom.", stmt
            )
        inputs = tuple(quantum_type(kind) or kind for kind in stmt.signature.inputs)
        ordinary = tuple(
            quantum_type(kind) or kind for kind in output_types(stmt.signature.output)
        )
        borrowed = tuple(
            i
            for i, kind in enumerate(stmt.signature.inputs)
            if quantum_type(kind) is not None
        )
        signature = func.Signature(
            inputs, packed_type((*ordinary, *(inputs[i] for i in borrowed)))
        )
        frame.output.args.append_from(
            types.MethodType[list(inputs), signature.output],
            stmt.body.blocks[0].args[0].name,
        )
        arguments = []
        for source, kind in zip(stmt.body.blocks[0].args[1:], inputs, strict=True):
            argument = frame.output.args.append_from(kind, source.name)
            if quantum_type(source.type) is None:
                arguments.append(argument)
                continue
            shape = register_shape(source.type)
            length = None
            if shape and all(size is not None for size in shape):
                count = 1
                for size in shape:
                    assert size is not None
                    count *= size
                length = Affine(count)
            location = Location(argument, length, shape=shape)
            arguments.append(location)
            frame.current[argument] = argument
        returned = emit.frame_call(frame, stmt, None, *arguments)
        assert isinstance(returned, tuple)
        assert all(isinstance(value, ir.SSAValue) for value in returned)
        returned = cast(tuple[ir.SSAValue, ...], returned)
        signature = func.Signature(
            inputs, packed_type(tuple(value.type for value in returned))
        )
        frame.output.args[0].type = types.MethodType[list(inputs), signature.output]
        code = stmt.from_stmt(stmt, regions=[ir.Region(frame.output)])
        code.source = None if stmt.source is None else replace(stmt.source)
        code.signature = signature
        source = emit.source
        method = ir.Method(
            kernel,
            code,
            nargs=source.nargs,
            mod=source.mod,
            py_func=source.py_func,
            sym_name=source.sym_name,
            arg_names=source.arg_names,
            fields=source.fields,
            file=source.file,
            lineno_begin=source.lineno_begin,
        )
        end = stmt.body.blocks[0].last_stmt
        assert isinstance(end, func.Return)
        value = frame.get(end.value)
        values = value if isinstance(value, tuple) else (value,)
        lengths = tuple(
            value.length if isinstance(value, Location) else None for value in values
        )
        parameters = set(frame.output.args[1:])
        lengths = tuple(
            length if length is not None and length.symbols() <= parameters else None
            for length in lengths
        )
        if not any(length is not None for length in lengths):
            lengths = (None,) * len(ordinary)
        emit.result = Contract(
            method,
            inputs,
            tuple(value.type for value in returned[: len(ordinary)]),
            borrowed,
            lengths,
            isinstance(stmt.signature.output, types.Generic)
            and stmt.signature.output.is_subseteq(types.Tuple)
            and bool(ordinary),
        )
        return ()

    @interp.impl(func.Return)
    def return_(
        self, emit: EmitWire, frame: WireFrame, stmt: func.Return
    ) -> interp.ReturnValue[Value]:
        """Return owned outputs and borrowed parameters, then end local allocation lifetimes."""
        if isinstance(frame.code, func.Invoke):
            return interp.ReturnValue(frame.get(stmt.value))
        function = emit.source.code
        assert isinstance(function, func.Function)
        if stmt.parent_stmt is not function:
            raise WireConversionError(
                "Wire conversion requires a function-level return.", stmt
            )
        borrowed = [
            frame.get(arg)
            for arg in function.body.blocks[0].args[1:]
            if quantum_type(arg.type) is not None
        ]
        borrowed_roots = {
            value.root for value in borrowed if isinstance(value, Location)
        }
        results: list[ir.SSAValue] = []
        seen = set()
        for value in ordinary_values(emit, frame, function, stmt):
            if isinstance(value, Location):
                if value.selection is not None:
                    raise UnsupportedCallContract(
                        "A quantum return must own one whole allocation.", stmt
                    )
                if value.root in borrowed_roots:
                    raise UnsupportedCallContract(
                        "A function cannot return a borrowed quantum parameter as an ordinary output.",
                        stmt,
                    )
                if value.root in seen:
                    raise WireConversionError(
                        "A function cannot return one qubit twice.", stmt
                    )
                seen.add(value.root)
                results.append(emit.take(frame, value))
            elif contains_location(value):
                raise UnsupportedCallContract(
                    "A quantum return must own one whole allocation.", stmt
                )
            else:
                results.append(emit.classical(frame, value))
        for value in borrowed:
            assert isinstance(value, Location)
            results.append(emit.take(frame, value))
        frame.finish()
        if not results:
            packed = frame.push(func.ConstantNone()).results[0]
        elif len(results) == 1:
            packed = results[0]
        else:
            packed = frame.push(py.tuple.New(tuple(results))).results[0]
        frame.push(func.Return(packed))
        return interp.ReturnValue(tuple(results))

    @interp.impl(func.Invoke)
    def invoke(self, emit: EmitWire, frame: WireFrame, stmt: func.Invoke):
        """Allocate a register or emit a call using its checked callee contract."""
        if stmt.callee is qubit.qalloc:
            size = emit.classical(frame, frame.get(stmt.inputs[0]))
            root = frame.push(stmts.AllocRegister(size)).results[0]
            frame.current[root] = root
            frame.local_roots.append(root)
            length = index_of(size)
            if length.constant is not None:
                length = Affine(max(0, length.const))
            return (Location(root, length, shape=register_shape(stmt.result.type)),)
        if is_library(stmt.callee):
            context = frame.code if isinstance(frame.code, func.Invoke) else stmt
            with emit.new_frame(context) as body:
                body.output = frame.output
                body.current = frame.current
                body.local_roots = frame.local_roots
                result = emit.frame_call(
                    body, stmt.callee.code, None, *frame.get_values(stmt.inputs)
                )
            return (result,)
        if any(value.type.is_subseteq(types.Tuple) for value in stmt.inputs):
            raise UnsupportedCallContract(
                "Wire calls require separate arguments for tuple fields.", stmt
            )
        contract = emit.contracts[stmt.callee]
        values = frame.get_values(stmt.inputs)
        locations = [values[i] for i in contract.borrowed]
        for i, location in enumerate(locations):
            if not isinstance(location, Location) or not (
                location.scalar or location.selection is None
            ):
                raise UnsupportedCallContract(
                    "A call requires whole registers or individual qubits.", stmt
                )
            for other in locations[i + 1 :]:
                if not isinstance(other, Location) or not separated(location, other):
                    raise UnresolvedArgumentSeparation(
                        "Call arguments require separate quantum locations.", stmt
                    )
        inputs = tuple(
            (
                emit.take(frame, value)
                if i in contract.borrowed and isinstance(value, Location)
                else emit.classical(frame, value)
            )
            for i, value in enumerate(values)
        )
        call = frame.push(func.Invoke(inputs, callee=contract.method))
        call.result.type = packed_type(contract.outputs)
        if len(contract.outputs) <= 1:
            outputs = [call.result][: len(contract.outputs)]
        else:
            outputs = []
            for i, kind in enumerate(contract.outputs):
                at = frame.push(py.Constant(i)).results[0]
                item = frame.push(py.GetItem(call.result, at))
                item.results[0].type = kind
                outputs.append(item.results[0])
        count = len(contract.ordinary)
        for location, result in zip(locations, outputs[count:], strict=True):
            assert isinstance(location, Location)
            emit.restore(frame, location, result)
        parameters = contract.method.callable_region.blocks[0].args[1:]
        mapping: dict[Sym, Affine] = {
            parameter: index_of(emit.classical(frame, value))
            for parameter, value in zip(parameters, values, strict=True)
            if parameter.type.is_subseteq(types.Int)
        }
        ordinary = []
        for result, kind, length in zip(
            outputs[:count], contract.ordinary, contract.lengths, strict=True
        ):
            if kind in (WireType, RegisterType):
                length = None if length is None else length.substitute(mapping)
                if length is not None and length.constant is not None:
                    length = Affine(max(0, length.const))
                ordinary.append(Location(result, length))
                frame.current[result] = result
                frame.local_roots.append(result)
            else:
                ordinary.append(result)
        if contract.packed:
            return (tuple(ordinary),)
        return (
            ordinary[0] if ordinary else frame.push(func.ConstantNone()).results[0],
        )


def ordinary_values(
    emit: EmitWire, frame: WireFrame, function: func.Function, stmt: func.Return
):
    """Read ordinary source results according to the declared function result layout."""
    kinds = output_types(function.signature.output)
    if not kinds:
        return ()
    value = frame.get(stmt.value)
    if isinstance(
        function.signature.output, types.Generic
    ) and function.signature.output.is_subseteq(types.Tuple):
        if isinstance(value, tuple):
            return value
        packed = emit.classical(frame, value)
        fields = []
        for i, kind in enumerate(kinds):
            index = frame.push(py.Constant(i)).result
            item = frame.push(py.GetItem(packed, index))
            item.result.type = kind
            fields.append(item.result)
        return tuple(fields)
    return (value,)


def separated(left: Location, right: Location) -> bool:
    """Check whether two call operands name provably separate quantum locations."""
    if left.root is not right.root:
        return True
    return (
        isinstance(left.selection, Affine)
        and isinstance(right.selection, Affine)
        and same(left.selection, right.selection) is False
    )


@qubit.dialect.register(key="emit.squin_wire")
class Qubits(interp.MethodTable):
    """Emit standalone qubit allocations."""

    @interp.impl(qubit.stmts.New)
    def new(self, emit: EmitWire, frame: WireFrame, stmt):
        """Create a qubit and record its lifetime in the current region."""
        root = frame.push(stmts.Alloc()).results[0]
        frame.current[root] = root
        frame.local_roots.append(root)
        return (Location(root),)


@py.assign.dialect.register(key="emit.squin_wire")
class Aliases(interp.MethodTable):
    """Resolve source aliases from their previously emitted operands."""

    @interp.impl(py.assign.Alias)
    def alias(self, emit: EmitWire, frame: WireFrame, stmt):
        """Retain the source value's emitted value or quantum location."""
        return (frame.get(stmt.value),)


@ilist.dialect.register(key="emit.squin_wire")
class Lists(interp.MethodTable):
    """Retain quantum lists as location sequences and emit classical lists."""

    @interp.impl(ilist.New)
    def new(self, emit: EmitWire, frame: WireFrame, stmt):
        """Build a source container or copy a classical list into the output."""
        values = frame.get_values(stmt.values)
        if holds_qubits(stmt.result.type) or any(
            contains_location(value) for value in values
        ):
            return (values,)
        return emit.eval_fallback(frame, stmt)


@py.tuple.dialect.register(key="emit.squin_wire")
class Tuples(interp.MethodTable):
    """Keep source tuple fields available for indexing and function returns."""

    @interp.impl(py.tuple.New)
    def new(self, emit: EmitWire, frame: WireFrame, stmt):
        """Record tuple fields without emitting a redundant source container."""
        return (frame.get_values(stmt.args),)


@py.binop.dialect.register(key="emit.squin_wire")
class Arithmetic(interp.MethodTable):
    """Resolve quantum list concatenation and emit ordinary arithmetic."""

    @interp.impl(py.Add)
    def add(self, emit: EmitWire, frame: WireFrame, stmt):
        """Concatenate known quantum sequences or copy the classical addition."""
        left, right = frame.get_values(stmt.args)
        if contains_location(left) or contains_location(right):
            a, b = emit.items(left, stmt), emit.items(right, stmt)
            if a is None or b is None:
                raise WireConversionError(
                    "Quantum concatenation requires known source lengths.", stmt
                )
            return (a + b,)
        return emit.eval_fallback(frame, stmt)


@py_len.dialect.register(key="emit.squin_wire")
class Lengths(interp.MethodTable):
    """Emit the length of a source sequence."""

    @interp.impl(py_len.Len)
    def length(self, emit: EmitWire, frame: WireFrame, stmt):
        """Read the source sequence's known or runtime length."""
        return (emit.length(frame, frame.get(stmt.value)),)


@py.indexing.dialect.register(key="emit.squin_wire")
class Indexing(interp.MethodTable):
    """Select source quantum locations and emit ordinary classical indexing."""

    @interp.impl(py.GetItem)
    def getitem(self, emit: EmitWire, frame: WireFrame, stmt):
        """Resolve one source element or slice using already emitted operands."""
        obj, at = frame.get_values(stmt.args)
        index = emit.classical(frame, at)
        if isinstance(obj, ir.SSAValue):
            return emit.eval_fallback(frame, stmt)
        if index.type.is_subseteq(types.Slice):
            start, stop, step = slice_bounds(index, stmt)
            if isinstance(obj, Location):
                return (obj.slice(start, stop, step, stmt),)
            if isinstance(obj, tuple) and all(
                bound is None or bound.constant is not None for bound in (start, stop)
            ):
                return (
                    obj[
                        slice(
                            None if start is None else start.const,
                            None if stop is None else stop.const,
                            step,
                        )
                    ],
                )
        else:
            position = index_of(index)
            if isinstance(obj, Location):
                return (obj.select(position, stmt),)
            if (
                isinstance(obj, tuple)
                and position.constant is not None
                and -len(obj) <= position.const < len(obj)
            ):
                return (obj[position.const],)
            if isinstance(obj, tuple) and not contains_location(obj):
                return (
                    frame.push(py.GetItem(emit.classical(frame, obj), index)).results[
                        0
                    ],
                )
        raise WireConversionError(
            "A quantum container requires a tuple read at a constant index or a resolved register element.",
            stmt,
        )


def slice_bounds(value: ir.SSAValue, at: ir.Statement):
    """Read supported slice bounds from an emitted slice operation."""
    node = value.owner
    if isinstance(node, py_slice.Slice):
        start = (
            None
            if node.start.type.is_subseteq(types.NoneType)
            else index_of(node.start)
        )
        stop = (
            None if node.stop.type.is_subseteq(types.NoneType) else index_of(node.stop)
        )
        step = (
            1 if node.step.type.is_subseteq(types.NoneType) else constant_int(node.step)
        )
    elif isinstance(node, py.Constant) and isinstance(
        sliced := node.value.unwrap(), slice
    ):
        start = None if sliced.start is None else Affine(sliced.start)
        stop = None if sliced.stop is None else Affine(sliced.stop)
        step = 1 if sliced.step is None else sliced.step
    else:
        raise WireConversionError("The slice bounds have no supported source form.", at)
    if step is None or step == 0:
        raise WireConversionError(
            "A quantum slice requires a nonzero constant step.", at
        )
    return start, stop, step
