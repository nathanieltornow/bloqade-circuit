"""Emit function boundaries, library wrappers and calls in the wire convention."""

import math
from typing import cast
from dataclasses import replace

from kirin import ir, types, interp
from kirin.dialects import py, func

from bloqade import qubit
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
from bloqade.squin.wire.signature import (
    Layout,
    Contract,
    packed_type,
    output_types,
)

from .base import EmitWire, WireFrame, index_of
from .values import Value, Location, contains_location

LIBRARIES = ("bloqade.squin.stdlib", "bloqade.qubit.stdlib", "bloqade.native.stdlib")
"""The modules whose kernels wrap source quantum operations for users."""


def is_library(method: ir.Method) -> bool:
    """Check whether `method` is a kernel of a module in `LIBRARIES`.

    The emitter runs the body of a library kernel at each call. It keeps an explicit
    call for every other kernel.
    """
    module = method.mod.__name__ if method.mod is not None else ""
    return module.startswith(LIBRARIES)


@func.dialect.register(key="emit.squin_wire")
class Functions(interp.MethodTable):
    """Emit function boundaries and calls in the calling convention of `Layout`."""

    @interp.impl(func.Function)
    def function(self, emit: EmitWire, frame: WireFrame, stmt: func.Function):
        """Emit the wire form of the source function and record its contract."""
        if isinstance(stmt.signature.output, types.BottomType):
            raise WireConversionError(
                "The function return has the impossible type Bottom.", stmt
            )
        layout = Layout.of(stmt.signature)
        block = stmt.body.blocks[0]
        inputs = tuple(quantum_type(kind) or kind for kind in stmt.signature.inputs)
        # The emitted body decides the output type, so the self argument gets its
        # final type after the body.
        this = frame.output.args.append_from(types.Any, block.args[0].name)
        arguments = [
            parameter(frame, source, kind)
            for source, kind in zip(block.args[1:], inputs, strict=True)
        ]
        returned = emit.frame_call(frame, stmt, None, *arguments)
        end = block.last_stmt
        assert isinstance(end, func.Return)
        ordinary = ordinary_values(emit, frame, stmt, end, layout, returned)
        parameters = frozenset(frame.output.args[1:])
        lengths = tuple(static_length(value, parameters) for value in ordinary)
        borrowed = [cast(Location, arguments[i]) for i in layout.borrowed]
        results = return_results(emit, frame, end, ordinary, borrowed)
        signature = func.Signature(
            inputs, packed_type(tuple(value.type for value in results))
        )
        this.type = types.MethodType[list(inputs), signature.output]
        method = wire_method(emit.source, stmt, frame.output, signature)
        emit.result = Contract(method, lengths)
        return ()

    @interp.impl(func.Return)
    def return_(
        self, emit: EmitWire, frame: WireFrame, stmt: func.Return
    ) -> interp.ReturnValue[Value]:
        """Return the source value to the function or library call that runs the body."""
        if stmt.parent_stmt is not frame.code:
            raise WireConversionError(
                "Wire conversion requires a function-level return.", stmt
            )
        return interp.ReturnValue(frame.get(stmt.value))

    @interp.impl(func.Invoke)
    def invoke(self, emit: EmitWire, frame: WireFrame, stmt: func.Invoke):
        """Allocate a register, run a library wrapper or call a converted function."""
        # `qalloc` builds its register with a list map, which the emitter cannot run.
        if stmt.callee is qubit.qalloc:
            size = emit.classical_value(frame, frame.get(stmt.inputs[0]))
            root = frame.push(stmts.AllocRegister(size)).results[0]
            frame.latest[root] = root
            frame.local_roots.append(root)
            length = index_of(size)
            if (count := length.constant) is not None:
                length = Affine(max(0, count))
            return (Location(root, length, shape=register_shape(stmt.result.type)),)
        if is_library(stmt.callee):
            # The wrapper emits into the caller's block and continues the caller's
            # wires. Its allocations live until the caller's region ends.
            with emit.new_frame(stmt.callee.code) as body:
                body.site = frame.site or stmt
                body.output = frame.output
                body.latest = frame.latest
                body.local_roots = frame.local_roots
                result = emit.frame_call(
                    body, stmt.callee.code, None, *frame.get_values(stmt.inputs)
                )
            return (result,)
        return (call(emit, frame, stmt),)


def parameter(
    frame: WireFrame, source: ir.BlockArgument, kind: types.TypeAttribute
) -> Value:
    """Add the wire parameter of a source parameter and return the source meaning.

    A quantum parameter becomes the location of a whole allocation. Its length is
    known if the source type gives every register dimension.
    """
    argument = frame.output.args.append_from(kind, source.name)
    if quantum_type(source.type) is None:
        return argument
    shape = register_shape(source.type)
    length = None
    if shape and all(size is not None for size in shape):
        length = Affine(math.prod(cast(tuple[int, ...], shape)))
    frame.latest[argument] = argument
    return Location(argument, length, shape=shape)


def ordinary_values(
    emit: EmitWire,
    frame: WireFrame,
    function: func.Function,
    end: func.Return,
    layout: Layout,
    value: Value,
) -> tuple[Value, ...]:
    """Split the returned source value into the ordinary results of `layout`."""
    if not layout.ordinary:
        return ()
    if not layout.packed:
        return (value,)
    if not isinstance(value, tuple):
        kinds = output_types(function.signature.output)
        return unpack(frame, emit.classical_value(frame, value), kinds)
    if len(value) != layout.ordinary:
        raise WireConversionError(
            "The returned tuple must match the declared result count.", end
        )
    return value


def static_length(value: Value, parameters: frozenset[ir.SSAValue]) -> Affine | None:
    """Return the length of a returned register if the parameters determine it."""
    if isinstance(value, Location) and value.root_length is not None:
        if value.root_length.symbols() <= parameters:
            return value.root_length
    return None


def return_results(
    emit: EmitWire,
    frame: WireFrame,
    end: func.Return,
    ordinary: tuple[Value, ...],
    borrowed: list[Location],
) -> tuple[ir.SSAValue, ...]:
    """Emit the wire return with the ordinary results first and the borrowed qubits last.

    A quantum ordinary result must own one whole fresh allocation. The function
    frees every other local allocation before it returns.
    """
    borrowed_roots = {location.root for location in borrowed}
    results: list[ir.SSAValue] = []
    seen = set()
    for value in ordinary:
        if isinstance(value, Location):
            if value.selection is not None:
                raise UnsupportedCallContract(
                    "A quantum return must own one whole allocation.", end
                )
            if value.root in borrowed_roots:
                raise UnsupportedCallContract(
                    "A function cannot return a borrowed quantum parameter as an ordinary output.",
                    end,
                )
            if value.root in seen:
                raise WireConversionError(
                    "A function cannot return one qubit twice.", end
                )
            seen.add(value.root)
            results.append(emit.acquire(frame, value))
        elif contains_location(value):
            raise UnsupportedCallContract(
                "A quantum return must own one whole allocation.", end
            )
        else:
            results.append(emit.classical_value(frame, value))
    results.extend(emit.acquire(frame, location) for location in borrowed)
    frame.free_local_allocations()
    if not results:
        packed = frame.push(func.ConstantNone()).results[0]
    elif len(results) == 1:
        packed = results[0]
    else:
        packed = frame.push(py.tuple.New(tuple(results))).results[0]
    frame.push(func.Return(packed))
    return tuple(results)


def wire_method(
    source: ir.Method, stmt: func.Function, body: ir.Block, signature: func.Signature
) -> ir.Method:
    """Build the wire method with the body `body` and the metadata of `source`."""
    code = stmt.from_stmt(stmt, regions=[ir.Region(body)])
    code.source = None if stmt.source is None else replace(stmt.source)
    code.signature = signature
    return ir.Method(
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


def call(emit: EmitWire, frame: WireFrame, stmt: func.Invoke) -> Value:
    """Call a converted function and continue the borrowed qubits with its results."""
    if any(value.type.is_subseteq(types.Tuple) for value in stmt.inputs):
        raise UnsupportedCallContract(
            "Wire calls require separate arguments for tuple fields.", stmt
        )
    callee = stmt.callee.code
    assert isinstance(callee, func.Function)
    layout = Layout.of(callee.signature)
    contract = emit.contracts[stmt.callee]
    values = frame.get_values(stmt.inputs)
    borrowed = quantum_operands(values, layout, stmt)
    operands = dict(zip(layout.borrowed, borrowed, strict=True))
    inputs = tuple(
        (
            emit.acquire(frame, operands[i])
            if i in operands
            else emit.classical_value(frame, x)
        )
        for i, x in enumerate(values)
    )
    made = frame.push(func.Invoke(inputs, callee=contract.method))
    made.result.type = contract.method.return_type
    outputs = call_results(frame, made.result, layout)
    for location, result in zip(borrowed, outputs[layout.ordinary :], strict=True):
        emit.restore(frame, location, result)
    ordinary = fresh_results(emit, frame, contract, values, outputs[: layout.ordinary])
    if layout.packed:
        return ordinary
    return ordinary[0] if ordinary else frame.push(func.ConstantNone()).results[0]


def quantum_operands(
    values: tuple[Value, ...], layout: Layout, stmt: func.Invoke
) -> list[Location]:
    """Return the borrowed call operands after checking that they are separate qubits.

    Each operand must be one qubit or one whole register.
    """
    operands = [values[i] for i in layout.borrowed]
    for i, operand in enumerate(operands):
        if not isinstance(operand, Location) or not (
            operand.scalar or operand.selection is None
        ):
            raise UnsupportedCallContract(
                "A call requires whole registers or individual qubits.", stmt
            )
        for other in operands[i + 1 :]:
            if not isinstance(other, Location) or not separated(operand, other):
                raise UnresolvedArgumentSeparation(
                    "Call arguments require separate quantum locations.", stmt
                )
    return cast(list[Location], operands)


def call_results(
    frame: WireFrame, result: ir.SSAValue, layout: Layout
) -> tuple[ir.SSAValue, ...]:
    """Split the result of a wire call into one value for each wire result."""
    kinds = layout.results(result.type)
    if not kinds:
        return ()
    if len(kinds) == 1:
        return (result,)
    return unpack(frame, result, kinds)


def fresh_results(
    emit: EmitWire,
    frame: WireFrame,
    contract: Contract,
    values: tuple[Value, ...],
    results: tuple[ir.SSAValue, ...],
) -> tuple[Value, ...]:
    """Give each quantum result of a call a new allocation identity in the caller.

    A fresh register gets the static length from the contract. The caller's integer
    arguments replace the callee's parameters in that length.
    """
    parameters = contract.method.callable_region.blocks[0].args[1:]
    mapping: dict[Sym, Affine] = {
        parameter: index_of(emit.classical_value(frame, value))
        for parameter, value in zip(parameters, values, strict=True)
        if parameter.type.is_subseteq(types.Int)
    }
    meanings: list[Value] = []
    for result, length in zip(results, contract.lengths, strict=True):
        if result.type not in (WireType, RegisterType):
            meanings.append(result)
            continue
        if length is not None:
            length = length.substitute(mapping)
            if (count := length.constant) is not None:
                length = Affine(max(0, count))
        meanings.append(Location(result, length))
        frame.latest[result] = result
        frame.local_roots.append(result)
    return tuple(meanings)


def unpack(
    frame: WireFrame, packed: ir.SSAValue, kinds: tuple[types.TypeAttribute, ...]
) -> tuple[ir.SSAValue, ...]:
    """Read each field of the tuple value `packed` and give it the type in `kinds`."""
    fields = []
    for i, kind in enumerate(kinds):
        index = frame.push(py.Constant(i)).results[0]
        item = frame.push(py.GetItem(packed, index))
        item.results[0].type = kind
        fields.append(item.results[0])
    return tuple(fields)


def separated(left: Location, right: Location) -> bool:
    """Check whether two call operands name provably separate quantum locations."""
    if left.root is not right.root:
        return True
    return (
        isinstance(left.selection, Affine)
        and isinstance(right.selection, Affine)
        and same(left.selection, right.selection) is False
    )
