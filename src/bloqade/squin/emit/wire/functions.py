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
        input_types = tuple(
            quantum_type(kind) or kind for kind in stmt.signature.inputs
        )
        # The emitted body decides the output type, so the self argument gets its
        # final type after the body.
        self_arg = frame.output.args.append_from(types.Any, block.args[0].name)
        parameter_values = [
            add_parameter(frame, source, kind)
            for source, kind in zip(block.args[1:], input_types, strict=True)
        ]
        # `None` is the meaning of the source self argument.
        returned = emit.frame_call(frame, stmt, None, *parameter_values)
        end = block.last_stmt
        assert isinstance(end, func.Return)

        ordinary = ordinary_values(emit, frame, stmt, end, layout, returned)
        wire_parameters = frozenset(frame.output.args[1:])
        lengths = tuple(static_length(value, wire_parameters) for value in ordinary)
        borrowed = [
            cast(Location, parameter_values[i]) for i in layout.borrowed_positions
        ]
        results = (
            *ordinary_wires(emit, frame, end, ordinary, borrowed),
            *(emit.acquire(frame, location) for location in borrowed),
        )
        frame.free_local_allocations()
        emit_return(frame, results)

        signature = func.Signature(
            input_types, packed_type(tuple(value.type for value in results))
        )
        self_arg.type = types.MethodType[list(input_types), signature.output]
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
        if stmt.callee is qubit.qalloc:
            return (allocate_register(emit, frame, stmt),)
        if is_library(stmt.callee):
            return (run_library(emit, frame, stmt),)
        return (call(emit, frame, stmt),)


def add_parameter(
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
    if not layout.ordinary_count:
        return ()
    if not layout.packed:
        return (value,)
    if not isinstance(value, tuple):
        kinds = output_types(function.signature.output)
        return unpack(frame, emit.classical_value(frame, value), kinds)
    if len(value) != layout.ordinary_count:
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


def ordinary_wires(
    emit: EmitWire,
    frame: WireFrame,
    end: func.Return,
    ordinary: tuple[Value, ...],
    borrowed: list[Location],
) -> list[ir.SSAValue]:
    """Return the wire value of each ordinary result of a function.

    A quantum ordinary result must own one whole fresh allocation. It must differ
    from the other results and from every borrowed parameter.
    """
    borrowed_roots = {location.root for location in borrowed}
    wires: list[ir.SSAValue] = []
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
            wires.append(emit.acquire(frame, value))
        elif contains_location(value):
            raise UnsupportedCallContract(
                "A quantum return must own one whole allocation.", end
            )
        else:
            wires.append(emit.classical_value(frame, value))
    return wires


def emit_return(frame: WireFrame, results: tuple[ir.SSAValue, ...]) -> None:
    """Emit a `Return` of `None`, of the single result or of a tuple of the results."""
    if not results:
        packed = frame.push(func.ConstantNone()).results[0]
    elif len(results) == 1:
        packed = results[0]
    else:
        packed = frame.push(py.tuple.New(results)).results[0]
    frame.push(func.Return(packed))


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


def allocate_register(emit: EmitWire, frame: WireFrame, stmt: func.Invoke) -> Location:
    """Emit the register of a `qalloc` call and return its location.

    The library body of `qalloc` builds the register with a list map, which the
    emitter cannot run. This function emits one `AllocRegister` in its place.
    """
    size = emit.classical_value(frame, frame.get(stmt.inputs[0]))
    root = frame.push(stmts.AllocRegister(size)).results[0]
    frame.latest[root] = root
    frame.local_roots.append(root)
    shape = register_shape(stmt.result.type)
    return Location(root, clamped(index_of(size)), shape=shape)


def run_library(emit: EmitWire, frame: WireFrame, stmt: func.Invoke) -> Value:
    """Emit the body of a library wrapper in place of its call and return its result.

    The wrapper emits into the caller's block and continues the caller's wires.
    Its allocations live until the caller's region ends.
    """
    with emit.new_frame(stmt.callee.code) as body:
        body.site = frame.site or stmt
        body.output = frame.output
        body.latest = frame.latest
        body.local_roots = frame.local_roots
        result = emit.frame_call(
            body, stmt.callee.code, None, *frame.get_values(stmt.inputs)
        )
    return result


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
    arguments = frame.get_values(stmt.inputs)
    borrowed = borrowed_operands(arguments, layout, stmt)
    location_at = dict(zip(layout.borrowed_positions, borrowed, strict=True))
    inputs = tuple(
        (
            emit.acquire(frame, location_at[i])
            if i in location_at
            else emit.classical_value(frame, argument)
        )
        for i, argument in enumerate(arguments)
    )
    made = frame.push(func.Invoke(inputs, callee=contract.method))
    made.result.type = contract.method.return_type
    outputs = call_results(frame, made.result, layout)
    count = layout.ordinary_count
    for location, result in zip(borrowed, outputs[count:], strict=True):
        emit.restore(frame, location, result)
    ordinary = ordinary_meanings(emit, frame, contract, arguments, outputs[:count])
    if layout.packed:
        return ordinary
    return ordinary[0] if ordinary else frame.push(func.ConstantNone()).results[0]


def borrowed_operands(
    arguments: tuple[Value, ...], layout: Layout, stmt: func.Invoke
) -> list[Location]:
    """Return the borrowed call operands after checking that they are separate qubits.

    Each operand must be one qubit or one whole register.
    """
    operands = [arguments[i] for i in layout.borrowed_positions]
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
    kinds = layout.result_types(result.type)
    if not kinds:
        return ()
    if len(kinds) == 1:
        return (result,)
    return unpack(frame, result, kinds)


def ordinary_meanings(
    emit: EmitWire,
    frame: WireFrame,
    contract: Contract,
    arguments: tuple[Value, ...],
    results: tuple[ir.SSAValue, ...],
) -> tuple[Value, ...]:
    """Return the source meaning of each ordinary result of a call.

    A classical result means itself. A quantum result becomes a new allocation in
    the caller. A fresh register gets the static length from the contract, with the
    caller's integer arguments in place of the callee's parameters.
    """
    parameters = contract.method.callable_region.blocks[0].args[1:]
    mapping: dict[Sym, Affine] = {
        parameter: index_of(emit.classical_value(frame, argument))
        for parameter, argument in zip(parameters, arguments, strict=True)
        if parameter.type.is_subseteq(types.Int)
    }
    meanings: list[Value] = []
    for result, length in zip(results, contract.lengths, strict=True):
        if result.type not in (WireType, RegisterType):
            meanings.append(result)
            continue
        if length is not None:
            length = clamped(length.substitute(mapping))
        meanings.append(Location(result, length))
        frame.latest[result] = result
        frame.local_roots.append(result)
    return tuple(meanings)


def clamped(length: Affine) -> Affine:
    """Return a negative constant length as zero, like the length of `range(n)`."""
    if (count := length.constant) is not None:
        return Affine(max(0, count))
    return length


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
