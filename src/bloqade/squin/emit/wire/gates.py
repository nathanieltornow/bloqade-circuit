"""Emit gate twins and runtime broadcasts with explicit register borrows."""

from kirin import ir, types
from kirin.dialects import py, scf, ilist
from kirin.dialects.py import assertion
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.analysis.indices import Affine
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import Twin

from .base import EmitWire, WireFrame
from .values import Location
from .control import region, resume, carry_wires, yield_wires


def scalar_gate(
    emit: EmitWire,
    frame: WireFrame,
    source: ir.Statement,
    locations: tuple[Location, ...],
):
    """Emit one wire gate and restore its operands to their source locations."""
    twin = TWINS[type(source)]
    trait = twin.get_present_trait(Twin)
    inputs = {
        wire_field: emit.acquire(frame, location)
        for (_, wire_field), location in zip(trait.wires, locations, strict=True)
    }
    fields = getattr(twin, "__kirin_stmt_fields")
    classical = {
        name: emit.classical_value(frame, frame.get(getattr(source, name)))
        for name in fields.std_args
        if name not in inputs
    }
    attributes = {name: getattr(source, name) for name in fields.attributes}
    made = frame.push(twin(**inputs, **classical, **attributes))
    # `make_twin` gives each wire field `x` the result field `x_out`.
    for wire_field, location in zip(inputs, locations, strict=True):
        emit.restore(frame, location, getattr(made, f"{wire_field}_out"))
    return tuple(getattr(made, wire_result) for _, wire_result in trait.results)


def emit_gate(emit: EmitWire, frame: WireFrame, source: ir.Statement):
    """Expand known broadcast operands or emit one loop for runtime lengths."""
    trait = TWINS[type(source)].get_present_trait(Twin)
    operands = tuple(frame.get(getattr(source, field)) for field, _ in trait.wires)
    groups = tuple(emit.known_items(value, source) for value in operands)
    if any(group is None for group in groups):
        return broadcast_loop(emit, frame, source, operands)
    known = tuple(group for group in groups if group is not None)
    if not trait.truncate and len({len(group) for group in known}) > 1:
        raise WireConversionError(
            f"'{source.name}' requires qubit lists of equal length.", source
        )
    outputs = [[] for _ in trait.results]
    for group in zip(*known):
        if not all(
            isinstance(location, Location) and location.scalar for location in group
        ):
            raise WireConversionError(
                f"A qubit of '{source.name}' has no place in a known allocation.",
                source,
            )
        values = scalar_gate(emit, frame, source, group)
        for collected, value in zip(outputs, values, strict=True):
            collected.append(value)
    return tuple(
        frame.push(
            ilist.New(tuple(values), elem_type=getattr(source, field).type.vars[0])
        ).results[0]
        for (field, _), values in zip(trait.results, outputs, strict=True)
    )


def broadcast_loop(emit: EmitWire, frame: WireFrame, source: ir.Statement, operands):
    """Emit indexed gate applications and collect classical results in loop-carried lists."""
    trait = TWINS[type(source)].get_present_trait(Twin)
    if not all(
        isinstance(value, Location) and not value.scalar or isinstance(value, tuple)
        for value in operands
    ):
        raise WireConversionError(
            "A runtime broadcast requires resolved register operands.", source
        )
    lengths = tuple(emit.length_value(frame, value) for value in operands)
    size = lengths[0]
    for other in lengths[1:]:
        if trait.truncate:
            condition = frame.push(py.cmp.Lt(size, other)).results[0]
            left, right = ir.Block(), ir.Block()
            left.args.append_from(types.Bool)
            right.args.append_from(types.Bool)
            left.stmts.append(scf.Yield(size))
            right.stmts.append(scf.Yield(other))
            size = frame.push(
                scf.IfElse(condition, ir.Region(left), ir.Region(right))
            ).results[0]
        else:
            equal = frame.push(py.cmp.Eq(size, other)).results[0]
            message = frame.push(
                py.Constant(f"'{source.name}' requires equal qubit-list lengths.")
            ).results[0]
            frame.push(assertion.Assert(equal, message))
    elements = tuple(getattr(source, field).type.vars[0] for field, _ in trait.results)
    empty = tuple(
        frame.push(ilist.New((), elem_type=element)).results[0] for element in elements
    )
    with region(emit, frame, source) as body:
        index = body.output.args.append_from(types.Int, "i")
        lists = tuple(
            body.output.args.append_from(ilist.IListType[element, types.Int])
            for element in elements
        )
        roots = carry_wires(frame, body)
        incoming = tuple(frame.latest[root] for root in roots)
        locations = []
        for operand in operands:
            if not isinstance(operand, Location):
                raise WireConversionError(
                    "A runtime broadcast cannot index a heterogeneous quantum container.",
                    source,
                )
            locations.append(operand.select(Affine.of(index), source))
        values = scalar_gate(emit, body, source, tuple(locations))
        collected = []
        for previous, value in zip(lists, values, strict=True):
            singleton = body.push(ilist.New((value,), elem_type=value.type)).results[0]
            appended = body.push(py.Add(previous, singleton))
            appended.results[0].type = previous.type
            collected.append(appended.results[0])
        yield_wires(body, roots, collected)
    start = frame.push(py.Constant(0)).results[0]
    step = frame.push(py.Constant(1)).results[0]
    iterable = frame.push(ilist_stmts.Range(start, size, step)).results[0]
    made = frame.push(scf.For(iterable, ir.Region(body.output), *empty, *incoming))
    return resume(frame, made, roots, empty, range(len(empty)))
