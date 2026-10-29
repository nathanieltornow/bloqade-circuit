"""Emit gate twins and runtime broadcasts with explicit register borrows."""

from kirin import ir, types
from kirin.dialects import py, scf, func, ilist
from kirin.dialects.py import assertion
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.analysis.indices import Affine
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import Twin

from .base import EmitWire, WireFrame
from .values import Location


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
        name: emit.take(frame, location)
        for (_, name), location in zip(trait.wires, locations, strict=True)
    }
    fields = getattr(twin, "__kirin_stmt_fields")
    quantum = {name for _, name in trait.wires}
    classical = {
        name: emit.classical(frame, frame.get(getattr(source, name)))
        for name in fields.std_args
        if name not in quantum
    }
    attributes = {name: getattr(source, name) for name in fields.attributes}
    made = frame.push(twin(**inputs, **classical, **attributes))
    for (_, name), location in zip(trait.wires, locations, strict=True):
        emit.restore(frame, location, getattr(made, f"{name}_out"))
    return tuple(getattr(made, result) for _, result in trait.results)


def emit_gate(emit: EmitWire, frame: WireFrame, source: ir.Statement):
    """Expand known broadcast operands or emit one loop for runtime lengths."""
    trait = TWINS[type(source)].get_present_trait(Twin)
    operands = tuple(frame.get(getattr(source, name)) for name, _ in trait.wires)
    groups = tuple(emit.items(value, source) for value in operands)
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
            ilist.New(tuple(values), elem_type=getattr(source, name).type.vars[0])
        ).results[0]
        for (name, _), values in zip(trait.results, outputs, strict=True)
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
    lengths = tuple(emit.length(frame, value) for value in operands)
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
    roots = tuple(frame.current)
    incoming = tuple(frame.current[root] for root in roots)
    empty = tuple(
        frame.push(ilist.New((), elem_type=getattr(source, name).type.vars[0])).results[
            0
        ]
        for name, _ in trait.results
    )
    context = frame.code if isinstance(frame.code, func.Invoke) else source
    with emit.new_frame(context, has_parent_access=True) as body:
        body.current_stmt = source
        index = body.output.args.append_from(types.Int, "i")
        lists = tuple(
            body.output.args.append_from(
                ilist.IListType[getattr(source, name).type.vars[0], types.Int]
            )
            for name, _ in trait.results
        )
        wires = tuple(body.output.args.append_from(value.type) for value in incoming)
        body.current = dict(zip(roots, wires, strict=True))
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
        body.push(scf.Yield(*collected, *(body.current[root] for root in roots)))
    start = frame.push(py.Constant(0)).results[0]
    step = frame.push(py.Constant(1)).results[0]
    iterable = frame.push(ilist_stmts.Range(start, size, step)).results[0]
    made = frame.push(scf.For(iterable, ir.Region(body.output), *empty, *incoming))
    frame.current.update(zip(roots, made.results[len(empty) :], strict=True))
    return tuple(made.results[: len(empty)])
