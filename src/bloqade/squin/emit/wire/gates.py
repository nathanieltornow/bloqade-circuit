"""Emit gate twins on qubits and register twins on whole registers or slices."""

from typing import cast

from kirin import ir, types
from kirin.dialects import ilist

from bloqade.squin.wire.twins import TWINS, REGISTER_TWINS
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import Twin, RegisterTwin

from .base import EmitWire, WireFrame
from .values import View, Value, Location, Register, is_qubit


def apply_twin(
    emit: EmitWire,
    frame: WireFrame,
    source: ir.Statement,
    twin: type[ir.Statement],
    wires: tuple[tuple[str, str], ...],
    locations: tuple[Location, ...],
) -> ir.Statement:
    """Emit `twin` on the operands at `locations` and restore them after it.

    `wires` pairs each qubit-list field of `source` with an operand field of
    `twin`. The classical arguments and attributes of `twin` come from `source`.
    """
    inputs = {
        wire_field: emit.acquire(frame, location)
        for (_, wire_field), location in zip(wires, locations, strict=True)
    }
    fields = getattr(twin, "__kirin_stmt_fields")
    classical = {
        name: emit.classical_value(frame, frame.get(getattr(source, name)))
        for name in fields.std_args
        if name not in inputs
    }
    attributes = {name: getattr(source, name) for name in fields.attributes}
    made = frame.push(twin(**inputs, **classical, **attributes))
    # `make_twin` gives each operand field `x` the result field `x_out`.
    for wire_field, location in zip(inputs, locations, strict=True):
        emit.restore(frame, location, getattr(made, f"{wire_field}_out"))
    return made


def scalar_gate(
    emit: EmitWire,
    frame: WireFrame,
    source: ir.Statement,
    locations: tuple[Location, ...],
) -> tuple[ir.SSAValue, ...]:
    """Emit one wire gate on single qubits and return its classical results."""
    twin = TWINS[type(source)]
    trait = twin.get_present_trait(Twin)
    made = apply_twin(emit, frame, source, twin, trait.wires, locations)
    return tuple(getattr(made, wire_result) for _, wire_result in trait.results)


def register_gate(
    emit: EmitWire,
    frame: WireFrame,
    source: ir.Statement,
    parts: tuple[Register | View, ...],
) -> tuple[ir.SSAValue, ...]:
    """Emit the register twin of `source` and return its classical list results.

    Operands of different known lengths fail, unless the statement stops at the
    shortest register.
    """
    twin = REGISTER_TWINS[type(source)]
    trait = twin.get_present_trait(RegisterTwin)
    counts = [None if part.length is None else part.length.constant for part in parts]
    known = {count for count in counts if count is not None}
    if not trait.truncate and len(known) > 1:
        raise WireConversionError(
            f"'{source.name}' requires qubit lists of equal length.", source
        )
    if trait.truncate:
        size = None if None in counts else min(known)
    else:
        size = next(iter(known), None)
    made = apply_twin(emit, frame, source, twin, trait.wires, parts)
    length = types.Any if size is None else types.Literal(size)
    results: list[ir.SSAValue] = []
    for _, result_field in trait.results:
        value = getattr(made, result_field)
        value.type = ilist.IListType[value.type.vars[0], length]
        results.append(value)
    return tuple(results)


def emit_gate(
    emit: EmitWire, frame: WireFrame, source: ir.Statement
) -> tuple[Value, ...]:
    """Emit a register twin on whole registers and step-1 slices, or scalar twins.

    A register twin takes each operand as its own register, so the operands must
    share no qubit. Overlapping slices, such as `cx(qs[0:2], qs[1:3])`, get one
    scalar twin per group in order, which is the order of the source broadcast.
    Scalar twins need operands whose qubits are all known.
    """
    trait = TWINS[type(source)].get_present_trait(Twin)
    operands = tuple(
        frame.get(getattr(source, source_field)) for source_field, _ in trait.wires
    )
    if all(
        isinstance(value, Register)
        or (isinstance(value, View) and value.span.step == 1)
        for value in operands
    ):
        parts = cast(tuple[Register | View, ...], operands)
        if apart(parts):
            return register_gate(emit, frame, source, parts)
    maybe_items = tuple(emit.known_items(value, source) for value in operands)
    if any(items is None for items in maybe_items):
        if all(
            isinstance(value, Register)
            or (isinstance(value, View) and value.span.step == 1)
            for value in operands
        ):
            raise WireConversionError(
                "A runtime broadcast over register parts that can share qubits has"
                " no wire form.",
                source,
            )
        if not all(isinstance(value, (Register, View, tuple)) for value in operands):
            raise WireConversionError(
                "A runtime broadcast requires resolved register operands.", source
            )
        raise WireConversionError(
            "A runtime broadcast cannot index a heterogeneous quantum container.",
            source,
        )
    operand_items = tuple(items for items in maybe_items if items is not None)
    if not trait.truncate and len({len(items) for items in operand_items}) > 1:
        raise WireConversionError(
            f"'{source.name}' requires qubit lists of equal length.", source
        )
    outputs: list[list[ir.SSAValue]] = [[] for _ in trait.results]
    # Each step takes the next qubit of every operand, like Python's `zip`.
    for qubits in zip(*operand_items):
        if not all(is_qubit(location) for location in qubits):
            raise WireConversionError(
                f"A qubit of '{source.name}' has no place in a known allocation.",
                source,
            )
        values = scalar_gate(emit, frame, source, qubits)
        for collected, value in zip(outputs, values, strict=True):
            collected.append(value)
    return tuple(
        frame.push(
            ilist.New(
                tuple(values), elem_type=getattr(source, source_field).type.vars[0]
            )
        ).results[0]
        for (source_field, _), values in zip(trait.results, outputs, strict=True)
    )


def apart(parts: tuple[Register | View, ...]) -> bool:
    """Check whether the register parts provably share no qubit.

    Parts of different allocations never meet. 2 slices of one allocation are apart
    if one ends before the other starts.
    """
    for i, left in enumerate(parts):
        for right in parts[i + 1 :]:
            if left.root is not right.root:
                continue
            if not (isinstance(left, View) and isinstance(right, View)):
                return False
            a, b = left.span, right.span
            if a.length is None or b.length is None:
                return False
            gaps = (
                (a.start + a.length - b.start).constant,
                (b.start + b.length - a.start).constant,
            )
            if not any(gap is not None and gap <= 0 for gap in gaps):
                return False
    return True
