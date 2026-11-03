"""Emit scalar wire operations on qubits and register operations on whole registers or slices."""

from typing import cast

from kirin import ir, types
from kirin.dialects import ilist

from bloqade.squin.wire.stmts import Operation
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.sources import SOURCES, Source

from .base import EmitWire, WireFrame
from .values import View, Value, Location, Register, is_qubit


def apply(
    emit: EmitWire,
    frame: WireFrame,
    source: ir.Statement,
    locations: tuple[Location, ...],
    register: bool,
) -> Operation:
    """Emit the wire form of `source` on the operands at `locations` and restore them.

    If `register` is True, each location is a register part and the statement acts on
    whole registers. Otherwise each location is one qubit.
    """
    entry = SOURCES[type(source)]
    inputs = tuple(emit.acquire(frame, location) for location in locations)
    params = tuple(
        emit.classical_value(frame, frame.get(value))
        for value in entry.param_operands(source)
    )
    build = entry.register if register else entry.scalar
    made = frame.push(build(inputs, params, entry.adjoint(source)))
    for location, output in zip(locations, made.outputs, strict=True):
        emit.restore(frame, location, output)
    return made


def register_gate(
    emit: EmitWire,
    frame: WireFrame,
    source: ir.Statement,
    parts: tuple[Register | View, ...],
) -> tuple[ir.SSAValue, ...]:
    """Emit the register form of `source` and return its classical list results.

    Operands of different known lengths fail, unless the statement stops at the
    shortest register.
    """
    truncate = SOURCES[type(source)].truncates
    counts = [None if part.length is None else part.length.constant for part in parts]
    known = {count for count in counts if count is not None}
    if not truncate and len(known) > 1:
        raise WireConversionError(
            f"'{source.name}' requires qubit lists of equal length.", source
        )
    if truncate:
        size = None if None in counts else min(known)
    else:
        size = next(iter(known), None)
    made = apply(emit, frame, source, parts, register=True)
    length = types.Any if size is None else types.Literal(size)
    for value in made.classical:
        value.type = ilist.IListType[element_type(value.type), length]
    return made.classical


def emit_gate(
    emit: EmitWire, frame: WireFrame, source: ir.Statement
) -> tuple[Value, ...]:
    """Emit one register operation on whole registers and step-1 slices, or scalar ones.

    A register operation takes each operand as its own register, so the operands
    must share no qubit. Overlapping slices, such as `cx(qs[0:2], qs[1:3])`, get one
    scalar operation per group in order, which is the order of the source broadcast.
    Scalar operations need operands whose qubits are all known.
    """
    entry: Source = SOURCES[type(source)]
    operands = tuple(frame.get(value) for value in entry.qubit_operands(source))
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
    if not entry.truncates and len({len(items) for items in operand_items}) > 1:
        raise WireConversionError(
            f"'{source.name}' requires qubit lists of equal length.", source
        )
    outputs: list[list[ir.SSAValue]] = [[] for _ in entry.results]
    # Each step takes the next qubit of every operand, like Python's `zip`.
    for qubits in zip(*operand_items):
        if not all(is_qubit(location) for location in qubits):
            raise WireConversionError(
                f"A qubit of '{source.name}' has no place in a known allocation.",
                source,
            )
        made = apply(emit, frame, source, qubits, register=False)
        for collected, value in zip(outputs, made.classical, strict=True):
            collected.append(value)
    return tuple(
        frame.push(
            ilist.New(tuple(values), elem_type=element_type(result.type))
        ).results[0]
        for result, values in zip(entry.result_values(source), outputs, strict=True)
    )


def element_type(kind: types.TypeAttribute) -> types.TypeAttribute:
    """Return the item type of the list type `kind`."""
    assert isinstance(kind, types.Generic)
    return kind.vars[0]


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
