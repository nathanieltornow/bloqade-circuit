"""This module builds one wire statement, a twin, for each squin statement on qubits.

A squin gate, noise channel, measurement or reset takes lists of qubits. Its twin
takes one wire for each list and returns one new wire for each. So the twin of
`squin.gate.CX(controls, targets)` is `CX(control, target)` with the results
`control_out` and `target_out`. Each squin statement also gets a register twin,
such as `BroadcastCX(controls, targets)`, that takes one register for each list.
The converter uses the register twin for a broadcast over whole registers or
contiguous slices, and one scalar twin for each group of qubits otherwise.

The generator reads the fields that kirin's `@statement` records on each squin
class. `TWIN_TABLE` declares each supported source and its quantum field mappings.
The generator checks those mappings against the source schema before registration.
A statement whose qubits come in nested lists, such as `CorrelatedQubitLoss`,
gets no twin.
"""

from dataclasses import dataclass
from collections.abc import Callable

from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.decl.info import ArgumentField, StatementFields

from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import QubitType, holds_qubits
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.noise import stmts as noise_stmts
from bloqade.native.dialects.gate import stmts as native_stmts

from .types import WireType, RegisterType
from .traits import (
    Twin,
    Noise,
    Threads,
    Unitary,
    Measurement,
    Preparation,
    RegisterTwin,
)
from ._dialect import dialect


def _is_qubit_list(kind: types.TypeAttribute) -> bool:
    """Return True if `kind` is a flat list of qubits, such as `IList[Qubit, N]`."""
    return (
        isinstance(kind, types.Generic)
        and kind.is_subseteq(ilist.IListType)
        and kind.vars[0].is_subseteq(QubitType)
        and not isinstance(kind.vars[0], types.BottomType)
    )


@dataclass(frozen=True)
class TwinDefinition:
    """Declare the source fields that supply each wire operand and scalar result."""

    source: type[ir.Statement]
    kind: ir.StmtTrait
    wires: tuple[tuple[str, str], ...] = ()
    results: tuple[tuple[str, str], ...] = ()
    class_name: str | None = None
    operation_name: str | None = None
    truncate: bool = False


ONE_QUBIT = (("qubits", "qubit"),)
CONTROL_TARGET = (("controls", "control"), ("targets", "target"))

TWIN_TABLE = (
    TwinDefinition(gate_stmts.X, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.Y, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.Z, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.H, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.T, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.S, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.SqrtX, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.SqrtY, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.Rx, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.Ry, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.Rz, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.CX, Unitary(), CONTROL_TARGET),
    TwinDefinition(gate_stmts.CY, Unitary(), CONTROL_TARGET),
    TwinDefinition(gate_stmts.CZ, Unitary(), CONTROL_TARGET),
    TwinDefinition(
        gate_stmts.CCZ,
        Unitary(),
        (("controls1", "control1"), ("controls2", "control2"), ("targets", "target")),
    ),
    TwinDefinition(
        gate_stmts.Swap, Unitary(), (("qubits1", "qubit1"), ("qubits2", "qubit2"))
    ),
    TwinDefinition(gate_stmts.U3, Unitary(), ONE_QUBIT),
    TwinDefinition(gate_stmts.PhasedXZ, Unitary(), ONE_QUBIT),
    TwinDefinition(noise_stmts.SingleQubitPauliChannel, Noise(), ONE_QUBIT),
    TwinDefinition(
        noise_stmts.TwoQubitPauliChannel, Noise(), CONTROL_TARGET, truncate=True
    ),
    TwinDefinition(noise_stmts.Depolarize, Noise(), ONE_QUBIT),
    TwinDefinition(noise_stmts.Depolarize2, Noise(), CONTROL_TARGET, truncate=True),
    TwinDefinition(noise_stmts.QubitLoss, Noise(), ONE_QUBIT),
    TwinDefinition(noise_stmts.QubitLeakage, Noise(), ONE_QUBIT),
    TwinDefinition(
        qubit_stmts.Measure, Measurement(), ONE_QUBIT, results=(("result", "result"),)
    ),
    TwinDefinition(qubit_stmts.Reset, Preparation(), ONE_QUBIT),
    TwinDefinition(
        native_stmts.CZ,
        Unitary(),
        CONTROL_TARGET,
        class_name="NativeCZ",
        operation_name="native_cz",
        truncate=True,
    ),
    TwinDefinition(
        native_stmts.R,
        Unitary(),
        ONE_QUBIT,
        class_name="NativeR",
        operation_name="native_r",
    ),
    TwinDefinition(
        native_stmts.Rz,
        Unitary(),
        ONE_QUBIT,
        class_name="NativeRz",
        operation_name="native_rz",
    ),
)
"""Declare the supported twins in registration order.

Native twins have distinct names because their angle fields and units differ
from the squin gates. `CorrelatedQubitLoss` takes nested lists and has no twin.
"""


def _schema(
    definition: TwinDefinition,
) -> tuple[StatementFields, dict[str, ArgumentField]]:
    """Return the kirin fields and the arguments of the source of `definition`.

    The function rejects a definition whose mappings break the source schema.
    """
    source = definition.source
    fields = getattr(source, "__kirin_stmt_fields")
    arguments = {**fields.std_args, **fields.kw_args}
    wires = dict(definition.wires)
    results = dict(definition.results)
    quantum = [name for name, field in arguments.items() if holds_qubits(field.type)]
    if not wires or list(wires) != quantum or len(wires) != len(definition.wires):
        raise ValueError(
            f"The twin of {source.__name__} requires every qubit operand in source order."
        )
    if any(not _is_qubit_list(arguments[name].type) for name in wires):
        raise ValueError(f"The twin of {source.__name__} requires flat qubit lists.")
    if list(results) != list(fields.results) or len(results) != len(definition.results):
        raise ValueError(
            f"The twin of {source.__name__} requires every result in source order."
        )
    for name in results:
        kind = fields.results[name].type
        if (
            not isinstance(kind, types.Generic)
            or not kind.is_subseteq(ilist.IListType)
            or holds_qubits(kind)
        ):
            raise ValueError(
                f"The twin of {source.__name__} requires classical list results."
            )
    names = [wires.get(name, name) for name in arguments]
    names += [f"{wire}_out" for wire in wires.values()]
    names += list(results.values()) + list(fields.attributes)
    if len(names) != len(set(names)):
        raise ValueError(
            f"The twin of {source.__name__} requires distinct field names."
        )
    return fields, arguments


def _build(
    definition: TwinDefinition,
    wires: dict[str, str],
    class_name: str,
    operation_name: str,
    linear: types.TypeAttribute,
    result_type: Callable[[types.TypeAttribute], types.TypeAttribute],
    trait: ir.StmtTrait,
) -> type[ir.Statement]:
    """Build a twin class whose qubit operands have the type `linear`.

    `wires` maps each qubit-list field of the source to an operand field. Each such
    field becomes one `linear` argument and one `linear` result.
    `result_type` maps the element type of each source list result to the type of
    the twin result. Classical arguments and attributes keep their source order.
    """
    fields, arguments = _schema(definition)
    annotations: dict[str, object] = {}
    namespace: dict[str, object] = {"__module__": __name__, "name": operation_name}
    for name, field in arguments.items():
        target = wires.get(name, name)
        annotations[target] = ir.SSAValue
        namespace[target] = info.argument(linear if name in wires else field.type)
    for wire in wires.values():
        annotations[f"{wire}_out"] = ir.ResultValue
        namespace[f"{wire}_out"] = info.result(linear)
    for name, target in definition.results:
        kind = fields.results[name].type
        assert isinstance(kind, types.Generic)
        element = kind.vars[0]
        annotations[target] = ir.ResultValue
        namespace[target] = info.result(result_type(element))
    for name, field in fields.attributes.items():
        annotations[name] = field.annotation
        namespace[name] = info.attribute(default=field.default)
    namespace["__annotations__"] = annotations
    namespace["traits"] = frozenset(
        {
            definition.kind,
            Threads(tuple((wire, f"{wire}_out") for wire in wires.values())),
            trait,
        }
    )
    twin = type(class_name, (ir.Statement,), namespace)
    return statement(dialect=dialect)(twin)


def make_twin(definition: TwinDefinition) -> type[ir.Statement]:
    """Build the twin of `definition` that acts on one wire per qubit list."""
    source = definition.source
    trait = Twin(source, definition.wires, definition.results, definition.truncate)
    return _build(
        definition,
        dict(definition.wires),
        definition.class_name or source.__name__,
        definition.operation_name or source.name,
        WireType,
        lambda element: element,
        trait,
    )


def make_register_twin(
    definition: TwinDefinition, scalar: type[ir.Statement]
) -> type[ir.Statement]:
    """Build the twin of `definition` that acts on one register per qubit list.

    `scalar` is the twin that the statement applies to each register position. Each
    register operand keeps the name of its source field, such as `controls`.
    """
    source = definition.source
    wires = tuple((name, name) for name, _ in definition.wires)
    trait = RegisterTwin(source, scalar, wires, definition.results, definition.truncate)
    return _build(
        definition,
        dict(wires),
        f"Broadcast{scalar.__name__}",
        f"broadcast_{scalar.name}",
        RegisterType,
        lambda element: ilist.IListType[element, types.Any],
        trait,
    )


TWINS: dict[type[ir.Statement], type[ir.Statement]] = {}
"""Map each supported source statement to its declared twin."""

REGISTER_TWINS: dict[type[ir.Statement], type[ir.Statement]] = {}
"""Map each supported source statement to its twin on whole registers."""

for _definition in TWIN_TABLE:
    _twin = make_twin(_definition)
    _register_twin = make_register_twin(_definition, _twin)
    TWINS[_definition.source] = _twin
    REGISTER_TWINS[_definition.source] = _register_twin
    globals()[_twin.__name__] = _twin
    globals()[_register_twin.__name__] = _register_twin


def twin_of(stmt: ir.Statement) -> Twin | None:
    """Return the `Twin` trait of the wire statement `stmt`, or None."""
    return stmt.get_trait(Twin)
