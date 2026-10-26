"""This module builds one wire statement, a twin, for each squin statement on qubits.

A squin gate, noise channel, measurement or reset takes lists of qubits. Its twin
takes one wire for each list and returns one new wire for each. So the twin of
`squin.gate.CX(controls, targets)` is `CX(control, target)` with the results
`control_out` and `target_out`. The converter applies one twin to each group of
qubits, so a broadcast becomes several twins.

The generator reads the fields that kirin's `@statement` records on each squin
class. A new squin statement on qubits therefore gets a twin without new code.
A statement whose qubits come in nested lists, such as `CorrelatedQubitLoss`,
gets no twin.
"""

import re

from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import QubitType
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.noise import stmts as noise_stmts

from .types import WireType
from .traits import Twin, Noise, Threads, Unitary, Measurement, Preparation
from ._dialect import dialect


def _is_qubit_list(kind: types.TypeAttribute) -> bool:
    """Return True if `kind` is a flat list of qubits, such as `IList[Qubit, N]`."""
    return (
        isinstance(kind, types.Generic)
        and kind.is_subseteq(ilist.IListType)
        and kind.vars[0].is_subseteq(QubitType)
        and not isinstance(kind.vars[0], types.BottomType)
    )


def _singular(name: str) -> str:
    """Return the field name for one qubit of the list field `name`.

    For example, `qubits` gives `qubit`, and `controls1` gives `control1`.
    """
    return re.sub(r"s(\d*)$", r"\1", name)


def _kind(source: type[ir.Statement]) -> ir.StmtTrait | None:
    """Return the kind trait of the twin of `source`, or None if it has no twin."""
    if issubclass(source, gate_stmts.Gate):
        return Unitary()
    if issubclass(source, noise_stmts.NoiseChannel):
        return Noise()
    if source is qubit_stmts.Measure:
        return Measurement()
    if source is qubit_stmts.Reset:
        return Preparation()
    return None


def make_twin(source: type[ir.Statement]) -> type[ir.Statement] | None:
    """Return a new wire statement class that copies `source`, or None.

    The twin keeps the classical arguments and the attributes of `source` in their
    order. Each qubit-list argument becomes one wire argument and one wire result.
    Each list result becomes a result of its element type. The method returns None
    if `source` has no kind or a qubit argument that is no flat list of qubits.
    """
    kind = _kind(source)
    if kind is None:
        return None
    fields = getattr(source, "__kirin_stmt_fields")
    annotations: dict[str, object] = {}
    namespace: dict[str, object] = {"__module__": __name__, "name": source.name}
    wires: list[tuple[str, str]] = []
    for name, field in {**fields.std_args, **fields.kw_args}.items():
        if _is_qubit_list(field.type):
            wire = _singular(name)
            wires.append((name, wire))
            annotations[wire] = ir.SSAValue
            namespace[wire] = info.argument(WireType)
        elif field.type.is_subseteq(QubitType) or "Qubit" in repr(field.type):
            return None
        else:
            annotations[name] = ir.SSAValue
            namespace[name] = info.argument(field.type)
    if not wires:
        return None
    for _, wire in wires:
        annotations[f"{wire}_out"] = ir.ResultValue
        namespace[f"{wire}_out"] = info.result(WireType)
    results: list[tuple[str, str]] = []
    for name, field in fields.results.items():
        kind_of = field.type
        if isinstance(kind_of, types.Generic) and kind_of.is_subseteq(ilist.IListType):
            kind_of = kind_of.vars[0]
            results.append((name, name))
        annotations[name] = ir.ResultValue
        namespace[name] = info.result(kind_of)
    for name, field in fields.attributes.items():
        annotations[name] = field.annotation
        namespace[name] = info.attribute(default=field.default)
    namespace["__annotations__"] = annotations
    namespace["traits"] = frozenset(
        {
            kind,
            Threads(tuple((wire, f"{wire}_out") for _, wire in wires)),
            Twin(source, tuple(wires), tuple(results)),
        }
    )
    twin = type(source.__name__, (ir.Statement,), namespace)
    return statement(dialect=dialect)(twin)


def _sources() -> list[type[ir.Statement]]:
    """Return the squin statement classes that may have a twin, in a stable order."""
    found: list[type[ir.Statement]] = []
    for module in (gate_stmts, noise_stmts, qubit_stmts):
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, ir.Statement)
                and value.__module__ == module.__name__
                and value.dialect is not None
            ):
                found.append(value)
    return found


TWINS: dict[type[ir.Statement], type[ir.Statement]] = {}
"""Map each squin statement class on qubits to its twin class."""

for _source in _sources():
    if (_twin := make_twin(_source)) is not None:
        TWINS[_source] = _twin
        globals()[_source.__name__] = _twin


def twin_of(stmt: ir.Statement) -> Twin | None:
    """Return the `Twin` trait of the wire statement `stmt`, or None."""
    return stmt.get_trait(Twin)
