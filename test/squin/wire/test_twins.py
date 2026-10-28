"""Test the wire twins that the generator builds from the squin statements."""

import pytest

from bloqade.qubit import stmts as qubit_stmts
from bloqade.squin import wire
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.wire import twins
from bloqade.squin.noise import stmts as noise_stmts
from bloqade.native.dialects.gate import stmts as native_stmts


def fields(cls) -> tuple[list[str], list[str], list[str]]:
    """Return the argument, result and attribute names of a kirin statement class."""
    declared = getattr(cls, "__kirin_stmt_fields")
    return list(declared.std_args), list(declared.results), list(declared.attributes)


def test_a_controlled_gate_gets_one_wire_per_qubit_list():
    assert fields(twins.TWINS[gate_stmts.CX]) == (
        ["control", "target"],
        ["control_out", "target_out"],
        [],
    )


def test_a_twin_keeps_the_classical_arguments_and_attributes_in_order():
    assert fields(twins.TWINS[gate_stmts.Rz])[0] == ["angle", "qubit"]
    assert fields(twins.TWINS[gate_stmts.T])[2] == ["adjoint"]


def test_two_qubit_noise_keeps_its_probabilities_as_a_classical_argument():
    twin = twins.TWINS[noise_stmts.TwoQubitPauliChannel]
    assert fields(twin)[0] == ["probabilities", "control", "target"]
    assert twin.get_present_trait(wire.Twin).wires == (
        ("controls", "control"),
        ("targets", "target"),
    )


def test_a_measurement_twin_returns_the_wire_and_one_result():
    assert fields(twins.TWINS[qubit_stmts.Measure])[1] == ["qubit_out", "result"]


def test_each_twin_carries_the_kind_of_its_source():
    assert twins.TWINS[gate_stmts.H].has_trait(wire.Unitary)
    assert twins.TWINS[noise_stmts.Depolarize].has_trait(wire.Noise)
    assert twins.TWINS[qubit_stmts.Measure].has_trait(wire.Measurement)
    assert twins.TWINS[qubit_stmts.Reset].has_trait(wire.Preparation)


def test_every_flat_qubit_statement_gets_a_twin():
    for module in (gate_stmts, noise_stmts, native_stmts):
        for value in module.dialect.stmts:
            if value is noise_stmts.CorrelatedQubitLoss:
                continue
            assert value in twins.TWINS, value.__name__


def test_a_statement_on_nested_qubit_lists_gets_no_twin():
    assert noise_stmts.CorrelatedQubitLoss not in twins.TWINS


def test_a_twin_links_back_to_its_source():
    trait = twins.TWINS[gate_stmts.CX].get_present_trait(wire.Twin)
    assert trait.source is gate_stmts.CX
    assert trait.wires == (("controls", "control"), ("targets", "target"))


def test_the_table_has_unique_sources_and_twin_names():
    assert len(twins.TWIN_TABLE) == len(twins.TWINS)
    assert len({cls.__name__ for cls in twins.TWINS.values()}) == len(twins.TWINS)
    assert len({cls.name for cls in twins.TWINS.values()}) == len(twins.TWINS)


def test_native_twins_keep_their_angle_fields_and_source_classes():
    assert fields(twins.NativeR)[0] == ["axis_angle", "rotation_angle", "qubit"]
    assert fields(twins.NativeRz)[0] == ["rotation_angle", "qubit"]
    assert twins.NativeCZ is twins.TWINS[native_stmts.CZ]
    assert twins.NativeCZ is not twins.TWINS[gate_stmts.CZ]
    for source in (native_stmts.CZ, native_stmts.R, native_stmts.Rz):
        assert twins.TWINS[source].has_trait(wire.Unitary)
        assert twins.TWINS[source].get_present_trait(wire.Twin).source is source


@pytest.mark.parametrize("source", [qubit_stmts.New, noise_stmts.CorrelatedQubitLoss])
def test_the_generator_rejects_an_unsupported_table_entry(source):
    with pytest.raises(ValueError, match="requires"):
        twins.make_twin(twins.TwinDefinition(source, wire.Noise()))


@pytest.mark.parametrize(
    "definition, reason",
    [
        (
            twins.TwinDefinition(
                gate_stmts.CX, wire.Unitary(), (("controls", "control"),)
            ),
            "every qubit operand",
        ),
        (
            twins.TwinDefinition(
                gate_stmts.Rz, wire.Unitary(), (("angle", "angle"), ("qubits", "qubit"))
            ),
            "every qubit operand",
        ),
        (
            twins.TwinDefinition(gate_stmts.Rz, wire.Unitary(), (("qubits", "angle"),)),
            "distinct field names",
        ),
        (
            twins.TwinDefinition(
                qubit_stmts.Measure, wire.Measurement(), twins.ONE_QUBIT
            ),
            "every result",
        ),
        (
            twins.TwinDefinition(
                noise_stmts.CorrelatedQubitLoss, wire.Noise(), twins.ONE_QUBIT
            ),
            "flat qubit lists",
        ),
    ],
)
def test_the_generator_checks_each_declared_field_before_registration(
    definition, reason
):
    with pytest.raises(ValueError, match=reason):
        twins.make_twin(definition)
