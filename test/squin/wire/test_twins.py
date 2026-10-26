"""Test the wire twins that the generator builds from the squin statements."""

from bloqade.qubit import stmts as qubit_stmts
from bloqade.squin import wire
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.wire import twins
from bloqade.squin.noise import stmts as noise_stmts


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


def test_a_measurement_twin_returns_the_wire_and_one_result():
    assert fields(twins.TWINS[qubit_stmts.Measure])[1] == ["qubit_out", "result"]


def test_each_twin_carries_the_kind_of_its_source():
    assert twins.TWINS[gate_stmts.H].has_trait(wire.Unitary)
    assert twins.TWINS[noise_stmts.Depolarize].has_trait(wire.Noise)
    assert twins.TWINS[qubit_stmts.Measure].has_trait(wire.Measurement)
    assert twins.TWINS[qubit_stmts.Reset].has_trait(wire.Preparation)


def test_every_flat_qubit_statement_gets_a_twin():
    for module in (gate_stmts, noise_stmts):
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == module.__name__:
                if getattr(value, "dialect", None) is None:
                    continue
                if value is noise_stmts.CorrelatedQubitLoss:
                    continue
                assert value in twins.TWINS, value.__name__


def test_a_statement_on_nested_qubit_lists_gets_no_twin():
    assert noise_stmts.CorrelatedQubitLoss not in twins.TWINS


def test_a_twin_links_back_to_its_source():
    trait = twins.TWINS[gate_stmts.CX].get_present_trait(wire.Twin)
    assert trait.source is gate_stmts.CX
    assert trait.wires == (("controls", "control"), ("targets", "target"))
