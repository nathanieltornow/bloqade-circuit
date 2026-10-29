"""Check direct wire emission through Kirin's region interpreter."""

from collections import Counter

import pytest
from test_roundtrip import nested, simulate

from bloqade import squin
from bloqade.squin.emit.wire import EmitWire
from bloqade.squin.passes.squin_to_wire import SquinToWire, remaining
from bloqade.squin.passes.wire_to_squin import WireToSquin
from bloqade.squin.analysis.validation.wire import WireValidation


def test_emission_visits_each_source_statement_once_and_preserves_source(monkeypatch):
    source = nested
    before = source.code.print_str()
    nodes = tuple(source.code.walk())
    uses = {value: set(value.uses) for node in nodes for value in node.results}
    visits = Counter()
    evaluate = EmitWire.frame_eval

    def record(emitter, frame, node):
        visits[node] += 1
        return evaluate(emitter, frame, node)

    monkeypatch.setattr(EmitWire, "frame_eval", record)
    emitter = EmitWire(source.dialects, source, {})
    for _ in range(2):
        visits.clear()
        wired = emitter.emit_method().method
        assert all(visits[node] == 1 for node in nodes)
        assert source.code.print_str() == before
        assert all(set(value.uses) == previous for value, previous in uses.items())
        assert not remaining(wired)
        WireValidation().check(wired)
        reconstructed = WireToSquin().emit(wired)
        assert simulate(reconstructed, 1, True) == [1, 1, 1, 0]
        assert simulate(reconstructed, 2, False) == [0, 0, 0, 0]


@squin.kernel
def allocated_length(n: int):
    qs = squin.qalloc(n)
    return len(qs)


@squin.kernel
def fresh_product(n: int):
    return squin.qalloc(n * n)


@squin.kernel
def uses_fresh_product(n: int):
    qs = fresh_product(n)
    squin.broadcast.x(qs)
    return squin.broadcast.measure(qs)


@pytest.mark.parametrize("size", [-2, 0, 3])
def test_runtime_length_uses_the_allocated_register(size):
    wired = SquinToWire().emit(allocated_length)
    reconstructed = WireToSquin().emit(wired)
    assert (
        simulate(reconstructed, size)
        == simulate(allocated_length, size)
        == max(0, size)
    )


@pytest.mark.parametrize("size", [0, 1, 2])
def test_fresh_result_lengths_keep_callee_local_values_inside_the_callee(size):
    wired = SquinToWire().emit(uses_fresh_product)
    reconstructed = WireToSquin().emit(wired)
    assert (
        simulate(reconstructed, size)
        == simulate(uses_fresh_product, size)
        == [1] * (size * size)
    )


@squin.kernel
def returns_tuple_from_branch(flag: bool):
    values = (1, 2)
    if flag:
        values = (3, 4)
    return values


@pytest.mark.parametrize("flag, expected", [(False, [1, 2]), (True, [3, 4])])
def test_return_unpacks_classical_tuple_results_from_regions(flag, expected):
    wired = SquinToWire().emit(returns_tuple_from_branch)
    assert simulate(WireToSquin().emit(wired), flag) == expected


@squin.kernel
def returns_typed_tuple_from_branch(flag: bool, n: int) -> tuple[int, int]:
    values = (n, n + 1)
    if flag:
        values = (n + 2, n + 3)
    return values


@pytest.mark.parametrize("flag, expected", [(False, [5, 6]), (True, [7, 8])])
def test_return_unpacks_known_tuple_fields_from_regions(flag, expected):
    wired = SquinToWire().emit(returns_typed_tuple_from_branch)
    assert simulate(WireToSquin().emit(wired), flag, 5) == expected
