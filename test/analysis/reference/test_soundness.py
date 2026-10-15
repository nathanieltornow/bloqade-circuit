"""Check the places of the reference analysis against the qubits that PyQrack simulates.

The test runs each kernel in PyQrack and records every value that a frame sets. If
the analysis gives a value a place, the test evaluates that place with the concrete
values of its root and its index symbols. The result must be the qubit or the list
of qubits that the value holds. The test also checks two facts for each pair of
records whose root and symbols hold the same concrete values. If two records have
the same place, they hold the same qubits. If `same` proves two indices different,
the records hold different qubits.
"""

import sys
from typing import Any
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

import pytest
import test_reference as ref_kernels
from kirin import ir, interp
from kirin.dialects import scf, func, ilist

from bloqade import squin
from bloqade.types import Qubit
from bloqade.pyqrack import DynamicMemorySimulator
from bloqade.pyqrack.reg import PyQrackQubit
from bloqade.analysis.reference import (
    Ref,
    Iter,
    Place,
    Slice,
    Affine,
    Returned,
    same,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

# The round-trip kernels live in the package `test.jeff`. Pytest puts the
# repository root on the path only when it collects that package, so a run of
# this file alone needs the root here.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test.jeff import test_roundtrip_squin as trip  # noqa: E402

Addrs = tuple[int, ...]
"""Hold the addresses of the qubits that a value holds, in order."""


def addresses(value: Any) -> Addrs | None:
    """Return the addresses of the qubits that `value` holds, or None.

    A qubit holds itself. A list holds the qubits of its items, in order. Any other
    value holds no qubit and gives None.
    """
    if isinstance(value, PyQrackQubit):
        return (value.addr,)
    if isinstance(value, ilist.IList):
        value = value.data
    if isinstance(value, (list, tuple)) and value:
        parts = [addresses(item) for item in value]
        if all(part is not None for part in parts):
            return tuple(a for part in parts for a in part)  # type: ignore[union-attr]
    return None


def kernels(method: ir.Method) -> list[ir.Method]:
    """Return `method` and every kernel that its calls reach, library kernels too."""
    seen: dict[ir.Method, None] = {}

    def visit(kernel: ir.Method) -> None:
        if kernel in seen:
            return
        seen[kernel] = None
        for node in kernel.callable_region.walk():
            if isinstance(node, func.Invoke):
                visit(node.callee)

    visit(method)
    return list(seen)


@dataclass(frozen=True)
class Record:
    """Hold one value that a PyQrack frame set, with the concrete values of its place.

    `env` maps the root and each index symbol of the place to its concrete value
    at the time of the record. A root maps to the addresses that it holds.
    """

    value: ir.SSAValue
    place: Place
    held: Addrs
    env: tuple[tuple[object, object], ...]

    def agrees(self, other: "Record") -> bool:
        """Return True if the two records give each shared key the same value."""
        mine = dict(self.env)
        return all(mine.get(k, v) == v for k, v in other.env)


def symbol_value(frame: interp.Frame, symbol: object) -> int | None:
    """Return the concrete value of the index symbol `symbol` in `frame`, or None.

    An SSA symbol holds its own value. An `Iter` of an `scf.for` loop holds the
    value of the loop variable.
    """
    if isinstance(symbol, Iter):
        if not isinstance(symbol.stmt, scf.For):
            return None
        symbol = symbol.stmt.body.blocks[0].args[0]
    value = frame.entries.get(symbol)  # type: ignore[arg-type]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def root_value(frame: interp.Frame, root: object) -> Any:
    """Return the concrete value of the root `root` in `frame`, or None."""
    if not isinstance(root, Returned):
        return frame.entries.get(root)  # type: ignore[arg-type]
    results = root.call.results
    position = root.position
    if len(results) == 1:
        value = frame.entries.get(results[0])
        if isinstance(value, tuple) and len(value) > position:
            value = value[position]
        elif position != 0:
            return None
    else:
        value = frame.entries.get(results[position])
    return value


def evaluate(index: Affine, env: dict[object, object]) -> int:
    """Return the concrete value of `index` under the symbol values `env`."""
    return index.const + sum(a * env[s] for s, a in index.terms)  # type: ignore


def place_of(place: Place, root: Any, env: dict[object, object]) -> Any:
    """Return the concrete value that `place` names, given its root value `root`."""
    value = root
    for step in place.path:
        items = list(value.data if isinstance(value, ilist.IList) else value)
        if isinstance(step, Slice):
            start = evaluate(step.start, env)
            count = None if step.length is None else evaluate(step.length, env)
            picked = []
            j = 0
            while count is None or j < count:
                k = start + j * step.step
                if count is None and not 0 <= k < len(items):
                    break
                picked.append(items[k])
                j += 1
            value = picked
        else:
            value = items[evaluate(step, env)]
    return value


def distinct(a: Place, b: Place) -> bool:
    """Return True if `same` proves that the places `a` and `b` share no item.

    The places must share a root. The first pair of steps that `same` does not
    prove equal must be two indices that `same` proves different.
    """
    if a.root != b.root:
        return False
    for x, y in zip(a.path, b.path):
        if isinstance(x, Slice) or isinstance(y, Slice):
            return False
        verdict = same(x, y)
        if verdict is False:
            return True
        if verdict is None:
            return False
    return False


class Checker:
    """Record the places of a run and check each new record against the others."""

    def __init__(self, refs: dict[ir.SSAValue, Ref]) -> None:
        self.refs = refs
        self.records: dict[object, list[Record]] = {}
        self.count = 0

    def observe(self, frame: interp.Frame, key: ir.SSAValue, value: Any) -> None:
        """Check the value `value` that `frame` sets for `key` against its place."""
        place = self.refs.get(key)
        held = addresses(value)
        if not isinstance(place, Place) or held is None:
            return
        root = root_value(frame, place.root)
        symbols = {s for step in place.path for s in _symbols(step)}
        env: dict[object, object] = {s: symbol_value(frame, s) for s in symbols}
        if root is None or addresses(root) is None or None in env.values():
            return
        named = addresses(place_of(place, root, env))
        assert named == held, f"{key} holds {held}, and {place!r} names {named}"
        env[place.root] = addresses(root)
        record = Record(key, place, held, tuple(env.items()))
        group = self.records.setdefault(place.root, [])
        for other in group:
            if not record.agrees(other):
                continue
            if record.place == other.place:
                assert record.held == other.held, (record, other)
            if distinct(record.place, other.place):
                assert not set(record.held) & set(other.held), (record, other)
        group.append(record)
        self.count += 1


def _symbols(step: Affine | Slice) -> frozenset[object]:
    """Return the index symbols that one step of a path uses."""
    if isinstance(step, Slice):
        length = step.length.symbols() if step.length is not None else frozenset()
        return step.start.symbols() | length
    return step.symbols()


@contextmanager
def checking(method: ir.Method) -> Iterator[Checker]:
    """Check every value that PyQrack sets while it runs `method` or its callees."""
    refs: dict[ir.SSAValue, Ref] = {}
    for kernel in kernels(method):
        frame, _ = QubitReferenceAnalysis(kernel.dialects).run(kernel)
        refs.update(frame.entries)
    checker = Checker(refs)
    original = interp.Frame.set_values

    def set_values(self, keys, values):
        keys, values = tuple(keys), tuple(values)
        original(self, keys, values)
        for key, value in zip(keys, values):
            checker.observe(self, key, value)

    interp.Frame.set_values = set_values  # type: ignore[method-assign]
    try:
        yield checker
    finally:
        interp.Frame.set_values = original  # type: ignore[method-assign]


def run_checked(kernel: ir.Method, *args: Any) -> int:
    """Run `kernel` in PyQrack under the checks and return the count of records."""
    with checking(kernel) as checker:
        DynamicMemorySimulator().run(kernel, args=args)
    return checker.count


# -- kernels that call the kernels of the reference tests -----------------------


@squin.kernel
def drive_through():
    q = squin.qubit.new()
    ref_kernels.through(q)


@squin.kernel
def drive_elements():
    qs = squin.qalloc(3)
    ref_kernels.elements(qs, 2)


@squin.kernel
def drive_pick(i: int):
    qs = squin.qalloc(4)
    squin.x(ref_kernels.pick(qs, i))
    squin.x(ref_kernels.pick_next(qs, i))


@squin.kernel
def ladder(qs: ilist.IList[Qubit, Any]):
    for i in range(3):
        squin.cx(qs[i], qs[i + 1])
        squin.cz(qs[2 * i], qs[2 * i + 1])


@squin.kernel
def drive_ladder():
    qs = squin.qalloc(6)
    ladder(qs)


REFERENCE = [
    (drive_through, ()),
    (drive_elements, ()),
    (drive_pick, (0,)),
    (drive_pick, (2,)),
    (drive_ladder, ()),
    (ref_kernels.maker, ()),
    (ref_kernels.twice, ()),
    (ref_kernels.sized, ()),
    (ref_kernels.swaps_in_loop, ()),
    (ref_kernels.rotation, (0.5,)),
    (ref_kernels.reads_twice, ()),
    (ref_kernels.reads_made_pair, ()),
    (ref_kernels.branches, (True,)),
    (ref_kernels.branches, (False,)),
    (ref_kernels.concatenated, ()),
    (ref_kernels.picks, ()),
    (ref_kernels.reads_nested_pair, ()),
]

ROUNDTRIP = [
    (kernel, args) for kernel, args, _ in trip.DETERMINISTIC if kernel is not trip.logic
] + [
    (kernel, ()) for kernel in (trip.bell, trip.ghz, trip.bell_list, trip.ghz_register)
]


def _id(case) -> str:
    kernel, args = case
    return f"{kernel.sym_name}{args}"


@pytest.mark.parametrize("case", REFERENCE + ROUNDTRIP, ids=_id)
def test_places_name_the_simulated_qubits(case):
    kernel, args = case
    run_checked(kernel, *args)


def test_the_harness_sees_places():
    """Check that the ladder kernel gives the harness more than 10 records."""
    assert run_checked(drive_ladder) > 10
