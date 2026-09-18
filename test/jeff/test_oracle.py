"""An independent oracle: a reference interpreter of jeff dialect IR,
written from the schema, against pyqrack on what the conversion produces.

Random jeff programs from `jeff_generator` are interpreted, and the values
compared with pyqrack on the converted kernel, and again after a trip through
the file format. Programs whose measurements the interpreter finds
nondeterministic, or whose arithmetic leaves the defined range, are
skipped; a refusal must be one of the constructs squin has no form for;
anything else is a defect.
"""

import math
import random
import tempfile
from pathlib import Path

import pytest
from kirin.dialects.ilist import IList

from bloqade.jeff import Refusal, JeffToSquin, emit_jeff, load_jeff
from bloqade.pyqrack import DynamicMemorySimulator

from .build import validate
from .jeff_generator import Gen, Refuse
from .jeff_interpreter import (
    Interp,
    Overflow,
    Nonfinite,
    Undefined,
    Nondeterministic,
)

COUNT = 40
SEED = 3000
ALLOWED_REFUSALS = (
    "squin's math dialect has no acosh",
    "has no squin statement",
    "have no unsigned form",
)


def _plain(value):
    """The interpreter's and the simulator's results as plain Python:
    bits and measurement results become ints, lists element-wise."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, IList):
        return [_plain(v) for v in value.data]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return int(value)


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _interpret(mt, args):
    """The reference values, or None if the program is outside what the
    interpreter compares."""
    try:
        values = Interp().run(mt, args)
    except (Nondeterministic, Nonfinite, Undefined, Overflow):
        return None
    return _plain(values[0] if len(values) == 1 else values)


def programs():
    found = []
    for seed in range(SEED, SEED + COUNT):
        generator = Gen(random.Random(seed))
        try:
            found.append((seed, *generator.program()))
        except Refuse:
            continue
    return found


@pytest.mark.parametrize("seed, program, args", programs(), ids=lambda x: str(x))
def test_conversion_agrees_with_the_interpreter(seed, program, args):
    validate(program)
    expected = _interpret(program, args)
    if expected is None:
        pytest.skip("outside the interpreter's deterministic range")
    try:
        converted = JeffToSquin().emit(program)
    except Refusal as refusal:
        for error in refusal.errors:
            assert any(reason in str(error) for reason in ALLOWED_REFUSALS), str(error)
        return
    simulator = DynamicMemorySimulator()
    assert _same(expected, _plain(simulator.run(converted, args=args)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.jeff"
        emit_jeff(program).write_out(str(path))
        reloaded = load_jeff(str(path))
    assert _same(
        expected, _plain(simulator.run(JeffToSquin().emit(reloaded), args=args))
    )
