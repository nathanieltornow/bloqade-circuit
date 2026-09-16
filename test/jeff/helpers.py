"""Shared helpers for the jeff tests."""

import tempfile
from pathlib import Path
from contextlib import contextmanager

import pytest
from kirin import types
from kirin.ir.exception import ValidationErrorGroup

from bloqade.jeff import emit_jeff, load_jeff
from bloqade.pyqrack import DynamicMemorySimulator
from bloqade.jeff.dialects import stmts

from .build import add, entry, method


def roundtrips(lowered) -> bool:
    """Check that emit, write, load, emit, write is byte-identical."""
    module = emit_jeff(lowered)
    with tempfile.TemporaryDirectory() as tmp:
        first, second = Path(tmp) / "a.jeff", Path(tmp) / "b.jeff"
        module.write_out(str(first))
        emit_jeff(load_jeff(str(first))).write_out(str(second))
        return first.read_bytes() == second.read_bytes()


SHOTS = 8


@contextmanager
def rejected(message: str):
    """The conversion refuses with a group holding an error that names `message`."""
    with pytest.raises(ValidationErrorGroup) as info:
        yield
    messages = [str(e) for e in info.value.errors]
    assert any(message in m for m in messages), messages


def simulate(kernel, *args):
    return DynamicMemorySimulator().run(kernel, args=args)


def as_python(value):
    """Simulator output as plain Python: bits become ints, lists and tuples
    are converted element-wise."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(as_python(v) for v in value)
    try:
        return [as_python(b) for b in value]
    except TypeError:
        return int(value)


def one_wire_program(fill, name="prog"):
    """A one-qubit jeff program: `fill(block, wire) -> wire`, then measure."""
    block, _ = entry()
    w = add(block, stmts.Alloc()).result
    out = fill(block, w)
    m = add(block, stmts.MeasureNd(out))
    add(block, stmts.Free(m.result_wire))
    return method(block, m.bit, types.Bool, name=name)
