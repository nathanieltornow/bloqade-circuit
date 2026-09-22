"""Tests that a generated jeff program survives the trip to squin and back.

Each program goes jeff to squin with `JeffToSquin`, squin to jeff with
`SquinToJeff`, and the jeff reference interpreter compares the result with the
original. A program that squin cannot express is skipped, with the reason that
the oracle allows.
"""

import pytest

from bloqade.jeff import Refusal, JeffToSquin, SquinToJeff

from .build import validate
from .test_oracle import ALLOWED_REFUSALS, _same, programs, _interpret

SURGERY = (
    "a value carried by a loop or branch",
    "a list or tuple read at a runtime index",
    "jeff has no form for 'slice'",
    "a constant of type slice",
    "a slice of a register",
    "a concatenation of registers",
    "a function that returns one qubit of a register",
    "a root that the callee owns",
    "a literal list passed to a call",
    "takes one qubit twice",
)
"""The refusals of a squin form that takes a register apart or builds one.

Squin has no register, so a jeff insert into a register or a join of two
registers becomes a list built from slices and sums, a wire extracted from a
register becomes one item of a list, and a register built from wires becomes
a literal list. Jeff cannot express these again.
"""


@pytest.mark.parametrize("seed, program, args", programs(), ids=lambda x: str(x))
def test_the_trip_to_squin_and_back_keeps_the_result(seed, program, args):
    expected = _interpret(program, args)
    if expected is None:
        pytest.skip("outside the interpreter's deterministic range")
    try:
        kernel = JeffToSquin().emit(program)
    except Refusal as refusal:
        for error in refusal.errors:
            assert any(reason in str(error) for reason in ALLOWED_REFUSALS), str(error)
        pytest.skip("squin cannot express the program")
    try:
        back = SquinToJeff().emit(kernel)
    except Refusal as refusal:
        for error in refusal.errors:
            assert any(reason in str(error) for reason in SURGERY), str(error)
        pytest.skip("the squin form takes a register apart, which jeff cannot express")
    validate(back)
    assert _same(expected, _interpret(back, args))
