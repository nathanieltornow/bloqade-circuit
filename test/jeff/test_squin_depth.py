"""Test that the squin depth from the per-qubit interpreter equals the jeff depth.

The jeff depth analysis lives in `bloqade.jeff.analysis.depth`. The test skips if
that package is absent. The squin interpreter runs each constant loop once per
value, so it also names a loop index into a register. There the jeff analysis gives
`Unknown`, and the test compares only the kernels with a jeff depth.
"""

import pytest
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.jeff import SquinToJeff
from bloqade.squin.analysis.per_qubit import circuit_depth

from . import test_roundtrip_squin as trip

jeff_depth = pytest.importorskip("bloqade.jeff.analysis.depth").circuit_depth


@squin.kernel
def repeated():
    q = squin.qubit.new()
    p = squin.qubit.new()
    for _ in range(3):
        squin.h(q)
        squin.cx(q, p)
    return squin.measure(p)


def test_a_constant_loop_gives_one_depth_on_both_sides():
    assert circuit_depth(repeated) == jeff_depth(SquinToJeff().emit(repeated)) == 7


KERNELS = sorted(
    {kernel for kernel, _, _ in trip.DETERMINISTIC}
    | {trip.bell, trip.ghz, trip.bell_list, trip.ghz_register, trip.grid_from_rows},
    key=lambda kernel: kernel.sym_name,
)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda kernel: kernel.sym_name)
def test_the_squin_depth_equals_the_jeff_depth(kernel):
    try:
        jeff = jeff_depth(SquinToJeff().emit(kernel))
    except ValidationErrorGroup:
        pytest.skip("squin-to-jeff refuses the kernel")
    squin = circuit_depth(kernel)
    if isinstance(jeff, int):
        assert squin == jeff
