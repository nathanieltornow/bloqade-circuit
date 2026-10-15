"""Test affine indices, the alias check, and Python slice semantics."""

from kirin import ir
from kirin.dialects import py

from bloqade.analysis.reference.index import (
    Iter,
    Slice,
    Affine,
    same,
    python_slice,
)

LOOP = Iter(py.Constant(0), range(3))
"""`LOOP` is the variable `i` of a loop that iterates over `range(3)`."""

N = ir.TestValue()
"""`N` is an integer parameter `n` whose range is unknown."""

i, n = Affine.of(LOOP), Affine.of(N)


def test_arithmetic_normalizes():
    assert (i + 1) - i == Affine(1)
    assert i.scale(2) - i - i == Affine(0)
    assert (i + n).substitute({N: Affine(4)}) == i + 4


def test_same_proves_equal_and_disjoint_indices():
    assert same(i, i) is True
    assert same(i, i + 1) is False
    assert same(i.scale(2), i.scale(2) + 1) is False
    assert same(i, Affine(5)) is False
    assert same(i, Affine(1)) is None
    assert same(n, Affine(0)) is None


def test_constant_slices_match_python():
    for size in range(6):
        for a in (None, -7, -2, 0, 1, 3, 9):
            for b in (None, -7, -1, 0, 2, 4, 9):
                for step in (-3, -1, 1, 2):
                    start = None if a is None else Affine(a)
                    stop = None if b is None else Affine(b)
                    got = python_slice(start, stop, step, size)
                    assert got is not None and got.length is not None
                    items = [got.item(Affine(j)).const for j in range(got.length.const)]
                    assert items == list(range(size))[a:b:step], (size, a, b, step)


def test_symbolic_slices():
    pair = python_slice(i, i + 2, 1, 4)
    assert pair == Slice(i, 1, Affine(2))
    assert pair.item(Affine(1)) == i + 1
    assert python_slice(n, None, 1, 4) is None
    assert python_slice(Affine(1), None, 2, None) == Slice(Affine(1), 2, None)


def test_a_slice_of_a_slice_reads_the_register():
    outer = Slice(Affine(2), 1, Affine(4))
    inner = python_slice(Affine(1), None, 1, 4)
    assert inner is not None
    assert outer.then(inner).item(Affine(0)) == Affine(3)
