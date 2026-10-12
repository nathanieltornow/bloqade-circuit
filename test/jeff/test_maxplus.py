"""Test the max-plus matrix power and forms that the depth analysis uses."""

import random

import pytest

from bloqade.jeff.analysis.depth import maxplus
from bloqade.jeff.analysis.depth.maxplus import Form


def naive_power(matrix, n):
    result = maxplus.identity(len(matrix))
    for _ in range(n):
        result = maxplus.multiply(result, matrix)
    return result


def random_matrix(rng, size):
    return [
        [None if rng.random() < 0.3 else rng.randint(0, 4) for _ in range(size)]
        for _ in range(size)
    ]


@pytest.mark.parametrize("seed", range(10))
def test_power_by_squaring_equals_repeated_multiplication(seed):
    rng = random.Random(seed)
    matrix = random_matrix(rng, rng.randint(1, 5))
    for n in range(21):
        assert maxplus.power(matrix, n) == naive_power(matrix, n)


def test_affine_power_zero_is_the_identity_map():
    matrix, offset = maxplus.affine_power([[1, None], [2, 3]], [5, None], 0)
    assert matrix == maxplus.identity(2)
    assert offset == [None, None]


def test_affine_power_applies_the_map_n_times():
    matrix = [[None, 0], [1, None]]
    offset = [None, 2]
    for n in range(8):
        x = [0, 0]
        for _ in range(n):
            x = [
                maxplus.maximum(
                    offset[j],
                    max(
                        (maxplus.plus(matrix[j][i], x[i]) for i in range(2)),
                        key=lambda e: -1 if e is None else e,
                    ),
                )
                for j in range(2)
            ]
        raised, raised_offset = maxplus.affine_power(matrix, offset, n)
        expected = [
            maxplus.maximum(
                raised_offset[j],
                maxplus.maximum(raised[j][0], raised[j][1]),
            )
            for j in range(2)
        ]
        assert x == expected


def test_a_form_substitutes_forms_for_its_variables():
    form = Form(1, ((0, 2), (1, 5)))
    result = form.substitute({0: Form(3), 1: Form(None, ((7, 1),))})
    assert result == Form(5, ((7, 6),))
