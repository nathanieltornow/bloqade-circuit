"""This module holds the max-plus algebra that the depth analysis uses.

Max-plus algebra replaces addition with `max` and multiplication with `+`. The
depth of a gate is the max over its input depths plus a cost. So every depth in a
program is a max-plus expression of the depths that enter the program.

`None` stands for minus infinity. Minus infinity is the max-plus zero: `max(x, None)`
is `x`, and `x + None` is `None`.

A `Form` is a max-plus expression over variables, such as `max(3, x0 + 2, x1 + 5)`.
A `Matrix` holds the coefficients of several forms over the same variables. For a
matrix `A` and a vector `x`, the product `A ⊗ x` has entry `max_i(A[j][i] + x[i])`
at row `j`.
"""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

Entry = int | None
"""A max-plus number. `None` stands for minus infinity."""

Matrix = list[list[Entry]]
"""A square max-plus matrix. Entry `[j][i]` is the coefficient of input `i` in output `j`."""


def plus(a: Entry, b: Entry) -> Entry:
    """Return the max-plus product of two numbers, which is their sum."""
    return None if a is None or b is None else a + b


def maximum(a: Entry, b: Entry) -> Entry:
    """Return the max-plus sum of two numbers, which is their max."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


@dataclass(frozen=True)
class Form:
    """A max-plus expression `max(const, max_v(v + terms[v]))` over integer variables.

    `terms` holds `(variable, offset)` pairs sorted by variable. A variable that
    does not appear in `terms` has the offset minus infinity. For example,
    `Form(1, ((0, 2),))` is `max(1, x0 + 2)`.
    """

    const: Entry = None
    terms: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def constant(value: Entry) -> "Form":
        """Return the form that holds only the number `value`."""
        return Form(value)

    @staticmethod
    def var(variable: int) -> "Form":
        """Return the form `x + 0` for the variable `x`."""
        return Form(None, ((variable, 0),))

    @staticmethod
    def _build(const: Entry, terms: Mapping[int, int]) -> "Form":
        """Return a form with its terms in variable order."""
        return Form(const, tuple(sorted(terms.items())))

    def join(self, other: "Form") -> "Form":
        """Return `max(self, other)` as one form."""
        terms = dict(self.terms)
        for variable, offset in other.terms:
            terms[variable] = max(terms.get(variable, offset), offset)
        return Form._build(maximum(self.const, other.const), terms)

    def shift(self, amount: int) -> "Form":
        """Return `self + amount`, which adds `amount` to the constant and every offset."""
        return Form(
            plus(self.const, amount),
            tuple((variable, offset + amount) for variable, offset in self.terms),
        )

    def offset(self, variable: int) -> Entry:
        """Return the offset of `variable`, or None if the form does not use it."""
        return dict(self.terms).get(variable)

    def substitute(self, env: Mapping[int, "Form"]) -> "Form":
        """Replace each variable `v` in `env` by the form `env[v]`.

        The method keeps a variable that `env` does not hold. The result is the
        composition of two max-plus maps, so it is again a form.
        """
        result = Form(self.const)
        for variable, offset in self.terms:
            inner = env.get(variable, Form.var(variable))
            result = result.join(inner.shift(offset))
        return result

    def value(self) -> Entry:
        """Return the number that the form holds.

        Raises:
            ValueError: If the form still uses a variable.
        """
        if self.terms:
            raise ValueError(f"the form {self} still uses variables")
        return self.const


def identity(size: int) -> Matrix:
    """Return the max-plus identity: 0 on the diagonal and minus infinity elsewhere."""
    return [[0 if row == col else None for col in range(size)] for row in range(size)]


def multiply(a: Matrix, b: Matrix) -> Matrix:
    """Return the max-plus product `a ⊗ b`.

    Entry `[i][j]` of the product is `max_k(a[i][k] + b[k][j])`.
    """
    size = len(a)
    result: Matrix = [[None] * size for _ in range(size)]
    for i in range(size):
        for k in range(size):
            left = a[i][k]
            if left is None:
                continue
            row = b[k]
            out = result[i]
            for j, right in enumerate(row):
                if right is not None:
                    out[j] = maximum(out[j], left + right)
    return result


def power(matrix: Matrix, n: int) -> Matrix:
    """Return `matrix` multiplied with itself `n` times, by repeated squaring.

    The cost is `O(k^3 log n)` for a `k × k` matrix. `power(matrix, 0)` is the
    identity.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError(f"a matrix power needs n >= 0, got {n}")
    result = identity(len(matrix))
    base = matrix
    while n:
        if n & 1:
            result = multiply(result, base)
        n >>= 1
        if n:
            base = multiply(base, base)
    return result


def affine_power(
    matrix: Matrix, offset: Sequence[Entry], n: int
) -> tuple[Matrix, list[Entry]]:
    """Return the map that sends `x` to `A ⊗ x ⊕ b`, applied `n` times.

    The result is a pair `(A_n, b_n)`.

    `matrix` is `A` and `offset` is `b`. The result sends `x` to `A_n ⊗ x ⊕ b_n`.

    The function appends one constant input to `x` with the value 0. The augmented
    matrix `[[A, b], [-inf, 0]]` then makes the map linear, and its power holds
    `A^n` and `b_n` as blocks.
    """
    size = len(matrix)
    augmented: Matrix = [[*matrix[row], offset[row]] for row in range(size)]
    augmented.append([*([None] * size), 0])
    raised = power(augmented, n)
    return [row[:size] for row in raised[:size]], [row[size] for row in raised[:size]]
