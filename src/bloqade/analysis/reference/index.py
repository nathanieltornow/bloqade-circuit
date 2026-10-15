"""This module holds the affine indices that name an item of a register.

An affine index has the form `c + a1*s1 + ... + an*sn`. Each `c` and `a` is a
constant integer. Each symbol `s` is an integer parameter or the iteration variable
of a loop or a map. Indices such as `i`, `i + 1`, `2*i + 1` and `n - 1 - i` are
affine, and `i * i` and `i % 2` are not.

Affine indices let the reference analysis tell `qs[i]` and `qs[i + 1]` apart. Two
affine indices are equal when their difference is 0, and that check is cheap.
"""

from __future__ import annotations

import math
from typing import Mapping
from dataclasses import dataclass

from kirin import ir


@dataclass(frozen=True)
class Iter:
    """The iteration variable of the loop or map statement `stmt`.

    If the loop runs over a constant range, `values` holds that range. Otherwise
    `values` is None. Two `Iter` symbols of one statement name the same iteration.
    """

    stmt: ir.Statement
    values: range | None = None

    def __repr__(self) -> str:
        """Return the name of the iteration variable, such as `%i`.

        A loop names the first argument of its body. A map names its result
        with the suffix `[*]`, such as `%ys[*]`.
        """
        for region in self.stmt.regions:
            if region.blocks and region.blocks[0].args:
                return _name(region.blocks[0].args[0])
        if self.stmt.results:
            return _name(self.stmt.results[0]) + "[*]"
        return f"%{self.stmt.name}"


def _name(value: ir.SSAValue) -> str:
    """Return the printed name of an SSA value, such as `%n`."""
    return f"%{value.name}" if value.name else "%?"


Sym = ir.SSAValue | Iter
"""A symbol of an affine index: an integer SSA value or an `Iter`."""


@dataclass(frozen=True, repr=False)
class Affine:
    """The index `const + sum(a * s for s, a in terms)`.

    `terms` holds each symbol at most once, and no symbol has the coefficient 0. So
    two equal indices have equal fields. Use `Affine.make` to build an index from a
    mapping that may hold zero coefficients.
    """

    const: int = 0
    terms: frozenset[tuple[Sym, int]] = frozenset()

    def __repr__(self) -> str:
        """Return the index as a sum, such as `2*%i + 1` or `%n - 1`."""
        named = sorted(
            (repr(s) if isinstance(s, Iter) else _name(s), a) for s, a in self.terms
        )
        parts: list[str] = []
        for name, a in named:
            factor = "" if abs(a) == 1 else f"{abs(a)}*"
            sign = "-" if a < 0 else "+"
            parts.append(f"{sign} {factor}{name}" if parts else f"{sign}{factor}{name}")
        text = " ".join(parts).removeprefix("+")
        if not text:
            return str(self.const)
        if self.const:
            text += f" {'-' if self.const < 0 else '+'} {abs(self.const)}"
        return text

    @staticmethod
    def make(const: int, coeffs: Mapping[Sym, int]) -> Affine:
        """Return the index with the constant `const` and the coefficients `coeffs`."""
        return Affine(const, frozenset((s, a) for s, a in coeffs.items() if a))

    @staticmethod
    def of(sym: Sym) -> Affine:
        """Return the index that is the symbol `sym` alone."""
        return Affine(0, frozenset({(sym, 1)}))

    @property
    def constant(self) -> int | None:
        """Return the value of an index without symbols, or None if it has one."""
        return None if self.terms else self.const

    def __add__(self, other: Affine | int) -> Affine:
        """Return the sum of two indices."""
        other = lift(other)
        coeffs = dict(self.terms)
        for s, a in other.terms:
            coeffs[s] = coeffs.get(s, 0) + a
        return Affine.make(self.const + other.const, coeffs)

    def __sub__(self, other: Affine | int) -> Affine:
        """Return the difference of two indices."""
        return self + lift(other).scale(-1)

    def scale(self, k: int) -> Affine:
        """Return the index multiplied by the constant `k`."""
        return Affine.make(self.const * k, {s: a * k for s, a in self.terms})

    def substitute(self, mapping: Mapping[Sym, Affine]) -> Affine:
        """Replace each symbol that `mapping` holds with its index.

        A call uses this to replace a parameter of the callee with the index of the
        argument that the caller passes.
        """
        out = Affine(self.const)
        for s, a in self.terms:
            out = out + mapping.get(s, Affine.of(s)).scale(a)
        return out

    def symbols(self) -> frozenset[Sym]:
        """Return the symbols that the index uses."""
        return frozenset(s for s, _ in self.terms)

    def bounds(self) -> tuple[int, int] | None:
        """Return the smallest and the largest value of the index.

        The bounds treat each symbol on its own, so they can be wider than the true
        range. If a symbol has no constant range, the result is None.
        """
        lo = hi = self.const
        for s, a in self.terms:
            if not (isinstance(s, Iter) and s.values):
                return None
            ends = (a * s.values[0], a * s.values[-1])
            lo, hi = lo + min(ends), hi + max(ends)
        return lo, hi


def lift(value: Affine | int) -> Affine:
    """Return `value` as an index. An integer becomes a constant index."""
    return value if isinstance(value, Affine) else Affine(value)


def same(a: Affine, b: Affine) -> bool | None:
    """Check whether the indices `a` and `b` name the same item.

    The result is True if `a` equals `b` for every value of the symbols. The result
    is False if `a` equals `b` for no value of the symbols. Otherwise the result is
    None. A shared `Iter` symbol stands for one iteration in both indices.
    """
    d = a - b
    if d.constant is not None:
        return d.constant == 0
    # The GCD test: if the GCD of the coefficients does not divide the constant,
    # `d` is never 0. For example, `2*i` and `2*i + 1` never meet.
    if d.const % math.gcd(*(c for _, c in d.terms)):
        return False
    limits = d.bounds()
    if limits is not None and (limits[0] > 0 or limits[1] < 0):
        return False
    return None


@dataclass(frozen=True, repr=False)
class Slice:
    """The items `start + j*step` of a register, for `j` from 0 to `length - 1`.

    If the slice runs to the end of a register of unknown length, `length` is None.
    """

    start: Affine
    step: int
    length: Affine | None

    def __repr__(self) -> str:
        """Return the slice as `start:step:length`, such as `1:1:2`.

        A slice that runs to the end of a register of unknown length prints
        `?` as its length.
        """
        length = "?" if self.length is None else repr(self.length)
        return f"{self.start!r}:{self.step}:{length}"

    def item(self, j: Affine) -> Affine:
        """Return the register index of item `j` of the slice."""
        return self.start + j.scale(self.step)

    def then(self, inner: Slice) -> Slice:
        """Return the slice `inner` of this slice as a slice of the register."""
        return Slice(self.item(inner.start), self.step * inner.step, inner.length)


def python_slice(
    start: Affine | None, stop: Affine | None, step: int, size: int | None
) -> Slice | None:
    """Return the slice that `x[start:stop:step]` reads from a list of `size` items.

    Python counts a negative bound from the end and clamps a bound past the end. If
    every bound and `size` are constants, the result follows those rules exactly.
    A bound with symbols must lie in the range 0 to `size`, or the result is None.
    The result is also None for a zero step, and for a negative step with symbols.
    """
    if step == 0:
        return None
    bounds = (start, stop)
    if size is not None and all(b is None or b.constant is not None for b in bounds):
        constants = (None if b is None else b.const for b in bounds)
        items = range(size)[slice(*constants, step)]
        return Slice(Affine(items.start), items.step, Affine(len(items)))
    if step < 0:
        return None
    start = start or Affine(0)

    def in_range(bound: Affine) -> bool:
        limits = bound.bounds()
        return (
            limits is not None
            and limits[0] >= 0
            and (size is None or limits[1] <= size)
        )

    if not in_range(start) or (stop is not None and not in_range(stop)):
        return None
    if stop is None:
        if size is None:
            return Slice(start, step, None)
        stop = Affine(size)
    span = stop - start
    if (n := span.constant) is not None:
        return Slice(start, step, Affine(max(0, -(-n // step))))
    # A symbolic length must stay affine, and `ceil(span / step)` is affine only
    # for a step of 1.
    limits = span.bounds()
    if step == 1 and limits is not None and limits[0] >= 0:
        return Slice(start, 1, span)
    return None
