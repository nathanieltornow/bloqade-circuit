"""Represent the affine indices that name an item of a register.

An affine index has the form `c + a1*s1 + ... + an*sn`. Each `c` and `a` is a
constant integer. Each symbol `s` is an integer parameter or the iteration variable
of an `scf.For` loop. Indices such as `i`, `i + 1`, `2*i + 1` and `n - 1 - i` are
affine, and `i * i` and `i % 2` are not.

The converter and wire validators use these expressions to compare register indices.
Two affine indices are equal when their difference is zero.
"""

from __future__ import annotations

import math
from typing import Mapping
from dataclasses import dataclass

from kirin import ir
from kirin.dialects import py, ilist
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.constants import constant_int


@dataclass(frozen=True)
class Iter:
    """The iteration variable of the `scf.For` loop `stmt`.

    If the loop runs over a constant range, `values` holds that range. Otherwise
    `values` is None. Two `Iter` symbols of one statement name the same iteration.
    """

    stmt: ir.Statement
    values: range | None = None

    def __repr__(self) -> str:
        """Return the name of the first argument of the loop body, such as `%i`."""
        for region in self.stmt.regions:
            if region.blocks and region.blocks[0].args:
                return _name(region.blocks[0].args[0])
        return f"%{self.stmt.name}"


@dataclass(frozen=True)
class Emitted:
    """A symbol for an integer of the output IR that has no source value."""

    value: ir.SSAValue

    def __repr__(self) -> str:
        """Return the name of the emitted value, such as `%i`."""
        return _name(self.value)


def _name(value: ir.SSAValue) -> str:
    """Return the printed name of an SSA value, such as `%n`."""
    return f"%{value.name}" if value.name else "%?"


Sym = ir.SSAValue | Iter | Emitted
"""Name an integer value, the iteration variable of a loop or an emitted integer."""


def _symbol_name(sym: Sym) -> str:
    """Return the printed name of a symbol, such as `%n`, `%i` or `%i'`."""
    return _name(sym) if isinstance(sym, ir.SSAValue) else repr(sym)


def _values(sym: Sym) -> range | None:
    """Return the constant range of the symbol `sym`, or None if it has none."""
    return sym.values if isinstance(sym, Iter) else None


@dataclass(frozen=True, repr=False)
class Affine:
    """The index `offset + sum(a * s for s, a in terms)`.

    `terms` holds each symbol at most once, and no symbol has the coefficient 0. So
    two equal indices have equal fields. Use `Affine.make` to build an index from a
    mapping that may hold zero coefficients. `offset` is the constant term, and
    `constant` is the value of an index without symbols.
    """

    offset: int = 0
    terms: frozenset[tuple[Sym, int]] = frozenset()

    def __repr__(self) -> str:
        """Return the index as a sum, such as `2*%i + 1` or `%n - 1`."""
        named = sorted((_symbol_name(s), a) for s, a in self.terms)
        parts: list[str] = []
        for name, a in named:
            factor = "" if abs(a) == 1 else f"{abs(a)}*"
            sign = "-" if a < 0 else "+"
            parts.append(f"{sign} {factor}{name}" if parts else f"{sign}{factor}{name}")
        text = " ".join(parts).removeprefix("+")
        if not text:
            return str(self.offset)
        if self.offset:
            text += f" {'-' if self.offset < 0 else '+'} {abs(self.offset)}"
        return text

    @staticmethod
    def make(offset: int, coeffs: Mapping[Sym, int]) -> Affine:
        """Return the index with the constant term `offset` and the coefficients `coeffs`."""
        return Affine(offset, frozenset((s, a) for s, a in coeffs.items() if a))

    @staticmethod
    def of(sym: Sym) -> Affine:
        """Return the index that is the symbol `sym` alone."""
        return Affine(0, frozenset({(sym, 1)}))

    @property
    def constant(self) -> int | None:
        """Return the value of an index without symbols, or None if it has one."""
        return None if self.terms else self.offset

    def __add__(self, other: Affine | int) -> Affine:
        """Return the sum of two indices."""
        other = lift(other)
        coeffs = dict(self.terms)
        for s, a in other.terms:
            coeffs[s] = coeffs.get(s, 0) + a
        return Affine.make(self.offset + other.offset, coeffs)

    def __sub__(self, other: Affine | int) -> Affine:
        """Return the difference of two indices."""
        return self + lift(other).scale(-1)

    def scale(self, k: int) -> Affine:
        """Return the index multiplied by the constant `k`."""
        return Affine.make(self.offset * k, {s: a * k for s, a in self.terms})

    def substitute(self, mapping: Mapping[Sym, Affine]) -> Affine:
        """Replace each symbol that `mapping` holds with its index.

        A call uses this to replace a parameter of the callee with the index of the
        argument that the caller passes.
        """
        out = Affine(self.offset)
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
        lo = hi = self.offset
        for s, a in self.terms:
            if not (values := _values(s)):
                return None
            ends = (a * values[0], a * values[-1])
            lo, hi = lo + min(ends), hi + max(ends)
        return lo, hi


def lift(value: Affine | int) -> Affine:
    """Return `value` as an index. An integer becomes a constant index."""
    return value if isinstance(value, Affine) else Affine(value)


def same(a: Affine, b: Affine) -> bool | None:
    """Check whether the indices `a` and `b` name the same item.

    The result is True if `a` equals `b` for every value of the symbols. The result
    is False if `a` equals `b` for no value of the symbols. Otherwise the result is
    None.

    A symbol has one value in both indices, so the check compares the indices at
    one moment. A shared `Iter` symbol stands for one iteration in both indices.
    So `same(i, i + 1)` is False, although item `i + 1` of iteration 0 is item `i`
    of iteration 1.
    """
    d = a - b
    if d.constant is not None:
        return d.constant == 0
    # The GCD test: if the GCD of the coefficients does not divide the constant,
    # `d` is never 0. For example, `2*i` and `2*i + 1` never meet.
    if d.offset % math.gcd(*(c for _, c in d.terms)):
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

    def then(self, inner: Slice) -> Slice | None:
        """Return the slice `inner` of this slice as a slice of the register.

        If `inner` runs to the end of this slice, its length comes from the length
        of this slice. For example, `(0:1:%i)[1:]` has the length `%i - 1`. The
        result is None if that length is not affine or can be negative.
        """
        length = inner.length
        if length is None and self.length is not None:
            span = self.length - inner.start
            if (n := span.constant) is not None:
                length = Affine(max(0, -(-n // inner.step)))
            elif inner.step != 1 or (limits := span.bounds()) is None or limits[0] < 0:
                return None
            else:
                length = span
        return Slice(self.item(inner.start), self.step * inner.step, length)


def python_slice(
    start: Affine | None, stop: Affine | None, step: int, size: int | None
) -> Slice | None:
    """Return the slice that `x[start:stop:step]` reads from a list of `size` items.

    Python counts a negative bound from the end and clamps a bound past the end. If
    every bound and `size` are constants, the result follows those rules exactly.
    A bound with symbols must lie in the range 0 to `size`, or the result is None.
    If `size` is None, a stop gives None, because Python clamps it to the size.
    The result is also None for a zero step, and for a negative step with symbols.
    """
    if step == 0:
        return None
    bounds = (start, stop)
    if size is not None and all(b is None or b.constant is not None for b in bounds):
        first, last = (None if b is None else b.constant for b in bounds)
        items = range(size)[slice(first, last, step)]
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
    # Python clamps a stop past the end, and a read past the end raises no error.
    # So without the size, the length of `x[1:3]` can be 2, 1 or 0.
    if stop is not None and size is None:
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


def loop_values(iterable: ir.SSAValue) -> range | None:
    """Return the constant range that a loop runs over, or None.

    The iterable is a constant list that holds a `range`, or an `ilist.range`
    call with constant arguments.
    """
    owner = iterable.owner
    if isinstance(owner, py.Constant):
        data = owner.value.unwrap()
        if isinstance(data, ilist.IList) and isinstance(data.data, range):
            return data.data
    if isinstance(owner, ilist_stmts.Range):
        bounds = [constant_int(arg) for arg in owner.args]
        if all(b is not None for b in bounds):
            return range(*bounds)  # type: ignore[arg-type]
    return None
