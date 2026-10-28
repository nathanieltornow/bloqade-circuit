"""Build the integer statements of an affine index, such as `2 * i + 1`.

The converter names a register element with an integer expression over symbols.
A symbol names an integer value or the iteration variable of an `scf.For` loop.
"""

from collections.abc import Mapping

from kirin import ir, types
from kirin.dialects import py, scf

from bloqade.analysis.indices import Sym, Iter, Affine


def _integer(stmt: ir.Statement, before: ir.Statement) -> ir.SSAValue:
    """Insert the integer arithmetic `stmt` before `before`, and return its result.

    A new `py.binop` statement has the result type `~T` until type inference runs,
    and kirin's `verify_type` rejects `~T` as an index.
    """
    stmt.insert_before(before)
    stmt.results[0].type = types.Int
    return stmt.results[0]


def symbol_value(symbol: Sym) -> ir.SSAValue:
    """Return the integer value that the index symbol `symbol` stands for."""
    if isinstance(symbol, Iter):
        if not isinstance(symbol.stmt, scf.For):
            raise ValueError(f"{symbol!r} has no loop variable")
        return symbol.stmt.body.blocks[0].args[0]
    return symbol


def index_value(
    index: Affine,
    before: ir.Statement,
    symbols: Mapping[Sym, ir.SSAValue] | None = None,
) -> ir.SSAValue:
    """Insert the statements that compute `index` before `before`, and return it.

    The sum takes each symbol times its coefficient, plus the constant.
    Structural rewrites supply replacement values for symbols whose owners they rebuild.
    """
    if (constant := index.constant) is not None:
        made = py.Constant(constant)
        made.insert_before(before)
        return made.result
    total: ir.SSAValue | None = None
    for symbol, a in sorted(index.terms, key=lambda t: repr(t[0])):
        term = (
            symbols[symbol]
            if symbols is not None and symbol in symbols
            else symbol_value(symbol)
        )
        if a != 1:
            factor = py.Constant(a)
            factor.insert_before(before)
            term = _integer(py.Mult(term, factor.result), before)
        total = term if total is None else _integer(py.Add(total, term), before)
    assert total is not None
    if index.const:
        offset = py.Constant(index.const)
        offset.insert_before(before)
        total = _integer(py.Add(total, offset.result), before)
    return total
