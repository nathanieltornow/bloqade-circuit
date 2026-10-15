"""This module holds `IndexAnalysis`, which maps each integer SSA value to an affine index.

The reference analysis keeps integers `Untracked`, so that the checks of the
emitters see no change. It asks `IndexAnalysis` for the index of each value that
reads a register. For example, in `for i in range(3): qs[i + 1]` the value
`i + 1` has the index `%i + 1`, where `%i` is the `Iter` symbol of the loop.
"""

from dataclasses import field, dataclass

from kirin import ir
from kirin.dialects import py, scf, ilist
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.constants import constant_int

from .index import Iter, Affine


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


@dataclass
class IndexAnalysis:
    """Map each integer SSA value to an affine index, and remember each result.

    A constant gives a constant index. `+` and `-` give the sum and the difference
    of their operand indices. A product with a constant scales the other operand.
    The variable of an `scf.for` loop gives its `Iter` symbol. Every other value,
    such as a parameter or `i * i`, is a symbol of its own. Such a symbol stands
    for one runtime value, so two reads at the same symbol name the same item.
    """

    cache: dict[ir.SSAValue, Affine] = field(default_factory=dict)

    def of(self, value: ir.SSAValue) -> Affine:
        """Return the affine index of the integer SSA value `value`."""
        if value not in self.cache:
            self.cache[value] = self._compute(value)
        return self.cache[value]

    def _compute(self, value: ir.SSAValue) -> Affine:
        """Return the affine index of `value` from the statement that defines it."""
        if (constant := constant_int(value)) is not None:
            return Affine(constant)
        if isinstance(value, ir.BlockArgument):
            loop = value.owner.parent_stmt
            if isinstance(loop, scf.For) and value.index == 0:
                return Affine.of(Iter(loop, loop_values(loop.iterable)))
            return Affine.of(value)
        match value.owner:
            case py.assign.Alias(value=aliased):
                return self.of(aliased)
            case py.binop.Add(lhs=lhs, rhs=rhs):
                return self.of(lhs) + self.of(rhs)
            case py.binop.Sub(lhs=lhs, rhs=rhs):
                return self.of(lhs) - self.of(rhs)
            case py.unary.USub(value=operand):
                return self.of(operand).scale(-1)
            case py.binop.Mult(lhs=lhs, rhs=rhs):
                left, right = self.of(lhs), self.of(rhs)
                if (k := left.constant) is not None:
                    return right.scale(k)
                if (k := right.constant) is not None:
                    return left.scale(k)
        return Affine.of(value)
