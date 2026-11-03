"""Turn registers of a constant size into separate wires.

`ExpandBroadcast` replaces a register operation with one scalar operation per
position when the register types carry their lengths. `UnpackRegister` then replaces
a local register with one `Alloc` per qubit, if every statement on the register's
linear chain reads a constant index. After both rules, 2 gates on one qubit are
neighbours on one wire, so a peephole rule can match them.

Both rules read static lengths from the types, so Kirin's type inference must run
first.
"""

from dataclasses import dataclass

from kirin import ir, types
from kirin.dialects import py, ilist
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import register_length


class ExpandBroadcast(RewriteRule):
    """Replace a register operation of static length with scalar operations.

    Each position `k` gets a `Borrow` at `k` from every register, the scalar
    operation, and a `Release` at `k`. The classical results of the positions form
    one list per classical result, such as the bits of a `RegisterMeasure`.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Expand `node` if it is a register operation whose lengths are known."""
        if not isinstance(node, stmts.RegisterOperation):
            return RewriteResult()
        lengths = [register_length(register.type) for register in node.linear]
        if any(length is None for length in lengths):
            return RewriteResult()
        known = [length for length in lengths if length is not None]
        # A program that broadcasts over unequal lengths breaks a precondition.
        # The rule leaves that operation for the emitter's checks to report.
        if not node.truncates and len(set(known)) > 1:
            return RewriteResult()
        registers = list(node.linear)
        collected: list[list[ir.SSAValue]] = [[] for _ in node.classical]
        for position in range(min(known, default=0)):
            index = py.Constant(position)
            index.insert_before(node)
            wires: list[ir.SSAValue] = []
            for k, register in enumerate(registers):
                borrow = stmts.Borrow(register, index.result)
                borrow.insert_before(node)
                borrow.register_out.type = register.type
                registers[k] = borrow.register_out
                wires.append(borrow.wire)
            scalar = node.scalar(tuple(wires))
            scalar.insert_before(node)
            for k, (register, output) in enumerate(zip(registers, scalar.outputs)):
                release = stmts.Release(register, index.result, output)
                release.insert_before(node)
                release.register_out.type = register.type
                registers[k] = release.register_out
            for items, value in zip(collected, scalar.classical, strict=True):
                items.append(value)
        for result, items in zip(node.classical, collected, strict=True):
            made = ilist.New(tuple(items), elem_type=element_type(result.type))
            made.insert_before(node)
            result.replace_by(made.result)
        for output, register in zip(node.outputs, registers, strict=True):
            output.replace_by(register)
        node.delete()
        return RewriteResult(has_done_something=True)


def element_type(kind: types.TypeAttribute) -> types.TypeAttribute:
    """Return the item type of the list type `kind`."""
    assert isinstance(kind, types.Generic)
    return kind.vars[0]


@dataclass(frozen=True)
class Step:
    """Record one statement on the linear chain of a register and the slot it reads.

    `slot` is the constant index of a `Borrow` or a `Release`. It is None for a
    `Length` and for the statement that frees the register.
    """

    stmt: ir.Statement
    slot: int | None = None


class UnpackRegister(RewriteRule):
    """Replace a local register of constant size with one wire per qubit.

    The rule starts at an `AllocRegister` with a constant size or at a
    `PackRegister`. It follows the register from each statement to the statement
    that consumes its continuing result. Every statement on that chain must be a
    `Borrow` or a `Release` at a constant index, a `Length`, or the final free, and
    it must sit in the block of the register. If any statement breaks these rules,
    the rule leaves the register unchanged. A loop, a branch or a call that takes
    the register breaks them.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Unpack the register that `node` creates, if its whole chain is static."""
        if isinstance(node, stmts.AllocRegister):
            size = constant_int(node.size)
            start: ir.SSAValue = node.result
        elif isinstance(node, stmts.PackRegister):
            size = len(node.wires)
            start = node.result
        else:
            return RewriteResult()
        if size is None or (steps := chain(node, start, size)) is None:
            return RewriteResult()
        if isinstance(node, stmts.PackRegister):
            slots: list[ir.SSAValue] = list(node.wires)
        else:
            slots = []
            for _ in range(size):
                alloc = stmts.Alloc()
                alloc.insert_before(node)
                slots.append(alloc.result)
        for step in steps:
            stmt = step.stmt
            if isinstance(stmt, stmts.Borrow):
                assert step.slot is not None
                stmt.wire.replace_by(slots[step.slot])
            elif isinstance(stmt, stmts.Release):
                assert step.slot is not None
                slots[step.slot] = stmt.wire
            elif isinstance(stmt, stmts.Length):
                count = py.Constant(size)
                count.insert_before(stmt)
                stmt.result.replace_by(count.result)
            else:
                for wire in slots:
                    stmts.Free(wire).insert_before(stmt)
        # Each register value has one use, which is the next statement on the
        # chain. Deleting from the end keeps every deleted result unused.
        for step in reversed(steps):
            step.stmt.delete()
        node.delete()
        return RewriteResult(has_done_something=True)


def chain(node: ir.Statement, start: ir.SSAValue, size: int) -> list[Step] | None:
    """Return the statements on the linear chain of the register `start`, or None.

    The result ends with the `FreeRegister` or `DiscardRegister` of the register. A
    `Borrow` must read a slot that holds its qubit, and a `Release` must fill an
    empty slot. The function returns None if a statement breaks the rules of
    `UnpackRegister`.
    """
    full = [True] * size
    steps: list[Step] = []
    current = start
    while True:
        if len(current.uses) != 1:
            return None
        (use,) = current.uses
        stmt = use.stmt
        if stmt.parent_block is not node.parent_block:
            return None
        if isinstance(stmt, (stmts.FreeRegister, stmts.DiscardRegister)):
            if not all(full):
                return None
            steps.append(Step(stmt))
            return steps
        if isinstance(stmt, stmts.Length):
            steps.append(Step(stmt))
            current = stmt.register_out
            continue
        if not isinstance(stmt, (stmts.Borrow, stmts.Release)):
            return None
        slot = constant_int(stmt.index)
        if slot is None or not 0 <= slot < size:
            return None
        # A `Borrow` empties a full slot, and a `Release` fills an empty one.
        if full[slot] != isinstance(stmt, stmts.Borrow):
            return None
        full[slot] = not full[slot]
        steps.append(Step(stmt, slot))
        current = stmt.register_out
