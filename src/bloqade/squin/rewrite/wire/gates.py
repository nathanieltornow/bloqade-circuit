"""Cancel and merge neighboring gates on the same wires.

Two gates are neighbors if the outputs of the first gate feed the inputs of the
second gate in the same order, with nothing in between. In wire squin this is a
property of the def-use chain, so the rules need no analysis. A noise channel or a
measurement between two gates breaks the chain, so the rules never cross one.
"""

from kirin import ir
from kirin.dialects import py
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.squin.gate import stmts as gate_stmts
from bloqade.squin.wire.traits import Twin, Threads, Unitary
from bloqade.squin.rewrite.wire.registers import local_producer

SELF_INVERSE = frozenset(
    {
        gate_stmts.X,
        gate_stmts.Y,
        gate_stmts.Z,
        gate_stmts.H,
        gate_stmts.CX,
        gate_stmts.CY,
        gate_stmts.CZ,
        gate_stmts.CCZ,
        gate_stmts.Swap,
    }
)
"""Hold the squin gate classes that undo themselves."""

ROTATIONS = frozenset({gate_stmts.Rx, gate_stmts.Ry, gate_stmts.Rz})
"""Hold the squin rotation classes whose angles add up."""


def neighbor(node: ir.Statement) -> ir.Statement | None:
    """Return the gate whose outputs feed every wire input of `node`, or None.

    The two gates must be twins of one squin class in one block. Output `k` of
    the first gate must feed input `k` of `node`.
    """
    threads = node.get_trait(Threads)
    if threads is None or not node.has_trait(Unitary):
        return None
    pairs = threads.threads(node)
    previous = local_producer(pairs[0][0], node)
    if type(previous) is not type(node):
        return None
    before = previous.get_present_trait(Threads).threads(previous)
    for (_, out), (into, _) in zip(before, pairs, strict=True):
        if out is not into:
            return None
    return previous


def _splice(first: ir.Statement, second: ir.Statement) -> None:
    """Connect the inputs of `first` to the users of the outputs of `second`."""
    before = first.get_present_trait(Threads).threads(first)
    after = second.get_present_trait(Threads).threads(second)
    for (into, _), (_, out) in zip(before, after, strict=True):
        out.replace_by(into)


class CancelInverses(RewriteRule):
    """Remove two neighboring gates that undo each other.

    A self-inverse gate, such as `h`, cancels with itself. A gate with an
    `adjoint` flag, such as `s`, cancels with its adjoint.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Remove `node` and its neighbor if they undo each other."""
        previous = neighbor(node)
        if previous is None:
            return RewriteResult()
        source = node.get_present_trait(Twin).source
        if source not in SELF_INVERSE:
            flags = (getattr(previous, "adjoint", None), getattr(node, "adjoint", None))
            if None in flags or flags[0] == flags[1]:
                return RewriteResult()
        _splice(previous, node)
        node.delete()
        previous.delete()
        return RewriteResult(has_done_something=True)


class MergeRotations(RewriteRule):
    """Join two neighboring rotations about the same axis into one rotation.

    The new rotation turns by the sum of the two angles.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Replace `node` and its neighbor with one rotation."""
        previous = neighbor(node)
        if previous is None or node.get_present_trait(Twin).source not in ROTATIONS:
            return RewriteResult()
        total = py.Add(previous.angle, node.angle)
        total.result.type = previous.angle.type.join(node.angle.type)
        total.insert_before(node)
        merged = type(node)(angle=total.result, qubit=previous.qubit)
        merged.insert_before(node)
        node.qubit_out.replace_by(merged.qubit_out)
        node.delete()
        previous.delete()
        return RewriteResult(has_done_something=True)
