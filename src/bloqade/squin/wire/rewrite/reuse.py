"""Reuse the qubit of a `Free` for a later `Alloc` in the same block.

In wire squin, the last use of a qubit is the statement that defines the wire that
`Free` consumes. `HoistFree` moves each `Free` right after that statement, so the
qubit dies as early as possible. `ReuseFreed` then turns a `Free` and a later
`Alloc` into one `reset`, so the kernel needs one qubit fewer.

The def-use chain of the wire holds the lifetime of the qubit, so the rules need
no analysis.
"""

from kirin import ir
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts

from .. import stmts
from ..twins import TWINS


def _only_frees_between(first: ir.Statement | None, last: ir.Statement) -> bool:
    """Return True if each statement after `first` and before `last` is a `Free`.

    If `first` is None, the walk starts at the top of the block of `last`.
    """
    node = last.prev_stmt
    while node is not first:
        if node is None or not isinstance(node, stmts.Free):
            return False
        node = node.prev_stmt
    return True


class HoistFree(RewriteRule):
    """Move each `Free` right after the statement that defines its wire."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Move `node` up, unless only other `Free` statements lie above it."""
        if not isinstance(node, stmts.Free):
            return RewriteResult()
        owner = node.wire.owner
        block = node.parent_block
        if isinstance(owner, ir.Block):
            if owner is not block or _only_frees_between(None, node):
                return RewriteResult()
            node.detach()
            first = block.first_stmt
            assert first is not None
            node.insert_before(first)
            return RewriteResult(has_done_something=True)
        if owner.parent_block is not block or _only_frees_between(owner, node):
            return RewriteResult()
        node.detach()
        node.insert_after(owner)
        return RewriteResult(has_done_something=True)


class ReuseFreed(RewriteRule):
    """Turn an `Alloc` into a `reset` of a qubit that a `Free` above it releases."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Replace `node` with a reset of the nearest freed qubit above it."""
        if not isinstance(node, stmts.Alloc):
            return RewriteResult()
        free = node.prev_stmt
        while free is not None and not isinstance(free, stmts.Free):
            free = free.prev_stmt
        if free is None:
            return RewriteResult()
        reset = TWINS[qubit_stmts.Reset](qubit=free.wire)
        reset.insert_before(node)
        node.result.replace_by(reset.qubit_out)
        node.delete()
        free.delete()
        return RewriteResult(has_done_something=True)
