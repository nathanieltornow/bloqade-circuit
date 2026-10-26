"""Remove the `Release` and `Borrow` pairs that the converter leaves between two gates.

The converter borrows each register qubit right before a gate and releases it
right after. So two gates on `qs[0]` sit on either side of a `Release` and a
`Borrow` at index 0. These rules join the two gates onto one wire.

Two indices are equal if they are one SSA value or two equal constants. Two
indices differ if they are two different constants.
"""

from kirin import ir
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.constants import constant_int

from .. import stmts


def same_index(a: ir.SSAValue, b: ir.SSAValue) -> bool | None:
    """Return True if `a` and `b` are equal indices, False if they differ, else None."""
    if a is b:
        return True
    x, y = constant_int(a), constant_int(b)
    if x is None or y is None:
        return None
    return x == y


def local_producer(value: ir.SSAValue, user: ir.Statement) -> ir.Statement | None:
    """Return the statement that defines `value`, if it sits in the block of `user`.

    A branch of `scf.if` reads a wire of the enclosing block, and the other branch
    reads the same wire. A rule that joins two statements across that boundary
    would break the other branch. So every rule asks for a producer in one block.
    """
    owner = value.owner
    if isinstance(owner, ir.Statement) and owner.parent_block is user.parent_block:
        return owner
    return None


def _above(first: ir.Statement, second: ir.Statement) -> bool:
    """Return True if `first` comes before `second` in their common block."""
    node = second.prev_stmt
    while node is not None:
        if node is first:
            return True
        node = node.prev_stmt
    return False


class ForwardRelease(RewriteRule):
    """Hand the wire of a `Release` straight to a `Borrow` at the same index.

    `Borrow(Release(r, i, w), i)` gives back the register `r` and the wire `w`.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Join `node` with the `Release` before it, if both use one index."""
        if not isinstance(node, stmts.Borrow):
            return RewriteResult()
        release = local_producer(node.register, node)
        if not isinstance(release, stmts.Release):
            return RewriteResult()
        if same_index(release.index, node.index) is not True:
            return RewriteResult()
        node.wire.replace_by(release.wire)
        node.register_out.replace_by(release.register)
        node.delete()
        release.delete()
        return RewriteResult(has_done_something=True)


class BorrowFirst(RewriteRule):
    """Move a `Borrow` before a `Release` at a different constant index.

    The swap brings each `Borrow` closer to the `Release` at its own index. Then
    `ForwardRelease` can join them.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Swap `node` with the `Release` before it, if their indices differ."""
        if not isinstance(node, stmts.Borrow):
            return RewriteResult()
        release = local_producer(node.register, node)
        if not isinstance(release, stmts.Release):
            return RewriteResult()
        if same_index(release.index, node.index) is not False:
            return RewriteResult()
        # `same_index` only returns False for two constants. The constant of the
        # `Borrow` can sit between the two statements, so it moves up first.
        constant = node.index.owner
        assert isinstance(constant, ir.Statement)
        if constant.parent_block is release.parent_block and not _above(
            constant, release
        ):
            constant.detach()
            constant.insert_before(release)
        borrow = stmts.Borrow(release.register, node.index)
        borrow.insert_before(release)
        moved = stmts.Release(borrow.register_out, release.index, release.wire)
        moved.insert_before(node)
        node.wire.replace_by(borrow.wire)
        node.register_out.replace_by(moved.register_out)
        node.delete()
        release.delete()
        return RewriteResult(has_done_something=True)


class DropRoundTrip(RewriteRule):
    """Remove a `Release` that puts an unchanged borrowed wire back at its index."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Remove `node` and its `Borrow` if nothing touched the wire between them."""
        if not isinstance(node, stmts.Release):
            return RewriteResult()
        borrow = local_producer(node.wire, node)
        if (
            not isinstance(borrow, stmts.Borrow)
            or node.register is not borrow.register_out
        ):
            return RewriteResult()
        if same_index(borrow.index, node.index) is not True:
            return RewriteResult()
        node.register_out.replace_by(borrow.register)
        node.delete()
        borrow.delete()
        return RewriteResult(has_done_something=True)
