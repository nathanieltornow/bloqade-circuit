"""This module holds queries that compare the places of the reference analysis.

`must_alias` proves that two places name the same items. `disjoint` proves that two
places share no item. Each query answers False when it cannot prove its claim, so a
pass can use the answer to merge or to separate the state of qubits.
"""

from kirin import ir
from kirin.dialects import scf, func, ilist

from .index import Slice, Affine, same
from .lattice import Ref, Root, Step, Place, Returned
from .analysis import RegisterLength, items_of


def leaves(ref: Ref, register_length: RegisterLength) -> tuple[Place, ...] | None:
    """Return the place of each item of tracked state that `ref` holds, or None.

    A register of registers holds the items of each row, in order. A literal list
    holds the items of each member. The result is None if a length is unknown or
    if a member is no place.
    """
    if isinstance(ref, Place) and ref.one_item:
        return (ref,)
    items = items_of(ref, register_length)
    if items is None:
        return None
    out: list[Place] = []
    for item in items:
        below = leaves(item, register_length)
        if below is None:
            return None
        out.extend(below)
    return tuple(out)


_REPEATING = (scf.For, func.Lambda, ilist.Map, ilist.ForEach)
"""Statements whose body can run more than once for one run of the function."""


def family(root: Root) -> bool:
    """Return True if `root` stands for a new item each time a loop or a map runs.

    An allocation inside a loop body or a function body that a map runs has one
    root for all its runs. A count keyed by such a root sums over those items.
    """
    while isinstance(root, Returned):
        if _repeats(root.call):
            return True
        root = root.inner
    owner = root.owner
    node = owner if isinstance(owner, ir.Statement) else owner.parent_stmt
    return node is not None and _repeats(node)


def _repeats(node: ir.Statement) -> bool:
    """Return True if a repeating statement encloses `node`, `node` excluded."""
    parent = node.parent_stmt
    while parent is not None:
        if isinstance(parent, _REPEATING):
            return True
        parent = parent.parent_stmt
    return False


def fresh(root: Root) -> bool:
    """Return True if `root` is an allocation, which no other root can alias.

    A parameter of a function can receive the same qubit as another parameter, so
    a parameter root is not fresh.
    """
    if isinstance(root, Returned):
        return True
    return not (
        isinstance(root, ir.BlockArgument)
        and isinstance(root.owner.parent_stmt, (func.Function, func.Lambda))
    )


def _picked(step: Step) -> tuple[Affine, int, int] | None:
    """Return the start, the step and the constant count of the items of `step`."""
    if isinstance(step, Affine):
        return step, 1, 1
    if step.length is None or (count := step.length.constant) is None:
        return None
    return step.start, step.step, count


def apart(x: Step, y: Step) -> bool:
    """Return True if the steps `x` and `y` at one level of two paths share no item.

    Two indices are apart if `same` proves them different. An index lies apart from
    a slice of step 1 if its distance to the slice start is outside the slice. Two
    slices with constant bounds are apart if they pick no common item.
    """
    if isinstance(x, Affine) and isinstance(y, Affine):
        return same(x, y) is False
    a, b = _picked(x), _picked(y)
    if a is None or b is None:
        return False
    if a[0].constant is not None and b[0].constant is not None:
        left = {a[0].const + j * a[1] for j in range(a[2])}
        right = {b[0].const + j * b[1] for j in range(b[2])}
        return not left & right
    if isinstance(x, Slice) and isinstance(y, Slice):
        return False
    index, run = (x, b) if isinstance(x, Affine) else (y, a)
    if run[1] != 1:
        return False
    distance = index - run[0]
    limits = distance.bounds()
    if distance.constant is not None:
        limits = (distance.constant, distance.constant)
    return limits is not None and (limits[1] < 0 or limits[0] >= run[2])


def must_alias(a: Ref, b: Ref) -> bool:
    """Return True if the places `a` and `b` name the same items at one moment.

    The places must share a root and have equal paths up to `same`. A root that
    stands for a family of items, such as an allocation in a loop, gives False,
    because two uses can belong to different runs of the allocation.
    """
    if not (isinstance(a, Place) and isinstance(b, Place)):
        return False
    if a.root != b.root or family(a.root) or len(a.path) != len(b.path):
        return False
    if a.one_item != b.one_item:
        return False
    for x, y in zip(a.path, b.path):
        if isinstance(x, Affine) and isinstance(y, Affine):
            if same(x, y) is not True:
                return False
        elif not (
            isinstance(x, Slice)
            and isinstance(y, Slice)
            and x.step == y.step
            and same(x.start, y.start) is True
            and x.length is not None
            and y.length is not None
            and same(x.length, y.length) is True
        ):
            return False
    return True


def disjoint(a: Ref, b: Ref) -> bool:
    """Return True if the places `a` and `b` share no item at one moment.

    Two different roots share no item if one of them is an allocation. Two places
    of one root share no item if a step of their paths lies apart. A place shares
    every item with each place below it.
    """
    if not (isinstance(a, Place) and isinstance(b, Place)):
        return False
    if a.root != b.root:
        return fresh(a.root) or fresh(b.root)
    return any(apart(x, y) for x, y in zip(a.path, b.path))
