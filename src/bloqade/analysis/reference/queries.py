"""This module holds queries that compare the places of the reference analysis.

`must_alias` proves that two places name the same items. `disjoint` proves that two
places share no item. Each query answers False when it cannot prove its claim, so a
pass can use the answer to merge or to separate the state of qubits.

`must_alias` and `disjoint` compare two places at one moment. A symbol of an index
has one value in both places. Inside a loop, a shared loop symbol stands for one
iteration. So `disjoint(qs[i], qs[i + 1])` is True, although `qs[i + 1]` of
iteration 0 is `qs[i]` of iteration 1. `disjoint_across_iterations` compares two
places in any two iterations of a loop.
"""

from collections.abc import Mapping

from kirin import ir
from kirin.dialects import scf, func, ilist

from .index import Sym, Iter, Slice, Affine, Renamed, same
from .lattice import Ref, Root, Step, Place, Returned, place
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

    Each symbol of an index has one value in both places. Inside a loop, a shared
    loop symbol stands for one iteration. So `qs[i]` must alias `qs[i]` in one
    iteration, and the two can name different items in two iterations.

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

    Each symbol of an index has one value in both places. Inside a loop, a shared
    loop symbol stands for one iteration. So `qs[i]` and `qs[i + 1]` are disjoint,
    although `qs[i + 1]` of iteration 0 is `qs[i]` of iteration 1. Use
    `disjoint_across_iterations` to compare two iterations.

    Two different roots share no item if one of them is an allocation. Two places
    of one root share no item if a step of their paths lies apart. A place shares
    every item with each place below it.
    """
    if not (isinstance(a, Place) and isinstance(b, Place)):
        return False
    if a.root != b.root:
        return fresh(a.root) or fresh(b.root)
    return any(apart(x, y) for x, y in zip(a.path, b.path))


def disjoint_across_iterations(a: Ref, b: Ref, loop: scf.For) -> bool:
    """Return True if `a` and `b` share no item in any two iterations of `loop`.

    `a` and `b` are places of values inside the body of `loop`. The result is
    True if `a` in iteration `j` and `b` in iteration `k` share no item, for
    every `j` and every `k`. The two iterations can also be equal.

    The query gives each symbol that `loop` defines a new copy in `b`. Such a
    symbol is the loop symbol of `loop`, of a loop inside it, or a value of its
    body. Then it runs `disjoint`. A symbol from outside `loop` keeps one value
    in all iterations, so it stays shared. For example, `qs[2*i]` and
    `qs[2*i + 1]` are disjoint across iterations, and `qs[i]` and `qs[i + 1]`
    are not.
    """
    if not isinstance(b, Place):
        return False
    symbols = {sym for step in b.path for sym in _symbols(step)}
    copies = {sym: Affine.of(Renamed(sym)) for sym in symbols if _defined_in(sym, loop)}
    path = tuple(_rename(step, copies) for step in b.path)
    return disjoint(a, place(b.root, path, b.depth))


def _symbols(step: Step) -> frozenset[Sym]:
    """Return the symbols of an index, or of the start and the length of a slice."""
    if isinstance(step, Affine):
        return step.symbols()
    length = frozenset() if step.length is None else step.length.symbols()
    return step.start.symbols() | length


def _rename(step: Step, copies: Mapping[Sym, Affine]) -> Step:
    """Return `step` with each symbol that `copies` holds replaced by its copy."""
    if isinstance(step, Affine):
        return step.substitute(copies)
    length = None if step.length is None else step.length.substitute(copies)
    return Slice(step.start.substitute(copies), step.step, length)


def _defined_in(sym: Sym, loop: scf.For) -> bool:
    """Return True if `loop` defines the symbol `sym`, so that it can change per run.

    `loop` defines its own loop symbol, the loop symbols of the loops inside it and
    each value of its body. A result of `loop` itself lies outside the body. A
    loop variable that its loop yields unchanged equals its initializer, so the
    initializer decides.
    """
    match sym:
        case Iter(stmt=stmt):
            node: ir.Statement | None = stmt
        case ir.ResultValue(owner=owner):
            node = owner.parent_stmt
        case ir.BlockArgument(owner=block):
            if (initializer := _passed_through(sym)) is not None:
                return _defined_in(initializer, loop)
            node = block.parent_stmt
        case _:
            return False
    while node is not None:
        if node is loop:
            return True
        node = node.parent_stmt
    return False


def _passed_through(arg: ir.BlockArgument) -> ir.SSAValue | None:
    """Return the initializer of a loop variable that its loop yields unchanged.

    kirin carries each outer variable that a loop body reads, also if the body
    never assigns it. Such a variable has one value in all iterations. The
    result is None for any other block argument.
    """
    loop = arg.owner.parent_stmt
    if not isinstance(loop, scf.For) or arg.index == 0:
        return None
    end = arg.owner.last_stmt
    if isinstance(end, scf.Yield) and end.values[arg.index - 1] is arg:
        return loop.initializers[arg.index - 1]
    return None
