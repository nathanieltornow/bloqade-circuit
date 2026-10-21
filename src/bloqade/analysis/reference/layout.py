"""This module holds the flat layout of a place in one register of its root.

Jeff keeps the qubits of a root in one register. A register of registers becomes
one register whose rows lie one after another. `flat` gives the position of a
place in that register. The squin-to-jeff emitter and its validation both use it.
"""

from .index import Slice, Affine
from .lattice import Root, Place
from .analysis import origin_of, typed_length


def strides(root: Root, levels: int) -> tuple[int, ...] | None:
    """Return the count of qubits that one item spans at each level of `root`.

    For a register of 3 rows of 2 qubits the result is `(2, 1)`. A row spans 2
    qubits, and a qubit spans 1. The result is None if an inner level has no
    static length, because the jeff register then has no fixed layout.
    """
    origin, _ = origin_of(root)
    spans = [1]
    for level in range(levels - 1, 0, -1):
        length = typed_length(origin.type, level)
        if length is None:
            return None
        spans.insert(0, spans[0] * length)
    return tuple(spans)


def flat(ref: Place) -> Affine | Slice | None:
    """Return where the place `ref` lies in the one jeff register of its root.

    One qubit gives its index in the register. A register place gives the
    slice of the register that holds its qubits. For example, with rows of 2
    qubits, `qss[i][j]` lies at `2*i + j`, and the row `qss[i]` is the slice
    that starts at `2*i` with 2 qubits. The result is None if the root has no
    fixed layout, or if the qubits of a register place do not lie next to
    each other in a step of 1 or a single step.
    """
    path = ref.path
    levels = ref.depth + sum(isinstance(step, Affine) for step in path)
    spans = strides(ref.root, levels)
    if spans is None:
        return None
    base = Affine(0)
    for level, step in enumerate(path):
        if isinstance(step, Slice):
            span = spans[level]
            if span == 1:
                return Slice(base + step.start, step.step, step.length)
            if step.step != 1 or step.length is None:
                return None
            return Slice(base + step.start.scale(span), 1, step.length.scale(span))
        base = base + step.scale(spans[level])
    if ref.one_item:
        return base
    if not path:
        # A whole root spans its own length times the span of one item.
        origin, _ = origin_of(ref.root)
        length = typed_length(origin.type, 0)
        return None if length is None else Slice(base, 1, Affine(length * spans[0]))
    return Slice(base, 1, Affine(spans[len(path) - 1]))
