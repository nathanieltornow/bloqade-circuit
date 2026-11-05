"""Name the calling convention of a wire function.

An `InOut` function returns its ordinary results and then each quantum parameter in
its position. Jeff import marks each function that it converts as `Owns`, because a
Jeff function takes its qubits and returns every quantum result in its own order.
"""

from typing import TypeAlias
from dataclasses import dataclass

from kirin import ir
from kirin.dialects import func

from .types import is_linear

KEY = "squin.wire.owned"
"""Name the function attribute that marks an `Owns` function."""


@dataclass(frozen=True)
class InOut:
    """Name a function that returns each of its `count` quantum parameters last."""

    count: int


@dataclass(frozen=True)
class Owns:
    """Name a function that consumes its quantum parameters."""


Convention: TypeAlias = InOut | Owns
"""Name the calling convention of a wire function."""


def convention(function: func.Function) -> Convention:
    """Return the calling convention of the wire function `function`."""
    if function.attributes.get(KEY) == ir.PyAttr(True):
        return Owns()
    return InOut(sum(is_linear(typ) for typ in function.signature.inputs))


def mark_owned(method: ir.Method) -> None:
    """Mark `method` as a function that consumes its quantum parameters."""
    method.code.attributes[KEY] = ir.PyAttr(True)


def clear_owned(method: ir.Method) -> None:
    """Remove the mark of `mark_owned`, so that `method` is an `InOut` function."""
    method.code.attributes.pop(KEY, None)
