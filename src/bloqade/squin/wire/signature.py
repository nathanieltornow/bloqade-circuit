"""Describe the calling convention of converted wire functions.

A source function borrows each quantum parameter. Its wire function returns the
ordinary source results first. Then it returns each quantum parameter in
declaration order, so that the caller gets the updated qubits back. For example,
the source call `answer = f(q, n)` becomes `answer, q_next = f_wire(q_current, n)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from kirin import ir, types
from kirin.dialects import func

from bloqade.analysis.indices import Affine

from .types import quantum_type


def output_types(output: types.TypeAttribute) -> tuple[types.TypeAttribute, ...]:
    """Return the type of each output of a function whose output type is `output`.

    `None` gives no output. A tuple gives one output per member, and any other type
    gives one output.
    """
    if output.is_subseteq(types.NoneType):
        return ()
    if isinstance(output, types.Generic) and output.is_subseteq(types.Tuple):
        return tuple(output.vars)
    return (output,)


def packed_type(outputs: Sequence[types.TypeAttribute]) -> types.TypeAttribute:
    """Return the output type of a function with the output types `outputs`."""
    if not outputs:
        return types.NoneType
    if len(outputs) == 1:
        return outputs[0]
    return types.Generic(tuple, *outputs)


@dataclass(frozen=True)
class Layout:
    """Describe the wire results that the signature of a source function implies."""

    borrowed: tuple[int, ...]
    """The argument positions of the quantum parameters."""
    ordinary: int
    """The number of ordinary source results."""
    packed: bool
    """Whether the source function returns its ordinary results as one tuple."""

    @classmethod
    def of(cls, signature: func.Signature) -> Layout:
        """Read the layout from the signature of a source function."""
        ordinary = len(output_types(signature.output))
        borrowed = tuple(
            i
            for i, kind in enumerate(signature.inputs)
            if quantum_type(kind) is not None
        )
        packed = (
            ordinary > 0
            and isinstance(signature.output, types.Generic)
            and signature.output.is_subseteq(types.Tuple)
        )
        return cls(borrowed, ordinary, packed)

    def results(self, output: types.TypeAttribute) -> tuple[types.TypeAttribute, ...]:
        """Return the type of each wire result for the wire output type `output`.

        A single wire result keeps its own type, even if that type is a tuple.
        """
        count = self.ordinary + len(self.borrowed)
        kinds = (output,) if count == 1 else output_types(output)
        if len(kinds) != count:
            raise ValueError(
                f"The wire output type {output} must have {count} results."
            )
        return kinds


@dataclass(frozen=True)
class Contract:
    """Hold what a caller needs from a converted function.

    The inferred signature of `method` gives the type of every wire result.
    """

    method: ir.Method
    """The converted wire function."""
    lengths: tuple[Affine | None, ...]
    """The static length of each ordinary result, in result order.

    A length is an affine expression in the integer parameters of `method`. The
    caller replaces those parameters with its own arguments. An entry is None for
    a classical result and for a register of unknown length.
    """
