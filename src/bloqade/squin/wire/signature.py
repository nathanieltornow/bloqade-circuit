"""Describe wire function results and borrowed parameter positions."""

from dataclasses import dataclass
from collections.abc import Sequence

from kirin import ir, types

from bloqade.analysis.indices import Affine


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
class Contract:
    """Describe a converted function's ordinary results and borrowed-parameter suffix."""

    method: ir.Method
    inputs: tuple[types.TypeAttribute, ...]
    ordinary: tuple[types.TypeAttribute, ...]
    borrowed: tuple[int, ...]
    lengths: tuple[Affine | None, ...]
    packed: bool

    @property
    def outputs(self) -> tuple[types.TypeAttribute, ...]:
        """Return all wire result types in their declared order."""
        return self.ordinary + tuple(self.inputs[i] for i in self.borrowed)
