"""Reuse callee results in the wire validation analyses."""

from abc import ABC
from typing import TypeVar
from dataclasses import field, dataclass

from kirin import ir, lattice
from kirin.analysis import Forward
from typing_extensions import Self
from kirin.analysis.forward import ForwardFrame

LatticeType = TypeVar("LatticeType", bound=lattice.BoundedLattice)


@dataclass
class CachedCalls(Forward[LatticeType], ABC):
    """Interpret each callee once for each list of abstract arguments.

    A wire validation analysis interprets the callee at each call. Without reuse, a
    chain of functions that each call the next function twice costs two to the
    power of the chain depth. An analysis result depends only on the callee and its
    arguments. The analysis records each validation error at a callee node, so a
    reused result loses no error.
    """

    calls: dict[tuple, tuple[ForwardFrame[LatticeType], LatticeType]] = field(
        default_factory=dict, init=False
    )
    """The frame and the result of each finished call, keyed by callee and arguments."""

    def initialize(self) -> Self:
        """Forget the results of the previous analysis run."""
        super().initialize()
        self.calls = {}
        return self

    def call(
        self, node: ir.Statement | ir.Method, *args: LatticeType, **kwargs: LatticeType
    ) -> tuple[ForwardFrame[LatticeType], LatticeType]:
        """Return the stored result of an equal call, or interpret the callee."""
        key = (node, args, tuple(kwargs.items()))
        if key not in self.calls:
            self.calls[key] = super().call(node, *args, **kwargs)
        return self.calls[key]
