"""This module holds the depth analysis and the `circuit_depth` entry point."""

import itertools
from typing import ClassVar
from dataclasses import field, dataclass
from collections.abc import Callable, Sequence

from kirin import ir, types
from kirin.analysis.forward import Forward, ForwardFrame

from bloqade.jeff.types import WireType, QuregType, is_subtype, qureg_length
from bloqade.jeff.dialects import stmts, kernel

from .lattice import NONE, ZERO, Depth, Results, Unknown, RegDepth, Classical, WireDepth
from .maxplus import Form

DEPTH_KEY = "jeff.depth"
"""The registry key of the rules for jeff statements."""


def unit_cost(stmt: ir.Statement) -> int:
    """Return 1 for a gate, a Pauli-product rotation, a reset or a measurement, else 0."""
    costly = (stmts.Gate, stmts.Ppr, stmts.Reset, stmts.MeasureNd, stmts.Measure)
    return 1 if isinstance(stmt, costly) else 0


@dataclass
class DepthAnalysis(Forward[Depth]):
    """A forward analysis that computes the exact circuit depth of a jeff program.

    `cost` gives the depth that a statement adds to its wires. The default is
    `unit_cost`. After `run`, `sink` holds the circuit depth as a form without
    variables, and `failure` holds the first `Unknown` that the analysis met.
    """

    keys: ClassVar[tuple[str, ...]] = (DEPTH_KEY,)
    lattice = Depth

    cost: Callable[[ir.Statement], int] = field(default=unit_cost)
    sink: Form = field(default=NONE, init=False)
    """The max depth over every wire that the program consumed so far."""
    failure: Unknown | None = field(default=None, init=False)
    _active: set[ir.Statement] = field(default_factory=set, init=False)
    """The functions that the analysis runs at the moment, to detect recursion."""
    _variables: itertools.count = field(default_factory=itertools.count, init=False)

    def initialize(self) -> "DepthAnalysis":
        """Reset the interpreter state, the sink and the failure."""
        super().initialize()
        self.sink = NONE
        self.failure = None
        self._active = set()
        self._variables = itertools.count()
        return self

    def run(self, method: ir.Method, *args: Depth, **kwargs: Depth):
        """Analyze `method` and every function that it calls.

        Each wire parameter starts at depth 0. A register parameter of static length
        starts with every slot at depth 0. The returned wires join the sink.
        """
        params = [
            self.parameter(arg.type)
            for arg in method.callable_region.blocks[0].args[1:]
        ]
        with self.eval_context():
            frame, result = self.call(method.code, self.method_self(method), *params)
            if isinstance(result, Results):
                for value in result.values:
                    self.consume(value)
        return frame, result

    def method_self(self, method: ir.Method) -> Depth:
        """Return `Classical` for the method object."""
        return Classical()

    def eval_fallback(self, frame: ForwardFrame[Depth], node: ir.Statement):
        """Return `Classical` for each result of a statement without a depth rule.

        A statement without a rule that returns a wire or a register gives `Unknown`.
        """
        if any(_quantum(result.type) for result in node.results):
            return tuple(
                self.unknown(f"a '{node.name}' statement without a depth rule")
                for _ in node.results
            )
        return tuple(Classical() for _ in node.results)

    def parameter(self, type_: types.TypeAttribute) -> Depth:
        """Return the depth of a function parameter of type `type_` at the program start."""
        if is_subtype(type_, WireType):
            return WireDepth(ZERO)
        if is_subtype(type_, QuregType):
            length = qureg_length(type_)
            if length is None:
                return self.unknown("a register parameter of unknown length")
            return RegDepth((ZERO,) * length)
        return Classical()

    def unknown(self, reason: str) -> Unknown:
        """Return `Unknown(reason)` and record it as the failure if it is the first one."""
        result = Unknown(reason)
        if self.failure is None:
            self.failure = result
        return result

    def fresh(self) -> int:
        """Return a variable that no form in the analysis uses yet."""
        return next(self._variables)

    def consume(self, value: Depth) -> None:
        """Join the depth of every wire in `value` into the sink."""
        if isinstance(value, WireDepth):
            self.sink = self.sink.join(value.form)
        elif isinstance(value, RegDepth):
            for slot in value.slots:
                if slot is not None:
                    self.sink = self.sink.join(slot)

    def run_region(
        self, stmt: ir.Statement, region: ir.Region, args: Sequence[Depth]
    ) -> tuple[Depth, ...] | None:
        """Run `region` of `stmt` with `args` and return the yielded values.

        The method returns None if the region does not yield.
        """
        with self.new_frame(stmt, has_parent_access=True) as inner:
            yielded = self.frame_call_region(inner, stmt, region, *args)
        return yielded if isinstance(yielded, tuple) else None


def _quantum(type_: types.TypeAttribute) -> bool:
    """Return True if `type_` is a wire or a register type."""
    return is_subtype(type_, WireType) or is_subtype(type_, QuregType)


def circuit_depth(
    method: ir.Method, cost: Callable[[ir.Statement], int] = unit_cost
) -> int | Unknown:
    """Return the exact circuit depth of the jeff program `method`.

    `cost` gives the depth that each statement adds, as in `DepthAnalysis`. A program
    that consumes no wire has depth 0. If the analysis meets a construct without an
    exact depth, the function returns that `Unknown` with its reason.
    """
    analysis = DepthAnalysis(kernel, cost=cost)
    analysis.run(method)
    if analysis.failure is not None:
        return analysis.failure
    return analysis.sink.value() or 0
