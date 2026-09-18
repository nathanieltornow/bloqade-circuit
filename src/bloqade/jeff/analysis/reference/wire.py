"""This module holds the reference analysis for jeff wires and registers."""

from dataclasses import dataclass

from kirin import ir, types, interp
from kirin.analysis.forward import ForwardFrame

from bloqade.jeff.types import is_linear, qureg_length
from bloqade.jeff.dialects import stmts
from bloqade.jeff.constants import const_int

from .lattice import UNTRACKED, Ref, Whole, Unknown, Positions
from .analysis import KEY, ReferenceAnalysis, as_positions

WIRE_KEY = "jeff.reference"
"""The registry key of the rules for jeff statements."""


@dataclass
class WireReferenceAnalysis(ReferenceAnalysis):
    """A reference analysis whose roots are jeff wires and registers.

    A gate, a reset and a measurement hand each wire on to a result, so the result
    refers to the root of the wire.
    """

    keys = (WIRE_KEY, KEY)

    def is_tracked_type(self, type_: types.TypeAttribute) -> bool:
        """Return True for a wire and for a register."""
        return is_linear(type_)

    def register_length(self, value: ir.SSAValue) -> int | None:
        """Return the constant allocation size or the typed length of a register."""
        if isinstance(value, ir.ResultValue) and isinstance(
            owner := value.owner, stmts.RegAlloc
        ):
            return const_int(owner.size)
        return qureg_length(value.type)


@stmts.wire.dialect.register(key=WIRE_KEY)
class _Wire(interp.MethodTable):
    """A method table that follows wires through allocation, extraction and insert."""

    @interp.impl(stmts.Alloc)
    @interp.impl(stmts.RegAlloc)
    def alloc(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Alloc | stmts.RegAlloc,
    ) -> tuple[Ref, ...]:
        """Return the new wire or register as a whole root."""
        return (Whole(stmt.result),)

    @interp.impl(stmts.Reset)
    def reset(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Reset,
    ) -> tuple[Ref, ...]:
        """Hand the wire on."""
        return (frame.get(stmt.wire),)

    @interp.impl(stmts.MeasureNd)
    def measure_nd(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.MeasureNd,
    ) -> tuple[Ref, ...]:
        """Hand the wire on and make the bit `Untracked`."""
        return (frame.get(stmt.wire), UNTRACKED)

    @interp.impl(stmts.RegLength)
    def length(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.RegLength,
    ) -> tuple[Ref, ...]:
        """Hand the register on and make the length `Untracked`."""
        return (frame.get(stmt.reg), UNTRACKED)

    @interp.impl(stmts.Extract)
    def extract(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Extract,
    ) -> tuple[Ref, ...]:
        """Hand the register on and make the wire a `Slot` of the register."""
        register = frame.get(stmt.reg)
        if not isinstance(register, Whole):
            return (register, Unknown("a wire of a register that has no root"))
        return (register, analysis.slot(register.root, stmt.index))

    @interp.impl(stmts.Insert)
    def insert(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Insert,
    ) -> tuple[Ref, ...]:
        """Hand the register on if the wire returns to the slot that it came from."""
        register, wire = frame.get(stmt.reg), frame.get(stmt.wire)
        if isinstance(register, Whole) and wire == analysis.slot(
            register.root, stmt.index
        ):
            return (register,)
        return (Unknown("a register that holds a wire from another slot"),)


@stmts.gate.dialect.register(key=WIRE_KEY)
class _Gate(interp.MethodTable):
    """A method table that hands each wire of a gate on to its result."""

    @interp.impl(stmts.Gate)
    @interp.impl(stmts.Ppr)
    def gate(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Gate | stmts.Ppr,
    ) -> tuple[Ref, ...]:
        """Return the references of the targets and then of the controls."""
        return frame.get_values((*stmt.targets, *stmt.controls))


@stmts.scf.dialect.register(key=WIRE_KEY)
class _Scf(interp.MethodTable):
    """A method table that joins the references that jeff loops and switches carry."""

    @interp.impl(stmts.For)
    def for_(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.For,
    ) -> tuple[Ref, ...]:
        """Run the body until the carried references stop changing."""
        carried = frame.get_values(stmt.state)
        return analysis.run_loop(frame, stmt, stmt.body, carried)

    @interp.impl(stmts.Switch)
    def switch(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Switch,
    ) -> tuple[Ref, ...]:
        """Join the references that the branches and the default yield."""
        inputs = frame.get_values(stmt.inputs)
        return analysis.run_branches(frame, stmt, stmt.regions, inputs)

    @interp.impl(stmts.Yield)
    def yield_(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Yield,
    ) -> interp.YieldValue[Ref]:
        """End the region with the yielded references."""
        return interp.YieldValue(frame.get_values(stmt.values))


@stmts.call.dialect.register(key=WIRE_KEY)
class _Call(interp.MethodTable):
    """A method table that translates the result of the jeff callee at each call."""

    @interp.impl(stmts.Call)
    def call(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Call,
    ) -> tuple[Ref, ...]:
        """Return the references of the outputs in the terms of the caller."""
        result = analysis.result_of(stmt.callee, frame)
        outputs = as_positions(result, len(stmt.results))
        return analysis.translate(stmt, frame.get_values(stmt.inputs), outputs)

    @interp.impl(stmts.Return)
    def return_(
        self,
        analysis: WireReferenceAnalysis,
        frame: ForwardFrame[Ref],
        stmt: stmts.Return,
    ) -> interp.ReturnValue[Ref]:
        """Return the references of the returned values as `Positions`."""
        return interp.ReturnValue(Positions(frame.get_values(stmt.values)))
