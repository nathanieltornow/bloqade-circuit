"""This module holds the depth rules of the jeff statements.

Each method table registers its rules under `DEPTH_KEY` for one jeff dialect.
"""

from dataclasses import field, dataclass

from kirin import ir, interp
from kirin.analysis.forward import ForwardFrame

from bloqade.constants import constant_int
from bloqade.jeff.dialects import stmts

from . import maxplus
from .lattice import (
    NONE,
    ZERO,
    Depth,
    Bottom,
    Results,
    Unknown,
    RegDepth,
    Classical,
    WireDepth,
)
from .maxplus import Form
from .analysis import DEPTH_KEY, DepthAnalysis


def _wire(analysis: DepthAnalysis, value: Depth, what: str) -> Form | Unknown:
    """Return the form of a wire depth, or `Unknown` for any other value."""
    if isinstance(value, WireDepth):
        return value.form
    if isinstance(value, Unknown):
        return value
    return analysis.unknown(f"{what} without a wire depth")


def _register(analysis: DepthAnalysis, value: Depth, what: str) -> RegDepth | Unknown:
    """Return a register depth, or `Unknown` for any other value."""
    if isinstance(value, RegDepth):
        return value
    if isinstance(value, Unknown):
        return value
    return analysis.unknown(f"{what} without a register depth")


def _index(analysis: DepthAnalysis, value: ir.SSAValue, length: int) -> int | Unknown:
    """Return the constant index `value` if it lies in `0..length`, else `Unknown`.

    The upper end `length` is inclusive, so the result also serves as a slice bound.
    """
    index = constant_int(value)
    if index is None:
        return analysis.unknown("a register index that is not a constant")
    if not 0 <= index <= length:
        return analysis.unknown(f"a register index {index} outside 0..{length}")
    return index


@stmts.wire.dialect.register(key=DEPTH_KEY)
class _Wire(interp.MethodTable):
    """A method table for allocation, measurement, release and register statements."""

    @interp.impl(stmts.Alloc)
    def alloc(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Alloc
    ) -> tuple[Depth, ...]:
        """Return depth 0 for the fresh wire."""
        return (WireDepth(ZERO),)

    @interp.impl(stmts.Reset)
    def reset(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Reset
    ) -> tuple[Depth, ...]:
        """Add the cost of the reset to the depth of the wire."""
        form = _wire(analysis, frame.get(stmt.wire), "a reset")
        if isinstance(form, Unknown):
            return (form,)
        return (WireDepth(form.shift(analysis.cost(stmt))),)

    @interp.impl(stmts.MeasureNd)
    def measure_nd(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.MeasureNd
    ) -> tuple[Depth, ...]:
        """Add the cost of the measurement to the wire and return a classical bit."""
        form = _wire(analysis, frame.get(stmt.wire), "a measurement")
        if isinstance(form, Unknown):
            return (form, Classical())
        return (WireDepth(form.shift(analysis.cost(stmt))), Classical())

    @interp.impl(stmts.Measure)
    @interp.impl(stmts.Free)
    @interp.impl(stmts.FreeZero)
    def consume(
        self,
        analysis: DepthAnalysis,
        frame: ForwardFrame[Depth],
        stmt: stmts.Measure | stmts.Free | stmts.FreeZero,
    ) -> tuple[Depth, ...]:
        """Join the depth of the wire plus the cost of `stmt` into the sink."""
        form = _wire(analysis, frame.get(stmt.wire), f"a '{stmt.name}' statement")
        if not isinstance(form, Unknown):
            analysis.sink = analysis.sink.join(form.shift(analysis.cost(stmt)))
        return tuple(Classical() for _ in stmt.results)

    @interp.impl(stmts.RegAlloc)
    def reg_alloc(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.RegAlloc
    ) -> tuple[Depth, ...]:
        """Return depth 0 for every slot of a register of constant size."""
        size = constant_int(stmt.size)
        if size is None or size < 0:
            return (analysis.unknown("a register of non-constant size"),)
        return (RegDepth((ZERO,) * size),)

    @interp.impl(stmts.RegCreate)
    def reg_create(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.RegCreate
    ) -> tuple[Depth, ...]:
        """Return a register whose slots hold the depths of the wires."""
        forms = [_wire(analysis, frame.get(w), "a register wire") for w in stmt.wires]
        for form in forms:
            if isinstance(form, Unknown):
                return (form,)
        return (RegDepth(tuple(forms)),)  # type: ignore[arg-type]

    @interp.impl(stmts.RegFree)
    @interp.impl(stmts.RegFreeZero)
    def reg_free(
        self,
        analysis: DepthAnalysis,
        frame: ForwardFrame[Depth],
        stmt: stmts.RegFree | stmts.RegFreeZero,
    ) -> tuple[Depth, ...]:
        """Join the depth of every filled slot into the sink."""
        analysis.consume(frame.get(stmt.reg))
        return ()

    @interp.impl(stmts.RegLength)
    def length(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.RegLength
    ) -> tuple[Depth, ...]:
        """Hand the register on and return a classical length."""
        return (frame.get(stmt.reg), Classical())

    @interp.impl(stmts.Extract)
    def extract(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Extract
    ) -> tuple[Depth, ...]:
        """Take the wire out of a constant slot and leave the slot empty."""
        reg = _register(analysis, frame.get(stmt.reg), "an extract")
        if isinstance(reg, Unknown):
            return (reg, reg)
        index = _index(analysis, stmt.index, len(reg.slots) - 1)
        if isinstance(index, Unknown):
            return (index, index)
        form = reg.slots[index]
        if form is None:
            failed = analysis.unknown(f"an extract from the empty slot {index}")
            return (failed, failed)
        slots = list(reg.slots)
        slots[index] = None
        return (RegDepth(tuple(slots)), WireDepth(form))

    @interp.impl(stmts.Insert)
    def insert(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Insert
    ) -> tuple[Depth, ...]:
        """Put the wire into an empty constant slot."""
        reg = _register(analysis, frame.get(stmt.reg), "an insert")
        if isinstance(reg, Unknown):
            return (reg,)
        form = _wire(analysis, frame.get(stmt.wire), "an insert")
        if isinstance(form, Unknown):
            return (form,)
        index = _index(analysis, stmt.index, len(reg.slots) - 1)
        if isinstance(index, Unknown):
            return (index,)
        if reg.slots[index] is not None:
            return (analysis.unknown(f"an insert into the filled slot {index}"),)
        slots = list(reg.slots)
        slots[index] = form
        return (RegDepth(tuple(slots)),)

    @interp.impl(stmts.ExtractSlice)
    def extract_slice(
        self,
        analysis: DepthAnalysis,
        frame: ForwardFrame[Depth],
        stmt: stmts.ExtractSlice,
    ) -> tuple[Depth, ...]:
        """Move the slots `start` to `start + length` into a new register."""
        reg = _register(analysis, frame.get(stmt.reg), "a slice extract")
        if isinstance(reg, Unknown):
            return (reg, reg)
        start = _index(analysis, stmt.start, len(reg.slots))
        if isinstance(start, Unknown):
            return (start, start)
        stop = _index(analysis, stmt.length, len(reg.slots) - start)
        if isinstance(stop, Unknown):
            return (stop, stop)
        stop += start
        rest = (*reg.slots[:start], *([None] * (stop - start)), *reg.slots[stop:])
        return (RegDepth(rest), RegDepth(reg.slots[start:stop]))

    @interp.impl(stmts.InsertSlice)
    def insert_slice(
        self,
        analysis: DepthAnalysis,
        frame: ForwardFrame[Depth],
        stmt: stmts.InsertSlice,
    ) -> tuple[Depth, ...]:
        """Move the slots of a register into empty slots from `start` on."""
        reg = _register(analysis, frame.get(stmt.reg), "a slice insert")
        if isinstance(reg, Unknown):
            return (reg,)
        piece = _register(analysis, frame.get(stmt.slice_reg), "a slice insert")
        if isinstance(piece, Unknown):
            return (piece,)
        start = _index(analysis, stmt.start, len(reg.slots) - len(piece.slots))
        if isinstance(start, Unknown):
            return (start,)
        stop = start + len(piece.slots)
        if any(slot is not None for slot in reg.slots[start:stop]):
            return (analysis.unknown("a slice insert into filled slots"),)
        return (RegDepth((*reg.slots[:start], *piece.slots, *reg.slots[stop:])),)

    @interp.impl(stmts.RegSplit)
    def split(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.RegSplit
    ) -> tuple[Depth, ...]:
        """Split the slots at a constant index into two registers."""
        reg = _register(analysis, frame.get(stmt.reg), "a register split")
        if isinstance(reg, Unknown):
            return (reg, reg)
        index = _index(analysis, stmt.index, len(reg.slots))
        if isinstance(index, Unknown):
            return (index, index)
        return (RegDepth(reg.slots[:index]), RegDepth(reg.slots[index:]))

    @interp.impl(stmts.RegJoin)
    def join(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.RegJoin
    ) -> tuple[Depth, ...]:
        """Join the slots of two registers into one register."""
        first = _register(analysis, frame.get(stmt.first), "a register join")
        if isinstance(first, Unknown):
            return (first,)
        second = _register(analysis, frame.get(stmt.second), "a register join")
        if isinstance(second, Unknown):
            return (second,)
        return (RegDepth((*first.slots, *second.slots)),)


@stmts.gate.dialect.register(key=DEPTH_KEY)
class _Gate(interp.MethodTable):
    """A method table that applies the depth rule of a gate."""

    @interp.impl(stmts.Gate)
    @interp.impl(stmts.Ppr)
    def gate(
        self,
        analysis: DepthAnalysis,
        frame: ForwardFrame[Depth],
        stmt: stmts.Gate | stmts.Ppr,
    ) -> tuple[Depth, ...]:
        """Give every output wire the max input depth plus the cost of the gate."""
        wires = (*stmt.targets, *stmt.controls)
        depth = NONE
        for wire in wires:
            form = _wire(analysis, frame.get(wire), f"a '{stmt.name}' input")
            if isinstance(form, Unknown):
                return tuple(form for _ in wires)
            depth = depth.join(form)
        result = WireDepth(depth.shift(analysis.cost(stmt)))
        return tuple(result for _ in wires)


def _trip_count(stmt: stmts.For) -> int | None:
    """Return the number of iterations of a loop with constant bounds, else None."""
    start, stop, step = (constant_int(v) for v in (stmt.start, stmt.stop, stmt.step))
    if start is None or stop is None or step is None or step == 0:
        return None
    return len(range(start, stop, step))


@dataclass
class _Carried:
    """The loop-carried wires of a `For` loop, flattened into numbered slots.

    `variables` holds one fresh variable per wire and per filled register slot.
    `incoming` holds the depth that enters the loop for each variable.
    """

    variables: list[int] = field(default_factory=list)
    incoming: list[Form] = field(default_factory=list)

    def symbolic(self, analysis: DepthAnalysis, value: Depth) -> Depth:
        """Return `value` with a fresh variable in place of each wire depth."""
        if isinstance(value, WireDepth):
            return WireDepth(self._add(analysis, value.form))
        if isinstance(value, RegDepth):
            return RegDepth(
                tuple(
                    None if slot is None else self._add(analysis, slot)
                    for slot in value.slots
                )
            )
        return value

    def _add(self, analysis: DepthAnalysis, form: Form) -> Form:
        variable = analysis.fresh()
        self.variables.append(variable)
        self.incoming.append(form)
        return Form.var(variable)


def _flatten(entering: Depth, leaving: Depth) -> list[Form] | None:
    """Return the forms of the wires in `leaving`, or None if its shape differs.

    The shape of `leaving` must equal the shape of `entering`. A wire stays a wire,
    a register keeps its length and its empty slots, and a classical value stays
    classical.
    """
    if isinstance(entering, WireDepth) and isinstance(leaving, WireDepth):
        return [leaving.form]
    if isinstance(entering, RegDepth) and isinstance(leaving, RegDepth):
        if len(entering.slots) != len(leaving.slots):
            return None
        forms: list[Form] = []
        for before, after in zip(entering.slots, leaving.slots):
            if (before is None) != (after is None):
                return None
            if after is not None:
                forms.append(after)
        return forms
    if isinstance(entering, Classical) and isinstance(leaving, Classical):
        return []
    return None


def _unflatten(template: Depth, forms: list[Form]) -> Depth:
    """Return `template` with its wire depths replaced, in order, by `forms`.

    The function removes the forms that it uses from the front of `forms`.
    """
    if isinstance(template, WireDepth):
        return WireDepth(forms.pop(0))
    if isinstance(template, RegDepth):
        return RegDepth(
            tuple(None if slot is None else forms.pop(0) for slot in template.slots)
        )
    return template


@stmts.scf.dialect.register(key=DEPTH_KEY)
class _Scf(interp.MethodTable):
    """A method table for the structured control flow of jeff."""

    @interp.impl(stmts.For)
    def for_(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.For
    ) -> tuple[Depth, ...]:
        """Apply the max-plus map of one iteration `n` times, by repeated squaring."""
        state = frame.get_values(stmt.state)
        for value in state:
            if isinstance(value, Unknown):
                return tuple(value for _ in stmt.results)
        n = _trip_count(stmt)
        if n is None:
            failed = analysis.unknown("a for loop without constant bounds")
            return tuple(failed for _ in stmt.results)
        if n == 0:
            return state

        carried = _Carried()
        symbolic = [carried.symbolic(analysis, value) for value in state]
        # The sink joins into the carried state, so that the map stays linear when
        # the body consumes a wire and allocates a fresh wire.
        sink = analysis.fresh()
        outer_sink = analysis.sink
        analysis.sink = Form.var(sink)
        yielded = analysis.run_region(stmt, stmt.body, (Classical(), *symbolic))
        body_sink = analysis.sink
        analysis.sink = outer_sink
        if yielded is None:
            failed = analysis.unknown("a for loop body that does not yield")
            return tuple(failed for _ in stmt.results)

        rows: list[Form] = []
        for entering, leaving in zip(symbolic, yielded, strict=True):
            if isinstance(leaving, Unknown):
                return tuple(leaving for _ in stmt.results)
            forms = _flatten(entering, leaving)
            if forms is None:
                failed = analysis.unknown("a for loop body that changes a register")
                return tuple(failed for _ in stmt.results)
            rows.extend(forms)
        rows.append(body_sink)

        variables = [*carried.variables, sink]
        matrix = [[row.offset(v) for v in variables] for row in rows]
        offset = [row.const for row in rows]
        raised, raised_offset = maxplus.affine_power(matrix, offset, n)

        env = dict(zip(variables, [*carried.incoming, NONE]))
        final = [
            Form(
                const, tuple((v, a) for v, a in zip(variables, row) if a is not None)
            ).substitute(env)
            for row, const in zip(raised, raised_offset)
        ]
        analysis.sink = analysis.sink.join(final.pop())
        return tuple(_unflatten(value, final) for value in state)

    @interp.impl(stmts.Switch)
    def switch(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Switch
    ) -> tuple[Depth, ...]:
        """Run the selected region, or join every region if the selector varies."""
        inputs = frame.get_values(stmt.inputs)
        selector = constant_int(stmt.selector)
        if selector is not None:
            cases = stmt.branches
            region = cases[selector] if 0 <= selector < len(cases) else stmt.default
            regions = [region]
        else:
            regions = list(stmt.regions)

        outer_sink = analysis.sink
        sink = outer_sink
        joined: tuple[Depth, ...] = tuple(Bottom() for _ in stmt.results)
        for region in regions:
            analysis.sink = outer_sink
            yielded = analysis.run_region(stmt, region, inputs)
            sink = sink.join(analysis.sink)
            if yielded is None:
                failed = analysis.unknown("a switch branch that does not yield")
                analysis.sink = sink
                return tuple(failed for _ in stmt.results)
            joined = tuple(a.join(b) for a, b in zip(joined, yielded, strict=True))
        analysis.sink = sink
        for value in joined:
            if isinstance(value, Unknown) and analysis.failure is None:
                analysis.failure = value
        return joined

    @interp.impl(stmts.While)
    def while_(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.While
    ) -> tuple[Depth, ...]:
        """Return `Unknown`, because a while loop has no static trip count."""
        failed = analysis.unknown("a while loop has no static trip count")
        return tuple(failed for _ in stmt.results)

    @interp.impl(stmts.Yield)
    def yield_(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Yield
    ) -> interp.YieldValue[Depth]:
        """End the region with the yielded depths."""
        return interp.YieldValue(frame.get_values(stmt.values))


@stmts.call.dialect.register(key=DEPTH_KEY)
class _Call(interp.MethodTable):
    """A method table that runs the callee at each call with the caller's depths."""

    @interp.impl(stmts.Call)
    def call(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Call
    ) -> tuple[Depth, ...]:
        """Run the callee with the depths of the inputs and return its results."""
        code = stmt.callee.code
        if code in analysis._active:
            failed = analysis.unknown(f"a recursive call of '{stmt.callee.sym_name}'")
            return tuple(failed for _ in stmt.results)
        analysis._active.add(code)
        try:
            inputs = frame.get_values(stmt.inputs)
            _, result = analysis.call(code, Classical(), *inputs)
        finally:
            analysis._active.discard(code)
        if isinstance(result, Results) and len(result.values) == len(stmt.results):
            return result.values
        failed = analysis.unknown("a call whose return does not match its results")
        return tuple(failed for _ in stmt.results)

    @interp.impl(stmts.Return)
    def return_(
        self, analysis: DepthAnalysis, frame: ForwardFrame[Depth], stmt: stmts.Return
    ) -> interp.ReturnValue[Depth]:
        """Return the depths of the returned values as `Results`."""
        return interp.ReturnValue(Results(frame.get_values(stmt.values)))
