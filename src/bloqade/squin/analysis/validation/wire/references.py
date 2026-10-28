"""Check how wire squin uses qubit references with `ReferenceValidation`.

`Unwrap` takes the qubit of a reference out as a wire, and `Wrap` puts it back. At
each point of a block, some places are out: their qubits travel as wires. The
rule has four parts:

- `Unwrap` must not take out a place that may overlap a place that is out.
- `Wrap` must put back a place that is out.
- No other statement may touch a place that is out, except statements that only
  build references, such as `qs[i]`.
- A body must end with the same places out as it started with.
- A function may return only while the qubits that are out leave through its
  return value, as the wires of its borrowed parameters do.

A `Reference` starts with its place out, and the `Wrap` right after it fills it.

`ReferenceCheck` is a kirin forward analysis. It reads the places of
`QubitReferenceAnalysis` and keeps the places that are out in `out`.
"""

from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir, interp
from kirin.lattice import EmptyLattice
from kirin.analysis import Forward
from kirin.dialects import py, scf, func
from kirin.validation import ValidationPass
from kirin.analysis.forward import ForwardFrame

from bloqade.squin.wire import stmts
from bloqade.analysis.reference import (
    Ref,
    Place,
    Untracked,
    disjoint,
    places_in,
    same_place,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

KEY = "squin.wire.references"
"""Name the registry key of the method tables of `ReferenceCheck`."""

TAKE = (stmts.Unwrap, stmts.UnwrapRegister)
PUT = (stmts.Wrap, stmts.WrapRegister)
NEW = (stmts.Reference, stmts.ReferenceRegister)


@dataclass
class ReferenceCheck(Forward[EmptyLattice]):
    """Record a validation error for each statement that breaks the reference rule."""

    keys = (KEY, "absint")
    lattice = EmptyLattice
    refs: Mapping[ir.SSAValue, Ref] = field(default_factory=dict)
    out: list[tuple[Place, ir.SSAValue | None]] = field(
        default_factory=list, init=False
    )
    """The places whose qubits travel as wires, each with the wire that took it out.

    A new `Reference` starts out without a wire.
    """

    def initialize(self):
        """Start a run with no place out."""
        super().initialize()
        self.out = []
        return self

    def run(self, method: ir.Method, *args: EmptyLattice, **kwargs: EmptyLattice):
        """Compute the places of `method` if `refs` is empty, then check it."""
        if not self.refs:
            frame, _ = QubitReferenceAnalysis(method.dialects).run(method)
            self.refs = frame.entries
        return super().run(method, *args, **kwargs)

    def method_self(self, method: ir.Method) -> EmptyLattice:
        """Return the top element for the method object."""
        return EmptyLattice.top()

    def error(self, node: ir.Statement, message: str) -> None:
        """Record the validation error `message` at `node`."""
        self.add_validation_error(node, ir.ValidationError(node, message))

    def place(self, node: ir.Statement, value: ir.SSAValue) -> Place | None:
        """Return the place of the reference `value`, or record an error."""
        ref = self.refs.get(value)
        if isinstance(ref, Place):
            return ref
        self.error(node, f"'{node.name}' reads a reference that no place names")
        return None

    def results(self, node: ir.Statement) -> tuple[EmptyLattice, ...]:
        """Return the top element for each result of `node`."""
        return tuple(EmptyLattice.top() for _ in node.results)

    def eval_fallback(
        self, frame: ForwardFrame[EmptyLattice], node: ir.Statement
    ) -> tuple[EmptyLattice, ...]:
        """Check that a statement touches no place that is out.

        A pure statement only builds values, such as the reference `qs[i]`, so it
        touches no qubit.
        """
        if node.has_trait(ir.Pure):
            return self.results(node)
        for value in node.args:
            for ref in places_in(self.refs.get(value, Untracked())):
                if any(not disjoint(ref, place) for place, _ in self.out):
                    self.error(
                        node, f"'{node.name}' touches a qubit that is out as a wire"
                    )
                    return self.results(node)
        return self.results(node)


@stmts.dialect.register(key=KEY)
class _References(interp.MethodTable):
    """Track the places that `Unwrap`, `Wrap` and `Reference` take out and put back."""

    @interp.impl(stmts.Unwrap)
    @interp.impl(stmts.UnwrapRegister)
    def take(
        self,
        check: ReferenceCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: ir.Statement,
    ) -> tuple[EmptyLattice, ...]:
        """Add the place of the reference to the places that are out."""
        place = check.place(stmt, stmt.args[0])
        if place is None:
            return check.results(stmt)
        if any(not disjoint(place, other) for other, _ in check.out):
            check.error(stmt, f"'{stmt.name}' takes a qubit that is already out")
        check.out.append((place, stmt.results[0]))
        return check.results(stmt)

    @interp.impl(stmts.Wrap)
    @interp.impl(stmts.WrapRegister)
    def put(
        self,
        check: ReferenceCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: ir.Statement,
    ) -> tuple[EmptyLattice, ...]:
        """Remove the place of the reference from the places that are out."""
        place = check.place(stmt, stmt.args[0])
        if place is None:
            return check.results(stmt)
        for k, (other, _) in enumerate(check.out):
            if same_place(place, other):
                del check.out[k]
                return check.results(stmt)
        check.error(stmt, f"'{stmt.name}' puts back a qubit that is not out")
        return check.results(stmt)

    @interp.impl(stmts.Reference)
    @interp.impl(stmts.ReferenceRegister)
    def new(
        self,
        check: ReferenceCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: ir.Statement,
    ) -> tuple[EmptyLattice, ...]:
        """Start the new reference out, and check that a `Wrap` fills it next."""
        fill = stmt.next_stmt
        if not (isinstance(fill, PUT) and fill.args[0] is stmt.results[0]):
            check.error(
                stmt, f"'{stmt.name}' needs a wrap of its result right after it"
            )
        place = check.place(stmt, stmt.results[0])
        if place is not None:
            check.out.append((place, None))
        return check.results(stmt)


@scf.dialect.register(key=KEY)
class _Regions(interp.MethodTable):
    """Check that each body leaves the same places out as it found."""

    @interp.impl(scf.For)
    @interp.impl(scf.IfElse)
    def structured(
        self,
        check: ReferenceCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: ir.Statement,
    ) -> tuple[EmptyLattice, ...]:
        """Run each body once from the places that are out before `stmt`."""
        before = list(check.out)
        for region in stmt.regions:
            if not region.blocks:
                continue
            check.out = list(before)
            args = [EmptyLattice.top() for _ in region.blocks[0].args]
            with check.new_frame(stmt, has_parent_access=True) as inner:
                check.frame_call_region(inner, stmt, region, *args)
            balanced = len(check.out) == len(before) and all(
                any(same_place(a, b) for b, _ in before) for a, _ in check.out
            )
            if not balanced:
                check.error(
                    stmt, f"a body of '{stmt.name}' changes which qubits are out"
                )
        check.out = before
        return check.results(stmt)

    @interp.impl(scf.Yield)
    def yield_(
        self, check: ReferenceCheck, frame: ForwardFrame[EmptyLattice], stmt: scf.Yield
    ) -> interp.YieldValue[EmptyLattice]:
        """End a body."""
        return interp.YieldValue(tuple(EmptyLattice.top() for _ in stmt.values))


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """Check calls and returns of one function."""

    @interp.impl(func.Invoke)
    def invoke(
        self,
        check: ReferenceCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: func.Invoke,
    ) -> tuple[EmptyLattice, ...]:
        """Check the arguments of a call without entering the callee.

        The places of `refs` belong to one function, so the suite checks each
        callee on its own.
        """
        return check.eval_fallback(frame, stmt)

    @interp.impl(func.Return)
    def return_(
        self,
        check: ReferenceCheck,
        frame: ForwardFrame[EmptyLattice],
        stmt: func.Return,
    ) -> interp.ReturnValue[EmptyLattice]:
        """Record an error if a qubit is out as a wire that the return drops."""
        returned = {stmt.value}
        if isinstance(made := stmt.value.owner, py.tuple.New):
            returned.update(made.args)
        if any(wire not in returned for _, wire in check.out):
            check.error(stmt, "the function returns while qubits are out as wires")
        return interp.ReturnValue(EmptyLattice.top())


class ReferenceValidation(ValidationPass[ForwardFrame[EmptyLattice]]):
    """Check that wire squin takes each qubit out of its reference at most once."""

    def name(self) -> str:
        """Return the pass name that validation reports show."""
        return "squin wire references"

    def run(
        self, method: ir.Method
    ) -> tuple[ForwardFrame[EmptyLattice], list[ir.ValidationError]]:
        """Run `ReferenceCheck` and return its frame and errors."""
        analysis = ReferenceCheck(method.dialects)
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()
