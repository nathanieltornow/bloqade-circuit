"""Build wire squin from reference squin with the kirin emitter `ToWire`.

`ToWire` is a kirin emitter with one method table for each dialect. It walks a
kernel in program order and keeps the current wire of each qubit root, such as a
`qubit.new` result or a `qalloc` register. The places of `QubitReferenceAnalysis`
name the root and the index of every qubit operand. So the emitter needs no alias
reasoning of its own.

A gate on a list of qubits becomes one twin for each group of qubits. A qubit of
a register comes out with `Borrow` before the twin and goes back with `Release`
after it. The rewrite rules in `bloqade.squin.wire.rewrite` remove the pairs of
`Release` and `Borrow` that end up next to each other.

A loop carries the wires of the roots that `QubitEffects` records for it, as
extra initializers. A branch reads those wires from outside and yields the new
wires. Every root that a body allocates and keeps to itself gets a `Free` at the
end of the body.
"""

from typing import TypeVar
from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence

from kirin import ir, types, interp
from kirin.emit import EmitABC, EmitFrame
from kirin.dialects import py, scf, func, ilist, ssacfg
from kirin.emit.abc import EmitTable
from kirin.worklist import WorkList

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.squin.gate import _dialect as gate_dialect
from bloqade.squin.noise import _dialect as noise_dialect
from bloqade.analysis.reference import (
    Ref,
    Iter,
    Root,
    Place,
    Affine,
    Untracked,
    roots,
    disjoint,
)
from bloqade.squin.analysis.lifetime import Effect, holds_qubits
from bloqade.analysis.reference.index import Sym
from bloqade.squin.analysis.reference import qubit_items
from bloqade.analysis.reference.layout import flat

from . import stmts
from .group import kernel
from .twins import TWINS
from .types import RegisterType
from .traits import Twin

KEY = "emit.squin.wire"
"""Name the registry key of the method tables of `ToWire`."""

Value = ir.SSAValue | None
"""Type a wire squin value, or None for a value that holds qubits."""

_S = TypeVar("_S", bound=ir.Statement)


@dataclass
class Frame(EmitFrame[Value]):
    """Hold the block under construction and the current wire of each root."""

    body: ir.Block = field(default_factory=ir.Block, kw_only=True)
    """The wire squin block under construction."""
    wires: dict[Root, ir.SSAValue] = field(default_factory=dict, kw_only=True)
    """The current wire or register of each root that this block holds."""

    def push(self, stmt: _S) -> _S:
        """Append `stmt` to the block and return it."""
        self.body.stmts.append(stmt)
        return stmt

    def value(self, value: ir.SSAValue) -> Value:
        """Return the wire squin value of the reference squin value `value`.

        A body reads the classical values of its parent frames directly.
        """
        if value in self.entries:
            return self.entries[value]
        if not isinstance(self.parent, Frame):
            raise interp.InterpreterError(f"{value} has no wire squin value")
        return self.parent.value(value)

    def scalar(self, value: ir.SSAValue) -> ir.SSAValue:
        """Return the wire squin value of the classical value `value`."""
        made = self.value(value)
        if made is None:
            raise interp.InterpreterError(
                f"{value} holds qubits where the statement needs a classical value"
            )
        return made

    def wire(self, root: Root) -> ir.SSAValue:
        """Return the current wire of `root` in this block.

        A body never reads a wire of its parent block directly. The loop or branch
        rule hands each wire that the body uses into its frame first.
        """
        if root not in self.wires:
            raise interp.InterpreterError(f"the block holds no wire of {root!r}")
        return self.wires[root]


@dataclass
class ToWire(EmitABC[Frame, Value]):
    """Build the wire squin method of one reference squin kernel.

    `refs` holds the place of each value, and `effects` holds the effect of each
    statement. If `assume_disjoint` is True, the emitter trusts that the qubits of
    one gate group differ when the places cannot prove it.
    """

    keys = (KEY,)
    void = None
    dialects: ir.DialectGroup
    refs: Mapping[ir.SSAValue, Ref]
    effects: Mapping[ir.Statement, Effect]
    assume_disjoint: bool = False
    method: ir.Method | None = field(default=None, init=False)
    """The wire squin method that the last run built."""

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> Frame:
        """Create the frame of a body under conversion."""
        return Frame(node, has_parent_access=has_parent_access)

    def reset(self) -> None:
        """Clear the state of the previous run."""
        self.callables = EmitTable(prefix="")
        self.callable_to_emit = WorkList()
        self.method = None

    def eval_fallback(self, frame: Frame, node: ir.Statement) -> tuple[Value, ...]:
        """Copy a classical statement, and drop a statement that only builds qubit values.

        A statement whose results hold qubits, such as `ilist.new` of qubits or an
        index into a register, leaves no statement behind. The places already name
        every qubit that it builds. A classical statement gets a copy with its
        operands mapped.
        """
        if node.regions:
            raise interp.InterpreterError(f"'{node.name}' has a body without a rule")
        if any(holds_qubits(r.type) for r in node.results):
            return tuple(None for _ in node.results)
        args = [frame.scalar(arg) for arg in node.args]
        made = frame.push(type(node).from_stmt(node, args=args))
        return tuple(made.results)

    def qubits(self, value: ir.SSAValue) -> Sequence[Ref]:
        """Return the place of each qubit that `value` holds, in list order."""
        items = qubit_items(self.refs[value])
        if items is None:
            raise interp.InterpreterError(f"{value} holds qubits that no place names")
        return items

    def check_apart(self, group: Sequence[Ref]) -> None:
        """Raise if two places of one gate group may name the same qubit."""
        if self.assume_disjoint:
            return
        for k, a in enumerate(group):
            for b in group[k + 1 :]:
                if not disjoint(a, b):
                    raise interp.InterpreterError(
                        f"the places {a!r} and {b!r} of one gate may name one qubit"
                    )

    def take(self, frame: Frame, ref: Ref) -> ir.SSAValue:
        """Return the wire of the qubit at `ref`, and take it out of its root.

        A whole root hands over its own wire. A qubit of a register comes out with
        `Borrow`.
        """
        if not (isinstance(ref, Place) and ref.one_item):
            raise interp.InterpreterError(f"{ref!r} names no single qubit")
        if not ref.path:
            return frame.wire(ref.root)
        borrow = frame.push(
            stmts.Borrow(frame.wire(ref.root), self.index(frame, self.located(ref)))
        )
        frame.wires[ref.root] = borrow.register_out
        return borrow.wire

    def give(self, frame: Frame, ref: Ref, wire: ir.SSAValue) -> None:
        """Make `wire` the current wire of `ref`, and put it back into a register."""
        assert isinstance(ref, Place)
        if not ref.path:
            frame.wires[ref.root] = wire
            return
        release = frame.push(
            stmts.Release(
                frame.wire(ref.root), self.index(frame, self.located(ref)), wire
            )
        )
        frame.wires[ref.root] = release.register_out

    def located(self, ref: Place) -> Affine:
        """Return the index of the qubit `ref` in the flat register of its root."""
        at = flat(ref)
        if not isinstance(at, Affine):
            raise interp.InterpreterError(f"{ref!r} has no index in one register")
        return at

    def index(self, frame: Frame, index: Affine) -> ir.SSAValue:
        """Return a wire squin integer for the affine index `index`.

        The method builds the sum of each symbol times its coefficient, plus the
        constant. Common subexpression elimination later merges equal sums.
        """
        if (constant := index.constant) is not None:
            return frame.push(py.Constant(constant)).result
        total: ir.SSAValue | None = None
        for symbol, a in sorted(index.terms, key=lambda t: repr(t[0])):
            term = self.symbol(frame, symbol)
            if a != 1:
                factor = frame.push(py.Constant(a)).result
                term = frame.push(py.Mult(term, factor)).result
            total = term if total is None else frame.push(py.Add(total, term)).result
        assert total is not None
        if index.const:
            offset = frame.push(py.Constant(index.const)).result
            total = frame.push(py.Add(total, offset)).result
        return total

    def symbol(self, frame: Frame, symbol: Sym) -> ir.SSAValue:
        """Return the wire squin integer of the index symbol `symbol`.

        The `Iter` symbol of an `scf.for` loop is its loop variable. Any other
        symbol is an integer value of the kernel.
        """
        if isinstance(symbol, Iter):
            if not isinstance(symbol.stmt, scf.For):
                raise interp.InterpreterError(f"{symbol!r} has no wire squin value")
            return frame.scalar(symbol.stmt.body.blocks[0].args[0])
        return frame.scalar(symbol)

    def apply_twin(self, frame: Frame, stmt: ir.Statement) -> tuple[Value, ...]:
        """Emit the twin of the squin statement `stmt` once for each group of qubits.

        The classical operands and the attributes of `stmt` go to every twin. A
        list result of `stmt`, such as the results of a measurement, becomes an
        `ilist.new` of the scalar results of the twins.
        """
        twin = TWINS.get(type(stmt))
        if twin is None:
            raise interp.InterpreterError(f"'{stmt.name}' has no wire twin")
        trait = twin.get_present_trait(Twin)
        fields = getattr(twin, "__kirin_stmt_fields")
        wire_fields = dict(trait.wires)
        classical = {
            name: frame.scalar(getattr(stmt, name))
            for name in fields.std_args
            if name not in wire_fields.values()
        }
        attributes = {name: getattr(stmt, name) for name in fields.attributes}
        lists = [self.qubits(getattr(stmt, source)) for source, _ in trait.wires]
        scalars: dict[str, list[ir.SSAValue]] = {name: [] for name, _ in trait.results}
        for group in zip(*lists, strict=True):
            self.check_apart(group)
            wires = {
                wire: self.take(frame, ref)
                for (_, wire), ref in zip(trait.wires, group, strict=True)
            }
            made = frame.push(twin(**classical, **wires, **attributes))
            for (_, wire), ref in zip(trait.wires, group, strict=True):
                self.give(frame, ref, getattr(made, f"{wire}_out"))
            for source, result in trait.results:
                scalars[source].append(getattr(made, result))
        values: list[Value] = []
        for name in getattr(type(stmt), "__kirin_stmt_fields").results:
            kind = getattr(stmt, name).type
            element = kind.vars[0] if isinstance(kind, types.Generic) else None
            made_list = ilist.New(values=tuple(scalars[name]), elem_type=element)
            values.append(frame.push(made_list).result)
        return tuple(values)

    def carried(self, frame: Frame, stmt: ir.Statement) -> list[Root]:
        """Return the roots of `frame` whose wires the loop or branch `stmt` uses.

        The order follows the order in which `frame` got its wires, so two runs
        give the same order.
        """
        effect = self.effects.get(stmt)
        if effect is None or effect.opaque:
            raise interp.InterpreterError(
                f"'{stmt.name}' touches qubits that no place names"
            )
        return [root for root in frame.wires if root in effect.uses]

    def close(
        self, inner: Frame, kept: Sequence[Root], outs: Sequence[ir.SSAValue]
    ) -> ir.Region:
        """End the block of `inner` with a yield and return the block as a region.

        The yield holds `outs`, then the wire of each root in `kept`. Each other
        root of `inner` comes from an allocation in the body, so the body frees it.
        """
        self.free(inner, kept)
        inner.push(scf.Yield(*outs, *(inner.wires[root] for root in kept)))
        return ir.Region(inner.body)

    def free(self, frame: Frame, kept: Sequence[Root]) -> None:
        """Free the wire of each root of `frame` that `kept` does not name."""
        for root, wire in frame.wires.items():
            if root in kept:
                continue
            if wire.type.is_subseteq(RegisterType):
                frame.push(stmts.FreeRegister(wire))
            else:
                frame.push(stmts.Free(wire))

    def classical(self, values: Sequence[ir.SSAValue]) -> list[int]:
        """Return the positions of the values in `values` that hold no qubits."""
        return [
            k
            for k, v in enumerate(values)
            if isinstance(self.refs.get(v, Untracked()), Untracked)
        ]


@ssacfg.dialect.register(key=KEY)
class _SsaCfg(ssacfg.Concrete):
    """Run the statements of an SSA control-flow region with kirin's concrete rule."""


@func.dialect.register(key=KEY)
class _Func(interp.MethodTable):
    """Emit functions, returns and the calls of `qalloc`."""

    @interp.impl(func.Function)
    def function(
        self, emit: ToWire, frame: Frame, stmt: func.Function
    ) -> interp.StatementResult[Value]:
        """Build the wire squin function of `stmt` with the same signature.

        Each parameter must be classical. A kernel with qubit parameters needs the
        borrowing calls of the design, which this emitter does not build.
        """
        block = ir.Block()
        args: list[Value] = []
        for param in stmt.body.blocks[0].args:
            if param.index > 0 and not isinstance(
                emit.refs.get(param, Untracked()), Untracked
            ):
                raise interp.InterpreterError(
                    f"the qubit parameter '{param.name}' needs a borrowing call"
                )
            args.append(block.args.append_from(param.type, param.name))
        frame.body = block
        emit.frame_call(frame, stmt, *args)
        made = func.Function(
            sym_name=stmt.sym_name,
            signature=stmt.signature,
            slots=stmt.slots,
            body=ir.Region(block),
        )
        emit.method = ir.Method(
            dialects=kernel,
            code=made,
            sym_name=stmt.sym_name,
            arg_names=list(stmt.slots),
        )
        return ()

    @interp.impl(func.Invoke)
    def invoke(
        self, emit: ToWire, frame: Frame, stmt: func.Invoke
    ) -> interp.StatementResult[Value]:
        """Emit a call of `qalloc` as one register allocation.

        The pipeline inlines every other kernel before the conversion.
        """
        if stmt.callee is not qalloc:
            raise interp.InterpreterError(
                f"the call of '{stmt.callee.sym_name}' needs a borrowing call"
            )
        (root,) = roots(emit.refs[stmt.result])
        size = frame.scalar(stmt.inputs[0])
        frame.wires[root] = frame.push(stmts.AllocRegister(size)).result
        return (None,)

    @interp.impl(func.Return)
    def return_(
        self, emit: ToWire, frame: Frame, stmt: func.Return
    ) -> interp.ReturnValue[Value]:
        """Free every wire of the function and return its classical result."""
        value = frame.scalar(stmt.value)
        emit.free(frame, ())
        frame.push(func.Return(value))
        return interp.ReturnValue(None)


@qubit_stmts.dialect.register(key=KEY)
class _Qubit(interp.MethodTable):
    """Emit allocations, and the twins of measurements and resets."""

    @interp.impl(qubit_stmts.New)
    def new(
        self, emit: ToWire, frame: Frame, stmt: qubit_stmts.New
    ) -> interp.StatementResult[Value]:
        """Allocate the wire of the new qubit."""
        (root,) = roots(emit.refs[stmt.result])
        frame.wires[root] = frame.push(stmts.Alloc()).result
        return (None,)

    @interp.impl(qubit_stmts.Measure)
    @interp.impl(qubit_stmts.Reset)
    def twin(
        self, emit: ToWire, frame: Frame, stmt: ir.Statement
    ) -> interp.StatementResult[Value]:
        """Emit the twin of the statement for each qubit."""
        return emit.apply_twin(frame, stmt)


def _twin_table(dialect: ir.Dialect) -> None:
    """Register one method table that emits the twins of the statements of `dialect`."""

    def rule(
        self, emit: ToWire, frame: Frame, stmt: ir.Statement
    ) -> interp.StatementResult[Value]:
        return emit.apply_twin(frame, stmt)

    namespace = {
        f"twin_{source.__name__}": interp.impl(source)(rule)
        for source in TWINS
        if source.dialect is dialect
    }
    table = type(f"_Twins_{dialect.name}", (interp.MethodTable,), namespace)
    dialect.register(key=KEY)(table)


_twin_table(gate_dialect.dialect)
_twin_table(noise_dialect.dialect)


@scf.dialect.register(key=KEY)
class _Scf(interp.MethodTable):
    """Emit loops and branches that carry the wires of their bodies."""

    @interp.impl(scf.Yield)
    def yield_(
        self, emit: ToWire, frame: Frame, stmt: scf.Yield
    ) -> interp.YieldValue[Value]:
        """End a body with the values that it yields."""
        return interp.YieldValue(tuple(frame.value(v) for v in stmt.values))

    @interp.impl(scf.For)
    def for_(
        self, emit: ToWire, frame: Frame, stmt: scf.For
    ) -> interp.StatementResult[Value]:
        """Emit a loop that carries the wires of the roots that its body uses.

        The new loop takes the classical initializers first, then one wire for each
        carried root. An initializer that holds qubits disappears, because its
        place names the qubits.
        """
        carried = emit.carried(frame, stmt)
        classical = emit.classical(stmt.initializers)
        old = stmt.body.blocks[0].args
        block = ir.Block()
        index = block.args.append_from(old[0].type, old[0].name)
        inputs: list[Value] = [index] + [None] * len(stmt.initializers)
        for k in classical:
            inputs[k + 1] = block.args.append_from(old[k + 1].type, old[k + 1].name)
        with emit.new_frame(stmt, has_parent_access=True) as inner:
            inner.body = block
            for root in carried:
                inner.wires[root] = block.args.append_from(frame.wire(root).type)
            yielded = emit.frame_call_region(inner, stmt, stmt.body, *inputs)
            assert isinstance(yielded, tuple)
            body = emit.close(inner, carried, [yielded[k] for k in classical])
        loop = frame.push(
            scf.For(
                frame.scalar(stmt.iterable),
                body,
                *(frame.scalar(stmt.initializers[k]) for k in classical),
                *(frame.wire(root) for root in carried),
            )
        )
        results: list[Value] = [None] * len(stmt.results)
        for k, result in zip(classical, loop.results):
            results[k] = result
        for root, result in zip(carried, loop.results[len(classical) :], strict=True):
            frame.wires[root] = result
        return tuple(results)

    @interp.impl(scf.IfElse)
    def if_else(
        self, emit: ToWire, frame: Frame, stmt: scf.IfElse
    ) -> interp.StatementResult[Value]:
        """Emit a branch whose bodies read the carried wires and yield new ones.

        Kirin's branch has no carried block arguments. So each body uses each outer
        wire exactly once, and the branch result holds the new wire.
        """
        carried = emit.carried(frame, stmt)
        classical = emit.classical(stmt.results)
        bodies: list[ir.Region] = []
        for body in (stmt.then_body, stmt.else_body):
            block = ir.Block()
            cond = block.args.append_from(stmt.cond.type, "cond")
            with emit.new_frame(stmt, has_parent_access=True) as inner:
                inner.body = block
                inner.wires = {root: frame.wire(root) for root in carried}
                if body.blocks:
                    yielded = emit.frame_call_region(inner, stmt, body, cond)
                    assert isinstance(yielded, tuple)
                    outs = [yielded[k] for k in classical]
                else:
                    outs = []
                bodies.append(emit.close(inner, carried, outs))
        branch = frame.push(scf.IfElse(frame.scalar(stmt.cond), bodies[0], bodies[1]))
        results: list[Value] = [None] * len(stmt.results)
        for k, result in zip(classical, branch.results):
            results[k] = result
        for root, result in zip(carried, branch.results[len(classical) :], strict=True):
            frame.wires[root] = result
        return tuple(results)
