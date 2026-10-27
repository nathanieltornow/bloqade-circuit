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

A call borrows its qubit arguments, as in Guppy and in MQT's QCO dialect. The
wire callee takes a wire or a register for each qubit parameter. It returns its
ordinary results first, then the wire or register of each qubit parameter in
parameter order. A returned qubit that the callee allocates comes back as a wire.
"""

from typing import TypeVar
from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence

from kirin import ir, types, interp
from kirin.emit import EmitABC, EmitFrame
from kirin.dialects import py, scf, func, ilist, ssacfg
from kirin.emit.abc import EmitTable
from kirin.worklist import WorkList
from kirin.dialects.py import len as py_len

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
    positions,
)
from bloqade.squin.analysis.effects import Effect, holds_qubits
from bloqade.analysis.reference.index import Sym
from bloqade.squin.analysis.reference import qubit_items
from bloqade.analysis.reference.layout import flat

from . import stmts
from .group import kernel
from .twins import TWINS
from .types import WireType, RegisterType
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


def linear_type(ref: Ref) -> types.TypeAttribute | None:
    """Return the wire squin type of a value with the place `ref`, or None.

    One qubit becomes a `Wire`, and a whole register becomes a `Register`. A value
    without qubits gives None.
    """
    if isinstance(ref, Untracked):
        return None
    if isinstance(ref, Place) and ref.one_item:
        return WireType
    if isinstance(ref, Place) and not ref.path:
        return RegisterType
    raise interp.InterpreterError(f"a call boundary cannot pass {ref!r}")


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
    """Describe the outputs of the wire squin function of one kernel.

    `positions` counts the ordinary outputs of the kernel. `packed` is True if the
    kernel returns a tuple. `borrowed` holds the qubit parameters, whose wires
    follow the ordinary outputs in parameter order.
    """

    positions: int
    packed: bool
    borrowed: tuple[ir.BlockArgument, ...]

    @property
    def total(self) -> int:
        """Return the count of outputs of the wire squin function."""
        return self.positions + len(self.borrowed)


def pack(frame: Frame, values: Sequence[ir.SSAValue]) -> ir.SSAValue:
    """Return one value that holds `values`, as a function returns them.

    No value gives `None`, one value gives itself, and more values give a tuple.
    """
    if not values:
        return frame.push(func.ConstantNone()).result
    if len(values) == 1:
        return values[0]
    return frame.push(py.tuple.New(tuple(values))).result


def unpack(
    frame: Frame, value: ir.SSAValue, kinds: Sequence[types.TypeAttribute]
) -> list[ir.SSAValue]:
    """Return one value for each output of `value`, the inverse of `pack`."""
    if len(kinds) <= 1:
        return [value][: len(kinds)]
    items: list[ir.SSAValue] = []
    for k, kind in enumerate(kinds):
        at = frame.push(py.Constant(k)).result
        item = frame.push(py.GetItem(value, at))
        item.result.type = kind
        items.append(item.result)
    return items


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
    kernels: Mapping[ir.Method, ir.Method] = field(default_factory=dict)
    """Map each kernel that a call reaches to its prepared copy."""
    assume_disjoint: bool = False
    functions: dict[ir.Statement, ir.Method] = field(default_factory=dict, init=False)
    """Map the code of each prepared kernel to its wire squin method."""
    layouts: dict[ir.Statement, Layout] = field(default_factory=dict, init=False)
    """Map the code of each prepared kernel to the layout of its outputs."""

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> Frame:
        """Create the frame of a body under conversion."""
        return Frame(node, has_parent_access=has_parent_access)

    def reset(self) -> None:
        """Clear the state of the previous run."""
        self.callables = EmitTable(prefix="")
        self.callable_to_emit = WorkList()
        self.functions = {}
        self.layouts = {}

    def declare(self, code: func.Function) -> ir.Method:
        """Return the wire squin method of the kernel code `code`, made on first request.

        The method has the signature of `Layout` and an empty body.
        """
        if code in self.functions:
            return self.functions[code]
        entry = code.body.blocks[0]
        params = entry.args[1:]
        borrowed = tuple(p for p in params if linear_type(self.refs[p]) is not None)
        inputs = [linear_type(self.refs[p]) or p.type for p in params]
        originals = output_types(code.signature.output)
        returned = entry.last_stmt
        if not isinstance(returned, func.Return):
            raise interp.InterpreterError(f"'{code.sym_name}' does not end in a return")
        refs = positions(self.refs[returned.value], len(originals))
        outputs = [linear_type(r) or t for r, t in zip(refs, originals, strict=True)]
        outputs += [linear_type(self.refs[p]) or p.type for p in borrowed]
        output = packed_type(outputs)
        block = ir.Block()
        block.args.append_from(types.MethodType[inputs, output], entry.args[0].name)
        for param, kind in zip(params, inputs, strict=True):
            block.args.append_from(kind, param.name)
        made = func.Function(
            sym_name=code.sym_name,
            signature=func.Signature(inputs=tuple(inputs), output=output),
            slots=code.slots,
            body=ir.Region(block),
        )
        method = ir.Method(
            dialects=kernel,
            code=made,
            sym_name=code.sym_name,
            arg_names=list(code.slots),
        )
        packed = code.signature.output.is_subseteq(types.Tuple) and bool(originals)
        self.layouts[code] = Layout(len(originals), packed, borrowed)
        self.functions[code] = method
        return method

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

        A whole root hands over its own wire or register. A qubit of a register
        comes out with `Borrow`.
        """
        if not isinstance(ref, Place) or not (ref.one_item or not ref.path):
            raise interp.InterpreterError(f"{ref!r} names no qubit and no register")
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
                term = self.integer(frame, py.Mult(term, factor))
            total = term if total is None else self.integer(frame, py.Add(total, term))
        assert total is not None
        if index.const:
            offset = frame.push(py.Constant(index.const)).result
            total = self.integer(frame, py.Add(total, offset))
        return total

    @staticmethod
    def integer(frame: Frame, stmt: ir.Statement) -> ir.SSAValue:
        """Push the integer arithmetic `stmt` and return its result, typed `int`.

        A new `py.binop` statement has the result type `~T` until type inference
        runs, and kirin's `verify_type` rejects `~T` as an index.
        """
        frame.push(stmt)
        stmt.results[0].type = types.Int
        return stmt.results[0]

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
    """Emit functions, returns and borrowing calls."""

    @interp.impl(func.Function)
    def function(
        self, emit: ToWire, frame: Frame, stmt: func.Function
    ) -> interp.StatementResult[Value]:
        """Fill the body of the wire squin method of the kernel code `stmt`."""
        method = emit.declare(stmt)
        block = method.callable_region.blocks[0]
        frame.body = block
        args: list[Value] = [block.args[0]]
        for param, arg in zip(stmt.body.blocks[0].args[1:], block.args[1:]):
            ref = emit.refs[param]
            if isinstance(ref, Untracked):
                args.append(arg)
                continue
            (root,) = roots(ref)
            frame.wires[root] = arg
            args.append(None)
        emit.frame_call(frame, stmt, *args)
        return ()

    @interp.impl(func.Invoke)
    def invoke(
        self, emit: ToWire, frame: Frame, stmt: func.Invoke
    ) -> interp.StatementResult[Value]:
        """Emit a call of `qalloc` as an allocation, and any other call as a borrowing call.

        The call takes each qubit argument out of its root, and puts the returned
        wire back. Each returned qubit that the callee allocates becomes a new root
        of the caller.
        """
        if stmt.callee is qalloc:
            (root,) = roots(emit.refs[stmt.result])
            size = frame.scalar(stmt.inputs[0])
            frame.wires[root] = frame.push(stmts.AllocRegister(size)).result
            return (None,)
        prepared = emit.kernels.get(stmt.callee)
        if prepared is None:
            raise interp.InterpreterError(
                f"'{stmt.callee.sym_name}' has no prepared copy"
            )
        code = prepared.code
        assert isinstance(code, func.Function)
        if code not in emit.functions:
            emit.callable_to_emit.append(code)
        callee = emit.declare(code)
        layout = emit.layouts[code]
        passed = [emit.refs[arg] for arg in stmt.inputs]
        emit.check_apart([ref for ref in passed if not isinstance(ref, Untracked)])
        inputs = [
            frame.scalar(arg) if isinstance(ref, Untracked) else emit.take(frame, ref)
            for arg, ref in zip(stmt.inputs, passed, strict=True)
        ]
        call = frame.push(func.Invoke(tuple(inputs), callee=callee))
        call.result.type = callee.return_type
        outs = unpack(frame, call.result, output_types(callee.return_type))
        given = [ref for ref in passed if not isinstance(ref, Untracked)]
        for ref, wire in zip(given, outs[layout.positions :], strict=True):
            emit.give(frame, ref, wire)
        results: list[Value] = []
        refs = positions(emit.refs[stmt.result], layout.positions)
        for ref, value in zip(refs, outs[: layout.positions], strict=True):
            if isinstance(ref, Untracked):
                results.append(value)
                continue
            (root,) = roots(ref)
            frame.wires[root] = value
            results.append(None)
        if not layout.packed:
            return (results[0] if results else None,)
        members = [
            value if value is not None else frame.push(func.ConstantNone()).result
            for value in results
        ]
        return (frame.push(py.tuple.New(tuple(members))).result,)

    @interp.impl(func.Return)
    def return_(
        self, emit: ToWire, frame: Frame, stmt: func.Return
    ) -> interp.ReturnValue[Value]:
        """Return the ordinary outputs, then the wire of each borrowed parameter.

        The function frees every other wire.
        """
        code = stmt.parent_stmt
        assert isinstance(code, func.Function)
        layout = emit.layouts[code]
        if layout.positions == 0:
            values: list[ir.SSAValue] = []
        elif not layout.packed:
            values = [stmt.value]
        elif isinstance(made := stmt.value.owner, py.tuple.New):
            values = list(made.args)
        else:
            raise interp.InterpreterError(f"'{code.sym_name}' returns an opaque tuple")
        outs: list[ir.SSAValue] = []
        kept: list[Root] = []
        for value in values:
            ref = emit.refs[value]
            if isinstance(ref, Untracked):
                outs.append(frame.scalar(value))
                continue
            (root,) = roots(ref)
            outs.append(frame.wire(root))
            kept.append(root)
        for param in layout.borrowed:
            (root,) = roots(emit.refs[param])
            outs.append(frame.wire(root))
            kept.append(root)
        emit.free(frame, kept)
        frame.push(func.Return(pack(frame, outs)))
        return interp.ReturnValue(None)


@py_len.dialect.register(key=KEY)
class _Len(interp.MethodTable):
    """Emit the length of a list of qubits."""

    @interp.impl(py_len.Len)
    def len_(
        self, emit: ToWire, frame: Frame, stmt: py_len.Len
    ) -> interp.StatementResult[Value]:
        """Emit a constant for a list of known length, and `Length` for a register.

        A classical list keeps its `len` statement.
        """
        ref = emit.refs.get(stmt.value, Untracked())
        if isinstance(ref, Untracked):
            return emit.eval_fallback(frame, stmt)
        if (items := qubit_items(ref)) is not None:
            return (frame.push(py.Constant(len(items))).result,)
        if not (isinstance(ref, Place) and not ref.path):
            raise interp.InterpreterError(f"the length of {ref!r} has no wire form")
        length = frame.push(stmts.Length(frame.wire(ref.root)))
        frame.wires[ref.root] = length.register_out
        return (length.result,)


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
