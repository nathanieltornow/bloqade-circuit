"""This module holds `PerQubit`, the base class of analyses that keep one value per qubit.

The walk runs a squin kernel in program order. It resolves the operands of each gate,
measurement and reset to concrete qubits with the places of the reference analysis.
Then it calls the hook `update` once for each group of qubits that one operation
acts on. A broadcast gate on 3 qubits gives 3 calls, and `cx` gives one call with 2
qubits.

A `Qubit` names one qubit of a run: the root of the place, the count of earlier runs
of the allocation, and the integer path below the root. The walk runs each loop
over a constant range once per value, so a loop index becomes an integer. A call
binds the parameters of the callee to the qubits and integers of the caller.

The walk stops with a reason if it cannot name a qubit. A loop over a range that
is not constant, an index that the walk cannot compute and a map that touches
qubits each stop it. Both branches of an `if` run, and `join` merges their states.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir
from kirin.dialects import scf, func, ilist
from kirin.dialects.ilist import stmts as ilist_stmts

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.jeff.forms import GATES, GATE_KERNELS, QUBIT_KERNELS, is_library
from bloqade.analysis.reference import (
    Ref,
    Iter,
    Root,
    Place,
    Slice,
    Affine,
    Members,
    Unknown,
    Returned,
    Untracked,
    place,
    roots,
    origin_of,
    typed_length,
)
from bloqade.squin.analysis.reference import QubitReferenceAnalysis
from bloqade.analysis.reference.integers import loop_values

S = TypeVar("S")

MAX_DEPTH = 64
"""Give the deepest chain of calls that the walk follows before it stops."""


@dataclass(frozen=True)
class Qubit:
    """Name one qubit of a run of a kernel.

    `root` is the root of the qubit, as the kernel that allocates or receives it
    names it. `run` counts the earlier runs of that allocation, so an allocation
    in a loop gives a new qubit in each iteration. `path` holds the integer indices
    from the root down to the qubit.
    """

    root: Root
    run: int
    path: tuple[int, ...]

    def __repr__(self) -> str:
        """Return the place of the qubit, such as `Slot(%qs, 2)` or `Whole(%t)#1`."""
        named = repr(place(self.root, tuple(Affine(i) for i in self.path), 0))
        return f"{named}#{self.run}" if self.run else named


@dataclass(frozen=True)
class _Bound:
    """Hold a concrete register or qubit: a root run, a path and a slice at the end.

    `cut` holds the start and the step of a trailing slice. `length` holds the
    count of items of a register, if the walk knows it.
    """

    root: Root
    run: int
    path: tuple[int, ...] = ()
    cut: tuple[int, int] | None = None
    length: int | None = None

    def at(self, j: int) -> "_Bound":
        """Return the item `j` of the register."""
        if self.cut is not None:
            j = self.cut[0] + j * self.cut[1]
        return _Bound(self.root, self.run, (*self.path, j))

    def sliced(self, start: int, step: int, length: int | None) -> "_Bound":
        """Return the slice of `length` items from `start` with the step `step`."""
        if self.cut is not None:
            start, step = self.cut[0] + start * self.cut[1], self.cut[1] * step
        return _Bound(self.root, self.run, self.path, (start, step), length)

    def size(self) -> int | None:
        """Return the count of items of the register, or None."""
        if self.length is not None:
            return self.length
        if self.cut is not None:
            return None
        origin, _ = origin_of(self.root)
        return typed_length(origin.type, len(self.path))

    def qubit(self) -> Qubit:
        """Return the qubit that the binding names."""
        assert self.cut is None
        return Qubit(self.root, self.run, self.path)


class _Stop(Exception):
    """Stop the walk with a reason."""


@dataclass
class _Activation:
    """Hold what the walk knows inside one run of one kernel."""

    kernel: ir.Method
    refs: Mapping[ir.SSAValue, Ref]
    analysis: QubitReferenceAnalysis
    depth: int
    ints: dict[ir.SSAValue, int] = field(default_factory=dict)
    """The value of each integer that the walk knows, such as a loop variable."""
    bound: dict[Root, object] = field(default_factory=dict)
    """The binding of each root that the walk has met in this run."""
    returned: dict[ir.Statement, list[Returned]] = field(default_factory=dict)
    """The `Returned` roots that each call statement hands to this kernel."""


@dataclass
class PerQubit(ABC, Generic[S]):
    """Walk a squin kernel and keep one value of type `S` for each qubit.

    A subclass implements `update`. After `run`, `failure` holds the reason if the
    walk stopped, and the returned state is then incomplete.
    """

    failure: str | None = field(default=None, init=False)
    _runs: dict[Root, int] = field(default_factory=dict, init=False)
    _frames: dict[ir.Method, tuple] = field(default_factory=dict, init=False)

    @abstractmethod
    def update(
        self, state: dict[Qubit, S], qubits: tuple[Qubit, ...], stmt: ir.Statement
    ) -> None:
        """Change `state` for one operation of `stmt` on the group `qubits`.

        `stmt` is a gate, a measurement or a reset, as a statement or as a call of
        a squin library kernel. `measures` and `resets` tell the kinds apart.
        """

    def join(self, a: S, b: S) -> S:
        """Return the value of a qubit after two branches that give `a` and `b`.

        The default is the larger value, which bounds both branches.
        """
        return max(a, b)  # type: ignore[type-var]

    def measures(self, stmt: ir.Statement) -> bool:
        """Return True if `stmt` measures its qubits."""
        return self._kind(stmt) is qubit_stmts.Measure

    def resets(self, stmt: ir.Statement) -> bool:
        """Return True if `stmt` resets its qubits."""
        return self._kind(stmt) is qubit_stmts.Reset

    def run(self, method: ir.Method) -> dict[Qubit, S]:
        """Walk `method` and return the value of each qubit that an operation touched.

        Each qubit or register parameter of `method` is a root with the run 0.
        """
        self.failure = None
        self._runs = {}
        state: dict[Qubit, S] = {}
        act = self._activation(method, 0)
        for param in method.callable_region.blocks[0].args[1:]:
            if isinstance(ref := act.refs.get(param), Place):
                act.bound[ref.root] = _Bound(ref.root, 0)
        try:
            self._block(act, method.callable_region.blocks[0], state)
        except _Stop as stop:
            self.failure = str(stop)
        return state

    # -- the walk ----------------------------------------------------------------

    def _activation(self, kernel: ir.Method, depth: int) -> _Activation:
        """Return a new run of `kernel` with the references of its own analysis."""
        if depth > MAX_DEPTH:
            raise _Stop("a chain of calls deeper than the walk follows")
        if kernel not in self._frames:
            analysis = QubitReferenceAnalysis(kernel.dialects)
            frame, _ = analysis.run(kernel)
            returned: dict[ir.Statement, list[Returned]] = {}
            for ref in frame.entries.values():
                for root in roots(ref):
                    if isinstance(root, Returned) and root not in returned.get(
                        root.call, []
                    ):
                        returned.setdefault(root.call, []).append(root)
            self._frames[kernel] = (frame.entries, analysis, returned)
        refs, analysis, returned = self._frames[kernel]
        return _Activation(kernel, refs, analysis, depth, returned=returned)

    def _block(self, act: _Activation, block: ir.Block, state: dict) -> None:
        """Walk the statements of `block` in order."""
        for stmt in block.stmts:
            self._stmt(act, stmt, state)

    def _stmt(self, act: _Activation, stmt: ir.Statement, state: dict) -> None:
        """Walk one statement."""
        match stmt:
            case scf.For():
                self._loop(act, stmt, state)
            case scf.IfElse():
                self._branches(act, stmt, state)
            case func.Invoke() if is_library(stmt.callee):
                self._library(act, stmt, state)
            case func.Invoke():
                self._call(act, stmt, state)
            case qubit_stmts.New():
                self._allocate(act, stmt.result, None)
            case qubit_stmts.Measure() | qubit_stmts.Reset():
                for qubit in self._qubits(act, stmt.args[0]):
                    self.update(state, (qubit,), stmt)
            case _ if type(stmt) in GATES:
                form = GATES[type(stmt)]
                self._gate(act, stmt, stmt.args[form.angles :], state)
            case _ if stmt.regions and _touches(stmt):
                raise _Stop(f"a '{stmt.name}' statement whose body touches qubits")

    def _loop(self, act: _Activation, stmt: scf.For, state: dict) -> None:
        """Walk the body once for each value of a loop over a constant range."""
        values = loop_values(stmt.iterable)
        if values is None and isinstance(stmt.iterable.owner, ilist_stmts.Range):
            bounds = [self._int(act, arg) for arg in stmt.iterable.owner.args]
            if all(b is not None for b in bounds):
                values = range(*bounds)  # type: ignore[arg-type]
        if values is None:
            if _touches(stmt):
                raise _Stop("a loop over a range that is not constant")
            return
        variable = stmt.body.blocks[0].args[0]
        for value in values:
            act.ints[variable] = value
            self._block(act, stmt.body.blocks[0], state)

    def _branches(self, act: _Activation, stmt: scf.IfElse, state: dict) -> None:
        """Walk both branches on copies of `state` and join the results."""
        then_state, else_state = dict(state), dict(state)
        self._block(act, stmt.then_body.blocks[0], then_state)
        self._block(act, stmt.else_body.blocks[0], else_state)
        for qubit in then_state.keys() | else_state.keys():
            if qubit in then_state and qubit in else_state:
                state[qubit] = self.join(then_state[qubit], else_state[qubit])
            else:
                state[qubit] = then_state.get(qubit, else_state.get(qubit))

    def _library(self, act: _Activation, stmt: func.Invoke, state: dict) -> None:
        """Walk a call of a squin library kernel: a gate, a qubit kernel or `qalloc`."""
        if (form := GATE_KERNELS.get(stmt.callee)) is not None:
            self._gate(act, stmt, stmt.inputs[form.angles :], state)
            return
        if stmt.callee is qalloc:
            self._allocate(act, stmt, self._int(act, stmt.inputs[0]))
            return
        match QUBIT_KERNELS.get(stmt.callee):
            case qubit_stmts.New:
                self._allocate(act, stmt, None)
            case qubit_stmts.Measure | qubit_stmts.Reset:
                for qubit in self._qubits(act, stmt.inputs[0]):
                    self.update(state, (qubit,), stmt)

    def _gate(
        self,
        act: _Activation,
        stmt: ir.Statement,
        operands: tuple[ir.SSAValue, ...],
        state: dict,
    ) -> None:
        """Call `update` once for each group of qubits of a gate."""
        lists = [self._qubits(act, value) for value in operands]
        if len({len(qubits) for qubits in lists}) > 1:
            raise _Stop("a gate on qubit lists of different lengths")
        for group in zip(*lists):
            self.update(state, tuple(group), stmt)

    def _allocate(
        self, act: _Activation, source: ir.Statement | ir.SSAValue, size: int | None
    ) -> None:
        """Start a new run of each root that the allocation `source` gives."""
        if isinstance(source, ir.SSAValue):
            named: list[Root] = [source]
        else:
            named = list(act.returned.get(source, []))
        for root in named:
            run = self._runs.get(root, -1) + 1
            self._runs[root] = run
            act.bound[root] = _Bound(root, run, length=size)

    def _call(self, act: _Activation, stmt: func.Invoke, state: dict) -> None:
        """Walk the callee with its parameters bound to the caller's values."""
        inner = self._activation(stmt.callee, act.depth + 1)
        params = stmt.callee.callable_region.blocks[0].args[1:]
        for param, arg in zip(params, stmt.inputs):
            ref = act.refs.get(arg)
            if isinstance(ref, Untracked):
                if (value := self._int(act, arg)) is not None:
                    inner.ints[param] = value
            elif (binding := self._bind(act, ref)) is not None:
                inner_ref = inner.refs.get(param)
                if isinstance(inner_ref, Place):
                    inner.bound[inner_ref.root] = binding
        self._block(inner, stmt.callee.callable_region.blocks[0], state)
        for root in act.returned.get(stmt, []):
            if (binding := inner.bound.get(root.inner)) is not None:
                act.bound[root] = binding

    # -- resolving places --------------------------------------------------------

    def _int(self, act: _Activation, value: ir.SSAValue) -> int | None:
        """Return the integer that `value` holds in this run, or None."""
        return self._evaluate(act, act.analysis.affine(value))

    def _evaluate(self, act: _Activation, index: Affine) -> int | None:
        """Return the value of the affine index `index` in this run, or None."""
        total = index.const
        for symbol, a in index.terms:
            if isinstance(symbol, Iter):
                if not isinstance(symbol.stmt, scf.For):
                    return None
                symbol = symbol.stmt.body.blocks[0].args[0]
            if (value := act.ints.get(symbol)) is None:
                return None
            total += a * value
        return total

    def _bind(self, act: _Activation, ref: Ref | None) -> object | None:
        """Return the binding of the reference `ref` in this run, or None."""
        match ref:
            case Place():
                return self._resolve(act, ref)
            case Members(members):
                bound = tuple(self._bind(act, m) for m in members)
                return None if any(b is None for b in bound) else bound
        return None

    def _resolve(self, act: _Activation, ref: Place) -> object:
        """Return the binding of the place `ref` in this run, or stop."""
        binding = act.bound.get(ref.root)
        if binding is None:
            raise _Stop(f"a place whose root the walk cannot name: {ref!r}")
        for step in ref.path:
            if isinstance(step, Slice):
                start = self._evaluate(act, step.start)
                length = (
                    None if step.length is None else self._evaluate(act, step.length)
                )
                if start is None or (step.length is not None and length is None):
                    raise _Stop(f"a slice that the walk cannot compute: {ref!r}")
                binding = _slice(binding, start, step.step, length)
            else:
                if (j := self._evaluate(act, step)) is None:
                    raise _Stop(f"an index that the walk cannot compute: {ref!r}")
                binding = _at(binding, j)
        return binding

    def _qubits(self, act: _Activation, value: ir.SSAValue) -> list[Qubit]:
        """Return the qubits that `value` holds in this run, or stop."""
        ref = act.refs.get(value)
        qubits = self._expand(act, ref)
        if qubits is None:
            raise _Stop(f"qubits that the walk cannot name: {ref!r}")
        return qubits

    def _expand(self, act: _Activation, ref: Ref | None) -> list[Qubit] | None:
        """Return the qubits of `ref` in order, or None."""
        match ref:
            case Place() if ref.one_item:
                binding = self._resolve(act, ref)
                return [binding.qubit()] if isinstance(binding, _Bound) else None
            case Place():
                return _items(self._resolve(act, ref), ref.depth)
            case Members(members):
                out: list[Qubit] = []
                for member in members:
                    part = self._expand(act, member)
                    if part is None:
                        return None
                    out += part
                return out
        return None

    def _kind(self, stmt: ir.Statement) -> type | None:
        """Return the qubit statement type that `stmt` is or wraps, or None."""
        if isinstance(stmt, func.Invoke):
            return QUBIT_KERNELS.get(stmt.callee)
        return type(stmt)


def _at(binding: object, j: int) -> object:
    """Return the item `j` of a register binding or a literal list."""
    if isinstance(binding, tuple):
        return binding[j]
    assert isinstance(binding, _Bound)
    return binding.at(j)


def _slice(binding: object, start: int, step: int, length: int | None) -> object:
    """Return a slice of a register binding or a literal list."""
    if isinstance(binding, tuple):
        if length is None:
            return binding[start::step]
        return tuple(binding[start + k * step] for k in range(length))
    assert isinstance(binding, _Bound)
    return binding.sliced(start, step, length)


def _items(binding: object, depth: int) -> list[Qubit] | None:
    """Return the qubits of a binding of the depth `depth`, or None."""
    if depth == 0:
        return [binding.qubit()] if isinstance(binding, _Bound) else None
    if isinstance(binding, tuple):
        size: int | None = len(binding)
    else:
        assert isinstance(binding, _Bound)
        size = binding.size()
    if size is None:
        return None
    out: list[Qubit] = []
    for j in range(size):
        part = _items(_at(binding, j), depth - 1)
        if part is None:
            return None
        out += part
    return out


def _touches(stmt: ir.Statement) -> bool:
    """Return True if a statement inside `stmt` can act on a qubit."""
    for node in stmt.walk():
        if node is stmt:
            continue
        if isinstance(node, func.Invoke):
            return True
        if type(node) in GATES or isinstance(
            node, (qubit_stmts.New, qubit_stmts.Measure, qubit_stmts.Reset)
        ):
            return True
        if isinstance(node, (ilist.Map, ilist.ForEach, func.Call)):
            return True
    return False


@dataclass
class MeasurementCount(PerQubit[int]):
    """Count the measurements of each qubit.

    After branches, the count is the larger count of the two branches.
    """

    def update(
        self, state: dict[Qubit, int], qubits: tuple[Qubit, ...], stmt: ir.Statement
    ) -> None:
        """Add 1 to each measured qubit."""
        if self.measures(stmt):
            for qubit in qubits:
                state[qubit] = state.get(qubit, 0) + 1


@dataclass
class QubitDepth(PerQubit[int]):
    """Compute the depth of each qubit: the longest chain of operations on it.

    A gate, a measurement and a reset each add 1. An operation on several qubits
    starts after the deepest of them, so operations on disjoint qubits share one
    layer. This matches `unit_cost` of the jeff depth analysis.
    """

    def update(
        self, state: dict[Qubit, int], qubits: tuple[Qubit, ...], stmt: ir.Statement
    ) -> None:
        """Set each qubit of the group to 1 more than the deepest of them."""
        depth = max((state.get(qubit, 0) for qubit in qubits), default=0) + 1
        for qubit in qubits:
            state[qubit] = depth


def circuit_depth(method: ir.Method) -> int | Unknown:
    """Return the circuit depth of the squin kernel `method`, or `Unknown`.

    The circuit depth is the largest depth over all qubits, and 0 without any
    operation. `Unknown` carries the reason why the walk stopped.
    """
    analysis = QubitDepth()
    state = analysis.run(method)
    if analysis.failure is not None:
        return Unknown(analysis.failure)
    return max(state.values(), default=0)
