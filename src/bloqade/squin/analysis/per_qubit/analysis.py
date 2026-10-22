"""This module holds `PerQubit`, the base class of analyses that keep one value per qubit.

`PerQubitInterpreter` is a kirin interpreter with the key `per_qubit`. It runs a
squin kernel concretely and falls back to the `main` rules of kirin for classical
statements. A classical value is a real Python value. A qubit is a `Qubit` object,
and a register is an `IList` of them. So the `main` rules of `py.indexing` and
`py.slice` pick the qubits of a register.

The rules of `impls.py` call the hook `update` once for each group of qubits that
one operation acts on. A broadcast gate on 3 qubits gives 3 calls, and `cx` gives
one call with 2 qubits.

`OPAQUE` stands for a classical value that the interpreter cannot compute, such as
a measurement result or a classical parameter of the kernel. A pure statement with
an opaque operand gives opaque results. If the condition of an `if` is opaque,
the interpreter runs both branches on copies of the state, and `join` merges the
two states. If the condition is a known value, the interpreter runs only the taken
branch.

The interpreter raises `Stop` with a reason if it cannot run a statement that
touches qubits. A loop over an opaque range that touches qubits, a gate on qubits
that the interpreter cannot name, a statement on qubits without a rule and a chain
of calls deeper than `MAX_DEPTH` each stop it. `PerQubit.run` stores the reason in `failure`.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from dataclasses import field, dataclass
from collections.abc import Iterable, Sequence

from kirin import ir, types, interp
from kirin.dialects import func, ilist

from bloqade.qubit import stmts as qubit_stmts
from bloqade.types import QubitType
from bloqade.analysis.reference import Root, Affine, Unknown, place

S = TypeVar("S")

KEY = "per_qubit"
"""Give the registry key of the rules of the per-qubit interpreter."""

MAX_DEPTH = 256
"""Give the deepest stack of kirin call frames that the interpreter runs."""


class Stop(interp.InterpreterError):
    """Stop the per-qubit interpreter with a reason."""


class Opaque:
    """Stand for a classical value that the interpreter cannot compute.

    A measurement result, a classical parameter of the kernel and a value that
    depends on one are opaque. A rule that needs the truth value, the index or the
    items of an opaque value stops the interpreter.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return `OPAQUE`."""
        return "OPAQUE"

    def __bool__(self) -> bool:
        """Stop, because a branch on the value has no known side."""
        raise Stop("a branch on a value that the interpreter cannot compute")

    def __index__(self) -> int:
        """Stop, because the value gives no known index."""
        raise Stop("an index that the interpreter cannot compute")

    def __iter__(self):
        """Stop, because the value gives no known items."""
        raise Stop("a loop over values that the interpreter cannot compute")


OPAQUE = Opaque()
"""Give the one opaque value."""


@dataclass(frozen=True)
class Qubit:
    """Name one qubit of a run of a kernel.

    `root` is the value that allocates or receives the qubit, such as the result
    of `qalloc` or a parameter of the kernel. `run` counts the earlier runs of
    that allocation, so an allocation in a loop gives a new qubit in each
    iteration. `path` holds the integer indices from the root down to the qubit.
    """

    root: Root
    run: int
    path: tuple[int, ...]

    def __repr__(self) -> str:
        """Return the place of the qubit, such as `Slot(%qs, 2)` or `Whole(%t)#1`."""
        named = repr(place(self.root, tuple(Affine(i) for i in self.path), 0))
        return f"{named}#{self.run}" if self.run else named


@dataclass(frozen=True)
class Register:
    """Hold a register parameter of the kernel whose length the type does not give.

    An integer index gives the item at that index. Any other use stops the
    interpreter, because the register has no known items.
    """

    root: Root
    run: int
    path: tuple[int, ...]
    item: types.TypeAttribute

    def __repr__(self) -> str:
        """Return the place of the register, such as `Register(%qs)`."""
        named = repr(place(self.root, tuple(Affine(i) for i in self.path), 1))
        return f"{named}#{self.run}" if self.run else named

    def __getitem__(self, index: object) -> object:
        """Return the item at the index `index`, which must be 0 or more."""
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise Stop("a part of a register of a length that the type does not give")
        return argument(self.item, self.root, self.run, (*self.path, index))

    def __len__(self) -> int:
        """Stop, because the type gives no length."""
        raise Stop("the length of a register that the type does not give")

    def __iter__(self):
        """Stop, because the type gives no length."""
        raise Stop("the items of a register of a length that the type does not give")


def argument(
    kind: types.TypeAttribute, root: Root, run: int, path: tuple[int, ...]
) -> object:
    """Return the value of a parameter of the type `kind`, or the part at `path`.

    A qubit gives a `Qubit`. A list of qubits of a literal length gives an
    `IList`. A list of qubits of another length gives a `Register`. Any other type
    gives `OPAQUE`.
    """
    if kind.is_subseteq(QubitType) and kind is not types.Bottom:
        return Qubit(root, run, path)
    if not quantum(kind):
        return OPAQUE
    assert isinstance(kind, types.Generic)
    item = kind.vars[0]
    length = kind.vars[1] if len(kind.vars) > 1 else None
    if isinstance(length, types.Literal) and isinstance(length.data, int):
        items = [argument(item, root, run, (*path, j)) for j in range(length.data)]
        return ilist.IList(items, elem=item)
    return Register(root, run, path, item)


def quantum(kind: types.TypeAttribute) -> bool:
    """Return True if a value of the type `kind` is a qubit or a list of qubits."""
    if kind is types.Bottom:
        return False
    if kind.is_subseteq(QubitType):
        return True
    return (
        isinstance(kind, types.Generic)
        and kind.is_subseteq(ilist.IListType)
        and bool(kind.vars)
        and quantum(kind.vars[0])
    )


def touches(node: ir.Statement, seen: set[ir.Statement] | None = None) -> bool:
    """Return True if a statement inside `node` can act on a qubit.

    A statement with an operand or a result of a qubit type acts on a qubit. A
    call acts on a qubit if the body of the callee does. A call of a function
    value, a map and a for-each can each act on a qubit.
    """
    seen = set() if seen is None else seen
    for inner in node.walk():
        if inner is node:
            continue
        if any(quantum(value.type) for value in (*inner.args, *inner.results)):
            return True
        if isinstance(inner, (func.Call, ilist.Map, ilist.ForEach)):
            return True
        if isinstance(inner, func.Invoke) and inner.callee.code not in seen:
            seen.add(inner.callee.code)
            if touches(inner.callee.code, seen):
                return True
    return False


def holds_qubits(values: Iterable[object]) -> bool:
    """Return True if a value in `values` holds a qubit or a function on qubits."""
    for value in values:
        if isinstance(value, (Qubit, Register)):
            return True
        if isinstance(value, ir.Method) and touches(value.code):
            return True
        if isinstance(value, ilist.IList) and holds_qubits(value.data):
            return True
        if isinstance(value, (tuple, list)) and holds_qubits(value):
            return True
    return False


def same(a: object, b: object) -> object:
    """Return the value that two branches give: the value if they agree, else `OPAQUE`."""
    if a is b:
        return a
    if isinstance(a, ilist.IList) and isinstance(b, ilist.IList):
        if len(a.data) != len(b.data):
            return OPAQUE
        return ilist.IList([same(x, y) for x, y in zip(a.data, b.data)], elem=a.elem)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(same(x, y) for x, y in zip(a, b))
    if type(a) is type(b) and isinstance(a, (bool, int, float, complex, str, Qubit)):
        return a if a == b else OPAQUE
    return OPAQUE


@dataclass
class PerQubitInterpreter(interp.Interpreter):
    """Run a squin kernel concretely and call the hooks of `analysis`.

    The rules of the key `per_qubit` come first. The `main` rules of kirin run
    the classical statements.
    """

    keys = (KEY, "main")
    analysis: "PerQubit" = field(kw_only=True)
    qubit_state: dict[Qubit, Any] = field(default_factory=dict, init=False)
    """The value of each qubit that an operation touched."""
    runs: dict[Root, int] = field(default_factory=dict, init=False)
    """The count of earlier runs of each allocation."""
    own: dict[interp.Signature, Any] = field(init=False, repr=False, compare=False)
    main: dict[interp.Signature, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Select the rules of the key `per_qubit` and of the key `main`."""
        super().__post_init__()
        self.own = self.dialects.registry.interpreter(keys=(KEY,))
        self.main = self.dialects.registry.interpreter(keys=("main",))

    def initialize(self):
        """Clear the state and the run counts before a run."""
        super().initialize()
        self.qubit_state = {}
        self.runs = {}
        return self

    def frame_eval(
        self, frame: interp.Frame, node: ir.Statement
    ) -> interp.StatementResult[Any]:
        """Run `node`, and give opaque results for a pure statement on an opaque value.

        A statement with a `per_qubit` rule handles opaque operands itself. A pure
        statement with a region, such as a lambda, runs normally.
        """
        if (
            interp.Signature(type(node)) not in self.own
            and node.has_trait(ir.Pure)
            and not node.regions
        ):
            values = frame.get_values(node.args)
            if any(isinstance(value, Opaque) for value in values):
                if holds_qubits(values):
                    raise Stop(
                        f"a '{node.name}' statement on qubits with a value"
                        " that the interpreter cannot compute"
                    )
                return tuple(OPAQUE for _ in node.results)
        return super().frame_eval(frame, node)

    def eval_fallback(
        self, frame: interp.Frame, node: ir.Statement
    ) -> interp.StatementResult[Any]:
        """Give opaque results for a classical statement without a rule, or stop.

        A statement without a rule stops the interpreter if it has a region, an
        operand that holds a qubit, or a result of a qubit type.
        """
        if (
            node.regions
            or holds_qubits(frame.get_values(node.args))
            or any(quantum(result.type) for result in node.results)
        ):
            raise Stop(
                f"a '{node.name}' statement that the interpreter has no rule for"
            )
        return tuple(OPAQUE for _ in node.results)

    def recursion_limit_reached(self) -> Any:
        """Stop, because the chain of calls is deeper than `MAX_DEPTH`."""
        raise Stop("a chain of calls deeper than the interpreter follows")

    def run_main(
        self, frame: interp.Frame, node: ir.Statement
    ) -> interp.StatementResult[Any]:
        """Run `node` with its `main` rule."""
        return self.main[interp.Signature(type(node))](self, frame, node)

    def allocate(self, root: Root, size: object) -> object:
        """Return the fresh qubits of a new run of the allocation `root`.

        If `size` is None, the allocation gives one qubit. Else it gives a list of
        `size` qubits.
        """
        if isinstance(size, Opaque):
            raise Stop("a register of a size that the interpreter cannot compute")
        run = self.runs.get(root, -1) + 1
        self.runs[root] = run
        if size is None:
            return Qubit(root, run, ())
        assert isinstance(size, int)
        return ilist.IList([Qubit(root, run, (j,)) for j in range(size)], QubitType)

    def qubits(self, value: object) -> list[Qubit]:
        """Return the qubits of a qubit or of a list of qubits, in order, or stop."""
        if isinstance(value, Qubit):
            return [value]
        if isinstance(value, ilist.IList):
            return [qubit for item in value.data for qubit in self.qubits(item)]
        raise Stop(f"qubits that the interpreter cannot name: {value!r}")

    def apply(self, stmt: ir.Statement, operands: Sequence[object]) -> None:
        """Call `update` once for each group of qubits of the operation `stmt`.

        Group `k` holds the item `k` of each operand, in the order of the operands.
        """
        lists = [self.qubits(value) for value in operands]
        if len({len(qubits) for qubits in lists}) > 1:
            raise Stop("a gate on qubit lists of different lengths")
        for group in zip(*lists):
            self.analysis.update(self.qubit_state, tuple(group), stmt)

    def join_states(self, a: dict[Qubit, Any], b: dict[Qubit, Any]) -> dict:
        """Return the state after two branches that give the states `a` and `b`."""
        out = dict(a)
        for qubit, value in b.items():
            out[qubit] = self.analysis.join(a[qubit], value) if qubit in a else value
        return out


@dataclass
class PerQubit(ABC, Generic[S]):
    """Run a squin kernel and keep one value of type `S` for each qubit.

    A subclass implements `update`. After `run`, `failure` holds the reason if the
    interpreter stopped, and the returned state is then incomplete.
    """

    failure: str | None = field(default=None, init=False)

    @abstractmethod
    def update(
        self, state: dict[Qubit, S], qubits: tuple[Qubit, ...], stmt: ir.Statement
    ) -> None:
        """Change `state` for one operation of `stmt` on the group `qubits`.

        `stmt` is a gate, a measurement or a reset statement. `measures` and
        `resets` tell the kinds apart.
        """

    def join(self, a: S, b: S) -> S:
        """Return the value of a qubit after two branches that give `a` and `b`.

        The default is the larger value, which bounds both branches.
        """
        return max(a, b)  # type: ignore[type-var]

    def measures(self, stmt: ir.Statement) -> bool:
        """Return True if `stmt` measures its qubits."""
        return isinstance(stmt, qubit_stmts.Measure)

    def resets(self, stmt: ir.Statement) -> bool:
        """Return True if `stmt` resets its qubits."""
        return isinstance(stmt, qubit_stmts.Reset)

    def run(self, method: ir.Method) -> dict[Qubit, S]:
        """Run `method` and return the value of each qubit that an operation touched.

        Each qubit or register parameter of `method` is a root with the run 0. Each
        classical parameter is `OPAQUE`.
        """
        self.failure = None
        runner = PerQubitInterpreter(
            method.dialects, analysis=self, max_depth=MAX_DEPTH
        )
        params = method.callable_region.blocks[0].args[1:]
        args = [argument(param.type, param, 0, ()) for param in params]
        try:
            runner.run(method, *args)
        except interp.InterpreterError as error:
            self.failure = str(error)
        return runner.qubit_state


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
    operation. `Unknown` carries the reason why the interpreter stopped.
    """
    analysis = QubitDepth()
    state = analysis.run(method)
    if analysis.failure is not None:
        return Unknown(analysis.failure)
    return max(state.values(), default=0)
