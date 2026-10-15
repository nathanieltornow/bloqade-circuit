"""This module holds the forward analysis that states what each value refers to."""

import weakref
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from collections.abc import Callable

from kirin import ir, types
from kirin.dialects import py
from kirin.analysis.forward import Forward, ForwardFrame

from .index import Iter, Slice, Affine, python_slice
from .lattice import (
    CARRIED,
    UNTRACKED,
    Ref,
    Root,
    Step,
    Items,
    Place,
    Bottom,
    Closure,
    Members,
    Unknown,
    Returned,
    Positions,
    Untracked,
    child,
    place,
    roots,
)
from .integers import IndexAnalysis, loop_values

KEY = "reference"
"""Name the registry key of the rules for kirin's own dialects."""

OUT_OF_SCOPE = "an index that names a variable of a loop or a call that has ended"
"""Give the reason of an `Unknown` whose index uses a symbol outside its scope."""


_CONTENTS: "weakref.WeakKeyDictionary[ir.Statement, frozenset[int]]" = (
    weakref.WeakKeyDictionary()
)
"""Hold the identities of the statements and blocks inside each scope statement."""


def encloses(scope: ir.Statement, node: ir.IRNode) -> bool:
    """Return True if `node` is `scope` or a statement or block inside `scope`.

    The walk up the parent links answers most queries. The body of a lambda that a
    constant method holds can lack those links. So the fallback checks the
    statements and blocks that `scope` holds.
    """
    current: ir.IRNode | None = node
    while current is not None:
        if current is scope:
            return True
        current = current.parent_node
    if scope not in _CONTENTS:
        held: set[int] = set()
        for stmt in scope.walk():
            held.add(id(stmt))
            for region in stmt.regions:
                held.update(id(block) for block in region.blocks)
        _CONTENTS[scope] = frozenset(held)
    return id(node) in _CONTENTS[scope]


def owned_by(root: Root, code: ir.Statement) -> bool:
    """Return True if the function body `code` allocates or receives `root`."""
    return encloses(code, root.call if isinstance(root, Returned) else root.owner)


def defined_inside(symbol: object, scope: ir.Statement) -> bool:
    """Return True if the index symbol `symbol` only has a value inside `scope`.

    An `Iter` symbol of `scope` or of a statement inside it has a value only during
    one iteration. An SSA value that a statement or a block inside `scope` defines
    has a value only inside `scope`. A result of `scope` itself lies outside.
    """
    if isinstance(symbol, Iter):
        return encloses(scope, symbol.stmt)
    if isinstance(symbol, ir.ResultValue) and symbol.owner is scope:
        return False
    return isinstance(symbol, ir.SSAValue) and encloses(scope, symbol.owner)


def step_symbols(step: Step) -> frozenset[object]:
    """Return the index symbols that one step of a path uses."""
    if isinstance(step, Slice):
        length = frozenset() if step.length is None else step.length.symbols()
        return step.start.symbols() | length
    return step.symbols()


def _substitute(step: Step, mapping: dict) -> Step:
    """Return `step` with each symbol that `mapping` holds replaced by its index."""
    if isinstance(step, Slice):
        length = None if step.length is None else step.length.substitute(mapping)
        return Slice(step.start.substitute(mapping), step.step, length)
    return step.substitute(mapping)


RegisterLength = Callable[[ir.SSAValue, tuple[Returned, ...]], int | None]
"""Type a function that returns the static length of a register root, or None.

It takes the SSA value that allocates or receives the root, and the calls that
lead to it, innermost first.
"""


def origin_of(root: Root) -> tuple[ir.SSAValue, tuple[Returned, ...]]:
    """Return the SSA value that allocates or receives `root`, and the calls to it.

    A `Returned` root leads to the root inside the callee. The calls on that path
    come innermost first. A root of the analyzed function has no call.
    """
    calls: list[Returned] = []
    while isinstance(root, Returned):
        calls.append(root)
        root = root.inner
    return root, tuple(reversed(calls))


def argument_of(value: ir.SSAValue, calls: tuple[Returned, ...]) -> ir.SSAValue:
    """Return the value that the callers pass for `value`, through `calls`.

    `calls` come innermost first. While `value` is a parameter of the callee of
    the next call, the value becomes the argument of that call at the same
    position. So a size `n` that a callee receives becomes the caller's `4`.
    """
    for returned in calls:
        entry = returned.callee.callable_region.blocks[0]
        if not (isinstance(value, ir.BlockArgument) and value.owner is entry):
            break
        value = returned.call.args[value.index - 1]
    return value


def _typed_length(kind: types.TypeAttribute, levels: int) -> int | None:
    """Return the static length of the list `levels` levels inside the type `kind`.

    For example, the type `IList[IList[Qubit, 2], 3]` has the length 3 at level 0
    and the length 2 at level 1. The result is None if the type gives no literal
    length.
    """
    for _ in range(levels):
        if not (isinstance(kind, types.Generic) and kind.vars):
            return None
        kind = kind.vars[0]
    if not (isinstance(kind, types.Generic) and len(kind.vars) > 1):
        return None
    length = kind.vars[1]
    if isinstance(length, types.Literal) and isinstance(length.data, int):
        return length.data
    return None


def place_length(place: Place, register_length: RegisterLength) -> int | None:
    """Return the static length of the register place `place`, or None.

    A whole root gets its length from `register_length`. A slice has the length
    that the slice gives. A row of a register of registers, such as `qss[0]`, gets
    its length from the type of the root.
    """
    path = place.path
    if path and isinstance(last := path[-1], Slice):
        return None if last.length is None else last.length.constant
    origin, calls = origin_of(place.root)
    if not path:
        return register_length(origin, calls)
    return _typed_length(origin.type, len(path))


def items_of(ref: Ref, register_length: RegisterLength) -> tuple[Ref, ...] | None:
    """Return the tracked items that `ref` holds, or None if they are unknown.

    One item is itself. A literal list holds its members. A register place of
    static length holds the place of each index, and `register_length` gives the
    length of a whole root. A register of registers holds its rows.
    """
    match ref:
        case Place() if ref.one_item:
            return (ref,)
        case Members(members):
            return members
        case Place():
            size = place_length(ref, register_length)
            if size is not None:
                return tuple(child(ref, Affine(i)) for i in range(size))
    return None


def _checked(index: Affine, size: int | None) -> Affine | Unknown:
    """Return `index` as an index into a register of length `size`, or `Unknown`.

    A negative constant counts from the end, as in Python, so it needs the length.
    An index whose bounds show that it can be negative or past the end is
    `Unknown`. An index without bounds, such as a parameter `i`, keeps its form.
    The analysis then assumes that the program reads inside the register. A read
    past the end raises an error at runtime, so a program that runs never does it.
    """
    if (constant := index.constant) is not None:
        if size is None:
            if constant < 0:
                return Unknown("a negative index into a register of unknown length")
            return index
        if not -size <= constant < size:
            return Unknown("a constant index out of range")
        return Affine(constant % size)
    limits = index.bounds()
    if limits is not None and limits[0] < 0:
        return Unknown("an index that can be negative")
    if limits is not None and size is not None and limits[1] >= size:
        return Unknown("an index that can lie past the end of the register")
    return index


@dataclass
class ReferenceAnalysis(Forward[Ref], ABC):
    """Compute which root each value refers to, as a forward analysis."""

    keys = (KEY, "absint")
    lattice = Ref
    indices: IndexAnalysis = field(default_factory=IndexAnalysis, init=False)
    """Map each integer value that the analysis has read to its affine index."""

    @abstractmethod
    def shape(self, type_: types.TypeAttribute) -> int | None:
        """Return the depth of the tracked state that a value of type `type_` holds.

        The result is 0 for one item of tracked state, 1 for a register of items,
        2 for a register of registers, and None for a value without tracked state.
        """

    @abstractmethod
    def register_length(
        self, root: ir.SSAValue, calls: tuple[Returned, ...]
    ) -> int | None:
        """Return the static length of the register root `root`, or None.

        If `root` lies inside a callee, `calls` lead to it, innermost first. So the
        length can come from a constant argument of a call.
        """

    def affine(self, value: ir.SSAValue) -> Affine:
        """Return the affine index of the integer SSA value `value`."""
        return self.indices.of(value)

    def run(
        self, method: ir.Method, *args: Ref, **kwargs: Ref
    ) -> tuple[ForwardFrame[Ref], Ref]:
        """Analyze `method` and every function that it calls.

        Without `args`, each parameter of a tracked type is a root, a tuple
        parameter with a tracked member is `Unknown`, and every other parameter is
        `Untracked`. The returned frame holds the reference of each value of
        `method`, and the returned value is the result of `method`.
        """
        if not args and not kwargs:
            params: list[Ref] = []
            for arg in method.callable_region.blocks[0].args[1:]:
                if (depth := self.shape(arg.type)) is not None:
                    params.append(place(arg, (), depth))
                elif (
                    isinstance(arg.type, types.Generic)
                    and arg.type.is_subseteq(types.Tuple)
                    and any(self.shape(member) is not None for member in arg.type.vars)
                ):
                    params.append(Unknown("a member of a tuple parameter"))
                else:
                    params.append(UNTRACKED)
            args = tuple(params)
        return super().run(method, *args, **kwargs)

    def recursion_limit_reached(self) -> Ref:
        """Return `Unknown` for the deepest call of a recursion."""
        return Unknown("the result of a recursive call")

    def method_self(self, method: ir.Method) -> Ref:
        """Return `Untracked` for the method object."""
        return UNTRACKED

    def eval_fallback(
        self, frame: ForwardFrame[Ref], node: ir.Statement
    ) -> tuple[Ref, ...]:
        """Give each tracked result of a statement without a rule `Unknown`."""
        return self.unknown_results(node, f"a value computed by '{node.name}'")

    def unknown_results(self, stmt: ir.Statement, reason: str) -> tuple[Ref, ...]:
        """Return `Unknown` for each tracked result of `stmt` and `Untracked` else."""
        return tuple(
            Unknown(reason) if self.shape(r.type) is not None else UNTRACKED
            for r in stmt.results
        )

    def length(self, ref: Place) -> int | None:
        """Return the static length of the register place `ref`, or None."""
        return place_length(ref, self.register_length)

    def index(self, ref: Ref, index: Affine | int | ir.SSAValue) -> Ref:
        """Return the reference of the item at `index` of the list or register `ref`.

        A register place gives the place one step below it. If the register has a
        static length, a constant index becomes an index in the range 0 to the
        length minus 1. An index into a slice becomes an index into the register
        below the slice. A literal list gives its member at a constant index.
        """
        if isinstance(index, ir.SSAValue):
            index = self.affine(index)
        elif isinstance(index, int):
            index = Affine(index)
        match ref:
            case Place() if not ref.one_item:
                checked = _checked(index, self.length(ref))
                if isinstance(checked, Unknown):
                    return checked
                return child(ref, checked)
            case Members(members):
                if (constant := index.constant) is None:
                    return Unknown("a list or tuple read at a runtime index")
                if not -len(members) <= constant < len(members):
                    return Unknown("a constant index out of range")
                return members[constant]
            case Unknown() | Bottom():
                return ref
        return Unknown("an index into a value that is not a register or a list")

    def slice_bounds(
        self, value: ir.SSAValue
    ) -> tuple[Affine | None, Affine | None, int] | None:
        """Return the start, the stop and the step of the slice value `value`.

        A missing start or stop is None, and a missing step is 1. The result is None
        if `value` is no slice, or if its step is no constant.
        """
        owner = value.owner
        if isinstance(owner, py.Constant):
            data = owner.value.unwrap()
            if not isinstance(data, slice):
                return None
            bounds = (data.start, data.stop, data.step)
            if not all(b is None or type(b) is int for b in bounds):
                return None
            start, stop = (None if b is None else Affine(b) for b in bounds[:2])
            return start, stop, 1 if data.step is None else data.step
        if not isinstance(owner, py.slice.Slice):
            return None

        def bound(arg: ir.SSAValue) -> Affine | None:
            return None if arg.type.is_subseteq(types.NoneType) else self.affine(arg)

        if owner.step.type.is_subseteq(types.NoneType):
            step = 1
        elif (step := self.affine(owner.step).constant) is None:
            return None
        return bound(owner.start), bound(owner.stop), step

    def slice(self, ref: Ref, value: ir.SSAValue) -> Ref:
        """Return the reference of the slice `value` of the list or register `ref`.

        A register place gives a register place whose last step is a `Slice`. A
        slice of a slice reads the register below it. A literal list with constant bounds
        gives the list of the picked members.
        """
        bounds = self.slice_bounds(value)
        if bounds is None:
            return Unknown("a slice whose bounds the analysis cannot read")
        start, stop, step = bounds
        match ref:
            case Place() if not ref.one_item:
                made = python_slice(start, stop, step, self.length(ref))
                if made is None:
                    return Unknown("a slice that the analysis cannot name")
                return self.sliced(ref, made)
            case Members(members):
                ends = [None if b is None else b.constant for b in (start, stop)]
                if any(
                    b is not None and e is None for b, e in zip((start, stop), ends)
                ):
                    return Unknown("a list read at a runtime slice")
                return type(ref)(members[slice(*ends, step)])
            case Unknown() | Bottom():
                return ref
        return Unknown("a slice of a value that is not a register or a list")

    def sliced(self, ref: Place, made: Slice) -> Place:
        """Return the slice `made` of the register place `ref`.

        `made` counts its items in `ref`. If `ref` is itself a slice, the result
        counts them in the register below it.
        """
        path = ref.path
        if path and isinstance(last := path[-1], Slice):
            path = (*path[:-1], last.then(made))
        else:
            path = (*path, made)
        return place(ref.root, path, ref.depth)

    def descend(self, ref: Ref, path: tuple[Step, ...]) -> Ref:
        """Return the reference that `path` reaches from `ref`, one step at a time.

        Each index step goes through `index`, so a constant index gets its range
        check again. Each slice step counts its items in the register below.
        """
        for step in path:
            if isinstance(step, Slice):
                if not (isinstance(ref, Place) and not ref.one_item):
                    return Unknown("a slice of a value that is not a register")
                ref = self.sliced(ref, step)
            else:
                ref = self.index(ref, step)
        return ref

    def leave(self, ref: Ref, scope: ir.Statement) -> Ref:
        """Return `ref` as it reads outside the loop, branch or call `scope`.

        A place whose index uses a symbol that only has a value inside `scope`
        becomes `Unknown`. Without this rule, two iterations of a loop would give
        one place to two different items.
        """
        match ref:
            case Place(_, path):
                symbols = (s for step in path for s in step_symbols(step))
                if any(defined_inside(s, scope) for s in symbols):
                    return Unknown(OUT_OF_SCOPE)
            case Members(members):
                return type(ref)(tuple(self.leave(m, scope) for m in members))
            case Closure(code, captured):
                return Closure(code, tuple(self.leave(m, scope) for m in captured))
        return ref

    def call_result(self, frame: ForwardFrame[Ref], call: ir.Statement) -> Ref:
        """Return the reference of the result of `call` in the terms of the caller.

        The analysis runs the callee with the caller's references as arguments, like
        kirin's type inference. Kirin gives a call that reaches itself with the same
        references `Bottom` until the result settles, and a recursion that runs to
        kirin's `max_depth` returns `Unknown` from its deepest call.
        """
        callee = call.get_present_trait(ir.StaticCall).get_callee(call)
        args = frame.get_values(call.args)
        _, result = self.call(callee.code, self.method_self(callee), *args)
        params = callee.callable_region.blocks[0].args[1:]
        mapping = {p: self.affine(a) for p, a in zip(params, call.args)}
        return self.in_caller_terms(call, callee.code, mapping, args, result)

    def function_of(
        self, frame: ForwardFrame[Ref], fn: ir.SSAValue
    ) -> tuple[ir.Statement, Ref] | None:
        """Return the body of the function value `fn` and its self reference, or None.

        A constant method gives its code and `Untracked`. A closure gives its
        `func.Lambda` statement and the closure, which reads the captured values.
        """
        owner = fn.owner
        if isinstance(owner, py.Constant):
            method = owner.value.unwrap()
            if isinstance(method, ir.Method):
                return method.code, self.method_self(method)
        ref = frame.get(fn)
        if isinstance(ref, Closure):
            return ref.code, ref
        return None

    def apply(
        self,
        frame: ForwardFrame[Ref],
        code: ir.Statement,
        self_ref: Ref,
        args: tuple[Ref, ...],
    ) -> Ref:
        """Run the function body `code` on `args` and return its result.

        The result is in the terms of the callee, so `in_caller_terms` must rename
        it. A closure body belongs to the kernel under analysis, so its references
        join the ones in `frame`. A closure that runs twice joins both runs.
        """
        inner, result = self.call(code, self_ref, *args)
        if isinstance(self_ref, Closure):
            for key, value in inner.entries.items():
                old = frame.entries.get(key)
                frame.entries[key] = value if old is None else old.join(value)
        return result

    def map_result(
        self,
        frame: ForwardFrame[Ref],
        stmt: ir.Statement,
        fn: ir.SSAValue,
        collection: ir.SSAValue,
        depth: int | None,
    ) -> Ref:
        """Return the reference of the list that `stmt` maps `fn` over `collection`.

        `depth` is the shape of the result type. The analysis runs the body once,
        with the item `collection[k]` at the `Iter` symbol `k` of `stmt`. A body
        that returns its item gives `collection` itself. A body that returns the
        place `r[c + a*k]` with `a` other than 0 gives the slice of `r` that starts
        at `c` with the step `a`. A body that allocates gives a new root at `stmt`.
        A literal list runs the body once for each member.
        """
        function = self.function_of(frame, fn)
        if function is None:
            return Unknown("a map of a function that the analysis cannot name")
        code, self_ref = function
        source = frame.get(collection)
        if isinstance(source, Members):
            results = []
            for member in source.refs:
                result = self.apply(frame, code, self_ref, (member,))
                if any(owned_by(r, code) for r in roots(result)):
                    return self._fresh(stmt, depth)
                results.append(self.in_caller_terms(stmt, code, {}, (member,), result))
            return Items(tuple(results)) if depth is not None else UNTRACKED
        params = code.get_present_trait(ir.CallableStmtInterface)
        params = params.get_callable_region(code).blocks[0].args[1:]
        mapping: dict = {}
        if isinstance(source, Place) and not source.one_item:
            size = self.length(source)
            k = Iter(stmt, None if size is None else range(size))
            item = self.index(source, Affine.of(k))
        elif isinstance(source, Untracked):
            values = loop_values(collection)
            size = None if values is None else len(values)
            k = Iter(stmt, None if values is None else range(len(values)))
            if values is not None and params:
                mapping[params[0]] = Affine(values.start) + Affine.of(k).scale(
                    values.step
                )
            item = UNTRACKED
        else:
            return Unknown("a map over a value that is not a register or a list")
        result = self.apply(frame, code, self_ref, (item,))
        if any(owned_by(r, code) for r in roots(result)):
            return self._fresh(stmt, depth)
        named = self.in_caller_terms(stmt, code, mapping, (item,), result)
        if depth is None:
            return UNTRACKED
        if named == item and isinstance(source, Place):
            return source
        return self._gathered(named, k, size, depth)

    def _fresh(self, stmt: ir.Statement, depth: int | None) -> Ref:
        """Return a new register root at the result of the map `stmt`."""
        if depth is None or depth == 0:
            return Unknown("a map whose body allocates a value that is not a list item")
        return place(stmt.results[0], (), depth)

    def _gathered(self, named: Ref, k: Iter, size: int | None, depth: int) -> Ref:
        """Return the list of the items that the place `named` names at each `k`.

        `named` must use `k` in the last step of its path only, as `c + a*k` with
        `a` other than 0. The list then is the slice of `size` items that starts
        at `c` with the step `a`.
        """
        if not isinstance(named, Place) or named.depth + 1 != depth:
            if isinstance(named, Unknown):
                return named
            return Unknown("a map whose items the analysis cannot name")
        path = named.path
        uses = [n for n, step in enumerate(path) if k in step_symbols(step)]
        if not uses:
            return Unknown("a map whose items are all one place")
        last = path[-1]
        if uses != [len(path) - 1] or isinstance(last, Slice) or size is None:
            return Unknown("a map whose items do not form a slice")
        a = dict(last.terms)[k]
        start = last - Affine.of(k).scale(a)
        return place(named.root, (*path[:-1], Slice(start, a, Affine(size))), depth)

    def in_caller_terms(
        self,
        call: ir.Statement,
        code: ir.Statement,
        mapping: dict,
        args: tuple[Ref, ...],
        result: Ref,
    ) -> Ref:
        """Return `result` of the function body `code` in the terms of the caller.

        A root that the callee allocates and returns whole at the path `p` becomes
        `Returned(call, p)`. The first entry of `p` is the position of the output.
        Each further entry picks a member of a tuple or a list. A parameter of the
        callee is a root only when the callee is the method under analysis, in a
        recursion, and it becomes the reference of the argument at its position.
        The method maps every other root that the callee owns to `Unknown`.

        `mapping` maps each integer parameter of the callee to the affine index of
        the caller's argument. An index that still uses a symbol of the callee
        becomes `Unknown`.
        """
        params = code.get_present_trait(ir.CallableStmtInterface)
        params = params.get_callable_region(code).blocks[0].args[1:]
        passed: dict[ir.SSAValue, Ref] = dict(zip(params, args))
        returned: dict[Root, Root] = {}

        def collect(ref: Ref, at: tuple[int, ...]) -> None:
            match ref:
                case Place(root, ()) if owned_by(root, code):
                    returned.setdefault(root, Returned(call, at, root))
                case Members(members):
                    for k, member in enumerate(members):
                        collect(member, (*at, k))

        outputs = result.refs if isinstance(result, Positions) else (result,)
        for position, ref in enumerate(outputs):
            collect(ref, (position,))

        def rename(ref: Ref) -> Ref:
            match ref:
                case Place(root, path):
                    levels = ref.depth + sum(isinstance(s, Affine) for s in path)
                    base: Ref = place(root, (), levels)
                    if owned_by(root, code):
                        if isinstance(root, ir.SSAValue) and root in passed:
                            base = passed[root]
                        elif root in returned:
                            base = place(returned[root], (), levels)
                        else:
                            return Unknown("a root that the callee owns")
                    path = tuple(_substitute(step, mapping) for step in path)
                    symbols = (s for step in path for s in step_symbols(step))
                    if any(defined_inside(s, code) for s in symbols):
                        return Unknown("an item at an index that the callee computes")
                    return self.descend(base, path)
                case Members(members):
                    return type(ref)(tuple(rename(m) for m in members))
                case Closure(stmt, captured):
                    return Closure(stmt, tuple(rename(m) for m in captured))
            return ref

        return rename(result)

    def run_loop(
        self,
        frame: ForwardFrame[Ref],
        stmt: ir.Statement,
        body: ir.Region,
        carried: tuple[Ref, ...],
    ) -> tuple[Ref, ...]:
        """Run a loop body until the references that it carries stop changing.

        The body gets `Untracked` for the loop index and the carried references.
        The results leave the scope of the loop, so an index that uses the loop
        variable becomes `Unknown`.
        """
        # A join moves a carried reference up, and each one can move up once.
        for _ in range(len(carried) + 1):
            with self.new_frame(stmt, has_parent_access=True) as inner:
                yielded = self.frame_call_region(inner, stmt, body, UNTRACKED, *carried)
            if not isinstance(yielded, tuple) or len(yielded) != len(carried):
                return self.unknown_results(stmt, CARRIED)
            joined = tuple(c.join(y) for c, y in zip(carried, yielded, strict=True))
            if joined == carried:
                frame.entries.update(inner.entries)
                return tuple(self.leave(c, stmt) for c in carried)
            carried = joined
        raise AssertionError("a carried reference moved up the lattice twice")
