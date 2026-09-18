"""This module holds the forward analysis that states what each value refers to."""

from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from collections.abc import Sequence

from kirin import ir, types
from kirin.dialects import func
from kirin.interp.frame import FrameABC
from kirin.analysis.forward import Forward, ForwardFrame

from bloqade.jeff.constants import const_int
from bloqade.jeff.signature import declared_outputs

from .lattice import (
    CARRIED,
    UNTRACKED,
    Ref,
    Call,
    Root,
    Slot,
    Items,
    Whole,
    Members,
    Unknown,
    Returned,
    Positions,
    root_in,
)

KEY = "reference"
"""The registry key of the rules for kirin's own dialects."""


def as_positions(ref: Ref, count: int) -> tuple[Ref, ...]:
    """Split the reference of a function result into `count` output positions."""
    if count == 0:
        return ()
    if isinstance(ref, Unknown):
        return tuple(ref for _ in range(count))
    if isinstance(ref, Positions) and len(ref.refs) == count:
        return ref.refs
    if count == 1 and not isinstance(ref, Positions):
        return (ref,)
    return tuple(
        Unknown("a return that does not match the declared output positions")
        for _ in range(count)
    )


@dataclass
class ReferenceAnalysis(Forward[Ref], ABC):
    """A forward analysis that states which root each value refers to.

    The analysis runs each function once, with its own parameters as roots. A call
    translates the result of the callee into the terms of the caller. For example,
    `flip(q)` returns `Whole(%q)`, so the call `flip(a)` returns `Whole(%a)`. A
    qubit that `fresh()` allocates and returns becomes `Returned(call, 0)` at each
    call. The frame that `run` returns holds the reference of every value in every
    function that the analysis ran, each in the terms of its own function.

    A subclass states which types hold tracked state, which calls allocate a
    register, and how long a register is. The rules for kirin dialects live under
    the key `KEY`.
    """

    keys = (KEY,)
    lattice = Ref

    results: dict[ir.Method, Ref] = field(default_factory=dict, init=False)
    """The result of each function that the analysis ran, each callee first."""

    @abstractmethod
    def is_tracked_type(self, type_: types.TypeAttribute) -> bool:
        """Return True if a value of type `type_` refers to tracked state."""

    def is_allocating_call(self, call: func.Invoke) -> bool:
        """Return True if `call` allocates its result, so the analysis skips the callee.

        The default returns False.
        """
        return False

    @abstractmethod
    def register_length(self, value: ir.SSAValue) -> int | None:
        """Return the static length of the register root `value`, or None."""

    def initialize(self) -> "ReferenceAnalysis":
        """Reset the interpreter state and the results."""
        super().initialize()
        self.results = {}
        return self

    def run(
        self, method: ir.Method, *args: Ref, **kwargs: Ref
    ) -> tuple[ForwardFrame[Ref], Ref]:
        """Analyze `method` and every function that it calls.

        The returned frame holds the reference of every value in these functions.
        The returned value is the result of `method`.
        """
        with self.eval_context():
            frame, result = self.call(
                method.code, self.method_self(method), *self.parameters(method)
            )
        self.results[method] = result
        return frame, result

    def parameters(self, method: ir.Method) -> list[Ref]:
        """Return the references of the parameters of `method` in its own terms.

        A parameter of a tracked type is a root. A tuple parameter with a tracked
        member is `Unknown`, and every other parameter is `Untracked`.
        """
        params: list[Ref] = []
        for arg in method.callable_region.blocks[0].args[1:]:
            if self.is_tracked_type(arg.type):
                params.append(Whole(arg))
            elif (
                isinstance(arg.type, types.Generic)
                and arg.type.is_subseteq(types.Tuple)
                and any(self.is_tracked_type(member) for member in arg.type.vars)
            ):
                params.append(Unknown("a member of a tuple parameter"))
            else:
                params.append(UNTRACKED)
        return params

    def result_of(self, method: ir.Method, frame: ForwardFrame[Ref]) -> Ref:
        """Return the result of `method` in terms of its own parameters.

        `frame` is the frame of the caller. The analysis runs `method` on the first
        request only, and then adds the references of its body to `frame`. If a
        frame of `method` already runs below `frame`, the call is recursive, and its
        result is `Unknown`.
        """
        if method not in self.results:
            running: FrameABC | None = frame
            while running is not None:
                if running.code is method.code:
                    return Unknown("the result of a recursive call")
                running = running.parent
            body, result = self.call(
                method.code, self.method_self(method), *self.parameters(method)
            )
            frame.entries.update(body.entries)
            self.results[method] = result
        return self.results[method]

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
            Unknown(reason) if self.is_tracked_type(r.type) else UNTRACKED
            for r in stmt.results
        )

    def origin(self, root: Root) -> ir.SSAValue:
        """Return the SSA value that allocates or receives `root`.

        A `Returned` root leads to the root inside the callee.
        """
        while isinstance(root, Returned):
            callee = root.call.callee
            count = len(declared_outputs(callee.return_type))
            inside = as_positions(self.results[callee], count)[root.position]
            # `translate` roots a `Returned` only at a whole root of the callee.
            assert isinstance(inside, Whole)
            root = inside.root
        return root

    def item(self, members: tuple[Ref, ...], index: int | None) -> Ref:
        """Return the reference of the member at a constant `index`, or `Unknown`."""
        if index is None:
            return Unknown("a list or tuple read at a runtime index")
        if not -len(members) <= index < len(members):
            return Unknown("a constant index out of range")
        return members[index]

    def slot(self, root: Root, index: int | ir.SSAValue) -> Ref:
        """Return the reference to the item at `index` of the register `root`.

        If the register has a static length, a constant index becomes an index in
        the range 0 to the length minus 1.
        """
        constant = index if isinstance(index, int) else const_int(index)
        if constant is None:
            return Slot(root, index)
        size = self.register_length(self.origin(root))
        if size is None:
            return Slot(root, constant)
        if not -size <= constant < size:
            return Unknown("a constant index out of range")
        return Slot(root, constant % size)

    def translate(
        self, call: Call, args: tuple[Ref, ...], outputs: tuple[Ref, ...]
    ) -> tuple[Ref, ...]:
        """Translate the output references of the callee of `call` into caller terms.

        A parameter of the callee becomes the reference of its argument in `args`.
        A root that the callee allocates and returns becomes a `Returned` root of
        `call`. An item at an index that the callee computes, and an allocation of
        the callee inside a list, are `Unknown`.
        """
        callee = call.callee
        params = callee.callable_region.blocks[0].args[1:]
        bound: dict[ir.SSAValue, Ref] = dict(zip(params, args))
        inputs: dict[ir.SSAValue, ir.SSAValue] = dict(zip(params, call.inputs))
        body = callee.callable_region
        seen: set[Root] = set()

        def translate(ref: Ref, position: int, nested: bool = False) -> Ref:
            match ref:
                case Whole(root) if root in bound:
                    return bound[root]
                case Slot(root, index) if root in bound:
                    if isinstance(index, ir.SSAValue):
                        if index not in inputs:
                            return Unknown(
                                "an item at an index that the callee computes"
                            )
                        index = inputs[index]
                    match bound[root]:
                        case Whole(outer):
                            return self.slot(outer, index)
                        case Items(members):
                            at = index if isinstance(index, int) else const_int(index)
                            return self.item(members, at)
                    return Unknown("an item of an argument that has no root")
                case Whole(root) if root_in(root, body):
                    if nested:
                        return Unknown("a list that holds an allocation of the call")
                    if root in seen:
                        return Unknown("an allocation returned at two positions")
                    seen.add(root)
                    return Whole(Returned(call, position))
                case Slot(root, _) if root_in(root, body):
                    return Unknown("an item of a register that the call allocates")
                case Members(members):
                    translated = (translate(m, position, True) for m in members)
                    return type(ref)(tuple(translated))
            return ref

        return tuple(translate(ref, position) for position, ref in enumerate(outputs))

    def run_loop(
        self,
        frame: ForwardFrame[Ref],
        stmt: ir.Statement,
        body: ir.Region,
        carried: tuple[Ref, ...],
    ) -> tuple[Ref, ...]:
        """Run a loop body until the references that it carries stop changing.

        The body gets `Untracked` for the loop index and the carried references.
        """
        while True:
            with self.new_frame(stmt, has_parent_access=True) as inner:
                yielded = self.frame_call_region(inner, stmt, body, UNTRACKED, *carried)
            if not isinstance(yielded, tuple) or len(yielded) != len(carried):
                return self.unknown_results(stmt, CARRIED)
            joined = tuple(c.join(y) for c, y in zip(carried, yielded, strict=True))
            if joined == carried:
                frame.entries.update(inner.entries)
                return carried
            carried = joined

    def run_branches(
        self,
        frame: ForwardFrame[Ref],
        stmt: ir.Statement,
        regions: Sequence[ir.Region],
        args: tuple[Ref, ...],
    ) -> tuple[Ref, ...]:
        """Run each branch with `args` and join the references that they yield."""
        joined: tuple[Ref, ...] | None = None
        complete = True
        for region in regions:
            with self.new_frame(stmt, has_parent_access=True) as inner:
                yielded = self.frame_call_region(inner, stmt, region, *args)
            frame.entries.update(inner.entries)
            if not isinstance(yielded, tuple) or len(yielded) != len(stmt.results):
                complete = False
            elif joined is None:
                joined = yielded
            else:
                joined = tuple(a.join(b) for a, b in zip(joined, yielded, strict=True))
        if joined is None or not complete:
            return self.unknown_results(stmt, CARRIED)
        return joined
