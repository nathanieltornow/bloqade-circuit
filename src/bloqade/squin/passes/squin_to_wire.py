"""Convert squin kernels to wire squin with explicit conversion and validation checks."""

from dataclasses import field, dataclass
from collections.abc import Mapping, Callable

from kirin import ir
from kirin.passes import TypeInfer
from kirin.rewrite import (
    Walk,
    Chain,
    Inline,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import qalloc
from bloqade.squin.wire.group import reference_kernel
from bloqade.squin.wire.errors import WireConversionError
from bloqade.analysis.reference import Ref, Place, Members, Untracked, roots, disjoint
from bloqade.squin.analysis.reference import QubitReferenceAnalysis
from bloqade.squin.passes.wire_target import ConversionTarget
from bloqade.squin.rewrite.wire.lower import lower, layout_of
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint, reachable
from bloqade.squin.rewrite.wire.promote import promote
from bloqade.squin.rewrite.wire.normalize import (
    MapToLoop,
    Concatenate,
    IndexInLoop,
    IndexInBranch,
    QubitIteration,
    BroadcastToLoop,
    TupleProjection,
    BoundListLengths,
)
from bloqade.squin.analysis.validation.wire.validation import WireValidation

LIBRARY = ("bloqade.squin.stdlib", "bloqade.qubit.stdlib", "bloqade.native.stdlib")
"""Name the library modules whose kernels the converter inlines."""


def is_library(kernel: ir.Method) -> bool:
    """Return True if `kernel` belongs to an inlined library and differs from `qalloc`.

    A library kernel, such as `squin.h`, wraps one squin statement. Inlining it
    exposes the statement to the converter. `QubitReferenceAnalysis` reads a call
    of `qalloc` as one register root, so that call stays.
    """
    module = kernel.mod.__name__ if kernel.mod is not None else ""
    return kernel is not qalloc and module.startswith(LIBRARY)


@dataclass
class InlineCalls(RewriteRule):
    """Inline each call that `select` picks, with kirin's `Inline` rule.

    A call of a closure, such as a function that the kernel defines inside its
    body, always gets inlined. `kept` collects the callee of each call that stays
    a call.
    """

    select: Callable[[func.Invoke], bool]
    kept: list[ir.Method] = field(default_factory=list)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Inline `node` if it is a call that `select` picks."""
        if isinstance(node, func.Call):
            return Inline(heuristic=lambda _: True).rewrite(node)
        if not isinstance(node, func.Invoke):
            return RewriteResult()
        if not self.select(node):
            if node.callee is not qalloc:
                self.kept.append(node.callee)
            return RewriteResult()
        return Inline(heuristic=lambda _: True).rewrite(node)


def _returns_only_fresh(kernel: ir.Method) -> bool:
    """Return True if every qubit that `kernel` returns comes from its own allocation.

    Guppy and MQT's QCO dialect never let a callee return a qubit that it
    borrows. If `kernel` returns a qubit parameter, the converter inlines the call.
    """
    _, result = QubitReferenceAnalysis(kernel.dialects).run(kernel)
    params = set(kernel.callable_region.blocks[0].args[1:])
    members = result.refs if isinstance(result, Members) else (result,)
    for member in members:
        if isinstance(member, Untracked):
            continue
        if not (isinstance(member, Place) and not member.path):
            return False
        if any(root in params for root in roots(member)):
            return False
    return True


def remaining(method: ir.Method) -> list[str]:
    """Return the operations and parameters that violate the wire conversion target."""
    return ConversionTarget.check(method)


@dataclass
class SquinToWire:
    """Convert a squin or native kernel to wire squin and keep the original kernel.

    A call stays a borrowing call if every qubit argument is one qubit or one
    whole register, if no two arguments overlap, and if the callee returns only
    qubits that it allocates. Kirin's `Inline` rule inlines every other call.
    A public wire function requires distinct quantum arguments from its caller.

    If `partial` is True, a statement whose qubits the places can't name keeps
    reference semantics, and the rest of the kernel still becomes wire squin.
    Each such statement adds one line to `report`. If `partial` is False, the
    converter raises a `WireConversionError` for it.
    """

    max_iterations: int = 32
    partial: bool = False
    prepared: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)
    refs: dict[ir.SSAValue, Ref] = field(default_factory=dict, init=False)
    report: list[str] = field(default_factory=list, init=False)
    """Hold one line for each statement that kept reference semantics."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return the wire squin method of `method`.

        The method prepares `method` and every kernel that a kept call reaches,
        lowers each copy onto `Unwrap`, twin and `Wrap`, and promotes the copies
        to wires. It raises a `WireConversionError` if a construct of `method` has
        no wire form, and a `ValidationErrorGroup` if a result breaks a rule of
        `WireValidation`.
        """
        if not method.dialects.data <= reference_kernel.data:
            raise ValueError("SquinToWire expects squin or native kernel dialects.")
        self.prepared, self.refs, self.report = {}, {}, []
        if self.max_iterations < 1:
            raise ValueError("The rewrite iteration limit must be positive.")
        reachable(method, skip=lambda callee: callee is qalloc or is_library(callee))
        self.prepare(method)
        copies = list(self.prepared.values())
        layouts = {copy.code: layout_of(copy.code, self.refs) for copy in copies}
        for copy in copies:
            reasons = lower(copy, self.refs, self.prepared, self.partial, layouts)
            self.report += [f"'{copy.sym_name}': {reason}" for reason in reasons]
        for copy in copies:
            CheckedFixpoint(
                Walk(Chain(TupleProjection(), DeadCodeElimination()))
            ).rewrite(copy.code)
            promote(copy, layouts[copy.code])
            CheckedFixpoint(Walk(CommonSubexpressionElimination())).rewrite(copy.code)
            CheckedFixpoint(Walk(DeadCodeElimination())).rewrite(copy.code)
        for copy in copies:
            if left := remaining(copy):
                message = (
                    f"'{copy.sym_name}' violates the wire conversion target at: "
                    + ", ".join(sorted(set(left)))
                )
                if not self.partial:
                    raise WireConversionError(message)
                self.report.append(message)
            WireValidation().check(copy)
        return self.prepared[method]

    def prepare(self, kernel: ir.Method) -> None:
        """Copy `kernel`, normalize the copy, and analyze it.

        The method inlines every library call and runs the rules of
        `bloqade.squin.rewrite.wire.normalize`. After each analysis, it expands broadcasts
        of unknown length and joins concatenated registers. It then inlines each
        call that `borrowable` rejects. An inlined body
        can hold new calls and new patterns, so the method repeats both steps until
        nothing changes. Then it prepares each kernel that a kept call reaches.
        """
        if kernel in self.prepared:
            return
        copy = kernel.similar()
        self.prepared[kernel] = copy
        self.normalize(copy, self.max_iterations)
        for _ in range(self.max_iterations):
            frame, _ = QubitReferenceAnalysis(copy.dialects).run(copy)
            rejected = InlineCalls(
                lambda call: not self.borrowable(call, frame.entries)
            )
            rules = Chain(
                Concatenate(frame.entries), BroadcastToLoop(frame.entries), rejected
            )
            if not Walk(rules).rewrite(copy.code).has_done_something:
                break
            self.normalize(copy, self.max_iterations)
        else:
            raise WireConversionError(
                f"Preparation of '{copy.sym_name}' did not converge within {self.max_iterations} iterations."
            )
        copy.verify()
        copy.verify_type()
        self.refs.update(frame.entries)
        for callee in rejected.kept:
            self.prepare(callee)

    @staticmethod
    def normalize(copy: ir.Method, max_iterations: int = 32) -> None:
        """Inline the library calls of `copy`, run the structural rules, and infer types.

        An inlined body keeps the types of its callee, and some of them are `Any`.
        The reference analysis reads the type of a value to decide whether it holds
        qubits, so a qubit of type `Any` would go untracked. Kirin's type inference
        gives such a value its precise type.
        """
        rules = Chain(
            InlineCalls(lambda call: is_library(call.callee)),
            QubitIteration(),
            MapToLoop(),
            IndexInBranch(),
            IndexInLoop(),
        )
        CheckedFixpoint(Walk(rules), max_iter=max_iterations).rewrite(copy.code)
        CheckedFixpoint(Walk(DeadCodeElimination())).rewrite(copy.code)
        TypeInfer(copy.dialects)(copy)
        Walk(BoundListLengths()).rewrite(copy.code)

    def borrowable(self, call: func.Invoke, refs: Mapping[ir.SSAValue, Ref]) -> bool:
        """Return True if `call` can stay a call that borrows its qubit arguments."""
        if call.callee is qalloc:
            return True
        if not isinstance(call.callee.code, func.Function):
            return False
        passed = [refs.get(arg, Untracked()) for arg in call.inputs]
        qubits = [ref for ref in passed if not isinstance(ref, Untracked)]
        for ref in qubits:
            if not (isinstance(ref, Place) and (ref.one_item or not ref.path)):
                return False
        for k, a in enumerate(qubits):
            if any(not disjoint(a, b) for b in qubits[k + 1 :]):
                return False
        return _returns_only_fresh(call.callee)
