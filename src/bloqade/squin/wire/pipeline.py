"""Convert squin kernels to wire squin and back with `SquinToWire` and `WireToSquin`."""

from dataclasses import field, dataclass
from collections.abc import Mapping, Callable

from kirin import ir
from kirin.passes import TypeInfer
from kirin.rewrite import (
    Walk,
    Chain,
    Inline,
    Fixpoint,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade import squin
from bloqade.qubit import qalloc
from bloqade.analysis.reference import (
    Ref,
    Place,
    Members,
    Untracked,
    roots,
    disjoint,
)
from bloqade.squin.analysis.lifetime import Effect, QubitEffects
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

from .to_wire import ToWire
from .from_wire import FromWire
from .normalize import (
    MapToLoop,
    Concatenate,
    IndexInLoop,
    IndexInBranch,
    QubitIteration,
)
from .validation import WireValidation

LIBRARY = ("bloqade.squin.stdlib", "bloqade.qubit.stdlib")
"""Name the modules whose kernels wrap one squin statement each."""


def is_library(kernel: ir.Method) -> bool:
    """Return True if `kernel` is a squin library kernel other than `qalloc`.

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


@dataclass
class SquinToWire:
    """Convert a squin kernel to a wire squin method and leave the kernel as it is.

    A call stays a borrowing call if every qubit argument is one qubit or one
    whole register, if no two arguments overlap, and if the callee returns only
    qubits that it allocates. Kirin's `Inline` rule inlines every other call. If
    `assume_disjoint` is True, the converter trusts that the qubits of one gate
    differ when the places cannot prove it.
    """

    assume_disjoint: bool = False
    prepared: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)
    refs: dict[ir.SSAValue, Ref] = field(default_factory=dict, init=False)
    effects: dict[ir.Statement, Effect] = field(default_factory=dict, init=False)

    def emit(self, method: ir.Method) -> ir.Method:
        """Return the wire squin method of `method`.

        The method raises a kirin `InterpreterError` if a construct of `method`
        has no wire form, and a `ValidationErrorGroup` if the result breaks a
        rule of `WireValidation`.
        """
        if method.dialects is not squin.kernel:
            raise ValueError("SquinToWire expects a method on `squin.kernel`.")
        self.prepared, self.refs, self.effects = {}, {}, {}
        self.prepare(method)
        emitter = ToWire(
            method.dialects,
            self.refs,
            self.effects,
            kernels=self.prepared,
            assume_disjoint=self.assume_disjoint,
        )
        top = self.prepared[method]
        emitter.run(top)
        for wired in emitter.functions.values():
            Fixpoint(Walk(CommonSubexpressionElimination())).rewrite(wired.code)
            Fixpoint(Walk(DeadCodeElimination())).rewrite(wired.code)
        result = emitter.functions[top.code]
        WireValidation().check(result)
        return result

    def prepare(self, kernel: ir.Method) -> None:
        """Copy `kernel`, normalize the copy, and analyze it.

        The method inlines every library call and runs the rules of
        `bloqade.squin.wire.normalize`. After each analysis, it joins concatenated
        registers and inlines each call that `borrowable` rejects. An inlined body
        can hold new calls and new patterns, so the method repeats both steps until
        nothing changes. Then it prepares each kernel that a kept call reaches.
        """
        if kernel in self.prepared:
            return
        copy = kernel.similar()
        self.prepared[kernel] = copy
        self.normalize(copy)
        while True:
            frame, _ = QubitReferenceAnalysis(copy.dialects).run(copy)
            rejected = InlineCalls(
                lambda call: not self.borrowable(call, frame.entries)
            )
            rules = Chain(Concatenate(frame.entries), rejected)
            if not Walk(rules).rewrite(copy.code).has_done_something:
                break
            self.normalize(copy)
        effects = QubitEffects(copy.dialects, refs=frame.entries)
        effects.run(copy)
        self.refs.update(frame.entries)
        self.effects.update(effects.effects)
        for callee in rejected.kept:
            self.prepare(callee)

    @staticmethod
    def normalize(copy: ir.Method) -> None:
        """Inline the library calls of `copy`, run the structural rules, and infer types.

        An inlined body keeps the types of its callee, and some of them are `Any`.
        `QubitEffects` must treat a value of type `Any` as a value that may hide a
        qubit. Kirin's type inference gives such a value its precise type.
        """
        rules = Chain(
            InlineCalls(lambda call: is_library(call.callee)),
            QubitIteration(),
            MapToLoop(),
            IndexInBranch(),
            IndexInLoop(),
        )
        Fixpoint(Walk(rules)).rewrite(copy.code)
        Fixpoint(Walk(DeadCodeElimination())).rewrite(copy.code)
        TypeInfer(copy.dialects)(copy)

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
        if not self.assume_disjoint:
            for k, a in enumerate(qubits):
                if any(not disjoint(a, b) for b in qubits[k + 1 :]):
                    return False
        return _returns_only_fresh(call.callee)


@dataclass
class WireToSquin:
    """Convert a wire squin method and every wire callee back to squin kernels."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return a squin kernel that runs the same program as `method`."""
        rewrite = FromWire()
        return rewrite.convert(method)
