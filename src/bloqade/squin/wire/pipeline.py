"""Convert squin kernels to wire squin and back with `SquinToWire` and `WireToSquin`."""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import (
    Walk,
    Inline,
    Fixpoint,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)

from bloqade import squin
from bloqade.qubit import qalloc
from bloqade.squin.analysis.lifetime import QubitEffects
from bloqade.squin.analysis.reference import QubitReferenceAnalysis

from .to_wire import ToWire
from .from_wire import FromWire
from .validation import LinearityValidation


def _inlined(method: ir.Method) -> ir.Method:
    """Return a copy of `method` with every call inlined except the calls of `qalloc`.

    `QubitReferenceAnalysis` reads a `qalloc` call as one register root. So the
    call stays, and the converter turns it into one `AllocRegister`.
    """
    copy = method.similar()
    Fixpoint(Walk(Inline(lambda code: code is not qalloc.code))).rewrite(copy.code)
    Fixpoint(Walk(DeadCodeElimination())).rewrite(copy.code)
    return copy


@dataclass
class SquinToWire:
    """Convert a squin kernel to a wire squin method and leave the kernel as it is.

    If `assume_disjoint` is True, the converter trusts that the qubits of one
    gate differ when the places cannot prove it.
    """

    assume_disjoint: bool = False

    def emit(self, method: ir.Method) -> ir.Method:
        """Return the wire squin method of `method`.

        The method raises a kirin `InterpreterError` if a construct of `method`
        has no wire form, and a `ValidationErrorGroup` if the result breaks the
        linearity rules.
        """
        if method.dialects is not squin.kernel:
            raise ValueError("SquinToWire expects a method on `squin.kernel`.")
        copy = _inlined(method)
        frame, _ = QubitReferenceAnalysis(copy.dialects).run(copy)
        effects = QubitEffects(copy.dialects, refs=frame.entries)
        effects.run(copy)
        emitter = ToWire(copy.dialects, frame.entries, effects.effects)
        emitter.assume_disjoint = self.assume_disjoint
        emitter.run(copy)
        assert emitter.method is not None
        wired = emitter.method
        Fixpoint(Walk(CommonSubexpressionElimination())).rewrite(wired.code)
        Fixpoint(Walk(DeadCodeElimination())).rewrite(wired.code)
        LinearityValidation().check(wired)
        return wired


@dataclass
class WireToSquin:
    """Convert a wire squin method back to a squin kernel."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return a squin kernel that runs the same program as `method`."""
        copy = method.similar(squin.kernel)
        FromWire().rewrite(copy)
        return copy
