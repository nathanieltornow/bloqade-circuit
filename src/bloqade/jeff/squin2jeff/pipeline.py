"""This module holds `SquinToJeff`, which converts a squin kernel to a jeff method."""

from dataclasses import dataclass

from kirin import ir

from bloqade import squin
from bloqade.jeff.errors import Refusal
from bloqade.jeff.analysis.validation.from_squin import SquinToJeffAnalysis

from .linearize import Linearize


@dataclass
class SquinToJeff:
    """A target that converts squin kernels to jeff dialect IR."""

    def emit(self, kernel: ir.Method) -> ir.Method:
        """Return the jeff method for `kernel` and leave `kernel` as it is.

        If jeff cannot express a construct of `kernel`, the method raises `Refusal`.
        """
        if not isinstance(kernel, ir.Method):
            raise TypeError(
                f"SquinToJeff.emit takes a squin kernel. "
                f"The argument has type {type(kernel).__name__}."
            )
        if kernel.dialects is not squin.kernel:
            raise TypeError(
                "SquinToJeff expects a method on the dialect group `squin.kernel`."
            )
        analysis = SquinToJeffAnalysis(kernel.dialects)
        analysis.run(kernel)
        if errors := analysis.get_validation_errors():
            raise Refusal(
                f"SquinToJeff cannot express '{kernel.sym_name}'. "
                f"Unsupported constructs: {len(errors)}.",
                errors,
            )
        emitter = Linearize(kernel.dialects, analysis.refs)
        emitter.run(kernel.code)
        return emitter.functions[kernel.code]
