"""This module holds `JeffToSquin`, which converts a jeff method to a squin kernel."""

from dataclasses import dataclass

from kirin import ir
from kirin.passes import Fold
from kirin.rewrite import Walk
from kirin.dialects.scf.unroll import PickIfElse

from bloqade import squin
from bloqade.jeff.errors import Refusal, JeffToSquinError
from bloqade.jeff.dialects import kernel as jeff_kernel
from bloqade.jeff.analysis.reference import WireReferenceAnalysis
from bloqade.jeff.analysis.validation import JeffToSquinValidation

from .jeff2py import JeffToPy
from .jeff2scf import JeffToScf
from .delinearize import Delinearize


@dataclass
class JeffToSquin:
    """A target that converts jeff dialect IR to squin kernels."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return the squin kernel for `method` and leave `method` as it is.

        If squin cannot express a construct of `method`, the method raises `Refusal`.
        Squin integers do not wrap, so a result outside the int32 range differs.
        """
        if not isinstance(method, ir.Method):
            raise TypeError(
                f"JeffToSquin.emit takes a kirin Method from `load_jeff`. "
                f"The argument has type {type(method).__name__}."
            )
        if method.dialects is not jeff_kernel:
            raise JeffToSquinError(
                method.code,
                "JeffToSquin expects a method on the dialect group `jeff.kernel`. "
                "This method is on another group. It may already be a squin kernel.",
            )
        references = WireReferenceAnalysis(jeff_kernel)
        references.run(method)
        _, errors = JeffToSquinValidation(results=references.results).run(method)
        if errors:
            raise Refusal(
                f"JeffToSquin cannot express '{method.sym_name}'. "
                f"Unsupported constructs: {len(errors)}.",
                errors,
            )
        # The passes below mix kirin statements into the jeff code of each kernel.
        mixed = jeff_kernel.union(squin.kernel)
        kernels = {function: function.similar(mixed) for function in references.results}
        for kernel in kernels.values():
            JeffToPy(kernel.dialects).unsafe_run(kernel)
            JeffToScf(kernel.dialects).unsafe_run(kernel)
            Delinearize(kernel.dialects, kernels=kernels).unsafe_run(kernel)
            kernel.dialects = squin.kernel
        # Squin's type inference types a callee when it first reaches a call.
        # So it runs after every function is squin code.
        for kernel in kernels.values():
            # Squin's type inference types neither branch of an `if` on a constant.
            fold = Fold(kernel.dialects)
            fold.unsafe_run(kernel)
            while Walk(PickIfElse()).rewrite(kernel.code).has_done_something:
                fold.unsafe_run(kernel)
            if run_pass := squin.kernel.run_pass:
                run_pass(kernel, fold=False)
        return kernels[method]
