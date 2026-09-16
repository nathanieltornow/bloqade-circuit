"""This module holds `JeffToSquin`, which converts a jeff method to a squin kernel."""

from dataclasses import field, dataclass

from kirin import ir
from kirin.passes import Fold
from kirin.rewrite import Walk
from kirin.dialects.scf.unroll import PickIfElse

from bloqade import squin
from bloqade.jeff.errors import Refusal, JeffToSquinError
from bloqade.jeff.dialects import kernel as jeff_kernel
from bloqade.jeff.analysis.validation import JeffToSquinValidation

from .jeff2py import JeffToPy
from .jeff2scf import JeffToScf
from .delinearize import Delinearize, handed_back


@dataclass
class JeffToSquin:
    """A target that converts jeff dialect IR to squin kernels."""

    kernels: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)
    """The squin kernel of each jeff function of the last conversion."""
    layouts: dict[ir.Method, tuple[ir.BlockArgument | None, ...]] = field(
        default_factory=dict, init=False
    )
    """The parameter that each output of a jeff function hands back."""

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
        _, errors = JeffToSquinValidation().run(method)
        if errors:
            raise Refusal(
                f"JeffToSquin cannot express '{method.sym_name}'. "
                f"Unsupported constructs: {len(errors)}.",
                errors,
            )
        self.kernels = {}
        self.layouts = {}
        main = self.kernel(method)
        # Squin's type inference types a callee when it first reaches a call.
        # So it runs after every function is squin code.
        for kernel in self.kernels.values():
            # Squin's type inference types neither branch of an `if` on a constant.
            fold = Fold(kernel.dialects)
            fold.unsafe_run(kernel)
            while Walk(PickIfElse()).rewrite(kernel.code).has_done_something:
                fold.unsafe_run(kernel)
            if run_pass := squin.kernel.run_pass:
                run_pass(kernel, fold=False)
        return main

    def handed(self, function: ir.Method) -> tuple[ir.BlockArgument | None, ...]:
        """Return the parameter that each output of `function` hands back, memoized."""
        if function not in self.layouts:
            self.layouts[function] = handed_back(function)
        return self.layouts[function]

    def kernel(self, function: ir.Method) -> ir.Method:
        """Return the squin kernel of the jeff `function`, and build it on first request.

        The kernel is registered before its body is rewritten, so that a call back
        to `function` inside its own call tree finds it.
        """
        if function in self.kernels:
            return self.kernels[function]
        # The passes mix kirin statements into the jeff code of the kernel.
        kernel = self.kernels[function] = function.similar(
            jeff_kernel.union(squin.kernel)
        )
        JeffToPy(kernel.dialects).unsafe_run(kernel)
        JeffToScf(kernel.dialects).unsafe_run(kernel)
        Delinearize(kernel.dialects, kernel=self.kernel, handed=self.handed).unsafe_run(
            kernel
        )
        kernel.dialects = squin.kernel
        return kernel
