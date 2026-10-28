"""Convert normalized squin through local transfers and one ordered wire traversal."""

from dataclasses import field, dataclass
from collections.abc import Callable

from kirin import ir
from kirin.passes import TypeInfer
from kirin.rewrite import Walk, Chain, Inline, DeadCodeElimination
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import qalloc
from bloqade.squin.wire.group import reference_kernel
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.passes.wire_target import ConversionTarget
from bloqade.squin.rewrite.wire.lower import Contract
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint, reachable
from bloqade.squin.rewrite.wire.promote import (
    Conversion,
    InlineRequired,
    ReferencesRequired,
)
from bloqade.squin.rewrite.wire.normalize import (
    MapToLoop,
    IndexInLoop,
    IndexInBranch,
    QubitIteration,
    TupleProjection,
    BoundListLengths,
)
from bloqade.squin.analysis.validation.wire import WireValidation

LIBRARY = ("bloqade.squin.stdlib", "bloqade.qubit.stdlib", "bloqade.native.stdlib")
"""Name the library modules whose kernels the converter inlines."""


def is_library(kernel: ir.Method) -> bool:
    """Return True if `kernel` belongs to an inlined library and differs from `qalloc`.

    A library kernel, such as `squin.h`, wraps one squin statement. Inlining it
    exposes the statement to the converter. The ordered converter treats `qalloc` as a register allocation.
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


def remaining(method: ir.Method) -> list[str]:
    """Return the operations and parameters that violate the complete wire conversion target."""
    return ConversionTarget.check(method)


@dataclass
class SquinToWire:
    """Convert squin functions using checked call contracts and ordered wire values.

    Complete conversion visits each normalized function once after resolving required inlining.
    Partial conversion can retry private copies to preserve allocations with unsupported uses.
    Every returned method passes wire validation, and failures preserve all source methods.
    """

    max_iterations: int = 32
    partial: bool = False
    prepared: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)
    contracts: dict[ir.Method, Contract] = field(default_factory=dict, init=False)
    report: list[str] = field(default_factory=list, init=False)
    diagnostics: dict[ir.Method, list[str]] = field(default_factory=dict, init=False)
    promote: bool = field(default=True, init=False)

    def emit(self, method: ir.Method) -> ir.Method:
        """Return verified wire squin while preserving the source and its callees."""
        return self._emit(method, promote=True)

    def lower(self, method: ir.Method) -> ir.Method:
        """Return the executable transfer form before eliminating reference transfers."""
        return self._emit(method, promote=False)

    def _emit(self, method: ir.Method, *, promote: bool) -> ir.Method:
        self.prepared, self.contracts, self.diagnostics, self.report = {}, {}, {}, []
        self.promote = promote
        if not method.dialects.data <= reference_kernel.data:
            raise ValueError("SquinToWire expects squin or native kernel dialects.")
        if self.max_iterations < 1:
            raise ValueError("The rewrite iteration limit must be positive.")
        reachable(method, skip=lambda callee: callee is qalloc or is_library(callee))
        result = self.convert(method).method
        for copy in reachable(result, skip=lambda callee: callee is qalloc):
            self.report.extend(self.diagnostics.get(copy, ()))
            if promote and (left := remaining(copy)):
                message = (
                    f"'{copy.sym_name}' violates the wire conversion target at: "
                    + ", ".join(sorted(set(left)))
                )
                if not self.partial:
                    raise WireConversionError(message)
                self.report.append(message)
            WireValidation().check(copy)
        return result

    def convert(self, source: ir.Method) -> Contract:
        """Convert a callee on demand and cache its checked wire signature."""
        if source in self.contracts:
            return self.contracts[source]
        inline: set[ir.Method] = set()
        blocked: set[int] = set()
        block_all = False
        reasons: set[str] = set()
        for _ in range(self.max_iterations):
            copy = source.similar()
            self.normalize(copy, self.max_iterations, inline)
            conversion = Conversion(
                copy, self.convert, self.partial, self.promote, blocked, block_all
            )
            try:
                conversion.run()
            except InlineRequired as error:
                if error.callee in inline:
                    raise WireConversionError(
                        f"Inlining '{error.callee.sym_name}' did not resolve its call contract."
                    ) from error
                inline.add(error.callee)
                blocked, block_all = set(), False
                continue
            except ReferencesRequired as error:
                reasons.add(error.reason)
                if error.allocations is None:
                    block_all = True
                else:
                    blocked = blocked | error.allocations
                continue
            assert conversion.contract is not None
            self.prepared[source] = copy
            self.contracts[source] = conversion.contract
            self.diagnostics[copy] = [
                f"'{copy.sym_name}': {reason}"
                for reason in sorted(reasons | set(conversion.report))
            ]
            return conversion.contract
        raise WireConversionError(
            f"Conversion of '{source.sym_name}' did not converge within {self.max_iterations} iterations."
        )

    @staticmethod
    def normalize(
        copy: ir.Method, max_iterations: int = 32, inline: set[ir.Method] | None = None
    ) -> None:
        """Expose source operations and carry selected register indices as explicit classical values."""
        selected = inline or set()
        rules = Chain(
            InlineCalls(
                lambda call: is_library(call.callee) or call.callee in selected
            ),
            QubitIteration(),
            MapToLoop(),
            IndexInBranch(),
            IndexInLoop(),
            TupleProjection(),
        )
        CheckedFixpoint(Walk(rules), max_iter=max_iterations).rewrite(copy.code)
        CheckedFixpoint(Walk(DeadCodeElimination())).rewrite(copy.code)
        TypeInfer(copy.dialects)(copy)
        Walk(BoundListLengths()).rewrite(copy.code)
        copy.verify()
        copy.verify_type()
