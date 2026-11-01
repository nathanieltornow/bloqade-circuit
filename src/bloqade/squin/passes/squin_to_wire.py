"""Emit complete wire IR directly from source squin methods."""

from dataclasses import field, dataclass

from kirin import ir
from kirin.passes import TypeInfer
from kirin.rewrite import Walk, Fixpoint, DeadCodeElimination
from kirin.dialects import func
from kirin.validation import ValidationSuite

from bloqade.squin.emit.wire import EmitWire
from bloqade.squin.wire.group import reference_kernel
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.passes.wire_target import ConversionTarget
from bloqade.squin.emit.wire.functions import is_library
from bloqade.squin.analysis.validation.wire import (
    WireValidation,
    StructureValidation,
)
from bloqade.squin.analysis.validation.wire.stages import SourceValidation


@dataclass
class SquinToWire:
    """Emit source methods once and require complete, validated wire results."""

    converted: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)
    """The validated wire method of each converted source method."""
    active: set[ir.Method] = field(default_factory=set, init=False)
    """The source methods whose conversion is in progress, used to reject recursion."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Build verified wire IR while preserving the source and its callees."""
        self.converted, self.active = {}, set()
        if not method.dialects.data <= reference_kernel.data:
            raise ValueError("SquinToWire expects squin or native kernel dialects.")
        return self.convert(method)

    def convert(self, source: ir.Method) -> ir.Method:
        """Convert the callees of `source`, then `source` itself, and cache the result."""
        if source in self.converted:
            return self.converted[source]
        if source in self.active:
            raise WireConversionError(
                f"Wire emission does not support recursive calls to '{source.sym_name}'.",
                source.code,
            )
        self.active.add(source)
        try:
            ValidationSuite(
                [StructureValidation, SourceValidation], fail_fast=True
            ).validate(source).raise_if_invalid()
            callees = {
                node.callee
                for node in source.code.walk()
                if isinstance(node, func.Invoke)
            }
            # Sorting by name gives a deterministic conversion order. A library
            # wrapper gets no conversion, because `run_library` emits its body.
            for callee in sorted(callees, key=lambda item: item.sym_name or ""):
                if not is_library(callee):
                    self.convert(callee)
            method = self.emit_and_validate(source)
            self.converted[source] = method
            return method
        finally:
            self.active.remove(source)

    def emit_and_validate(self, source: ir.Method) -> ir.Method:
        """Emit the wire method of `source`, infer its types and validate it.

        Dead-code removal runs first, so type inference reads only live operations.
        """
        method = EmitWire(source.dialects, source, self.converted).emit_method()
        Fixpoint(Walk(DeadCodeElimination())).rewrite(method.code)
        TypeInfer(method.dialects)(method)
        if violations := ConversionTarget.check(method):
            raise WireConversionError(
                f"'{method.sym_name}' violates the wire conversion target at: "
                + ", ".join(sorted(set(violations))),
                method.code,
            )
        WireValidation().check(method)
        return method
