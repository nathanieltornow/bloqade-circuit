"""Emit complete wire IR directly from source squin methods."""

from dataclasses import field, dataclass

from kirin import ir
from kirin.rewrite import Walk, Fixpoint, DeadCodeElimination

from bloqade.qubit import qalloc
from bloqade.squin.emit.wire import EmitWire
from bloqade.squin.wire.group import reference_kernel
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.signature import Contract
from bloqade.squin.emit.wire.methods import is_library
from bloqade.squin.passes.wire_target import ConversionTarget
from bloqade.squin.passes.wire_typeinfer import infer_wire_types
from bloqade.squin.analysis.validation.wire import WireValidation
from bloqade.squin.analysis.validation.wire.stages import SourceValidation, check_stage


def remaining(method: ir.Method) -> list[str]:
    """Return the operations and parameters that violate the complete wire conversion target."""
    return ConversionTarget.check(method)


@dataclass
class SquinToWire:
    """Emit source methods once and require complete, validated wire results."""

    prepared: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)
    contracts: dict[ir.Method, Contract] = field(default_factory=dict, init=False)
    active: set[ir.Method] = field(default_factory=set, init=False)

    def emit(self, method: ir.Method) -> ir.Method:
        """Build verified wire IR while preserving the source and its callees."""
        self.prepared, self.contracts, self.active = {}, {}, set()
        if not method.dialects.data <= reference_kernel.data:
            raise ValueError("SquinToWire expects squin or native kernel dialects.")
        return self.convert(method).method

    def convert(self, source: ir.Method) -> Contract:
        """Emit callees before their caller and propagate the first conversion failure."""
        if source in self.contracts:
            return self.contracts[source]
        if source in self.active:
            raise WireConversionError(
                f"Wire emission does not support recursive calls to '{source.sym_name}'.",
                source.code,
            )
        self.active.add(source)
        try:
            checked = check_stage(source, SourceValidation)
            for callee in sorted(checked.callees, key=lambda item: item.sym_name or ""):
                if callee is not qalloc and not is_library(callee):
                    self.convert(callee)
            contract = EmitWire(source.dialects, source, self.contracts).emit_method()
            output = contract.method
            Fixpoint(Walk(DeadCodeElimination())).rewrite(output.code)
            infer_wire_types(output)
            if left := remaining(output):
                raise WireConversionError(
                    f"'{output.sym_name}' violates the wire conversion target at: "
                    + ", ".join(sorted(set(left))),
                    output.code,
                )
            WireValidation().check(output)
            self.prepared[source] = output
            self.contracts[source] = contract
            return contract
        finally:
            self.active.remove(source)
