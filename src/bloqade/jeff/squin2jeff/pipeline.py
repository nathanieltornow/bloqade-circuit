"""Convert squin through verified wire values before emitting Jeff."""

from dataclasses import dataclass

from kirin import ir, interp
from kirin.validation import ValidationSuite
from kirin.ir.exception import ValidationErrorGroup

from bloqade.squin.wire.group import kernel as wire_kernel, reference_kernel
from bloqade.squin.wire.errors import WireConversionError
from bloqade.jeff.analysis.validation import LinearityValidation, StructureValidation
from bloqade.squin.passes.wire_target import ConversionTarget
from bloqade.squin.rewrite.wire.driver import reachable
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.analysis.validation.wire import WireValidation

from .emitter import WireEmitter
from .isolate import IsolateRegions


@dataclass
class WireToJeff:
    """Emit Jeff from complete wire squin and verify every emitted function."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Translate explicit wire values without reconstructing reference identities."""
        if method.dialects is not wire_kernel:
            raise ValueError("WireToJeff expects squin.wire kernel dialects.")
        for callee in reachable(method):
            if illegal := ConversionTarget.check(callee):
                raise WireConversionError(
                    f"Jeff requires complete wire conversion: {', '.join(sorted(set(illegal)))}."
                )
            WireValidation().check(callee)
        emitter = WireEmitter(method.dialects)
        try:
            emitter.run(method.code)
        except interp.InterpreterError as error:
            raise ValidationErrorGroup(
                f"Jeff emission failed for '{method.sym_name}'.",
                [ir.ValidationError(method.code, str(error))],
            ) from error
        for function in emitter.functions.values():
            IsolateRegions(function.dialects)(function)
            ValidationSuite([StructureValidation, LinearityValidation]).validate(
                function
            ).raise_if_invalid()
        return emitter.functions[method.code]


@dataclass
class SquinToJeff:
    """Convert a squin kernel to wire squin, then emit Jeff directly from those wires."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return verified Jeff while preserving the source kernel and its callees."""
        if not method.dialects.data <= reference_kernel.data:
            raise ValueError(
                "SquinToJeff expects squin.kernel or native kernel dialects."
            )
        try:
            wired = SquinToWire().emit(method)
        except WireConversionError as error:
            raise ValidationErrorGroup(
                f"SquinToJeff cannot convert '{method.sym_name}'.",
                [ir.ValidationError(method.code, str(error))],
            ) from error
        return WireToJeff().emit(wired)
