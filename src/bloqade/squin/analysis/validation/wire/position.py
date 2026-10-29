"""Check qubit positions through a Kirin validation analysis."""

from kirin import ir
from kirin.validation import ValidationPass
from kirin.analysis.forward import ForwardFrame

from bloqade.squin.wire.types import is_linear

from .origins import PositionAnalysis
from .identities import Handle, Origin


class PositionValidation(ValidationPass[ForwardFrame[Origin]]):
    """Require gates, releases, calls and yields to preserve their declared qubit positions."""

    def name(self) -> str:
        """Return the pass name in validation reports."""
        return "squin wire positions"

    def run(
        self, method: ir.Method
    ) -> tuple[ForwardFrame[Origin], list[ir.ValidationError]]:
        """Run origin analysis with a distinct identity for each quantum parameter."""
        check = PositionAnalysis(method.dialects)
        inputs = tuple(
            Origin(Handle(arg, arg.type)) if is_linear(arg.type) else Origin.top()
            for arg in method.args
        )
        frame, _ = check.run(method, *inputs)
        return frame, check.get_validation_errors()
