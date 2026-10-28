"""Check qubit positions through explicit wire operations and structured regions."""

from kirin import ir
from kirin.validation import ValidationPass

from .origins import Origins


class PositionValidation(ValidationPass[Origins]):
    """Require gates, releases, calls and yields to preserve their declared qubit positions."""

    def name(self) -> str:
        """Return the pass name in validation reports."""
        return "squin wire positions"

    def run(self, method: ir.Method) -> tuple[Origins, list[ir.ValidationError]]:
        """Follow explicit origins and report positional contract violations."""
        check = Origins(references=False)
        check.run(method)
        return check, check.errors
