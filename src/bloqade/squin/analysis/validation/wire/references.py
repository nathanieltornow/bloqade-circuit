"""Check ownership transfers between source references and explicit wires."""

from kirin import ir
from kirin.validation import ValidationPass

from .origins import Origins


class ReferenceValidation(ValidationPass[Origins]):
    """Check that reference operations respect outstanding unwraps and wraps."""

    def name(self) -> str:
        """Return the pass name in validation reports."""
        return "squin wire references"

    def run(self, method: ir.Method) -> tuple[Origins, list[ir.ValidationError]]:
        """Check transfers using handles from an ordered traversal of definitions."""
        check = Origins(positions=False)
        check.run(method)
        return check, check.errors
