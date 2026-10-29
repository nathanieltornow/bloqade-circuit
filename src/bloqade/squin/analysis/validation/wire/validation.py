"""Check wire squin with the kirin validation suite `WireValidation`.

`StructureValidation` runs kirin's own `verify` and `verify_type`.
`LinearityValidation` checks that each wire and register has one use.
`BorrowValidation` checks that each borrowed element returns and that each moved
element is filled or discarded.
`PositionValidation` checks the order of wires across control flow and calls.
"""

from dataclasses import replace, dataclass

from kirin import ir
from kirin.validation import ValidationPass, ValidationSuite

from bloqade.squin.analysis.validation.wire.borrows import BorrowValidation
from bloqade.squin.analysis.validation.wire.position import PositionValidation
from bloqade.squin.analysis.validation.wire.linearity import LinearityValidation


class StructureValidation(ValidationPass[None]):
    """Run kirin's own structure and type checks on a method.

    The suite runs this check on wire methods, and `check_stage` runs it on source.
    """

    def name(self) -> str:
        """Return the pass name that validation reports show."""
        return "squin wire structure"

    def run(self, method: ir.Method) -> tuple[None, list[ir.ValidationError]]:
        """Return the error of `verify` or `verify_type`, if either check fails."""
        try:
            method.code.verify()
            method.code.verify_type()
        except ir.ValidationError as error:
            error.source = None if error.source is None else replace(error.source)
            return None, [error]
        return None, []


@dataclass
class WireValidation:
    """Run the wire structure, linearity, borrow and position checks."""

    def check(self, method: ir.Method) -> None:
        """Raise a `ValidationErrorGroup` that lists every broken rule of `method`."""
        suite = ValidationSuite(
            [
                StructureValidation,
                LinearityValidation,
                BorrowValidation,
                PositionValidation,
            ]
        )
        suite.validate(method).raise_if_invalid()
