"""Report source locations and conditions that prevent conversion."""

from kirin import ir
from kirin.ir.exception import PotentialValidationError


class WireConversionError(Exception):
    """Report the source operation and condition that prevent wire conversion."""

    def __init__(self, condition: str, node: ir.Statement | None = None):
        """Retain the failed condition and source location for callers and diagnostic tests."""
        self.condition = condition
        self.node = node
        self.source = node.source if node is not None else None
        super().__init__(
            condition if self.source is None else f"{condition}\n{self.source!r}"
        )


class UnsupportedCallContract(WireConversionError):
    """Report an unsupported layout of call arguments or results."""


class UnresolvedArgumentSeparation(WireConversionError):
    """Report call operands whose quantum locations lack proven separation."""


class UnresolvedBorrowSeparation(PotentialValidationError):
    """Report register indices whose separation can depend on caller-provided values."""
