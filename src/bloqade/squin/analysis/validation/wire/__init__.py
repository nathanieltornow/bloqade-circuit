"""Check wire structure, types, uses, borrows and positions."""

from .borrows import (
    Slots as Slots,
    BorrowCheck as BorrowCheck,
    BorrowValidation as BorrowValidation,
)
from .position import PositionValidation as PositionValidation
from .linearity import LinearityValidation as LinearityValidation
from .validation import (
    WireValidation as WireValidation,
    StructureValidation as StructureValidation,
)
