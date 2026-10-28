"""Hold the squin wire dialect, a linear form of squin for optimization passes.

A wire is one qubit at one point in the program, and it has exactly one use. Each
statement consumes its input wires and returns new wires. `SquinToWire` builds a
wire squin method from a squin kernel, and `WireToSquin` turns it back.
"""

from . import impls as impls, twins as twins, rewrite as rewrite
from .group import kernel as kernel
from .stmts import (
    Free as Free,
    Wrap as Wrap,
    Alloc as Alloc,
    Borrow as Borrow,
    Length as Length,
    Unwrap as Unwrap,
    Release as Release,
    Reference as Reference,
    FreeRegister as FreeRegister,
    WrapRegister as WrapRegister,
    AllocRegister as AllocRegister,
    UnwrapRegister as UnwrapRegister,
    ReferenceRegister as ReferenceRegister,
)
from .twins import TWINS as TWINS
from .types import (
    Wire as Wire,
    Register as Register,
    WireType as WireType,
    RegisterType as RegisterType,
    is_linear as is_linear,
)
from .traits import (
    Twin as Twin,
    Noise as Noise,
    Threads as Threads,
    Unitary as Unitary,
    Measurement as Measurement,
    Preparation as Preparation,
)
from ._dialect import dialect as dialect
from .pipeline import SquinToWire as SquinToWire, WireToSquin as WireToSquin
from .validation import (
    Slots as Slots,
    BorrowCheck as BorrowCheck,
    WireValidation as WireValidation,
    BorrowValidation as BorrowValidation,
    LinearityValidation as LinearityValidation,
    ReferenceValidation as ReferenceValidation,
    StructureValidation as StructureValidation,
)
