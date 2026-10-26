"""Hold the squin wire dialect, a linear form of squin for optimization passes.

A wire is one qubit at one point in the program, and it has exactly one use. Each
statement consumes its input wires and returns new wires. `SquinToWire` builds a
wire squin method from a squin kernel, and `WireToSquin` turns it back.
"""

from . import twins as twins, rewrite as rewrite
from .group import kernel as kernel
from .stmts import (
    Free as Free,
    Alloc as Alloc,
    Borrow as Borrow,
    Release as Release,
    FreeRegister as FreeRegister,
    AllocRegister as AllocRegister,
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
from .validation import LinearityValidation as LinearityValidation
