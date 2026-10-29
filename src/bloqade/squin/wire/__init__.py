"""Hold the squin wire dialect, which makes quantum state flow explicit.

A wire is one qubit at one point in the program, and it has exactly one use. Each
statement consumes its input wires and returns new wires. The conversion passes live in `bloqade.squin.passes`.
"""

from . import twins as twins
from .group import kernel as kernel
from .stmts import (
    Free as Free,
    Alloc as Alloc,
    Borrow as Borrow,
    Length as Length,
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
from .errors import WireConversionError as WireConversionError
from .traits import (
    Twin as Twin,
    Noise as Noise,
    Threads as Threads,
    Unitary as Unitary,
    Measurement as Measurement,
    Preparation as Preparation,
)
from ._dialect import dialect as dialect
