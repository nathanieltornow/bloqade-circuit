"""Track qubit references through squin kernels.

Each SSA value maps to an allocation, a parameter, a returned allocation,
or an element of a register. Calls preserve references to their arguments
and give returned allocations a root at the call site.
"""

from . import impls as impls
from .lattice import (
    Ref as Ref,
    Slot as Slot,
    Whole as Whole,
    Unknown as Unknown,
    Returned as Returned,
    Classical as Classical,
    Positions as Positions,
    QubitList as QubitList,
)
from .analysis import (
    References as References,
    ReferenceAnalysis as ReferenceAnalysis,
    analyze as analyze,
)
