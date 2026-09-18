"""This package holds the analysis that states which root each value refers to.

`ReferenceAnalysis` holds the rules for kirin's own dialects. `QubitReferenceAnalysis`
adds the squin qubit and register roots, and `WireReferenceAnalysis` adds the jeff
wire and register roots.
"""

from . import impls as impls
from .wire import WireReferenceAnalysis as WireReferenceAnalysis
from .qubit import (
    QubitReferenceAnalysis as QubitReferenceAnalysis,
    is_allocator as is_allocator,
)
from .lattice import (
    CARRIED as CARRIED,
    UNTRACKED as UNTRACKED,
    Ref as Ref,
    Call as Call,
    Root as Root,
    Slot as Slot,
    Items as Items,
    Whole as Whole,
    Bottom as Bottom,
    Members as Members,
    Unknown as Unknown,
    Returned as Returned,
    Positions as Positions,
    Untracked as Untracked,
    root_in as root_in,
)
from .analysis import (
    ReferenceAnalysis as ReferenceAnalysis,
    as_positions as as_positions,
)
