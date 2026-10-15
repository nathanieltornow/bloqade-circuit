"""This package holds the analysis that states which root each value refers to.

`ReferenceAnalysis` holds the rules for kirin's own dialects. A subclass adds the
roots of one kind of tracked state, such as squin qubits or jeff wires.
"""

from . import impls as impls
from .index import (
    Iter as Iter,
    Slice as Slice,
    Affine as Affine,
    same as same,
)
from .lattice import (
    CARRIED as CARRIED,
    UNTRACKED as UNTRACKED,
    Ref as Ref,
    Item as Item,
    Part as Part,
    Root as Root,
    Slot as Slot,
    Step as Step,
    Items as Items,
    Place as Place,
    Whole as Whole,
    Bottom as Bottom,
    Closure as Closure,
    Members as Members,
    Unknown as Unknown,
    Register as Register,
    Returned as Returned,
    Positions as Positions,
    Untracked as Untracked,
    child as child,
    place as place,
    roots as roots,
    positions as positions,
)
from .analysis import (
    KEY as KEY,
    OUT_OF_SCOPE as OUT_OF_SCOPE,
    RegisterLength as RegisterLength,
    ReferenceAnalysis as ReferenceAnalysis,
    items_of as items_of,
    origin_of as origin_of,
    argument_of as argument_of,
    place_length as place_length,
)
from .integers import IndexAnalysis as IndexAnalysis
