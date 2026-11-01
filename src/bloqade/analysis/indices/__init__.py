"""Compute and compare the affine indices that name an item of a register."""

from . import impls as impls
from .affine import (
    Sym as Sym,
    Iter as Iter,
    Slice as Slice,
    Affine as Affine,
    Emitted as Emitted,
    lift as lift,
    loop_values as loop_values,
    python_slice as python_slice,
)
from .lattice import (
    Index as Index,
    KnownIndex as KnownIndex,
    UnknownIndex as UnknownIndex,
    UnreachableIndex as UnreachableIndex,
)
from .analysis import Indices as Indices, IndexAnalysis as IndexAnalysis
