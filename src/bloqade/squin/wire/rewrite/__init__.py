"""Hold the rewrite rules and passes that run on wire squin."""

from .gates import CancelInverses as CancelInverses, MergeRotations as MergeRotations
from .reuse import HoistFree as HoistFree, ReuseFreed as ReuseFreed
from .passes import Simplify as Simplify, ReuseQubits as ReuseQubits
from .registers import (
    BorrowFirst as BorrowFirst,
    DropRoundTrip as DropRoundTrip,
    ForwardRelease as ForwardRelease,
)
