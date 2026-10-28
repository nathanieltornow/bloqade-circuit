"""Hold the rewrite rules and passes that run on wire squin."""

from bloqade.squin.rewrite.wire.gates import (
    CancelInverses as CancelInverses,
    MergeRotations as MergeRotations,
)
from bloqade.squin.rewrite.wire.reuse import (
    HoistFree as HoistFree,
    ReuseFreed as ReuseFreed,
)
from bloqade.squin.rewrite.wire.registers import (
    BorrowFirst as BorrowFirst,
    DropRoundTrip as DropRoundTrip,
    ForwardRelease as ForwardRelease,
)
