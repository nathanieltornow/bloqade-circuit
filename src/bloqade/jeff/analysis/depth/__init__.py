"""This module holds the analysis that computes the exact circuit depth of a jeff program.

The depth of a wire is the cost of the longest chain of operations that ends at the
wire. A gate takes the max depth over its input wires and adds its cost. Two gates
on disjoint wires therefore share one layer. The circuit depth is the max depth over
every wire that the program consumes or returns.

Each depth rule uses only `max` and `+`, so each depth is a max-plus `Form` over the
depths that enter a region. See `bloqade.jeff.analysis.depth.maxplus`. The analysis runs a
`For` body once with one variable per carried wire. The yielded forms give a matrix
`A` and a vector `b` with `next = A ⊗ state ⊕ b`. The analysis then raises the map to
the trip count `n` by repeated squaring. The result is exact for every `n` and costs
`O(k^3 log n)` for `k` carried wires.

A multiple of the depth of one iteration overestimates when the longest chain moves
between wires. For example, a body that applies a gate to `w0` and yields
`(w1, w0)` gives each wire a gate in every second iteration only.

If the analysis cannot compute an exact depth, it reports `Unknown` with a reason.
A `While` loop, a `For` loop with non-constant bounds, a register of non-constant
length and a register index that is not a constant each give `Unknown`.
"""

from . import impls as impls
from .lattice import (
    Depth as Depth,
    Bottom as Bottom,
    Results as Results,
    Unknown as Unknown,
    RegDepth as RegDepth,
    Classical as Classical,
    WireDepth as WireDepth,
)
from .maxplus import Form as Form
from .analysis import (
    DEPTH_KEY as DEPTH_KEY,
    DepthAnalysis as DepthAnalysis,
    unit_cost as unit_cost,
    circuit_depth as circuit_depth,
)
