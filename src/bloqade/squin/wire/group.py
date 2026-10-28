"""Define `kernel`, the dialect group of wire squin methods.

The group holds the squin, native gate and wire dialects. A wire squin method
can therefore keep its classical statements unchanged.
"""

from bloqade.squin.groups import kernel as squin_kernel
from bloqade.native.dialects import gate as native_gate

from ._dialect import dialect

reference_kernel = squin_kernel.union([native_gate])
kernel = reference_kernel.union([dialect])
