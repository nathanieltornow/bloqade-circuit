"""Define `kernel`, the dialect group of wire squin methods.

The group holds every squin dialect and the wire dialect. A wire squin method
can therefore keep its classical statements unchanged.
"""

from bloqade.squin.groups import kernel as squin_kernel

from ._dialect import dialect

kernel = squin_kernel.union([dialect])
