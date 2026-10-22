"""This package computes the lifetime of each qubit root of a squin kernel.

`QubitEffects` records what each statement does to qubits, with one method table
for each dialect. `block_lifetimes` reads those effects for one block.
"""

from . import impls as impls
from .analysis import (
    KEY as KEY,
    ESCAPES as ESCAPES,
    Effect as Effect,
    Lifetime as Lifetime,
    QubitEffects as QubitEffects,
    BlockLifetimes as BlockLifetimes,
    value_of as value_of,
    holds_qubits as holds_qubits,
    block_lifetimes as block_lifetimes,
    may_hide_qubits as may_hide_qubits,
)
