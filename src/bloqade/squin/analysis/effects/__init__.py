"""Record what each statement of a squin kernel does to qubits.

`QubitEffects` records the roots that each statement allocates or uses, with one
method table for each dialect. The wire converter reads the effect of a loop or a
branch to decide which wires the body carries.
"""

from . import impls as impls
from .analysis import (
    KEY as KEY,
    Effect as Effect,
    QubitEffects as QubitEffects,
    holds_qubits as holds_qubits,
    may_hide_qubits as may_hide_qubits,
)
