"""Run squin kernels with a kirin interpreter that keeps one value per qubit."""

from . import impls as impls
from .analysis import (
    KEY as KEY,
    OPAQUE as OPAQUE,
    MAX_DEPTH as MAX_DEPTH,
    Stop as Stop,
    Qubit as Qubit,
    Opaque as Opaque,
    PerQubit as PerQubit,
    Register as Register,
    QubitDepth as QubitDepth,
    MeasurementCount as MeasurementCount,
    PerQubitInterpreter as PerQubitInterpreter,
    circuit_depth as circuit_depth,
)
