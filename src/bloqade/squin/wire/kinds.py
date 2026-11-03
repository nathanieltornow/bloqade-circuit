"""Name the gates and noise channels of the wire dialect and state their shape.

A `Gate` or a `Noise` statement holds its kind as an attribute. The shape tables give
each kind its number of wires and parameters, which the statements check. A
measurement and a reset have one fixed shape each.
"""

import enum
from dataclasses import dataclass


class GateKind(enum.Enum):
    """Name each unitary gate of the wire dialect."""

    X = "x"
    Y = "y"
    Z = "z"
    H = "h"
    T = "t"
    S = "s"
    SQRT_X = "sqrt_x"
    SQRT_Y = "sqrt_y"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    CCZ = "ccz"
    SWAP = "swap"
    U3 = "u3"
    PHASED_XZ = "phased_x_z"
    NATIVE_CZ = "native_cz"
    NATIVE_R = "native_r"
    NATIVE_RZ = "native_rz"

    def __repr__(self) -> str:
        """Return the name, so that the IR prints `kind=cx`."""
        return self.value


class NoiseKind(enum.Enum):
    """Name each noise channel of the wire dialect."""

    SINGLE_QUBIT_PAULI_CHANNEL = "single_qubit_pauli_channel"
    TWO_QUBIT_PAULI_CHANNEL = "two_qubit_pauli_channel"
    DEPOLARIZE = "depolarize"
    DEPOLARIZE2 = "depolarize2"
    QUBIT_LOSS = "qubit_loss"
    QUBIT_LEAKAGE = "qubit_leakage"

    def __repr__(self) -> str:
        """Return the name, so that the IR prints `kind=depolarize`."""
        return self.value


@dataclass(frozen=True)
class Shape:
    """State the shape of one gate or noise channel."""

    wires: int
    """The number of qubits that the operation acts on."""
    params: int = 0
    """The number of classical parameters, such as angles or probabilities."""
    truncate: bool = False
    """Whether a broadcast of the operation stops at the shortest register."""


GATE_SHAPES: dict[GateKind, Shape] = {
    GateKind.X: Shape(1),
    GateKind.Y: Shape(1),
    GateKind.Z: Shape(1),
    GateKind.H: Shape(1),
    GateKind.T: Shape(1),
    GateKind.S: Shape(1),
    GateKind.SQRT_X: Shape(1),
    GateKind.SQRT_Y: Shape(1),
    GateKind.RX: Shape(1, 1),
    GateKind.RY: Shape(1, 1),
    GateKind.RZ: Shape(1, 1),
    GateKind.CX: Shape(2),
    GateKind.CY: Shape(2),
    GateKind.CZ: Shape(2),
    GateKind.CCZ: Shape(3),
    GateKind.SWAP: Shape(2),
    GateKind.U3: Shape(1, 3),
    GateKind.PHASED_XZ: Shape(1, 3),
    GateKind.NATIVE_CZ: Shape(2, truncate=True),
    GateKind.NATIVE_R: Shape(1, 2),
    GateKind.NATIVE_RZ: Shape(1, 1),
}
"""Give the shape of each gate. Angles are in turns."""

NOISE_SHAPES: dict[NoiseKind, Shape] = {
    NoiseKind.SINGLE_QUBIT_PAULI_CHANNEL: Shape(1, 3),
    NoiseKind.TWO_QUBIT_PAULI_CHANNEL: Shape(2, 1, truncate=True),
    NoiseKind.DEPOLARIZE: Shape(1, 1),
    NoiseKind.DEPOLARIZE2: Shape(2, 1, truncate=True),
    NoiseKind.QUBIT_LOSS: Shape(1, 1),
    NoiseKind.QUBIT_LEAKAGE: Shape(1, 2),
}
"""Give the shape of each noise channel."""

MEASURE_SHAPE = Shape(1)
"""Give the shape of a measurement."""

RESET_SHAPE = Shape(1)
"""Give the shape of a reset."""
