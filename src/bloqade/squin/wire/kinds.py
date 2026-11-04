"""Name the gates and noise channels of the wire dialect with their operand counts.

A `Gate` or a `Noise` statement holds its kind as an attribute. Each kind states how
many qubits and classical parameters it takes, which the statements check.
"""

import enum


class GateKind(enum.Enum):
    """Name each unitary gate of the wire dialect.

    A member holds its printed name, its number of qubits and its number of angles in
    turns. For example, `CX` takes 2 qubits, and `RZ` takes 1 qubit and 1 angle. If
    `truncates` is True, the register form of the gate stops at the shortest register.
    The register form of every other gate needs registers of equal length.
    """

    X = ("x", 1)
    Y = ("y", 1)
    Z = ("z", 1)
    H = ("h", 1)
    T = ("t", 1)
    S = ("s", 1)
    SQRT_X = ("sqrt_x", 1)
    SQRT_Y = ("sqrt_y", 1)
    RX = ("rx", 1, 1)
    RY = ("ry", 1, 1)
    RZ = ("rz", 1, 1)
    CX = ("cx", 2)
    CY = ("cy", 2)
    CZ = ("cz", 2)
    CCZ = ("ccz", 3)
    SWAP = ("swap", 2)
    U3 = ("u3", 1, 3)
    PHASED_XZ = ("phased_x_z", 1, 3)
    NATIVE_CZ = ("native_cz", 2, 0, True)
    NATIVE_R = ("native_r", 1, 2)
    NATIVE_RZ = ("native_rz", 1, 1)

    def __init__(
        self, label: str, wires: int, params: int = 0, truncates: bool = False
    ) -> None:
        """Record the name and the operand counts of the gate."""
        self.label = label
        self.wires = wires
        self.params = params
        self.truncates = truncates

    def __repr__(self) -> str:
        """Return the name, so that the IR prints `kind=cx`."""
        return self.label


class NoiseKind(enum.Enum):
    """Name each noise channel of the wire dialect.

    A member holds its printed name, its number of qubits and its number of
    probabilities. If `truncates` is True, the register form of the channel stops at
    the shortest register.
    """

    SINGLE_QUBIT_PAULI_CHANNEL = ("single_qubit_pauli_channel", 1, 3)
    TWO_QUBIT_PAULI_CHANNEL = ("two_qubit_pauli_channel", 2, 1, True)
    DEPOLARIZE = ("depolarize", 1, 1)
    DEPOLARIZE2 = ("depolarize2", 2, 1, True)
    QUBIT_LOSS = ("qubit_loss", 1, 1)
    QUBIT_LEAKAGE = ("qubit_leakage", 1, 2)

    def __init__(
        self, label: str, wires: int, params: int = 0, truncates: bool = False
    ) -> None:
        """Record the name and the operand counts of the channel."""
        self.label = label
        self.wires = wires
        self.params = params
        self.truncates = truncates

    def __repr__(self) -> str:
        """Return the name, so that the IR prints `kind=depolarize`."""
        return self.label
