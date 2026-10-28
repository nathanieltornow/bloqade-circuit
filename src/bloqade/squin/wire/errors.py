"""Define `WireConversionError`, which `SquinToWire` raises for a kernel it can't convert."""


class WireConversionError(Exception):
    """Report a construct of a squin kernel that has no wire form.

    The message names the construct, such as two places of one gate that may name
    one qubit, or a statement whose qubits no place names.
    """
