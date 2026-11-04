"""Mark the functions that consume their quantum inputs.

Jeff import marks each function that it converts. Such a function takes its qubits
and returns every quantum result in its own order. Every other wire function is an
in-out function, which returns each quantum parameter in its position.
"""

from kirin import ir

KEY = "squin.wire.owned"


def owns_inputs(function: ir.Statement) -> bool:
    """Check whether the function `function` consumes its quantum inputs."""
    return function.attributes.get(KEY) == ir.PyAttr(True)


def mark_owned(method: ir.Method) -> None:
    """Mark `method` as a function that consumes its quantum inputs."""
    method.code.attributes[KEY] = ir.PyAttr(True)
