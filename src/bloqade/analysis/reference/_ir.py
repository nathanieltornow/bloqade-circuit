"""Read constants, signatures, and containment relationships from Kirin IR."""

from typing import Any
from collections.abc import Iterator

from kirin import ir, types
from kirin.dialects import func, ilist


def _payload(value: ir.SSAValue) -> object:
    owner = value.owner
    if not isinstance(owner, ir.Statement) or not owner.has_trait(ir.ConstantLike):
        return None
    attribute = owner.attributes.get("value")
    if not isinstance(attribute, ir.Data):
        return None
    data = attribute.unwrap()
    return data.data if isinstance(data, ilist.IList) else data


def const_int(value: ir.SSAValue) -> int | None:
    """Return a constant integer, or None for other values, including booleans."""
    data = _payload(value)
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    return None


def function_of(method: ir.Method[..., Any]) -> func.Function:
    """Return the function wrapped by a method.

    Raises:
        TypeError: If the method does not wrap a function.
    """
    code = method.code
    if not isinstance(code, func.Function):
        raise TypeError(f"'{method.sym_name}' is not a function: {code.name}")
    return code


def declared_outputs(signature: func.Signature) -> tuple[types.TypeAttribute, ...]:
    """Return the declared output types, unpacking tuples and omitting None."""
    output = signature.output
    if output.is_subseteq(types.NoneType):
        return ()
    if isinstance(output, types.Generic) and output.is_subseteq(types.Tuple):
        return tuple(output.vars)
    return (output,)


def ancestors(node: ir.Statement) -> Iterator[tuple[ir.Block, ir.Region]]:
    """Yield enclosing (block, region) pairs, starting with the nearest."""
    current = node
    while current is not None:
        block = current.parent
        if block is None or block.parent is None:
            return
        yield block, block.parent
        current = block.parent.parent_node


def statement_in(node: ir.Statement, container: ir.Region | ir.Block) -> bool:
    """Check whether a statement lies inside a block or region at any depth."""
    for block, region in ancestors(node):
        if block is container or region is container:
            return True
    return False


def value_in(value: ir.SSAValue, container: ir.Region | ir.Block) -> bool:
    """Check whether a value is defined inside a block or region at any depth."""
    if isinstance(value, ir.BlockArgument):
        block = value.block
        return (
            block is container
            or block.parent is container
            or (
                block.parent is not None
                and block.parent.parent_node is not None
                and statement_in(block.parent.parent_node, container)
            )
        )
    owner = value.owner
    return isinstance(owner, ir.Statement) and statement_in(owner, container)
