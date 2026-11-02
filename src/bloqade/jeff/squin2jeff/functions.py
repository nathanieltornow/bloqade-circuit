"""Replace wire functions, calls and returns with their Jeff forms."""

from dataclasses import dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.dialects import py, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.jeff.types import QuregType, is_linear
from bloqade.jeff.dialects import stmts
from bloqade.squin.wire.ownership import owns_inputs
from bloqade.jeff.dialects.stmts.call import declared_outputs

from .kinds import JeffEmissionError, put, fitted, replace, jeff_kind


@dataclass
class Functions(RewriteRule):
    """Retype each function and replace each call and return with Jeff.

    A wire function returns one value, which is a tuple when it has several
    outputs. A Jeff function returns each output as a separate value.
    """

    methods: Mapping[ir.Method, ir.Method]
    """The Jeff copy of each wire method that a call can name.

    The pipeline rewrites each callee before its callers, so a call reads the Jeff
    signature of its callee.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is a function, a call or a return."""
        if isinstance(node, func.Function):
            retype(node)
        elif isinstance(node, func.Invoke):
            self.call(node)
        elif isinstance(node, func.Return):
            returned(node)
        else:
            return RewriteResult()
        return RewriteResult(has_done_something=True)

    def call(self, node: func.Invoke) -> None:
        """Replace `node` with a Jeff call of the Jeff copy of its callee.

        A tuple result becomes a `py.tuple.New` of the call results, and a later
        rewrite reads its fields. The arguments already hold Jeff values. A
        borrowing callee returns each register argument as the same register, so
        that result keeps the static length of the argument.
        """
        callee = self.methods[node.callee]
        outputs = declared_outputs(callee.return_type)
        made = put(stmts.Call(callee, tuple(node.inputs), outputs), node)
        if not owns_inputs(node.callee):
            borrowed = [value for value in node.inputs if is_linear(value.type)]
            suffix = made.results[len(made.results) - len(borrowed) :]
            for argument, result in zip(borrowed, suffix, strict=True):
                if argument.type.is_subseteq(QuregType):
                    result.type = argument.type
        if node.result.type.is_subseteq(types.NoneType):
            replace(node, put(func.ConstantNone(), node).result)
        elif node.result.type.is_subseteq(types.Tuple):
            replace(node, put(py.tuple.New(tuple(made.results)), node).result)
        else:
            replace(node, made.results[0])


def retype(node: func.Function) -> None:
    """Give the function `node` the Jeff types of its parameters and outputs."""
    inputs = tuple(jeff_kind(kind) for kind in node.signature.inputs)
    outputs = tuple(jeff_kind(kind) for kind in declared_outputs(node.signature.output))
    output = (
        types.NoneType
        if not outputs
        else outputs[0] if len(outputs) == 1 else types.Generic(tuple, *outputs)
    )
    block = node.body.blocks[0]
    for argument, kind in zip(block.args[1:], inputs, strict=True):
        argument.type = kind
    block.args[0].type = types.MethodType[list(inputs), output]
    node.signature = func.Signature(inputs=inputs, output=output)


def returned(node: func.Return) -> None:
    """Replace `node` with a Jeff return of each output in the declared order."""
    function = node.parent_stmt
    if not isinstance(function, func.Function):
        raise JeffEmissionError(node, "Jeff requires returns at function scope.")
    owner = node.value.owner
    if isinstance(owner, func.ConstantNone):
        values: tuple[ir.SSAValue, ...] = ()
    elif isinstance(owner, py.tuple.New):
        values = tuple(owner.args)
    else:
        values = (node.value,)
    kinds = declared_outputs(function.signature.output)
    if len(values) != len(kinds):
        raise JeffEmissionError(
            node, "The return values must match the declared function outputs."
        )
    if any(value.type.is_subseteq(types.Tuple) for value in values):
        raise JeffEmissionError(
            node, "Jeff returns require flat scalar or array values."
        )
    outputs = (fitted(v, k, node) for v, k in zip(values, kinds, strict=True))
    put(stmts.Return(*outputs), node)
    node.delete()
