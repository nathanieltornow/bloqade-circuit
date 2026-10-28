"""Infer types while preserving literal tuple indices inside structured regions."""

from kirin import ir
from kirin.passes import TypeInfer
from kirin.rewrite import Walk
from kirin.analysis import const
from kirin.rewrite.abc import RewriteRule, RewriteResult
from kirin.passes.hint_const import HintConst

from bloqade.constants import constant_int


class LiteralIndices(RewriteRule):
    """Restore exact integer hints after constant propagation widens loop values."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Keep each integer literal available to tuple field type inference."""
        for value in node.results:
            if (integer := constant_int(value)) is not None:
                value.hints["const"] = const.Value(integer)
        return RewriteResult()


class WireHints(HintConst):
    """Run Kirin's constant analysis and retain every syntactic integer literal."""

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        """Restore literal hints before type inference checks quantum tuple fields."""
        result = super().unsafe_run(mt)
        return Walk(LiteralIndices()).rewrite(mt.code).join(result)


def infer_wire_types(method: ir.Method) -> None:
    """Infer copied method types with exact tuple selectors on the pinned Kirin version."""
    inference = TypeInfer(method.dialects)
    inference.hint_const = WireHints(method.dialects)
    inference(method)
