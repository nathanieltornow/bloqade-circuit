"""Define which statements a complete wire conversion can keep.

The target accepts wire operations and classical operations from the source
dialect group. Every operand, result and block argument must exclude qubit
references. Partial conversion reports the same violations and keeps them.
"""

from dataclasses import field, dataclass

from kirin import ir
from kirin.rewrite import Walk
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade import qubit
from bloqade.squin import gate, noise
from bloqade.squin.wire import stmts
from bloqade.native.dialects import gate as native_gate
from bloqade.squin.wire.group import kernel, reference_kernel
from bloqade.squin.analysis.reference import holds_qubits

REFERENCE_STATEMENTS = (
    stmts.Unwrap,
    stmts.Wrap,
    stmts.UnwrapRegister,
    stmts.WrapRegister,
    stmts.Reference,
    stmts.ReferenceRegister,
)

CLASSICAL_DIALECTS = reference_kernel.data - {
    qubit.dialect,
    gate.dialect,
    noise.dialect,
    native_gate.dialect,
}


@dataclass
class ConversionTarget(RewriteRule):
    """Collect illegal statements with Kirin's walker without changing the method."""

    remaining: list[str] = field(default_factory=list)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Record operations that keep references or lack a supported wire form."""
        legal_dialect = (
            node.dialect in CLASSICAL_DIALECTS
            or node.dialect is stmts.dialect
            or isinstance(
                node, (qubit.stmts.IsZero, qubit.stmts.IsOne, qubit.stmts.IsLost)
            )
        )
        legal_values = not any(
            holds_qubits(value.type) for value in (*node.args, *node.results)
        )
        legal_call = not isinstance(node, func.Call) and (
            not isinstance(node, func.Invoke) or node.callee.dialects is kernel
        )
        legal_signature = not isinstance(node, func.Function) or not any(
            holds_qubits(kind)
            for kind in (*node.signature.inputs, node.signature.output)
        )
        if not (
            legal_dialect and legal_values and legal_call and legal_signature
        ) or isinstance(node, REFERENCE_STATEMENTS):
            self.remaining.append(node.name)
        return RewriteResult()

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        """Record block parameters that still hold qubit references."""
        if any(holds_qubits(arg.type) for arg in node.args):
            self.remaining.append("block argument")
        return RewriteResult()

    @classmethod
    def check(cls, method: ir.Method) -> list[str]:
        """Return every conversion violation in the method and its nested regions."""
        target = cls()
        Walk(target).rewrite(method.code)
        return target.remaining
