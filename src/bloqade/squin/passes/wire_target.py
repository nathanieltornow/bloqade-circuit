"""Define which statements a complete wire conversion can keep.

The target accepts wire operations and classical operations from the source
dialect group. Every operand, result and block argument must exclude qubit
references. Complete conversion rejects every remaining violation.
"""

from dataclasses import field, dataclass

from kirin import ir
from kirin.dialects import func

from bloqade import qubit
from bloqade.squin import gate, noise
from bloqade.types import holds_qubits
from bloqade.squin.wire import stmts
from bloqade.native.dialects import gate as native_gate
from bloqade.squin.wire.group import kernel, reference_kernel
from bloqade.squin.analysis.validation.wire.stages import SourceAnalysis

CLASSICAL_DIALECTS = reference_kernel.data - {
    qubit.dialect,
    gate.dialect,
    noise.dialect,
    native_gate.dialect,
}


@dataclass
class ConversionTarget(SourceAnalysis):
    """Collect illegal statements through Kirin's validation analysis frames."""

    remaining: list[str] = field(default_factory=list)

    def check_node(self, node: ir.Statement) -> None:
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
        if not (legal_dialect and legal_values and legal_call and legal_signature):
            self.remaining.append(node.name)

    def frame_call_region(self, frame, node, region, *args):
        """Record reference parameters before Kirin checks the region body."""
        if any(holds_qubits(arg.type) for block in region.blocks for arg in block.args):
            self.remaining.append("block argument")
        return super().frame_call_region(frame, node, region, *args)

    @classmethod
    def check(cls, method: ir.Method) -> list[str]:
        """Return every conversion violation in the method and its nested regions."""
        target = cls(method.dialects)
        target.run(method)
        return target.remaining
