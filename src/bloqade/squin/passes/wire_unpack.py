"""Turn the constant-size registers of a wire method into separate wires."""

from dataclasses import dataclass

from kirin import ir
from kirin.passes import Pass
from kirin.rewrite import Walk, Fixpoint, DeadCodeElimination
from kirin.rewrite.abc import RewriteResult

from bloqade.squin.rewrite.wire.unpack import UnpackRegister, ExpandBroadcast


@dataclass
class UnpackRegisters(Pass):
    """Expand static broadcasts, then unpack each register whose indices are constant.

    The pass works on one method and leaves its callees alone. A register that a
    loop, a branch or a call takes stays a register, but its static broadcasts still
    become scalar operations.
    """

    name = "unpack_registers"

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        """Run the expansion, then the unpacking, then dead-code removal."""
        result = Walk(ExpandBroadcast()).rewrite(mt.code)
        result = Walk(UnpackRegister()).rewrite(mt.code).join(result)
        dead = Fixpoint(Walk(DeadCodeElimination())).rewrite(mt.code)
        return dead.join(result)
