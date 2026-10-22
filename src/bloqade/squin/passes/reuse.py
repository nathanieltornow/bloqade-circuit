"""This module holds `ReuseQubits`, a pass that reuses dead qubits for later allocations.

A qubit is dead after the last statement that touches it. If a later
`squin.qubit.new()` in the same block needs a qubit, the pass resets the dead
qubit and hands it over. So the kernel allocates fewer qubits, and the peak count
of live qubits can drop.

For example, two ancillas that the kernel measures one after the other share one
qubit:

    a = squin.qubit.new()          a = squin.qubit.new()
    squin.h(a)                     squin.h(a)
    squin.measure(a)               squin.measure(a)
    b = squin.qubit.new()          squin.reset(a)
    squin.x(b)                     squin.x(a)

The pass reads the places of `QubitReferenceAnalysis`. It only reuses single
qubits whose lifetimes it can bound in one block. See
`bloqade.squin.analysis.lifetime` for the rules.
"""

from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir
from kirin.passes import Pass
from kirin.rewrite import Walk
from kirin.dialects import func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade import squin
from bloqade.analysis.reference import Ref
from bloqade.squin.analysis.lifetime import value_of, block_lifetimes
from bloqade.squin.analysis.reference import QubitReferenceAnalysis


@dataclass
class ReuseInBlock(RewriteRule):
    """Hand each dead qubit of a block to the next allocation of that block.

    `refs` holds the place of each value, from `QubitReferenceAnalysis`. `reused`
    counts the allocations that the rule removed.
    """

    refs: Mapping[ir.SSAValue, Ref]
    reused: int = field(default=0, init=False)

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        """Replace each allocation of `node` that can take a dead qubit."""
        lives = block_lifetimes(node, self.refs)
        stmts = list(node.stmts)
        # `defined` maps each result of the block to the index of its statement.
        defined: dict[ir.SSAValue, int] = {
            value: index for index, stmt in enumerate(stmts) for value in stmt.results
        }
        # `busy` maps each qubit value to the index of its last use so far.
        busy: dict[ir.SSAValue, float] = {}
        for root, life in lives.lifetimes.items():
            value = value_of(root, node)
            if value is not None and root not in lives.allocations.values():
                busy[value] = life.end
        changed = False
        for index, stmt in enumerate(stmts):
            root = lives.allocations.get(index)
            if root is None:
                continue
            end = lives.lifetimes[root].end
            donor = next(
                (
                    value
                    for value, last in busy.items()
                    if last < index and defined[value] < index
                ),
                None,
            )
            if donor is None:
                if (own := value_of(root, node)) is not None:
                    busy[own] = end
                continue
            reset = func.Invoke((donor,), callee=squin.reset)
            reset.result.type = squin.reset.return_type
            reset.insert_before(stmt)
            stmt.results[0].replace_by(donor)
            stmt.delete()
            busy[donor] = end
            self.reused += 1
            changed = True
        return RewriteResult(has_done_something=changed)


@dataclass
class ReuseQubits(Pass):
    """Reset dead qubits and reuse them for later single-qubit allocations.

    After a run, `reused` holds the count of allocations that the pass removed.
    """

    name = "reuse_qubits"
    reused: int = field(default=0, init=False)

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        """Reuse the dead qubits of each block of `mt` in place."""
        frame, _ = QubitReferenceAnalysis(mt.dialects).run(mt)
        rule = ReuseInBlock(frame.entries)
        result = Walk(rule).rewrite(mt.code)
        self.reused = rule.reused
        return result
