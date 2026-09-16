"""Track qubit references with forward dataflow analysis.

Transfer rules use the ``qubit.reference`` registry key. Each callee runs
with its caller's argument references. Returned allocations receive roots
at the call site. Recursive calls produce unknown quantum results.

Unknown references include a reason for the loss of information.
"""

from typing import Any
from dataclasses import field, dataclass

from kirin import ir, types
from kirin.dialects import py, func
from kirin.analysis.forward import Forward, ForwardFrame

from .lattice import (
    CLASSICAL,
    Ref,
    Root,
    Slot,
    Whole,
    Members,
    Unknown,
    Returned,
    Positions,
    roots,
    root_in,
    static_length,
    is_quantum_type,
)

# -- a callee's return, in the caller's terms ----------------------------------


def as_positions(ref: Ref, count: int) -> tuple[Ref, ...]:
    """Unpack a return reference into the declared number of output positions."""
    if count == 0:
        return ()
    if isinstance(ref, Unknown):
        return tuple(ref for _ in range(count))
    if isinstance(ref, Positions) and len(ref.refs) == count:
        return ref.refs
    if count == 1 and not isinstance(ref, Positions):
        return (ref,)
    return tuple(
        Unknown("a return that does not match the declared output positions")
        for _ in range(count)
    )


def rebased(
    returned: tuple[Ref, ...], call: func.Invoke, body: ir.Region
) -> tuple[Ref, ...]:
    """Replace callee-local allocation roots with call-site roots.

    Preserve references to caller-owned qubits. Mark duplicate allocations,
    locally allocated register elements, and lists of local allocations unknown.
    """
    seen: set[Root] = set()
    resolved = []
    for position, ref in enumerate(returned):
        match ref:
            case Whole(root) if root_in(root, body):
                if root in seen:
                    resolved.append(Unknown("an allocation returned at two positions"))
                else:
                    seen.add(root)
                    resolved.append(
                        Whole(Returned(call, position, static_length(root)))
                    )
            case Slot(root, _) if root_in(root, body):
                resolved.append(Unknown("an element of a register the call allocates"))
            case Members() if any(root_in(root, body) for root in roots(ref)):
                resolved.append(Unknown("a list holding an allocation of the call"))
            case _:
                resolved.append(ref)
    return tuple(resolved)


# -- the analysis --------------------------------------------------------------


References = dict[ir.SSAValue, Ref]
"""Map SSA values to their reference-analysis results."""


@dataclass
class ReferenceAnalysis(Forward[Ref]):
    """Track references and cache callee results by function and argument references."""

    keys = ("qubit.reference",)
    lattice = Ref

    active: set[ir.Statement] = field(default_factory=set, init=False)
    memo: dict[tuple[ir.Statement, tuple[Ref, ...]], tuple[Ref, ...]] = field(
        default_factory=dict, init=False
    )

    def method_self(self, method: ir.Method) -> Ref:
        """Classify the method object as classical."""
        return CLASSICAL

    def eval_fallback(
        self, frame: ForwardFrame[Ref], node: ir.Statement
    ) -> tuple[Ref, ...]:
        """Mark unsupported quantum results unknown and other results classical."""
        if isinstance(node, py.binop.Add):
            reason = "a concatenation of registers"
        else:
            reason = f"a value computed by '{node.name}'"
        return unknown_results(node, reason)


def _parameter(arg: ir.BlockArgument) -> Ref:
    """Classify a parameter by type.

    Use quantum parameters as roots. Mark tuples with quantum members unknown;
    classify other parameters as classical.
    """
    if is_quantum_type(arg.type):
        return Whole(arg)
    if (
        isinstance(arg.type, types.Generic)
        and arg.type.is_subseteq(types.Tuple)
        and any(is_quantum_type(member) for member in arg.type.vars)
    ):
        return Unknown("a member of a tuple parameter")
    return CLASSICAL


def analyze(kernel: ir.Method[..., Any]) -> References:
    """Map kernel SSA values to references, using quantum parameters as roots."""
    analysis = ReferenceAnalysis(kernel.dialects)
    params = [_parameter(arg) for arg in kernel.callable_region.blocks[0].args[1:]]
    frame, _ = analysis.run(kernel, *params)
    return dict(frame.entries)


def unknown_results(stmt: ir.Statement, reason: str) -> tuple[Ref, ...]:
    """Mark quantum results unknown with the given reason and other results classical."""
    return tuple(
        Unknown(reason) if is_quantum_type(r.type) else CLASSICAL for r in stmt.results
    )
