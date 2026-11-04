"""Convert Jeff through local classical, region and quantum rewrite rules."""

from dataclasses import field, dataclass

from kirin import ir, types
from kirin.passes import Fold, TypeInfer
from kirin.rewrite import Walk
from kirin.dialects import func
from kirin.validation import ValidationSuite
from kirin.dialects.scf.unroll import PickIfElse

from bloqade.jeff.dialects import stmts, kernel as jeff_kernel
from bloqade.squin.wire.group import kernel as wire_kernel
from bloqade.squin.wire.ownership import mark_owned
from bloqade.jeff.analysis.validation import (
    LinearityValidation,
    StructureValidation,
    JeffToSquinValidation,
)
from bloqade.squin.rewrite.wire.driver import CheckedFixpoint, reachable
from bloqade.squin.analysis.validation.wire import WireValidation

from .jeff2py import JeffToPy
from .rewrite import Retype, Quantum
from .jeff2scf import JeffToScf


def jeff_callee(node: ir.Statement) -> ir.Method | None:
    """Return the callee of a Jeff `Call`, or None for any other statement."""
    return node.callee if isinstance(node, stmts.Call) else None


@dataclass
class JeffToWire:
    """Convert supported Jeff programs to explicit wire values on private method copies."""

    methods: dict[ir.Method, ir.Method] = field(default_factory=dict, init=False)

    def emit(self, method: ir.Method) -> ir.Method:
        """Return verified wire code with consuming function parameters and explicit result order."""
        self.methods = {}
        if method.dialects is not jeff_kernel:
            raise ValueError("JeffToWire expects jeff.kernel dialects.")
        functions = reachable(method, callee_of=jeff_callee)
        ValidationSuite(
            [StructureValidation, LinearityValidation, JeffToSquinValidation]
        ).validate(method).raise_if_invalid()
        self.methods = {
            source: source.similar(jeff_kernel.union(wire_kernel))
            for source in functions
        }
        for copy in self.methods.values():
            mark_owned(copy)
            JeffToPy(copy.dialects).unsafe_run(copy)
            JeffToScf(copy.dialects).unsafe_run(copy)
            Walk(Retype()).rewrite(copy.code)
            Walk(Quantum(self.methods)).rewrite(copy.code)
            copy.dialects = wire_kernel
            Fold(copy.dialects).unsafe_run(copy)
            CheckedFixpoint(Walk(PickIfElse())).rewrite(copy.code)
            TypeInfer(copy.dialects)(copy)
            assert isinstance(copy.code, func.Function)
            returned = copy.callable_region.blocks[0].last_stmt
            assert isinstance(returned, func.Return)
            inputs = tuple(
                value.type for value in copy.callable_region.blocks[0].args[1:]
            )
            copy.code.signature = func.Signature(
                inputs=inputs, output=returned.value.type
            )
            copy.callable_region.blocks[0].args[0].type = types.MethodType[
                list(inputs), returned.value.type
            ]
            WireValidation().check(copy)
        return self.methods[method]
