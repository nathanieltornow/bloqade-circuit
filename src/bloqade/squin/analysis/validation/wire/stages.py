"""Check source representation through Kirin's validation and interpreter infrastructure."""

from dataclasses import field, replace, dataclass

from kirin import ir, interp
from kirin.lattice import EmptyLattice
from kirin.analysis import Forward
from kirin.dialects import scf, func
from kirin.validation import ValidationPass, ValidationSuite

from .validation import StructureValidation

KEY = "squin.source.validation"


@dataclass
class SourceAnalysis(Forward[EmptyLattice]):
    """Visit source functions and structured regions through operation method tables."""

    keys = (KEY,)
    lattice = EmptyLattice
    callees: set[ir.Method] = field(default_factory=set, init=False)

    def initialize(self):
        """Reset collected callees before checking a source method."""
        super().initialize()
        self.callees.clear()
        return self

    def method_self(self, method: ir.Method) -> EmptyLattice:
        """Return the empty abstract value for the function object."""
        return EmptyLattice.top()

    def error(self, node: ir.IRNode, message: str) -> None:
        """Record the failed source condition at its operation."""
        error = ir.ValidationError(node, message)
        error.source = None if node.source is None else replace(node.source)
        self.add_validation_error(node, error)

    def check_node(self, node: ir.Statement) -> None:
        """Require static calls and supported structured region owners."""
        if isinstance(node, func.Invoke):
            self.callees.add(node.callee)
        if isinstance(node, func.Call):
            self.error(node, "Wire emission requires a static call target.")
        if node.regions and not isinstance(node, (func.Function, scf.For, scf.IfElse)):
            self.error(node, "Wire emission requires function, loop or branch regions.")

    def frame_eval(self, frame, node):
        """Check each operation before its method table runs."""
        self.check_node(node)
        return super().frame_eval(frame, node)

    def frame_call(self, frame, node, *args, **kwargs):
        """Check the function operation before Kirin enters its body."""
        self.check_node(node)
        return super().frame_call(frame, node, *args, **kwargs)

    def frame_call_region(self, frame, node, region, *args):
        """Check region parameters and terminators before interpreting a source block."""
        if len(region.blocks) != 1:
            self.error(
                region, "Wire emission requires single-block structured regions."
            )
            return None
        block = region.blocks[0]
        if len(block.args) != len(args):
            self.error(node, "Region arguments must match the supplied inputs.")
            return None
        end = block.last_stmt
        if isinstance(node, (scf.For, scf.IfElse)):
            if not isinstance(end, scf.Yield):
                self.error(node, "A structured body must end with a yield.")
                return None
            if len(end.values) != len(node.results):
                self.error(
                    end, "Each yield must match its parent operation's result count."
                )
                return None
        return super().frame_call_region(frame, node, region, *args)

    def eval_fallback(self, frame, node):
        """Treat ordinary source operations and static calls as opaque to structural validation."""
        return tuple(EmptyLattice.top() for _ in node.results)


@scf.dialect.register(key=KEY)
class Regions(interp.MethodTable):
    """Check each structured body once with a separate Kirin frame."""

    @interp.impl(scf.For)
    @interp.impl(scf.IfElse)
    def structured(self, check: SourceAnalysis, frame, stmt):
        """Pass abstract region arguments to Kirin's region interpreter."""
        for region in stmt.regions:
            count = len(stmt.initializers) + 1 if isinstance(stmt, scf.For) else 1
            with check.new_frame(stmt, has_parent_access=True) as inner:
                check.frame_call_region(
                    inner, stmt, region, *(EmptyLattice.top() for _ in range(count))
                )
        return tuple(EmptyLattice.top() for _ in stmt.results)

    @interp.impl(scf.Yield)
    def yield_(self, check: SourceAnalysis, frame, stmt):
        """End a region with an abstract value for each yielded operand."""
        return interp.YieldValue(tuple(EmptyLattice.top() for _ in stmt.values))


@func.dialect.register(key=KEY)
class Functions(interp.MethodTable):
    """End source function validation at its return statement."""

    @interp.impl(func.Return)
    def return_(self, check: SourceAnalysis, frame, stmt):
        """Return an empty abstract value to the function driver."""
        return interp.ReturnValue(EmptyLattice.top())


class SourceValidation(ValidationPass[SourceAnalysis]):
    """Check the structured source representation before wire emission."""

    def name(self) -> str:
        """Return the diagnostic name for source representation checks."""
        return "squin source"

    def run(self, method: ir.Method):
        """Return the source analysis and its validation errors."""
        check = SourceAnalysis(method.dialects)
        check.run(method)
        return check, check.get_validation_errors()


def check_stage(method: ir.Method, stage: type[ValidationPass]):
    """Check structure, types and the declared stage contract before emission."""
    result = ValidationSuite([StructureValidation, stage], fail_fast=True).validate(
        method
    )
    result.raise_if_invalid()
    return result.frames[stage().name()]
