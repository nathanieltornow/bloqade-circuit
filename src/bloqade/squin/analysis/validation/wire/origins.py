"""Check wire positions through Kirin frames and operation method tables."""

from dataclasses import field, dataclass

from kirin import ir, interp
from kirin.analysis import Forward
from kirin.dialects import py, scf, func
from kirin.analysis.forward import ForwardFrame

from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Affine, IndexAnalysis
from bloqade.squin.wire.types import is_linear
from bloqade.squin.wire.traits import Threads
from bloqade.squin.wire.ownership import KEY as OWNED

from .identities import Handle, Origin

KEY = "squin.wire.positions"


@dataclass
class PositionAnalysis(Forward[Origin]):
    """Follow allocation origins in Kirin frames and report positional contract failures."""

    keys = (KEY,)
    lattice = Origin
    functions: list[ir.Statement] = field(default_factory=list, init=False)
    indices: IndexAnalysis = field(default_factory=IndexAnalysis, init=False)

    def initialize(self):
        """Reset function context and integer facts before each validation run."""
        super().initialize()
        self.functions = []
        self.indices = IndexAnalysis()
        return self

    def method_self(self, method: ir.Method) -> Origin:
        """Give the function object an unknown quantum origin."""
        return Origin.top()

    def error(self, node: ir.IRNode, message: str) -> None:
        """Record a validation error at the operation that violates its contract."""
        self.add_validation_error(node, ir.ValidationError(node, message))

    def expect(
        self, node: ir.Statement, actual: Origin, expected: Origin, message: str
    ) -> None:
        """Require both values to identify the same allocation and element."""
        if not isinstance(actual.data, Handle) or actual != expected:
            self.error(node, message)

    @property
    def borrowing(self) -> bool:
        """Identify functions whose quantum parameters require positional return continuity."""
        return self.functions[-1].attributes.get(OWNED) != ir.PyAttr(True)

    def frame_call(
        self,
        frame: ForwardFrame[Origin],
        node: ir.Statement,
        *args: Origin,
        **kwargs: Origin,
    ) -> Origin:
        """Check borrowed returns after Kirin interprets the function body."""
        if node in self.functions:
            self.error(node, "Wire validation does not support recursive calls.")
            return Origin.top()
        self.functions.append(node)
        try:
            result = super().frame_call(frame, node, *args, **kwargs)
            if self.borrowing:
                region = node.get_present_trait(
                    ir.CallableStmtInterface
                ).get_callable_region(node)
                parameters = [
                    arg for arg in region.blocks[0].args[1:] if is_linear(arg.type)
                ]
                outputs = result.data if isinstance(result.data, tuple) else (result,)
                end = region.blocks[0].last_stmt
                assert end is not None
                if len(outputs) < len(parameters):
                    self.error(
                        end, "The return suffix must include every borrowed parameter."
                    )
                elif parameters:
                    for i, (actual, parameter) in enumerate(
                        zip(outputs[-len(parameters) :], parameters)
                    ):
                        self.expect(
                            end,
                            actual,
                            frame.get(parameter),
                            f"return suffix position {i} must continue borrowed parameter {i}",
                        )
            return result
        finally:
            self.functions.pop()

    def eval_fallback(self, frame: ForwardFrame[Origin], node: ir.Statement):
        """Follow declared wire continuations and assign unknown origins to other results."""
        outputs = {result: Origin.top() for result in node.results}
        if threads := node.get_trait(Threads):
            for operand, result in threads.threads(node):
                outputs[result] = frame.get(operand)
        return tuple(outputs.values())


@stmts.dialect.register(key=KEY)
class WirePositions(interp.MethodTable):
    """Track origins through allocation and register operations."""

    @interp.impl(stmts.Alloc)
    @interp.impl(stmts.AllocRegister)
    @interp.impl(stmts.Put)
    @interp.impl(stmts.PackRegister)
    @interp.impl(stmts.JoinRegister)
    @interp.impl(stmts.SplitRegister)
    @interp.impl(stmts.TakeSlice)
    @interp.impl(stmts.PutSlice)
    def allocate(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: ir.Statement
    ):
        """Assign independent result identities to allocations and consuming register operations."""
        size = (
            constant_int(stmt.size) if isinstance(stmt, stmts.AllocRegister) else None
        )
        return tuple(
            Origin(Handle(value, value.type, None if size is None else Affine(size)))
            for value in stmt.results
        )

    @interp.impl(stmts.Borrow)
    @interp.impl(stmts.Take)
    def borrow(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: stmts.Borrow
    ):
        """Keep the register's origin and identify the selected element."""
        register = frame.get(stmt.register)
        selected = (
            register.data.pick(check.indices.of(stmt.index))
            if isinstance(register.data, Handle)
            else Origin.top()
        )
        return register, selected

    @interp.impl(stmts.Release)
    def release(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: stmts.Release
    ):
        """Require a release to restore the qubit from its original register element."""
        register = frame.get(stmt.register)
        expected = (
            register.data.pick(check.indices.of(stmt.index))
            if isinstance(register.data, Handle)
            else Origin.top()
        )
        check.expect(
            stmt,
            frame.get(stmt.wire),
            expected,
            "A release must restore the qubit from its register slot.",
        )
        return (register,)


@py.assign.dialect.register(key=KEY)
class Aliases(interp.MethodTable):
    """Preserve origins across aliases."""

    @interp.impl(py.assign.Alias)
    def alias(
        self,
        check: PositionAnalysis,
        frame: ForwardFrame[Origin],
        stmt: py.assign.Alias,
    ):
        """Read the operand's origin from the current frame."""
        return (frame.get(stmt.value),)


@py.tuple.dialect.register(key=KEY)
class Tuples(interp.MethodTable):
    """Keep separate origins for tuple fields."""

    @interp.impl(py.tuple.New)
    def new(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: py.tuple.New
    ):
        """Collect the origins of the tuple's fields in declaration order."""
        return (Origin(frame.get_values(stmt.args)),)


@py.indexing.dialect.register(key=KEY)
class Indexing(interp.MethodTable):
    """Read known tuple fields from their frame values."""

    @interp.impl(py.GetItem)
    def getitem(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: py.GetItem
    ):
        """Select a tuple field when its index is a known integer."""
        obj = frame.get(stmt.obj).data
        index = check.indices.of(stmt.index).constant
        if (
            isinstance(obj, tuple)
            and index is not None
            and -len(obj) <= index < len(obj)
        ):
            return (obj[index],)
        return (Origin.top(),)


@func.dialect.register(key=KEY)
class Functions(interp.MethodTable):
    """Validate calls with independent frames and preserve each result's origin."""

    @interp.impl(func.Invoke)
    def invoke(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: func.Invoke
    ):
        """Run the callee with input origins and distinguish its fresh results by call site."""
        inputs = frame.get_values(stmt.inputs)
        _, result = check.call(stmt.callee.code, Origin.top(), *inputs)
        roots = frozenset(root for value in inputs for root in value.roots())
        return (result.at_call(stmt, roots),)

    @interp.impl(func.Return)
    def return_(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: func.Return
    ):
        """Return the value's origin to the function validation hook."""
        return interp.ReturnValue(frame.get(stmt.value))


@scf.dialect.register(key=KEY)
class ControlFlow(interp.MethodTable):
    """Check positional continuity while Kirin interprets each structured body once."""

    @interp.impl(scf.Yield)
    def yield_(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: scf.Yield
    ):
        """Return yielded origins to the enclosing region handler."""
        return interp.YieldValue(frame.get_values(stmt.values))

    @interp.impl(scf.For)
    def loop(self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: scf.For):
        """Compare one iteration's outputs with its incoming wire origins."""
        initial = frame.get_values(stmt.initializers)
        with check.new_frame(stmt, has_parent_access=True) as body:
            yielded = check.frame_call_region(
                body, stmt, stmt.body, Origin.top(), *initial
            )
        assert isinstance(yielded, tuple)
        end = stmt.body.blocks[0].last_stmt
        assert isinstance(end, scf.Yield)
        for i, (argument, actual, expected) in enumerate(
            zip(stmt.body.blocks[0].args[1:], yielded, initial)
        ):
            if check.borrowing and is_linear(argument.type):
                check.expect(
                    end,
                    actual,
                    expected,
                    f"yield position {i} must continue loop input position {i}",
                )
        return tuple(before.join(after) for before, after in zip(initial, yielded))

    @interp.impl(scf.IfElse)
    def branch(
        self, check: PositionAnalysis, frame: ForwardFrame[Origin], stmt: scf.IfElse
    ):
        """Analyze both paths independently and compare their wire result origins."""
        yielded = []
        for region in (stmt.then_body, stmt.else_body):
            with check.new_frame(stmt, has_parent_access=True) as body:
                result = check.frame_call_region(
                    body, stmt, region, *(Origin.top() for _ in region.blocks[0].args)
                )
            assert isinstance(result, tuple)
            yielded.append(result)
        left, right = yielded
        for i, (value, first, second) in enumerate(zip(stmt.results, left, right)):
            if check.borrowing and is_linear(value.type):
                check.expect(
                    stmt,
                    second,
                    first,
                    f"both branches must continue the same qubit at yield position {i}",
                )
        return tuple(first.join(second) for first, second in zip(left, right))
