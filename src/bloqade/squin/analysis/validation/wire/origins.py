"""Verify explicit qubit origins and reference transfers with ordered region checks.

Each check follows operation operands and declared wire continuations.
Calls verify their callees, and structured regions get separate transfer states.
"""

from dataclasses import field, dataclass

from kirin import ir, types
from kirin.rewrite import Walk
from kirin.dialects import py, scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.constants import constant_int
from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import WireType, RegisterType, is_linear
from bloqade.squin.wire.traits import Threads
from bloqade.analysis.reference.index import Affine
from bloqade.squin.analysis.reference import holds_qubits
from bloqade.squin.rewrite.wire.handles import (
    Handle,
    Handles,
    Unknown,
    Descriptor,
    merge,
    roots,
    distinct,
    quantum_type,
    typed_length,
)


@dataclass
class Origins(RewriteRule):
    """Follow definition identities while checking positions or reference ownership."""

    positions: bool = True
    references: bool = True
    handles: Handles = field(default_factory=Handles)
    out: dict[Handle, ir.SSAValue | None] = field(default_factory=dict)
    errors: list[ir.ValidationError] = field(default_factory=list)
    active: set[ir.Statement] = field(default_factory=set)
    method: ir.Method | None = None
    returned: Descriptor = None

    def error(self, node: ir.Statement, message: str) -> None:
        """Record a contract violation at its operation."""
        self.errors.append(ir.ValidationError(node, message))

    def expect(
        self, node: ir.Statement, actual: Descriptor, expected: Descriptor, message: str
    ) -> None:
        """Require two values to continue the same allocation and element."""
        if self.positions and not (isinstance(actual, Handle) and actual == expected):
            self.error(node, message)

    def walk(self, region: ir.Region) -> None:
        """Visit one region while structured rules visit nested bodies explicitly."""
        Walk(
            self, skip=lambda node: isinstance(node, ir.Region) and node is not region
        ).rewrite(region)

    def run(
        self, method: ir.Method, inputs: tuple[Descriptor, ...] | None = None
    ) -> Descriptor:
        """Check one function with declared parameter origins or caller-provided origins."""
        self.method = method
        if method.code in self.active:
            self.error(method.code, "Wire validation does not support recursive calls.")
            return Unknown()
        self.active.add(method.code)
        for i, parameter in enumerate(method.callable_region.blocks[0].args[1:]):
            if inputs is not None:
                self.handles.values[parameter] = inputs[i]
            elif (kind := quantum_type(parameter.type)) is not None:
                length = typed_length(parameter.type)
                self.handles.values[parameter] = Handle(
                    parameter, kind, None if length is None else Affine(length)
                )
        self.walk(method.callable_region)
        self.active.remove(method.code)
        return self.returned

    def child(self) -> "Origins":
        """Give a structured body its own outstanding transfers and the shared definition map."""
        return Origins(
            self.positions,
            self.references,
            self.handles,
            dict(self.out),
            self.errors,
            self.active,
            self.method,
        )

    def region(self, region: ir.Region) -> "Origins":
        """Check a body and require it to restore its incoming reference ownership."""
        inner = self.child()
        inner.walk(region)
        if self.references and inner.out.keys() != self.out.keys():
            assert region.parent_node is not None
            self.error(
                region.parent_node,
                "A structured body changes which qubits are out as wires.",
            )
        return inner

    def allocate(
        self, node: ir.Statement, kind: types.TypeAttribute, size: int | None = None
    ) -> None:
        """Assign a new origin to a source or wire allocation."""
        self.handles.values[node.results[0]] = Handle(
            node.results[0], kind, None if size is None else Affine(size)
        )

    def reference_access(self, node: ir.Statement) -> None:
        """Track ownership transfers and require wraps to restore their original qubits."""
        reference = self.handles.get(node.args[0])
        if not isinstance(reference, Handle):
            if self.references:
                self.error(node, f"'{node.name}' reads a reference that has no place.")
            if node.results:
                self.handles.values[node.results[0]] = Unknown()
            return
        if isinstance(node, (stmts.Unwrap, stmts.UnwrapRegister)):
            if self.references and any(
                not distinct(reference, held) for held in self.out
            ):
                self.error(node, f"'{node.name}' takes a qubit that is already out.")
            self.out[reference] = node.results[0]
            self.handles.values[node.results[0]] = reference
        else:
            self.expect(
                node,
                self.handles.get(node.args[1]),
                reference,
                f"'{node.name}' must restore the qubits of its reference.",
            )
            if reference not in self.out:
                if self.references:
                    self.error(
                        node, f"'{node.name}' puts back a qubit that is not out."
                    )
            else:
                del self.out[reference]

    def observer(self, node: ir.Statement) -> None:
        """Reject a source operation that accesses a reference whose qubit travels as a wire."""

        def leaves(value):
            if isinstance(value, Handle):
                return (value,)
            if isinstance(value, tuple):
                return tuple(item for member in value for item in leaves(member))
            if isinstance(value, Unknown):
                return (value,)
            return ()

        if not self.references or not self.out:
            return
        for value in node.args:
            if not holds_qubits(value.type):
                continue
            for item in leaves(self.handles.get(value)):
                if isinstance(item, Unknown) or any(
                    not distinct(item, held) for held in self.out
                ):
                    self.error(
                        node, f"'{node.name}' touches a qubit that is out as a wire."
                    )
                    return

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Check an operation and record the identities of its results."""
        if isinstance(node, scf.For):
            return self.loop(node)
        if isinstance(node, scf.IfElse):
            return self.branch(node)
        if isinstance(node, scf.Yield):
            return RewriteResult()
        if isinstance(node, func.Return):
            self.return_(node)
            return RewriteResult()
        if isinstance(node, (stmts.Alloc, qubit_stmts.New)):
            self.allocate(node, WireType)
        elif isinstance(node, stmts.AllocRegister):
            self.allocate(node, RegisterType, constant_int(node.size))
        elif isinstance(node, func.Invoke) and node.callee is qalloc:
            size = constant_int(node.inputs[0])
            self.allocate(node, RegisterType, None if size is None else max(0, size))
        elif isinstance(node, (stmts.Reference, stmts.ReferenceRegister)):
            fill = node.next_stmt
            valid = (
                isinstance(fill, (stmts.Wrap, stmts.WrapRegister))
                and fill.args[0] is node.results[0]
            )
            if not valid:
                if self.references:
                    self.error(
                        node,
                        f"'{node.name}' needs a wrap of its result right after it.",
                    )
                self.allocate(
                    node,
                    WireType if isinstance(node, stmts.Reference) else RegisterType,
                )
            else:
                assert fill is not None
                self.handles.values[node.results[0]] = self.handles.get(fill.args[1])
            handle = self.handles.get(node.results[0])
            if isinstance(handle, Handle):
                self.out[handle] = None
        elif isinstance(
            node, (stmts.Unwrap, stmts.UnwrapRegister, stmts.Wrap, stmts.WrapRegister)
        ):
            self.reference_access(node)
        elif isinstance(node, stmts.Borrow):
            register = self.handles.get(node.register)
            self.handles.values[node.register_out] = register
            self.handles.values[node.wire] = (
                register.pick(self.handles.indices.of(node.index))
                if isinstance(register, Handle)
                else Unknown()
            )
        elif isinstance(node, stmts.Release):
            register = self.handles.get(node.register)
            expected = (
                register.pick(self.handles.indices.of(node.index))
                if isinstance(register, Handle)
                else Unknown()
            )
            self.expect(
                node,
                self.handles.get(node.wire),
                expected,
                "A release must restore the qubit from its register slot.",
            )
            self.handles.values[node.register_out] = register
        elif isinstance(node, func.Invoke):
            self.observer(node)
            inner = Origins(
                self.positions, self.references, errors=self.errors, active=self.active
            )
            result = inner.run(
                node.callee, tuple(self.handles.get(value) for value in node.inputs)
            )
            passed = set()
            for value in node.inputs:
                passed.update(roots(self.handles.get(value)) or ())

            def returned(value):
                if isinstance(value, Handle) and value.root not in passed:
                    return Handle(
                        (node, value.root),
                        value.kind,
                        value.length,
                        value.selection,
                        value.shape,
                    )
                if isinstance(value, tuple):
                    return tuple(returned(item) for item in value)
                return value

            self.handles.values[node.result] = returned(result)
        elif self.handles.record(node):
            pass
        elif threads := node.get_trait(Threads):
            for operand, result in threads.threads(node):
                self.handles.values[result] = self.handles.get(operand)
        else:
            self.observer(node)
        return RewriteResult()

    def return_(self, node: func.Return) -> None:
        """Check borrowed return positions and outstanding transfers at the function exit."""
        result = self.handles.get(node.value)
        self.returned = result
        assert self.method is not None
        parameters = [
            arg
            for arg in self.method.callable_region.blocks[0].args[1:]
            if is_linear(arg.type)
        ]
        outputs = result if isinstance(result, tuple) else (result,)
        if self.positions and len(outputs) < len(parameters):
            self.error(node, "The return suffix must include every borrowed parameter.")
        elif parameters:
            for i, (actual, parameter) in enumerate(
                zip(outputs[-len(parameters) :], parameters)
            ):
                self.expect(
                    node,
                    actual,
                    self.handles.get(parameter),
                    f"return suffix position {i} must continue borrowed parameter {i}",
                )
        returned = {node.value}
        if isinstance(node.value.owner, py.tuple.New):
            returned.update(node.value.owner.args)
        if self.references and any(
            value not in returned for value in self.out.values()
        ):
            self.error(node, "The function returns while qubits are out as wires.")

    def loop(self, node: scf.For) -> RewriteResult:
        """Check one iteration with explicit input origins and require positional continuity."""
        block = node.body.blocks[0]
        end = block.last_stmt
        assert isinstance(end, scf.Yield)
        initial = [self.handles.get(value) for value in node.initializers]
        for i, (argument, descriptor) in enumerate(zip(block.args[1:], initial)):
            if holds_qubits(argument.type) and end.values[i] not in (
                argument,
                node.initializers[i],
            ):
                descriptor = Unknown()
            self.handles.values[argument] = descriptor
        self.region(node.body)
        for i, (result, value, argument, before) in enumerate(
            zip(node.results, end.values, block.args[1:], initial)
        ):
            actual = self.handles.get(value)
            if is_linear(argument.type):
                self.expect(
                    end,
                    actual,
                    self.handles.get(argument),
                    f"yield position {i} must continue loop input position {i}",
                )
                self.handles.values[result] = actual
            else:
                self.handles.values[result] = merge(before, actual)
        return RewriteResult()

    def branch(self, node: scf.IfElse) -> RewriteResult:
        """Check both branches from the same incoming origins and outstanding transfers."""
        self.region(node.then_body)
        self.region(node.else_body)
        left, right = (
            node.then_body.blocks[0].last_stmt,
            node.else_body.blocks[0].last_stmt,
        )
        if not isinstance(left, scf.Yield) or not isinstance(right, scf.Yield):
            return RewriteResult()
        for i, (result, a, b) in enumerate(
            zip(node.results, left.values, right.values)
        ):
            first, second = self.handles.get(a), self.handles.get(b)
            if is_linear(result.type):
                self.expect(
                    node,
                    second,
                    first,
                    f"both branches must continue the same qubit at yield position {i}",
                )
            self.handles.values[result] = merge(first, second)
        return RewriteResult()
