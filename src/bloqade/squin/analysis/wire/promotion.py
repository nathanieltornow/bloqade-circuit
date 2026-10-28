"""Plan reference promotion without changing the input program.

Reference analysis identifies allocations and their aliases.
Eligibility checks select whole allocations whose accesses promotion can express.
A forward interpreter tracks the current value and outstanding borrows of each selected allocation.
Loop inputs and region results use symbolic definitions until the rewrite creates their SSA values.
"""

from dataclasses import field, dataclass
from collections.abc import Mapping

from kirin import ir, types, interp
from kirin.lattice import EmptyLattice
from kirin.rewrite import Walk
from kirin.analysis import ForwardExtra, ForwardFrame
from kirin.dialects import scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.squin.wire import stmts
from bloqade.squin.wire.types import WireType, RegisterType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import ReferenceAccess
from bloqade.analysis.reference import Ref, Root, Place, Affine, Returned, places_in
from bloqade.analysis.reference.index import same
from bloqade.squin.analysis.wire.slots import SlotAnalysis
from bloqade.analysis.reference.queries import flat

KEY = "squin.wire.promotion"


@dataclass(frozen=True, eq=False)
class Definition:
    """Name a wire value that the rewrite will create."""

    type: types.TypeAttribute


Value = ir.SSAValue | Definition


@dataclass(frozen=True)
class Slot:
    """Identify one whole allocation and its linear value type."""

    root: Root
    allocation: ir.Statement
    type: types.TypeAttribute


@dataclass(frozen=True)
class Operation:
    """Describe a wire operation that the rewrite must insert at an existing statement."""

    kind: type[ir.Statement]
    operands: tuple[Value, ...]
    results: tuple[Definition, ...]
    index: Affine | None = None


@dataclass(frozen=True)
class Region:
    """Describe the extra inputs and outputs of one structured control-flow operation."""

    inputs: tuple[Value, ...]
    arguments: tuple[Definition, ...]
    results: tuple[Definition, ...]


@dataclass
class PromotionPlan:
    """Record a complete promotion decision for one unchanged method.

    The plan holds symbolic values and references to existing IR nodes.
    Building the plan creates no IR nodes and changes no use lists.
    Apply the plan once to the method that supplied it.
    """

    code: ir.Statement
    applied: bool = field(default=False, init=False)
    slots: dict[Root, Slot] = field(default_factory=dict)
    before: dict[ir.Statement, list[Operation]] = field(default_factory=dict)
    regions: dict[ir.Statement, Region] = field(default_factory=dict)
    yields: dict[scf.Yield, tuple[Value, ...]] = field(default_factory=dict)
    replacements: dict[ir.SSAValue, Value] = field(default_factory=dict)
    removed: set[ir.Statement] = field(default_factory=set)

    def insert(self, node, kind, operands=(), result_types=(), index=None):
        """Record an insertion and return its symbolic results."""
        results = tuple(Definition(kind) for kind in result_types)
        self.before.setdefault(node, []).append(
            Operation(kind, tuple(operands), results, index)
        )
        return results


def allocation(root: Root) -> ir.Statement | None:
    """Return the local allocation or reference initializer for a root."""
    if isinstance(root, Returned):
        return (
            root.call
            if isinstance(root.call, func.Invoke) and root.call.callee is qalloc
            else None
        )
    owner = root.owner
    if isinstance(owner, (qubit_stmts.New, stmts.Reference, stmts.ReferenceRegister)):
        return owner
    return None


@dataclass
class Eligibility(RewriteRule):
    """Check every selected allocation's access before planning any replacements."""

    refs: Mapping[ir.SSAValue, Ref]
    blocked: set[Root]
    allocations: dict[ir.Statement, Slot]
    ordered: dict[Root, Slot] = field(default_factory=dict)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Record allocations in program order and exclude unsupported reference accesses."""
        if slot := self.allocations.get(node):
            self.ordered[slot.root] = slot
        if access := node.get_trait(ReferenceAccess):
            ref = self.refs.get(access.reference(node))
            if isinstance(ref, Place):
                if ref.path and not (ref.one_item and isinstance(flat(ref), Affine)):
                    self.blocked.add(ref.root)
                scope = node.parent_stmt
                while scope is not None and not isinstance(scope, func.Function):
                    if not isinstance(scope, (scf.For, scf.IfElse)):
                        self.blocked.add(ref.root)
                    scope = scope.parent_stmt
        return RewriteResult()


@dataclass(frozen=True)
class Contents:
    """Hold the current value and the indices of outstanding register borrows.

    A missing value means that an unwrap transferred the whole allocation to an explicit wire.
    A later wrap must restore that value before another reference access.
    """

    value: Value | None
    borrowed: frozenset[Affine] = frozenset()


@dataclass
class Frame(ForwardFrame[EmptyLattice]):
    """Track reference contents during one forward visit to a structured region."""

    contents: dict[Root, Contents] = field(default_factory=dict)
    local: list[Root] = field(default_factory=list)


@dataclass
class PromotionAnalysis(ForwardExtra[Frame, EmptyLattice]):
    """Compute each access's current value through Kirin's structured control-flow rules."""

    keys = (KEY, "absint")
    lattice = EmptyLattice
    plan: PromotionPlan
    refs: Mapping[ir.SSAValue, Ref]
    touched: Mapping[ir.Statement, set[Place]]
    allocations: dict[ir.Statement, Slot] = field(default_factory=dict, init=False)

    def initialize(self):
        """Index the selected allocations before the forward visit."""
        self.allocations = {slot.allocation: slot for slot in self.plan.slots.values()}
        return super().initialize()

    def initialize_frame(self, node, *, has_parent_access=False):
        """Create a separate reference state for each function, loop or branch body."""
        return Frame(node, has_parent_access=has_parent_access)

    def method_self(self, method):
        """Give the method object the neutral analysis value."""
        return EmptyLattice.top()

    def frame_eval(self, frame, node):
        """Start each selected allocation at its original program position."""
        if slot := self.allocations.get(node):
            if isinstance(node, (stmts.Reference, stmts.ReferenceRegister)):
                value = None
            elif slot.type == WireType:
                (value,) = self.plan.insert(node, stmts.Alloc, result_types=(WireType,))
            else:
                assert isinstance(node, func.Invoke)
                (value,) = self.plan.insert(
                    node, stmts.AllocRegister, node.inputs, (RegisterType,)
                )
            frame.contents[slot.root] = Contents(value)
            frame.local.append(slot.root)
        return super().frame_eval(frame, node)

    def require(self, frame: Frame, root: Root, *, restored=False) -> Contents:
        """Require a current value and optionally require every borrowed element to be present."""
        state = frame.contents.get(root)
        if state is None or state.value is None:
            raise WireConversionError(
                f"Promotion requires an available value for {root!r}."
            )
        if restored and state.borrowed:
            raise WireConversionError(
                f"Promotion requires every borrowed element of {root!r} at the region boundary."
            )
        return state

    def value(self, frame: Frame, root: Root) -> Value:
        """Return a fully restored value for a region boundary."""
        state = self.require(frame, root, restored=True)
        assert state.value is not None
        return state.value

    def finish(self, frame: Frame, node: ir.Statement) -> None:
        """Free local allocations that still belong to the region at its exit."""
        for root in frame.local:
            state = frame.contents[root]
            if state.value is None:
                continue
            self.require(frame, root, restored=True)
            kind = (
                stmts.Free
                if self.plan.slots[root].type == WireType
                else stmts.FreeRegister
            )
            self.plan.insert(node, kind, (state.value,))
            frame.contents[root] = Contents(None)

    def crossing(self, node: ir.Statement, frame: Frame) -> tuple[Root, ...]:
        """Select touched allocations that already exist at this region boundary."""
        touched = {place.root for place in self.touched.get(node, ())}
        return tuple(
            root
            for root in self.plan.slots
            if root in touched and root in frame.contents
        )

    def body(self, frame: Frame, node: ir.Statement, region: ir.Region, contents):
        """Visit one structured body with its planned incoming reference state."""
        if len(region.blocks) != 1:
            raise WireConversionError(
                "Promotion requires a single block in each structured region."
            )
        block = region.blocks[0]
        if not isinstance(block.last_stmt, scf.Yield):
            raise WireConversionError(
                "Promotion requires structured bodies to end with a yield."
            )
        with self.new_frame(node, has_parent_access=True) as inner:
            inner.contents = dict(contents)
            self.frame_call_region(
                inner, node, region, *(EmptyLattice.top() for _ in block.args)
            )
        return inner

    def eval_fallback(self, frame: Frame, node: ir.Statement):
        """Plan reference transfers and leave existing wire and classical operations in place."""
        access = node.get_trait(ReferenceAccess)
        if access is None:
            return tuple(EmptyLattice.top() for _ in node.results)
        ref = self.refs.get(access.reference(node))
        if not isinstance(ref, Place) or ref.root not in self.plan.slots:
            return tuple(EmptyLattice.top() for _ in node.results)
        root = ref.root
        if ref.path:
            index = flat(ref)
            assert isinstance(index, Affine)
            state = self.require(frame, root)
            if access.takes:
                if any(same(index, held) is not False for held in state.borrowed):
                    raise WireConversionError(
                        "Promotion requires distinct simultaneous register borrows."
                    )
                register, value = self.plan.insert(
                    node, stmts.Borrow, (state.value,), (RegisterType, WireType), index
                )
                self.plan.replacements[access.value(node)] = value
                frame.contents[root] = Contents(register, state.borrowed | {index})
            else:
                if index not in state.borrowed:
                    raise WireConversionError(
                        "Promotion requires a matching borrow before a register release."
                    )
                (register,) = self.plan.insert(
                    node,
                    stmts.Release,
                    (state.value, access.value(node)),
                    (RegisterType,),
                    index,
                )
                frame.contents[root] = Contents(register, state.borrowed - {index})
        elif access.takes:
            state = self.require(frame, root, restored=True)
            assert state.value is not None
            self.plan.replacements[access.value(node)] = state.value
            frame.contents[root] = Contents(None)
        else:
            state = frame.contents.get(root)
            if state is None or state.value is not None:
                raise WireConversionError(
                    "Promotion requires an empty reference before a wrap."
                )
            frame.contents[root] = Contents(access.value(node))
        self.plan.removed.add(node)
        return tuple(EmptyLattice.top() for _ in node.results)


@func.dialect.register(key=KEY)
class Functions(interp.MethodTable):
    """Keep calls explicit and finish local lifetimes at the function return."""

    @interp.impl(func.Invoke)
    def invoke(self, analysis: PromotionAnalysis, frame: Frame, stmt: func.Invoke):
        """Treat a lowered call's explicit wire operands and results as existing SSA values."""
        return (EmptyLattice.top(),)

    @interp.impl(func.Return)
    def return_(self, analysis: PromotionAnalysis, frame: Frame, stmt: func.Return):
        """End the lifetimes of local allocations that the function retains."""
        analysis.finish(frame, stmt)
        return interp.ReturnValue(EmptyLattice.top())


@scf.dialect.register(key=KEY)
class ControlFlow(interp.MethodTable):
    """Plan loop arguments and branch results from explicit incoming and outgoing values."""

    @interp.impl(scf.For)
    def loop(self, analysis: PromotionAnalysis, frame: Frame, stmt: scf.For):
        """Visit the loop body once with symbolic values for the current iteration."""
        roots = analysis.crossing(stmt, frame)
        inputs = tuple(analysis.value(frame, root) for root in roots)
        arguments = tuple(Definition(analysis.plan.slots[root].type) for root in roots)
        contents = dict(frame.contents)
        contents.update(
            (root, Contents(value)) for root, value in zip(roots, arguments)
        )
        inner = analysis.body(frame, stmt, stmt.body, contents)
        outputs = tuple(analysis.value(inner, root) for root in roots)
        if roots:
            results = tuple(
                Definition(analysis.plan.slots[root].type) for root in roots
            )
            analysis.plan.regions[stmt] = Region(inputs, arguments, results)
            terminator = stmt.body.blocks[0].last_stmt
            assert isinstance(terminator, scf.Yield)
            analysis.plan.yields[terminator] = outputs
            frame.contents.update(
                (root, Contents(value)) for root, value in zip(roots, results)
            )
        return tuple(EmptyLattice.top() for _ in stmt.results)

    @interp.impl(scf.IfElse)
    def branch(self, analysis: PromotionAnalysis, frame: Frame, stmt: scf.IfElse):
        """Plan both branches from the same incoming values and merge their outgoing positions."""
        roots = analysis.crossing(stmt, frame)
        inputs = tuple(analysis.value(frame, root) for root in roots)
        for body in (stmt.then_body, stmt.else_body):
            inner = analysis.body(frame, stmt, body, frame.contents)
            if roots:
                terminator = body.blocks[0].last_stmt
                assert isinstance(terminator, scf.Yield)
                analysis.plan.yields[terminator] = tuple(
                    analysis.value(inner, root) for root in roots
                )
        if roots:
            results = tuple(
                Definition(analysis.plan.slots[root].type) for root in roots
            )
            analysis.plan.regions[stmt] = Region(inputs, (), results)
            frame.contents.update(
                (root, Contents(value)) for root, value in zip(roots, results)
            )
        return tuple(EmptyLattice.top() for _ in stmt.results)

    @interp.impl(scf.Yield)
    def yield_(self, analysis: PromotionAnalysis, frame: Frame, stmt: scf.Yield):
        """End local lifetimes before the enclosing rule records its outgoing values."""
        analysis.finish(frame, stmt)
        return interp.YieldValue(tuple(EmptyLattice.top() for _ in stmt.values))


def plan_promotion(method: ir.Method) -> PromotionPlan:
    """Check eligibility and compute a complete plan without changing the method."""
    analysis = SlotAnalysis(method.dialects)
    frame, _ = analysis.run(method)
    plan = PromotionPlan(method.code)
    if analysis.pin_all:
        return plan
    allocations = {}
    roots = {place.root for ref in frame.entries.values() for place in places_in(ref)}
    for root in roots:
        if (node := allocation(root)) is not None:
            kind = (
                WireType
                if isinstance(node, (qubit_stmts.New, stmts.Reference))
                else RegisterType
            )
            allocations[node] = Slot(root, node, kind)
    eligibility = Eligibility(frame.entries, set(analysis.pinned), allocations)
    Walk(eligibility, region_first=True).rewrite(method.code)
    plan.slots = {
        root: slot
        for root, slot in eligibility.ordered.items()
        if root not in eligibility.blocked
    }
    if not plan.slots:
        return plan
    if len(method.callable_region.blocks) != 1:
        raise WireConversionError(
            "Promotion requires a function with one structured entry block."
        )
    PromotionAnalysis(
        method.dialects, plan=plan, refs=frame.entries, touched=analysis.touched
    ).run(method)
    return plan
