"""Promote the `Unwrap` and `Wrap` of a lowered kernel to wires, as MLIR's `mem2reg` does.

A slot is a place whose qubits can travel as one wire or one register. One qubit
is a slot, and so is each item of a `qalloc` register of static length that the
kernel only reads at constant indices. Such a register splits into one slot per
qubit, as SROA splits an aggregate. Any other register is one slot as a whole.
`RegisterAccesses` first turns each qubit access of such a register into a
`Borrow` or a `Release` on the register.

The promotion then runs in three steps, each a kirin rewrite rule:

1. `Boundaries` makes every slot that crosses a loop or a branch explicit. A loop
   carries the slot as a wire, with an `Unwrap` before it and a `Wrap` after it.
   Its body starts with a `Wrap` of the block argument and ends with an
   `Unwrap` before the yield. A branch reads the wire by capture.
2. `Forward` joins each `Unwrap` with the nearest earlier `Wrap` of the same slot
   in its block, if nothing between them touches the slot.
3. `MaterializeStart` and `MaterializeEnd` turn the remaining ends of each chain
   into `Alloc`, `Free` and wire parameters.

The reference analysis gives the place of each reference, and `QubitEffects`
gives the places that each loop and branch touches.
"""

from dataclasses import dataclass
from collections.abc import Mapping

from kirin import ir, types
from kirin.rewrite import Walk, Chain, Fixpoint, DeadCodeElimination
from kirin.dialects import py, scf, func
from kirin.rewrite.abc import RewriteRule, RewriteResult

from bloqade.qubit import stmts as qubit_stmts, qalloc
from bloqade.types import QubitType
from bloqade.analysis.reference import (
    Ref,
    Root,
    Slot,
    Place,
    Whole,
    Affine,
    Register,
    Returned,
    disjoint,
    origin_of,
)
from bloqade.squin.analysis.effects import (
    Effect,
    QubitEffects,
    places_in,
    holds_qubits,
)
from bloqade.squin.analysis.reference import (
    QubitReferenceAnalysis,
    qubit_register_length,
)
from bloqade.analysis.reference.layout import flat
from bloqade.analysis.reference.analysis import encloses

from . import stmts
from .index import index_value
from .lower import Layout
from .types import WireType, RegisterType
from .reference_check import same_place

TAKE = (stmts.Unwrap, stmts.UnwrapRegister)
PUT = (stmts.Wrap, stmts.WrapRegister)


def _put(stmt: ir.Statement, before: ir.Statement) -> ir.Statement:
    """Insert `stmt` before the statement `before` and return `stmt`."""
    stmt.insert_before(before)
    return stmt


def root_value(root: Root) -> ir.SSAValue | None:
    """Return the SSA value that holds the root `root` in its function, or None."""
    if isinstance(root, ir.SSAValue):
        return root
    if root.path == (0,) and len(root.call.results) == 1:
        return root.call.results[0]
    return None


def allocation(root: Root) -> ir.Statement | None:
    """Return the statement that allocates `root` in its function, or None.

    A `qubit.new`, a `Reference` and a call of `qalloc` allocate. A parameter does
    not.
    """
    if isinstance(root, Returned):
        return root.call if root.call.callee is qalloc else None
    owner = root.owner
    if isinstance(owner, (qubit_stmts.New, stmts.Reference, stmts.ReferenceRegister)):
        return owner
    return None


@dataclass(frozen=True)
class SlotPolicy:
    """Decide which places of a lowered kernel are slots.

    `accessed` maps each root to the places that the kernel touches in it.
    """

    accessed: Mapping[Root, set[Place]]

    def scalar_root(self, root: Root) -> bool:
        """Return True if every place of `root` that the kernel touches is a slot.

        A root that holds one qubit qualifies. A `qalloc` register qualifies if its
        length is static and the kernel reads it only at constant indices.
        """
        places = self.accessed.get(root, set())
        if all(isinstance(p, Whole) for p in places):
            return True
        if not isinstance(root, Returned) or root.call.callee is not qalloc:
            return False
        origin, calls = origin_of(root)
        if qubit_register_length(origin, calls) is None:
            return False
        return all(isinstance(p, Slot) and p.index.constant is not None for p in places)

    def slot(self, ref: Ref | None) -> Place | None:
        """Return `ref` if it is a slot, and None otherwise.

        One qubit of a root that splits is a slot. So is a whole register of a
        root that stays one register.
        """
        if not isinstance(ref, Place) or root_value(ref.root) is None:
            return None
        if ref.one_item:
            return ref if self.scalar_root(ref.root) else None
        if isinstance(ref, Register) and not self.scalar_root(ref.root):
            return ref
        return None


def take_out(ref: ir.SSAValue, slot: Place, before: ir.Statement) -> ir.SSAValue:
    """Insert the statement that takes the slot `slot` of `ref` out, and return it."""
    if slot.one_item:
        return _put(stmts.Unwrap(ref), before).wire
    return _put(stmts.UnwrapRegister(ref), before).register


def put_back(
    ref: ir.SSAValue, slot: Place, value: ir.SSAValue, before: ir.Statement
) -> None:
    """Insert the statement that puts `value` back into the slot `slot` of `ref`."""
    if slot.one_item:
        _put(stmts.Wrap(ref, value), before)
    else:
        _put(stmts.WrapRegister(ref, value), before)


def linear_type(slot: Place) -> types.TypeAttribute:
    """Return the type of the wire or register that carries the slot `slot`."""
    return WireType if slot.one_item else RegisterType


@dataclass
class RegisterAccesses(RewriteRule):
    """Turn each qubit access of a register that stays whole into a register access.

    `Unwrap(qs[i])` becomes `UnwrapRegister(qs)`, `Borrow` at `i` and
    `WrapRegister(qs)`. `Wrap(qs[i], w)` becomes `UnwrapRegister(qs)`, `Release`
    of `w` at `i` and `WrapRegister(qs)`. The register itself is then one slot.
    """

    refs: Mapping[ir.SSAValue, Ref]
    policy: SlotPolicy

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it takes out or puts back one qubit of such a register."""
        if not isinstance(node, (stmts.Unwrap, stmts.Wrap)):
            return RewriteResult()
        place = self.refs.get(node.qubit)
        if not isinstance(place, Place) or not place.path:
            return RewriteResult()
        if self.policy.scalar_root(place.root):
            return RewriteResult()
        register = root_value(place.root)
        at = flat(place)
        if register is None or not isinstance(at, Affine):
            return RewriteResult()
        index = index_value(at, node)
        whole = _put(stmts.UnwrapRegister(register), node).register
        if isinstance(node, stmts.Unwrap):
            borrow = _put(stmts.Borrow(whole, index), node)
            _put(stmts.WrapRegister(register, borrow.register_out), node)
            node.wire.replace_by(borrow.wire)
        else:
            release = _put(stmts.Release(whole, index, node.wire), node)
            _put(stmts.WrapRegister(register, release.register_out), node)
        node.delete()
        return RewriteResult(has_done_something=True)


def reference(place: Place, before: ir.Statement) -> ir.SSAValue:
    """Insert a reference to the slot `place` before `before`, and return it."""
    value = root_value(place.root)
    assert value is not None
    if isinstance(place, (Whole, Register)):
        return value
    assert isinstance(place, Slot) and place.index.constant is not None
    at = _put(py.Constant(place.index.constant), before)
    item = _put(py.GetItem(value, at.result), before)
    item.result.type = QubitType
    return item.result


def _defined_outside(root: Root, scope: ir.Statement) -> bool:
    """Return True if `root` comes from outside the loop or branch `scope`."""
    owner = root.call if isinstance(root, Returned) else root.owner
    return not encloses(scope, owner)


@dataclass
class Boundaries(RewriteRule):
    """Make each slot that a loop or a branch touches cross its boundary as a wire.

    Run the rule once with `Walk`. The rule rebuilds each loop and branch, and
    `Walk` visits only the statements that existed before it started.
    """

    effects: Mapping[ir.Statement, Effect]
    policy: SlotPolicy

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rebuild `node` if it is a loop or a branch that touches slots."""
        if not isinstance(node, (scf.For, scf.IfElse)) or node.parent_block is None:
            return RewriteResult()
        effect = self.effects.get(node)
        if effect is None or effect.opaque:
            return RewriteResult()
        carried = sorted(
            {
                place
                for place in effect.places
                if self.policy.slot(place) and _defined_outside(place.root, node)
            },
            key=repr,
        )
        if not carried:
            return RewriteResult()
        refs = [reference(place, node) for place in carried]
        starts = [take_out(ref, slot, node) for ref, slot in zip(refs, carried)]
        if isinstance(node, scf.For):
            made = self.loop(node, carried, refs, starts)
        else:
            made = self.branch(node, carried, refs, starts)
        kept = len(node.results)
        for old, new in zip(node.results, made.results[:kept], strict=True):
            old.replace_by(new)
        after = node.next_stmt
        assert after is not None
        outs = zip(carried, refs, made.results[kept:], strict=True)
        for slot, ref, result in outs:
            put_back(ref, slot, result, after)
        node.delete()
        return RewriteResult(has_done_something=True)

    @staticmethod
    def enclose(
        block: ir.Block,
        slots: list[Place],
        refs: list[ir.SSAValue],
        wires: list[ir.SSAValue],
    ) -> None:
        """Put `wires` into `refs` at the start of `block`, and take them out at its end.

        The yield at the end of `block` gets the taken wires as extra values.
        """
        first = block.first_stmt
        assert first is not None
        for slot, ref, wire in zip(slots, refs, wires, strict=True):
            put_back(ref, slot, wire, first)
        returned = block.last_stmt
        assert isinstance(returned, scf.Yield)
        outs = [take_out(ref, slot, returned) for slot, ref in zip(slots, refs)]
        _put(scf.Yield(*returned.values, *outs), returned)
        returned.delete()

    def loop(
        self,
        node: scf.For,
        slots: list[Place],
        refs: list[ir.SSAValue],
        starts: list[ir.SSAValue],
    ) -> ir.Statement:
        """Return a copy of the loop `node` that carries one wire per slot."""
        block = node.body.blocks[0]
        args = [block.args.append_from(linear_type(slot)) for slot in slots]
        self.enclose(block, slots, refs, args)
        region = node.body
        region.detach()
        made = scf.For(node.iterable, region, *node.initializers, *starts)
        return _put(made, node)

    def branch(
        self,
        node: scf.IfElse,
        slots: list[Place],
        refs: list[ir.SSAValue],
        starts: list[ir.SSAValue],
    ) -> ir.Statement:
        """Return a copy of the branch `node` whose bodies read and yield the wires."""
        for region in (node.then_body, node.else_body):
            if not region.blocks:
                block = ir.Block()
                block.args.append_from(node.cond.type, "cond")
                block.stmts.append(scf.Yield())
                region.blocks.append(block)
            self.enclose(region.blocks[0], slots, refs, starts)
        then_body, else_body = node.then_body, node.else_body
        then_body.detach()
        else_body.detach()
        return _put(scf.IfElse(node.cond, then_body, else_body), node)


@dataclass
class Scan:
    """Decide, for one slot, which statements of a block touch it.

    `refs` holds the places of the kernel, and `effects` holds the effect of each
    statement.
    """

    refs: Mapping[ir.SSAValue, Ref]
    effects: Mapping[ir.Statement, Effect]

    def place(self, value: ir.SSAValue) -> Place | None:
        """Return the place of the reference `value`, or None."""
        ref = self.refs.get(value)
        return ref if isinstance(ref, Place) else None

    def touches(self, stmt: ir.Statement, slot: Place) -> bool:
        """Return True if `stmt` may touch the qubit of `slot`.

        A pure statement only builds values. A statement with a body touches the
        places of its effect. Any other statement touches the places of its
        operands.
        """
        if stmt.has_trait(ir.Pure):
            return False
        if stmt.regions:
            effect = self.effects.get(stmt)
            if effect is None or effect.opaque:
                return True
            return any(not disjoint(slot, place) for place in effect.places)
        for value in stmt.args:
            ref = self.refs.get(value)
            if ref is None:
                continue
            if any(not disjoint(slot, place) for place in places_in(ref)):
                return True
        return False


@dataclass
class Forward(RewriteRule):
    """Join an `Unwrap` of a slot with the nearest earlier `Wrap` of it in its block.

    The `Unwrap` reads the wire of the `Wrap`, and both statements go away. The
    rule gives up if a statement between them may touch the slot.
    """

    scan: Scan
    policy: SlotPolicy

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Forward the wire of an earlier `Wrap` to `node`, if `node` is an `Unwrap`."""
        if not isinstance(node, TAKE) or node.parent_block is None:
            return RewriteResult()
        slot = self.policy.slot(self.scan.place(node.args[0]))
        if slot is None:
            return RewriteResult()
        stmt = node.prev_stmt
        while stmt is not None:
            if isinstance(stmt, PUT):
                other = self.scan.place(stmt.args[0])
                if other is not None and same_place(slot, other):
                    node.results[0].replace_by(stmt.args[1])
                    node.delete()
                    stmt.delete()
                    return RewriteResult(has_done_something=True)
            if self.scan.touches(stmt, slot):
                return RewriteResult()
            stmt = stmt.prev_stmt
        return RewriteResult()


def _first(node: ir.Statement, scan: Scan, slot: Place) -> bool:
    """Return True if no statement before `node` in its block touches `slot`."""
    stmt = node.prev_stmt
    while stmt is not None:
        if scan.touches(stmt, slot):
            return False
        stmt = stmt.prev_stmt
    return True


def _last(node: ir.Statement, scan: Scan, slot: Place) -> bool:
    """Return True if no statement after `node` in its block touches `slot`."""
    stmt = node.next_stmt
    while stmt is not None:
        if scan.touches(stmt, slot):
            return False
        stmt = stmt.next_stmt
    return True


def _local(root: Root, block: ir.Block) -> bool:
    """Return True if `block` allocates `root` itself."""
    made = allocation(root)
    return made is not None and made.parent_block is block


@dataclass
class MaterializeStart(RewriteRule):
    """Turn the first `Unwrap` of a slot into its allocation or its parameter.

    A slot whose root the block allocates gets an `Alloc` or an `AllocRegister`. A
    qubit or register parameter of the function becomes a wire or register
    parameter.
    """

    scan: Scan
    policy: SlotPolicy

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is the first access to its slot in its block."""
        if not isinstance(node, TAKE) or node.parent_block is None:
            return RewriteResult()
        slot = self.policy.slot(self.scan.place(node.args[0]))
        if slot is None or not _first(node, self.scan, slot):
            return RewriteResult()
        block = node.parent_block
        root = slot.root
        made = allocation(root)
        if _local(root, block):
            if isinstance(made, (stmts.Reference, stmts.ReferenceRegister)):
                return RewriteResult()
            if slot.one_item:
                fresh = _put(stmts.Alloc(), node).result
            else:
                assert isinstance(made, func.Invoke)
                fresh = _put(stmts.AllocRegister(made.inputs[0]), node).result
            node.results[0].replace_by(fresh)
        elif self.parameter(root, block) and len(root.uses) == 1:
            root.type = linear_type(slot)
            node.results[0].replace_by(root)
        else:
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)

    @staticmethod
    def parameter(root: Root, block: ir.Block) -> bool:
        """Return True if `root` is a parameter of the function whose entry is `block`."""
        return (
            isinstance(root, ir.BlockArgument)
            and root.owner is block
            and root.index > 0
            and isinstance(block.parent_stmt, func.Function)
        )


@dataclass
class MaterializeEnd(RewriteRule):
    """Turn the last `Wrap` of a slot that its block allocates into a `Free`."""

    scan: Scan
    policy: SlotPolicy

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Rewrite `node` if it is the last access to a local slot in its block."""
        if not isinstance(node, PUT) or node.parent_block is None:
            return RewriteResult()
        slot = self.policy.slot(self.scan.place(node.args[0]))
        if slot is None or not _local(slot.root, node.parent_block):
            return RewriteResult()
        if not _last(node, self.scan, slot):
            return RewriteResult()
        if slot.one_item:
            _put(stmts.Free(node.args[1]), node)
        else:
            _put(stmts.FreeRegister(node.args[1]), node)
        node.delete()
        return RewriteResult(has_done_something=True)


class DropInvariantReferences(RewriteRule):
    """Remove a qubit reference that a loop carries and yields back unchanged.

    Kirin's lowering carries every variable of a loop body, so a loop over `x(q)`
    carries the reference `q`. After promotion, the qubit itself travels as a
    wire, and the carried reference keeps only its allocation alive.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Remove the invariant qubit references that the loop `node` carries."""
        if not isinstance(node, scf.For) or node.parent_block is None:
            return RewriteResult()
        block = node.body.blocks[0]
        returned = block.last_stmt
        if not isinstance(returned, scf.Yield):
            return RewriteResult()
        # A body yields an invariant reference back as its block argument, or as
        # the initializer once an earlier drop has replaced the argument.
        dropped = [
            k
            for k, arg in enumerate(block.args[1:])
            if holds_qubits(arg.type)
            and returned.values[k] in (arg, node.initializers[k])
        ]
        if not dropped:
            return RewriteResult()
        for k in dropped:
            block.args[k + 1].replace_by(node.initializers[k])
            node.results[k].replace_by(node.initializers[k])
        kept_values = [v for k, v in enumerate(returned.values) if k not in dropped]
        _put(scf.Yield(*kept_values), returned)
        returned.delete()
        for k in reversed(dropped):
            block.args[k + 1].delete()
        region = node.body
        region.detach()
        inits = [v for k, v in enumerate(node.initializers) if k not in dropped]
        made = _put(scf.For(node.iterable, region, *inits), node)
        kept = [r for k, r in enumerate(node.results) if k not in dropped]
        for old, new in zip(kept, made.results, strict=True):
            old.replace_by(new)
        node.delete()
        return RewriteResult(has_done_something=True)


class DeadAllocations(RewriteRule):
    """Remove a `qubit.new`, a `Reference` or a `qalloc` call that nothing reads."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        """Remove `node` if it allocates a reference without uses."""
        fresh = (qubit_stmts.New, stmts.Reference, stmts.ReferenceRegister)
        dead = isinstance(node, fresh) or (
            isinstance(node, func.Invoke) and node.callee is qalloc
        )
        if not dead or node.results[0].uses:
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)


def analyze(method: ir.Method) -> tuple[Mapping[ir.SSAValue, Ref], QubitEffects]:
    """Return the places and the effects of the current form of `method`."""
    frame, _ = QubitReferenceAnalysis(method.dialects).run(method)
    effects = QubitEffects(method.dialects, refs=frame.entries)
    effects.run(method)
    return frame.entries, effects


def promote(method: ir.Method, layout: Layout | None = None) -> RewriteResult:
    """Promote the slots of the lowered `method` in place.

    If `layout` is given, the function gets the signature of its wire function.
    """
    refs, effects = analyze(method)
    policy = SlotPolicy(effects.accessed)
    result = Walk(RegisterAccesses(refs, policy)).rewrite(method.code)
    refs, effects = analyze(method)
    policy = SlotPolicy(effects.accessed)
    result = Walk(Boundaries(effects.effects, policy)).rewrite(method.code).join(result)
    refs, effects = analyze(method)
    policy = SlotPolicy(effects.accessed)
    scan = Scan(refs, effects.effects)
    result = Fixpoint(Walk(Forward(scan, policy))).rewrite(method.code).join(result)
    # The reads `qs[i]` of the promoted accesses are dead now, and loops still
    # carry the references that kirin's lowering added. A parameter becomes a wire
    # parameter only when its last use is the `Unwrap` that takes it out.
    tidy = Chain(DeadCodeElimination(), DropInvariantReferences())
    result = Fixpoint(Walk(tidy)).rewrite(method.code).join(result)
    # `DropInvariantReferences` rebuilds loops, so the effects need a fresh run.
    refs, effects = analyze(method)
    policy = SlotPolicy(effects.accessed)
    scan = Scan(refs, effects.effects)
    ends = Chain(MaterializeStart(scan, policy), MaterializeEnd(scan, policy))
    result = Fixpoint(Walk(ends)).rewrite(method.code).join(result)
    cleanup = Chain(DeadCodeElimination(), DropInvariantReferences(), DeadAllocations())
    result = Fixpoint(Walk(cleanup)).rewrite(method.code).join(result)
    if layout is not None:
        retype(method, layout)
    return result


def retype(method: ir.Method, layout: Layout) -> None:
    """Give the function of `method` the signature of its wire function."""
    code = method.code
    assert isinstance(code, func.Function)
    block = code.body.blocks[0]
    inputs = tuple(arg.type for arg in block.args[1:])
    block.args[0].type = types.MethodType[list(inputs), layout.output]
    code.signature = func.Signature(inputs=inputs, output=layout.output)
