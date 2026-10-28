"""Lower gates and calls locally to explicit reference transfers and wire operations.

The ordered converter supplies handles for operands that it has already visited.
Each lowering returns its new statements in construction order for immediate conversion.
"""

from dataclasses import field, dataclass
from collections.abc import Sequence

from kirin import ir, types
from kirin.dialects import py, func, ilist

from bloqade.types import QubitType
from bloqade.squin.wire import stmts
from bloqade.analysis.indices import Sym, Affine
from bloqade.squin.wire.twins import TWINS
from bloqade.squin.wire.types import WireType, RegisterType
from bloqade.squin.wire.errors import WireConversionError
from bloqade.squin.wire.traits import Twin
from bloqade.squin.rewrite.wire.handles import Handle, Handles, Unknown, distinct
from bloqade.squin.rewrite.wire.normalize import BroadcastToLoop


def output_types(output: types.TypeAttribute) -> tuple[types.TypeAttribute, ...]:
    """Return the type of each output of a function whose output type is `output`.

    `None` gives no output. A tuple gives one output per member, and any other type
    gives one output.
    """
    if output.is_subseteq(types.NoneType):
        return ()
    if isinstance(output, types.Generic) and output.is_subseteq(types.Tuple):
        return tuple(output.vars)
    return (output,)


def packed_type(outputs: Sequence[types.TypeAttribute]) -> types.TypeAttribute:
    """Return the output type of a function with the output types `outputs`."""
    if not outputs:
        return types.NoneType
    if len(outputs) == 1:
        return outputs[0]
    return types.Generic(tuple, *outputs)


@dataclass(frozen=True)
class Contract:
    """Describe a converted function's ordinary results and borrowed-parameter suffix."""

    method: ir.Method
    inputs: tuple[types.TypeAttribute, ...]
    ordinary: tuple[types.TypeAttribute, ...]
    borrowed: tuple[int, ...]
    lengths: tuple[Affine | None, ...]
    packed: bool

    @property
    def outputs(self) -> tuple[types.TypeAttribute, ...]:
        """Return all wire result types in their declared order."""
        return self.ordinary + tuple(self.inputs[i] for i in self.borrowed)


@dataclass
class LocalLowering:
    """Build a replacement beside its source statement and retain the new statements in order."""

    before: ir.Statement
    handles: Handles
    created: list[ir.Statement] = field(default_factory=list)

    def put(self, node):
        """Insert and record one statement for the ordered converter."""
        node.insert_before(self.before)
        self.created.append(node)
        return node

    def take(self, reference: ir.SSAValue) -> ir.SSAValue:
        """Create a transfer with the scalar or register type of its handle."""
        handle = self.handles.get(reference)
        if not isinstance(handle, Handle):
            raise WireConversionError(
                "A quantum operand has no place in a known allocation."
            )
        kind = stmts.Unwrap if handle.scalar else stmts.UnwrapRegister
        return self.put(kind(reference)).results[0]

    def give(self, reference: ir.SSAValue, value: ir.SSAValue) -> None:
        """Restore a call or gate result to its source reference."""
        kind = stmts.Wrap if value.type == WireType else stmts.WrapRegister
        self.put(kind(reference, value))

    def items(self, value: ir.SSAValue, count: int) -> list[ir.SSAValue]:
        """Materialize one source reference per element of a known-length operand."""
        if isinstance(value.owner, ilist.New):
            return list(value.owner.values)
        items = []
        for i in range(count):
            index = self.put(py.Constant(i)).result
            item = self.put(py.GetItem(value, index))
            item.result.type = QubitType
            items.append(item.result)
        return items

    def gate(self) -> list[ir.Statement]:
        """Lower one source gate or expand its runtime broadcast into a loop."""
        node = self.before
        twin = TWINS[type(node)]
        trait = twin.get_present_trait(Twin)
        operands = [getattr(node, name) for name, _ in trait.wires]
        groups = [self.handles.items(value) for value in operands]
        for value, items in zip(operands, groups):
            if items is None and not isinstance(self.handles.get(value), Handle):
                raise WireConversionError(
                    f"A qubit operand of '{node.name}' has no place."
                )
        if any(items is None for items in groups):
            rule = BroadcastToLoop(
                {
                    value: None if items is None else len(items)
                    for value, items in zip(operands, groups)
                }
            )
            rule.rewrite(node)
            return rule.created
        known = [items for items in groups if items is not None]
        lengths = [len(items) for items in known]
        if not trait.truncate and len(set(lengths)) > 1:
            raise WireConversionError(
                f"'{node.name}' requires qubit lists of equal length."
            )
        for group in zip(*known):
            for i, a in enumerate(group):
                if not isinstance(a, Handle) or not a.scalar:
                    reason = (
                        a.reason
                        if isinstance(a, Unknown)
                        else "an unresolved reference"
                    )
                    raise WireConversionError(
                        f"A qubit of '{node.name}' has no place: {reason}."
                    )
                for b in group[i + 1 :]:
                    if not isinstance(b, Handle) or not distinct(a, b):
                        raise WireConversionError(
                            f"The operands of '{node.name}' may name one qubit."
                        )
        lists = [self.items(value, count) for value, count in zip(operands, lengths)]
        fields = getattr(twin, "__kirin_stmt_fields")
        wire_names = {wire for _, wire in trait.wires}
        classical = {
            name: getattr(node, name)
            for name in fields.std_args
            if name not in wire_names
        }
        attributes = {name: getattr(node, name) for name in fields.attributes}
        scalars = {name: [] for name, _ in trait.results}
        for group in zip(*lists):
            taken = {
                wire: self.put(stmts.Unwrap(reference)).wire
                for (_, wire), reference in zip(trait.wires, group)
            }
            made = self.put(twin(**classical, **taken, **attributes))
            for (_, wire), reference in zip(trait.wires, group):
                self.give(reference, getattr(made, f"{wire}_out"))
            for source, result in trait.results:
                scalars[source].append(getattr(made, result))
        for name, values in scalars.items():
            old = getattr(node, name)
            listed = self.put(
                ilist.New(values=tuple(values), elem_type=old.type.vars[0])
            )
            old.replace_by(listed.result)
        node.delete()
        return self.created

    def unpack(
        self, value: ir.SSAValue, kinds: Sequence[types.TypeAttribute]
    ) -> list[ir.SSAValue]:
        """Read each field of a wire call's result."""
        if len(kinds) <= 1:
            return [value][: len(kinds)]
        values = []
        for i, kind in enumerate(kinds):
            index = self.put(py.Constant(i)).result
            item = self.put(py.GetItem(value, index))
            item.result.type = kind
            values.append(item.result)
        return values

    def call(self, contract: Contract) -> list[ir.Statement]:
        """Lower a call using its checked result layout and borrowed parameter positions."""
        node = self.before
        assert isinstance(node, func.Invoke)
        inputs = [
            self.take(value) if i in contract.borrowed else value
            for i, value in enumerate(node.inputs)
        ]
        call = self.put(func.Invoke(tuple(inputs), callee=contract.method))
        call.result.type = packed_type(contract.outputs)
        values = self.unpack(call.result, contract.outputs)
        count = len(contract.ordinary)
        for i, value in zip(contract.borrowed, values[count:]):
            self.give(node.inputs[i], value)
        ordinary = []
        parameters = contract.method.callable_region.blocks[0].args[1:]
        mapping: dict[Sym, Affine] = {
            parameter: self.handles.indices.of(value)
            for parameter, value in zip(parameters, node.inputs)
            if parameter.type.is_subseteq(types.Int)
        }
        for value, kind, length in zip(
            values[:count], contract.ordinary, contract.lengths
        ):
            if kind == WireType:
                ref = self.put(stmts.Reference()).result
                self.give(ref, value)
                ordinary.append(ref)
            elif kind == RegisterType:
                length = None if length is None else length.substitute(mapping)
                count = None if length is None else length.constant
                count = None if count is None else max(0, count)
                ref = self.put(stmts.ReferenceRegister()).result
                ref.type = ilist.IListType[
                    QubitType, types.Any if count is None else types.Literal(count)
                ]
                if length is not None:
                    self.handles.extents[ref] = length
                self.give(ref, value)
                ordinary.append(ref)
            else:
                ordinary.append(value)
        if contract.packed:
            replacement = self.put(py.tuple.New(tuple(ordinary))).result
        elif ordinary:
            replacement = ordinary[0]
        else:
            replacement = self.put(func.ConstantNone()).result
        node.result.replace_by(replacement)
        node.delete()
        return self.created
