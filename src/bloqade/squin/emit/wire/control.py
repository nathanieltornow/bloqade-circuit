"""Emit structured regions once with explicit incoming and outgoing wire values."""

from kirin import ir, interp
from kirin.dialects import scf

from bloqade.squin.wire.errors import WireConversionError

from .base import EmitWire, WireFrame
from .values import contains_location


@scf.dialect.register(key="emit.squin_wire")
class ControlFlow(interp.MethodTable):
    """Construct loop and branch bodies with separate current-wire maps."""

    @interp.impl(scf.Yield)
    def yield_(self, emit: EmitWire, frame: WireFrame, stmt: scf.Yield):
        """Return source meanings to the region handler that constructs the output yield."""
        return interp.YieldValue(frame.get_values(stmt.values))

    @interp.impl(scf.For)
    def loop(self, emit: EmitWire, frame: WireFrame, stmt: scf.For):
        """Convert the body once and carry all available outer wires through the loop."""
        initial = frame.get_values(stmt.initializers)
        positions = tuple(
            i for i, value in enumerate(initial) if not contains_location(value)
        )
        classical = tuple(emit.classical(frame, initial[i]) for i in positions)
        roots = tuple(frame.current)
        incoming = tuple(frame.current[root] for root in roots)
        iterable = emit.classical(frame, frame.get(stmt.iterable))
        with emit.new_frame(stmt, has_parent_access=True) as body:
            body.current_stmt = stmt
            arguments = list(initial)
            index = body.output.args.append_from(
                stmt.body.blocks[0].args[0].type, stmt.body.blocks[0].args[0].name
            )
            for position, value in zip(positions, classical, strict=True):
                source = stmt.body.blocks[0].args[position + 1]
                arguments[position] = body.output.args.append_from(
                    value.type, source.name
                )
            wires = tuple(
                body.output.args.append_from(value.type) for value in incoming
            )
            body.current = dict(zip(roots, wires, strict=True))
            provisional = body.push(scf.Yield(*body.output.args[1:]))
            made = scf.For(iterable, ir.Region(body.output), *classical, *incoming)
            made.attributes.update(stmt.attributes)
            frame.push(made)
            provisional.delete()
            yielded = emit.frame_call_region(body, stmt, stmt.body, index, *arguments)
            if not isinstance(yielded, tuple) or len(yielded) != len(initial):
                raise WireConversionError(
                    "A loop must yield its source state in initializer order.", stmt
                )
            for i, value in enumerate(initial):
                if i not in positions and value != yielded[i]:
                    raise WireConversionError(
                        "Cannot resolve a value carried by a loop or branch.", stmt
                    )
            outputs = tuple(emit.classical(body, yielded[i]) for i in positions)
            body.finish()
            body.push(scf.Yield(*outputs, *(body.current[root] for root in roots)))
        for result, value in zip(
            made.results,
            (*outputs, *(body.current[root] for root in roots)),
            strict=True,
        ):
            result.type = value.type
        frame.current.update(zip(roots, made.results[len(positions) :], strict=True))
        result = list(initial)
        for position, value in zip(
            positions, made.results[: len(positions)], strict=True
        ):
            result[position] = value
        return tuple(result)

    @interp.impl(scf.IfElse)
    def branch(self, emit: EmitWire, frame: WireFrame, stmt: scf.IfElse):
        """Convert both paths from the same incoming state and join their explicit wire results."""
        condition = emit.classical(frame, frame.get(stmt.cond))
        roots = tuple(frame.current)
        bodies, yields = [], []
        for region in (stmt.then_body, stmt.else_body):
            with emit.new_frame(stmt, has_parent_access=True) as body:
                body.current_stmt = stmt
                arguments = tuple(
                    body.output.args.append_from(arg.type, arg.name)
                    for arg in region.blocks[0].args
                )
                body.current = dict(frame.current)
                yielded = emit.frame_call_region(body, stmt, region, *arguments)
                if not isinstance(yielded, tuple):
                    raise WireConversionError("A branch must end with a yield.", stmt)
                bodies.append(body)
                yields.append(yielded)
        left, right = yields
        if len(left) != len(right):
            raise WireConversionError("Branch results must have matching counts.", stmt)
        positions = []
        results = list(left)
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if contains_location(a) or contains_location(b):
                if a != b:
                    raise WireConversionError(
                        "A branch reference has no place in one known allocation.", stmt
                    )
            else:
                positions.append(i)
        for body, yielded in zip(bodies, yields, strict=True):
            outputs = tuple(emit.classical(body, yielded[i]) for i in positions)
            body.finish()
            body.push(scf.Yield(*outputs, *(body.current[root] for root in roots)))
        made = scf.IfElse(
            condition, ir.Region(bodies[0].output), ir.Region(bodies[1].output)
        )
        made.attributes.update(stmt.attributes)
        frame.push(made)
        frame.current.update(zip(roots, made.results[len(positions) :], strict=True))
        for position, value in zip(
            positions, made.results[: len(positions)], strict=True
        ):
            results[position] = value
        return tuple(results)
