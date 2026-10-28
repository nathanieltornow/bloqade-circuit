# Squin wire conversion

`SquinToWire` converts qubit references into explicit sequences of wire values.
A reference names a logical qubit or a list of logical qubits.
A wire names one version of a logical qubit during execution.
Each operation consumes its input wire and produces the next version.

```python
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin

wired = SquinToWire().emit(source)
reconstructed = WireToSquin().emit(wired)
```

## Package layout

| Package | Responsibility |
| --- | --- |
| `squin.wire` | Define the dialect, statements, types and operation traits. |
| `squin.rewrite.wire` | Normalize source operations and construct explicit wire values. |
| `squin.analysis.validation.wire` | Check structure, ownership and qubit continuity. |
| `squin.passes` | Run conversion, cleanup and validation on private method copies. |
| `squin.analysis.wire` | Supply reference-analysis registrations for other consumers of the wire dialect. |

The converter and wire validators never run `ReferenceAnalysis`, `QubitReferenceAnalysis` or `SlotAnalysis`.
The conversion tests replace `ReferenceAnalysis.run` with a function that fails to enforce this boundary.

## Transfers and gate semantics

Let `origin(w)` name the logical qubit that wire `w` continues.
For each input and output pair that `Threads` declares, the operation must satisfy:

```text
origin(output_wire) = origin(input_wire)
```

The operation changes the quantum state according to its gate or channel.
`Swap` exchanges quantum states while each output continues its corresponding input qubit.
Measurement produces a classical result and continues the measured qubit.

A twin is the wire form of one source operation on one group of qubits.
`TWIN_TABLE` declares source fields, wire fields, classical results and broadcast length rules.
`make_twin` checks these declarations against Kirin's statement schema.
`WireToSquin` reconstructs the source operation with singleton qubit lists.
This reconstruction defines each twin's gate or channel semantics.

The local lowering of `squin.cx(a, b)` constructs:

```text
wa = Unwrap(a)
wb = Unwrap(b)
va, vb = wire.cx(wa, wb)
Wrap(a, va)
Wrap(b, vb)
```

`Unwrap` transfers access from a reference to a wire.
`Wrap` restores access to that reference.
The ordered converter immediately resolves these transfers against its current wire values.
Complete conversion requires every reference transfer to disappear.
`SquinToWire.lower` exposes the executable transfer form for inspection and testing.

## Conversion state

`Handles` records the meaning of each source value when the converter visits its definition.
A handle contains an allocation identity and an optional register index or slice.
Literal lists and tuples contain descriptors for their members.
Nested register parameters use one flat register with their rows in source order.
Their handles retain the source dimensions so indexing can compute each element's flat position.
Aliases reuse their operand's descriptor.
An unresolved descriptor retains every possible allocation identity when those identities are known.

For example, both `a = qs[1]` and `b = a` identify element 1 of allocation `qs`.
A gate changes the current wire for that allocation.
The handles for `a` and `b` keep the same identity.

The region's `current` map stores the latest wire or register value for each allocation.
Each entry also records the register indices that currently have outstanding borrows.
An empty current value means that an explicit wire temporarily owns the whole allocation.
The region also records its local allocations so it can free them at exit.

The shared index utility represents integer expressions as sums of symbols with constant integer coefficients.
It proves that `i` equals `i + 0` and differs from `i + 1`.
The converter uses this arithmetic for indices, slices and call result lengths.
The utility's module lives under `analysis.reference.index`.
It never invokes the reference-analysis interpreter.

## Pass contracts

| Stage | Kirin mechanism | Required result |
| --- | --- | --- |
| Normalize | `Inline`, `RewriteRule`, `Walk`, `CheckedFixpoint`, `TypeInfer` | Expose source gates and normalize closures, maps and selected indices. |
| Convert | `OrderedPromotion`, `RewriteRule`, `Walk` | Resolve definitions, lower operations locally and thread current values through structured regions. |
| Clean up | Tuple projection, dead-code elimination and common-subexpression elimination | Remove dead source references and redundant conversion scaffolding. |
| Check | `ConversionTarget`, `ValidationPass`, `ValidationSuite` | Enforce complete conversion and every wire ownership contract. |

The converter applies these actions in source order:

1. Record allocation and alias definitions in `Handles`.
2. Construct local transfers around each supported gate or call.
3. Resolve each transfer with the current allocation value.
4. Convert each structured body with its own current-value map.
5. Construct the return values and free the remaining local allocations.

`Walk` visits one region at a time.
The rule for an enclosing loop or conditional owns the visit to its nested bodies.
Local lowerings return their newly constructed statements in construction order.
The converter visits that list immediately because Kirin's walker snapshots its original worklist.

A whole-allocation unwrap consumes the current value and marks the entry empty.
A matching wrap stores the next current value.
An element unwrap emits `Borrow` and records its index as outstanding.
An element wrap emits `Release` and removes that outstanding index.
Every simultaneous operand needs a proof that it names a distinct qubit.
Every simultaneous register borrow needs a proof that its index differs from outstanding indices.

Unknown-length broadcasts become loops during the same conversion visit.
Each iteration lowers a singleton broadcast through the same gate rule.
Measurement broadcasts carry their classical result list through the loop.

## Loops and branches

The loop rule carries every available outer allocation through extra loop parameters.
This includes allocations that the loop leaves unchanged.
The rule creates these parameters before visiting the loop body.
The body starts with the parameters as its current values.
Its final current values become extra yield operands.
The loop results become the enclosing region's current values.
For zero iterations, the loop returns its initial values.

```text
r0 = AllocRegister(n)
r_after = For(indices, initial=r0):
    r_iter = iteration_argument
    r_hole, q0 = Borrow(r_iter, i)
    q1 = X(q0)
    r_next = Release(r_hole, i, q1)
    Yield(r_next)
FreeRegister(r_after)
```

Each branch starts with a separate copy of the incoming current values.
Both branches yield each outer allocation at the same result position.
The conditional's results become the enclosing region's current values.
Every register must contain all its borrowed elements at these boundaries.

Source reference variables must retain their allocation identity across a loop.
Normalization can turn a changing element reference into a changing integer index.
If an allocation identity still changes, complete conversion fails.
Partial conversion preserves every possible source allocation and joins the descriptors conservatively.
Each structured region must contain one block with the expected return or yield terminator.

## Calls and returns

The converter converts a callee when it first encounters a call that can retain its boundary.
A successful conversion caches a `Contract` with input types, ordinary outputs and borrowed parameter positions.
The contract also records known expressions for fresh register lengths.
Later calls reuse this converted method and contract.

Every quantum parameter borrows its input allocation.
The function returns the latest version of each borrowed parameter in a suffix, in parameter order.
Ordinary results come before that suffix.
A fresh ordinary quantum result must own one whole allocation.
It must differ from every other result and borrowed parameter.

For a source call `answer = f(q, n)`, the wire convention is:

```text
q0 = Unwrap(q)
answer, q1 = call f_wire(q0, n)
Wrap(q, q1)
```

The caller transfers its current values into the call.
The returned suffix restores those values at the same positions.
Each fresh quantum result creates a new caller-side allocation identity.
Two calls to the same allocating function create distinct identities.

Quantum call operands must name separate scalar qubits or whole registers.
If arguments overlap, use views or contain unresolved containers, the converter inlines the call and retries a private copy.
A callee that returns a borrowed alias also requires inlining.
A public entry point that returns a borrowed alias fails conversion.
The call-graph check rejects recursion.

## Complete and partial conversion

`SquinToWire.emit` performs complete conversion by default.
The conversion target rejects surviving source quantum operations, reference transfers and reference types in signatures or containers.
It also rejects unresolved calls and operations from unsupported dialects.
Every retained callee must satisfy the same target.

`SquinToWire(partial=True)` can preserve unsupported reference operations.
Its `report` lists the reasons and remaining conversion violations.
If a late use prevents promotion, the converter retries a private copy with the affected allocations preserved.
If an unresolved use can name any allocation, the retry preserves every allocation in the function.
A gate that touches a preserved allocation preserves all its operand allocations.
Partial conversion still runs every wire validation pass.

The ordered rewrite visits each normalized statement once during one conversion attempt.
Normalization, required inlining, partial retries, cleanup and validation can each perform additional traversals.
`max_iterations` limits normalization and conversion retries to 32 by default.
A failure leaves the source method and its source callees unchanged.
Repeated calls clear the converter's contracts and diagnostics.

## Validation contracts

| Check | Required property |
| --- | --- |
| `StructureValidation` | Kirin verifies method structure and types. |
| `LinearityValidation` | Each wire, register and quantum tuple field has one use on each execution path. |
| `BorrowValidation` | Releases match borrowed indices and registers restore every element before crossing a boundary. |
| `PositionValidation` | Releases, loop yields, branches and call return suffixes continue the correct qubits in order. |
| `ReferenceValidation` | Source operations respect outstanding transfers between references and wires. |

Linearity and borrow validation use Kirin forward analyses with operation method tables.
Borrow validation tracks outstanding register indices in its `Slots` lattice.
This validation state describes holes in an explicit wire register.

Position and reference validation use the ordered `Origins` rule with Kirin's `Walk`.
The rule follows allocation identities, operation continuations, calls and structured boundaries.
It checks each callee and gives every call's fresh outputs distinct identities.
The reference check rejects source accesses to qubits that currently travel as explicit wires.

Quantum tuple fields require constant indices within their bounds.
Tuple construction, extraction, calls and region boundaries must preserve quantum field types.
Function parameters and region state carry separate wire or register values.
Function results can use a flat tuple of ordinary outputs followed by borrowed parameters.

## Direct Jeff emission

```python
from bloqade.jeff import SquinToJeff, WireToJeff

wired = SquinToWire().emit(source)
jeff = WireToJeff().emit(wired)
jeff_from_source = SquinToJeff().emit(source)
```

`SquinToJeff` runs `SquinToWire`, then `WireToJeff`.
`WireToJeff` requires complete wire conversion and validates every reachable wire method.
Its Kirin emission method tables map explicit wire values to Jeff values.
The emitter preserves the borrowed-parameter suffix and explicit loop state.
`IsolateRegions` adds explicit region inputs for captured values with Kirin rewrite rules.
Jeff structure and linearity validation check each emitted function.

Jeff gate angles use radians.
Squin gate statements use turns, so the emitter multiplies them by `2*pi`.
The emitter maps squin's control-first operand order to Jeff's target-first operand order.
It restores the declared wire result order after emission.

Wire registers preserve runtime lengths but erase static length annotations.
The Jeff emitter rejects noise channels, qubit-loss tests and gates without a registered Jeff form.
It rejects dynamic list construction, tuple-valued region state and runtime tuple indices.
It also rejects arithmetic that requires integer-to-float conversion in Jeff.

## Verification and limits

```sh
uv run --locked pytest -q test/squin/wire test/analysis/reference test/jeff
```

The tests compare source execution with execution after conversion and reconstruction.
A separate dense-matrix interpreter executes wire operations directly in the tests.
Its unitary and noisy results must match independently constructed Cirq circuits.
Measurement tests enumerate outcomes and check their exact probabilities, including conditional gates.
Malformed-program tests exchange qubits and break ownership contracts to check validator rejection.
Tests also disable reference analysis during conversion, validation, reconstruction and direct Jeff emission.

Complete conversion can reject a valid source program when it cannot prove operand separation or resolve its reference expressions.
Public quantum parameters require distinct input qubits.
Recursion and arbitrary control-flow graphs exceed the current conversion contract.
The conversion tests have no expected-failure markers.
The pipeline has no gate cancellation, angle merging or qubit reuse pass.
