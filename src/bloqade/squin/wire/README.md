# Squin wire conversion

`SquinToWire` converts qubit references into explicit sequences of wire values.
A reference names a logical qubit or a list of logical qubits.
A wire names one version of a logical qubit during execution.
Each operation consumes its input wire and produces the next version.
The quantum state can span several entangled qubits.

## Package layout

`bloqade.squin.wire` defines the dialect, statements, types and traits.
`bloqade.squin.analysis.wire` computes the facts that promotion needs.
`bloqade.squin.analysis.validation.wire` checks the wire contracts.
`bloqade.squin.rewrite.wire` contains the transformation rules.
`bloqade.squin.passes` contains the conversion entry points.

```python
from bloqade.squin.passes.squin_to_wire import SquinToWire
from bloqade.squin.passes.wire_to_squin import WireToSquin
```

## The semantic contract

Let `origin(w)` name the logical qubit that a wire `w` continues.
Let `place(r)` name the logical qubits that a reference `r` accesses.
For each pair that `Threads` declares, the operation must satisfy:

```text
origin(output_wire) = origin(input_wire)
```

The operation still changes the quantum state according to its gate or channel.
For example, `Swap` exchanges quantum states while each output continues its corresponding input qubit.
Measurement produces a classical result and continues the measured qubit.

A twin is the wire form of one source operation on one group of qubits.
`TWIN_TABLE` declares its source fields, wire fields, classical results and broadcast length rule.
`make_twin` checks these declarations against Kirin's statement schema.
`WireToSquin` reconstructs the source operation with singleton qubit lists.
This reconstruction defines the twin's gate or channel semantics.

For example, lowering `squin.cx(a, b)` first produces:

```text
wa = Unwrap(a)
wb = Unwrap(b)
va, vb = wire.cx(wa, wb)
Wrap(a, va)
Wrap(b, vb)
```

`Unwrap` transfers access from a reference to a wire.
`Wrap` restores access to that reference.
Promotion connects the wire values between operations and removes these temporary transfers.

## Reference analysis

`QubitReferenceAnalysis` computes the identity of each qubit reference with Kirin's `Forward` analysis.
A place combines an allocation or parameter with a path through its registers.
For example, `qs[i + 1]` identifies the register `qs` and the index `i + 1`.
The index analysis represents integer expressions as sums of symbols with constant integer coefficients.
This representation proves that `i` equals `i + 0` and differs from `i + 1`.

The analysis follows aliases, slices, literal containers, calls, closures and structured control flow.
Each run clears cached index facts because rewrites can change the expressions.
If the analysis cannot identify a tracked reference, it returns `Unknown`.
An unknown reference cannot establish that two gate operands name distinct qubits.

`SlotAnalysis` adds the information that promotion needs.
A slot is a qubit or register that promotion can carry as one linear value.
The analysis records which slots each loop and branch accesses.
A blocking operation prevents promotion of every root that it can access.
If a blocking operand has an unknown reference, the analysis prevents promotion of every root.

Every simultaneous gate operand requires a proof that it names a distinct qubit.
Every simultaneous register borrow requires a proof that its index differs from outstanding borrows.
Missing reference facts cause conversion to fail.
Public wire functions require distinct quantum arguments.
Preparation checks each retained call against that convention.
If a reference call can alias its arguments, preparation inlines it and checks the resulting operations.

## Pass contracts

| Stage | Kirin mechanism | Required result |
| --- | --- | --- |
| Prepare | `Inline`, `RewriteRule`, `Walk`, `CheckedFixpoint`, `TypeInfer`, reference analysis | Calls expose supported reference behavior, and unknown-length broadcasts use loops. |
| Lower | `LowerTwins`, `LowerCalls`, `LowerLength`, `LowerReturn`, `LowerEntry` | Quantum operations use wire twins between explicit reference transfers. |
| Promote | `SlotAnalysis`, `PromotionAnalysis`, `RewriteRule`, `Walk` | A read-only plan connects eligible allocations through wire values, block arguments and results. |
| Check conversion | `ConversionTarget` with `Walk` | Complete conversion contains supported operations and excludes qubit reference types. |
| Validate | Kirin `ValidationSuite` and forward analyses | The result satisfies structure, types, linear uses, borrow balance, positions and reference access rules. |

## Promotion algorithm

Promotion converts each eligible allocation into a sequence of explicit wire values.
A register remains one allocation, including when every access uses a constant index.
`plan_promotion` computes all decisions before `apply_promotion` changes the program.

1. `SlotAnalysis` identifies each reference's allocation and records the accesses inside each structured region.
2. `Eligibility` checks each allocation's accesses and excludes allocations with unsupported uses or paths.
3. `PromotionAnalysis` follows execution order and records the current value of each selected allocation.
4. `LoopArguments` and `ApplyPromotion` create the planned values and replace the reference transfers.
5. Cleanup removes unused reference expressions, invariant reference arguments and the selected reference allocations.

An unsupported use prevents promotion of its whole allocation.
An unknown blocking reference prevents promotion of every allocation.
Partial conversion preserves these references and can still promote independent allocations.
Complete conversion rejects any references that remain.

### The forward state

For each allocation `r`, the analysis records `Contents(value, borrowed)`.
`value` names the current wire or register.
`borrowed` contains the register indices that currently belong to separate wires.
A missing `value` records a whole-allocation transfer from reference access to explicit wire operations.

| Input operation | Required state | Planned operation and next state |
| --- | --- | --- |
| Allocation | The allocation starts here. | Create a wire allocation at this position and record its value. |
| Whole `Unwrap(r)` | `r` holds a value with every element present. | Replace the result with that value and mark `r` empty. |
| Whole `Wrap(r, w)` | `r` is empty. | Record `w` as its current value. |
| Element `Unwrap(r[i])` | `r` holds a register and `i` differs from every outstanding borrow. | Insert `Borrow`, record the remaining register, and add `i` to `borrowed`. |
| Element `Wrap(r[i], w)` | `borrowed` contains `i`. | Insert `Release`, record its register result, and remove `i` from `borrowed`. |

A gate keeps its existing wire operands and results throughout promotion.
The rewrite substitutes planned values into the operands that previously came from `Unwrap`.
Synthetic reference initializers for parameters and call results start empty.
Their following `Wrap` operations supply the existing wire values.

At each region exit, the plan frees local allocations that the region still owns.
A transfer to a return value removes that allocation from the region's ownership.
Allocations inside a loop remain inside that loop.
Promotion preserves unused allocations and their frees.
It performs no register splitting or lifetime shortening.

### Loops and branches

A loop carries each accessed outer allocation as an initializer, a body argument and a result.
The analysis visits the body once with symbolic values for those new arguments.
The body's final values become the extra yield operands.
The loop's results become the current values after the loop.
For zero iterations, the loop returns the initializers.

Both branches start with the same incoming state.
Each branch yields its final value at the same result position for each accessed outer allocation.
The branch results become the current values after the conditional.
Every register must contain all its borrowed elements at these boundaries.

For example, this pseudocode gives a loop's explicit register connections:

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

`Definition` names each future value while the plan has no corresponding IR node.
Planning creates no IR nodes and changes no SSA use lists.
SSA means that each program value has one definition.
Apply a plan once, immediately after analysis, to its unchanged source method.
The driver performs no intervening rewrites.

### Local rewrites and method-wide decisions

The rewrite of one transfer uses a precomputed replacement.
Finding that replacement requires the allocation's earlier transfers, aliases and enclosing control flow.
A branch can supply different reaching definitions, which are the values available at the end of each path.
A loop must connect its final value to the next iteration's argument.
These decisions require information beyond the statement that the rewrite replaces.

`PromotionAnalysis` uses Kirin's `ForwardExtra` interpreter and dialect method tables.
Its supported regions contain one block and use `scf.For` or `scf.IfElse` with `scf.Yield` terminators.
The analysis visits each structured body once because symbolic region arguments represent repeated execution.
`ApplyPromotion` uses Kirin's `Walk` to apply the plan after loop arguments exist.
The walker rewrites each body before rebuilding its enclosing operation.
Cleanup uses `CheckedFixpoint` and has no role in choosing wire connections or lifetimes.

## Validation contracts

| Check | Required property |
| --- | --- |
| `StructureValidation` | Every method passes Kirin's structure and type checks, including partial conversions. |
| `LinearityValidation` | Each wire, register and quantum tuple field has one use on each execution path. |
| `BorrowValidation` | Each release matches a borrowed index, and registers leave each boundary with every slot restored. |
| `PositionValidation` | Loop yields and call return suffixes continue their inputs in order, and releases restore the original qubits. |
| `ReferenceValidation` | Reference operations respect the outstanding transfers between references and wires. |

Borrow validation represents outstanding slots with a set of normalized index expressions.
An unmatched release causes an error.
Two symbolic indices require a separation proof before their borrows can overlap.
An unknown set cannot establish that a register holds every qubit.
Both branches must continue the same qubit at each wire result position.
A function returns each borrowed parameter in a suffix, in parameter order.

Quantum tuple fields require constant indices within their bounds.
Tuple construction, extraction, calls and region boundaries must preserve quantum field types.
Function parameters and region state carry separate wire or register values.
Function results can use a flat tuple of ordinary outputs followed by borrowed parameters.
An ordinary quantum result must own a whole allocation distinct from all other results and borrowed parameters.
Preparation inlines calls that return aliases of their borrowed parameters.
A public entry point that returns such an alias causes conversion to fail.

Complete conversion checks every prepared method, including its nested regions.
The conversion target rejects source quantum operations, reference transfers and reference types inside containers or signatures.
It also rejects unresolved calls and operations from unsupported dialects.
`partial=True` keeps unsupported reference operations and reports them.
Partial conversion still requires every validation check to pass.

## Public pass boundaries

`SquinToWire.emit` copies the source and its retained callees before changing their programs.
Preparation checks structure and types after normalization.
The converter computes every call layout before it changes any signature.
Lowering records reference facts for each reference that a call result creates.
Tuple projection resolves the temporary tuples that lowering constructs.
Promotion computes a read-only plan after lowering finishes its structural changes.
The final target check and validation suite cover every converted function.
A failed conversion leaves the source methods unchanged.
Repeated calls clear the converter's preparation state and diagnostics.

`CheckedFixpoint` reports an error if rewriting reaches its iteration limit.
The default limit is 32 iterations.
Preparation also has a limit of 32 iterations, which `max_iterations` can change.
The call-graph check rejects recursion before the forward analyses run.

`WireToSquin.emit` validates its input methods before applying reconstruction rules.
It checks the structure and types of every reconstructed method.
Borrow and release operations belong to the `squin.wire` dialect.
They describe register ownership during conversion.

The pipeline uses common-subexpression elimination and dead-code elimination for conversion cleanup.
It has no gate cancellation, angle merging or qubit reuse pass.

## Direct Jeff emission

```python
from bloqade.jeff import SquinToJeff, WireToJeff

wired = SquinToWire().emit(source)
jeff = WireToJeff().emit(wired)
jeff_from_source = SquinToJeff().emit(source)
```

`SquinToJeff` runs `SquinToWire`, then `WireToJeff`.
`WireToJeff` requires complete wire conversion and validates every reachable wire method.
`WireEmitter` uses Kirin emission method tables to map each explicit wire value to a Jeff value.
It maps allocations, frees, borrows, releases, measurements and gates directly.
A call preserves the ordinary outputs and the borrowed-parameter suffix.
Loops carry explicit quantum state through block arguments and yields.
Branches yield explicit quantum state at matching result positions.
`IsolateRegions` adds explicit region inputs for captured values with Kirin rewrite rules.
Jeff structure and linearity validation check each emitted function.

Jeff gate angles use radians.
Squin gate statements use turns, so the emitter multiplies them by `2*pi`.
The emitter also maps the squin control-first operand order to Jeff's target-first operand order.
The output mapping restores the declared wire result order.

Wire registers preserve runtime lengths but currently erase static length annotations.
The emitter rejects noise channels, qubit-loss tests and gates without a registered Jeff form.
It rejects dynamic list construction, tuple-valued region state and runtime tuple indices.
Jeff cannot convert runtime integers to floating-point values.
The emitter reports unsupported mixed arithmetic and integer true division.
It rejects unary minus and inversion on booleans because those operations require Python integer semantics.
`SquinToJeffValidation` remains a standalone reference-form checker for the original restrictions.
The current converter uses the wire checks and its emission method tables.

## Relationship to MLIR and MQT

[MLIR dialect conversion](https://mlir.llvm.org/docs/DialectConversion/) defines a target that decides which operations may remain.
`ConversionTarget` gives complete wire conversion that explicit acceptance rule.
[MLIR mem2reg](https://mlir.llvm.org/docs/Passes/#-mem2reg) promotes memory slots through operation interfaces.
`ReferenceAccess` supplies the corresponding access contract for this converter.
The squin reference analysis identifies the slots behind Python aliases and container expressions.

[MQT's QCO utilities](https://github.com/munich-quantum-toolkit/core/blob/66968a5f44d2b0eddb108b6c7c5c49c463c57205/mlir/lib/Dialect/QCO/Utils/FunctionUtils.cpp)
check tensor restoration and trace wire origins through operations.
The wire validators use these same requirements for register boundaries and positional continuity.
[MQT's gate table](https://github.com/munich-quantum-toolkit/core/blob/66968a5f44d2b0eddb108b6c7c5c49c463c57205/mlir/include/mqt/Conversion/GateTable.def)
declares supported gate conversions explicitly.
`TWIN_TABLE` declares the squin and native conversions explicitly.

## Verification and limits

Run the conversion and reference tests with:

```sh
uv run --locked pytest -q test/squin/wire test/analysis/reference test/jeff
```

The tests compare source execution with execution after conversion and reconstruction.
A separate dense-matrix interpreter executes wire operations directly in the tests.
Its unitary and noisy results must match independently constructed Cirq circuits.
Measurement tests enumerate outcomes and check their exact probabilities, including conditional gates.
Reset tests check the mixed state of an entangled partner.
Direct Jeff tests check emission without invoking either reference conversion pass.
They also check Jeff serialization and execution after reconstruction.
Reference tests also compare computed places with qubit identities from simulator runs.
Malformed-program tests check that validators reject missing qubits, reordered outputs and unsupported operations.
These tests cover concrete programs and mutations.
A proof of semantic preservation for every accepted program remains separate work.

The recursive reference-analysis tests require Kirin's newer recursive fixpoint support.
Conversion rejects recursive input explicitly with the current Kirin version.
The conversion tests have no expected-failure markers.
