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

## Compilation stages

| Stage | Kirin mechanism | Result |
| --- | --- | --- |
| Check source | `ValidationSuite`, `Forward`, `interp.MethodTable` | Check source structure and types, then collect static callees. |
| Emit | `EmitABC`, `EmitFrame`, `interp.MethodTable`, `ssacfg.Concrete` | Build a fresh wire method from the source method. |
| Clean up | `Fixpoint`, `Walk`, `DeadCodeElimination` | Delete unused classical expressions, including source slice objects. |
| Infer types | `TypeInfer` | Resolve types of emitted classical operations and specialized library operations. |
| Check output | `ConversionTarget`, wire validation passes | Require complete wire conversion and valid ownership. |

`SquinToWire.emit()` reads the source method and constructs a separate output method.
It converts user callees first and supplies their checked contracts to the caller's emitter.
A contract records input types, ordinary result types and borrowed parameter positions.
It also records known fresh-register lengths in terms of the callee's integer parameters.
Each callee must pass the same conversion target and validation.

`SourceValidation` checks source structure, types, region forms and static calls before emission.
Emission visits each source statement once per method conversion.
Kirin's region interpreter controls statement order and fills the frame after each operation.
Standard library wrappers execute through nested emission frames with shared output and wire state.
Each library call interprets its wrapper body once.
Validation, type inference and cleanup perform their own traversals.
The full pipeline has no established linear-time bound.
The pinned Kirin version requires literal tuple-index hints during output type inference.

Unsupported source forms stop conversion at their first failed condition.
The converter preserves the source and propagates the error without retrying through inlining.

## Emission state

`WireFrame` extends Kirin's `EmitFrame` with an output block and allocation state.
Its inherited entries map source SSA values to their emitted meanings.
An SSA value names one compiler value with one definition.

| State | Meaning |
| --- | --- |
| `entries` | Map each source value to an emitted classical value, quantum location or container of values. |
| `current` | Map each allocation identity to its latest wire or register value. |
| `local_roots` | Record allocations that this region must free unless the function returns them. |
| `output` | Hold the emitted block and its arguments. |

A `Location` identifies an allocation and an optional element or slice.
Its root is the emitted allocation result or a function parameter.
Its shape records nested register dimensions.
Its length records a known allocation size.
The frame keeps locations stable while `current` changes after each quantum operation.
An alias copies its operand's frame entry.
An index selects from that entry using the already emitted integer operand.
A source list holds its element locations in a tuple.

Emission uses these frame entries for all source reference resolution.
Each reference operation reads the meanings of its operands from that frame.
The shared `analysis.indices` utility reads integer arithmetic from emitted values.
It represents expressions such as `i + 1` and proves simple index separation.
Runtime register lengths come from wire `Length` operations.
Nested register parameters use one flat register with their rows in source order.

The operation tables live in `squin.emit.wire`:

| File | Responsibility |
| --- | --- |
| `base.py` | Define the emitter, frame and operations on current wires. |
| `methods.py` | Emit functions and calls, and interpret source reference expressions. |
| `control.py` | Convert loop and branch regions through Kirin's region interpreter. |
| `gates.py` | Emit wire gates and runtime broadcast loops. |
| `values.py` | Define locations and compose element and slice selections. |

## Gates and allocation lifetimes

Each quantum operation consumes its input wire and produces the next version of that qubit.
`Threads` declares which output continues each input.
`Swap` exchanges quantum states while each output continues its corresponding input qubit.
Measurement produces a classical result and continues the measured qubit.

A twin is the wire form of one source operation on one group of qubits.
`TWIN_TABLE` declares source fields, wire fields, classical results and broadcast length rules.
`make_twin` checks these declarations against Kirin's statement schema.
`WireToSquin` reconstructs the source operation with singleton qubit lists.

A scalar gate handler gets each operand location from the frame.
For a standalone qubit, it consumes the current wire and records the output wire.
For a register element, it emits `Borrow`, the gate and `Release`.
`Borrow` produces the selected wire and a register with that element temporarily unavailable.
`Release` restores the updated wire to that register element.

```text
r0 = AllocRegister(2)
r1, q0 = Borrow(r0, 0)
q1 = X(q0)
r2 = Release(r1, 0, q1)
FreeRegister(r2)
```

Allocation handlers record their roots in the current frame.
At region exit, the frame frees each local allocation whose current value remains available.
Returning a fresh allocation consumes that value before cleanup.

A known-length broadcast emits one scalar gate per operand group.
An unknown-length broadcast emits a loop with the same scalar gate handler.
Measurement loops carry their classical result lists alongside the wire registers.
Equal-length broadcasts check their operand lengths before entering the loop.
Operations with shortest-list semantics use the minimum length.

## Loops and branches

The loop handler reads every available outer allocation from `current`.
It creates output parameters for classical loop state and those allocations.
It attaches the output loop before interpreting its body so integer indices can use the loop's range.
The body's frame starts with those parameters as its current wire values.
Kirin interprets the source body once.
The handler then emits a yield with the final classical values and current wires.
The loop results replace the enclosing frame's current values.
For zero iterations, the loop returns its initial values.

```text
r_after = For(indices, initial=r0):
    r_iter = iteration_argument
    r_hole, q0 = Borrow(r_iter, i)
    q1 = X(q0)
    r_next = Release(r_hole, i, q1)
    Yield(r_next)
```

Each branch starts with a separate copy of the incoming current-wire map.
The handler interprets both source regions once.
Both output regions yield the incoming allocations in the same order.
The conditional's results become the enclosing frame's current values.
Every register must contain all its borrowed elements at these boundaries.

Source reference variables must retain their allocation identity across a loop.
Both branches must return the same location for a quantum reference result.
If a reference changes its allocation or element across a boundary, conversion fails.
A program can express changing element selections with a classical integer index.
Each structured region must contain one block with its expected terminator.

## Calls and returns

Every quantum parameter borrows its input allocation.
The function returns the latest value of each borrowed parameter in declaration order.
Ordinary results precede this borrowed-parameter suffix.
A fresh ordinary quantum result must own one whole allocation.
It must differ from every other result and borrowed parameter.

For a source call `answer = f(q, n)`, the wire convention is:

```text
answer, q1 = call f_wire(q0, n)
```

The caller consumes its current inputs and restores them from the borrowed-parameter suffix.
Calls on register elements use `Borrow` and `Release` around the call.
Each fresh quantum result creates a new allocation identity in the caller.
Two calls to the same allocating function create distinct identities.

Quantum call operands must name separate scalar qubits or whole registers.
Calls with register slices, quantum containers or overlapping quantum arguments fail.
A returned borrowed alias also fails.
`UnsupportedCallContract` identifies unsupported argument and result layouts.
`UnresolvedArgumentSeparation` identifies quantum operands whose separation lacks proof.
Callee validation failures propagate directly to the caller.
Each conversion error retains its operation, source location and failed condition.

The conversion driver rejects recursive calls.
A failed conversion leaves the source method and its callees unchanged.
Repeated public conversions clear the method and contract caches.

## Validation contracts

| Check | Required property |
| --- | --- |
| `StructureValidation` | Kirin verifies method structure and types. |
| `LinearityValidation` | Each wire, register and quantum tuple field has one use on each execution path. |
| `BorrowValidation` | Releases match borrowed indices and registers restore every element before crossing a boundary. |
| `PositionValidation` | Releases, loop yields, branches and call return suffixes continue the correct qubits in order. |

Linearity and borrow validation use Kirin forward analyses with operation method tables.
Borrow validation tracks outstanding register indices in its `Slots` lattice.
This validation state describes holes in an explicit wire register.

Position validation uses Kirin's `Forward` analysis with operation method tables.
Its `Origin` lattice stores exact allocation identities and tuple fields in Kirin frames.
An unknown origin records insufficient information to prove continuity.
The analysis follows operation continuations and checks loop yields, branch results and borrowed return suffixes.
It checks each callee and gives every call's fresh outputs distinct identities.

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

## Jeff import through wires

```python
from bloqade.jeff import JeffToWire

imported = JeffToWire().emit(jeff)
reconstructed = WireToSquin().emit(imported)
```

`JeffToSquin` runs this same route through `JeffToWire` and `WireToSquin`.
Jeff import validates the source and converts private copies of the reachable functions.
Local rewrite rules translate classical operations, structured regions and quantum operations.
The converter rejects recursive call graphs.

Imported functions use consuming quantum parameters.
Their `squin.wire.owned` marker requires explicit quantum outputs and permits changes in result positions.
The wire validators still enforce linear uses and correct register ownership.
Source squin conversion retains its borrowed-parameter convention.

`Take` and `Put` move a qubit out of a register and fill an empty register element.
`TakeSlice` and `PutSlice` move contiguous register ranges.
`PackRegister`, `SplitRegister` and `JoinRegister` construct and partition registers.
`DiscardRegister` frees the elements that remain in a register.
An outstanding `Borrow` requires a matching `Release` before discard.

The complete round-trip tests run every conversion stage explicitly:

```python
wired = SquinToWire().emit(source)
jeff = WireToJeff().emit(wired)
imported = JeffToWire().emit(jeff)
reconstructed = WireToSquin().emit(imported)
```

## Verification and limits

```sh
uv run --locked pytest -q test/squin/wire test/analysis test/jeff
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
Closures, list maps and direct iteration over qubit registers also exceed that contract.
Indexed loops, broadcasts and calls with whole-register or scalar-qubit arguments can convert directly.
The conversion tests have no expected-failure markers.
The pipeline emits wire operations and removes unused classical expressions.
It omits common-subexpression elimination, gate cancellation, angle merging and qubit reuse.
