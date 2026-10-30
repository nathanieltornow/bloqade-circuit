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
Each callee must pass the same conversion target and validation.

`check_stage` runs `StructureValidation` for source structure and types before emission.
`SourceValidation` then checks region forms and static calls, and it collects the callees.
Emission visits each source statement once per method conversion.
Kirin's region interpreter controls statement order and fills the frame after each operation.
Standard library wrappers execute through nested emission frames with shared output and wire state.
Each library call interprets its wrapper body once.
The frame's `site` names the user call, and every operation of the wrapper takes that call's source location.
Validation, type inference and cleanup perform their own traversals.
Wire validation interprets each callee once for each list of abstract arguments.
The full pipeline has no established linear-time bound.
The pinned Kirin version requires literal tuple-index hints during output type inference.
`src/bloqade/squin/passes/wire_typeinfer.py` adds those hints.

Unsupported source forms stop conversion at their first failed condition.
The converter preserves the source and propagates the error without retrying through inlining.

## Emission state

`WireFrame` extends Kirin's `EmitFrame` with an output block and allocation state.
Its inherited entries map source SSA values to their emitted meanings.
An SSA value names one compiler value with one definition.

| State | Meaning |
| --- | --- |
| `entries` | Map each source value to an emitted classical value, quantum location or container of values. |
| `latest` | Map each allocation root to its latest wire or register value. |
| `local_roots` | Record allocations that this region must free unless the function returns them. |
| `output` | Hold the emitted block and its arguments. |
| `site` | Name the user call that runs a library wrapper. It is None in user code. |

A `Location` is one of six forms, and each form has a `root`.
The root is the emitted allocation result, a call result or a function parameter.

| Form | Meaning | Example |
| --- | --- | --- |
| `Qubit(root)` | A standalone qubit. | `q` |
| `Element(root, index)` | One qubit of a register. | `qs[i]` |
| `Register(root, length)` | A whole flat register. | `qs` |
| `View(root, span)` | A slice of the qubits of a register. | `qs[1:3]`, `grid[1]` of 2 dimensions |
| `Grid(root, length, row_shape)` | A whole nested register. | `grid` |
| `Rows(root, span, row_shape)` | A row whose items are rows. | `cube[1]` of 3 dimensions |

`select` and `select_slice` in `values.py` derive the location of an item or a slice.
A nested register has one flat root register that holds its rows in order.
The frame keeps locations stable while `latest` changes after each quantum operation.
An alias copies its operand's frame entry.
An index selects from that entry using the affine index of the source integer operand.
A source list holds its element locations in a tuple.

Emission uses these frame entries for all source reference resolution.
Each reference operation reads the meanings of its operands from that frame.
`IndexAnalysis` in `bloqade.analysis.indices` reads integer arithmetic from definitions.
It represents expressions such as `i + 1` and proves simple index separation.
The emitter analyzes source integers, so the symbols of an index are source values.
A loop symbol carries the constant range of its source loop.
`emit_affine` maps each symbol to its emitted value through the frame entries.
An integer that is not affine, such as `i * i`, is one opaque symbol.
Two uses of one opaque symbol name the same element, and two different opaque symbols never provably differ.

A wire register index lies between 0 and the register length minus 1, as in Jeff.
The converter counts a negative constant index from the end and checks constant bounds.
A runtime index must be a valid wire index, which is a precondition of the source program.
The converter emits a runtime index unchanged and checks neither its sign nor its upper bound.
So `qs[k]` with a runtime `k` of -1 is invalid input, although Python would read it as the last element.
Jeff has no assertion statement, so the converter cannot emit a runtime check that Jeff keeps.
Runtime register lengths come from wire `Length` operations.
Nested register parameters use one flat register with their rows in source order.

The operation tables live in `squin.emit.wire`:

| File | Responsibility |
| --- | --- |
| `base.py` | Define the emitter, the frame and the operations on latest wires, and copy classical operations. |
| `functions.py` | Emit function boundaries, library wrappers and calls. |
| `references.py` | Resolve qubit references in allocations, aliases, lists, tuples, additions, lengths and indexing. |
| `control.py` | Convert loop and branch regions through Kirin's region interpreter. |
| `gates.py` | Emit wire gates and runtime broadcast loops. |
| `values.py` | Define the location forms and select elements, slices and rows. |

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
For a standalone qubit, it consumes the latest wire and records the output wire.
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
At region exit, `free_local_allocations` frees each local allocation that the region still holds.
Returning a fresh allocation consumes that value before cleanup.

A known-length broadcast emits one scalar gate per operand group.
An unknown-length broadcast emits a loop with the same scalar gate handler.
Measurement loops carry their classical result lists alongside the wire registers.
Equal-length broadcasts check their operand lengths before entering the loop.
Operations with shortest-list semantics use the minimum length.

## Loops and branches

The loop handler reads every available outer allocation from `latest`.
It creates output parameters for classical loop state and those allocations.
It builds the output loop after its body, and `scf.For` reads its result types from the final yield.
The body's frame starts with those parameters as its latest wire values.
Kirin interprets the source body once.
The handler then emits a yield with the final classical values and latest wires.
The loop results replace the enclosing frame's latest values.
For zero iterations, the loop returns its initial values.

```text
r_after = For(indices, initial=r0):
    r_iter = iteration_argument
    r_hole, q0 = Borrow(r_iter, i)
    q1 = X(q0)
    r_next = Release(r_hole, i, q1)
    Yield(r_next)
```

Each branch starts with a separate copy of the incoming `latest` map.
The handler interprets both source regions once.
Both output regions yield the incoming allocations in the same order.
The conditional's results become the enclosing frame's latest values.
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

`Layout` reads this convention from the source signature.
It gives the borrowed parameter positions, the ordinary result count and the tuple packing of the source return.
The caller reads the type of each wire result from the callee's inferred signature.
`Contract` holds the callee's wire method and one static length for each ordinary result.
The length is None for a classical result and for a register of unknown length.
A length is an affine expression in the callee's integer parameters.
The caller replaces those parameters with its own arguments, so `qalloc(n)` in the callee gives a known length at a call with a constant `n`.
With a symbolic `n`, a broadcast over the result reads the runtime length through `Length`.

The caller consumes its latest inputs and restores them from the borrowed-parameter suffix.
Calls on register elements use `Borrow` and `Release` around the call.
The helper below borrows `q`, and its loop carries the wire of `q`:

```python
@squin.kernel
def kick(q: Qubit, k: int) -> int:
    for _ in range(k):
        squin.x(q)
    return k + 1


@squin.kernel
def main():
    qs = squin.qalloc(2)
    return kick(qs[1], 3)
```

```text
func kick(q: Wire, k: int) -> tuple[int, Wire]:
    q_end = For(range(0, k), initial=q):
        q_iter = iteration_argument
        q_next = X(q_iter)
        Yield(q_next)
    Return(tuple(k + 1, q_end))

func main() -> int:
    r0 = AllocRegister(2)
    r1, q0 = Borrow(r0, 1)
    result = call kick(q0, 3)
    total = result[0]
    q1 = result[1]
    r2 = Release(r1, 1, q1)
    FreeRegister(r2)
    Return(total)
```

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
Repeated public conversions clear the contract cache.

## Validation contracts

| Check | Required property |
| --- | --- |
| `StructureValidation` | Kirin verifies method structure and types. |
| `LinearityValidation` | Each wire, register and quantum tuple field has one use on each execution path. |
| `BorrowValidation` | Releases match borrowed indices and registers restore every element before crossing a boundary. |
| `PositionValidation` | Releases, loop yields, branches and call return suffixes continue the correct qubits in order. |

Linearity and borrow validation use Kirin forward analyses with operation method tables.
Borrow validation tracks outstanding register indices in its `Slots` lattice.
Its `Moved` entries name elements that `Take` or `TakeSlice` moved out of a register.

Position validation uses Kirin's `Forward` analysis with operation method tables.
Its `Origin` lattice stores exact allocation identities and tuple fields in Kirin frames.
An unknown origin records insufficient information to prove continuity.
The analysis follows operation continuations and checks loop yields, branch results and borrowed return suffixes.
It checks each callee and gives every call's fresh outputs distinct identities.
`CachedCalls` reuses the result of a call with the same callee and abstract arguments.
Each analysis records its errors at callee nodes, so a reused result loses no error.

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
