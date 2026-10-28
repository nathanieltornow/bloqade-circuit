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

With `assume_disjoint=True`, the caller promises that uncertain operands name distinct qubits.
The default requires the analysis to prove that property.

## Pass contracts

| Stage | Kirin mechanism | Required result |
| --- | --- | --- |
| Prepare | `Inline`, `RewriteRule`, `Walk`, `Fixpoint`, `TypeInfer`, reference analysis | Calls expose supported reference behavior, and unknown-length broadcasts use loops. |
| Lower | `LowerTwins`, `LowerCalls`, `LowerLength`, `LowerReturn`, `LowerEntry` | Quantum operations use wire twins between explicit reference transfers. |
| Promote | `SlotAnalysis` and rewrite rules | Eligible slots travel through wire values, block arguments and results. |
| Check conversion | `ConversionTarget` with `Walk` | Complete conversion contains supported operations and excludes qubit reference types. |
| Validate | Kirin `ValidationSuite` and forward analyses | The result satisfies structure, types, linear uses, borrow balance, positions and reference access rules. |

Promotion first rewrites register accesses as `Borrow` and `Release`.
`Boundaries` adds the wire values that loops and branches carry.
`Forward` replaces a reference read with the preceding write's wire value when intervening operations permit it.
Materialization creates wire allocations and frees at the ends of those sequences.
Cleanup removes dead references and allocations.
Promotion reruns slot analysis after changes that rebuild statements or remove reference operands.

The rewrite rules use Kirin's walkers to visit the program.
A rule can rebuild an entire loop or branch.
Reference analysis follows data across blocks and calls.
The forwarding rule also searches adjacent statements within a block.
These dependencies require analysis beyond a single statement.

## Validation contracts

| Check | Required property |
| --- | --- |
| `StructureValidation` | Every method passes Kirin's structure and type checks, including partial conversions. |
| `LinearityValidation` | Each wire and register has one use on each applicable execution path. |
| `BorrowValidation` | Each release matches a borrowed index, and registers leave each boundary with every slot restored. |
| `PositionValidation` | Loop yields and call return suffixes continue their inputs in order, and releases restore the original qubits. |
| `ReferenceValidation` | Reference operations respect the outstanding transfers between references and wires. |

Borrow validation represents outstanding slots with a set of normalized index expressions.
An unmatched release causes an error.
An unknown set cannot establish that a register holds every qubit.
Both branches must continue the same qubit at each wire result position.
A function returns each borrowed parameter in a suffix, in parameter order.

Complete conversion checks every prepared method, including its nested regions.
The conversion target rejects source quantum operations, reference transfers and reference types inside containers or signatures.
It also rejects unresolved calls and operations from unsupported dialects.
`partial=True` keeps unsupported reference operations and reports them.
Partial conversion still requires every validation check to pass.

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
uv run --locked pytest -q test/squin/wire test/analysis/reference
```

The tests compare source execution with execution after conversion and reconstruction.
Reference tests also compare computed places with qubit identities from simulator runs.
Malformed-program tests check that validators reject missing qubits, reordered outputs and unsupported operations.
These tests cover concrete programs and mutations.
A proof of semantic preservation for every accepted program remains separate work.
