# Squin

This crate defines squin, a quantum circuit language with linear wires and registers, on Kirin.
A `Wire` names one qubit at one point in the program, and each statement consumes its input wires and returns their next values.
Gates sit in the dataflow graphs of `circuit` regions, and blocks hold the control flow, such as `conditional`, `repeat` and Kirin's `for`.
`squin::verify` checks the types, the single use of each wire, the scope and captures of each region, and the placement of each statement.

The workspace pins Kirin to commit `93629607358289f168d4c10405ce32a11c07d04a` of the fork `nathanieltornow/kirin`.
The fork fixes how the parser ends a digraph body at its `yield` clause.
Kirin requires Rust 1.98 or later.
The Nix development shell supplies a compatible compiler, Cargo, rustfmt, and Clippy.

Run these commands from the repository root:

```sh
nix develop -c cargo check --workspace --all-targets --locked
nix develop -c cargo fmt --all --check
```
