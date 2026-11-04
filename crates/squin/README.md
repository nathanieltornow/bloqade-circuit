# Squin

This crate contains the Cargo setup for the Rust Squin dialect.
It has no dialect operations yet.

The workspace pins Kirin to commit `a8e48ec1c394bc9a6e04464c7459b5f75864bfe3` on its `rust` branch.
Kirin requires Rust 1.98 or later.
The Nix development shell supplies a compatible compiler, Cargo, rustfmt, and Clippy.

Run these commands from the repository root:

```sh
nix develop -c cargo check --workspace --all-targets --locked
nix develop -c cargo fmt --all --check
```
