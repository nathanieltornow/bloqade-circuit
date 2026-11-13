//! Define squin, a quantum circuit language with linear wires and registers.
//!
//! A `Wire` names one qubit at one point in the program. Each statement
//! consumes its input wires and returns their next values. A `Register` holds
//! an array of qubits, and `borrow` lends one of them to a wire. Gates sit in
//! the dataflow graphs of `circuit` regions, and blocks hold the control flow.
//!
//! `ir` defines and queries programs, and it imports no other module.
//! `verify`, `rewrite` and `analysis` import only `ir`. Each module owns the statement traits that it
//! asks and their implementations.
#![forbid(unsafe_code)]

pub mod analysis;
pub mod ir;
pub mod rewrite;
pub mod verify;

pub use ir::Squin;
pub use ir::traits::{Pairs, Threads};
pub use ir::types::{ArithType, ArithValue, Length, Scalar, SquinType};
pub use verify::{Rule, Violation, verify};
