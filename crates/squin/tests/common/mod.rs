//! Share the kernel parser of the squin tests.

use kirin::prelude::*;
use squin::Squin;

/// Parse a kernel with the parameters `parameters` and the statements `body`.
///
/// `parameters` is a list such as `"%c: Bool, %a: Float"`. The kernel has one
/// function, which returns an `Int`.
#[allow(dead_code)]
pub fn kernel(parameters: &str, body: &str) -> Pipeline<StageInfo<Squin>> {
    let types: Vec<&str> = parameters
        .split(',')
        .filter_map(|parameter| parameter.split(':').nth(1))
        .map(str::trim)
        .collect();
    let signature = format!("fn @main({}) -> Int", types.join(", "));
    let program = format!(
        "stage @kernel {signature};\n\
         specialize @kernel {signature} cfg {{\n  ^entry({parameters}) {{\n{body}\n  }}\n}}\n"
    );
    let mut pipeline: Pipeline<StageInfo<Squin>> = Pipeline::new();
    if let Err(error) = pipeline.parse(&program) {
        panic!("the kernel must parse: {error:?}\n{program}");
    }
    pipeline
}

/// Tell why a program text does not give a verified program.
#[derive(Debug)]
#[allow(dead_code)]
pub enum Rejected {
    Parse(kirin::parsers::FunctionParseError),
    Invalid(Vec<squin::Violation>),
}

/// Parse `text` and verify every function of its stage `@kernel`.
#[allow(dead_code)]
pub fn check(text: &str) -> Result<Pipeline<StageInfo<Squin>>, Rejected> {
    let mut pipeline: Pipeline<StageInfo<Squin>> = Pipeline::new();
    pipeline.parse(text).map_err(Rejected::Parse)?;
    let stage = pipeline
        .stage_by_name("kernel")
        .expect("the text declares @kernel");
    squin::verify(pipeline.stage(stage).unwrap()).map_err(Rejected::Invalid)?;
    Ok(pipeline)
}
