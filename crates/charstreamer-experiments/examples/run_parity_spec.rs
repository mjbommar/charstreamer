use std::env;
use std::error::Error;
use std::path::Path;

use charstreamer_experiments::{compare_charboundary_legacy_features, read_parity_check_spec};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!(
            "usage: cargo run -p charstreamer-experiments --example run_parity_spec -- <spec.json>"
        );
        std::process::exit(2);
    }

    let spec = read_parity_check_spec(Path::new(&args[1]))?;
    let report = compare_charboundary_legacy_features(&spec)?;

    println!("compared_rows={}", report.compared_rows);
    println!("exact_match={}", report.exact_match);
    if report.exact_match {
        return Ok(());
    }

    eprintln!("mismatched_rows={:?}", report.mismatched_rows);
    for mismatch in &report.mismatch_details {
        eprintln!("row {} rust={:?}", mismatch.row_index, mismatch.rust_row);
        eprintln!(
            "row {} python={:?}",
            mismatch.row_index, mismatch.python_row
        );
    }
    std::process::exit(1);
}
