use std::env;
use std::error::Error;
use std::path::Path;

use charstreamer_core::BinaryMetrics;
use charstreamer_experiments::{
    CorpusEvaluationReport, read_boundary_experiment_spec, run_boundary_experiment, write_report,
};

fn print_metrics(label: &str, metrics: BinaryMetrics) {
    println!(
        "{label}: precision={:.4} recall={:.4} f1={:.4} tp={} fp={} fn={}",
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.true_positives,
        metrics.false_positives,
        metrics.false_negatives,
    );
}

fn print_evaluation(report: &CorpusEvaluationReport) {
    println!(
        "{}: docs={} precision={:.4} recall={:.4} f1={:.4} throughput={:.1} chars/sec",
        report.name,
        report.documents,
        report.span_metrics.precision,
        report.span_metrics.recall,
        report.span_metrics.f1,
        report.throughput.chars_per_second,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "usage: cargo run -p charstreamer-experiments --example run_experiment_spec -- <spec.json> [report.json]"
        );
        std::process::exit(2);
    }

    let spec_path = Path::new(&args[1]);
    let spec = read_boundary_experiment_spec(spec_path)?;
    let report = run_boundary_experiment(&spec)?;

    println!("experiment: {}", report.spec.name);
    println!(
        "train_docs={} valid_docs={} train_rows={} positives={} negatives={} threshold={:.2} training_seconds={:.3}",
        report.train_documents,
        report.valid_documents,
        report.train_rows,
        report.train_positives,
        report.train_negatives,
        report.threshold,
        report.training_seconds,
    );
    print_metrics("candidate_metrics", report.candidate_metrics);
    print_evaluation(&report.validation);
    for evaluation in &report.evaluations {
        print_evaluation(evaluation);
    }

    if let Some(output_path) = args.get(2) {
        write_report(output_path, &report)?;
        println!("report: {}", output_path);
    }

    Ok(())
}
