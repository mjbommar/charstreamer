use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use charstreamer_experiments::{
    BoundaryExperimentReport, read_boundary_experiment_spec, run_boundary_experiment,
};

fn markdown_summary(reports: &[BoundaryExperimentReport]) -> String {
    let mut lines = Vec::with_capacity(reports.len() + 6);
    lines.push("# Experiment Sweep".to_string());
    lines.push(String::new());
    lines.push(
        "| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |".to_string(),
    );
    lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |".to_string());
    for report in reports {
        let (eval_f1, eval_cps) = report
            .evaluations
            .first()
            .map(|evaluation| {
                (
                    format!("{:.4}", evaluation.span_metrics.f1),
                    format!("{:.1}", evaluation.throughput.chars_per_second),
                )
            })
            .unwrap_or_else(|| ("-".to_string(), "-".to_string()));
        lines.push(format!(
            "| {} | {:.4} | {:.4} | {:.1} | {:.3} | {} | {} |",
            report.spec.name,
            report.validation.span_metrics.f1,
            report.candidate_metrics.f1,
            report.validation.throughput.chars_per_second,
            report.training_seconds,
            eval_f1,
            eval_cps,
        ));
    }
    lines.push(String::new());
    if let Some(best_val) = reports.iter().max_by(|a, b| {
        a.validation
            .span_metrics
            .f1
            .partial_cmp(&b.validation.span_metrics.f1)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        lines.push(format!(
            "Best validation F1: `{}` at `{:.4}`",
            best_val.spec.name, best_val.validation.span_metrics.f1
        ));
    }
    if let Some(best_eval) = reports.iter().max_by(|a, b| {
        let a_score = a
            .evaluations
            .first()
            .map(|evaluation| evaluation.span_metrics.f1)
            .unwrap_or(0.0);
        let b_score = b
            .evaluations
            .first()
            .map(|evaluation| evaluation.span_metrics.f1)
            .unwrap_or(0.0);
        a_score
            .partial_cmp(&b_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        let eval_f1 = best_eval
            .evaluations
            .first()
            .map(|evaluation| evaluation.span_metrics.f1)
            .unwrap_or(0.0);
        lines.push(format!(
            "Best first-eval F1: `{}` at `{:.4}`",
            best_eval.spec.name, eval_f1
        ));
    }
    lines.join("\n")
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: cargo run -p charstreamer-experiments --example run_experiment_sweep -- [--markdown-out report.md] <spec1.json> [spec2.json ...]"
        );
        std::process::exit(2);
    }

    let mut markdown_out = None;
    let mut spec_paths = Vec::new();
    let mut index = 1;
    while index < args.len() {
        if args[index] == "--markdown-out" {
            let Some(path) = args.get(index + 1) else {
                eprintln!("--markdown-out requires a path");
                std::process::exit(2);
            };
            markdown_out = Some(path.clone());
            index += 2;
            continue;
        }
        spec_paths.push(args[index].clone());
        index += 1;
    }
    if spec_paths.is_empty() {
        eprintln!("expected at least one spec path");
        std::process::exit(2);
    }

    let mut rows = Vec::new();
    let mut reports = Vec::new();
    for spec_path in &spec_paths {
        let spec = read_boundary_experiment_spec(Path::new(spec_path))?;
        let report = run_boundary_experiment(&spec)?;
        let mut row = format!(
            "{:<30} {:>8.4} {:>8.4} {:>12.1} {:>12.3}",
            report.spec.name,
            report.validation.span_metrics.f1,
            report.candidate_metrics.f1,
            report.validation.throughput.chars_per_second,
            report.training_seconds,
        );
        if let Some(first_eval) = report.evaluations.first() {
            row.push_str(&format!(
                " {:>8.4} {:>12.1}",
                first_eval.span_metrics.f1, first_eval.throughput.chars_per_second
            ));
        }
        rows.push(row);
        reports.push(report);
    }

    println!(
        "{:<30} {:>8} {:>8} {:>12} {:>12} {:>8} {:>12}",
        "experiment", "val_f1", "cand_f1", "val_cps", "train_s", "eval_f1", "eval_cps"
    );
    for row in rows {
        println!("{row}");
    }

    if let Some(path) = markdown_out {
        fs::write(&path, markdown_summary(&reports))?;
        println!("markdown: {path}");
    }

    Ok(())
}
