use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use charstreamer_core::ByteSpan;
use charstreamer_experiments::{read_boundary_experiment_spec, train_boundary_pipeline};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let spec_path = PathBuf::from(
        args.next()
            .ok_or("usage: inspect_text_spec <spec.json> <text-1.txt> [text-2.txt ...]")?,
    );
    let text_paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if text_paths.is_empty() {
        return Err("usage: inspect_text_spec <spec.json> <text-1.txt> [text-2.txt ...]".into());
    }

    let segment_limit = env_usize("CHARSTREAMER_SEGMENT_LIMIT", 12);
    let candidate_limit = env_usize("CHARSTREAMER_CANDIDATE_LIMIT", 12);
    let excerpt_limit = env_usize("CHARSTREAMER_EXCERPT_LIMIT", 1200);
    let benchmark_iterations = env_usize("CHARSTREAMER_BENCH_ITERS", 200);

    let spec = read_boundary_experiment_spec(&spec_path)?;
    let (pipeline, report) = train_boundary_pipeline(&spec)?;

    println!("spec: {}", report.spec.name);
    println!(
        "trained rows: {}  threshold: {:.4}  eval docs in manifest: {}",
        report.train_rows,
        pipeline.threshold(),
        report.evaluations.len()
    );

    for (index, text_path) in text_paths.iter().enumerate() {
        let text = fs::read_to_string(text_path)?;
        let inspection = pipeline.inspect_text(&text)?;
        let throughput = pipeline.benchmark_text(&text, benchmark_iterations)?;
        let excerpt = &text[..excerpt_limit.min(text.len())];

        if index > 0 {
            println!();
            println!("---");
        }
        println!("text: {}", text_path.display());
        println!(
            "text bytes: {}  candidates: {}  predicted segments: {}",
            inspection.bytes,
            inspection.candidate_count,
            inspection.predicted_spans.len()
        );
        println!(
            "throughput: {:.1} chars/sec over {} iterations",
            throughput.chars_per_second, throughput.iterations
        );
        println!();
        println!("excerpt:");
        println!("{}", excerpt);
        println!();
        println!("first {} predicted segments:", segment_limit);
        print_segments(&text, &inspection.predicted_spans, segment_limit);
        println!();
        println!("first {} scored candidates:", candidate_limit);
        print_candidates(
            &text,
            &inspection.candidates,
            &inspection.scores,
            candidate_limit,
        );
    }

    Ok(())
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn normalize_snippet(text: &str) -> String {
    text.replace('\n', "\\n").replace('\r', "\\r")
}

fn print_segments(text: &str, spans: &[ByteSpan], limit: usize) {
    for (index, span) in spans.iter().take(limit).enumerate() {
        let snippet = &text[span.start.as_usize()..span.end.as_usize()];
        println!(
            "{:>2}. [{:>5}..{:>5}] {}",
            index + 1,
            span.start.as_usize(),
            span.end.as_usize(),
            normalize_snippet(snippet)
        );
    }
}

fn print_candidates(
    text: &str,
    candidates: &[charstreamer_core::BytePos],
    scores: &[f32],
    limit: usize,
) {
    for (index, (position, score)) in candidates
        .iter()
        .zip(scores.iter().copied())
        .take(limit)
        .enumerate()
    {
        let byte_index = position.as_usize();
        let start = byte_index.saturating_sub(20);
        let end = (byte_index + 21).min(text.len());
        let snippet = &text[start..end];
        let byte = text.as_bytes()[byte_index] as char;
        println!(
            "{:>2}. byte {:>5} {:?} score {:>6.4}  {}",
            index + 1,
            byte_index,
            byte,
            score,
            normalize_snippet(snippet)
        );
    }
}
