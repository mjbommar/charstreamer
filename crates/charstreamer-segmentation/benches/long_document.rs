use std::fs;
use std::path::PathBuf;

use charstreamer_segmentation::CombinedSegmenter;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn fallback_long_document() -> String {
    let sample = r#"Case Note: Synthetic v. Example
Docket: 26-CV-1042
Date: April 28, 2026

# Background
The court reviewed the contract and the attached invoices. The vendor argued that the late delivery was excused by a port closure.

# Findings
First, the shipment logs show a gap of six days. Second, the buyer sent two written notices before canceling the order.

- The refund request was timely.
- The replacement goods were accepted without objection.
- Further interest is denied.

"I called the warehouse twice," Maria said. "Nobody could locate the missing pallets."

Conclusion: judgment is entered for the buyer in part.

"#;
    let mut text = String::with_capacity(3_500_000);
    while text.len() < 3_300_000 {
        text.push_str(sample);
    }
    text
}

fn load_long_document() -> (String, String) {
    let path = std::env::var("CHARSTREAMER_BENCH_TEXT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("data/bench/war_and_peace.txt")
        });
    match fs::read_to_string(&path) {
        Ok(text) => (path.display().to_string(), text),
        Err(_) => (
            "fallback_synthetic_long_document".to_string(),
            fallback_long_document(),
        ),
    }
}

fn bench_combined_segmenter(c: &mut Criterion) {
    let (name, text) = load_long_document();
    let segmenter = CombinedSegmenter::default();
    let mut group = c.benchmark_group("combined_segmenter_long_document");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, text| {
        b.iter(|| {
            let annotation = segmenter.annotate(text);
            criterion::black_box(annotation.spans.len());
            criterion::black_box(annotation.tagged.len());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_combined_segmenter);
criterion_main!(benches);
