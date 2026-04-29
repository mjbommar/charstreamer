use std::path::PathBuf;
use std::time::Instant;

use charstreamer_segmentation::{CombinedSegmenter, SegmenterConfig, render_spans};

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

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("data/bench/war_and_peace.txt")
        });
    let (path_label, text) = match std::fs::read_to_string(&path) {
        Ok(text) => (path.display().to_string(), text),
        Err(_) => (
            "fallback_synthetic_long_document".to_string(),
            fallback_long_document(),
        ),
    };
    let structure_only = CombinedSegmenter::new(SegmenterConfig {
        include_sentences: false,
        ..SegmenterConfig::default()
    });
    let sentence_only = CombinedSegmenter::new(SegmenterConfig {
        include_paragraphs: false,
        include_metadata: false,
        include_sections: false,
        include_list_items: false,
        include_dialogue: false,
        ..SegmenterConfig::default()
    });
    let segmenter = CombinedSegmenter::default();

    let started = Instant::now();
    let structure_spans = structure_only.spans(&text);
    let structure_s = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let sentence_spans = sentence_only.spans(&text);
    let sentence_s = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let spans = segmenter.spans(&text);
    let spans_s = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let tagged = render_spans(&text, &spans);
    let render_s = started.elapsed().as_secs_f64();

    let total_s = spans_s + render_s;
    println!("path={path_label}");
    println!("bytes={}", text.len());
    println!(
        "structure_spans={} structure_s={structure_s:.6}",
        structure_spans.len()
    );
    println!(
        "sentence_spans={} sentence_s={sentence_s:.6}",
        sentence_spans.len()
    );
    println!("spans={}", spans.len());
    println!("tagged_bytes={}", tagged.len());
    println!("spans_s={spans_s:.6}");
    println!("render_s={render_s:.6}");
    println!("total_s={total_s:.6}");
    println!(
        "total_mib_s={:.3}",
        text.len() as f64 / total_s.max(f64::MIN_POSITIVE) / (1024.0 * 1024.0)
    );
}
