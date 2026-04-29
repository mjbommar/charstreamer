use std::collections::HashSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::corpus::AnnotatedDocument;
use crate::data::{ByteSpan, PipelineWorkspace};
use crate::error::PipelineError;
use crate::pipeline::Pipeline;
use crate::text::TextBytes;
use crate::traits::{BatchPredictor, CandidateScanner, Decoder, FeatureKernel};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BinaryMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThroughputReport {
    pub total_chars: usize,
    pub total_documents: usize,
    pub iterations: usize,
    pub elapsed_seconds: f64,
    pub chars_per_second: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineEvaluation {
    pub metrics: BinaryMetrics,
    pub throughput: ThroughputReport,
}

#[must_use]
pub fn metrics_from_scores(scores: &[f32], labels: &[u8], threshold: f32) -> BinaryMetrics {
    assert_eq!(
        scores.len(),
        labels.len(),
        "scores and labels must have matching lengths",
    );

    let mut true_positives = 0_usize;
    let mut false_positives = 0_usize;
    let mut false_negatives = 0_usize;
    let mut true_negatives = 0_usize;

    for (&score, &label) in scores.iter().zip(labels) {
        let predicted = score >= threshold;
        match (predicted, label) {
            (true, 1) => true_positives += 1,
            (true, _) => false_positives += 1,
            (false, 1) => false_negatives += 1,
            (false, _) => true_negatives += 1,
        }
    }

    metrics_from_counts(
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    )
}

#[must_use]
pub fn best_threshold_from_scores(scores: &[f32], labels: &[u8]) -> f32 {
    assert_eq!(
        scores.len(),
        labels.len(),
        "scores and labels must have matching lengths",
    );

    let mut best_threshold = 0.5_f32;
    let mut best_metrics = BinaryMetrics::default();

    for step in 5..=95 {
        let threshold = step as f32 / 100.0;
        let metrics = metrics_from_scores(scores, labels, threshold);
        if metrics.f1 > best_metrics.f1
            || (metrics.f1 == best_metrics.f1 && metrics.precision > best_metrics.precision)
        {
            best_metrics = metrics;
            best_threshold = threshold;
        }
    }

    best_threshold
}

pub fn evaluate_pipeline<S, K, M, D>(
    pipeline: &Pipeline<S, K, M, D>,
    documents: &[AnnotatedDocument],
) -> Result<BinaryMetrics, PipelineError>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    M: BatchPredictor<f32, f32>,
    D: Decoder<f32, ByteSpan>,
{
    let mut workspace = PipelineWorkspace::<f32, f32>::default();
    let mut predicted_spans = Vec::new();
    let mut true_positives = 0_usize;
    let mut false_positives = 0_usize;
    let mut false_negatives = 0_usize;

    for document in documents {
        predicted_spans.clear();
        pipeline.run_into(
            TextBytes::from_utf8(&document.text),
            &mut workspace,
            &mut predicted_spans,
        )?;

        let gold = boundary_set_from_spans(&document.sentence_spans);
        let predicted = boundary_set_from_spans(&predicted_spans);
        true_positives += gold.intersection(&predicted).count();
        false_positives += predicted.difference(&gold).count();
        false_negatives += gold.difference(&predicted).count();
    }

    Ok(metrics_from_counts(
        true_positives,
        false_positives,
        false_negatives,
        0,
    ))
}

pub fn benchmark_pipeline<S, K, M, D>(
    pipeline: &Pipeline<S, K, M, D>,
    documents: &[AnnotatedDocument],
    iterations: usize,
) -> Result<ThroughputReport, PipelineError>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    M: BatchPredictor<f32, f32>,
    D: Decoder<f32, ByteSpan>,
{
    let iterations = iterations.max(1);
    let mut workspace = PipelineWorkspace::<f32, f32>::default();
    let mut spans = Vec::new();
    let total_chars: usize = documents.iter().map(|document| document.text.len()).sum();

    let started = Instant::now();
    for _ in 0..iterations {
        for document in documents {
            spans.clear();
            pipeline.run_into(
                TextBytes::from_utf8(&document.text),
                &mut workspace,
                &mut spans,
            )?;
        }
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let chars_per_second = if elapsed_seconds > 0.0 {
        total_chars as f64 * iterations as f64 / elapsed_seconds
    } else {
        0.0
    };

    Ok(ThroughputReport {
        total_chars,
        total_documents: documents.len(),
        iterations,
        elapsed_seconds,
        chars_per_second,
    })
}

fn metrics_from_counts(
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    true_negatives: usize,
) -> BinaryMetrics {
    let precision = ratio(true_positives, true_positives + false_positives);
    let recall = ratio(true_positives, true_positives + false_negatives);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let accuracy = ratio(
        true_positives + true_negatives,
        true_positives + false_positives + false_negatives + true_negatives,
    );

    BinaryMetrics {
        accuracy,
        precision,
        recall,
        f1,
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn boundary_set_from_spans(spans: &[ByteSpan]) -> HashSet<usize> {
    let mut boundaries = HashSet::with_capacity(spans.len().saturating_mul(2));
    for span in spans {
        boundaries.insert(span.start.as_usize());
        boundaries.insert(span.end.as_usize());
    }
    boundaries
}

#[cfg(test)]
mod tests {
    use crate::metrics::{best_threshold_from_scores, metrics_from_scores};

    #[test]
    fn selects_threshold_with_better_f1() {
        let scores = [0.9, 0.8, 0.4, 0.1];
        let labels = [1, 1, 0, 0];
        let threshold = best_threshold_from_scores(&scores, &labels);
        assert!((0.4..=0.8).contains(&threshold));
    }

    #[test]
    fn computes_binary_metrics() {
        let metrics = metrics_from_scores(&[0.9, 0.2, 0.7, 0.1], &[1, 0, 0, 1], 0.5);
        assert_eq!(metrics.true_positives, 1);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.false_negatives, 1);
        assert_eq!(metrics.true_negatives, 1);
    }
}
