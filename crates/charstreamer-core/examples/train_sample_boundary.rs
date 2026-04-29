use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use charstreamer_core::{
    BatchPredictor, BinaryMetrics, BoundaryDatasetBuildOptions, Pipeline, ThresholdSpanDecoder,
    ThroughputReport, TrainablePredictor, TrainingPositionPolicy, benchmark_pipeline,
    best_threshold_from_scores, build_boundary_dataset, evaluate_pipeline, load_alea_jsonl,
    load_multilegal_jsonl, metrics_from_scores, split_documents,
};
use charstreamer_kernels::{ByteSet256, ByteSetScanner, CompositeFeatureKernel};
use charstreamer_models_native::{LogisticFitOptions, LogisticFitReport, LogisticModel};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct FeatureBlockSummary {
    name: String,
    offset: usize,
    width: usize,
}

#[derive(Debug, Serialize)]
struct CorpusEvaluationReport {
    dataset_path: String,
    documents: usize,
    span_metrics: BinaryMetrics,
    throughput: ThroughputReport,
}

#[derive(Debug, Serialize)]
struct SampleTrainingReport {
    format: &'static str,
    model_kind: &'static str,
    scanner_bytes: String,
    fit_options: LogisticFitOptions,
    fit_report: LogisticFitReport,
    training_seconds: f64,
    threshold: f32,
    train_documents: usize,
    valid_documents: usize,
    train_rows: usize,
    train_positives: usize,
    train_negatives: usize,
    candidate_metrics: BinaryMetrics,
    alea: CorpusEvaluationReport,
    multilegal: Option<CorpusEvaluationReport>,
    feature_blocks: Vec<FeatureBlockSummary>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root should be two levels above the crate")
        .to_path_buf();
    let legal_root = workspace_root.join("../legal-sentence-paper");
    let alea_path = env_path(
        "CHARSTREAMER_ALEA_PATH",
        legal_root.join("data/alea-legal-benchmark/train.jsonl"),
    );
    let multilegal_path = env_path(
        "CHARSTREAMER_MULTILEGAL_PATH",
        legal_root.join("data/MultiLegalSBD/CD_scotus.jsonl"),
    );
    let model_out = opt_env_path("CHARSTREAMER_MODEL_OUT");
    let report_out = opt_env_path("CHARSTREAMER_REPORT_OUT");
    let alea_limit = env_usize("CHARSTREAMER_ALEA_LIMIT", 1_500);
    let multilegal_limit = env_usize("CHARSTREAMER_MULTILEGAL_LIMIT", 128);

    let documents = load_alea_jsonl(&alea_path, Some(alea_limit))?;
    let (train_docs, valid_docs) = split_documents(documents, 9, 10);

    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!;:\"'\n\r"));
    let kernel = CompositeFeatureKernel::boundary_demo();

    let dataset_options = BoundaryDatasetBuildOptions {
        negative_keep_rate: 1.0,
        seed: Some(7),
        position_policy: TrainingPositionPolicy::ScannedCandidatesOnly,
    };
    let train_started = Instant::now();
    let train_dataset = build_boundary_dataset(&train_docs, &scanner, &kernel, &dataset_options)?;
    let fit_options = LogisticFitOptions {
        epochs: env_usize("CHARSTREAMER_EPOCHS", 35),
        learning_rate: env_f32("CHARSTREAMER_LR", 0.03),
        batch_size: env_usize("CHARSTREAMER_BATCH", 1024),
        l2: env_f32("CHARSTREAMER_L2", 1e-4),
        ..LogisticFitOptions::default()
    };
    let (model, fit_report) = LogisticModel::fit(
        train_dataset.as_view(),
        &fit_options,
        &mut charstreamer_core::FitScratch::default(),
    )?;
    let training_seconds = train_started.elapsed().as_secs_f64();

    let valid_dataset = build_boundary_dataset(&valid_docs, &scanner, &kernel, &dataset_options)?;
    let mut valid_scores = vec![0.0_f32; valid_dataset.rows()];
    model.predict_into(valid_dataset.features.as_view(), &mut valid_scores)?;
    let threshold = best_threshold_from_scores(&valid_scores, &valid_dataset.labels);
    let candidate_metrics = metrics_from_scores(&valid_scores, &valid_dataset.labels, threshold);

    let pipeline = Pipeline::new(
        ByteSetScanner::new(ByteSet256::from_bytes(b".?!;:\"'\n\r")),
        CompositeFeatureKernel::boundary_demo(),
        model.clone(),
        ThresholdSpanDecoder::new(threshold),
    );
    let alea_metrics = evaluate_pipeline(&pipeline, &valid_docs)?;
    let alea_throughput = benchmark_pipeline(&pipeline, &valid_docs, 3)?;

    println!("trained native logistic boundary model");
    println!("alea path: {}", alea_path.display());
    println!(
        "train docs: {}  valid docs: {}",
        train_docs.len(),
        valid_docs.len()
    );
    println!(
        "train rows: {}  positives: {}  negatives: {}",
        train_dataset.rows(),
        train_dataset.positives,
        train_dataset.negatives
    );
    println!(
        "training: {:.2}s  epochs: {}  final loss: {:.4}  pos_weight: {:.3}",
        training_seconds,
        fit_report.epochs,
        fit_report.final_loss,
        fit_report.positive_class_weight
    );
    println!(
        "validation candidate metrics: threshold {:.2}  precision {:.4}  recall {:.4}  f1 {:.4}",
        threshold, candidate_metrics.precision, candidate_metrics.recall, candidate_metrics.f1
    );
    println!(
        "validation span metrics (ALEA): precision {:.4}  recall {:.4}  f1 {:.4}  accuracy {:.4}",
        alea_metrics.precision, alea_metrics.recall, alea_metrics.f1, alea_metrics.accuracy
    );
    println!(
        "validation throughput (ALEA): {:.0} chars/sec across {} docs",
        alea_throughput.chars_per_second, alea_throughput.total_documents
    );

    let mut multilegal_report = None;
    if multilegal_path.exists() {
        let multilegal_docs = load_multilegal_jsonl(&multilegal_path, Some(multilegal_limit))?;
        let multilegal_metrics = evaluate_pipeline(&pipeline, &multilegal_docs)?;
        let multilegal_throughput = benchmark_pipeline(&pipeline, &multilegal_docs, 2)?;
        println!("multilegal path: {}", multilegal_path.display());
        println!(
            "span metrics (MultiLegal): precision {:.4}  recall {:.4}  f1 {:.4}  accuracy {:.4}",
            multilegal_metrics.precision,
            multilegal_metrics.recall,
            multilegal_metrics.f1,
            multilegal_metrics.accuracy
        );
        println!(
            "throughput (MultiLegal): {:.0} chars/sec across {} docs",
            multilegal_throughput.chars_per_second, multilegal_throughput.total_documents
        );
        multilegal_report = Some(CorpusEvaluationReport {
            dataset_path: multilegal_path.display().to_string(),
            documents: multilegal_docs.len(),
            span_metrics: multilegal_metrics,
            throughput: multilegal_throughput,
        });
    }

    if let Some(path) = model_out {
        model.save_json_with_schema(&path, kernel.schema())?;
        println!("saved model artifact: {}", path.display());
    }

    if let Some(path) = report_out {
        let report = SampleTrainingReport {
            format: "charstreamer.sample-training-report.v1",
            model_kind: "logistic",
            scanner_bytes: String::from(".?!;:\\\"'\\n\\r"),
            fit_options: fit_options.clone(),
            fit_report,
            training_seconds,
            threshold,
            train_documents: train_docs.len(),
            valid_documents: valid_docs.len(),
            train_rows: train_dataset.rows(),
            train_positives: train_dataset.positives,
            train_negatives: train_dataset.negatives,
            candidate_metrics,
            alea: CorpusEvaluationReport {
                dataset_path: alea_path.display().to_string(),
                documents: valid_docs.len(),
                span_metrics: alea_metrics,
                throughput: alea_throughput,
            },
            multilegal: multilegal_report,
            feature_blocks: kernel
                .schema()
                .blocks()
                .iter()
                .map(|block| FeatureBlockSummary {
                    name: block.name.to_string(),
                    offset: block.offset,
                    width: block.width,
                })
                .collect(),
        };
        fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
        println!("saved training report: {}", path.display());
    }

    Ok(())
}

fn env_path(key: &str, default: PathBuf) -> PathBuf {
    env::var_os(key).map(PathBuf::from).unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn opt_env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key).map(PathBuf::from)
}
