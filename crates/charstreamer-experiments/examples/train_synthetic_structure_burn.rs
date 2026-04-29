use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::{TensorData, backend::Backend};
use charstreamer_core::{
    BytePos, ByteWindowSpec, CandidateBuffer, FeatureKernel, FeatureMatrix, FeatureScratch,
    TextBytes,
};
use charstreamer_kernels::{
    AsciiClassAppender, BoundaryShapeAppender, ByteClass, CompositeFeatureKernel,
    DirectionalByteClassCountAppender, DirectionalUnicodeCategoryGroupCountAppender,
    EncodedByteWindowAppender, LineByteCountAppender, UnicodeCategoryGroup,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

type InferBackend = NdArray<f32>;
type TrainBackend = Autodiff<InferBackend>;

const DEFAULT_LABELS: &[&str] = &["metadata", "section", "list_item", "dialogue"];

#[derive(Debug)]
enum TrainError {
    InvalidArgument(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    Feature(charstreamer_core::FeatureError),
    Burn(String),
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => f.write_str(message),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::Feature(error) => write!(f, "feature extraction error: {error}"),
            Self::Burn(message) => f.write_str(message),
        }
    }
}

impl Error for TrainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Feature(error) => Some(error),
            Self::InvalidArgument(_) | Self::Burn(_) => None,
        }
    }
}

impl From<std::io::Error> for TrainError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for TrainError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<charstreamer_core::FeatureError> for TrainError {
    fn from(error: charstreamer_core::FeatureError) -> Self {
        Self::Feature(error)
    }
}

#[derive(Clone, Debug)]
struct Config {
    inputs: Vec<PathBuf>,
    report_path: PathBuf,
    inspect_texts: Vec<String>,
    labels: Vec<String>,
    epochs: usize,
    batch_size: usize,
    hidden_dim: usize,
    hidden_dim2: usize,
    learning_rate: f64,
    seed: u64,
    split_numerator: usize,
    split_denominator: usize,
    negative_keep_rate: f32,
    max_records: Option<usize>,
    encoded_left: usize,
    encoded_right: usize,
    count_radius: usize,
    validation_predict_batch_size: usize,
    min_span_bytes: usize,
    merge_gap_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            report_path: PathBuf::from("/tmp/charstreamer-structure-burn-report.json"),
            inspect_texts: Vec::new(),
            labels: DEFAULT_LABELS
                .iter()
                .map(|label| (*label).to_string())
                .collect(),
            epochs: 20,
            batch_size: 512,
            hidden_dim: 96,
            hidden_dim2: 48,
            learning_rate: 1.0e-3,
            seed: 29,
            split_numerator: 8,
            split_denominator: 10,
            negative_keep_rate: 1.0,
            max_records: None,
            encoded_left: 7,
            encoded_right: 7,
            count_radius: 32,
            validation_predict_batch_size: 16_384,
            min_span_bytes: 4,
            merge_gap_bytes: 2,
        }
    }
}

#[derive(Clone, Deserialize)]
struct SyntheticRecord {
    text: String,
    #[serde(default)]
    spans: Vec<SyntheticSpan>,
}

#[derive(Clone, Debug, Deserialize)]
struct SyntheticSpan {
    label: String,
    start: usize,
    end: usize,
    #[serde(default)]
    left_open: bool,
    #[serde(default)]
    right_open: bool,
}

#[derive(Clone, Debug)]
struct CleanRecord {
    text: String,
    spans: Vec<SyntheticSpan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LineCandidate {
    start: usize,
    end: usize,
    feature_pos: usize,
}

#[derive(Debug)]
struct StructureDataset {
    features: FeatureMatrix<f32>,
    targets: Vec<u8>,
    rows: usize,
    output_dim: usize,
    documents: usize,
    chars: usize,
    positives: Vec<usize>,
}

#[derive(Module, Debug)]
struct StructureMlp<B: Backend> {
    input: Linear<B>,
    activation1: Relu,
    hidden2: Linear<B>,
    activation2: Relu,
    output: Linear<B>,
}

impl<B: Backend> StructureMlp<B> {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        hidden_dim2: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim).init(device),
            activation1: Relu::new(),
            hidden2: LinearConfig::new(hidden_dim, hidden_dim2).init(device),
            activation2: Relu::new(),
            output: LinearConfig::new(hidden_dim2, output_dim).init(device),
        }
    }

    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden1 = self.activation1.forward(self.input.forward(input));
        let hidden2 = self.activation2.forward(self.hidden2.forward(hidden1));
        self.output.forward(hidden2)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct BinaryMetricReport {
    precision: f64,
    recall: f64,
    f1: f64,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    true_negatives: usize,
}

#[derive(Clone, Debug, Serialize)]
struct OutputMetricReport {
    label: String,
    threshold: f32,
    positives: usize,
    negatives: usize,
    metrics: BinaryMetricReport,
}

#[derive(Clone, Debug, Serialize)]
struct DatasetReport {
    documents: usize,
    rows: usize,
    chars: usize,
    positives_by_output: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingReport {
    inputs: Vec<PathBuf>,
    labels: Vec<String>,
    feature_dim: usize,
    hidden_dim: usize,
    hidden_dim2: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
    negative_keep_rate: f32,
    loaded_documents: usize,
    invalid_documents: usize,
    train: DatasetReport,
    validation: DatasetReport,
    feature_seconds_train: f64,
    feature_seconds_validation: f64,
    train_seconds: f64,
    validation_predict_seconds: f64,
    validation_rows_per_second: f64,
    validation_end_to_end_chars_per_second: f64,
    macro_f1: f64,
    output_metrics: Vec<OutputMetricReport>,
}

#[derive(Clone, Debug)]
struct DecodedSpan {
    label: String,
    start: usize,
    end: usize,
    score: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let started = Instant::now();
    let (mut records, invalid_documents) = load_records(&config)?;
    let loaded_documents = records.len();
    if loaded_documents < 2 {
        return Err(Box::new(TrainError::InvalidArgument(
            "need at least two valid records".to_string(),
        )));
    }

    let mut rng = SmallRng::seed_from_u64(config.seed);
    records.shuffle(&mut rng);
    let split_at =
        loaded_documents.saturating_mul(config.split_numerator) / config.split_denominator;
    if split_at == 0 || split_at >= loaded_documents {
        return Err(Box::new(TrainError::InvalidArgument(
            "invalid train/validation split".to_string(),
        )));
    }
    let validation_records = records.split_off(split_at);
    let train_records = records;
    let kernel = build_kernel(&config);

    let feature_started = Instant::now();
    let train_dataset = build_dataset(&train_records, &kernel, &config, true)?;
    let feature_seconds_train = feature_started.elapsed().as_secs_f64();

    let feature_started = Instant::now();
    let validation_dataset = build_dataset(&validation_records, &kernel, &config, false)?;
    let feature_seconds_validation = feature_started.elapsed().as_secs_f64();

    let train_started = Instant::now();
    let model = train_model(&train_dataset, &config)?;
    let train_seconds = train_started.elapsed().as_secs_f64();

    let predict_started = Instant::now();
    let scores = predict_probabilities(&model, &validation_dataset, &config)?;
    let validation_predict_seconds = predict_started.elapsed().as_secs_f64();

    let output_metrics = tune_and_score_outputs(&scores, &validation_dataset, &config.labels);
    let macro_f1 = output_metrics
        .iter()
        .map(|metric| metric.metrics.f1)
        .sum::<f64>()
        / output_metrics.len().max(1) as f64;
    let validation_end_to_end_seconds = feature_seconds_validation + validation_predict_seconds;
    let report = TrainingReport {
        inputs: config.inputs.clone(),
        labels: config.labels.clone(),
        feature_dim: train_dataset.features.cols,
        hidden_dim: config.hidden_dim,
        hidden_dim2: config.hidden_dim2,
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        seed: config.seed,
        negative_keep_rate: config.negative_keep_rate,
        loaded_documents,
        invalid_documents,
        train: dataset_report(&train_dataset, &config.labels),
        validation: dataset_report(&validation_dataset, &config.labels),
        feature_seconds_train,
        feature_seconds_validation,
        train_seconds,
        validation_predict_seconds,
        validation_rows_per_second: validation_dataset.rows as f64
            / validation_predict_seconds.max(f64::MIN_POSITIVE),
        validation_end_to_end_chars_per_second: validation_dataset.chars as f64
            / validation_end_to_end_seconds.max(f64::MIN_POSITIVE),
        macro_f1,
        output_metrics,
    };
    fs::write(&config.report_path, serde_json::to_vec_pretty(&report)?)?;

    println!(
        "loaded_docs={} invalid_docs={} train_docs={} valid_docs={} train_rows={} valid_rows={} feature_dim={} elapsed_s={:.3}",
        loaded_documents,
        invalid_documents,
        train_dataset.documents,
        validation_dataset.documents,
        train_dataset.rows,
        validation_dataset.rows,
        train_dataset.features.cols,
        started.elapsed().as_secs_f64(),
    );
    println!(
        "structure_mlp hidden={}/{} epochs={} batch={} lr={} train_s={:.3} valid_predict_s={:.3} valid_rows_per_s={:.1} valid_e2e_chars_per_s={:.1} macro_f1={:.4}",
        config.hidden_dim,
        config.hidden_dim2,
        config.epochs,
        config.batch_size,
        config.learning_rate,
        train_seconds,
        validation_predict_seconds,
        report.validation_rows_per_second,
        report.validation_end_to_end_chars_per_second,
        report.macro_f1,
    );
    for metric in &report.output_metrics {
        println!(
            "{} f1={:.4} p={:.4} r={:.4} pos={} th={:.2}",
            metric.label,
            metric.metrics.f1,
            metric.metrics.precision,
            metric.metrics.recall,
            metric.positives,
            metric.threshold,
        );
    }
    println!("report: {}", config.report_path.display());
    inspect_texts(&config, &kernel, &model, &report.output_metrics)?;
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, TrainError> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => config.inputs.push(next_path(&mut args, "--input")?),
            "--report" => config.report_path = next_path(&mut args, "--report")?,
            "--inspect-text" => config
                .inspect_texts
                .push(next_value(&mut args, "--inspect-text")?),
            "--labels" => config.labels = split_csv(&next_value(&mut args, "--labels")?),
            "--epochs" => config.epochs = parse_next(&mut args, "--epochs")?,
            "--batch-size" => config.batch_size = parse_next(&mut args, "--batch-size")?,
            "--hidden-dim" => config.hidden_dim = parse_next(&mut args, "--hidden-dim")?,
            "--hidden-dim2" => config.hidden_dim2 = parse_next(&mut args, "--hidden-dim2")?,
            "--learning-rate" => config.learning_rate = parse_next(&mut args, "--learning-rate")?,
            "--seed" => config.seed = parse_next(&mut args, "--seed")?,
            "--split-numerator" => {
                config.split_numerator = parse_next(&mut args, "--split-numerator")?
            }
            "--split-denominator" => {
                config.split_denominator = parse_next(&mut args, "--split-denominator")?;
            }
            "--negative-keep-rate" => {
                config.negative_keep_rate = parse_next(&mut args, "--negative-keep-rate")?;
            }
            "--max-records" => {
                config.max_records = nonzero_option(parse_next(&mut args, "--max-records")?)
            }
            "--encoded-left" => config.encoded_left = parse_next(&mut args, "--encoded-left")?,
            "--encoded-right" => config.encoded_right = parse_next(&mut args, "--encoded-right")?,
            "--count-radius" => config.count_radius = parse_next(&mut args, "--count-radius")?,
            "--validation-predict-batch-size" => {
                config.validation_predict_batch_size =
                    parse_next(&mut args, "--validation-predict-batch-size")?;
            }
            "--min-span-bytes" => {
                config.min_span_bytes = parse_next(&mut args, "--min-span-bytes")?
            }
            "--merge-gap-bytes" => {
                config.merge_gap_bytes = parse_next(&mut args, "--merge-gap-bytes")?
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(TrainError::InvalidArgument(format!(
                    "unknown argument `{other}`; use --help"
                )));
            }
        }
    }
    if config.inputs.is_empty() {
        return Err(TrainError::InvalidArgument(
            "at least one --input JSONL path is required".to_string(),
        ));
    }
    if config.labels.is_empty() || config.epochs == 0 || config.batch_size == 0 {
        return Err(TrainError::InvalidArgument(
            "labels, epochs, and batch size must be non-empty".to_string(),
        ));
    }
    if config.split_denominator == 0 || config.split_numerator >= config.split_denominator {
        return Err(TrainError::InvalidArgument(
            "split must satisfy numerator < denominator".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.negative_keep_rate) {
        return Err(TrainError::InvalidArgument(
            "negative keep rate must be between 0 and 1".to_string(),
        ));
    }
    Ok(config)
}

fn print_usage() {
    eprintln!(
        "usage: cargo run -p charstreamer-experiments --example train_synthetic_structure_burn -- \\
  --input <jsonl> [--report <report.json>] [--inspect-text <text>]"
    );
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, TrainError> {
    args.next()
        .ok_or_else(|| TrainError::InvalidArgument(format!("{flag} requires a value")))
}

fn next_path(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<PathBuf, TrainError> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn parse_next<T>(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<T, TrainError>
where
    T: std::str::FromStr,
    T::Err: Display,
{
    let value = next_value(args, flag)?;
    value.parse::<T>().map_err(|error| {
        TrainError::InvalidArgument(format!("invalid value `{value}` for {flag}: {error}"))
    })
}

fn nonzero_option(value: usize) -> Option<usize> {
    if value == 0 { None } else { Some(value) }
}

fn split_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn load_records(config: &Config) -> Result<(Vec<CleanRecord>, usize), TrainError> {
    let allowed_labels = config
        .labels
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut records = Vec::new();
    let mut invalid = 0_usize;
    for path in &config.inputs {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if config.max_records.is_some_and(|max| records.len() >= max) {
                return Ok((records, invalid));
            }
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed: SyntheticRecord = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(_) => {
                    invalid += 1;
                    continue;
                }
            };
            match clean_record(parsed, &allowed_labels) {
                Some(record) => records.push(record),
                None => invalid += 1,
            }
        }
    }
    Ok((records, invalid))
}

fn clean_record(record: SyntheticRecord, allowed_labels: &BTreeSet<&str>) -> Option<CleanRecord> {
    if record.text.is_empty() {
        return None;
    }
    let mut spans = Vec::new();
    for span in record.spans {
        if !allowed_labels.contains(span.label.as_str()) {
            continue;
        }
        if span.start > span.end || span.end > record.text.len() {
            return None;
        }
        if !record.text.is_char_boundary(span.start) || !record.text.is_char_boundary(span.end) {
            return None;
        }
        if span.start == span.end {
            continue;
        }
        spans.push(span);
    }
    Some(CleanRecord {
        text: record.text,
        spans,
    })
}

fn build_kernel(config: &Config) -> CompositeFeatureKernel {
    CompositeFeatureKernel::new(vec![
        Box::new(EncodedByteWindowAppender::new(ByteWindowSpec::new(
            config.encoded_left,
            config.encoded_right,
        ))),
        Box::new(AsciiClassAppender::new()),
        Box::new(BoundaryShapeAppender::new()),
        Box::new(DirectionalByteClassCountAppender::new(
            "directional_byte_class_counts",
            ByteWindowSpec::new(config.count_radius, config.count_radius),
            vec![
                ByteClass::AsciiUpper,
                ByteClass::AsciiLower,
                ByteClass::AsciiDigit,
                ByteClass::AsciiWhitespace,
                ByteClass::AsciiPunctuation,
                ByteClass::LineBreak,
                ByteClass::OpenBracket,
                ByteClass::CloseBracket,
            ],
        )),
        Box::new(DirectionalUnicodeCategoryGroupCountAppender::new(
            "directional_unicode_group_counts",
            ByteWindowSpec::new(config.count_radius, config.count_radius),
            vec![
                UnicodeCategoryGroup::L,
                UnicodeCategoryGroup::N,
                UnicodeCategoryGroup::P,
                UnicodeCategoryGroup::S,
                UnicodeCategoryGroup::Z,
                UnicodeCategoryGroup::C,
            ],
        )),
        Box::new(LineByteCountAppender::new(
            "line_structure_counts",
            vec![b'\n', b'#', b'-', b'*', b':', b'"', b'\'', b'<', b'>', b','],
        )),
    ])
}

fn build_dataset(
    records: &[CleanRecord],
    kernel: &CompositeFeatureKernel,
    config: &Config,
    sample_negatives: bool,
) -> Result<StructureDataset, TrainError> {
    let output_dim = config.labels.len();
    let mut dataset = StructureDataset {
        features: FeatureMatrix {
            rows: 0,
            cols: kernel.schema().total_dim(),
            data: Vec::new(),
        },
        targets: Vec::new(),
        rows: 0,
        output_dim,
        documents: 0,
        chars: 0,
        positives: vec![0; output_dim],
    };
    let mut rng = SmallRng::seed_from_u64(config.seed ^ if sample_negatives { 0x91 } else { 0xC3 });
    let mut candidate_buffer = CandidateBuffer::new();
    let mut feature_scratch = FeatureScratch::default();
    let mut doc_features = FeatureMatrix::<f32>::default();
    let mut selected_targets = Vec::new();

    for record in records {
        candidate_buffer.clear();
        selected_targets.clear();
        for candidate in line_candidates(&record.text) {
            let row_targets = targets_for_candidate(record, candidate, &config.labels);
            let has_positive = row_targets.iter().any(|&target| target != 0);
            if sample_negatives && !has_positive && rng.random::<f32>() > config.negative_keep_rate
            {
                continue;
            }
            candidate_buffer.push(BytePos::from_usize(candidate.feature_pos));
            for (index, value) in row_targets.iter().copied().enumerate() {
                dataset.positives[index] += usize::from(value != 0);
            }
            selected_targets.extend_from_slice(&row_targets);
        }
        if candidate_buffer.is_empty() {
            continue;
        }
        doc_features.resize_zeroed(candidate_buffer.len(), dataset.features.cols);
        kernel.extract_into(
            TextBytes::from_utf8(&record.text),
            candidate_buffer.as_slice(),
            doc_features.as_view_mut(),
            &mut feature_scratch,
        )?;
        dataset.features.data.extend_from_slice(&doc_features.data);
        dataset.features.rows += doc_features.rows;
        dataset.targets.extend_from_slice(&selected_targets);
        dataset.rows += candidate_buffer.len();
        dataset.documents += 1;
        dataset.chars += record.text.len();
    }
    Ok(dataset)
}

fn line_candidates(text: &str) -> Vec<LineCandidate> {
    let mut candidates = Vec::new();
    let mut start = 0_usize;
    for (offset, ch) in text.char_indices() {
        if !matches!(ch, '\n' | '\r') {
            continue;
        }
        push_line_candidate(text, start, offset, &mut candidates);
        start = offset + ch.len_utf8();
    }
    push_line_candidate(text, start, text.len(), &mut candidates);
    candidates
}

fn push_line_candidate(text: &str, start: usize, end: usize, out: &mut Vec<LineCandidate>) {
    let trimmed_start = next_nonspace_position(text, start, end).unwrap_or(end);
    let trimmed_end = previous_nonspace_end(text, start, end).unwrap_or(trimmed_start);
    if trimmed_end <= trimmed_start {
        return;
    }
    out.push(LineCandidate {
        start: trimmed_start,
        end: trimmed_end,
        feature_pos: trimmed_start,
    });
}

fn targets_for_candidate(
    record: &CleanRecord,
    candidate: LineCandidate,
    labels: &[String],
) -> Vec<u8> {
    labels
        .iter()
        .map(|label| {
            record.spans.iter().any(|span| {
                span.label == *label
                    && !span.left_open
                    && !span.right_open
                    && overlap_ratio(candidate.start, candidate.end, span.start, span.end) >= 0.35
            }) as u8
        })
        .collect()
}

fn overlap_ratio(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> f32 {
    let overlap_start = a_start.max(b_start);
    let overlap_end = a_end.min(b_end);
    if overlap_end <= overlap_start || a_end <= a_start {
        return 0.0;
    }
    (overlap_end - overlap_start) as f32 / (a_end - a_start) as f32
}

fn train_model(
    dataset: &StructureDataset,
    config: &Config,
) -> Result<StructureMlp<InferBackend>, TrainError> {
    let device = Default::default();
    TrainBackend::seed(&device, config.seed);
    let mut model = StructureMlp::<TrainBackend>::new(
        dataset.features.cols,
        config.hidden_dim,
        config.hidden_dim2,
        dataset.output_dim,
        &device,
    );
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);
    let mut indices = (0..dataset.rows).collect::<Vec<_>>();
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut batch_features = Vec::new();
    let mut batch_targets = Vec::new();
    for epoch in 0..config.epochs {
        indices.shuffle(&mut rng);
        for batch_rows in indices.chunks(config.batch_size) {
            gather_features(
                &dataset.features.data,
                dataset.features.cols,
                batch_rows,
                &mut batch_features,
            );
            gather_targets(
                &dataset.targets,
                dataset.output_dim,
                batch_rows,
                &mut batch_targets,
            );
            let features = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(
                    batch_features.clone(),
                    [batch_rows.len(), dataset.features.cols],
                ),
                &device,
            );
            let targets = Tensor::<TrainBackend, 2, Int>::from_data(
                TensorData::new(
                    batch_targets.clone(),
                    [batch_rows.len(), dataset.output_dim],
                ),
                &device,
            );
            let logits = model.forward_logits(features);
            let loss = loss_fn.forward(logits, targets);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }
        println!("epoch {}/{} complete", epoch + 1, config.epochs);
    }
    Ok(model.valid())
}

fn predict_probabilities(
    model: &StructureMlp<InferBackend>,
    dataset: &StructureDataset,
    config: &Config,
) -> Result<Vec<f32>, TrainError> {
    let device = Default::default();
    let mut scores = Vec::with_capacity(dataset.rows * dataset.output_dim);
    let batch_size = config.validation_predict_batch_size.max(1);
    for row_start in (0..dataset.rows).step_by(batch_size) {
        let row_end = row_start.saturating_add(batch_size).min(dataset.rows);
        let data_start = row_start * dataset.features.cols;
        let data_end = row_end * dataset.features.cols;
        let features = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(
                dataset.features.data[data_start..data_end].to_vec(),
                [row_end - row_start, dataset.features.cols],
            ),
            &device,
        );
        let batch_scores = sigmoid(model.forward_logits(features))
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| TrainError::Burn(format!("burn tensor readback failed: {error}")))?;
        scores.extend(batch_scores);
    }
    Ok(scores)
}

fn gather_features(data: &[f32], cols: usize, rows: &[usize], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(rows.len() * cols);
    for &row in rows {
        let start = row * cols;
        out.extend_from_slice(&data[start..start + cols]);
    }
}

fn gather_targets(data: &[u8], output_dim: usize, rows: &[usize], out: &mut Vec<i64>) {
    out.clear();
    out.reserve(rows.len() * output_dim);
    for &row in rows {
        let start = row * output_dim;
        out.extend(
            data[start..start + output_dim]
                .iter()
                .copied()
                .map(i64::from),
        );
    }
}

fn tune_and_score_outputs(
    scores: &[f32],
    dataset: &StructureDataset,
    labels: &[String],
) -> Vec<OutputMetricReport> {
    labels
        .iter()
        .enumerate()
        .map(|(output_index, label)| {
            let positives = dataset.positives[output_index];
            let negatives = dataset.rows.saturating_sub(positives);
            let (threshold, metrics) = best_threshold_for_output(
                scores,
                &dataset.targets,
                dataset.output_dim,
                output_index,
            );
            OutputMetricReport {
                label: label.clone(),
                threshold,
                positives,
                negatives,
                metrics,
            }
        })
        .collect()
}

fn best_threshold_for_output(
    scores: &[f32],
    targets: &[u8],
    output_dim: usize,
    output_index: usize,
) -> (f32, BinaryMetricReport) {
    let mut best_threshold = 0.5_f32;
    let mut best_metrics = BinaryMetricReport::default();
    for step in 1..100 {
        let threshold = step as f32 / 100.0;
        let metrics = metric_for_threshold(scores, targets, output_dim, output_index, threshold);
        if step == 1 || metrics.f1 > best_metrics.f1 {
            best_threshold = threshold;
            best_metrics = metrics;
        }
    }
    (best_threshold, best_metrics)
}

fn metric_for_threshold(
    scores: &[f32],
    targets: &[u8],
    output_dim: usize,
    output_index: usize,
    threshold: f32,
) -> BinaryMetricReport {
    let mut true_positives = 0_usize;
    let mut false_positives = 0_usize;
    let mut false_negatives = 0_usize;
    let mut true_negatives = 0_usize;
    for row in 0..(targets.len() / output_dim) {
        let index = row * output_dim + output_index;
        let predicted = scores[index] >= threshold;
        let actual = targets[index] != 0;
        match (predicted, actual) {
            (true, true) => true_positives += 1,
            (true, false) => false_positives += 1,
            (false, true) => false_negatives += 1,
            (false, false) => true_negatives += 1,
        }
    }
    binary_metrics_from_counts(
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    )
}

fn binary_metrics_from_counts(
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    true_negatives: usize,
) -> BinaryMetricReport {
    let precision = if true_positives + false_positives == 0 {
        0.0
    } else {
        true_positives as f64 / (true_positives + false_positives) as f64
    };
    let recall = if true_positives + false_negatives == 0 {
        0.0
    } else {
        true_positives as f64 / (true_positives + false_negatives) as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    BinaryMetricReport {
        precision,
        recall,
        f1,
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    }
}

fn dataset_report(dataset: &StructureDataset, labels: &[String]) -> DatasetReport {
    let positives_by_output = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.clone(), dataset.positives[index]))
        .collect();
    DatasetReport {
        documents: dataset.documents,
        rows: dataset.rows,
        chars: dataset.chars,
        positives_by_output,
    }
}

fn inspect_texts(
    config: &Config,
    kernel: &CompositeFeatureKernel,
    model: &StructureMlp<InferBackend>,
    metrics: &[OutputMetricReport],
) -> Result<(), TrainError> {
    if config.inspect_texts.is_empty() {
        return Ok(());
    }
    let thresholds = metrics
        .iter()
        .map(|metric| (metric.label.clone(), metric.threshold))
        .collect::<BTreeMap<_, _>>();
    for (index, text) in config.inspect_texts.iter().enumerate() {
        let candidates = line_candidates(text);
        let dataset = score_dataset_for_candidates(text, &candidates, kernel, config)?;
        let scores = predict_probabilities(model, &dataset, config)?;
        let semantic_spans = decode_semantic_spans(
            text,
            &candidates,
            &scores,
            &config.labels,
            &thresholds,
            config.min_span_bytes,
            config.merge_gap_bytes,
        );
        let merged = render_merged_annotation(text, &semantic_spans);
        println!(
            "\ninspect_text_{index} bytes={} chars={} line_candidates={} semantic_spans={}",
            text.len(),
            text.chars().count(),
            candidates.len(),
            semantic_spans.len()
        );
        println!("--- text start ---\n{text}\n--- text end ---");
        for candidate_index in 0..candidates.len() {
            let candidate = candidates[candidate_index];
            print!(
                "  line {:>4}..{:<4} {:?}",
                candidate.start,
                candidate.end,
                snippet(text, candidate.start, candidate.end, 100)
            );
            for (label_index, label) in config.labels.iter().enumerate() {
                let raw_score = scores[candidate_index * config.labels.len() + label_index];
                let score = adjusted_label_score(text, candidate, label, raw_score);
                let threshold = thresholds.get(label).copied().unwrap_or(0.5);
                if score >= threshold {
                    print!(" {label}={score:.3}");
                }
            }
            println!();
        }
        println!("  semantic spans:");
        for span in &semantic_spans {
            println!(
                "    {:<9} {:>4}..{:<4} score={:.3} {:?}",
                span.label,
                span.start,
                span.end,
                span.score,
                snippet(text, span.start, span.end, 140)
            );
        }
        println!("  merged annotation:\n{merged}");
    }
    Ok(())
}

fn score_dataset_for_candidates(
    text: &str,
    candidates: &[LineCandidate],
    kernel: &CompositeFeatureKernel,
    config: &Config,
) -> Result<StructureDataset, TrainError> {
    let mut buffer = CandidateBuffer::new();
    for candidate in candidates {
        buffer.push(BytePos::from_usize(candidate.feature_pos));
    }
    let mut features = FeatureMatrix::<f32> {
        rows: buffer.len(),
        cols: kernel.schema().total_dim(),
        data: Vec::new(),
    };
    features.resize_zeroed(buffer.len(), kernel.schema().total_dim());
    kernel.extract_into(
        TextBytes::from_utf8(text),
        buffer.as_slice(),
        features.as_view_mut(),
        &mut FeatureScratch::default(),
    )?;
    Ok(StructureDataset {
        features,
        targets: vec![0; buffer.len() * config.labels.len()],
        rows: buffer.len(),
        output_dim: config.labels.len(),
        documents: 1,
        chars: text.len(),
        positives: vec![0; config.labels.len()],
    })
}

fn decode_semantic_spans(
    text: &str,
    candidates: &[LineCandidate],
    scores: &[f32],
    labels: &[String],
    thresholds: &BTreeMap<String, f32>,
    min_span_bytes: usize,
    merge_gap_bytes: usize,
) -> Vec<DecodedSpan> {
    let mut spans = Vec::new();
    for (label_index, label) in labels.iter().enumerate() {
        let threshold = thresholds.get(label).copied().unwrap_or(0.5);
        let mut current: Option<DecodedSpan> = None;
        for (candidate_index, candidate) in candidates.iter().copied().enumerate() {
            let raw_score = scores[candidate_index * labels.len() + label_index];
            let score = adjusted_label_score(text, candidate, label, raw_score);
            if score < threshold {
                continue;
            }
            let start = line_start_with_prefix(text, candidate.start);
            let end = line_end_with_newline(text, candidate.end);
            if let Some(span) = &mut current {
                if start
                    <= span
                        .end
                        .saturating_add(label_merge_gap(label, merge_gap_bytes))
                {
                    span.end = span.end.max(end);
                    span.score = span.score.max(score);
                    continue;
                }
                if span.end.saturating_sub(span.start) >= min_span_bytes {
                    spans.push(span.clone());
                }
            }
            current = Some(DecodedSpan {
                label: label.clone(),
                start,
                end,
                score,
            });
        }
        if let Some(span) = current {
            if span.end.saturating_sub(span.start) >= min_span_bytes {
                spans.push(span);
            }
        }
    }
    spans.sort_by_key(|span| (span.start, span.end, label_priority(&span.label)));
    resolve_overlapping_semantic_spans(spans)
}

fn label_merge_gap(label: &str, default_gap: usize) -> usize {
    match label {
        "list_item" | "dialogue" => 0,
        _ => default_gap,
    }
}

fn structural_prior_score(text: &str, candidate: LineCandidate, label: &str) -> f32 {
    let line = &text[candidate.start..candidate.end];
    match label {
        "metadata" => metadata_prior_score(text, candidate, line),
        "section" => section_prior_score(line),
        "list_item" => list_item_prior_score(line),
        "dialogue" => dialogue_prior_score(line),
        _ => 0.0,
    }
}

fn adjusted_label_score(text: &str, candidate: LineCandidate, label: &str, raw_score: f32) -> f32 {
    let prior = structural_prior_score(text, candidate, label);
    if label == "metadata"
        && prior == 0.0
        && !is_before_first_blank_line(text, candidate.start)
        && raw_score < 0.50
    {
        return 0.0;
    }
    raw_score.max(prior)
}

fn metadata_prior_score(text: &str, candidate: LineCandidate, line: &str) -> f32 {
    if !is_before_first_blank_line(text, candidate.start) {
        return 0.0;
    }
    if has_metadata_colon(line) {
        return 0.96;
    }
    let lowered = line.trim_start().to_ascii_lowercase();
    if lowered.starts_with("case ")
        || lowered.starts_with("case:")
        || lowered.starts_with("docket")
        || lowered.starts_with("date:")
        || lowered.starts_with("no.")
    {
        return 0.92;
    }
    0.0
}

fn section_prior_score(line: &str) -> f32 {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        return 0.98;
    }
    if trimmed.len() <= 80
        && !trimmed.ends_with('.')
        && trimmed.chars().any(|ch| ch.is_ascii_alphabetic())
        && trimmed
            .chars()
            .filter(|ch| ch.is_ascii_alphabetic())
            .all(|ch| ch.is_ascii_uppercase())
    {
        return 0.86;
    }
    0.0
}

fn list_item_prior_score(line: &str) -> f32 {
    let trimmed = line.trim_start();
    if trimmed.starts_with(['-', '*', '•']) {
        return 0.98;
    }
    let mut chars = trimmed.chars().peekable();
    let mut digits = 0_usize;
    while chars.peek().is_some_and(|ch| ch.is_ascii_digit()) {
        digits += 1;
        chars.next();
    }
    if digits > 0 && matches!(chars.next(), Some('.' | ')')) {
        return 0.94;
    }
    0.0
}

fn dialogue_prior_score(line: &str) -> f32 {
    let trimmed = line.trim_start();
    if trimmed.starts_with(['"', '\'', '“', '‘']) {
        return 0.91;
    }
    let quote_count = trimmed
        .chars()
        .filter(|&ch| matches!(ch, '"' | '“' | '”'))
        .count();
    if quote_count >= 2 {
        return 0.82;
    }
    0.0
}

fn is_before_first_blank_line(text: &str, position: usize) -> bool {
    let prefix = &text[..position.min(text.len())];
    !prefix.contains("\n\n") && !prefix.contains("\r\n\r\n")
}

fn has_metadata_colon(line: &str) -> bool {
    let trimmed = line.trim_start();
    let Some(colon) = trimmed.find(':') else {
        return false;
    };
    colon <= 40
        && trimmed[..colon].chars().any(|ch| ch.is_ascii_alphabetic())
        && !trimmed[..colon].contains('.')
}

fn resolve_overlapping_semantic_spans(mut spans: Vec<DecodedSpan>) -> Vec<DecodedSpan> {
    spans.sort_by_key(|span| (span.start, label_priority(&span.label), span.end));
    let mut resolved: Vec<DecodedSpan> = Vec::new();
    for span in spans {
        if let Some(existing) = resolved
            .iter()
            .position(|prior| ranges_overlap(span.start, span.end, prior.start, prior.end))
        {
            let prior = &resolved[existing];
            if span.score > prior.score + 0.05
                || ((span.score - prior.score).abs() <= 0.05
                    && label_priority(&span.label) < label_priority(&prior.label))
            {
                resolved[existing] = span;
            }
            continue;
        }
        resolved.push(span);
    }
    resolved.sort_by_key(|span| (span.start, span.end));
    resolved
}

fn render_merged_annotation(text: &str, semantic_spans: &[DecodedSpan]) -> String {
    let mut spans = paragraph_spans(text)
        .into_iter()
        .map(|(start, end)| DecodedSpan {
            label: "paragraph".to_string(),
            start,
            end,
            score: 1.0,
        })
        .collect::<Vec<_>>();
    spans.extend_from_slice(semantic_spans);
    let suppress_sentence_spans = semantic_spans
        .iter()
        .filter(|span| matches!(span.label.as_str(), "metadata" | "section" | "list_item"))
        .cloned()
        .collect::<Vec<_>>();
    for (start, end) in sentence_spans(text, &suppress_sentence_spans) {
        spans.push(DecodedSpan {
            label: "sentence".to_string(),
            start,
            end,
            score: 1.0,
        });
    }
    render_spans(text, &spans)
}

fn paragraph_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = next_nonspace_position(text, 0, text.len()).unwrap_or(text.len());
    let bytes = text.as_bytes();
    let mut index = 0_usize;
    while index < bytes.len() {
        if !bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        let run_start = index;
        let mut newlines = 0_usize;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            if matches!(bytes[index], b'\n' | b'\r') {
                newlines += 1;
            }
            index += 1;
        }
        if newlines >= 2 {
            let end = previous_nonspace_end(text, start, run_start).unwrap_or(start);
            if end > start {
                spans.push((start, end));
            }
            start = next_nonspace_position(text, index, text.len()).unwrap_or(text.len());
        }
    }
    if start < text.len() {
        let end = previous_nonspace_end(text, start, text.len()).unwrap_or(text.len());
        if end > start {
            spans.push((start, end));
        }
    }
    spans
}

fn sentence_spans(text: &str, suppress: &[DecodedSpan]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut cursor = next_allowed_nonspace(text, 0, suppress);
    for break_end in sentence_break_candidates(text) {
        let Some(start) = cursor else {
            break;
        };
        if break_end <= start {
            continue;
        }
        if overlaps_any(start, break_end, suppress) {
            cursor = next_allowed_nonspace(text, break_end, suppress);
            continue;
        }
        spans.push((start, break_end));
        cursor = next_allowed_nonspace(text, break_end, suppress);
    }
    if let Some(start) = cursor {
        if start < text.len() && !overlaps_any(start, text.len(), suppress) {
            spans.push((start, text.len()));
        }
    }
    spans
}

fn sentence_break_candidates(text: &str) -> Vec<usize> {
    let mut candidates = Vec::new();
    for (offset, ch) in text.char_indices() {
        if !matches!(ch, '.' | '!' | '?' | '…') {
            continue;
        }
        let terminal_end = offset + ch.len_utf8();
        if previous_char_is_ascii_digit(text, offset)
            && next_char_is_ascii_digit(text, terminal_end)
        {
            continue;
        }
        let break_end = absorb_trailing_closers(text, terminal_end);
        let Some(next_start) = next_nonspace_position(text, break_end, text.len()) else {
            candidates.push(break_end);
            continue;
        };
        let Some(next_ch) = text[next_start..].chars().next() else {
            continue;
        };
        if next_ch.is_uppercase()
            || matches!(next_ch, '"' | '\'' | '“' | '‘' | '#' | '-' | '*' | '•')
        {
            candidates.push(break_end);
        }
    }
    candidates
}

fn render_spans(text: &str, spans: &[DecodedSpan]) -> String {
    let mut valid = spans
        .iter()
        .filter(|span| span.start < span.end && span.end <= text.len())
        .cloned()
        .collect::<Vec<_>>();
    valid.sort_by_key(|span| (span.start, span.end, label_priority(&span.label)));
    let valid = remove_crossing_spans(valid);
    let mut opens: BTreeMap<usize, Vec<&DecodedSpan>> = BTreeMap::new();
    let mut closes: BTreeMap<usize, Vec<&DecodedSpan>> = BTreeMap::new();
    for span in &valid {
        opens.entry(span.start).or_default().push(span);
        closes.entry(span.end).or_default().push(span);
    }
    for values in opens.values_mut() {
        values.sort_by_key(|span| (label_priority(&span.label), std::cmp::Reverse(span.end)));
    }
    for values in closes.values_mut() {
        values.sort_by_key(|span| std::cmp::Reverse(label_priority(&span.label)));
    }
    let mut rendered = String::new();
    let mut cursor = 0_usize;
    let mut boundaries = text
        .char_indices()
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    boundaries.push(text.len());
    boundaries.sort_unstable();
    boundaries.dedup();
    for position in boundaries {
        if position > cursor {
            rendered.push_str(&text[cursor..position]);
            cursor = position;
        }
        if let Some(values) = closes.get(&position) {
            for span in values {
                rendered.push_str("</|");
                rendered.push_str(&span.label);
                rendered.push_str("|>");
            }
        }
        if let Some(values) = opens.get(&position) {
            for span in values {
                rendered.push_str("<|");
                rendered.push_str(&span.label);
                rendered.push_str("|>");
            }
        }
    }
    rendered
}

fn remove_crossing_spans(spans: Vec<DecodedSpan>) -> Vec<DecodedSpan> {
    let mut accepted: Vec<DecodedSpan> = Vec::new();
    'outer: for span in spans {
        for prior in &accepted {
            let crosses =
                span.start < prior.start && prior.start < span.end && span.end < prior.end
                    || prior.start < span.start && span.start < prior.end && prior.end < span.end;
            if crosses {
                continue 'outer;
            }
        }
        accepted.push(span);
    }
    accepted
}

fn label_priority(label: &str) -> usize {
    match label {
        "paragraph" => 0,
        "metadata" => 1,
        "section" => 2,
        "list_item" => 3,
        "dialogue" => 4,
        "sentence" => 5,
        _ => 10,
    }
}

fn line_start_with_prefix(text: &str, position: usize) -> usize {
    text[..position.min(text.len())]
        .rfind(['\n', '\r'])
        .map_or(0, |index| index + 1)
}

fn line_end_with_newline(text: &str, position: usize) -> usize {
    let position = position.min(text.len());
    match text[position..].find(['\n', '\r']) {
        Some(relative) => position + relative,
        None => text.len(),
    }
}

fn next_nonspace_position(text: &str, start: usize, end: usize) -> Option<usize> {
    text.get(start.min(text.len())..end.min(text.len()))?
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(relative, _)| start + relative)
}

fn previous_nonspace_end(text: &str, start: usize, end: usize) -> Option<usize> {
    text.get(start.min(text.len())..end.min(text.len()))?
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(relative, ch)| start + relative + ch.len_utf8())
}

fn next_allowed_nonspace(text: &str, offset: usize, suppress: &[DecodedSpan]) -> Option<usize> {
    let mut cursor = offset;
    loop {
        let next = next_nonspace_position(text, cursor, text.len())?;
        if let Some(span) = suppress
            .iter()
            .find(|span| next >= span.start && next < span.end)
        {
            cursor = span.end;
            continue;
        }
        return Some(next);
    }
}

fn overlaps_any(start: usize, end: usize, spans: &[DecodedSpan]) -> bool {
    spans
        .iter()
        .any(|span| ranges_overlap(start, end, span.start, span.end))
}

fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

fn previous_char_is_ascii_digit(text: &str, offset: usize) -> bool {
    text[..offset]
        .chars()
        .next_back()
        .is_some_and(|ch| ch.is_ascii_digit())
}

fn next_char_is_ascii_digit(text: &str, offset: usize) -> bool {
    text[offset..]
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_digit())
}

fn char_at(text: &str, offset: usize) -> Option<char> {
    text[offset.min(text.len())..].chars().next()
}

fn is_closing_quote_or_bracket_char(ch: char) -> bool {
    matches!(ch, '"' | '\'' | ')' | ']' | '}' | '>' | '”' | '’' | '»')
}

fn absorb_trailing_closers(text: &str, offset: usize) -> usize {
    let mut cursor = offset;
    while cursor < text.len() {
        let Some(ch) = char_at(text, cursor) else {
            break;
        };
        if !is_closing_quote_or_bracket_char(ch) {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn snippet(text: &str, start: usize, end: usize, max_chars: usize) -> String {
    let bounded_start = start.min(text.len());
    let bounded_end = end.min(text.len()).max(bounded_start);
    let slice = &text[bounded_start..bounded_end];
    let mut snippet = String::new();
    for ch in slice.chars().take(max_chars) {
        match ch {
            '\n' => snippet.push_str("\\n"),
            '\r' => snippet.push_str("\\r"),
            other => snippet.push(other),
        }
    }
    if slice.chars().count() > max_chars {
        snippet.push_str("...");
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_candidates_skip_blank_lines() {
        let candidates = line_candidates("A\n\n- B\n");
        assert_eq!(candidates.len(), 2);
        assert_eq!(
            candidates[1],
            LineCandidate {
                start: 3,
                end: 6,
                feature_pos: 3,
            }
        );
    }

    #[test]
    fn merged_annotation_suppresses_sentences_inside_metadata() {
        let text = "Case: X\n\nBody one. Body two.";
        let spans = vec![DecodedSpan {
            label: "metadata".to_string(),
            start: 0,
            end: 7,
            score: 1.0,
        }];
        let rendered = render_merged_annotation(text, &spans);
        assert!(rendered.contains("<|metadata|>Case: X</|metadata|>"));
        assert!(rendered.contains("<|sentence|>Body one.</|sentence|>"));
        assert!(!rendered.contains("<|sentence|><|metadata|>"));
    }

    #[test]
    fn merged_annotation_keeps_sentence_after_section_header() {
        let text = "# Background\nThe court reviewed the invoices. The vendor objected.";
        let spans = vec![DecodedSpan {
            label: "section".to_string(),
            start: 0,
            end: 12,
            score: 1.0,
        }];
        let rendered = render_merged_annotation(text, &spans);
        assert!(rendered.contains("<|section|># Background</|section|>"));
        assert!(rendered.contains("<|sentence|>The court reviewed the invoices.</|sentence|>"));
        assert!(rendered.contains("<|sentence|>The vendor objected.</|sentence|>"));
    }
}
