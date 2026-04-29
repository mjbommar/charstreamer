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
    EncodedByteWindowAppender, LegalBoundaryHeuristicAppender, LineByteCountAppender,
    UnicodeCategoryGroup,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

type InferBackend = NdArray<f32>;
type TrainBackend = Autodiff<InferBackend>;

const TASKS: [BoundaryTask; 2] = [BoundaryTask::SentenceBreak, BoundaryTask::ParagraphBreak];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum BoundaryTask {
    SentenceBreak,
    ParagraphBreak,
}

impl BoundaryTask {
    fn as_str(self) -> &'static str {
        match self {
            Self::SentenceBreak => "sentence_break",
            Self::ParagraphBreak => "paragraph_break",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::SentenceBreak => "sentence",
            Self::ParagraphBreak => "paragraph",
        }
    }
}

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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            report_path: PathBuf::from("/tmp/charstreamer-synthetic-boundary-burn-report.json"),
            inspect_texts: Vec::new(),
            epochs: 16,
            batch_size: 2048,
            hidden_dim: 64,
            hidden_dim2: 32,
            learning_rate: 1.0e-3,
            seed: 19,
            split_numerator: 8,
            split_denominator: 10,
            negative_keep_rate: 1.0,
            max_records: None,
            encoded_left: 7,
            encoded_right: 7,
            count_radius: 24,
            validation_predict_batch_size: 32_768,
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
    right_open: bool,
}

#[derive(Clone, Debug)]
struct CleanRecord {
    text: String,
    spans: Vec<SyntheticSpan>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BoundaryCandidate {
    feature_pos: usize,
    break_end: usize,
}

#[derive(Debug)]
struct BoundaryDataset {
    features: FeatureMatrix<f32>,
    targets: Vec<u8>,
    masks: Vec<u8>,
    rows: usize,
    output_dim: usize,
    documents: usize,
    chars: usize,
    positives: Vec<usize>,
    eligible: Vec<usize>,
}

#[derive(Module, Debug)]
struct BoundaryMlp<B: Backend> {
    input: Linear<B>,
    activation1: Relu,
    hidden2: Linear<B>,
    activation2: Relu,
    output: Linear<B>,
}

impl<B: Backend> BoundaryMlp<B> {
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
    task: String,
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
    eligible_by_output: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingReport {
    inputs: Vec<PathBuf>,
    model: String,
    outputs: Vec<String>,
    feature_dim: usize,
    hidden_dim: usize,
    hidden_dim2: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
    negative_keep_rate: f32,
    encoded_left: usize,
    encoded_right: usize,
    count_radius: usize,
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

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let started = Instant::now();
    let (mut records, invalid_documents) = load_records(&config)?;
    let loaded_documents = records.len();
    if loaded_documents < 2 {
        return Err(Box::new(TrainError::InvalidArgument(
            "need at least two valid synthetic records".to_string(),
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

    let output_metrics = tune_and_score_outputs(&scores, &validation_dataset);
    let macro_f1 = output_metrics
        .iter()
        .map(|metric| metric.metrics.f1)
        .sum::<f64>()
        / output_metrics.len().max(1) as f64;
    let validation_end_to_end_seconds = feature_seconds_validation + validation_predict_seconds;
    let outputs = TASKS
        .iter()
        .map(|task| task.as_str().to_string())
        .collect::<Vec<_>>();
    let report = TrainingReport {
        inputs: config.inputs.clone(),
        model: "burn_mlp".to_string(),
        outputs,
        feature_dim: train_dataset.features.cols,
        hidden_dim: config.hidden_dim,
        hidden_dim2: config.hidden_dim2,
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        seed: config.seed,
        negative_keep_rate: config.negative_keep_rate,
        encoded_left: config.encoded_left,
        encoded_right: config.encoded_right,
        count_radius: config.count_radius,
        loaded_documents,
        invalid_documents,
        train: dataset_report(&train_dataset),
        validation: dataset_report(&validation_dataset),
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
        "boundary_model=burn_mlp hidden={}/{} epochs={} batch={} lr={} train_s={:.3} valid_predict_s={:.3} valid_rows_per_s={:.1} valid_e2e_chars_per_s={:.1} macro_f1={:.4}",
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
            metric.task,
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
    if config.epochs == 0 || config.batch_size == 0 || config.hidden_dim == 0 {
        return Err(TrainError::InvalidArgument(
            "epochs, batch-size, and hidden dims must be greater than zero".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.negative_keep_rate) {
        return Err(TrainError::InvalidArgument(
            "--negative-keep-rate must be between 0 and 1".to_string(),
        ));
    }
    Ok(config)
}

fn print_usage() {
    eprintln!(
        "usage: cargo run -p charstreamer-experiments --example train_synthetic_boundary_burn -- \\
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

fn load_records(config: &Config) -> Result<(Vec<CleanRecord>, usize), TrainError> {
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
            match clean_record(parsed) {
                Some(record) => records.push(record),
                None => invalid += 1,
            }
        }
    }
    Ok((records, invalid))
}

fn clean_record(record: SyntheticRecord) -> Option<CleanRecord> {
    if record.text.is_empty() {
        return None;
    }
    let mut spans = Vec::new();
    for span in record.spans {
        if !matches!(span.label.as_str(), "sentence" | "paragraph") {
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
        Box::new(LegalBoundaryHeuristicAppender::new()),
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
) -> Result<BoundaryDataset, TrainError> {
    let output_dim = TASKS.len();
    let mut dataset = BoundaryDataset {
        features: FeatureMatrix {
            rows: 0,
            cols: kernel.schema().total_dim(),
            data: Vec::new(),
        },
        targets: Vec::new(),
        masks: Vec::new(),
        rows: 0,
        output_dim,
        documents: 0,
        chars: 0,
        positives: vec![0; output_dim],
        eligible: vec![0; output_dim],
    };
    let mut rng = SmallRng::seed_from_u64(config.seed ^ if sample_negatives { 0xB5 } else { 0x71 });
    let mut candidate_buffer = CandidateBuffer::new();
    let mut feature_scratch = FeatureScratch::default();
    let mut doc_features = FeatureMatrix::<f32>::default();
    let mut selected_targets = Vec::new();
    let mut selected_masks = Vec::new();

    for record in records {
        let candidates = boundary_candidates(record, true);
        if candidates.is_empty() {
            continue;
        }
        candidate_buffer.clear();
        selected_targets.clear();
        selected_masks.clear();

        for candidate in candidates {
            let row_targets = targets_for_candidate(record, candidate);
            let row_masks = output_masks_for_candidate(&record.text, candidate, &row_targets);
            let has_positive = row_targets.iter().any(|&target| target != 0);
            if sample_negatives && !has_positive && rng.random::<f32>() > config.negative_keep_rate
            {
                continue;
            }
            candidate_buffer.push(BytePos::from_usize(candidate.feature_pos));
            for (index, value) in row_targets.iter().copied().enumerate() {
                dataset.positives[index] += usize::from(value != 0);
            }
            for (index, value) in row_masks.iter().copied().enumerate() {
                dataset.eligible[index] += usize::from(value != 0);
            }
            selected_targets.extend_from_slice(&row_targets);
            selected_masks.extend_from_slice(&row_masks);
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
        dataset.masks.extend_from_slice(&selected_masks);
        dataset.rows += candidate_buffer.len();
        dataset.documents += 1;
        dataset.chars += record.text.len();
    }

    Ok(dataset)
}

fn boundary_candidates(record: &CleanRecord, include_gold: bool) -> Vec<BoundaryCandidate> {
    let mut candidates = BTreeSet::new();
    for candidate in sentence_punctuation_candidates(&record.text) {
        candidates.insert(candidate);
    }
    for candidate in paragraph_structure_candidates(&record.text) {
        candidates.insert(candidate);
    }
    if include_gold {
        for span in &record.spans {
            if span.right_open || span.end == 0 {
                continue;
            }
            if matches!(span.label.as_str(), "sentence" | "paragraph") {
                if let Some(candidate) = candidate_for_break_end(&record.text, span.end) {
                    candidates.insert(candidate);
                }
            }
        }
    }
    candidates.into_iter().collect()
}

fn sentence_punctuation_candidates(text: &str) -> Vec<BoundaryCandidate> {
    let mut candidates = Vec::new();
    for (offset, ch) in text.char_indices() {
        if !is_sentence_terminal_char(ch) {
            continue;
        }
        let terminal_end = offset + ch.len_utf8();
        if previous_char_is_ascii_digit(text, offset)
            && next_char_is_ascii_digit(text, terminal_end)
        {
            continue;
        }
        let break_end = absorb_trailing_closers(text, terminal_end);
        let Some(next_start) = next_nonspace_position(text, break_end) else {
            candidates.push(BoundaryCandidate {
                feature_pos: offset,
                break_end,
            });
            continue;
        };
        let Some(next_ch) = text[next_start..].chars().next() else {
            continue;
        };
        if next_ch.is_uppercase()
            || matches!(next_ch, '"' | '\'' | '“' | '‘' | '#' | '-' | '*' | '•')
        {
            candidates.push(BoundaryCandidate {
                feature_pos: offset,
                break_end,
            });
        }
    }
    candidates
}

fn paragraph_structure_candidates(text: &str) -> Vec<BoundaryCandidate> {
    let mut candidates = Vec::new();
    if !text.is_empty() {
        if let Some(candidate) = candidate_for_break_end(text, text.len()) {
            candidates.push(candidate);
        }
    }
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
        if newlines >= 2 && run_start > 0 {
            if let Some(candidate) = candidate_for_break_end(text, run_start) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

fn targets_for_candidate(record: &CleanRecord, candidate: BoundaryCandidate) -> Vec<u8> {
    TASKS
        .iter()
        .map(|task| {
            record.spans.iter().any(|span| {
                span.label == task.label() && !span.right_open && span.end == candidate.break_end
            }) as u8
        })
        .collect()
}

fn output_masks_for_candidate(text: &str, candidate: BoundaryCandidate, targets: &[u8]) -> Vec<u8> {
    TASKS
        .iter()
        .enumerate()
        .map(|(index, task)| {
            if targets[index] != 0 {
                return 1;
            }
            (match task {
                BoundaryTask::SentenceBreak => sentence_candidate_like(text, candidate),
                BoundaryTask::ParagraphBreak => paragraph_candidate_like(text, candidate),
            }) as u8
        })
        .collect()
}

fn sentence_candidate_like(text: &str, candidate: BoundaryCandidate) -> bool {
    let Some(ch) = char_at(text, candidate.feature_pos) else {
        return false;
    };
    if !is_sentence_terminal_char(ch) {
        return false;
    }
    let terminal_end = candidate.feature_pos + ch.len_utf8();
    absorb_trailing_closers(text, terminal_end) == candidate.break_end
}

fn paragraph_candidate_like(text: &str, candidate: BoundaryCandidate) -> bool {
    candidate.break_end == text.len() || newline_count_after(text, candidate.break_end) >= 2
}

fn train_model(
    dataset: &BoundaryDataset,
    config: &Config,
) -> Result<BoundaryMlp<InferBackend>, TrainError> {
    let device = Default::default();
    TrainBackend::seed(&device, config.seed);
    let mut model = BoundaryMlp::<TrainBackend>::new(
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
    model: &BoundaryMlp<InferBackend>,
    dataset: &BoundaryDataset,
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

fn tune_and_score_outputs(scores: &[f32], dataset: &BoundaryDataset) -> Vec<OutputMetricReport> {
    TASKS
        .iter()
        .enumerate()
        .map(|(output_index, task)| {
            let positives = dataset.positives[output_index];
            let eligible = dataset.eligible[output_index];
            let negatives = eligible.saturating_sub(positives);
            let (threshold, metrics) = best_threshold_for_output(
                scores,
                &dataset.targets,
                &dataset.masks,
                dataset.output_dim,
                output_index,
            );
            OutputMetricReport {
                task: task.as_str().to_string(),
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
    masks: &[u8],
    output_dim: usize,
    output_index: usize,
) -> (f32, BinaryMetricReport) {
    let mut best_threshold = 0.5_f32;
    let mut best_metrics = BinaryMetricReport::default();
    for step in 1..100 {
        let threshold = step as f32 / 100.0;
        let metrics =
            metric_for_threshold(scores, targets, masks, output_dim, output_index, threshold);
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
    masks: &[u8],
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
        if masks[index] == 0 {
            continue;
        }
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

fn inspect_texts(
    config: &Config,
    kernel: &CompositeFeatureKernel,
    model: &BoundaryMlp<InferBackend>,
    output_metrics: &[OutputMetricReport],
) -> Result<(), TrainError> {
    if config.inspect_texts.is_empty() {
        return Ok(());
    }
    let thresholds = output_metrics
        .iter()
        .map(|metric| (metric.task.clone(), metric.threshold))
        .collect::<BTreeMap<_, _>>();
    for (index, text) in config.inspect_texts.iter().enumerate() {
        let record = CleanRecord {
            text: text.clone(),
            spans: Vec::new(),
        };
        let candidates = boundary_candidates(&record, false);
        let dataset = score_dataset_for_candidates(text, &candidates, kernel)?;
        let scores = predict_probabilities(model, &dataset, config)?;
        let sentence_threshold = thresholds.get("sentence_break").copied().unwrap_or(0.5);
        let paragraph_threshold = thresholds.get("paragraph_break").copied().unwrap_or(0.5);
        let sentence_breaks = selected_breaks(text, &candidates, &scores, 0, sentence_threshold);
        let paragraph_breaks = selected_breaks(text, &candidates, &scores, 1, paragraph_threshold);

        println!(
            "\ninspect_text_{index} bytes={} chars={} candidates={}",
            text.len(),
            text.chars().count(),
            candidates.len()
        );
        println!("--- text start ---\n{text}\n--- text end ---");
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let sentence_score = scores[candidate_index * TASKS.len()];
            let paragraph_score = scores[candidate_index * TASKS.len() + 1];
            println!(
                "  break_end={:>4} sentence={:.3} paragraph={:.3} {:?}",
                candidate.break_end,
                sentence_score,
                paragraph_score,
                point_context(text, candidate.break_end, 54)
            );
        }
        println!(
            "  sentence_breaks threshold={sentence_threshold:.2}: {:?}",
            sentence_breaks
        );
        println!(
            "  sentence annotated:\n{}",
            render_sentence_breaks(text, &sentence_breaks)
        );
        println!(
            "  paragraph_breaks threshold={paragraph_threshold:.2}: {:?}",
            paragraph_breaks
        );
    }
    Ok(())
}

fn score_dataset_for_candidates(
    text: &str,
    candidates: &[BoundaryCandidate],
    kernel: &CompositeFeatureKernel,
) -> Result<BoundaryDataset, TrainError> {
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
    let mut masks = Vec::with_capacity(buffer.len() * TASKS.len());
    let mut eligible = vec![0; TASKS.len()];
    for candidate in candidates {
        let targets = [0, 0];
        let row_masks = output_masks_for_candidate(text, *candidate, &targets);
        for (index, value) in row_masks.iter().copied().enumerate() {
            eligible[index] += usize::from(value != 0);
        }
        masks.extend_from_slice(&row_masks);
    }
    Ok(BoundaryDataset {
        features,
        targets: vec![0; buffer.len() * TASKS.len()],
        masks,
        rows: buffer.len(),
        output_dim: TASKS.len(),
        documents: 1,
        chars: text.len(),
        positives: vec![0; TASKS.len()],
        eligible,
    })
}

fn selected_breaks(
    text: &str,
    candidates: &[BoundaryCandidate],
    scores: &[f32],
    output_index: usize,
    threshold: f32,
) -> Vec<usize> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let score = scores[index * TASKS.len() + output_index];
            let targets = [0, 0];
            let masks = output_masks_for_candidate(text, *candidate, &targets);
            (masks[output_index] != 0 && score >= threshold).then_some(candidate.break_end)
        })
        .collect()
}

fn dataset_report(dataset: &BoundaryDataset) -> DatasetReport {
    let positives_by_output = TASKS
        .iter()
        .enumerate()
        .map(|(index, task)| (task.as_str().to_string(), dataset.positives[index]))
        .collect();
    let eligible_by_output = TASKS
        .iter()
        .enumerate()
        .map(|(index, task)| (task.as_str().to_string(), dataset.eligible[index]))
        .collect();
    DatasetReport {
        documents: dataset.documents,
        rows: dataset.rows,
        chars: dataset.chars,
        positives_by_output,
        eligible_by_output,
    }
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

fn next_nonspace_position(text: &str, offset: usize) -> Option<usize> {
    text[offset..]
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(relative, _)| offset + relative)
}

fn newline_count_after(text: &str, offset: usize) -> usize {
    text[offset.min(text.len())..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .filter(|&ch| matches!(ch, '\n' | '\r'))
        .count()
}

fn previous_nonspace_position(text: &str, offset: usize) -> Option<usize> {
    text[..offset.min(text.len())]
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(position, _)| position)
}

fn previous_char(text: &str, offset: usize) -> Option<(usize, char)> {
    text[..offset.min(text.len())].char_indices().next_back()
}

fn char_at(text: &str, offset: usize) -> Option<char> {
    text[offset.min(text.len())..].chars().next()
}

fn is_sentence_terminal_char(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '…')
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

fn candidate_for_break_end(text: &str, break_end: usize) -> Option<BoundaryCandidate> {
    if break_end == 0 || break_end > text.len() || !text.is_char_boundary(break_end) {
        return None;
    }
    let feature_pos = terminal_feature_pos_for_break_end(text, break_end)
        .or_else(|| previous_nonspace_position(text, break_end))?;
    Some(BoundaryCandidate {
        feature_pos,
        break_end,
    })
}

fn terminal_feature_pos_for_break_end(text: &str, break_end: usize) -> Option<usize> {
    let mut cursor = break_end.min(text.len());
    while let Some((position, ch)) = previous_char(text, cursor) {
        if ch.is_whitespace() || is_closing_quote_or_bracket_char(ch) {
            cursor = position;
            continue;
        }
        return is_sentence_terminal_char(ch).then_some(position);
    }
    None
}

fn render_sentence_breaks(text: &str, breaks: &[usize]) -> String {
    let mut rendered = String::new();
    let mut cursor = 0_usize;
    let mut span_start = next_nonspace_position(text, 0).unwrap_or(0);
    for &break_end in breaks {
        if break_end <= span_start || break_end > text.len() {
            continue;
        }
        rendered.push_str(&text[cursor..span_start]);
        rendered.push_str("<|sentence|>");
        rendered.push_str(&text[span_start..break_end]);
        rendered.push_str("<|/sentence|>");
        cursor = break_end;
        span_start = next_nonspace_position(text, break_end).unwrap_or(text.len());
    }
    if span_start < text.len() {
        rendered.push_str(&text[cursor..span_start]);
        rendered.push_str("<|sentence|>");
        rendered.push_str(&text[span_start..]);
        rendered.push_str("<|/sentence|>");
    } else {
        rendered.push_str(&text[cursor..]);
    }
    rendered
}

fn point_context(text: &str, position: usize, radius: usize) -> String {
    let position = position.min(text.len());
    let mut start = position.saturating_sub(radius);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = position.saturating_add(radius).min(text.len());
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    let left = snippet(text, start, position, radius);
    let right = snippet(text, position, end, radius);
    format!("{left}|{right}")
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
    fn sentence_candidates_use_terminal_byte_as_feature_position() {
        let candidates = sentence_punctuation_candidates("One. Two.");
        assert_eq!(
            candidates,
            vec![
                BoundaryCandidate {
                    feature_pos: 3,
                    break_end: 4,
                },
                BoundaryCandidate {
                    feature_pos: 8,
                    break_end: 9,
                },
            ]
        );
    }

    #[test]
    fn sentence_candidates_absorb_closing_quotes() {
        let candidates = sentence_punctuation_candidates("\"One.\" Two.");
        assert_eq!(
            candidates[0],
            BoundaryCandidate {
                feature_pos: 4,
                break_end: 6,
            }
        );
    }

    #[test]
    fn paragraph_candidates_use_terminal_feature_position_before_quote() {
        let candidates = paragraph_structure_candidates("\"One.\"\n\nTwo.");
        assert!(candidates.contains(&BoundaryCandidate {
            feature_pos: 4,
            break_end: 6,
        }));
    }

    #[test]
    fn adjacent_sentence_boundary_is_positive() {
        let record = CleanRecord {
            text: "One. Two.".to_string(),
            spans: vec![
                SyntheticSpan {
                    label: "sentence".to_string(),
                    start: 0,
                    end: 4,
                    right_open: false,
                },
                SyntheticSpan {
                    label: "sentence".to_string(),
                    start: 4,
                    end: 9,
                    right_open: false,
                },
            ],
        };
        let targets = targets_for_candidate(
            &record,
            BoundaryCandidate {
                feature_pos: 3,
                break_end: 4,
            },
        );
        assert_eq!(targets, vec![1, 0]);
    }

    #[test]
    fn open_sentence_end_is_not_positive() {
        let record = CleanRecord {
            text: "One.".to_string(),
            spans: vec![SyntheticSpan {
                label: "sentence".to_string(),
                start: 0,
                end: 4,
                right_open: true,
            }],
        };
        let targets = targets_for_candidate(
            &record,
            BoundaryCandidate {
                feature_pos: 3,
                break_end: 4,
            },
        );
        assert_eq!(targets, vec![0, 0]);
    }

    #[test]
    fn dataset_masks_stay_aligned_with_targets_across_documents() {
        let records = vec![
            CleanRecord {
                text: "One. Two.".to_string(),
                spans: vec![SyntheticSpan {
                    label: "sentence".to_string(),
                    start: 0,
                    end: 4,
                    right_open: false,
                }],
            },
            CleanRecord {
                text: "\"Three.\"\n\nFour.".to_string(),
                spans: vec![SyntheticSpan {
                    label: "paragraph".to_string(),
                    start: 0,
                    end: 8,
                    right_open: false,
                }],
            },
        ];
        let config = Config::default();
        let kernel = build_kernel(&config);
        let dataset =
            build_dataset(&records, &kernel, &config, false).expect("dataset should build");
        assert_eq!(dataset.targets.len(), dataset.rows * TASKS.len());
        assert_eq!(dataset.masks.len(), dataset.targets.len());
        assert!(dataset.eligible.iter().all(|&value| value > 0));
    }
}
