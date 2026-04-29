use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use charstreamer_backend_burn::{BurnMultiLabelMlpFitOptions, BurnMultiLabelMlpModel};
use charstreamer_core::{
    BytePos, ByteWindowSpec, CandidateBuffer, FeatureKernel, FeatureMatrix, FeatureScratch,
    TextBytes,
};
use charstreamer_kernels::{
    AsciiClassAppender, BoundaryShapeAppender, ByteClass, CompositeFeatureKernel,
    DirectionalByteClassCountAppender, DirectionalUnicodeCategoryGroupCountAppender,
    EncodedByteWindowAppender, LineByteCountAppender, LineByteNgramHashAppender,
    LineContextMetricsAppender, LineShapeMetricsAppender, UnicodeCategoryGroup,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const DEFAULT_LABELS: &[&str] = &["paragraph", "metadata", "section", "list_item", "dialogue"];

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
    validation_inputs: Vec<PathBuf>,
    out_dir: Option<PathBuf>,
    report_path: PathBuf,
    inspect_texts: Vec<String>,
    labels: Vec<String>,
    label_aliases: BTreeMap<String, String>,
    label_positive_repeats: BTreeMap<String, usize>,
    label_loss_weights: BTreeMap<String, f32>,
    label_positive_loss_weights: BTreeMap<String, f32>,
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
    line_ngram_buckets: usize,
    line_ngram_min_n: usize,
    line_ngram_max_n: usize,
    line_context_metrics: bool,
    validation_predict_batch_size: usize,
    min_span_bytes: usize,
    merge_gap_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            validation_inputs: Vec::new(),
            out_dir: None,
            report_path: PathBuf::from("/tmp/charstreamer-structure-burn-report.json"),
            inspect_texts: Vec::new(),
            labels: DEFAULT_LABELS
                .iter()
                .map(|label| (*label).to_string())
                .collect(),
            label_aliases: BTreeMap::new(),
            label_positive_repeats: BTreeMap::new(),
            label_loss_weights: BTreeMap::new(),
            label_positive_loss_weights: BTreeMap::new(),
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
            line_ngram_buckets: 0,
            line_ngram_min_n: 3,
            line_ngram_max_n: 5,
            line_context_metrics: false,
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
    validation_inputs: Vec<PathBuf>,
    labels: Vec<String>,
    label_aliases: BTreeMap<String, String>,
    label_positive_repeats: BTreeMap<String, usize>,
    label_loss_weights: BTreeMap<String, f32>,
    label_positive_loss_weights: BTreeMap<String, f32>,
    feature_dim: usize,
    line_ngram_buckets: usize,
    line_ngram_min_n: usize,
    line_ngram_max_n: usize,
    line_context_metrics: bool,
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
    let (mut records, invalid_train_documents) = load_records(&config, &config.inputs)?;
    if records.is_empty() {
        return Err(Box::new(TrainError::InvalidArgument(
            "need at least one valid training record".to_string(),
        )));
    }

    let mut rng = SmallRng::seed_from_u64(config.seed);
    records.shuffle(&mut rng);
    let (train_records, validation_records, invalid_documents) =
        if config.validation_inputs.is_empty() {
            let loaded_documents = records.len();
            if loaded_documents < 2 {
                return Err(Box::new(TrainError::InvalidArgument(
                    "need at least two valid records for split validation".to_string(),
                )));
            }
            let split_at =
                loaded_documents.saturating_mul(config.split_numerator) / config.split_denominator;
            if split_at == 0 || split_at >= loaded_documents {
                return Err(Box::new(TrainError::InvalidArgument(
                    "invalid train/validation split".to_string(),
                )));
            }
            let validation_records = records.split_off(split_at);
            (records, validation_records, invalid_train_documents)
        } else {
            let (mut validation_records, invalid_validation_documents) =
                load_records(&config, &config.validation_inputs)?;
            if validation_records.is_empty() {
                return Err(Box::new(TrainError::InvalidArgument(
                    "need at least one valid validation record".to_string(),
                )));
            }
            validation_records.shuffle(&mut SmallRng::seed_from_u64(config.seed ^ 0x5EED));
            (
                records,
                validation_records,
                invalid_train_documents + invalid_validation_documents,
            )
        };
    let loaded_documents = train_records.len() + validation_records.len();
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
        validation_inputs: config.validation_inputs.clone(),
        labels: config.labels.clone(),
        label_aliases: config.label_aliases.clone(),
        label_positive_repeats: config.label_positive_repeats.clone(),
        label_loss_weights: config.label_loss_weights.clone(),
        label_positive_loss_weights: config.label_positive_loss_weights.clone(),
        feature_dim: train_dataset.features.cols,
        line_ngram_buckets: config.line_ngram_buckets,
        line_ngram_min_n: config.line_ngram_min_n,
        line_ngram_max_n: config.line_ngram_max_n,
        line_context_metrics: config.line_context_metrics,
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
    if let Some(out_dir) = &config.out_dir {
        write_structure_bundle(out_dir, &model, &report, &config)?;
    }

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
            "--validation-input" => config
                .validation_inputs
                .push(next_path(&mut args, "--validation-input")?),
            "--out" => config.out_dir = Some(next_path(&mut args, "--out")?),
            "--report" => config.report_path = next_path(&mut args, "--report")?,
            "--inspect-text" => config
                .inspect_texts
                .push(next_value(&mut args, "--inspect-text")?),
            "--labels" => config.labels = split_csv(&next_value(&mut args, "--labels")?),
            "--label-alias" => {
                parse_string_map_into(
                    &next_value(&mut args, "--label-alias")?,
                    &mut config.label_aliases,
                    "--label-alias",
                )?;
            }
            "--label-positive-repeat" => {
                parse_usize_map_into(
                    &next_value(&mut args, "--label-positive-repeat")?,
                    &mut config.label_positive_repeats,
                    "--label-positive-repeat",
                )?;
            }
            "--label-loss-weight" => {
                parse_f32_map_into(
                    &next_value(&mut args, "--label-loss-weight")?,
                    &mut config.label_loss_weights,
                    "--label-loss-weight",
                )?;
            }
            "--label-positive-loss-weight" => {
                parse_f32_map_into(
                    &next_value(&mut args, "--label-positive-loss-weight")?,
                    &mut config.label_positive_loss_weights,
                    "--label-positive-loss-weight",
                )?;
            }
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
            "--line-ngram-buckets" => {
                config.line_ngram_buckets = parse_next(&mut args, "--line-ngram-buckets")?;
            }
            "--line-ngram-min-n" => {
                config.line_ngram_min_n = parse_next(&mut args, "--line-ngram-min-n")?;
            }
            "--line-ngram-max-n" => {
                config.line_ngram_max_n = parse_next(&mut args, "--line-ngram-max-n")?;
            }
            "--line-context-metrics" => {
                config.line_context_metrics = true;
            }
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
    if config.line_ngram_buckets > 0
        && (config.line_ngram_min_n == 0 || config.line_ngram_min_n > config.line_ngram_max_n)
    {
        return Err(TrainError::InvalidArgument(
            "line n-gram configuration must satisfy 0 < min_n <= max_n".to_string(),
        ));
    }
    Ok(config)
}

fn print_usage() {
    eprintln!(
        "usage: cargo run -p charstreamer-experiments --example train_synthetic_structure_burn -- \\
  --input <jsonl> [--out <model-dir>] [--report <report.json>] [--inspect-text <text>]"
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

fn parse_string_map_into(
    value: &str,
    out: &mut BTreeMap<String, String>,
    flag: &str,
) -> Result<(), TrainError> {
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (key, value) = part.split_once(':').ok_or_else(|| {
            TrainError::InvalidArgument(format!("{flag} entries must be from:to"))
        })?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return Err(TrainError::InvalidArgument(format!(
                "{flag} entries must be non-empty from:to pairs"
            )));
        }
        out.insert(key.to_string(), value.to_string());
    }
    Ok(())
}

fn parse_usize_map_into(
    value: &str,
    out: &mut BTreeMap<String, usize>,
    flag: &str,
) -> Result<(), TrainError> {
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (key, value) = part.split_once(':').ok_or_else(|| {
            TrainError::InvalidArgument(format!("{flag} entries must be label:n"))
        })?;
        let key = key.trim();
        let repeat = value.trim().parse::<usize>().map_err(|error| {
            TrainError::InvalidArgument(format!(
                "invalid repeat value `{value}` for {flag}: {error}"
            ))
        })?;
        if key.is_empty() || repeat == 0 {
            return Err(TrainError::InvalidArgument(format!(
                "{flag} entries must use non-empty labels and positive repeats"
            )));
        }
        out.insert(key.to_string(), repeat);
    }
    Ok(())
}

fn parse_f32_map_into(
    value: &str,
    out: &mut BTreeMap<String, f32>,
    flag: &str,
) -> Result<(), TrainError> {
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (key, value) = part.split_once(':').ok_or_else(|| {
            TrainError::InvalidArgument(format!("{flag} entries must be label:x"))
        })?;
        let key = key.trim();
        let weight = value.trim().parse::<f32>().map_err(|error| {
            TrainError::InvalidArgument(format!(
                "invalid weight value `{value}` for {flag}: {error}"
            ))
        })?;
        if key.is_empty() || !weight.is_finite() || weight <= 0.0 {
            return Err(TrainError::InvalidArgument(format!(
                "{flag} entries must use non-empty labels and positive finite weights"
            )));
        }
        out.insert(key.to_string(), weight);
    }
    Ok(())
}

fn load_records(
    config: &Config,
    paths: &[PathBuf],
) -> Result<(Vec<CleanRecord>, usize), TrainError> {
    let allowed_labels = config
        .labels
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut records = Vec::new();
    let mut invalid = 0_usize;
    for path in paths {
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
            match clean_record(parsed, &allowed_labels, &config.label_aliases) {
                Some(record) => records.push(record),
                None => invalid += 1,
            }
        }
    }
    Ok((records, invalid))
}

fn clean_record(
    record: SyntheticRecord,
    allowed_labels: &BTreeSet<&str>,
    label_aliases: &BTreeMap<String, String>,
) -> Option<CleanRecord> {
    if record.text.is_empty() {
        return None;
    }
    let mut spans = Vec::new();
    for mut span in record.spans {
        if let Some(alias) = label_aliases.get(span.label.as_str()) {
            span.label.clone_from(alias);
        }
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
    let mut appenders: Vec<Box<dyn charstreamer_core::FeatureAppender<f32> + Send + Sync>> = vec![
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
        Box::new(LineShapeMetricsAppender::new()),
    ];
    if config.line_ngram_buckets > 0 {
        appenders.push(Box::new(LineByteNgramHashAppender::new(
            config.line_ngram_buckets,
            config.line_ngram_min_n,
            config.line_ngram_max_n,
        )));
    }
    if config.line_context_metrics {
        appenders.push(Box::new(LineContextMetricsAppender::new()));
    }
    CompositeFeatureKernel::new(appenders)
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
            let repeats = if sample_negatives {
                positive_repeat(&row_targets, &config.labels, &config.label_positive_repeats)
            } else {
                1
            };
            for _ in 0..repeats {
                candidate_buffer.push(BytePos::from_usize(candidate.feature_pos));
                for (index, value) in row_targets.iter().copied().enumerate() {
                    dataset.positives[index] += usize::from(value != 0);
                }
                selected_targets.extend_from_slice(&row_targets);
            }
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
                    && overlap_ratio(candidate.start, candidate.end, span.start, span.end) >= 0.35
            }) as u8
        })
        .collect()
}

fn positive_repeat(
    row_targets: &[u8],
    labels: &[String],
    label_positive_repeats: &BTreeMap<String, usize>,
) -> usize {
    row_targets
        .iter()
        .zip(labels)
        .filter_map(|(&target, label)| {
            if target == 0 {
                None
            } else {
                label_positive_repeats.get(label).copied()
            }
        })
        .max()
        .unwrap_or(1)
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
) -> Result<BurnMultiLabelMlpModel, TrainError> {
    let (model, _) = BurnMultiLabelMlpModel::fit_multilabel(
        dataset.features.as_view(),
        &dataset.targets,
        &BurnMultiLabelMlpFitOptions {
            hidden_dim1: config.hidden_dim,
            hidden_dim2: config.hidden_dim2,
            output_dim: dataset.output_dim,
            class_weights: class_weights(&config.labels, &config.label_loss_weights),
            positive_weights: class_weights(&config.labels, &config.label_positive_loss_weights),
            epochs: config.epochs,
            batch_size: config.batch_size,
            learning_rate: config.learning_rate,
            seed: config.seed,
        },
        &mut charstreamer_core::FitScratch::default(),
    )
    .map_err(|error| TrainError::Burn(format!("burn multi-label training failed: {error}")))?;
    Ok(model)
}

fn class_weights(labels: &[String], label_loss_weights: &BTreeMap<String, f32>) -> Vec<f32> {
    if label_loss_weights.is_empty() {
        return Vec::new();
    }
    labels
        .iter()
        .map(|label| label_loss_weights.get(label).copied().unwrap_or(1.0))
        .collect()
}

fn predict_probabilities(
    model: &BurnMultiLabelMlpModel,
    dataset: &StructureDataset,
    config: &Config,
) -> Result<Vec<f32>, TrainError> {
    let mut scores = vec![0.0_f32; dataset.rows * dataset.output_dim];
    let _ = config;
    model
        .predict_flat_into(dataset.features.as_view(), &mut scores)
        .map_err(|error| TrainError::Burn(format!("burn prediction failed: {error}")))?;
    Ok(scores)
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

fn write_structure_bundle(
    out_dir: &PathBuf,
    model: &BurnMultiLabelMlpModel,
    report: &TrainingReport,
    config: &Config,
) -> Result<(), TrainError> {
    fs::create_dir_all(out_dir)?;
    model
        .save_named_mpk(out_dir.join("semantic_structure"))
        .map_err(|error| TrainError::Burn(format!("structure model save failed: {error}")))?;
    let payload_path = out_dir.join("semantic_structure.mpk");
    let payload_bytes = fs::metadata(&payload_path)?.len();
    let thresholds = report
        .output_metrics
        .iter()
        .map(|metric| (metric.label.clone(), metric.threshold))
        .collect::<BTreeMap<_, _>>();
    let structure_value = serde_json::json!({
        "engine": "burn_multilabel_mlp_structure_v1",
        "labels": config.labels.clone(),
        "features": {
            "encoded_left": config.encoded_left,
            "encoded_right": config.encoded_right,
            "count_radius": config.count_radius,
            "line_ngram_buckets": config.line_ngram_buckets,
            "line_ngram_min_n": config.line_ngram_min_n,
            "line_ngram_max_n": config.line_ngram_max_n,
            "line_context_metrics": config.line_context_metrics,
            "feature_dim": report.feature_dim,
            "hidden_dim1": config.hidden_dim,
            "hidden_dim2": config.hidden_dim2,
            "output_dim": config.labels.len()
        },
        "thresholds": thresholds
    });
    let manifest_path = out_dir.join("manifest.json");
    let mut manifest = if manifest_path.is_file() {
        serde_json::from_slice::<serde_json::Value>(&fs::read(&manifest_path)?)?
    } else {
        serde_json::json!({
            "format": "charstreamer.model-bundle.v1",
            "name": "charstreamer-default",
            "version": env!("CARGO_PKG_VERSION"),
            "engine": "burn_shallow_mlp_sentence_v1",
            "task": "combined_segmentation",
            "files": []
        })
    };
    manifest["task"] = serde_json::json!("combined_segmentation");
    manifest["structure"] = structure_value;
    let files = manifest["files"].as_array_mut().ok_or_else(|| {
        TrainError::InvalidArgument("manifest files must be an array".to_string())
    })?;
    files.retain(|file| {
        file.get("role").and_then(|role| role.as_str()) != Some("semantic_structure")
    });
    files.push(serde_json::json!({
        "path": "semantic_structure.mpk",
        "role": "semantic_structure",
        "bytes": payload_bytes
    }));
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    fs::write(
        out_dir.join("structure-training-report.json"),
        serde_json::to_vec_pretty(report)?,
    )?;
    Ok(())
}

fn inspect_texts(
    config: &Config,
    kernel: &CompositeFeatureKernel,
    model: &BurnMultiLabelMlpModel,
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
        let merged = render_spans(text, &semantic_spans);
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
                let score = raw_score;
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
            let score = raw_score;
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
    remove_crossing_spans(spans)
}

fn label_merge_gap(label: &str, default_gap: usize) -> usize {
    match label {
        "list_item" | "dialogue" => 0,
        _ => default_gap,
    }
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
        values.sort_by_key(|span| (std::cmp::Reverse(span.end), label_priority(&span.label)));
    }
    for values in closes.values_mut() {
        values.sort_by_key(|span| {
            (
                std::cmp::Reverse(span.start),
                std::cmp::Reverse(label_priority(&span.label)),
            )
        });
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
    let mut ordered = spans;
    ordered.sort_by_key(|span| {
        (
            label_priority(&span.label),
            span.start,
            std::cmp::Reverse(span.end),
        )
    });
    let mut accepted: Vec<DecodedSpan> = Vec::new();
    for span in ordered {
        if !accepted
            .iter()
            .any(|prior| ranges_cross(span.start, span.end, prior.start, prior.end))
        {
            accepted.push(span);
        }
    }
    accepted.sort_by_key(|span| (span.start, span.end, label_priority(&span.label)));
    accepted
}

fn ranges_cross(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    ranges_overlap(a_start, a_end, b_start, b_end)
        && !range_contains(a_start, a_end, b_start, b_end)
        && !range_contains(b_start, b_end, a_start, a_end)
}

fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

fn range_contains(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_start && b_end <= a_end
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
}
