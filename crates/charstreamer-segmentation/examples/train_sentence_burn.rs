use charstreamer_backend_burn::{BurnShallowMlpFitOptions, BurnShallowMlpModel};
use charstreamer_core::{
    BatchPredictor, BytePos, CandidateBuffer, DatasetView, FeatureKernel, FeatureMatrix,
    FeatureScratch, FitScratch, TextBytes, TrainablePredictor, metrics_from_scores,
};
use charstreamer_segmentation::{
    BurnSentenceFeatureConfig, burn_sentence_kernel, sentence_boundary_candidates,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const MODEL_FORMAT: &str = "charstreamer.model-bundle.v1";
const MODEL_NAME: &str = "charstreamer-default";
const MODEL_ENGINE: &str = "burn_shallow_mlp_sentence_v1";
const SENTENCE_TAG: &str = "<|sentence|>";
const PARAGRAPH_TAG: &str = "<|paragraph|>";

#[derive(Clone, Debug)]
struct Args {
    inputs: Vec<PathBuf>,
    out_dir: PathBuf,
    report_path: Option<PathBuf>,
    max_records: Option<usize>,
    hidden_dim: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
    threshold: Option<f32>,
    encoded_left: usize,
    encoded_right: usize,
    count_radius: usize,
    negative_keep_rate: f32,
    version: String,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut inputs = Vec::new();
        let mut out_dir = None;
        let mut report_path = None;
        let mut max_records = None;
        let mut hidden_dim = 64_usize;
        let mut epochs = 12_usize;
        let mut batch_size = 256_usize;
        let mut learning_rate = 1.0e-3_f64;
        let mut seed = 7_u64;
        let mut threshold = None;
        let mut encoded_left = 7_usize;
        let mut encoded_right = 7_usize;
        let mut count_radius = 24_usize;
        let mut negative_keep_rate = 0.02_f32;
        let mut version = env!("CARGO_PKG_VERSION").to_owned();

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => inputs.push(PathBuf::from(value_after(&arg, &mut args)?)),
                "--out" => out_dir = Some(PathBuf::from(value_after(&arg, &mut args)?)),
                "--report" => report_path = Some(PathBuf::from(value_after(&arg, &mut args)?)),
                "--max-records" => {
                    max_records = Some(parse_value(&arg, &mut args)?);
                }
                "--hidden-dim" => hidden_dim = parse_value(&arg, &mut args)?,
                "--epochs" => epochs = parse_value(&arg, &mut args)?,
                "--batch-size" => batch_size = parse_value(&arg, &mut args)?,
                "--learning-rate" => learning_rate = parse_value(&arg, &mut args)?,
                "--seed" => seed = parse_value(&arg, &mut args)?,
                "--threshold" => threshold = Some(parse_value(&arg, &mut args)?),
                "--encoded-left" => encoded_left = parse_value(&arg, &mut args)?,
                "--encoded-right" => encoded_right = parse_value(&arg, &mut args)?,
                "--count-radius" => count_radius = parse_value(&arg, &mut args)?,
                "--negative-keep-rate" => negative_keep_rate = parse_value(&arg, &mut args)?,
                "--version" => version = value_after(&arg, &mut args)?,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }

        if inputs.is_empty() {
            return Err("at least one --input JSONL file is required".to_owned());
        }
        let out_dir = out_dir.ok_or_else(|| "--out model directory is required".to_owned())?;
        if hidden_dim == 0 || epochs == 0 || batch_size == 0 {
            return Err("--hidden-dim, --epochs, and --batch-size must be positive".to_owned());
        }
        if learning_rate <= 0.0 {
            return Err("--learning-rate must be positive".to_owned());
        }
        if !(0.0..=1.0).contains(&negative_keep_rate) {
            return Err("--negative-keep-rate must be between 0 and 1".to_owned());
        }

        Ok(Self {
            inputs,
            out_dir,
            report_path,
            max_records,
            hidden_dim,
            epochs,
            batch_size,
            learning_rate,
            seed,
            threshold,
            encoded_left,
            encoded_right,
            count_radius,
            negative_keep_rate,
            version,
        })
    }
}

fn value_after(arg: &str, args: &mut impl Iterator<Item = String>) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after `{arg}`"))
}

fn parse_value<T>(arg: &str, args: &mut impl Iterator<Item = String>) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = value_after(arg, args)?;
    raw.parse::<T>()
        .map_err(|error| format!("invalid value for `{arg}`: {error}"))
}

fn print_help() {
    println!(
        "train_sentence_burn --input DATA.jsonl --out MODEL_DIR [options]\n\
\n\
Options:\n\
  --input PATH           JSONL input with spans or inline sentence tags. Can be repeated.\n\
  --out PATH             Output model bundle directory.\n\
  --report PATH          Optional training report JSON path.\n\
  --max-records N        Stop after N records across inputs.\n\
  --hidden-dim N         Hidden dimension for the shallow Burn MLP. Default: 64.\n\
  --epochs N             Training epochs. Default: 12.\n\
  --batch-size N         Training batch size. Default: 256.\n\
  --learning-rate LR     Adam learning rate. Default: 0.001.\n\
  --seed N               Burn/RNG seed. Default: 7.\n\
  --threshold T          Fixed threshold. Default: tune on validation split.\n\
  --encoded-left N       Encoded byte window left width. Default: 7.\n\
  --encoded-right N      Encoded byte window right width. Default: 7.\n\
  --count-radius N       Directional count feature radius. Default: 24.\n\
  --negative-keep-rate R Keep-rate for negative all-position rows. Default: 0.02.\n\
  --version VERSION      Model bundle version. Default: crate version."
    );
}

#[derive(Debug, Deserialize)]
struct JsonlRecord {
    text: String,
    #[serde(default)]
    spans: Vec<JsonlSpan>,
}

#[derive(Clone, Debug, Deserialize)]
struct JsonlSpan {
    #[serde(default)]
    label: String,
    #[serde(default)]
    start: usize,
    #[serde(default)]
    end: usize,
    #[serde(default)]
    char_start: Option<usize>,
    #[serde(default)]
    char_end: Option<usize>,
    #[serde(default)]
    right_open: bool,
}

#[derive(Clone, Debug)]
struct RowSet {
    features: FeatureMatrix<f32>,
    labels: Vec<u8>,
    positives: usize,
    negatives: usize,
}

impl RowSet {
    fn new(cols: usize) -> Self {
        Self {
            features: FeatureMatrix {
                rows: 0,
                cols,
                data: Vec::new(),
            },
            labels: Vec::new(),
            positives: 0,
            negatives: 0,
        }
    }

    fn push(&mut self, row: &[f32], label: u8) {
        self.features.data.extend_from_slice(row);
        self.features.rows += 1;
        self.labels.push(label);
        if label == 1 {
            self.positives += 1;
        } else {
            self.negatives += 1;
        }
    }

    fn view(&self) -> DatasetView<'_, f32, u8> {
        DatasetView {
            features: self.features.as_view(),
            labels: &self.labels,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ExtractionStats {
    records_seen: usize,
    records_used: usize,
    records_without_candidates: usize,
    candidate_count: usize,
    positive_candidate_count: usize,
    unmatched_positive_boundaries: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse().map_err(|error| {
        eprintln!("{error}\n");
        print_help();
        error
    })?;

    let feature_config = BurnSentenceFeatureConfig {
        encoded_left: args.encoded_left,
        encoded_right: args.encoded_right,
        count_radius: args.count_radius,
        ..BurnSentenceFeatureConfig::default()
    };
    let kernel = burn_sentence_kernel(&feature_config);
    let feature_dim = kernel.schema().total_dim();
    let mut train = RowSet::new(feature_dim);
    let mut valid = RowSet::new(feature_dim);
    let mut stats = ExtractionStats::default();

    for input in &args.inputs {
        read_input(
            input,
            args.max_records,
            args.negative_keep_rate,
            &kernel,
            &mut train,
            &mut valid,
            &mut stats,
        )?;
        if args
            .max_records
            .is_some_and(|max_records| stats.records_seen >= max_records)
        {
            break;
        }
    }

    if train.labels.is_empty() || valid.labels.is_empty() {
        return Err("training and validation splits must both contain candidate rows".into());
    }
    if train.positives == 0 || valid.positives == 0 {
        return Err("training and validation splits must both contain positive examples".into());
    }

    let options = BurnShallowMlpFitOptions {
        hidden_dim: args.hidden_dim,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        seed: args.seed,
    };
    let (model, fit_report) =
        BurnShallowMlpModel::fit(train.view(), &options, &mut FitScratch::default())?;

    let mut train_scores = vec![0.0_f32; train.labels.len()];
    model.predict_into(train.features.as_view(), &mut train_scores)?;
    let mut valid_scores = vec![0.0_f32; valid.labels.len()];
    model.predict_into(valid.features.as_view(), &mut valid_scores)?;

    let threshold = args
        .threshold
        .unwrap_or_else(|| best_threshold_dense(&valid_scores, &valid.labels));
    let train_metrics = metrics_from_scores(&train_scores, &train.labels, threshold);
    let valid_metrics = metrics_from_scores(&valid_scores, &valid.labels, threshold);

    fs::create_dir_all(&args.out_dir)?;
    let model_stem = args.out_dir.join("sentence_boundary");
    model.save_named_mpk(&model_stem)?;
    let model_path = args.out_dir.join("sentence_boundary.mpk");
    let model_bytes = fs::metadata(&model_path)?.len();

    let mut manifest_feature_config = feature_config;
    manifest_feature_config.feature_dim = feature_dim;
    manifest_feature_config.hidden_dim = args.hidden_dim;

    let manifest = json!({
        "format": MODEL_FORMAT,
        "name": MODEL_NAME,
        "version": args.version,
        "engine": MODEL_ENGINE,
        "task": "sentence_boundary",
        "features": manifest_feature_config,
        "thresholds": {
            "sentence.end": threshold
        },
        "files": [
            {
                "path": "sentence_boundary.mpk",
                "role": "sentence_boundary",
                "bytes": model_bytes
            }
        ],
        "metrics": {
            "train": train_metrics,
            "validation": valid_metrics
        },
        "training": {
            "fit_report": fit_report,
            "stats": {
                "records_seen": stats.records_seen,
                "records_used": stats.records_used,
                "records_without_candidates": stats.records_without_candidates,
                "candidate_count": stats.candidate_count,
                "positive_candidate_count": stats.positive_candidate_count,
                "unmatched_positive_boundaries": stats.unmatched_positive_boundaries,
                "train_rows": train.labels.len(),
                "train_positive_rows": train.positives,
                "train_negative_rows": train.negatives,
                "validation_rows": valid.labels.len(),
                "validation_positive_rows": valid.positives,
                "validation_negative_rows": valid.negatives,
                "negative_keep_rate": args.negative_keep_rate
            }
        }
    });
    write_json_pretty(&args.out_dir.join("manifest.json"), &manifest)?;

    let report_path = args
        .report_path
        .clone()
        .unwrap_or_else(|| args.out_dir.join("training-report.json"));
    let report = json!({
        "model_dir": args.out_dir,
        "manifest": manifest,
    });
    write_json_pretty(&report_path, &report)?;

    println!(
        "trained {MODEL_ENGINE}: train_f1={:.4} valid_f1={:.4} threshold={:.2} rows={} valid_rows={} model={}",
        train_metrics.f1,
        valid_metrics.f1,
        threshold,
        train.labels.len(),
        valid.labels.len(),
        args.out_dir.display()
    );
    Ok(())
}

fn read_input(
    input: &Path,
    max_records: Option<usize>,
    negative_keep_rate: f32,
    kernel: &impl FeatureKernel<f32>,
    train: &mut RowSet,
    valid: &mut RowSet,
    stats: &mut ExtractionStats,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(input)?);
    for line in reader.lines() {
        if max_records.is_some_and(|max_records| stats.records_seen >= max_records) {
            break;
        }
        stats.records_seen += 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: JsonlRecord = match serde_json::from_str(&line) {
            Ok(record) => record,
            Err(_) => continue,
        };
        let record = normalize_record(record);
        extract_record(
            &record,
            stats.records_seen,
            negative_keep_rate,
            kernel,
            train,
            valid,
            stats,
        )?;
    }
    Ok(())
}

fn normalize_record(record: JsonlRecord) -> JsonlRecord {
    if record.spans.is_empty()
        && (record.text.contains(SENTENCE_TAG) || record.text.contains(PARAGRAPH_TAG))
    {
        return parse_inline_boundary_record(&record.text);
    }

    let spans = record
        .spans
        .into_iter()
        .filter_map(|span| normalize_span(&record.text, span))
        .collect();
    JsonlRecord {
        text: record.text,
        spans,
    }
}

fn parse_inline_boundary_record(annotated: &str) -> JsonlRecord {
    let mut text = String::with_capacity(annotated.len());
    let mut spans = Vec::new();
    let mut sentence_start = 0_usize;
    let mut offset = 0_usize;

    while offset < annotated.len() {
        let rest = &annotated[offset..];
        let marker_len = if rest.starts_with(SENTENCE_TAG) {
            Some(SENTENCE_TAG.len())
        } else if rest.starts_with(PARAGRAPH_TAG) {
            Some(PARAGRAPH_TAG.len())
        } else {
            None
        };

        if let Some(marker_len) = marker_len {
            let sentence_end = text.len();
            if sentence_end > sentence_start {
                spans.push(JsonlSpan {
                    label: "sentence".to_owned(),
                    start: sentence_start,
                    end: sentence_end,
                    char_start: None,
                    char_end: None,
                    right_open: false,
                });
                sentence_start = sentence_end;
            }
            offset += marker_len;
            continue;
        }

        let ch = rest
            .chars()
            .next()
            .expect("remaining annotated text must contain one UTF-8 scalar");
        text.push(ch);
        offset += ch.len_utf8();
    }

    if text.len() > sentence_start {
        spans.push(JsonlSpan {
            label: "sentence".to_owned(),
            start: sentence_start,
            end: text.len(),
            char_start: None,
            char_end: None,
            right_open: true,
        });
    }

    JsonlRecord { text, spans }
}

fn normalize_span(text: &str, span: JsonlSpan) -> Option<JsonlSpan> {
    if !span.label.eq_ignore_ascii_case("sentence") {
        return None;
    }
    let (start, end) = if let (Some(char_start), Some(char_end)) = (span.char_start, span.char_end)
    {
        char_span_to_byte_span(text, char_start, char_end)?
    } else if span.end <= text.len()
        && span.start < span.end
        && text.is_char_boundary(span.start)
        && text.is_char_boundary(span.end)
    {
        (span.start, span.end)
    } else {
        char_span_to_byte_span(text, span.start, span.end)?
    };
    Some(JsonlSpan {
        label: "sentence".to_owned(),
        start,
        end,
        char_start: None,
        char_end: None,
        right_open: span.right_open,
    })
}

fn char_span_to_byte_span(
    text: &str,
    char_start: usize,
    char_end: usize,
) -> Option<(usize, usize)> {
    if char_start >= char_end {
        return None;
    }
    let mut offsets = Vec::with_capacity(text.chars().count() + 1);
    offsets.push(0);
    for (byte_offset, _) in text.char_indices().skip(1) {
        offsets.push(byte_offset);
    }
    offsets.push(text.len());
    let start = *offsets.get(char_start)?;
    let end = *offsets.get(char_end)?;
    (start < end).then_some((start, end))
}

fn extract_record(
    record: &JsonlRecord,
    record_index: usize,
    negative_keep_rate: f32,
    kernel: &impl FeatureKernel<f32>,
    train: &mut RowSet,
    valid: &mut RowSet,
    stats: &mut ExtractionStats,
) -> Result<(), Box<dyn std::error::Error>> {
    if record.text.is_empty() {
        return Ok(());
    }

    let candidates = sentence_boundary_candidates(&record.text);
    if candidates.is_empty() {
        stats.records_without_candidates += 1;
        return Ok(());
    }

    let positives = sentence_end_boundaries(&record.text, &record.spans);
    let matched_positive_count = candidates
        .iter()
        .filter(|candidate| positives.contains(&candidate.break_end))
        .count();
    stats.unmatched_positive_boundaries += positives.len().saturating_sub(matched_positive_count);

    let mut candidate_buffer = CandidateBuffer::new();
    for candidate in &candidates {
        candidate_buffer.push(BytePos::from_usize(candidate.feature_pos));
    }

    let mut features = FeatureMatrix::<f32>::default();
    features.resize_zeroed(candidate_buffer.len(), kernel.schema().total_dim());
    kernel.extract_into(
        TextBytes::from_utf8(&record.text),
        candidate_buffer.as_slice(),
        features.as_view_mut(),
        &mut FeatureScratch::default(),
    )?;

    let target = if record_index.is_multiple_of(5) {
        valid
    } else {
        train
    };

    for (row_index, candidate) in candidates.iter().enumerate() {
        let label = u8::from(positives.contains(&candidate.break_end));
        if label == 0 && !keep_negative(record_index, *candidate, negative_keep_rate) {
            continue;
        }
        target.push(features.as_view().row(row_index), label);
        stats.candidate_count += 1;
        if label == 1 {
            stats.positive_candidate_count += 1;
        }
    }

    stats.records_used += 1;
    Ok(())
}

fn keep_negative(
    record_index: usize,
    candidate: charstreamer_segmentation::SentenceBoundaryCandidate,
    keep_rate: f32,
) -> bool {
    if keep_rate >= 1.0 {
        return true;
    }
    if keep_rate <= 0.0 {
        return false;
    }
    let value = ((record_index as u64).wrapping_mul(1_146_959_810_393_466_583)
        ^ (candidate.feature_pos as u64).wrapping_mul(1_099_511_628_211)
        ^ (candidate.break_end as u64))
        % 10_000;
    (value as f32 / 10_000.0) < keep_rate
}

fn sentence_end_boundaries(text: &str, spans: &[JsonlSpan]) -> BTreeSet<usize> {
    let mut boundaries = BTreeSet::new();
    for span in spans {
        if span.label != "sentence" || span.right_open {
            continue;
        }
        if span.start >= span.end || span.end > text.len() {
            continue;
        }
        if !text.is_char_boundary(span.start) || !text.is_char_boundary(span.end) {
            continue;
        }
        if let Some(end) = previous_nonspace_end(text, span.start, span.end) {
            boundaries.insert(end);
        }
    }
    boundaries
}

fn previous_nonspace_end(text: &str, start: usize, end: usize) -> Option<usize> {
    text.get(start..end)?
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(offset, ch)| start + offset + ch.len_utf8())
}

fn best_threshold_dense(scores: &[f32], labels: &[u8]) -> f32 {
    let mut best_threshold = 0.5_f32;
    let mut best_f1 = -1.0_f32;
    let mut best_precision = -1.0_f32;
    for step in 1..=99 {
        let threshold = step as f32 / 100.0;
        let metrics = metrics_from_scores(scores, labels, threshold);
        if metrics.f1 > best_f1
            || ((metrics.f1 - best_f1).abs() <= f32::EPSILON && metrics.precision > best_precision)
        {
            best_threshold = threshold;
            best_f1 = metrics.f1;
            best_precision = metrics.precision;
        }
    }
    best_threshold
}

fn write_json_pretty(
    path: &Path,
    value: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}
