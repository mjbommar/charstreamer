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
    AsciiClassAppender, ByteClass, CompositeFeatureKernel, DirectionalByteClassCountAppender,
    DirectionalUnicodeCategoryGroupCountAppender, EncodedByteWindowAppender, LineByteCountAppender,
    UnicodeCategoryGroup,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

type InferBackend = NdArray<f32>;
type TrainBackend = Autodiff<InferBackend>;

const DEFAULT_LABELS: &[&str] = &[
    "sentence",
    "paragraph",
    "section",
    "dialogue",
    "list_item",
    "metadata",
];

const DEFAULT_TASKS: &[SemanticTask] =
    &[SemanticTask::Inside, SemanticTask::Start, SemanticTask::End];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
enum Architecture {
    Linear,
    #[default]
    Mlp1,
    Mlp2,
    Mlp3,
}

impl Architecture {
    fn parse(value: &str) -> Result<Self, TrainError> {
        match value {
            "linear" => Ok(Self::Linear),
            "mlp1" => Ok(Self::Mlp1),
            "mlp2" => Ok(Self::Mlp2),
            "mlp3" => Ok(Self::Mlp3),
            other => Err(TrainError::InvalidArgument(format!(
                "unsupported architecture `{other}`; expected linear,mlp1,mlp2,mlp3"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Mlp1 => "mlp1",
            Self::Mlp2 => "mlp2",
            Self::Mlp3 => "mlp3",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum SemanticTask {
    Inside,
    Start,
    End,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
enum ClassWeighting {
    #[default]
    None,
    Balanced,
    SqrtBalanced,
}

impl ClassWeighting {
    fn parse(value: &str) -> Result<Self, TrainError> {
        match value {
            "none" => Ok(Self::None),
            "balanced" => Ok(Self::Balanced),
            "sqrt-balanced" => Ok(Self::SqrtBalanced),
            other => Err(TrainError::InvalidArgument(format!(
                "unsupported class weighting `{other}`; expected none,balanced,sqrt-balanced"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Balanced => "balanced",
            Self::SqrtBalanced => "sqrt-balanced",
        }
    }
}

impl SemanticTask {
    fn parse(value: &str) -> Result<Self, TrainError> {
        match value {
            "inside" => Ok(Self::Inside),
            "start" => Ok(Self::Start),
            "end" => Ok(Self::End),
            other => Err(TrainError::InvalidArgument(format!(
                "unsupported task `{other}`; expected inside,start,end"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Inside => "inside",
            Self::Start => "start",
            Self::End => "end",
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
    inspect_max_spans_per_label: usize,
    labels: Vec<String>,
    tasks: Vec<SemanticTask>,
    architecture: Architecture,
    epochs: usize,
    batch_size: usize,
    hidden_dim: usize,
    hidden_dim2: usize,
    hidden_dim3: usize,
    learning_rate: f64,
    seed: u64,
    split_numerator: usize,
    split_denominator: usize,
    negative_keep_rate: f32,
    max_records: Option<usize>,
    max_positions_per_doc: Option<usize>,
    encoded_left: usize,
    encoded_right: usize,
    count_radius: usize,
    validation_predict_batch_size: usize,
    class_weighting: ClassWeighting,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            report_path: PathBuf::from("/tmp/charstreamer-synthetic-burn-report.json"),
            inspect_texts: Vec::new(),
            inspect_max_spans_per_label: 4,
            labels: DEFAULT_LABELS
                .iter()
                .map(|label| (*label).to_string())
                .collect(),
            tasks: DEFAULT_TASKS.to_vec(),
            architecture: Architecture::Mlp1,
            epochs: 8,
            batch_size: 1024,
            hidden_dim: 128,
            hidden_dim2: 64,
            hidden_dim3: 32,
            learning_rate: 1.0e-3,
            seed: 7,
            split_numerator: 8,
            split_denominator: 10,
            negative_keep_rate: 0.35,
            max_records: None,
            max_positions_per_doc: None,
            encoded_left: 7,
            encoded_right: 7,
            count_radius: 24,
            validation_predict_batch_size: 16_384,
            class_weighting: ClassWeighting::None,
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

#[derive(Clone, Debug)]
struct OutputHead {
    label: String,
    task: SemanticTask,
}

#[derive(Debug)]
struct MultiLabelDataset {
    features: FeatureMatrix<f32>,
    targets: Vec<u8>,
    output_dim: usize,
    rows: usize,
    chars: usize,
    documents: usize,
    positives: Vec<usize>,
}

trait SemanticLogitModule<B: Backend>: Module<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

#[derive(Module, Debug)]
struct SemanticLinear<B: Backend> {
    output: Linear<B>,
}

impl<B: Backend> SemanticLinear<B> {
    fn new(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            output: LinearConfig::new(input_dim, output_dim).init(device),
        }
    }
}

impl<B: Backend> SemanticLogitModule<B> for SemanticLinear<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.output.forward(input)
    }
}

#[derive(Module, Debug)]
struct SemanticMlp1<B: Backend> {
    input: Linear<B>,
    activation: Relu,
    output: Linear<B>,
}

impl<B: Backend> SemanticMlp1<B> {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim).init(device),
            activation: Relu::new(),
            output: LinearConfig::new(hidden_dim, output_dim).init(device),
        }
    }
}

impl<B: Backend> SemanticLogitModule<B> for SemanticMlp1<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.activation.forward(self.input.forward(input));
        self.output.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct SemanticMlp2<B: Backend> {
    input: Linear<B>,
    activation1: Relu,
    hidden2: Linear<B>,
    activation2: Relu,
    output: Linear<B>,
}

impl<B: Backend> SemanticMlp2<B> {
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
}

impl<B: Backend> SemanticLogitModule<B> for SemanticMlp2<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden1 = self.activation1.forward(self.input.forward(input));
        let hidden2 = self.activation2.forward(self.hidden2.forward(hidden1));
        self.output.forward(hidden2)
    }
}

#[derive(Module, Debug)]
struct SemanticMlp3<B: Backend> {
    input: Linear<B>,
    activation1: Relu,
    hidden2: Linear<B>,
    activation2: Relu,
    hidden3: Linear<B>,
    activation3: Relu,
    output: Linear<B>,
}

impl<B: Backend> SemanticMlp3<B> {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        hidden_dim2: usize,
        hidden_dim3: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim).init(device),
            activation1: Relu::new(),
            hidden2: LinearConfig::new(hidden_dim, hidden_dim2).init(device),
            activation2: Relu::new(),
            hidden3: LinearConfig::new(hidden_dim2, hidden_dim3).init(device),
            activation3: Relu::new(),
            output: LinearConfig::new(hidden_dim3, output_dim).init(device),
        }
    }
}

impl<B: Backend> SemanticLogitModule<B> for SemanticMlp3<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden1 = self.activation1.forward(self.input.forward(input));
        let hidden2 = self.activation2.forward(self.hidden2.forward(hidden1));
        let hidden3 = self.activation3.forward(self.hidden3.forward(hidden2));
        self.output.forward(hidden3)
    }
}

#[derive(Debug)]
enum SemanticModel {
    Linear(SemanticLinear<InferBackend>),
    Mlp1(SemanticMlp1<InferBackend>),
    Mlp2(SemanticMlp2<InferBackend>),
    Mlp3(SemanticMlp3<InferBackend>),
}

impl SemanticModel {
    fn predict_probabilities(
        &self,
        dataset: &MultiLabelDataset,
        batch_size: usize,
    ) -> Result<Vec<f32>, TrainError> {
        match self {
            Self::Linear(model) => predict_probabilities_for_model(model, dataset, batch_size),
            Self::Mlp1(model) => predict_probabilities_for_model(model, dataset, batch_size),
            Self::Mlp2(model) => predict_probabilities_for_model(model, dataset, batch_size),
            Self::Mlp3(model) => predict_probabilities_for_model(model, dataset, batch_size),
        }
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
}

#[derive(Clone, Debug, Serialize)]
struct TrainingReport {
    inputs: Vec<PathBuf>,
    labels: Vec<String>,
    tasks: Vec<String>,
    outputs: Vec<String>,
    architecture: String,
    feature_dim: usize,
    hidden_dim: usize,
    hidden_dim2: usize,
    hidden_dim3: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
    split_numerator: usize,
    split_denominator: usize,
    negative_keep_rate: f32,
    encoded_left: usize,
    encoded_right: usize,
    count_radius: usize,
    class_weighting: String,
    class_weights: Option<Vec<f32>>,
    loaded_documents: usize,
    invalid_documents: usize,
    train: DatasetReport,
    validation: DatasetReport,
    feature_seconds_train: f64,
    feature_seconds_validation: f64,
    train_seconds: f64,
    validation_predict_seconds: f64,
    validation_rows_per_second: f64,
    validation_chars_per_second: f64,
    validation_end_to_end_seconds: f64,
    validation_end_to_end_rows_per_second: f64,
    validation_end_to_end_chars_per_second: f64,
    macro_f1: f64,
    macro_f1_by_task: BTreeMap<String, f64>,
    macro_f1_by_label: BTreeMap<String, f64>,
    output_metrics: Vec<OutputMetricReport>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args().skip(1))?;
    let started = Instant::now();
    let (mut records, invalid_documents) = load_records(&config)?;
    let loaded_documents = records.len();
    if loaded_documents < 2 {
        return Err(Box::new(TrainError::InvalidArgument(
            "need at least two valid synthetic records to train and validate".to_string(),
        )));
    }

    let mut rng = SmallRng::seed_from_u64(config.seed);
    records.shuffle(&mut rng);
    let split_at =
        loaded_documents.saturating_mul(config.split_numerator) / config.split_denominator;
    if split_at == 0 || split_at >= loaded_documents {
        return Err(Box::new(TrainError::InvalidArgument(format!(
            "invalid split produced train={split_at} validation={}",
            loaded_documents.saturating_sub(split_at)
        ))));
    }
    let validation_records = records.split_off(split_at);
    let train_records = records;

    let heads = output_heads(&config);
    let kernel = build_kernel(&config);
    let output_dim = heads.len();

    let feature_started = Instant::now();
    let train_dataset = build_dataset(&train_records, &heads, &kernel, &config, true)?;
    let feature_seconds_train = feature_started.elapsed().as_secs_f64();

    let feature_started = Instant::now();
    let validation_dataset = build_dataset(&validation_records, &heads, &kernel, &config, false)?;
    let feature_seconds_validation = feature_started.elapsed().as_secs_f64();

    if train_dataset.rows == 0 || validation_dataset.rows == 0 {
        return Err(Box::new(TrainError::InvalidArgument(
            "empty train or validation matrix after feature extraction".to_string(),
        )));
    }

    let train_started = Instant::now();
    let class_weights = class_weights(&train_dataset, config.class_weighting);
    let model = train_model(&train_dataset, &config, class_weights.as_deref())?;
    let train_seconds = train_started.elapsed().as_secs_f64();

    let predict_started = Instant::now();
    let scores =
        model.predict_probabilities(&validation_dataset, config.validation_predict_batch_size)?;
    let validation_predict_seconds = predict_started.elapsed().as_secs_f64();

    let output_metrics = tune_and_score_outputs(&heads, &scores, &validation_dataset);
    let validation_end_to_end_seconds = feature_seconds_validation + validation_predict_seconds;
    let macro_f1 = output_metrics
        .iter()
        .map(|metric| metric.metrics.f1)
        .sum::<f64>()
        / output_metrics.len().max(1) as f64;
    let macro_f1_by_task = macro_by_task(&output_metrics);
    let macro_f1_by_label = macro_by_label(&output_metrics);

    let outputs = heads
        .iter()
        .map(|head| output_name(head))
        .collect::<Vec<_>>();
    let report = TrainingReport {
        inputs: config.inputs.clone(),
        labels: config.labels.clone(),
        tasks: config
            .tasks
            .iter()
            .map(|task| task.as_str().to_string())
            .collect(),
        outputs: outputs.clone(),
        architecture: config.architecture.as_str().to_string(),
        feature_dim: train_dataset.features.cols,
        hidden_dim: config.hidden_dim,
        hidden_dim2: config.hidden_dim2,
        hidden_dim3: config.hidden_dim3,
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        seed: config.seed,
        split_numerator: config.split_numerator,
        split_denominator: config.split_denominator,
        negative_keep_rate: config.negative_keep_rate,
        encoded_left: config.encoded_left,
        encoded_right: config.encoded_right,
        count_radius: config.count_radius,
        class_weighting: config.class_weighting.as_str().to_string(),
        class_weights,
        loaded_documents,
        invalid_documents,
        train: dataset_report(&heads, &train_dataset),
        validation: dataset_report(&heads, &validation_dataset),
        feature_seconds_train,
        feature_seconds_validation,
        train_seconds,
        validation_predict_seconds,
        validation_rows_per_second: validation_dataset.rows as f64
            / validation_predict_seconds.max(f64::MIN_POSITIVE),
        validation_chars_per_second: validation_dataset.chars as f64
            / validation_predict_seconds.max(f64::MIN_POSITIVE),
        validation_end_to_end_seconds,
        validation_end_to_end_rows_per_second: validation_dataset.rows as f64
            / validation_end_to_end_seconds.max(f64::MIN_POSITIVE),
        validation_end_to_end_chars_per_second: validation_dataset.chars as f64
            / validation_end_to_end_seconds.max(f64::MIN_POSITIVE),
        macro_f1,
        macro_f1_by_task,
        macro_f1_by_label,
        output_metrics,
    };

    fs::write(&config.report_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "loaded_docs={} invalid_docs={} train_docs={} valid_docs={} train_rows={} valid_rows={} feature_dim={} outputs={} elapsed_s={:.3}",
        loaded_documents,
        invalid_documents,
        train_dataset.documents,
        validation_dataset.documents,
        train_dataset.rows,
        validation_dataset.rows,
        train_dataset.features.cols,
        output_dim,
        started.elapsed().as_secs_f64(),
    );
    println!(
        "burn_{} hidden={}/{}/{} epochs={} batch={} lr={} train_s={:.3} valid_predict_s={:.3} valid_rows_per_s={:.1} valid_e2e_chars_per_s={:.1} macro_f1={:.4}",
        config.architecture.as_str(),
        config.hidden_dim,
        config.hidden_dim2,
        config.hidden_dim3,
        config.epochs,
        config.batch_size,
        config.learning_rate,
        train_seconds,
        validation_predict_seconds,
        report.validation_rows_per_second,
        report.validation_end_to_end_chars_per_second,
        report.macro_f1,
    );
    for task in &config.tasks {
        let key = task.as_str();
        if let Some(value) = report.macro_f1_by_task.get(key) {
            println!("{key}_macro_f1={value:.4}");
        }
    }
    for metric in report
        .output_metrics
        .iter()
        .filter(|metric| metric.task == "inside" || metric.positives >= 10)
    {
        println!(
            "{}_{} f1={:.4} p={:.4} r={:.4} pos={} th={:.2}",
            metric.label,
            metric.task,
            metric.metrics.f1,
            metric.metrics.precision,
            metric.metrics.recall,
            metric.positives,
            metric.threshold,
        );
    }
    println!("report: {}", config.report_path.display());
    inspect_texts(&config, &heads, &kernel, &model, &report.output_metrics)?;

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
            "--inspect-max-spans-per-label" => {
                config.inspect_max_spans_per_label =
                    parse_next(&mut args, "--inspect-max-spans-per-label")?;
            }
            "--labels" => config.labels = split_csv(&next_value(&mut args, "--labels")?),
            "--tasks" => {
                config.tasks = split_csv(&next_value(&mut args, "--tasks")?)
                    .into_iter()
                    .map(|value| SemanticTask::parse(&value))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--architecture" => {
                config.architecture =
                    Architecture::parse(&next_value(&mut args, "--architecture")?)?;
            }
            "--epochs" => config.epochs = parse_next(&mut args, "--epochs")?,
            "--batch-size" => config.batch_size = parse_next(&mut args, "--batch-size")?,
            "--hidden-dim" => config.hidden_dim = parse_next(&mut args, "--hidden-dim")?,
            "--hidden-dim2" => config.hidden_dim2 = parse_next(&mut args, "--hidden-dim2")?,
            "--hidden-dim3" => config.hidden_dim3 = parse_next(&mut args, "--hidden-dim3")?,
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
            "--max-positions-per-doc" => {
                config.max_positions_per_doc =
                    nonzero_option(parse_next(&mut args, "--max-positions-per-doc")?);
            }
            "--encoded-left" => config.encoded_left = parse_next(&mut args, "--encoded-left")?,
            "--encoded-right" => config.encoded_right = parse_next(&mut args, "--encoded-right")?,
            "--count-radius" => config.count_radius = parse_next(&mut args, "--count-radius")?,
            "--validation-predict-batch-size" => {
                config.validation_predict_batch_size =
                    parse_next(&mut args, "--validation-predict-batch-size")?;
            }
            "--class-weighting" => {
                config.class_weighting =
                    ClassWeighting::parse(&next_value(&mut args, "--class-weighting")?)?;
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
    if config.labels.is_empty() || config.tasks.is_empty() {
        return Err(TrainError::InvalidArgument(
            "--labels and --tasks must be non-empty".to_string(),
        ));
    }
    if config.epochs == 0
        || config.batch_size == 0
        || config.hidden_dim == 0
        || config.hidden_dim2 == 0
        || config.hidden_dim3 == 0
    {
        return Err(TrainError::InvalidArgument(
            "epochs, batch-size, and hidden dims must be greater than zero".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.negative_keep_rate) {
        return Err(TrainError::InvalidArgument(
            "--negative-keep-rate must be between 0 and 1".to_string(),
        ));
    }
    if config.split_denominator == 0 || config.split_numerator >= config.split_denominator {
        return Err(TrainError::InvalidArgument(
            "split must satisfy 0 <= numerator < denominator".to_string(),
        ));
    }

    Ok(config)
}

fn print_usage() {
    eprintln!(
        "usage: cargo run -p charstreamer-experiments --example train_synthetic_burn -- \\
  --input <jsonl> [--input <jsonl> ...] [--report <report.json>] \\
  [--labels sentence,paragraph,section,dialogue,list_item,metadata] \\
  [--tasks inside,start,end] [--class-weighting none|balanced|sqrt-balanced] \\
  [--architecture linear|mlp1|mlp2|mlp3] [--epochs 8] [--batch-size 1024] \\
  [--inspect-text <text>]"
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
    let allowed_labels: BTreeSet<&str> = config.labels.iter().map(String::as_str).collect();
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

fn output_heads(config: &Config) -> Vec<OutputHead> {
    config
        .labels
        .iter()
        .flat_map(|label| {
            config.tasks.iter().copied().map(|task| OutputHead {
                label: label.clone(),
                task,
            })
        })
        .collect()
}

fn output_name(head: &OutputHead) -> String {
    format!("{}.{}", head.label, head.task.as_str())
}

fn build_kernel(config: &Config) -> CompositeFeatureKernel {
    CompositeFeatureKernel::new(vec![
        Box::new(EncodedByteWindowAppender::new(ByteWindowSpec::new(
            config.encoded_left,
            config.encoded_right,
        ))),
        Box::new(AsciiClassAppender::new()),
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
    heads: &[OutputHead],
    kernel: &CompositeFeatureKernel,
    config: &Config,
    sample_negatives: bool,
) -> Result<MultiLabelDataset, TrainError> {
    let mut dataset = MultiLabelDataset {
        features: FeatureMatrix {
            rows: 0,
            cols: kernel.schema().total_dim(),
            data: Vec::new(),
        },
        targets: Vec::new(),
        output_dim: heads.len(),
        rows: 0,
        chars: 0,
        documents: 0,
        positives: vec![0; heads.len()],
    };
    let mut feature_scratch = FeatureScratch::default();
    let mut doc_features = FeatureMatrix::<f32>::default();
    let mut candidates = CandidateBuffer::new();
    let mut selected_positions = Vec::new();
    let mut selected_targets = Vec::new();
    let mut rng = SmallRng::seed_from_u64(config.seed ^ if sample_negatives { 0x51 } else { 0xA7 });

    for record in records {
        let positions = positions_for_record(record, config, &mut rng);
        if positions.is_empty() {
            continue;
        }

        candidates.clear();
        selected_positions.clear();
        selected_targets.clear();
        for position in positions {
            let row_targets = targets_for_position(record, position, heads);
            let has_positive = row_targets.iter().any(|&value| value != 0);
            if sample_negatives && !has_positive && rng.random::<f32>() > config.negative_keep_rate
            {
                continue;
            }
            candidates.push(BytePos::from_usize(position));
            selected_positions.push(position);
            selected_targets.extend_from_slice(&row_targets);
            for (index, value) in row_targets.into_iter().enumerate() {
                dataset.positives[index] += usize::from(value != 0);
            }
        }

        if candidates.is_empty() {
            continue;
        }

        doc_features.resize_zeroed(candidates.len(), dataset.features.cols);
        kernel.extract_into(
            TextBytes::from_utf8(&record.text),
            candidates.as_slice(),
            doc_features.as_view_mut(),
            &mut feature_scratch,
        )?;

        dataset.features.data.extend_from_slice(&doc_features.data);
        dataset.features.rows += doc_features.rows;
        dataset.targets.extend_from_slice(&selected_targets);
        dataset.rows += candidates.len();
        dataset.chars += record.text.len();
        dataset.documents += 1;
    }

    Ok(dataset)
}

fn positions_for_record(record: &CleanRecord, config: &Config, rng: &mut SmallRng) -> Vec<usize> {
    let mut positions = Vec::with_capacity(record.text.len() + 1);
    for (offset, _) in record.text.char_indices() {
        positions.push(offset);
    }
    positions.push(record.text.len());

    let Some(max_positions) = config.max_positions_per_doc else {
        return positions;
    };
    if positions.len() <= max_positions {
        return positions;
    }

    let mut required = BTreeSet::new();
    for span in &record.spans {
        if !span.left_open {
            required.insert(span.start);
        }
        if !span.right_open {
            required.insert(span.end);
        }
    }

    let mut sampled = required.into_iter().collect::<Vec<_>>();
    let remaining = max_positions.saturating_sub(sampled.len());
    let mut optional = positions
        .into_iter()
        .filter(|position| !sampled.contains(position))
        .collect::<Vec<_>>();
    optional.shuffle(rng);
    sampled.extend(optional.into_iter().take(remaining));
    sampled.sort_unstable();
    sampled
}

fn targets_for_position(record: &CleanRecord, position: usize, heads: &[OutputHead]) -> Vec<u8> {
    let mut targets = vec![0_u8; heads.len()];
    for span in &record.spans {
        for (index, head) in heads.iter().enumerate() {
            if head.label != span.label {
                continue;
            }
            let positive = match head.task {
                SemanticTask::Inside => position >= span.start && position < span.end,
                SemanticTask::Start => !span.left_open && position == span.start,
                SemanticTask::End => !span.right_open && position == span.end,
            };
            if positive {
                targets[index] = 1;
            }
        }
    }
    targets
}

fn train_model(
    dataset: &MultiLabelDataset,
    config: &Config,
    class_weights: Option<&[f32]>,
) -> Result<SemanticModel, TrainError> {
    match config.architecture {
        Architecture::Linear => {
            let model = train_model_impl::<SemanticLinear<TrainBackend>, _>(
                dataset,
                config,
                class_weights,
                |device, input_dim, output_dim| SemanticLinear::new(input_dim, output_dim, device),
            )?;
            Ok(SemanticModel::Linear(model))
        }
        Architecture::Mlp1 => {
            let model = train_model_impl::<SemanticMlp1<TrainBackend>, _>(
                dataset,
                config,
                class_weights,
                |device, input_dim, output_dim| {
                    SemanticMlp1::new(input_dim, config.hidden_dim, output_dim, device)
                },
            )?;
            Ok(SemanticModel::Mlp1(model))
        }
        Architecture::Mlp2 => {
            let model = train_model_impl::<SemanticMlp2<TrainBackend>, _>(
                dataset,
                config,
                class_weights,
                |device, input_dim, output_dim| {
                    SemanticMlp2::new(
                        input_dim,
                        config.hidden_dim,
                        config.hidden_dim2,
                        output_dim,
                        device,
                    )
                },
            )?;
            Ok(SemanticModel::Mlp2(model))
        }
        Architecture::Mlp3 => {
            let model = train_model_impl::<SemanticMlp3<TrainBackend>, _>(
                dataset,
                config,
                class_weights,
                |device, input_dim, output_dim| {
                    SemanticMlp3::new(
                        input_dim,
                        config.hidden_dim,
                        config.hidden_dim2,
                        config.hidden_dim3,
                        output_dim,
                        device,
                    )
                },
            )?;
            Ok(SemanticModel::Mlp3(model))
        }
    }
}

fn train_model_impl<M, F>(
    dataset: &MultiLabelDataset,
    config: &Config,
    class_weights: Option<&[f32]>,
    build: F,
) -> Result<M::InnerModule, TrainError>
where
    M: SemanticLogitModule<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: SemanticLogitModule<InferBackend>,
    F: FnOnce(&<TrainBackend as Backend>::Device, usize, usize) -> M,
{
    let device = Default::default();
    TrainBackend::seed(&device, config.seed);
    let mut model = build(&device, dataset.features.cols, dataset.output_dim);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .with_weights(class_weights.map(<[f32]>::to_vec))
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

fn class_weights(dataset: &MultiLabelDataset, mode: ClassWeighting) -> Option<Vec<f32>> {
    if mode == ClassWeighting::None || dataset.rows == 0 || dataset.output_dim == 0 {
        return None;
    }

    let mut weights = dataset
        .positives
        .iter()
        .map(|&positives| {
            let positives = positives.max(1) as f32;
            let prevalence = positives / dataset.rows as f32;
            match mode {
                ClassWeighting::None => 1.0,
                ClassWeighting::Balanced => 1.0 / prevalence,
                ClassWeighting::SqrtBalanced => (1.0 / prevalence).sqrt(),
            }
        })
        .collect::<Vec<_>>();
    let mean = weights.iter().sum::<f32>() / weights.len().max(1) as f32;
    if mean > 0.0 {
        for weight in &mut weights {
            *weight /= mean;
        }
    }
    Some(weights)
}

fn predict_probabilities_for_model<M>(
    model: &M,
    dataset: &MultiLabelDataset,
    batch_size: usize,
) -> Result<Vec<f32>, TrainError>
where
    M: SemanticLogitModule<InferBackend>,
{
    let device = Default::default();
    let mut scores = Vec::with_capacity(dataset.rows * dataset.output_dim);
    for row_start in (0..dataset.rows).step_by(batch_size.max(1)) {
        let row_end = row_start
            .saturating_add(batch_size.max(1))
            .min(dataset.rows);
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
    heads: &[OutputHead],
    scores: &[f32],
    dataset: &MultiLabelDataset,
) -> Vec<OutputMetricReport> {
    heads
        .iter()
        .enumerate()
        .map(|(output_index, head)| {
            let positives = dataset.positives[output_index];
            let negatives = dataset.rows.saturating_sub(positives);
            let (threshold, metrics) = best_threshold_for_output(
                scores,
                &dataset.targets,
                dataset.output_dim,
                output_index,
            );
            OutputMetricReport {
                label: head.label.clone(),
                task: head.task.as_str().to_string(),
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

fn macro_by_task(metrics: &[OutputMetricReport]) -> BTreeMap<String, f64> {
    let mut grouped: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for metric in metrics {
        grouped
            .entry(metric.task.clone())
            .or_default()
            .push(metric.metrics.f1);
    }
    grouped
        .into_iter()
        .map(|(task, values)| {
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            (task, mean)
        })
        .collect()
}

fn macro_by_label(metrics: &[OutputMetricReport]) -> BTreeMap<String, f64> {
    let mut grouped: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for metric in metrics {
        grouped
            .entry(metric.label.clone())
            .or_default()
            .push(metric.metrics.f1);
    }
    grouped
        .into_iter()
        .map(|(label, values)| {
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            (label, mean)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct DecodedInspectionSpan {
    start: usize,
    end: usize,
    avg_score: f32,
    max_score: f32,
}

fn inspect_texts(
    config: &Config,
    heads: &[OutputHead],
    kernel: &CompositeFeatureKernel,
    model: &SemanticModel,
    output_metrics: &[OutputMetricReport],
) -> Result<(), TrainError> {
    if config.inspect_texts.is_empty() {
        return Ok(());
    }

    let thresholds = output_metrics
        .iter()
        .map(|metric| {
            (
                format!("{}.{}", metric.label, metric.task),
                metric.threshold,
            )
        })
        .collect::<BTreeMap<_, _>>();

    for (index, text) in config.inspect_texts.iter().enumerate() {
        let (positions, scores) = score_text(config, heads, kernel, model, text)?;
        println!(
            "\ninspect_text_{index} bytes={} chars={} positions={}",
            text.len(),
            text.chars().count(),
            positions.len()
        );
        println!("--- text start ---\n{text}\n--- text end ---");

        for (head_index, head) in heads.iter().enumerate() {
            let key = output_name(head);
            let threshold = thresholds.get(&key).copied().unwrap_or(0.5);
            if head.task != SemanticTask::Inside {
                let points =
                    decode_label_points(&positions, &scores, heads.len(), head_index, threshold);
                if points.is_empty() {
                    println!(
                        "  {}.{} threshold={threshold:.2}: no points",
                        head.label,
                        head.task.as_str()
                    );
                    continue;
                }
                println!(
                    "  {}.{} threshold={threshold:.2}: {} point(s)",
                    head.label,
                    head.task.as_str(),
                    points.len()
                );
                for (position, score) in points.iter().take(config.inspect_max_spans_per_label) {
                    println!(
                        "    {:>4} score={:.3} {:?}",
                        position,
                        score,
                        point_context(text, *position, 48)
                    );
                }
                continue;
            }
            let spans = decode_label_spans(
                text,
                &positions,
                &scores,
                heads.len(),
                head_index,
                threshold,
            );
            if spans.is_empty() {
                println!("  {} threshold={threshold:.2}: no spans", head.label);
                continue;
            }
            println!(
                "  {} threshold={threshold:.2}: {} span(s)",
                head.label,
                spans.len()
            );
            for span in spans.iter().take(config.inspect_max_spans_per_label) {
                println!(
                    "    {:>4}..{:<4} avg={:.3} max={:.3} {:?}",
                    span.start,
                    span.end,
                    span.avg_score,
                    span.max_score,
                    snippet(text, span.start, span.end, 160)
                );
            }
        }

        inspect_basic_sentence_breaks(text, heads, &positions, &scores, &thresholds);
    }

    Ok(())
}

fn score_text(
    config: &Config,
    heads: &[OutputHead],
    kernel: &CompositeFeatureKernel,
    model: &SemanticModel,
    text: &str,
) -> Result<(Vec<usize>, Vec<f32>), TrainError> {
    let mut candidates = CandidateBuffer::new();
    let mut positions = Vec::new();
    for (offset, _) in text.char_indices() {
        candidates.push(BytePos::from_usize(offset));
        positions.push(offset);
    }
    if positions.last().copied() != Some(text.len()) {
        candidates.push(BytePos::from_usize(text.len()));
        positions.push(text.len());
    }

    let mut features = FeatureMatrix::<f32> {
        rows: candidates.len(),
        cols: kernel.schema().total_dim(),
        data: Vec::new(),
    };
    features.resize_zeroed(candidates.len(), kernel.schema().total_dim());
    kernel.extract_into(
        TextBytes::from_utf8(text),
        candidates.as_slice(),
        features.as_view_mut(),
        &mut FeatureScratch::default(),
    )?;

    let dataset = MultiLabelDataset {
        rows: features.rows,
        features,
        targets: vec![0; candidates.len() * heads.len()],
        output_dim: heads.len(),
        chars: text.len(),
        documents: 1,
        positives: vec![0; heads.len()],
    };
    let scores = model.predict_probabilities(&dataset, config.validation_predict_batch_size)?;
    Ok((positions, scores))
}

fn decode_label_spans(
    text: &str,
    positions: &[usize],
    scores: &[f32],
    output_dim: usize,
    output_index: usize,
    threshold: f32,
) -> Vec<DecodedInspectionSpan> {
    let mut spans = Vec::new();
    let mut start = None;
    let mut score_sum = 0.0_f32;
    let mut max_score = f32::NEG_INFINITY;
    let mut count = 0_usize;

    for (row_index, &position) in positions.iter().enumerate() {
        let score = scores[row_index * output_dim + output_index];
        let positive = position < text.len() && score >= threshold;
        match (start, positive) {
            (None, true) => {
                start = Some(position);
                score_sum = score;
                max_score = score;
                count = 1;
            }
            (Some(_), true) => {
                score_sum += score;
                max_score = max_score.max(score);
                count += 1;
            }
            (Some(span_start), false) => {
                if position > span_start {
                    spans.push(DecodedInspectionSpan {
                        start: span_start,
                        end: position,
                        avg_score: score_sum / count.max(1) as f32,
                        max_score,
                    });
                }
                start = None;
                score_sum = 0.0;
                max_score = f32::NEG_INFINITY;
                count = 0;
            }
            (None, false) => {}
        }
    }

    if let Some(span_start) = start {
        if text.len() > span_start {
            spans.push(DecodedInspectionSpan {
                start: span_start,
                end: text.len(),
                avg_score: score_sum / count.max(1) as f32,
                max_score,
            });
        }
    }

    spans
}

fn decode_label_points(
    positions: &[usize],
    scores: &[f32],
    output_dim: usize,
    output_index: usize,
    threshold: f32,
) -> Vec<(usize, f32)> {
    positions
        .iter()
        .enumerate()
        .filter_map(|(row_index, &position)| {
            let score = scores[row_index * output_dim + output_index];
            (score >= threshold).then_some((position, score))
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct SentenceBreakCandidate {
    end: usize,
    next_start: Option<usize>,
    score: f32,
    end_score: f32,
    next_start_score: f32,
    inside_drop: f32,
    selected: bool,
}

fn inspect_basic_sentence_breaks(
    text: &str,
    heads: &[OutputHead],
    positions: &[usize],
    scores: &[f32],
    thresholds: &BTreeMap<String, f32>,
) {
    let Some(inside_index) = find_head_index(heads, "sentence", SemanticTask::Inside) else {
        return;
    };
    let Some(end_index) = find_head_index(heads, "sentence", SemanticTask::End) else {
        return;
    };
    let start_index = find_head_index(heads, "sentence", SemanticTask::Start);
    let end_threshold = thresholds
        .get("sentence.end")
        .copied()
        .unwrap_or(0.20)
        .max(0.01);
    let break_threshold = (end_threshold * 0.55).max(0.08);

    let mut candidates = Vec::new();
    for end in sentence_punctuation_candidates(text) {
        let next_start = next_nonspace_position(text, end);
        let end_score = score_at_position(positions, scores, heads.len(), end_index, end);
        let next_start_score = start_index
            .and_then(|index| {
                next_start.map(|position| {
                    score_at_position(positions, scores, heads.len(), index, position)
                })
            })
            .unwrap_or(0.0);
        let left_inside = score_at_position(
            positions,
            scores,
            heads.len(),
            inside_index,
            end.saturating_sub(1),
        );
        let right_inside = next_start
            .map(|position| {
                score_at_position(positions, scores, heads.len(), inside_index, position)
            })
            .unwrap_or(0.0);
        let inside_drop = (left_inside - right_inside).max(0.0);
        let score = 0.70 * end_score + 0.20 * next_start_score + 0.10 * inside_drop;
        candidates.push(SentenceBreakCandidate {
            end,
            next_start,
            score,
            end_score,
            next_start_score,
            inside_drop,
            selected: score >= break_threshold,
        });
    }

    println!(
        "\n  basic_sentence_change_decoder threshold={break_threshold:.3}: {} candidate(s)",
        candidates.len()
    );
    if candidates.is_empty() {
        return;
    }
    for candidate in &candidates {
        println!(
            "    end={:>4} score={:.3} end_p={:.3} next_start_p={:.3} drop={:.3} selected={} {:?}",
            candidate.end,
            candidate.score,
            candidate.end_score,
            candidate.next_start_score,
            candidate.inside_drop,
            candidate.selected,
            point_context(text, candidate.next_start.unwrap_or(candidate.end), 54)
        );
    }

    let selected_breaks = candidates
        .iter()
        .filter(|candidate| candidate.selected)
        .map(|candidate| candidate.end)
        .collect::<Vec<_>>();
    if selected_breaks.is_empty() {
        println!("  basic_sentence_change_decoder annotated: <no selected breaks>");
    } else {
        println!(
            "  basic_sentence_change_decoder annotated:\n{}",
            render_sentence_breaks(text, &selected_breaks)
        );
    }

    let prior_breaks = candidates
        .iter()
        .map(|candidate| candidate.end)
        .collect::<Vec<_>>();
    println!(
        "  punctuation_prior_sentence_decoder annotated:\n{}",
        render_sentence_breaks(text, &prior_breaks)
    );
}

fn find_head_index(heads: &[OutputHead], label: &str, task: SemanticTask) -> Option<usize> {
    heads
        .iter()
        .position(|head| head.label == label && head.task == task)
}

fn score_at_position(
    positions: &[usize],
    scores: &[f32],
    output_dim: usize,
    output_index: usize,
    position: usize,
) -> f32 {
    let index = match positions.binary_search(&position) {
        Ok(index) => index,
        Err(next) => next.saturating_sub(1),
    };
    scores[index * output_dim + output_index]
}

fn sentence_punctuation_candidates(text: &str) -> Vec<usize> {
    let mut candidates = Vec::new();
    for (offset, ch) in text.char_indices() {
        if !matches!(ch, '.' | '!' | '?') {
            continue;
        }
        let end = offset + ch.len_utf8();
        if previous_char_is_ascii_digit(text, offset) && next_char_is_ascii_digit(text, end) {
            continue;
        }
        let Some(next_start) = next_nonspace_position(text, end) else {
            candidates.push(end);
            continue;
        };
        let Some(next_ch) = text[next_start..].chars().next() else {
            candidates.push(end);
            continue;
        };
        if next_ch.is_uppercase() || matches!(next_ch, '"' | '\'' | '#' | '-' | '*') {
            candidates.push(end);
        }
    }
    candidates
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

fn dataset_report(heads: &[OutputHead], dataset: &MultiLabelDataset) -> DatasetReport {
    let positives_by_output = heads
        .iter()
        .enumerate()
        .map(|(index, head)| (output_name(head), dataset.positives[index]))
        .collect();
    DatasetReport {
        documents: dataset.documents,
        rows: dataset.rows,
        chars: dataset.chars,
        positives_by_output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence_heads() -> Vec<OutputHead> {
        vec![
            OutputHead {
                label: "sentence".to_string(),
                task: SemanticTask::Inside,
            },
            OutputHead {
                label: "sentence".to_string(),
                task: SemanticTask::Start,
            },
            OutputHead {
                label: "sentence".to_string(),
                task: SemanticTask::End,
            },
            OutputHead {
                label: "paragraph".to_string(),
                task: SemanticTask::Inside,
            },
        ]
    }

    #[test]
    fn unicode_byte_offsets_are_validated_as_bytes() {
        let text = "A “quoted” sentence.".to_string();
        assert!(text.len() > text.chars().count());
        let record = SyntheticRecord {
            text: text.clone(),
            spans: vec![SyntheticSpan {
                label: "sentence".to_string(),
                start: 0,
                end: text.len(),
                left_open: false,
                right_open: false,
            }],
        };
        let labels = BTreeSet::from(["sentence"]);
        let cleaned = clean_record(record, &labels).expect("valid byte offsets should load");
        assert_eq!(cleaned.spans[0].end, text.len());
    }

    #[test]
    fn overlapping_labels_set_independent_inside_targets() {
        let text = "One sentence.".to_string();
        let record = CleanRecord {
            text: text.clone(),
            spans: vec![
                SyntheticSpan {
                    label: "paragraph".to_string(),
                    start: 0,
                    end: text.len(),
                    left_open: false,
                    right_open: false,
                },
                SyntheticSpan {
                    label: "sentence".to_string(),
                    start: 0,
                    end: text.len(),
                    left_open: false,
                    right_open: false,
                },
            ],
        };
        let targets = targets_for_position(&record, 4, &sentence_heads());
        assert_eq!(targets, vec![1, 0, 0, 1]);
    }

    #[test]
    fn adjacent_same_label_spans_mark_end_and_start_at_shared_boundary() {
        let text = "One. Two.".to_string();
        let boundary = "One.".len();
        let record = CleanRecord {
            text: text.clone(),
            spans: vec![
                SyntheticSpan {
                    label: "sentence".to_string(),
                    start: 0,
                    end: boundary,
                    left_open: false,
                    right_open: false,
                },
                SyntheticSpan {
                    label: "sentence".to_string(),
                    start: boundary,
                    end: text.len(),
                    left_open: false,
                    right_open: false,
                },
            ],
        };
        let targets = targets_for_position(&record, boundary, &sentence_heads());
        assert_eq!(targets, vec![1, 1, 1, 0]);
    }

    #[test]
    fn open_edges_do_not_create_start_or_end_targets() {
        let text = "continued sentence fragment".to_string();
        let record = CleanRecord {
            text: text.clone(),
            spans: vec![SyntheticSpan {
                label: "sentence".to_string(),
                start: 0,
                end: text.len(),
                left_open: true,
                right_open: true,
            }],
        };
        let left_targets = targets_for_position(&record, 0, &sentence_heads());
        let right_targets = targets_for_position(&record, text.len(), &sentence_heads());
        assert_eq!(left_targets, vec![1, 0, 0, 0]);
        assert_eq!(right_targets, vec![0, 0, 0, 0]);
    }
}
