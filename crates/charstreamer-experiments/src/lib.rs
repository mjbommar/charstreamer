use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use charstreamer_backend_burn::{
    BurnDeepMlpFitOptions, BurnDeepMlpFitReport, BurnDeepMlpModel, BurnShallowMlpFitOptions,
    BurnShallowMlpFitReport, BurnShallowMlpModel, BurnWindowCnnFitOptions, BurnWindowCnnFitReport,
    BurnWindowCnnModel, BurnWindowGruFitOptions, BurnWindowGruFitReport, BurnWindowGruModel,
    BurnWindowLstmFitOptions, BurnWindowLstmFitReport, BurnWindowLstmModel,
};
use charstreamer_core::{
    BatchPredictor, BinaryMetrics, BoundaryDatasetBuildOptions, CandidateScanner, CorpusError,
    Decoder, FeatureAppender, FeatureError, FeatureKernel, FeatureMatrix, FeatureScratch, FitError,
    Pipeline, PipelineError, PredictError, TextBytes, ThresholdSpanDecoder, ThroughputReport,
    TrainablePredictor, benchmark_pipeline, best_threshold_from_scores, build_boundary_dataset,
    evaluate_pipeline, load_alea_jsonl, load_multilegal_jsonl, metrics_from_scores,
    split_documents,
};
use charstreamer_kernels::{
    AsciiClassAppender, BoundaryHeuristicAppender, ByteClass, ByteClassCountAppender, ByteSet256,
    ByteSetScanner, ByteWindowAppender, CharBoundaryLegacyAppender, CompositeFeatureKernel,
    DirectionalByteClassCountAppender, DirectionalUnicodeCategoryCountAppender,
    DirectionalUnicodeCategoryGroupCountAppender, EncodedByteWindowAppender, LegacyFeatureTables,
    LegalBoundaryHeuristicAppender, LineByteCountAppender, LineStartScanner,
    SelectedByteCountAppender, StrideScanner, UnicodeCategory, UnicodeCategoryGroup,
    Utf8CharSetScanner,
};
use charstreamer_models_native::{
    LogisticFitOptions, LogisticFitReport, LogisticModel, ModelIoError,
};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum ExperimentError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Corpus(CorpusError),
    Feature(FeatureError),
    Pipeline(PipelineError),
    Fit(FitError),
    Predict(PredictError),
    ModelIo(ModelIoError),
    Python(String),
    Unsupported(String),
}

impl Display for ExperimentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error while running experiment: {error}"),
            Self::Json(error) => write!(f, "JSON error while running experiment: {error}"),
            Self::Corpus(error) => write!(f, "corpus error while running experiment: {error}"),
            Self::Feature(error) => write!(f, "feature error while running experiment: {error}"),
            Self::Pipeline(error) => {
                write!(f, "pipeline error while running experiment: {error}")
            }
            Self::Fit(error) => write!(f, "fit error while running experiment: {error}"),
            Self::Predict(error) => write!(f, "predict error while running experiment: {error}"),
            Self::ModelIo(error) => write!(f, "model I/O error while running experiment: {error}"),
            Self::Python(message) => write!(f, "python subprocess failed: {message}"),
            Self::Unsupported(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ExperimentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Corpus(error) => Some(error),
            Self::Feature(error) => Some(error),
            Self::Pipeline(error) => Some(error),
            Self::Fit(error) => Some(error),
            Self::Predict(error) => Some(error),
            Self::ModelIo(error) => Some(error),
            Self::Python(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<std::io::Error> for ExperimentError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ExperimentError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<CorpusError> for ExperimentError {
    fn from(error: CorpusError) -> Self {
        Self::Corpus(error)
    }
}

impl From<FeatureError> for ExperimentError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}

impl From<PipelineError> for ExperimentError {
    fn from(error: PipelineError) -> Self {
        Self::Pipeline(error)
    }
}

impl From<FitError> for ExperimentError {
    fn from(error: FitError) -> Self {
        Self::Fit(error)
    }
}

impl From<PredictError> for ExperimentError {
    fn from(error: PredictError) -> Self {
        Self::Predict(error)
    }
}

impl From<ModelIoError> for ExperimentError {
    fn from(error: ModelIoError) -> Self {
        Self::ModelIo(error)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScannerSpec {
    ByteSet { bytes: Vec<u8> },
    Utf8CharSet { chars: Vec<char> },
    CharBoundarySentenceCandidates { constants_py_path: PathBuf },
    LineStart,
    Stride { stride: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FeatureComponentSpec {
    ByteWindow {
        left: usize,
        right: usize,
    },
    EncodedByteWindow {
        left: usize,
        right: usize,
    },
    AsciiNeighborClasses,
    BoundaryHeuristics,
    LegalBoundaryHeuristics,
    SelectedByteCounts {
        left: usize,
        right: usize,
        bytes: Vec<u8>,
    },
    ByteClassCounts {
        left: usize,
        right: usize,
        classes: Vec<ByteClass>,
    },
    DirectionalByteClassCounts {
        left: usize,
        right: usize,
        classes: Vec<ByteClass>,
    },
    DirectionalUnicodeCategoryCounts {
        left: usize,
        right: usize,
        categories: Vec<UnicodeCategory>,
    },
    DirectionalUnicodeCategoryGroupCounts {
        left: usize,
        right: usize,
        groups: Vec<UnicodeCategoryGroup>,
    },
    LineByteCounts {
        bytes: Vec<u8>,
    },
    CharBoundaryLegacy {
        left_window: usize,
        right_window: usize,
        constants_py_path: PathBuf,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelSpec {
    BurnShallowMlp {
        fit: BurnShallowMlpFitOptions,
    },
    BurnDeepMlp {
        fit: BurnDeepMlpFitOptions,
    },
    BurnWindowCnn {
        fit: BurnWindowCnnFitOptions,
    },
    BurnWindowGru {
        fit: BurnWindowGruFitOptions,
    },
    BurnWindowLstm {
        fit: BurnWindowLstmFitOptions,
    },
    NativeLogistic {
        fit: LogisticFitOptions,
    },
    PythonSklearnRandomForest {
        fit: PythonSklearnRandomForestFitOptions,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum ThresholdPolicy {
    #[default]
    TuneOnValidation,
    Fixed {
        value: f32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonSklearnRandomForestFitOptions {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<String>,
    pub class_weight: Option<String>,
    pub n_jobs: isize,
    pub random_state: Option<u64>,
}

impl Default for PythonSklearnRandomForestFitOptions {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            class_weight: Some("balanced".to_string()),
            n_jobs: -1,
            random_state: Some(7),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CorpusFormat {
    Alea,
    MultiLegal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationCorpusSpec {
    pub name: String,
    pub path: PathBuf,
    pub format: CorpusFormat,
    pub limit: Option<usize>,
    pub throughput_iterations: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryExperimentSpec {
    pub name: String,
    pub training_path: PathBuf,
    pub training_format: CorpusFormat,
    pub training_limit: Option<usize>,
    pub split_numerator: usize,
    pub split_denominator: usize,
    pub scanner: ScannerSpec,
    pub features: Vec<FeatureComponentSpec>,
    pub model: ModelSpec,
    pub dataset_options: BoundaryDatasetBuildOptions,
    #[serde(default)]
    pub threshold_policy: ThresholdPolicy,
    pub validation_throughput_iterations: usize,
    pub evaluation_corpora: Vec<EvaluationCorpusSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelFitReport {
    BurnShallowMlp(BurnShallowMlpFitReport),
    BurnDeepMlp(BurnDeepMlpFitReport),
    BurnWindowCnn(BurnWindowCnnFitReport),
    BurnWindowGru(BurnWindowGruFitReport),
    BurnWindowLstm(BurnWindowLstmFitReport),
    NativeLogistic(LogisticFitReport),
    PythonSklearnRandomForest(PythonSklearnRandomForestFitReport),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonSklearnRandomForestFitReport {
    pub rows: usize,
    pub cols: usize,
    pub model_path: PathBuf,
    pub sklearn_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusEvaluationReport {
    pub name: String,
    pub path: PathBuf,
    pub format: CorpusFormat,
    pub documents: usize,
    pub span_metrics: BinaryMetrics,
    pub throughput: ThroughputReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryExperimentReport {
    pub spec: BoundaryExperimentSpec,
    pub training_seconds: f64,
    pub threshold: f32,
    pub train_documents: usize,
    pub valid_documents: usize,
    pub train_rows: usize,
    pub train_positives: usize,
    pub train_negatives: usize,
    pub fit_report: ModelFitReport,
    pub candidate_metrics: BinaryMetrics,
    pub validation: CorpusEvaluationReport,
    pub evaluations: Vec<CorpusEvaluationReport>,
}

pub struct TrainedBoundaryPipeline {
    threshold: f32,
    pipeline: Pipeline<
        CompiledScanner,
        CompositeFeatureKernel,
        CompiledBinaryModel,
        ThresholdSpanDecoder,
    >,
}

#[derive(Clone, Debug)]
pub struct TextInspectionReport {
    pub bytes: usize,
    pub candidate_count: usize,
    pub candidates: Vec<charstreamer_core::BytePos>,
    pub scores: Vec<f32>,
    pub predicted_spans: Vec<charstreamer_core::ByteSpan>,
}

impl TrainedBoundaryPipeline {
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn inspect_text(&self, text: &str) -> Result<TextInspectionReport, ExperimentError> {
        let mut workspace = charstreamer_core::PipelineWorkspace::<f32, f32>::default();
        let mut predicted_spans = Vec::new();
        self.pipeline.run_into(
            TextBytes::from_utf8(text),
            &mut workspace,
            &mut predicted_spans,
        )?;
        Ok(TextInspectionReport {
            bytes: text.len(),
            candidate_count: workspace.candidates.len(),
            candidates: workspace.candidates.positions().to_vec(),
            scores: workspace.scores.data.clone(),
            predicted_spans,
        })
    }

    pub fn benchmark_text(
        &self,
        text: &str,
        iterations: usize,
    ) -> Result<ThroughputReport, ExperimentError> {
        let iterations = iterations.max(1);
        let mut workspace = charstreamer_core::PipelineWorkspace::<f32, f32>::default();
        let mut predicted_spans = Vec::new();
        let started = Instant::now();
        for _ in 0..iterations {
            predicted_spans.clear();
            self.pipeline.run_into(
                TextBytes::from_utf8(text),
                &mut workspace,
                &mut predicted_spans,
            )?;
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let chars_per_second = if elapsed_seconds > 0.0 {
            text.len() as f64 * iterations as f64 / elapsed_seconds
        } else {
            0.0
        };
        Ok(ThroughputReport {
            total_chars: text.len(),
            total_documents: 1,
            iterations,
            elapsed_seconds,
            chars_per_second,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParityCheckSpec {
    pub text: String,
    pub char_positions: Vec<usize>,
    pub left_window: usize,
    pub right_window: usize,
    pub constants_py_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParityRowMismatch {
    pub row_index: usize,
    pub rust_row: Vec<i32>,
    pub python_row: Vec<i32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParityCheckReport {
    pub compared_rows: usize,
    pub exact_match: bool,
    pub mismatched_rows: Vec<usize>,
    pub mismatch_details: Vec<ParityRowMismatch>,
}

#[derive(Clone, Debug, Deserialize)]
struct PythonLegacyTablesPayload {
    abbreviations: Vec<String>,
    list_markers: Vec<String>,
    list_conjunctions: Vec<String>,
    list_intros: Vec<String>,
    terminal_sentence_chars: Vec<String>,
    terminal_paragraph_chars: Vec<String>,
    primary_terminators: Vec<String>,
    secondary_terminators: Vec<String>,
    opening_quotes: Vec<String>,
    closing_quotes: Vec<String>,
    punctuation_chars: Vec<String>,
    whitespace_chars: Vec<String>,
}

#[derive(Clone, Debug)]
enum CompiledScanner {
    ByteSet(Box<ByteSetScanner>),
    Utf8CharSet(Utf8CharSetScanner),
    LineStart(LineStartScanner),
    Stride(StrideScanner),
}

impl CandidateScanner for CompiledScanner {
    fn scan_into(
        &self,
        text: TextBytes<'_>,
        range: charstreamer_core::ScanRange,
        out: &mut charstreamer_core::CandidateBuffer,
    ) {
        match self {
            Self::ByteSet(scanner) => scanner.scan_into(text, range, out),
            Self::Utf8CharSet(scanner) => scanner.scan_into(text, range, out),
            Self::LineStart(scanner) => scanner.scan_into(text, range, out),
            Self::Stride(scanner) => scanner.scan_into(text, range, out),
        }
    }
}

#[derive(Debug)]
enum CompiledBinaryModel {
    BurnShallowMlp(Box<BurnShallowMlpModel>),
    BurnDeepMlp(Box<BurnDeepMlpModel>),
    BurnWindowCnn(Box<BurnWindowCnnModel>),
    BurnWindowGru(Box<BurnWindowGruModel>),
    BurnWindowLstm(Box<BurnWindowLstmModel>),
    NativeLogistic(Box<LogisticModel>),
    PythonSklearnRandomForest(Box<PythonSklearnRandomForestModel>),
}

impl BatchPredictor<f32, f32> for CompiledBinaryModel {
    fn predict_into(
        &self,
        features: charstreamer_core::FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        match self {
            Self::BurnShallowMlp(model) => model.predict_into(features, out),
            Self::BurnDeepMlp(model) => model.predict_into(features, out),
            Self::BurnWindowCnn(model) => model.predict_into(features, out),
            Self::BurnWindowGru(model) => model.predict_into(features, out),
            Self::BurnWindowLstm(model) => model.predict_into(features, out),
            Self::NativeLogistic(model) => model.predict_into(features, out),
            Self::PythonSklearnRandomForest(model) => model.predict_into(features, out),
        }
    }
}

impl CompiledBinaryModel {
    fn prefers_corpus_batching(&self) -> bool {
        matches!(self, Self::PythonSklearnRandomForest(_))
    }
}

#[derive(Debug)]
struct PythonSklearnRandomForestModel {
    model_path: PathBuf,
}

impl PythonSklearnRandomForestModel {
    fn train(
        dataset: charstreamer_core::DatasetView<'_, f32, u8>,
        options: &PythonSklearnRandomForestFitOptions,
    ) -> Result<(Self, PythonSklearnRandomForestFitReport), ExperimentError> {
        if dataset.features.rows != dataset.labels.len() {
            return Err(ExperimentError::Unsupported(
                "dataset feature rows and labels must have matching lengths".to_string(),
            ));
        }
        if dataset.features.rows == 0 || dataset.features.cols == 0 {
            return Err(ExperimentError::Unsupported(
                "python sklearn random forest training requires non-empty features and labels"
                    .to_string(),
            ));
        }

        let model_path = temp_model_path("python-sklearn-rf", "joblib");
        let features = feature_rows_as_vec(dataset.features);
        let labels = dataset.labels.to_vec();
        let input = serde_json::json!({
            "model_path": model_path,
            "features": features,
            "labels": labels,
            "options": options,
        });
        let script = r#"
import json
import pathlib
import sys

from joblib import dump
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier

payload = json.loads(sys.stdin.read())
options = payload["options"]

model = RandomForestClassifier(
    n_estimators=options["n_estimators"],
    max_depth=options["max_depth"],
    min_samples_split=options["min_samples_split"],
    min_samples_leaf=options["min_samples_leaf"],
    max_features=options["max_features"],
    class_weight=options["class_weight"],
    n_jobs=options["n_jobs"],
    random_state=options["random_state"],
)
model.fit(payload["features"], payload["labels"])

model_path = pathlib.Path(payload["model_path"])
model_path.parent.mkdir(parents=True, exist_ok=True)
dump(model, model_path)
print(json.dumps({
    "rows": len(payload["features"]),
    "cols": len(payload["features"][0]) if payload["features"] else 0,
    "model_path": str(model_path),
    "sklearn_version": sklearn_version,
}))
"#;
        let report: PythonSklearnRandomForestFitReport = run_python_json_script(script, &input)?;
        Ok((Self { model_path }, report))
    }
}

impl Drop for PythonSklearnRandomForestModel {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.model_path);
    }
}

impl BatchPredictor<f32, f32> for PythonSklearnRandomForestModel {
    fn predict_into(
        &self,
        features: charstreamer_core::FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        if out.len() < features.rows {
            return Err(PredictError::new("score output buffer is too small"));
        }
        if features.rows == 0 {
            return Ok(());
        }

        let input = serde_json::json!({
            "model_path": self.model_path,
            "features": feature_rows_as_vec(features),
        });
        let script = r#"
import json
import sys

from joblib import load

payload = json.loads(sys.stdin.read())
model = load(payload["model_path"])
probas = model.predict_proba(payload["features"])
scores = [float(row[1]) if len(row) > 1 else float(row[0]) for row in probas]
print(json.dumps(scores))
"#;
        let scores: Vec<f32> = run_python_json_script(script, &input)
            .map_err(|error| PredictError::new(error.to_string()))?;
        for (slot, score) in out.iter_mut().zip(scores.into_iter()) {
            *slot = score;
        }
        Ok(())
    }
}

pub fn load_charboundary_legacy_tables(
    constants_py_path: impl AsRef<Path>,
) -> Result<LegacyFeatureTables, ExperimentError> {
    let script = r#"
import importlib.util
import json
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("cb_constants", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def ordered(values):
    if isinstance(values, (set, frozenset)):
        return sorted(values)
    return list(values)

payload = {
    "abbreviations": list(module.DEFAULT_ABBREVIATIONS),
    "list_markers": list(module.LIST_MARKERS),
    "list_conjunctions": list(module.LIST_CONJUNCTIONS),
    "list_intros": list(module.LIST_INTROS),
    "terminal_sentence_chars": ordered(module.TERMINAL_SENTENCE_CHAR_LIST),
    "terminal_paragraph_chars": ordered(module.TERMINAL_PARAGRAPH_CHAR_LIST),
    "primary_terminators": ordered(module.PRIMARY_TERMINATORS),
    "secondary_terminators": ordered(module.SECONDARY_TERMINATORS),
    "opening_quotes": ordered(module.OPENING_QUOTES),
    "closing_quotes": ordered(module.CLOSING_QUOTES),
    "punctuation_chars": ordered(module.PUNCTUATION_CHAR_LIST),
    "whitespace_chars": ordered(module.WS_CHAR_LIST),
}
print(json.dumps(payload))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(constants_py_path.as_ref())
        .output()?;
    if !output.status.success() {
        return Err(ExperimentError::Python(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    let payload: PythonLegacyTablesPayload = serde_json::from_slice(&output.stdout)?;

    Ok(LegacyFeatureTables {
        abbreviations: payload.abbreviations.into_iter().collect(),
        list_markers: payload.list_markers,
        list_conjunctions: payload.list_conjunctions,
        list_intros: payload.list_intros,
        terminal_sentence_chars: single_char_set(payload.terminal_sentence_chars),
        terminal_paragraph_chars: single_char_set(payload.terminal_paragraph_chars),
        primary_terminators: single_char_set(payload.primary_terminators),
        secondary_terminators: single_char_set(payload.secondary_terminators),
        opening_quotes: single_char_set(payload.opening_quotes),
        closing_quotes: single_char_set(payload.closing_quotes),
        punctuation_chars: single_char_set(payload.punctuation_chars),
        whitespace_chars: single_char_set(payload.whitespace_chars),
    })
}

fn resolve_relative_path(path: &mut PathBuf, base: &Path) {
    if path.is_relative() {
        *path = base.join(&*path);
    }
}

impl ScannerSpec {
    pub fn resolve_relative_to(&mut self, base: &Path) {
        if let Self::CharBoundarySentenceCandidates { constants_py_path } = self {
            resolve_relative_path(constants_py_path, base);
        }
    }
}

impl FeatureComponentSpec {
    pub fn resolve_relative_to(&mut self, base: &Path) {
        if let Self::CharBoundaryLegacy {
            constants_py_path, ..
        } = self
        {
            resolve_relative_path(constants_py_path, base);
        }
    }
}

impl EvaluationCorpusSpec {
    pub fn resolve_relative_to(&mut self, base: &Path) {
        resolve_relative_path(&mut self.path, base);
    }
}

impl BoundaryExperimentSpec {
    pub fn resolve_relative_to(&mut self, base: &Path) {
        resolve_relative_path(&mut self.training_path, base);
        self.scanner.resolve_relative_to(base);
        for feature in &mut self.features {
            feature.resolve_relative_to(base);
        }
        for evaluation in &mut self.evaluation_corpora {
            evaluation.resolve_relative_to(base);
        }
    }
}

impl ParityCheckSpec {
    pub fn resolve_relative_to(&mut self, base: &Path) {
        resolve_relative_path(&mut self.constants_py_path, base);
    }
}

pub fn read_boundary_experiment_spec(
    path: impl AsRef<Path>,
) -> Result<BoundaryExperimentSpec, ExperimentError> {
    let path = path.as_ref();
    let mut spec: BoundaryExperimentSpec = serde_json::from_slice(&fs::read(path)?)?;
    if let Some(base) = path.parent() {
        spec.resolve_relative_to(base);
    }
    Ok(spec)
}

pub fn read_parity_check_spec(path: impl AsRef<Path>) -> Result<ParityCheckSpec, ExperimentError> {
    let path = path.as_ref();
    let mut spec: ParityCheckSpec = serde_json::from_slice(&fs::read(path)?)?;
    if let Some(base) = path.parent() {
        spec.resolve_relative_to(base);
    }
    Ok(spec)
}

fn compile_scanner(spec: &ScannerSpec) -> Result<CompiledScanner, ExperimentError> {
    Ok(match spec {
        ScannerSpec::ByteSet { bytes } => {
            CompiledScanner::ByteSet(Box::new(ByteSetScanner::new(ByteSet256::from_bytes(bytes))))
        }
        ScannerSpec::Utf8CharSet { chars } => {
            CompiledScanner::Utf8CharSet(Utf8CharSetScanner::new(chars.clone()))
        }
        ScannerSpec::CharBoundarySentenceCandidates { constants_py_path } => {
            let tables = load_charboundary_legacy_tables(constants_py_path)?;
            let mut chars: Vec<char> = tables.terminal_sentence_chars.iter().copied().collect();
            chars.extend(tables.terminal_paragraph_chars.iter().copied());
            CompiledScanner::Utf8CharSet(Utf8CharSetScanner::new(chars))
        }
        ScannerSpec::LineStart => CompiledScanner::LineStart(LineStartScanner::new()),
        ScannerSpec::Stride { stride } => CompiledScanner::Stride(StrideScanner::new(*stride)),
    })
}

pub fn compile_feature_kernel(
    specs: &[FeatureComponentSpec],
) -> Result<CompositeFeatureKernel, ExperimentError> {
    let mut appenders: Vec<Box<dyn FeatureAppender<f32> + Send + Sync>> =
        Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            FeatureComponentSpec::ByteWindow { left, right } => appenders.push(Box::new(
                ByteWindowAppender::new(charstreamer_core::ByteWindowSpec::new(*left, *right)),
            )),
            FeatureComponentSpec::EncodedByteWindow { left, right } => {
                appenders.push(Box::new(EncodedByteWindowAppender::new(
                    charstreamer_core::ByteWindowSpec::new(*left, *right),
                )))
            }
            FeatureComponentSpec::AsciiNeighborClasses => {
                appenders.push(Box::new(AsciiClassAppender::new()))
            }
            FeatureComponentSpec::BoundaryHeuristics => {
                appenders.push(Box::new(BoundaryHeuristicAppender::new()))
            }
            FeatureComponentSpec::LegalBoundaryHeuristics => {
                appenders.push(Box::new(LegalBoundaryHeuristicAppender::new()))
            }
            FeatureComponentSpec::SelectedByteCounts { left, right, bytes } => {
                appenders.push(Box::new(SelectedByteCountAppender::new(
                    "selected_byte_counts",
                    charstreamer_core::ByteWindowSpec::new(*left, *right),
                    bytes.clone(),
                )))
            }
            FeatureComponentSpec::ByteClassCounts {
                left,
                right,
                classes,
            } => appenders.push(Box::new(ByteClassCountAppender::new(
                "byte_class_counts",
                charstreamer_core::ByteWindowSpec::new(*left, *right),
                classes.clone(),
            ))),
            FeatureComponentSpec::DirectionalByteClassCounts {
                left,
                right,
                classes,
            } => appenders.push(Box::new(DirectionalByteClassCountAppender::new(
                "directional_byte_class_counts",
                charstreamer_core::ByteWindowSpec::new(*left, *right),
                classes.clone(),
            ))),
            FeatureComponentSpec::DirectionalUnicodeCategoryCounts {
                left,
                right,
                categories,
            } => appenders.push(Box::new(DirectionalUnicodeCategoryCountAppender::new(
                "directional_unicode_category_counts",
                charstreamer_core::ByteWindowSpec::new(*left, *right),
                categories.clone(),
            ))),
            FeatureComponentSpec::DirectionalUnicodeCategoryGroupCounts {
                left,
                right,
                groups,
            } => appenders.push(Box::new(DirectionalUnicodeCategoryGroupCountAppender::new(
                "directional_unicode_category_group_counts",
                charstreamer_core::ByteWindowSpec::new(*left, *right),
                groups.clone(),
            ))),
            FeatureComponentSpec::LineByteCounts { bytes } => appenders.push(Box::new(
                LineByteCountAppender::new("line_byte_counts", bytes.clone()),
            )),
            FeatureComponentSpec::CharBoundaryLegacy {
                left_window,
                right_window,
                constants_py_path,
            } => appenders.push(Box::new(CharBoundaryLegacyAppender::new(
                *left_window,
                *right_window,
                load_charboundary_legacy_tables(constants_py_path)?,
            ))),
        }
    }

    Ok(CompositeFeatureKernel::new(appenders))
}

pub fn compare_charboundary_legacy_features(
    spec: &ParityCheckSpec,
) -> Result<ParityCheckReport, ExperimentError> {
    let tables = load_charboundary_legacy_tables(&spec.constants_py_path)?;
    let appender = CharBoundaryLegacyAppender::new(spec.left_window, spec.right_window, tables);
    let byte_positions = char_positions_to_byte_offsets(&spec.text, &spec.char_positions)?;
    let mut candidates = charstreamer_core::CandidateBuffer::new();
    for byte_offset in byte_positions {
        candidates.push(charstreamer_core::BytePos::from_usize(byte_offset));
    }

    let mut matrix = FeatureMatrix::<f32>::default();
    matrix.resize_zeroed(candidates.len(), appender.block().width);
    appender.append_into(
        TextBytes::from_utf8(&spec.text),
        candidates.as_slice(),
        matrix.as_view_mut(),
        &mut FeatureScratch::default(),
    )?;

    let python_rows = python_charboundary_feature_rows(spec)?;
    let rust_rows: Vec<Vec<i32>> = matrix
        .data
        .chunks(matrix.cols)
        .map(|row| row.iter().map(|value| *value as i32).collect())
        .collect();

    let mut mismatched_rows = Vec::new();
    let mut mismatch_details = Vec::new();
    for (row_index, (rust_row, python_row)) in rust_rows.iter().zip(&python_rows).enumerate() {
        if rust_row != python_row {
            mismatched_rows.push(row_index);
            if mismatch_details.len() < 8 {
                mismatch_details.push(ParityRowMismatch {
                    row_index,
                    rust_row: rust_row.clone(),
                    python_row: python_row.clone(),
                });
            }
        }
    }

    Ok(ParityCheckReport {
        compared_rows: python_rows.len(),
        exact_match: mismatched_rows.is_empty(),
        mismatched_rows,
        mismatch_details,
    })
}

pub fn run_boundary_experiment(
    spec: &BoundaryExperimentSpec,
) -> Result<BoundaryExperimentReport, ExperimentError> {
    let (_, report) = train_boundary_pipeline(spec)?;
    Ok(report)
}

pub fn train_boundary_pipeline(
    spec: &BoundaryExperimentSpec,
) -> Result<(TrainedBoundaryPipeline, BoundaryExperimentReport), ExperimentError> {
    let training_documents = load_documents(
        &spec.training_path,
        &spec.training_format,
        spec.training_limit,
    )?;
    let (train_docs, valid_docs) = split_documents(
        training_documents,
        spec.split_numerator,
        spec.split_denominator,
    );
    let scanner = compile_scanner(&spec.scanner)?;
    let kernel = compile_feature_kernel(&spec.features)?;

    let started = Instant::now();
    let train_dataset =
        build_boundary_dataset(&train_docs, &scanner, &kernel, &spec.dataset_options)?;
    let (model, fit_report) = fit_model(&spec.model, train_dataset.as_view())?;
    let training_seconds = started.elapsed().as_secs_f64();

    let valid_dataset =
        build_boundary_dataset(&valid_docs, &scanner, &kernel, &spec.dataset_options)?;
    let mut valid_scores = vec![0.0_f32; valid_dataset.rows()];
    if !valid_scores.is_empty() {
        model.predict_into(valid_dataset.features.as_view(), &mut valid_scores)?;
    }
    let threshold = match spec.threshold_policy {
        ThresholdPolicy::TuneOnValidation => {
            if valid_scores.is_empty() {
                0.5
            } else {
                best_threshold_from_scores(&valid_scores, &valid_dataset.labels)
            }
        }
        ThresholdPolicy::Fixed { value } => value,
    };
    let candidate_metrics = metrics_from_scores(&valid_scores, &valid_dataset.labels, threshold);
    let decoder = ThresholdSpanDecoder::new(threshold);
    let pipeline = Pipeline::new(scanner, kernel, model, decoder);

    let (validation, evaluations) = if pipeline.model().prefers_corpus_batching() {
        let validation = CorpusEvaluationReport {
            name: "validation".to_string(),
            path: spec.training_path.clone(),
            format: spec.training_format.clone(),
            documents: valid_docs.len(),
            span_metrics: evaluate_corpus_batched(
                pipeline.scanner(),
                pipeline.kernel(),
                pipeline.model(),
                pipeline.decoder(),
                &valid_docs,
            )?,
            throughput: benchmark_corpus_batched(
                pipeline.scanner(),
                pipeline.kernel(),
                pipeline.model(),
                pipeline.decoder(),
                &valid_docs,
                spec.validation_throughput_iterations,
            )?,
        };

        let mut evaluations = Vec::with_capacity(spec.evaluation_corpora.len());
        for evaluation in &spec.evaluation_corpora {
            let documents = load_documents(&evaluation.path, &evaluation.format, evaluation.limit)?;
            evaluations.push(CorpusEvaluationReport {
                name: evaluation.name.clone(),
                path: evaluation.path.clone(),
                format: evaluation.format.clone(),
                documents: documents.len(),
                span_metrics: evaluate_corpus_batched(
                    pipeline.scanner(),
                    pipeline.kernel(),
                    pipeline.model(),
                    pipeline.decoder(),
                    &documents,
                )?,
                throughput: benchmark_corpus_batched(
                    pipeline.scanner(),
                    pipeline.kernel(),
                    pipeline.model(),
                    pipeline.decoder(),
                    &documents,
                    evaluation.throughput_iterations,
                )?,
            });
        }
        (validation, evaluations)
    } else {
        let validation = CorpusEvaluationReport {
            name: "validation".to_string(),
            path: spec.training_path.clone(),
            format: spec.training_format.clone(),
            documents: valid_docs.len(),
            span_metrics: evaluate_pipeline(&pipeline, &valid_docs)?,
            throughput: benchmark_pipeline(
                &pipeline,
                &valid_docs,
                spec.validation_throughput_iterations,
            )?,
        };

        let mut evaluations = Vec::with_capacity(spec.evaluation_corpora.len());
        for evaluation in &spec.evaluation_corpora {
            let documents = load_documents(&evaluation.path, &evaluation.format, evaluation.limit)?;
            evaluations.push(CorpusEvaluationReport {
                name: evaluation.name.clone(),
                path: evaluation.path.clone(),
                format: evaluation.format.clone(),
                documents: documents.len(),
                span_metrics: evaluate_pipeline(&pipeline, &documents)?,
                throughput: benchmark_pipeline(
                    &pipeline,
                    &documents,
                    evaluation.throughput_iterations,
                )?,
            });
        }
        (validation, evaluations)
    };

    let report = BoundaryExperimentReport {
        spec: spec.clone(),
        training_seconds,
        threshold,
        train_documents: train_docs.len(),
        valid_documents: valid_docs.len(),
        train_rows: train_dataset.rows(),
        train_positives: train_dataset.positives,
        train_negatives: train_dataset.negatives,
        fit_report,
        candidate_metrics,
        validation,
        evaluations,
    };

    Ok((
        TrainedBoundaryPipeline {
            threshold,
            pipeline,
        },
        report,
    ))
}

pub fn write_report(
    path: impl AsRef<Path>,
    report: &BoundaryExperimentReport,
) -> Result<(), ExperimentError> {
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn fit_model(
    spec: &ModelSpec,
    dataset: charstreamer_core::DatasetView<'_, f32, u8>,
) -> Result<(CompiledBinaryModel, ModelFitReport), ExperimentError> {
    match spec {
        ModelSpec::BurnShallowMlp { fit } => {
            let (model, report) = BurnShallowMlpModel::fit(
                dataset,
                fit,
                &mut charstreamer_core::FitScratch::default(),
            )?;
            Ok((
                CompiledBinaryModel::BurnShallowMlp(Box::new(model)),
                ModelFitReport::BurnShallowMlp(report),
            ))
        }
        ModelSpec::BurnDeepMlp { fit } => {
            let (model, report) =
                BurnDeepMlpModel::fit(dataset, fit, &mut charstreamer_core::FitScratch::default())?;
            Ok((
                CompiledBinaryModel::BurnDeepMlp(Box::new(model)),
                ModelFitReport::BurnDeepMlp(report),
            ))
        }
        ModelSpec::BurnWindowCnn { fit } => {
            let (model, report) = BurnWindowCnnModel::fit(
                dataset,
                fit,
                &mut charstreamer_core::FitScratch::default(),
            )?;
            Ok((
                CompiledBinaryModel::BurnWindowCnn(Box::new(model)),
                ModelFitReport::BurnWindowCnn(report),
            ))
        }
        ModelSpec::BurnWindowGru { fit } => {
            let (model, report) = BurnWindowGruModel::fit(
                dataset,
                fit,
                &mut charstreamer_core::FitScratch::default(),
            )?;
            Ok((
                CompiledBinaryModel::BurnWindowGru(Box::new(model)),
                ModelFitReport::BurnWindowGru(report),
            ))
        }
        ModelSpec::BurnWindowLstm { fit } => {
            let (model, report) = BurnWindowLstmModel::fit(
                dataset,
                fit,
                &mut charstreamer_core::FitScratch::default(),
            )?;
            Ok((
                CompiledBinaryModel::BurnWindowLstm(Box::new(model)),
                ModelFitReport::BurnWindowLstm(report),
            ))
        }
        ModelSpec::NativeLogistic { fit } => {
            let (model, report) =
                LogisticModel::fit(dataset, fit, &mut charstreamer_core::FitScratch::default())?;
            Ok((
                CompiledBinaryModel::NativeLogistic(Box::new(model)),
                ModelFitReport::NativeLogistic(report),
            ))
        }
        ModelSpec::PythonSklearnRandomForest { fit } => {
            let (model, report) = PythonSklearnRandomForestModel::train(dataset, fit)?;
            Ok((
                CompiledBinaryModel::PythonSklearnRandomForest(Box::new(model)),
                ModelFitReport::PythonSklearnRandomForest(report),
            ))
        }
    }
}

#[derive(Debug)]
struct CorpusBatchDoc {
    text: String,
    gold_spans: Vec<charstreamer_core::ByteSpan>,
    candidates: Vec<charstreamer_core::BytePos>,
    score_start: usize,
    score_len: usize,
}

fn prepare_corpus_batch<S, K>(
    scanner: &S,
    kernel: &K,
    documents: &[charstreamer_core::AnnotatedDocument],
) -> Result<(FeatureMatrix<f32>, Vec<CorpusBatchDoc>), PipelineError>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
{
    let feature_dim = kernel.schema().total_dim();
    let mut all_features = FeatureMatrix::<f32> {
        rows: 0,
        cols: feature_dim,
        data: Vec::new(),
    };
    let mut candidates = charstreamer_core::CandidateBuffer::new();
    let mut doc_matrix = FeatureMatrix::<f32>::default();
    let mut feature_scratch = FeatureScratch::default();
    let mut docs = Vec::with_capacity(documents.len());

    for document in documents {
        let text = TextBytes::from_utf8(&document.text);
        scanner.scan_into(
            text,
            charstreamer_core::ScanRange::full(text),
            &mut candidates,
        );

        let score_start = all_features.rows;
        let score_len = candidates.len();
        if score_len > 0 {
            doc_matrix.resize_zeroed(score_len, feature_dim);
            kernel.extract_into(
                text,
                candidates.as_slice(),
                doc_matrix.as_view_mut(),
                &mut feature_scratch,
            )?;
            all_features.data.extend_from_slice(&doc_matrix.data);
            all_features.rows += score_len;
        }

        docs.push(CorpusBatchDoc {
            text: document.text.clone(),
            gold_spans: document.sentence_spans.clone(),
            candidates: candidates.positions().to_vec(),
            score_start,
            score_len,
        });
    }

    Ok((all_features, docs))
}

fn evaluate_corpus_batched<S, K, D>(
    scanner: &S,
    kernel: &K,
    model: &CompiledBinaryModel,
    decoder: &D,
    documents: &[charstreamer_core::AnnotatedDocument],
) -> Result<BinaryMetrics, PipelineError>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    D: Decoder<f32, charstreamer_core::ByteSpan>,
{
    let (features, docs) = prepare_corpus_batch(scanner, kernel, documents)?;
    let mut scores = vec![0.0_f32; features.rows];
    model.predict_into(features.as_view(), &mut scores)?;

    let mut predicted_spans = Vec::new();
    let mut true_positives = 0_usize;
    let mut false_positives = 0_usize;
    let mut false_negatives = 0_usize;

    for doc in docs {
        predicted_spans.clear();
        decoder.decode_into(
            TextBytes::from_utf8(&doc.text),
            charstreamer_core::CandidateSlice {
                data: &doc.candidates,
            },
            &scores[doc.score_start..doc.score_start + doc.score_len],
            &mut predicted_spans,
        )?;

        let gold = boundary_set_from_spans(&doc.gold_spans);
        let predicted = boundary_set_from_spans(&predicted_spans);
        true_positives += gold.intersection(&predicted).count();
        false_positives += predicted.difference(&gold).count();
        false_negatives += gold.difference(&predicted).count();
    }

    Ok(binary_metrics_from_counts(
        true_positives,
        false_positives,
        false_negatives,
        0,
    ))
}

fn benchmark_corpus_batched<S, K, D>(
    scanner: &S,
    kernel: &K,
    model: &CompiledBinaryModel,
    decoder: &D,
    documents: &[charstreamer_core::AnnotatedDocument],
    iterations: usize,
) -> Result<ThroughputReport, PipelineError>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    D: Decoder<f32, charstreamer_core::ByteSpan>,
{
    let iterations = iterations.max(1);
    let total_chars: usize = documents.iter().map(|document| document.text.len()).sum();
    let started = Instant::now();

    for _ in 0..iterations {
        let (features, docs) = prepare_corpus_batch(scanner, kernel, documents)?;
        let mut scores = vec![0.0_f32; features.rows];
        model.predict_into(features.as_view(), &mut scores)?;
        let mut predicted_spans = Vec::new();
        for doc in &docs {
            predicted_spans.clear();
            decoder.decode_into(
                TextBytes::from_utf8(&doc.text),
                charstreamer_core::CandidateSlice {
                    data: &doc.candidates,
                },
                &scores[doc.score_start..doc.score_start + doc.score_len],
                &mut predicted_spans,
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

fn load_documents(
    path: impl AsRef<Path>,
    format: &CorpusFormat,
    limit: Option<usize>,
) -> Result<Vec<charstreamer_core::AnnotatedDocument>, ExperimentError> {
    match format {
        CorpusFormat::Alea => Ok(load_alea_jsonl(path, limit)?),
        CorpusFormat::MultiLegal => Ok(load_multilegal_jsonl(path, limit)?),
    }
}

fn single_char_set(values: Vec<String>) -> std::collections::HashSet<char> {
    values
        .into_iter()
        .filter_map(|value| {
            let mut chars = value.chars();
            match (chars.next(), chars.next()) {
                (Some(ch), None) => Some(ch),
                _ => None,
            }
        })
        .collect()
}

fn feature_rows_as_vec(features: charstreamer_core::FeatureMatrixView<'_, f32>) -> Vec<Vec<f32>> {
    (0..features.rows)
        .map(|row| features.row(row).to_vec())
        .collect()
}

fn temp_model_path(prefix: &str, extension: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{now}.{extension}", std::process::id()))
}

fn run_python_json_script<T: for<'de> Deserialize<'de>>(
    script: &str,
    input: &serde_json::Value,
) -> Result<T, ExperimentError> {
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .expect("stdin should be available")
                .write_all(input.to_string().as_bytes())?;
            child.wait_with_output()
        })?;
    if !output.status.success() {
        return Err(ExperimentError::Python(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn binary_metrics_from_counts(
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

fn boundary_set_from_spans(
    spans: &[charstreamer_core::ByteSpan],
) -> std::collections::HashSet<usize> {
    let mut boundaries = std::collections::HashSet::with_capacity(spans.len().saturating_mul(2));
    for span in spans {
        boundaries.insert(span.start.as_usize());
        boundaries.insert(span.end.as_usize());
    }
    boundaries
}

fn char_positions_to_byte_offsets(
    text: &str,
    positions: &[usize],
) -> Result<Vec<usize>, ExperimentError> {
    let char_starts: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    let mut offsets = Vec::with_capacity(positions.len());
    for &position in positions {
        let Some(offset) = char_starts.get(position) else {
            return Err(ExperimentError::Unsupported(format!(
                "character position {position} is outside the text",
            )));
        };
        offsets.push(*offset);
    }
    Ok(offsets)
}

fn python_charboundary_feature_rows(
    spec: &ParityCheckSpec,
) -> Result<Vec<Vec<i32>>, ExperimentError> {
    let script = r#"
import importlib.util
import json
import pathlib
import sys
import types

constants_path = pathlib.Path(sys.argv[1])
charboundary_dir = constants_path.parent
payload = json.loads(sys.stdin.read())

pkg = types.ModuleType("charboundary")
pkg.__path__ = [str(charboundary_dir.parent)]
sys.modules["charboundary"] = pkg

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load("charboundary.constants", constants_path)
load("charboundary.encoders", charboundary_dir / "encoders.py")
features_module = load("charboundary.features", charboundary_dir / "features.py")

extractor = features_module.FeatureExtractor(use_numpy=False)
rows = extractor.get_char_features(
    payload["text"],
    left_window=payload["left_window"],
    right_window=payload["right_window"],
    positions=payload["char_positions"],
)
print(json.dumps(rows))
"#;

    let input = serde_json::json!({
        "text": spec.text,
        "char_positions": spec.char_positions,
        "left_window": spec.left_window,
        "right_window": spec.right_window,
    });

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&spec.constants_py_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .expect("stdin should be available")
                .write_all(input.to_string().as_bytes())?;
            child.wait_with_output()
        })?;

    if !output.status.success() {
        return Err(ExperimentError::Python(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[cfg(test)]
mod tests {
    use charstreamer_core::TrainingPositionPolicy;
    use charstreamer_kernels::{ByteClass, UnicodeCategory, UnicodeCategoryGroup};
    use std::path::Path;

    use super::{
        BoundaryExperimentSpec, CorpusFormat, EvaluationCorpusSpec, FeatureComponentSpec,
        ModelSpec, PythonSklearnRandomForestFitOptions, ScannerSpec, ThresholdPolicy,
    };

    #[test]
    fn experiment_spec_serializes() {
        let spec = BoundaryExperimentSpec {
            name: "test".to_string(),
            training_path: "train.jsonl".into(),
            training_format: CorpusFormat::Alea,
            training_limit: Some(100),
            split_numerator: 9,
            split_denominator: 10,
            scanner: ScannerSpec::ByteSet {
                bytes: b".?!".to_vec(),
            },
            features: vec![FeatureComponentSpec::EncodedByteWindow { left: 5, right: 3 }],
            model: ModelSpec::NativeLogistic {
                fit: charstreamer_models_native::LogisticFitOptions::default(),
            },
            dataset_options: charstreamer_core::BoundaryDatasetBuildOptions {
                negative_keep_rate: 0.1,
                seed: Some(7),
                position_policy: TrainingPositionPolicy::ScannedCandidatesOnly,
            },
            threshold_policy: ThresholdPolicy::TuneOnValidation,
            validation_throughput_iterations: 2,
            evaluation_corpora: vec![EvaluationCorpusSpec {
                name: "scotus".to_string(),
                path: "CD_scotus.jsonl".into(),
                format: CorpusFormat::MultiLegal,
                limit: Some(20),
                throughput_iterations: 1,
            }],
        };
        let json = serde_json::to_string(&spec).expect("spec should serialize");
        assert!(json.contains("\"name\":\"test\""));
    }

    #[test]
    fn python_random_forest_spec_serializes() {
        let spec = BoundaryExperimentSpec {
            name: "python_rf".to_string(),
            training_path: "train.jsonl".into(),
            training_format: CorpusFormat::Alea,
            training_limit: Some(10),
            split_numerator: 4,
            split_denominator: 5,
            scanner: ScannerSpec::CharBoundarySentenceCandidates {
                constants_py_path: "../charboundary/constants.py".into(),
            },
            features: vec![FeatureComponentSpec::CharBoundaryLegacy {
                left_window: 5,
                right_window: 3,
                constants_py_path: "../charboundary/constants.py".into(),
            }],
            model: ModelSpec::PythonSklearnRandomForest {
                fit: PythonSklearnRandomForestFitOptions {
                    n_estimators: 32,
                    max_depth: Some(16),
                    min_samples_split: 8,
                    min_samples_leaf: 4,
                    max_features: None,
                    class_weight: Some("balanced_subsample".to_string()),
                    n_jobs: -1,
                    random_state: Some(7),
                },
            },
            dataset_options: charstreamer_core::BoundaryDatasetBuildOptions {
                negative_keep_rate: 0.001,
                seed: Some(7),
                position_policy: TrainingPositionPolicy::AllUtf8ScalarPositions,
            },
            threshold_policy: ThresholdPolicy::TuneOnValidation,
            validation_throughput_iterations: 1,
            evaluation_corpora: Vec::new(),
        };
        let json = serde_json::to_string(&spec).expect("spec should serialize");
        assert!(json.contains("\"PythonSklearnRandomForest\""));
        assert!(json.contains("\"balanced_subsample\""));
    }

    #[test]
    fn count_feature_specs_serialize() {
        let specs = vec![
            FeatureComponentSpec::SelectedByteCounts {
                left: 24,
                right: 24,
                bytes: vec![b',', b';', b':'],
            },
            FeatureComponentSpec::ByteClassCounts {
                left: 16,
                right: 16,
                classes: vec![
                    ByteClass::AsciiUpper,
                    ByteClass::AsciiLower,
                    ByteClass::AsciiDigit,
                ],
            },
            FeatureComponentSpec::DirectionalByteClassCounts {
                left: 12,
                right: 12,
                classes: vec![ByteClass::AsciiUpper, ByteClass::AsciiWhitespace],
            },
            FeatureComponentSpec::DirectionalUnicodeCategoryCounts {
                left: 6,
                right: 6,
                categories: vec![UnicodeCategory::Lu, UnicodeCategory::Po],
            },
            FeatureComponentSpec::DirectionalUnicodeCategoryGroupCounts {
                left: 6,
                right: 6,
                groups: vec![UnicodeCategoryGroup::L, UnicodeCategoryGroup::P],
            },
            FeatureComponentSpec::LineByteCounts {
                bytes: vec![b'(', b')'],
            },
        ];
        let json = serde_json::to_string(&specs).expect("feature specs should serialize");
        assert!(json.contains("\"SelectedByteCounts\""));
        assert!(json.contains("\"ByteClassCounts\""));
        assert!(json.contains("\"DirectionalByteClassCounts\""));
        assert!(json.contains("\"DirectionalUnicodeCategoryCounts\""));
        assert!(json.contains("\"DirectionalUnicodeCategoryGroupCounts\""));
        assert!(json.contains("\"LineByteCounts\""));
    }

    #[test]
    fn resolving_relative_paths_updates_nested_specs() {
        let mut spec = BoundaryExperimentSpec {
            name: "resolve".to_string(),
            training_path: "data/train.jsonl".into(),
            training_format: CorpusFormat::Alea,
            training_limit: None,
            split_numerator: 1,
            split_denominator: 2,
            scanner: ScannerSpec::CharBoundarySentenceCandidates {
                constants_py_path: "../charboundary/constants.py".into(),
            },
            features: vec![FeatureComponentSpec::CharBoundaryLegacy {
                left_window: 5,
                right_window: 3,
                constants_py_path: "../charboundary/constants.py".into(),
            }],
            model: ModelSpec::NativeLogistic {
                fit: charstreamer_models_native::LogisticFitOptions::default(),
            },
            dataset_options: charstreamer_core::BoundaryDatasetBuildOptions::default(),
            threshold_policy: ThresholdPolicy::TuneOnValidation,
            validation_throughput_iterations: 1,
            evaluation_corpora: vec![EvaluationCorpusSpec {
                name: "eval".to_string(),
                path: "data/eval.jsonl".into(),
                format: CorpusFormat::MultiLegal,
                limit: Some(5),
                throughput_iterations: 1,
            }],
        };

        spec.resolve_relative_to(Path::new("/tmp/specs"));
        assert_eq!(spec.training_path, Path::new("/tmp/specs/data/train.jsonl"));
        match &spec.scanner {
            ScannerSpec::CharBoundarySentenceCandidates { constants_py_path } => {
                assert_eq!(
                    constants_py_path,
                    Path::new("/tmp/specs/../charboundary/constants.py")
                );
            }
            _ => panic!("expected charboundary scanner"),
        }
        match &spec.features[0] {
            FeatureComponentSpec::CharBoundaryLegacy {
                constants_py_path, ..
            } => {
                assert_eq!(
                    constants_py_path,
                    Path::new("/tmp/specs/../charboundary/constants.py")
                );
            }
            _ => panic!("expected legacy feature spec"),
        }
        assert_eq!(
            spec.evaluation_corpora[0].path,
            Path::new("/tmp/specs/data/eval.jsonl")
        );
    }
}
