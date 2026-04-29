use charstreamer_backend_burn::{BurnModelIoError, BurnMultiLabelMlpModel, BurnShallowMlpModel};
use charstreamer_core::{
    BatchPredictor, BytePos, ByteWindowSpec, CandidateBuffer, FeatureKernel, FeatureMatrix,
    FeatureScratch, TextBytes,
};
use charstreamer_kernels::{
    AsciiClassAppender, BoundaryShapeAppender, ByteClass, CompositeFeatureKernel,
    DirectionalByteClassCountAppender, DirectionalUnicodeCategoryGroupCountAppender,
    EncodedByteWindowAppender, LineByteCountAppender, LineByteNgramHashAppender,
    LineContextMetricsAppender, LineShapeMetricsAppender, UnicodeCategoryGroup,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

/// Semantic label emitted by trained CharStreamer models.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Paragraph,
    Metadata,
    Section,
    ListItem,
    Dialogue,
    Sentence,
}

impl Label {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paragraph => "paragraph",
            Self::Metadata => "metadata",
            Self::Section => "section",
            Self::ListItem => "list_item",
            Self::Dialogue => "dialogue",
            Self::Sentence => "sentence",
        }
    }
}

/// One half-open byte span in UTF-8 text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnotationSpan {
    pub label: Label,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

impl AnnotationSpan {
    #[must_use]
    pub fn new(label: Label, start: usize, end: usize, score: f32) -> Self {
        Self {
            label,
            start,
            end,
            score,
        }
    }
}

/// Full annotation result. Standoff spans are canonical; tagged text is a view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub spans: Vec<AnnotationSpan>,
    pub tagged: String,
}

/// Runtime configuration for model-backed segmentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmenterConfig {
    pub include_sentences: bool,
    pub min_span_bytes: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            include_sentences: true,
            min_span_bytes: 1,
        }
    }
}

const MODEL_FORMAT: &str = "charstreamer.model-bundle.v1";
const MODEL_NAME: &str = "charstreamer-default";
const BURN_SENTENCE_ENGINE: &str = "burn_shallow_mlp_sentence_v1";
const BURN_STRUCTURE_ENGINE: &str = "burn_multilabel_mlp_structure_v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BurnSentenceFeatureConfig {
    pub encoded_left: usize,
    pub encoded_right: usize,
    pub count_radius: usize,
    pub feature_dim: usize,
    pub hidden_dim: usize,
}

impl Default for BurnSentenceFeatureConfig {
    fn default() -> Self {
        Self {
            encoded_left: 15,
            encoded_right: 15,
            count_radius: 64,
            feature_dim: 0,
            hidden_dim: 256,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BurnStructureFeatureConfig {
    pub encoded_left: usize,
    pub encoded_right: usize,
    pub count_radius: usize,
    #[serde(default)]
    pub line_ngram_buckets: usize,
    #[serde(default = "default_line_ngram_min_n")]
    pub line_ngram_min_n: usize,
    #[serde(default = "default_line_ngram_max_n")]
    pub line_ngram_max_n: usize,
    #[serde(default)]
    pub line_context_metrics: bool,
    pub feature_dim: usize,
    pub hidden_dim1: usize,
    pub hidden_dim2: usize,
    pub output_dim: usize,
}

const fn default_line_ngram_min_n() -> usize {
    3
}

const fn default_line_ngram_max_n() -> usize {
    5
}

impl Default for BurnStructureFeatureConfig {
    fn default() -> Self {
        Self {
            encoded_left: 7,
            encoded_right: 7,
            count_radius: 32,
            line_ngram_buckets: 0,
            line_ngram_min_n: default_line_ngram_min_n(),
            line_ngram_max_n: default_line_ngram_max_n(),
            line_context_metrics: false,
            feature_dim: 0,
            hidden_dim1: 96,
            hidden_dim2: 48,
            output_dim: 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct StructureBundleManifest {
    engine: String,
    #[serde(default)]
    features: BurnStructureFeatureConfig,
    labels: Vec<Label>,
    #[serde(default)]
    thresholds: BTreeMap<String, f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelBundleManifest {
    format: String,
    name: String,
    engine: String,
    #[serde(default)]
    features: BurnSentenceFeatureConfig,
    #[serde(default)]
    thresholds: BTreeMap<String, f32>,
    #[serde(default)]
    structure: Option<StructureBundleManifest>,
    files: Vec<ModelBundleFile>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelBundleFile {
    path: String,
    #[serde(default)]
    role: Option<String>,
}

#[derive(Debug)]
pub struct ModelArtifactError {
    message: String,
}

impl ModelArtifactError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ModelArtifactError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelArtifactError {}

impl From<std::io::Error> for ModelArtifactError {
    fn from(error: std::io::Error) -> Self {
        Self::new(format!("model artifact I/O error: {error}"))
    }
}

impl From<serde_json::Error> for ModelArtifactError {
    fn from(error: serde_json::Error) -> Self {
        Self::new(format!("model artifact JSON error: {error}"))
    }
}

impl From<charstreamer_core::FeatureError> for ModelArtifactError {
    fn from(error: charstreamer_core::FeatureError) -> Self {
        Self::new(format!("model feature extraction error: {error}"))
    }
}

impl From<charstreamer_core::PredictError> for ModelArtifactError {
    fn from(error: charstreamer_core::PredictError) -> Self {
        Self::new(format!("model prediction error: {error}"))
    }
}

impl From<BurnModelIoError> for ModelArtifactError {
    fn from(error: BurnModelIoError) -> Self {
        Self::new(format!("burn model artifact error: {error}"))
    }
}

/// Model-backed sentence segmenter.
///
/// This type emits only labels produced by the loaded model. It does not
/// synthesize structural labels or silently fall back to rule code.
#[derive(Debug)]
pub struct BurnSentenceSegmenter {
    config: SegmenterConfig,
    model: BurnShallowMlpModel,
    feature_config: BurnSentenceFeatureConfig,
    threshold: f32,
    structure: Option<BurnStructureRuntime>,
}

#[derive(Debug)]
struct BurnStructureRuntime {
    model: BurnMultiLabelMlpModel,
    feature_config: BurnStructureFeatureConfig,
    labels: Vec<Label>,
    thresholds: BTreeMap<String, f32>,
}

impl BurnSentenceSegmenter {
    pub fn from_dir(
        path: impl AsRef<Path>,
        config: SegmenterConfig,
    ) -> Result<Self, ModelArtifactError> {
        let root = path.as_ref();
        let manifest_path = root.join("manifest.json");
        let manifest: ModelBundleManifest =
            serde_json::from_slice(&fs::read(&manifest_path).map_err(|error| {
                ModelArtifactError::new(format!(
                    "failed to read model manifest `{}`: {error}",
                    manifest_path.display()
                ))
            })?)?;
        validate_sentence_manifest(&manifest)?;

        let kernel = burn_sentence_kernel(&manifest.features);
        let expected_dim = kernel.schema().total_dim();
        if manifest.features.feature_dim != 0 && manifest.features.feature_dim != expected_dim {
            return Err(ModelArtifactError::new(format!(
                "model feature_dim {} does not match runtime feature_dim {}",
                manifest.features.feature_dim, expected_dim
            )));
        }

        let model_file = find_sentence_model_file(root, &manifest)?;
        let model = BurnShallowMlpModel::load_named_mpk(
            expected_dim,
            manifest.features.hidden_dim,
            model_file,
        )?;
        let threshold = *manifest.thresholds.get("sentence.end").unwrap_or(&0.5);
        let structure = match &manifest.structure {
            Some(structure_manifest) => {
                validate_structure_manifest(structure_manifest)?;
                let kernel = burn_structure_kernel(&structure_manifest.features);
                let expected_dim = kernel.schema().total_dim();
                if structure_manifest.features.feature_dim != 0
                    && structure_manifest.features.feature_dim != expected_dim
                {
                    return Err(ModelArtifactError::new(format!(
                        "structure feature_dim {} does not match runtime feature_dim {}",
                        structure_manifest.features.feature_dim, expected_dim
                    )));
                }
                let output_dim = structure_manifest.labels.len();
                if structure_manifest.features.output_dim != 0
                    && structure_manifest.features.output_dim != output_dim
                {
                    return Err(ModelArtifactError::new(format!(
                        "structure output_dim {} does not match labels length {}",
                        structure_manifest.features.output_dim, output_dim
                    )));
                }
                let model_file = find_model_file(root, &manifest, "semantic_structure")?;
                let model = BurnMultiLabelMlpModel::load_named_mpk(
                    expected_dim,
                    structure_manifest.features.hidden_dim1,
                    structure_manifest.features.hidden_dim2,
                    output_dim,
                    model_file,
                )?;
                Some(BurnStructureRuntime {
                    model,
                    feature_config: BurnStructureFeatureConfig {
                        feature_dim: expected_dim,
                        output_dim,
                        ..structure_manifest.features.clone()
                    },
                    labels: structure_manifest.labels.clone(),
                    thresholds: structure_manifest.thresholds.clone(),
                })
            }
            None => None,
        };

        Ok(Self {
            config,
            model,
            feature_config: BurnSentenceFeatureConfig {
                feature_dim: expected_dim,
                ..manifest.features
            },
            threshold,
            structure,
        })
    }

    pub fn annotate(&self, text: &str) -> Result<Annotation, ModelArtifactError> {
        let spans = self.spans(text)?;
        let tagged = render_spans(text, &spans);
        Ok(Annotation { spans, tagged })
    }

    pub fn spans(&self, text: &str) -> Result<Vec<AnnotationSpan>, ModelArtifactError> {
        let mut spans = Vec::new();
        if let Some(structure) = &self.structure {
            spans.extend(self.structure_spans(text, structure)?);
        }
        if self.config.include_sentences {
            spans.extend(self.sentence_spans(text)?);
        }
        Ok(normalize_spans(text, spans, self.config.min_span_bytes))
    }

    fn structure_spans(
        &self,
        text: &str,
        runtime: &BurnStructureRuntime,
    ) -> Result<Vec<AnnotationSpan>, ModelArtifactError> {
        let candidates = line_candidates(text);
        if candidates.is_empty() || runtime.labels.is_empty() {
            return Ok(Vec::new());
        }
        let kernel = burn_structure_kernel(&runtime.feature_config);
        let mut candidate_buffer = CandidateBuffer::new();
        for candidate in &candidates {
            candidate_buffer.push(BytePos::from_usize(candidate.feature_pos));
        }
        let mut features = FeatureMatrix::<f32>::default();
        features.resize_zeroed(candidate_buffer.len(), kernel.schema().total_dim());
        kernel.extract_into(
            TextBytes::from_utf8(text),
            candidate_buffer.as_slice(),
            features.as_view_mut(),
            &mut FeatureScratch::default(),
        )?;
        let output_dim = runtime.labels.len();
        let mut scores = vec![0.0_f32; candidate_buffer.len() * output_dim];
        runtime
            .model
            .predict_flat_into(features.as_view(), &mut scores)?;
        Ok(decode_structure_spans(
            &candidates,
            &scores,
            &runtime.labels,
            &runtime.thresholds,
        ))
    }

    fn sentence_spans(&self, text: &str) -> Result<Vec<AnnotationSpan>, ModelArtifactError> {
        let candidates = sentence_boundary_candidates(text);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let kernel = burn_sentence_kernel(&self.feature_config);
        let mut candidate_buffer = CandidateBuffer::new();
        for candidate in &candidates {
            candidate_buffer.push(BytePos::from_usize(candidate.feature_pos));
        }

        let mut features = FeatureMatrix::<f32>::default();
        features.resize_zeroed(candidate_buffer.len(), kernel.schema().total_dim());
        kernel.extract_into(
            TextBytes::from_utf8(text),
            candidate_buffer.as_slice(),
            features.as_view_mut(),
            &mut FeatureScratch::default(),
        )?;

        let mut scores = vec![0.0_f32; candidate_buffer.len()];
        self.model.predict_into(features.as_view(), &mut scores)?;

        let mut spans = Vec::new();
        let mut cursor = next_nonspace_position(text, 0, text.len()).unwrap_or(text.len());
        for (candidate, score) in candidates.iter().zip(scores.iter().copied()) {
            if score < self.threshold || candidate.break_end <= cursor {
                continue;
            }
            let end = previous_nonspace_end(text, cursor, candidate.break_end)
                .unwrap_or(candidate.break_end);
            if end > cursor {
                spans.push(AnnotationSpan::new(Label::Sentence, cursor, end, score));
            }
            cursor =
                next_nonspace_position(text, candidate.break_end, text.len()).unwrap_or(text.len());
        }

        Ok(spans)
    }
}

fn validate_sentence_manifest(manifest: &ModelBundleManifest) -> Result<(), ModelArtifactError> {
    if manifest.format != MODEL_FORMAT {
        return Err(ModelArtifactError::new(format!(
            "unsupported model format `{}`",
            manifest.format
        )));
    }
    if manifest.name != MODEL_NAME {
        return Err(ModelArtifactError::new(format!(
            "unsupported model name `{}`",
            manifest.name
        )));
    }
    if manifest.engine != BURN_SENTENCE_ENGINE {
        return Err(ModelArtifactError::new(format!(
            "unsupported model engine `{}`",
            manifest.engine
        )));
    }
    if manifest.features.hidden_dim == 0 {
        return Err(ModelArtifactError::new(
            "model manifest requires positive features.hidden_dim",
        ));
    }
    if manifest.files.is_empty() {
        return Err(ModelArtifactError::new(
            "model manifest requires at least one payload file",
        ));
    }
    Ok(())
}

fn validate_structure_manifest(
    manifest: &StructureBundleManifest,
) -> Result<(), ModelArtifactError> {
    if manifest.engine != BURN_STRUCTURE_ENGINE {
        return Err(ModelArtifactError::new(format!(
            "unsupported structure model engine `{}`",
            manifest.engine
        )));
    }
    if manifest.features.hidden_dim1 == 0 || manifest.features.hidden_dim2 == 0 {
        return Err(ModelArtifactError::new(
            "structure manifest requires positive hidden dimensions",
        ));
    }
    if manifest.labels.is_empty() {
        return Err(ModelArtifactError::new(
            "structure manifest requires at least one label",
        ));
    }
    if manifest.labels.contains(&Label::Sentence) {
        return Err(ModelArtifactError::new(
            "structure manifest must not include the sentence label",
        ));
    }
    Ok(())
}

fn find_sentence_model_file(
    root: &Path,
    manifest: &ModelBundleManifest,
) -> Result<PathBuf, ModelArtifactError> {
    find_model_file(root, manifest, "sentence_boundary")
}

fn find_model_file(
    root: &Path,
    manifest: &ModelBundleManifest,
    role: &str,
) -> Result<PathBuf, ModelArtifactError> {
    let file = manifest
        .files
        .iter()
        .find(|file| file.role.as_deref() == Some(role))
        .ok_or_else(|| {
            ModelArtifactError::new(format!(
                "model manifest does not include a `{role}` payload"
            ))
        })?;
    let path = Path::new(&file.path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(ModelArtifactError::new(format!(
            "unsafe model payload path `{}`",
            file.path
        )));
    }
    let path = root.join(path);
    if !path.is_file() {
        return Err(ModelArtifactError::new(format!(
            "model payload is missing: {}",
            path.display()
        )));
    }
    Ok(path)
}

pub fn burn_sentence_kernel(config: &BurnSentenceFeatureConfig) -> CompositeFeatureKernel {
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

pub fn burn_structure_kernel(config: &BurnStructureFeatureConfig) -> CompositeFeatureKernel {
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
            "line_byte_counts",
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SentenceBoundaryCandidate {
    pub feature_pos: usize,
    pub break_end: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LineCandidate {
    pub start: usize,
    pub end: usize,
    pub feature_pos: usize,
}

/// Returns every UTF-8 scalar end position as a model scoring candidate.
///
/// Candidate generation is exhaustive over valid scalar boundaries. Task
/// decisions belong to the model, not to rule code.
#[must_use]
pub fn sentence_boundary_candidates(text: &str) -> Vec<SentenceBoundaryCandidate> {
    text.char_indices()
        .map(|(offset, ch)| SentenceBoundaryCandidate {
            feature_pos: offset,
            break_end: offset + ch.len_utf8(),
        })
        .collect()
}

#[must_use]
pub fn line_candidates(text: &str) -> Vec<LineCandidate> {
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

fn decode_structure_spans(
    candidates: &[LineCandidate],
    scores: &[f32],
    labels: &[Label],
    thresholds: &BTreeMap<String, f32>,
) -> Vec<AnnotationSpan> {
    let mut spans = Vec::new();
    for (label_index, label) in labels.iter().copied().enumerate() {
        let threshold = *thresholds.get(label.as_str()).unwrap_or(&0.5);
        let mut current: Option<AnnotationSpan> = None;
        for (candidate_index, candidate) in candidates.iter().copied().enumerate() {
            let score = scores[candidate_index * labels.len() + label_index];
            if score < threshold {
                if let Some(span) = current.take() {
                    spans.push(span);
                }
                continue;
            }
            if let Some(span) = &mut current {
                if candidate.start <= span.end.saturating_add(2) {
                    span.end = span.end.max(candidate.end);
                    span.score = span.score.max(score);
                    continue;
                }
                spans.push(span.clone());
            }
            current = Some(AnnotationSpan::new(
                label,
                candidate.start,
                candidate.end,
                score,
            ));
        }
        if let Some(span) = current {
            spans.push(span);
        }
    }
    spans.sort_by_key(|span| (span.start, span.end, label_priority(span.label)));
    spans
}

/// Renders nested inline tags from standoff spans.
#[must_use]
pub fn render_spans(text: &str, spans: &[AnnotationSpan]) -> String {
    let mut valid = spans
        .iter()
        .filter(|span| {
            span.start < span.end
                && span.end <= text.len()
                && text.is_char_boundary(span.start)
                && text.is_char_boundary(span.end)
        })
        .cloned()
        .collect::<Vec<_>>();
    valid.sort_by_key(|span| (span.start, span.end, label_priority(span.label)));
    let valid = remove_crossing_spans(valid);
    let mut events = Vec::with_capacity(valid.len() * 2);
    for span in valid {
        events.push(RenderEvent {
            position: span.start,
            label: span.label,
            start: span.start,
            end: span.end,
            kind: RenderEventKind::Open,
        });
        events.push(RenderEvent {
            position: span.end,
            label: span.label,
            start: span.start,
            end: span.end,
            kind: RenderEventKind::Close,
        });
    }
    events.sort_by(render_event_order);

    let mut rendered = String::new();
    let mut cursor = 0_usize;
    for event in events {
        if event.position > cursor {
            rendered.push_str(&text[cursor..event.position]);
            cursor = event.position;
        }
        match event.kind {
            RenderEventKind::Close => {
                rendered.push_str("</|");
                rendered.push_str(event.label.as_str());
                rendered.push_str("|>");
            }
            RenderEventKind::Open => {
                rendered.push_str("<|");
                rendered.push_str(event.label.as_str());
                rendered.push_str("|>");
            }
        }
    }
    if cursor < text.len() {
        rendered.push_str(&text[cursor..]);
    }
    rendered
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RenderEventKind {
    Close,
    Open,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RenderEvent {
    position: usize,
    label: Label,
    start: usize,
    end: usize,
    kind: RenderEventKind,
}

fn render_event_order(left: &RenderEvent, right: &RenderEvent) -> std::cmp::Ordering {
    left.position
        .cmp(&right.position)
        .then_with(|| match (left.kind, right.kind) {
            (RenderEventKind::Close, RenderEventKind::Open) => std::cmp::Ordering::Less,
            (RenderEventKind::Open, RenderEventKind::Close) => std::cmp::Ordering::Greater,
            (RenderEventKind::Open, RenderEventKind::Open) => right
                .end
                .cmp(&left.end)
                .then_with(|| label_priority(left.label).cmp(&label_priority(right.label))),
            (RenderEventKind::Close, RenderEventKind::Close) => right
                .start
                .cmp(&left.start)
                .then_with(|| label_priority(right.label).cmp(&label_priority(left.label))),
        })
}

fn normalize_spans(
    text: &str,
    spans: Vec<AnnotationSpan>,
    min_span_bytes: usize,
) -> Vec<AnnotationSpan> {
    let mut spans = spans
        .into_iter()
        .filter(|span| {
            span.start < span.end
                && span.end <= text.len()
                && text.is_char_boundary(span.start)
                && text.is_char_boundary(span.end)
                && span.end.saturating_sub(span.start) >= min_span_bytes
        })
        .collect::<Vec<_>>();
    spans = remove_crossing_spans(spans);
    spans.sort_by_key(|span| (span.start, span.end, label_priority(span.label)));
    spans
}

fn remove_crossing_spans(spans: Vec<AnnotationSpan>) -> Vec<AnnotationSpan> {
    let mut ordered = spans;
    ordered.sort_by_key(|span| {
        (
            label_priority(span.label),
            span.start,
            std::cmp::Reverse(span.end),
        )
    });
    let mut accepted: Vec<AnnotationSpan> = Vec::with_capacity(ordered.len());
    for span in ordered {
        if !accepted
            .iter()
            .any(|prior| ranges_cross(span.start, span.end, prior.start, prior.end))
        {
            accepted.push(span);
        }
    }
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

fn label_priority(label: Label) -> usize {
    match label {
        Label::Paragraph => 0,
        Label::Metadata => 1,
        Label::Section => 2,
        Label::ListItem => 3,
        Label::Dialogue => 4,
        Label::Sentence => 5,
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

#[cfg(test)]
mod tests {
    use super::*;
    use charstreamer_backend_burn::{BurnShallowMlpFitOptions, BurnShallowMlpModel};
    use charstreamer_core::{DatasetView, FeatureMatrix, FitScratch, TrainablePredictor};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn render_standoff_spans_are_sorted_and_valid() {
        let text = "A. B.";
        let spans = vec![
            AnnotationSpan::new(Label::Sentence, 3, 5, 0.8),
            AnnotationSpan::new(Label::Sentence, 0, 2, 0.9),
        ];
        let rendered = render_spans(text, &spans);
        assert_eq!(
            rendered,
            "<|sentence|>A.</|sentence|> <|sentence|>B.</|sentence|>"
        );
    }

    #[test]
    fn render_same_start_spans_as_valid_nested_tags() {
        let text = "# Background";
        let spans = vec![
            AnnotationSpan::new(Label::Sentence, 0, text.len(), 0.9),
            AnnotationSpan::new(Label::Section, 0, text.len(), 0.8),
            AnnotationSpan::new(Label::Metadata, 0, text.len(), 0.7),
        ];
        let rendered = render_spans(text, &spans);
        assert_eq!(
            rendered,
            "<|metadata|><|section|><|sentence|># Background</|sentence|></|section|></|metadata|>"
        );
    }

    #[test]
    fn normalize_keeps_structure_when_sentence_crosses_it() {
        let text = "# Background\nThe court reviewed the invoice. The shipment was late.\n\n- Notice was timely.";
        let spans = vec![
            AnnotationSpan::new(Label::Sentence, 0, 44, 0.99),
            AnnotationSpan::new(Label::Section, 0, 12, 0.57),
            AnnotationSpan::new(Label::Paragraph, 13, text.len(), 0.88),
            AnnotationSpan::new(Label::ListItem, 69, text.len(), 0.39),
        ];
        let normalized = normalize_spans(text, spans, 1);
        let labels = normalized.iter().map(|span| span.label).collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![Label::Section, Label::Paragraph, Label::ListItem]
        );
    }

    #[test]
    fn sentence_candidates_are_exhaustive_utf8_scalar_ends() {
        let candidates = sentence_boundary_candidates("Aé.");
        let break_ends = candidates
            .iter()
            .map(|candidate| candidate.break_end)
            .collect::<Vec<_>>();
        assert_eq!(break_ends, vec![1, 3, 4]);
    }

    #[test]
    fn burn_sentence_bundle_loads_and_annotates() {
        let feature_config = BurnSentenceFeatureConfig {
            hidden_dim: 4,
            ..BurnSentenceFeatureConfig::default()
        };
        let kernel = burn_sentence_kernel(&feature_config);
        let feature_dim = kernel.schema().total_dim();
        let features = FeatureMatrix {
            rows: 4,
            cols: feature_dim,
            data: vec![0.0; 4 * feature_dim],
        };
        let labels = [1_u8, 1, 0, 0];
        let (model, _) = BurnShallowMlpModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &BurnShallowMlpFitOptions {
                hidden_dim: feature_config.hidden_dim,
                epochs: 1,
                batch_size: 2,
                learning_rate: 1.0e-3,
                seed: 3,
            },
            &mut FitScratch::default(),
        )
        .expect("toy Burn model should train");

        let model_dir = std::env::temp_dir().join(format!(
            "charstreamer-burn-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&model_dir).expect("temp model dir should be created");
        model
            .save_named_mpk(model_dir.join("sentence_boundary"))
            .expect("toy Burn model should serialize");

        let manifest = json!({
            "format": MODEL_FORMAT,
            "name": MODEL_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "engine": BURN_SENTENCE_ENGINE,
            "task": "sentence_boundary",
            "features": {
                "encoded_left": feature_config.encoded_left,
                "encoded_right": feature_config.encoded_right,
                "count_radius": feature_config.count_radius,
                "feature_dim": feature_dim,
                "hidden_dim": feature_config.hidden_dim
            },
            "thresholds": {
                "sentence.end": 0.0
            },
            "files": [
                {
                    "path": "sentence_boundary.mpk",
                    "role": "sentence_boundary"
                }
            ]
        });
        fs::write(
            model_dir.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should be written");

        let segmenter = BurnSentenceSegmenter::from_dir(&model_dir, SegmenterConfig::default())
            .expect("model bundle should load");
        let annotation = segmenter
            .annotate("One sentence. Another sentence.")
            .expect("loaded model should annotate");
        assert!(annotation.tagged.contains("<|sentence|>"));

        fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
    }
}
