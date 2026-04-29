use charstreamer_backend_burn::{BurnModelIoError, BurnShallowMlpModel};
use charstreamer_core::{
    BatchPredictor, BytePos, ByteWindowSpec, CandidateBuffer, FeatureKernel, FeatureMatrix,
    FeatureScratch, TextBytes,
};
use charstreamer_kernels::{
    AsciiClassAppender, BoundaryShapeAppender, ByteClass, CompositeFeatureKernel,
    DirectionalByteClassCountAppender, DirectionalUnicodeCategoryGroupCountAppender,
    EncodedByteWindowAppender, LegalBoundaryHeuristicAppender, LineByteCountAppender,
    UnicodeCategoryGroup,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

/// Semantic label emitted by the default combined segmenter.
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

/// Configuration for the default production segmenter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmenterConfig {
    pub include_paragraphs: bool,
    pub include_sentences: bool,
    pub include_metadata: bool,
    pub include_sections: bool,
    pub include_list_items: bool,
    pub include_dialogue: bool,
    pub suppress_sentences_in_structural_spans: bool,
    pub min_span_bytes: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            include_paragraphs: true,
            include_sentences: true,
            include_metadata: true,
            include_sections: true,
            include_list_items: true,
            include_dialogue: true,
            suppress_sentences_in_structural_spans: true,
            min_span_bytes: 1,
        }
    }
}

/// Production combined segmenter.
///
/// This is a deterministic, standoff-first implementation of the current
/// two-stage design: structural line/span detection, paragraph detection,
/// sentence-boundary detection, and one final merge/render pass. The trained
/// experiment models can replace the scoring functions behind the same span
/// contract once model serialization is added.
#[derive(Clone, Debug)]
pub struct CombinedSegmenter {
    config: SegmenterConfig,
}

impl Default for CombinedSegmenter {
    fn default() -> Self {
        Self::new(SegmenterConfig::default())
    }
}

impl CombinedSegmenter {
    #[must_use]
    pub fn new(config: SegmenterConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn config(&self) -> &SegmenterConfig {
        &self.config
    }

    /// Annotates UTF-8 text, returning canonical standoff spans plus tagged text.
    #[must_use]
    pub fn annotate(&self, text: &str) -> Annotation {
        let spans = self.spans(text);
        let tagged = render_spans(text, &spans);
        Annotation { spans, tagged }
    }

    /// Returns canonical standoff spans without rendering.
    #[must_use]
    pub fn spans(&self, text: &str) -> Vec<AnnotationSpan> {
        let mut spans = Vec::new();
        let structural = self.structural_spans(text);

        if self.config.include_paragraphs {
            spans.extend(
                paragraph_spans(text)
                    .into_iter()
                    .map(|(start, end)| AnnotationSpan::new(Label::Paragraph, start, end, 1.0)),
            );
        }

        spans.extend(structural.iter().cloned());

        if self.config.include_sentences {
            let suppress = if self.config.suppress_sentences_in_structural_spans {
                structural
                    .iter()
                    .filter(|span| {
                        matches!(
                            span.label,
                            Label::Metadata | Label::Section | Label::ListItem
                        )
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            spans.extend(
                sentence_spans(text, &suppress)
                    .into_iter()
                    .map(|(start, end)| AnnotationSpan::new(Label::Sentence, start, end, 1.0)),
            );
        }

        normalize_spans(text, spans, self.config.min_span_bytes)
    }

    fn structural_spans(&self, text: &str) -> Vec<AnnotationSpan> {
        let mut spans = Vec::new();
        let mut metadata_current: Option<AnnotationSpan> = None;
        let metadata_zone_end = first_blank_line_start(text).unwrap_or(0);

        for line in line_candidates(text) {
            let line_text = &text[line.start..line.end];
            let candidate = if self.config.include_metadata {
                metadata_score(line, line_text, metadata_zone_end)
                    .map(|score| AnnotationSpan::new(Label::Metadata, line.start, line.end, score))
            } else {
                None
            }
            .or_else(|| {
                self.config
                    .include_sections
                    .then(|| section_score(line_text))
                    .flatten()
                    .map(|score| AnnotationSpan::new(Label::Section, line.start, line.end, score))
            })
            .or_else(|| {
                self.config
                    .include_list_items
                    .then(|| list_item_score(line_text))
                    .flatten()
                    .map(|score| AnnotationSpan::new(Label::ListItem, line.start, line.end, score))
            })
            .or_else(|| {
                self.config
                    .include_dialogue
                    .then(|| dialogue_score(line_text))
                    .flatten()
                    .map(|score| AnnotationSpan::new(Label::Dialogue, line.start, line.end, score))
            });

            let Some(span) = candidate else {
                if let Some(current) = metadata_current.take() {
                    spans.push(current);
                }
                continue;
            };

            if span.label == Label::Metadata {
                if let Some(current) = &mut metadata_current
                    && span.start <= current.end.saturating_add(2)
                {
                    current.end = span.end;
                    current.score = current.score.max(span.score);
                    continue;
                }
                if let Some(current) = metadata_current.replace(span) {
                    spans.push(current);
                }
            } else {
                if let Some(current) = metadata_current.take() {
                    spans.push(current);
                }
                spans.push(span);
            }
        }

        if let Some(current) = metadata_current {
            spans.push(current);
        }

        resolve_overlapping_semantic_spans(spans)
    }
}

const MODEL_FORMAT: &str = "charstreamer.model-bundle.v1";
const MODEL_NAME: &str = "charstreamer-default";
const BURN_SENTENCE_ENGINE: &str = "burn_shallow_mlp_sentence_v1";

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
            encoded_left: 7,
            encoded_right: 7,
            count_radius: 24,
            feature_dim: 0,
            hidden_dim: 64,
        }
    }
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

/// Model-backed segmenter using a Burn sentence boundary model plus the native
/// structural span detector.
#[derive(Debug)]
pub struct BurnSentenceSegmenter {
    config: SegmenterConfig,
    model: BurnShallowMlpModel,
    feature_config: BurnSentenceFeatureConfig,
    threshold: f32,
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

        Ok(Self {
            config,
            model,
            feature_config: BurnSentenceFeatureConfig {
                feature_dim: expected_dim,
                ..manifest.features
            },
            threshold,
        })
    }

    pub fn annotate(&self, text: &str) -> Result<Annotation, ModelArtifactError> {
        let spans = self.spans(text)?;
        let tagged = render_spans(text, &spans);
        Ok(Annotation { spans, tagged })
    }

    pub fn spans(&self, text: &str) -> Result<Vec<AnnotationSpan>, ModelArtifactError> {
        let mut fallback_config = self.config.clone();
        fallback_config.include_sentences = false;
        let mut spans = CombinedSegmenter::new(fallback_config).spans(text);

        if self.config.include_sentences {
            let suppress = if self.config.suppress_sentences_in_structural_spans {
                spans
                    .iter()
                    .filter(|span| {
                        matches!(
                            span.label,
                            Label::Metadata | Label::Section | Label::ListItem
                        )
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            spans.extend(self.sentence_spans(text, &suppress)?);
        }

        Ok(normalize_spans(text, spans, self.config.min_span_bytes))
    }

    fn sentence_spans(
        &self,
        text: &str,
        suppress: &[AnnotationSpan],
    ) -> Result<Vec<AnnotationSpan>, ModelArtifactError> {
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
        let mut cursor = next_allowed_nonspace(text, 0, suppress);
        for (candidate, score) in candidates.iter().zip(scores.iter().copied()) {
            if score < self.threshold {
                continue;
            }
            let Some(start) = cursor else {
                break;
            };
            if candidate.break_end <= start {
                continue;
            }
            if let Some(overlap) = first_overlapping_span(start, candidate.break_end, suppress) {
                if overlap.start > start
                    && let Some(end) = previous_nonspace_end(text, start, overlap.start)
                    && end > start
                {
                    spans.push(AnnotationSpan::new(Label::Sentence, start, end, 1.0));
                }
                cursor = next_allowed_nonspace(text, overlap.end, suppress);
                continue;
            }
            spans.push(AnnotationSpan::new(
                Label::Sentence,
                start,
                candidate.break_end,
                score,
            ));
            cursor = next_allowed_nonspace(text, candidate.break_end, suppress);
        }

        if let Some(start) = cursor
            && start < text.len()
            && !overlaps_any(start, text.len(), suppress)
        {
            spans.push(AnnotationSpan::new(Label::Sentence, start, text.len(), 1.0));
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

fn find_sentence_model_file(
    root: &Path,
    manifest: &ModelBundleManifest,
) -> Result<PathBuf, ModelArtifactError> {
    let file = manifest
        .files
        .iter()
        .find(|file| file.role.as_deref() == Some("sentence_boundary"))
        .or_else(|| {
            manifest
                .files
                .iter()
                .find(|file| file.path.ends_with(".mpk"))
        })
        .ok_or_else(|| {
            ModelArtifactError::new(
                "model manifest does not include a sentence boundary `.mpk` payload",
            )
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LineCandidate {
    start: usize,
    end: usize,
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
    spans.sort_by_key(|span| (span.start, span.end, label_priority(span.label)));
    remove_crossing_spans(spans)
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
        start: line_start_with_prefix(text, trimmed_start),
        end: trimmed_end,
    });
}

fn metadata_score(line: LineCandidate, line_text: &str, metadata_zone_end: usize) -> Option<f32> {
    if line.start > metadata_zone_end {
        return None;
    }
    if has_metadata_colon(line_text) {
        return Some(0.96);
    }
    let lowered = line_text.trim_start().to_ascii_lowercase();
    (lowered.starts_with("case ")
        || lowered.starts_with("case:")
        || lowered.starts_with("docket")
        || lowered.starts_with("date:")
        || lowered.starts_with("no."))
    .then_some(0.92)
}

fn section_score(line: &str) -> Option<f32> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        return Some(0.98);
    }
    (trimmed.len() <= 80
        && !trimmed.ends_with('.')
        && trimmed.chars().any(|ch| ch.is_ascii_alphabetic())
        && trimmed
            .chars()
            .filter(|ch| ch.is_ascii_alphabetic())
            .all(|ch| ch.is_ascii_uppercase()))
    .then_some(0.86)
}

fn list_item_score(line: &str) -> Option<f32> {
    let trimmed = line.trim_start();
    if trimmed.starts_with(['-', '*', '•']) {
        return Some(0.98);
    }
    let mut chars = trimmed.chars().peekable();
    let mut digits = 0_usize;
    while chars.peek().is_some_and(|ch| ch.is_ascii_digit()) {
        digits += 1;
        chars.next();
    }
    (digits > 0 && matches!(chars.next(), Some('.' | ')'))).then_some(0.94)
}

fn dialogue_score(line: &str) -> Option<f32> {
    let trimmed = line.trim_start();
    if trimmed.starts_with(['"', '\'', '“', '‘']) {
        return Some(0.91);
    }
    let quote_count = trimmed
        .chars()
        .filter(|&ch| matches!(ch, '"' | '“' | '”'))
        .count();
    (quote_count >= 2).then_some(0.82)
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

fn sentence_spans(text: &str, suppress: &[AnnotationSpan]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut cursor = next_allowed_nonspace(text, 0, suppress);
    for break_end in sentence_break_candidates(text) {
        let Some(start) = cursor else {
            break;
        };
        if break_end <= start {
            continue;
        }
        if let Some(overlap) = first_overlapping_span(start, break_end, suppress) {
            if overlap.start > start
                && let Some(end) = previous_nonspace_end(text, start, overlap.start)
                && end > start
            {
                spans.push((start, end));
            }
            cursor = next_allowed_nonspace(text, overlap.end, suppress);
            continue;
        }
        spans.push((start, break_end));
        cursor = next_allowed_nonspace(text, break_end, suppress);
    }
    if let Some(start) = cursor
        && start < text.len()
        && !overlaps_any(start, text.len(), suppress)
    {
        spans.push((start, text.len()));
    }
    spans
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SentenceBoundaryCandidate {
    pub feature_pos: usize,
    pub break_end: usize,
}

pub fn sentence_boundary_candidates(text: &str) -> Vec<SentenceBoundaryCandidate> {
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
        let Some(next_start) = next_nonspace_position(text, break_end, text.len()) else {
            candidates.push(SentenceBoundaryCandidate {
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
            candidates.push(SentenceBoundaryCandidate {
                feature_pos: offset,
                break_end,
            });
        }
    }
    candidates
}

fn sentence_break_candidates(text: &str) -> Vec<usize> {
    sentence_boundary_candidates(text)
        .into_iter()
        .map(|candidate| candidate.break_end)
        .collect()
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
            (RenderEventKind::Open, RenderEventKind::Open) => label_priority(left.label)
                .cmp(&label_priority(right.label))
                .then_with(|| right.end.cmp(&left.end)),
            (RenderEventKind::Close, RenderEventKind::Close) => label_priority(right.label)
                .cmp(&label_priority(left.label))
                .then_with(|| right.start.cmp(&left.start)),
        })
}

fn resolve_overlapping_semantic_spans(mut spans: Vec<AnnotationSpan>) -> Vec<AnnotationSpan> {
    spans.sort_by_key(|span| (span.start, label_priority(span.label), span.end));
    let mut resolved: Vec<AnnotationSpan> = Vec::with_capacity(spans.len());
    let mut active_indices: Vec<usize> = Vec::new();
    for span in spans {
        active_indices.retain(|&index| resolved[index].end > span.start);
        if let Some(existing) = active_indices.iter().copied().find(|&index| {
            let prior = &resolved[index];
            ranges_overlap(span.start, span.end, prior.start, prior.end)
        }) {
            let prior = &resolved[existing];
            if span.score > prior.score + 0.05
                || ((span.score - prior.score).abs() <= 0.05
                    && label_priority(span.label) < label_priority(prior.label))
            {
                resolved[existing] = span;
            }
            continue;
        }
        active_indices.push(resolved.len());
        resolved.push(span);
    }
    resolved.sort_by_key(|span| (span.start, span.end));
    resolved
}

fn remove_crossing_spans(spans: Vec<AnnotationSpan>) -> Vec<AnnotationSpan> {
    let mut accepted = Vec::with_capacity(spans.len());
    let mut active: Vec<AnnotationSpan> = Vec::new();
    for span in spans {
        active.retain(|prior| prior.end > span.start);
        let crosses = active.iter().any(|prior| {
            prior.start < span.start && span.start < prior.end && prior.end < span.end
        });
        if !crosses {
            active.push(span.clone());
            accepted.push(span);
        }
    }
    accepted
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

fn line_start_with_prefix(text: &str, position: usize) -> usize {
    text[..position.min(text.len())]
        .rfind(['\n', '\r'])
        .map_or(0, |index| index + 1)
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

fn next_allowed_nonspace(text: &str, offset: usize, suppress: &[AnnotationSpan]) -> Option<usize> {
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

fn overlaps_any(start: usize, end: usize, spans: &[AnnotationSpan]) -> bool {
    spans
        .iter()
        .any(|span| ranges_overlap(start, end, span.start, span.end))
}

fn first_overlapping_span(
    start: usize,
    end: usize,
    spans: &[AnnotationSpan],
) -> Option<&AnnotationSpan> {
    spans
        .iter()
        .filter(|span| ranges_overlap(start, end, span.start, span.end))
        .min_by_key(|span| (span.start, span.end))
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

fn first_blank_line_start(text: &str) -> Option<usize> {
    match (text.find("\n\n"), text.find("\r\n\r\n")) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use charstreamer_backend_burn::{BurnShallowMlpFitOptions, BurnShallowMlpModel};
    use charstreamer_core::{DatasetView, FeatureMatrix, FitScratch, TrainablePredictor};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn combined_segmenter_merges_structural_and_sentence_spans() {
        let text = "Case: X\nDocket: 1\n\n# Facts\nOne. Two.\n\n- Item one.\n\n\"Hello.\"";
        let annotation = CombinedSegmenter::default().annotate(text);
        assert!(
            annotation
                .tagged
                .contains("<|metadata|>Case: X\nDocket: 1</|metadata|>")
        );
        assert!(annotation.tagged.contains("<|section|># Facts</|section|>"));
        assert!(annotation.tagged.contains("<|sentence|>One.</|sentence|>"));
        assert!(
            annotation
                .tagged
                .contains("<|list_item|>- Item one.</|list_item|>")
        );
        assert!(
            annotation
                .tagged
                .contains("<|dialogue|><|sentence|>\"Hello.\"</|sentence|></|dialogue|>")
        );
    }

    #[test]
    fn standoff_spans_are_sorted_and_valid() {
        let text = "A. B.\n\nC.";
        let spans = CombinedSegmenter::default().spans(text);
        assert!(spans.windows(2).all(|pair| {
            (pair[0].start, pair[0].end, label_priority(pair[0].label))
                <= (pair[1].start, pair[1].end, label_priority(pair[1].label))
        }));
        assert!(spans.iter().all(|span| {
            span.start < span.end
                && text.is_char_boundary(span.start)
                && text.is_char_boundary(span.end)
        }));
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
