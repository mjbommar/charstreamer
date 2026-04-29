use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::data::{BytePos, ByteSpan, CandidateBuffer, DatasetView, FeatureMatrix};
use crate::error::PipelineError;
use crate::text::TextBytes;
use crate::traits::{CandidateScanner, FeatureKernel};

const SENTENCE_TAG: &str = "<|sentence|>";
const PARAGRAPH_TAG: &str = "<|paragraph|>";

#[derive(Debug)]
pub enum CorpusError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidRecord(String),
}

impl Display for CorpusError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error while loading corpus: {error}"),
            Self::Json(error) => write!(f, "JSON decode error while loading corpus: {error}"),
            Self::InvalidRecord(message) => f.write_str(message),
        }
    }
}

impl Error for CorpusError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::InvalidRecord(_) => None,
        }
    }
}

impl From<std::io::Error> for CorpusError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CorpusError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// One annotated text plus sentence spans in canonical byte offsets.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnotatedDocument {
    pub text: String,
    pub sentence_spans: Vec<ByteSpan>,
    pub boundary_markers: Vec<BytePos>,
}

impl AnnotatedDocument {
    #[must_use]
    pub fn boundary_positions(&self) -> Vec<BytePos> {
        if !self.boundary_markers.is_empty() {
            return self.boundary_markers.clone();
        }

        let mut positions = Vec::with_capacity(self.sentence_spans.len());
        for span in &self.sentence_spans {
            if span.end.as_usize() > span.start.as_usize() {
                positions.push(BytePos::from_usize(span.end.as_usize() - 1));
            }
        }
        positions
    }
}

/// Dense feature matrix plus binary labels built from candidate positions.
#[derive(Clone, Debug, Default)]
pub struct BoundaryDataset {
    pub features: FeatureMatrix<f32>,
    pub labels: Vec<u8>,
    pub positives: usize,
    pub negatives: usize,
}

impl BoundaryDataset {
    #[must_use]
    pub fn rows(&self) -> usize {
        self.labels.len()
    }

    #[must_use]
    pub fn as_view(&self) -> DatasetView<'_, f32, u8> {
        DatasetView {
            features: self.features.as_view(),
            labels: &self.labels,
        }
    }
}

/// Sampling knobs for candidate-based training datasets.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrainingPositionPolicy {
    ScannedCandidatesOnly,
    AllUtf8ScalarPositions,
}

/// Sampling knobs for candidate-based training datasets.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BoundaryDatasetBuildOptions {
    pub negative_keep_rate: f32,
    pub seed: Option<u64>,
    pub position_policy: TrainingPositionPolicy,
}

impl Default for BoundaryDatasetBuildOptions {
    fn default() -> Self {
        Self {
            negative_keep_rate: 1.0,
            seed: Some(7),
            position_policy: TrainingPositionPolicy::ScannedCandidatesOnly,
        }
    }
}

#[derive(Deserialize)]
struct AleaRecord {
    text: String,
}

#[derive(Deserialize)]
struct MultiLegalRecord {
    text: String,
    spans: Vec<MultiLegalSpan>,
}

#[derive(Clone, Deserialize)]
struct MultiLegalSpan {
    start: usize,
    end: usize,
    label: String,
}

pub fn load_alea_jsonl(
    path: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<AnnotatedDocument>, CorpusError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    for line in reader.lines() {
        if limit.is_some_and(|max| documents.len() >= max) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: AleaRecord = serde_json::from_str(&line)?;
        documents.push(parse_alea_annotated_text(&record.text));
    }

    Ok(documents)
}

pub fn load_multilegal_jsonl(
    path: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<AnnotatedDocument>, CorpusError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    for line in reader.lines() {
        if limit.is_some_and(|max| documents.len() >= max) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: MultiLegalRecord = serde_json::from_str(&line)?;
        documents.push(parse_multilegal_record(record)?);
    }

    Ok(documents)
}

#[must_use]
pub fn split_documents(
    mut documents: Vec<AnnotatedDocument>,
    numerator: usize,
    denominator: usize,
) -> (Vec<AnnotatedDocument>, Vec<AnnotatedDocument>) {
    assert!(
        denominator > 0,
        "split denominator must be greater than zero"
    );
    assert!(
        numerator <= denominator,
        "split numerator must not exceed the denominator",
    );

    let split_at = documents.len().saturating_mul(numerator) / denominator;
    let validation = documents.split_off(split_at);
    (documents, validation)
}

pub fn build_boundary_dataset<S, K>(
    documents: &[AnnotatedDocument],
    scanner: &S,
    kernel: &K,
    options: &BoundaryDatasetBuildOptions,
) -> Result<BoundaryDataset, PipelineError>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
{
    let mut rng = match options.seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_rng(&mut rand::rng()),
    };
    let mut candidates = CandidateBuffer::new();
    let mut all_positions = CandidateBuffer::new();
    let mut selected = CandidateBuffer::new();
    let mut doc_matrix = FeatureMatrix::<f32>::default();
    let mut feature_scratch = crate::data::FeatureScratch::default();
    let feature_dim = kernel.schema().total_dim();

    let mut dataset = BoundaryDataset {
        features: FeatureMatrix {
            rows: 0,
            cols: feature_dim,
            data: Vec::new(),
        },
        labels: Vec::new(),
        positives: 0,
        negatives: 0,
    };

    for document in documents {
        let text = TextBytes::from_utf8(&document.text);
        scanner.scan_into(text, crate::data::ScanRange::full(text), &mut candidates);
        if candidates.is_empty()
            && options.position_policy == TrainingPositionPolicy::ScannedCandidatesOnly
        {
            continue;
        }

        let positives: HashSet<usize> = document
            .boundary_positions()
            .into_iter()
            .map(BytePos::as_usize)
            .collect();

        let candidate_positions: HashSet<usize> = candidates
            .positions()
            .iter()
            .map(|position| position.as_usize())
            .collect();

        all_positions.clear();
        selected.clear();
        let labels_start = dataset.labels.len();

        match options.position_policy {
            TrainingPositionPolicy::ScannedCandidatesOnly => {
                for &position in candidates.positions() {
                    let is_positive = positives.contains(&position.as_usize());
                    let is_sampled = rng.random::<f32>() <= options.negative_keep_rate;
                    if is_positive || is_sampled {
                        selected.push(position);
                        dataset.labels.push(u8::from(is_positive));
                        if is_positive {
                            dataset.positives += 1;
                        } else {
                            dataset.negatives += 1;
                        }
                    }
                }
            }
            TrainingPositionPolicy::AllUtf8ScalarPositions => {
                for (byte_index, _) in document.text.char_indices() {
                    all_positions.push(BytePos::from_usize(byte_index));
                }
                for &position in all_positions.positions() {
                    let offset = position.as_usize();
                    let is_positive = positives.contains(&offset);
                    let is_candidate = candidate_positions.contains(&offset);
                    let is_sampled = rng.random::<f32>() <= options.negative_keep_rate;
                    if is_positive || is_candidate || is_sampled {
                        selected.push(position);
                        dataset.labels.push(u8::from(is_positive));
                        if is_positive {
                            dataset.positives += 1;
                        } else {
                            dataset.negatives += 1;
                        }
                    }
                }
            }
        }

        if selected.is_empty() {
            dataset.labels.truncate(labels_start);
            continue;
        }

        doc_matrix.resize_zeroed(selected.len(), feature_dim);
        kernel.extract_into(
            text,
            selected.as_slice(),
            doc_matrix.as_view_mut(),
            &mut feature_scratch,
        )?;
        dataset.features.data.extend_from_slice(&doc_matrix.data);
        dataset.features.rows += doc_matrix.rows;
    }

    Ok(dataset)
}

fn parse_alea_annotated_text(annotated: &str) -> AnnotatedDocument {
    let mut text = String::with_capacity(annotated.len());
    let mut sentence_spans = Vec::new();
    let mut boundary_markers = Vec::new();
    let mut sentence_start = 0_usize;
    let mut offset = 0_usize;

    while offset < annotated.len() {
        let rest = &annotated[offset..];
        if rest.starts_with(SENTENCE_TAG) {
            let end = text.len();
            if end > sentence_start {
                sentence_spans.push(ByteSpan::new(sentence_start, end));
                boundary_markers.push(BytePos::from_usize(end - 1));
                sentence_start = end;
            }
            offset += SENTENCE_TAG.len();
            continue;
        }
        if rest.starts_with(PARAGRAPH_TAG) {
            let end = text.len();
            if end > sentence_start {
                sentence_spans.push(ByteSpan::new(sentence_start, end));
                boundary_markers.push(BytePos::from_usize(end - 1));
                sentence_start = end;
            }
            offset += PARAGRAPH_TAG.len();
            continue;
        }

        let ch = rest
            .chars()
            .next()
            .expect("remaining slice must contain one UTF-8 scalar");
        text.push(ch);
        offset += ch.len_utf8();
    }

    if text.len() > sentence_start {
        sentence_spans.push(ByteSpan::new(sentence_start, text.len()));
    }

    AnnotatedDocument {
        text,
        sentence_spans,
        boundary_markers,
    }
}

fn parse_multilegal_record(record: MultiLegalRecord) -> Result<AnnotatedDocument, CorpusError> {
    let offsets = scalar_to_byte_offsets(&record.text);
    let mut sentence_spans = Vec::new();

    for span in record.spans {
        if span.label != "Sentence" {
            continue;
        }
        let start = offsets.get(span.start).copied().ok_or_else(|| {
            CorpusError::InvalidRecord(format!(
                "span start {} exceeds the number of UTF-8 scalars in a MultiLegal example",
                span.start
            ))
        })?;
        let end = offsets.get(span.end).copied().ok_or_else(|| {
            CorpusError::InvalidRecord(format!(
                "span end {} exceeds the number of UTF-8 scalars in a MultiLegal example",
                span.end
            ))
        })?;
        if end > start {
            sentence_spans.push(ByteSpan::new(start, end));
        }
    }

    sentence_spans.sort_by_key(|span| span.start);
    let boundary_markers = sentence_spans
        .iter()
        .filter(|span| span.end.as_usize() > span.start.as_usize())
        .map(|span| BytePos::from_usize(span.end.as_usize() - 1))
        .collect();

    Ok(AnnotatedDocument {
        text: record.text,
        sentence_spans,
        boundary_markers,
    })
}

fn scalar_to_byte_offsets(text: &str) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(text.chars().count() + 1);
    offsets.push(0);
    for (byte_offset, _) in text.char_indices().skip(1) {
        offsets.push(byte_offset);
    }
    if !text.is_empty() {
        offsets.push(text.len());
    }
    offsets
}

#[cfg(test)]
mod tests {
    use crate::corpus::{load_alea_jsonl, parse_alea_annotated_text, split_documents};

    #[test]
    fn parse_alea_text_into_clean_text_and_spans() {
        let document =
            parse_alea_annotated_text("One.<|sentence|> Two.<|sentence|><|paragraph|>Three.");
        assert_eq!(document.text, "One. Two.Three.");
        let spans: Vec<&str> = document
            .sentence_spans
            .iter()
            .map(|span| &document.text[span.start.as_usize()..span.end.as_usize()])
            .collect();
        assert_eq!(spans, vec!["One.", " Two.", "Three."]);
    }

    #[test]
    fn split_documents_preserves_order() {
        let documents = vec![
            parse_alea_annotated_text("A.<|sentence|>"),
            parse_alea_annotated_text("B.<|sentence|>"),
            parse_alea_annotated_text("C.<|sentence|>"),
            parse_alea_annotated_text("D.<|sentence|>"),
        ];
        let (train, valid) = split_documents(documents, 3, 4);
        assert_eq!(train.len(), 3);
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0].text, "D.");
    }

    #[test]
    fn loads_real_alea_jsonl_example() {
        let path = "/home/mjbommar/projects/personal/legal-sentence-paper/data/alea-legal-benchmark/train.jsonl";
        let documents = load_alea_jsonl(path, Some(1)).expect("ALEA dataset should load");
        assert_eq!(documents.len(), 1);
        assert!(!documents[0].text.is_empty());
        assert!(!documents[0].sentence_spans.is_empty());
    }
}
