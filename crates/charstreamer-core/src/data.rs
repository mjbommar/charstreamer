use crate::text::TextBytes;

/// Byte position in the canonical byte-oriented text space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BytePos(pub u32);

impl BytePos {
    /// Creates a byte position from a `usize`, panicking if it does not fit in `u32`.
    #[must_use]
    pub fn from_usize(value: usize) -> Self {
        Self(u32::try_from(value).expect("byte position must fit in u32"))
    }

    /// Returns the position as a `usize`.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Unicode scalar position for derived UTF-8 views.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScalarPos(pub u32);

/// Half-open byte span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteSpan {
    pub start: BytePos,
    pub end: BytePos,
}

impl ByteSpan {
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start: BytePos::from_usize(start),
            end: BytePos::from_usize(end),
        }
    }
}

/// Range scanned for candidate positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScanRange {
    pub start: BytePos,
    pub end: BytePos,
}

impl ScanRange {
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start: BytePos::from_usize(start),
            end: BytePos::from_usize(end),
        }
    }

    #[must_use]
    pub fn full(text: TextBytes<'_>) -> Self {
        Self::new(0, text.len())
    }
}

/// Chunk range with overlap metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkRange {
    pub start: BytePos,
    pub end: BytePos,
    pub left_overlap: u32,
    pub right_overlap: u32,
}

/// Owned output range after overlap handling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnedRange {
    pub start: BytePos,
    pub end: BytePos,
}

/// Fixed-width byte window specification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteWindowSpec {
    pub left: usize,
    pub right: usize,
}

impl ByteWindowSpec {
    #[must_use]
    pub fn new(left: usize, right: usize) -> Self {
        Self { left, right }
    }

    #[must_use]
    pub fn width(self) -> usize {
        self.left + self.right + 1
    }
}

/// Fixed-width scalar window specification for future Unicode-aware kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarWindowSpec {
    pub left: usize,
    pub right: usize,
}

/// Sparse window specification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrideWindowSpec {
    pub left: usize,
    pub right: usize,
    pub stride: usize,
}

/// Reusable candidate buffer.
#[derive(Clone, Debug, Default)]
pub struct CandidateBuffer {
    data: Vec<BytePos>,
}

impl CandidateBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn push(&mut self, position: BytePos) {
        self.data.push(position);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> CandidateSlice<'_> {
        CandidateSlice { data: &self.data }
    }

    #[must_use]
    pub fn positions(&self) -> &[BytePos] {
        &self.data
    }
}

/// Borrowed candidate slice.
#[derive(Clone, Copy, Debug)]
pub struct CandidateSlice<'a> {
    pub data: &'a [BytePos],
}

impl<'a> CandidateSlice<'a> {
    #[must_use]
    pub fn len(self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.data.is_empty()
    }
}

/// Semantic alias for more generic position-oriented pipelines.
pub type PositionBuffer = CandidateBuffer;

/// Semantic alias for more generic position-oriented pipelines.
pub type PositionSlice<'a> = CandidateSlice<'a>;

/// One stable block of columns in the feature matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureBlock {
    pub name: &'static str,
    pub offset: usize,
    pub width: usize,
}

impl FeatureBlock {
    #[must_use]
    pub fn new(name: &'static str, width: usize) -> Self {
        Self {
            name,
            offset: 0,
            width,
        }
    }

    #[must_use]
    pub fn with_offset(self, offset: usize) -> Self {
        Self { offset, ..self }
    }
}

/// Explicit feature layout shared by training and inference.
#[derive(Clone, Debug, Default)]
pub struct FeatureSchema {
    blocks: Vec<FeatureBlock>,
    total_dim: usize,
}

impl FeatureSchema {
    #[must_use]
    pub fn new(blocks: Vec<FeatureBlock>) -> Self {
        let total_dim = blocks
            .iter()
            .map(|block| block.offset + block.width)
            .max()
            .unwrap_or(0);
        Self { blocks, total_dim }
    }

    #[must_use]
    pub fn blocks(&self) -> &[FeatureBlock] {
        &self.blocks
    }

    #[must_use]
    pub fn total_dim(&self) -> usize {
        self.total_dim
    }

    #[must_use]
    pub fn block(&self, name: &str) -> Option<&FeatureBlock> {
        self.blocks.iter().find(|block| block.name == name)
    }
}

/// Owned feature matrix with row-major contiguous storage.
#[derive(Clone, Debug, Default)]
pub struct FeatureMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T> FeatureMatrix<T>
where
    T: Clone + Default,
{
    pub fn resize_zeroed(&mut self, rows: usize, cols: usize) {
        let required = rows.saturating_mul(cols);
        self.rows = rows;
        self.cols = cols;
        self.data.resize(required, T::default());
        self.data.fill(T::default());
    }
}

impl<T> FeatureMatrix<T> {
    #[must_use]
    pub fn as_view(&self) -> FeatureMatrixView<'_, T> {
        FeatureMatrixView {
            rows: self.rows,
            cols: self.cols,
            row_stride: self.cols,
            col_offset: 0,
            data: &self.data,
        }
    }

    #[must_use]
    pub fn as_view_mut(&mut self) -> FeatureMatrixViewMut<'_, T> {
        FeatureMatrixViewMut {
            rows: self.rows,
            cols: self.cols,
            row_stride: self.cols,
            col_offset: 0,
            data: &mut self.data,
        }
    }
}

/// Borrowed immutable matrix view.
#[derive(Clone, Copy, Debug)]
pub struct FeatureMatrixView<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_offset: usize,
    pub data: &'a [T],
}

impl<'a, T> FeatureMatrixView<'a, T> {
    #[must_use]
    pub fn row(self, row: usize) -> &'a [T] {
        let start = row * self.row_stride + self.col_offset;
        let end = start + self.cols;
        &self.data[start..end]
    }
}

/// Borrowed mutable matrix view.
#[derive(Debug)]
pub struct FeatureMatrixViewMut<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_offset: usize,
    pub data: &'a mut [T],
}

impl<'a, T> FeatureMatrixViewMut<'a, T> {
    #[must_use]
    pub fn reborrow(&mut self) -> FeatureMatrixViewMut<'_, T> {
        FeatureMatrixViewMut {
            rows: self.rows,
            cols: self.cols,
            row_stride: self.row_stride,
            col_offset: self.col_offset,
            data: &mut *self.data,
        }
    }

    #[must_use]
    pub fn subview(self, offset: usize, cols: usize) -> Self {
        assert!(
            offset + cols <= self.cols,
            "subview must stay inside the parent view"
        );
        Self {
            rows: self.rows,
            cols,
            row_stride: self.row_stride,
            col_offset: self.col_offset + offset,
            data: self.data,
        }
    }

    #[must_use]
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.row_stride + self.col_offset;
        let end = start + self.cols;
        &mut self.data[start..end]
    }
}

impl<'a, T> FeatureMatrixViewMut<'a, T>
where
    T: Clone,
{
    pub fn fill(&mut self, value: T) {
        for row in 0..self.rows {
            self.row_mut(row).fill(value.clone());
        }
    }
}

/// Borrowed mutable feature row.
#[derive(Debug)]
pub struct FeatureRowMut<'a, T> {
    pub data: &'a mut [T],
}

/// Semantic alias for score matrices.
pub type ScoreMatrix<T> = FeatureMatrix<T>;

/// Semantic alias for immutable score views.
pub type ScoreMatrixView<'a, T> = FeatureMatrixView<'a, T>;

/// Semantic alias for mutable score views.
pub type ScoreMatrixViewMut<'a, T> = FeatureMatrixViewMut<'a, T>;

/// Reusable score buffer.
#[derive(Clone, Debug, Default)]
pub struct ScoreBuffer<S> {
    pub data: Vec<S>,
}

impl<S> ScoreBuffer<S>
where
    S: Clone,
{
    pub fn resize_fill(&mut self, len: usize, value: S) {
        self.data.resize(len, value.clone());
        self.data.fill(value);
    }
}

/// Borrowed dataset view shared by native trainers and evaluators.
#[derive(Clone, Copy, Debug)]
pub struct DatasetView<'a, F, L> {
    pub features: FeatureMatrixView<'a, F>,
    pub labels: &'a [L],
}

/// Scratch storage shared by feature extractors.
#[derive(Clone, Debug, Default)]
pub struct FeatureScratch {
    pub bytes: Vec<u8>,
}

/// Reusable scratch for native training code.
#[derive(Clone, Debug, Default)]
pub struct FitScratch {
    pub floats: Vec<f32>,
    pub floats_aux: Vec<f32>,
    pub indices: Vec<usize>,
}

/// Reusable workspace for the narrow end-to-end slice.
#[derive(Clone, Debug, Default)]
pub struct PipelineWorkspace<F, S> {
    pub candidates: CandidateBuffer,
    pub features: FeatureMatrix<F>,
    pub scores: ScoreBuffer<S>,
    pub feature_scratch: FeatureScratch,
}

/// Workspace for score-matrix pipelines.
#[derive(Clone, Debug, Default)]
pub struct ScoringWorkspace<F, S> {
    pub positions: CandidateBuffer,
    pub features: FeatureMatrix<F>,
    pub scores: ScoreMatrix<S>,
    pub feature_scratch: FeatureScratch,
}

/// Decoded label at one scored position.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelAtPos<L> {
    pub position: BytePos,
    pub label: L,
    pub score: f32,
}

/// Decoded labeled span over byte offsets.
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledSpan<L> {
    pub span: ByteSpan,
    pub label: L,
    pub score: f32,
}
