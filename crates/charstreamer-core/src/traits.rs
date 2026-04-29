use crate::data::{
    CandidateBuffer, CandidateSlice, DatasetView, FeatureBlock, FeatureMatrixView,
    FeatureMatrixViewMut, FeatureSchema, FeatureScratch, FitScratch, ScanRange, ScoreMatrixView,
    ScoreMatrixViewMut,
};
use crate::error::{DecodeError, FeatureError, FitError, PredictError};
use crate::text::TextBytes;

/// Candidate scanner over canonical byte input.
pub trait CandidateScanner {
    fn scan_into(&self, text: TextBytes<'_>, range: ScanRange, out: &mut CandidateBuffer);
}

/// One reusable column block appender.
pub trait FeatureAppender<T> {
    fn block(&self) -> FeatureBlock;

    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        out: FeatureMatrixViewMut<'_, T>,
        scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError>;
}

/// Composite feature kernel over one schema.
pub trait FeatureKernel<T> {
    fn schema(&self) -> &FeatureSchema;

    fn extract_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        out: FeatureMatrixViewMut<'_, T>,
        scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError>;
}

/// Batch model over row-major feature views.
pub trait BatchPredictor<F, S> {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, F>,
        out: &mut [S],
    ) -> Result<(), PredictError>;
}

/// Batch scorer over row-major feature views producing per-position score rows.
pub trait BatchScorer<F, S> {
    fn score_dim(&self) -> usize;

    fn score_into(
        &self,
        features: FeatureMatrixView<'_, F>,
        out: ScoreMatrixViewMut<'_, S>,
    ) -> Result<(), PredictError>;
}

/// Native trainer over the same matrix contract used for inference.
pub trait TrainablePredictor<F, L> {
    type FitOptions;
    type FitReport;

    fn fit(
        dataset: DatasetView<'_, F, L>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError>
    where
        Self: Sized;
}

/// Decoder from candidate scores to task-level outputs.
pub trait Decoder<S, O> {
    fn decode_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        scores: &[S],
        out: &mut Vec<O>,
    ) -> Result<(), DecodeError>;
}

/// Decoder from score matrices to task-level outputs.
pub trait ScoreDecoder<S, O> {
    fn decode_scores_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        scores: ScoreMatrixView<'_, S>,
        out: &mut Vec<O>,
    ) -> Result<(), DecodeError>;
}
