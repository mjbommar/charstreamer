use crate::data::{
    ByteSpan, CandidateBuffer, FeatureMatrix, PipelineWorkspace, ScanRange, ScoringWorkspace,
};
use crate::error::PipelineError;
use crate::text::TextBytes;
use crate::traits::{
    BatchPredictor, BatchScorer, CandidateScanner, Decoder, FeatureKernel, ScoreDecoder,
};

/// Primitive-first pipeline that wires scanner, features, model, and decoder together.
#[derive(Clone, Debug)]
pub struct Pipeline<S, K, M, D> {
    scanner: S,
    kernel: K,
    model: M,
    decoder: D,
}

impl<S, K, M, D> Pipeline<S, K, M, D> {
    #[must_use]
    pub fn new(scanner: S, kernel: K, model: M, decoder: D) -> Self {
        Self {
            scanner,
            kernel,
            model,
            decoder,
        }
    }

    #[must_use]
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    #[must_use]
    pub fn scanner(&self) -> &S {
        &self.scanner
    }

    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    #[must_use]
    pub fn decoder(&self) -> &D {
        &self.decoder
    }
}

impl<S, K, M, D> Pipeline<S, K, M, D>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    M: BatchPredictor<f32, f32>,
    D: Decoder<f32, ByteSpan>,
{
    pub fn scan_candidates(&self, text: TextBytes<'_>, out: &mut CandidateBuffer) {
        self.scanner.scan_into(text, ScanRange::full(text), out);
    }

    pub fn extract_features(
        &self,
        text: TextBytes<'_>,
        candidates: &CandidateBuffer,
        features: &mut FeatureMatrix<f32>,
        workspace: &mut PipelineWorkspace<f32, f32>,
    ) -> Result<(), PipelineError> {
        features.resize_zeroed(candidates.len(), self.kernel.schema().total_dim());
        self.kernel.extract_into(
            text,
            candidates.as_slice(),
            features.as_view_mut(),
            &mut workspace.feature_scratch,
        )?;
        Ok(())
    }

    pub fn predict_scores(
        &self,
        features: &FeatureMatrix<f32>,
        workspace: &mut PipelineWorkspace<f32, f32>,
    ) -> Result<(), PipelineError> {
        workspace.scores.resize_fill(features.rows, 0.0);
        self.model
            .predict_into(features.as_view(), &mut workspace.scores.data)?;
        Ok(())
    }

    pub fn decode(
        &self,
        text: TextBytes<'_>,
        candidates: &CandidateBuffer,
        scores: &[f32],
        out: &mut Vec<ByteSpan>,
    ) -> Result<(), PipelineError> {
        self.decoder
            .decode_into(text, candidates.as_slice(), scores, out)?;
        Ok(())
    }

    pub fn run_into(
        &self,
        text: TextBytes<'_>,
        workspace: &mut PipelineWorkspace<f32, f32>,
        out: &mut Vec<ByteSpan>,
    ) -> Result<(), PipelineError> {
        self.scan_candidates(text, &mut workspace.candidates);
        workspace
            .features
            .resize_zeroed(workspace.candidates.len(), self.kernel.schema().total_dim());
        self.kernel.extract_into(
            text,
            workspace.candidates.as_slice(),
            workspace.features.as_view_mut(),
            &mut workspace.feature_scratch,
        )?;
        workspace.scores.resize_fill(workspace.features.rows, 0.0);
        self.model
            .predict_into(workspace.features.as_view(), &mut workspace.scores.data)?;
        self.decoder.decode_into(
            text,
            workspace.candidates.as_slice(),
            &workspace.scores.data,
            out,
        )?;
        Ok(())
    }

    pub fn run(&self, text: TextBytes<'_>) -> Result<Vec<ByteSpan>, PipelineError> {
        let mut workspace = PipelineWorkspace::<f32, f32>::default();
        let mut spans = Vec::new();
        self.run_into(text, &mut workspace, &mut spans)?;
        Ok(spans)
    }
}

/// More generic pipeline that produces score matrices and delegates decoding to
/// task-specific score decoders.
#[derive(Clone, Debug)]
pub struct ScoringPipeline<S, K, M, D> {
    scanner: S,
    kernel: K,
    model: M,
    decoder: D,
}

impl<S, K, M, D> ScoringPipeline<S, K, M, D> {
    #[must_use]
    pub fn new(scanner: S, kernel: K, model: M, decoder: D) -> Self {
        Self {
            scanner,
            kernel,
            model,
            decoder,
        }
    }

    #[must_use]
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}

impl<S, K, M, D> ScoringPipeline<S, K, M, D>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    M: BatchScorer<f32, f32>,
{
    pub fn scan_positions(&self, text: TextBytes<'_>, out: &mut CandidateBuffer) {
        self.scanner.scan_into(text, ScanRange::full(text), out);
    }

    pub fn extract_features(
        &self,
        text: TextBytes<'_>,
        positions: &CandidateBuffer,
        workspace: &mut ScoringWorkspace<f32, f32>,
    ) -> Result<(), PipelineError> {
        workspace
            .features
            .resize_zeroed(positions.len(), self.kernel.schema().total_dim());
        self.kernel.extract_into(
            text,
            positions.as_slice(),
            workspace.features.as_view_mut(),
            &mut workspace.feature_scratch,
        )?;
        Ok(())
    }

    pub fn score_positions(
        &self,
        workspace: &mut ScoringWorkspace<f32, f32>,
    ) -> Result<(), PipelineError> {
        workspace
            .scores
            .resize_zeroed(workspace.features.rows, self.model.score_dim());
        self.model
            .score_into(workspace.features.as_view(), workspace.scores.as_view_mut())?;
        Ok(())
    }
}

impl<S, K, M, D> ScoringPipeline<S, K, M, D>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    M: BatchScorer<f32, f32>,
{
    pub fn run_scores_into(
        &self,
        text: TextBytes<'_>,
        workspace: &mut ScoringWorkspace<f32, f32>,
    ) -> Result<(), PipelineError> {
        self.scan_positions(text, &mut workspace.positions);
        let positions = workspace.positions.clone();
        self.extract_features(text, &positions, workspace)?;
        self.score_positions(workspace)?;
        Ok(())
    }
}

impl<S, K, M, D> ScoringPipeline<S, K, M, D> {
    #[must_use]
    pub fn decoder(&self) -> &D {
        &self.decoder
    }
}

impl<S, K, M, D> ScoringPipeline<S, K, M, D>
where
    S: CandidateScanner,
    K: FeatureKernel<f32>,
    M: BatchScorer<f32, f32>,
{
    pub fn decode_scores<O>(
        &self,
        text: TextBytes<'_>,
        positions: &CandidateBuffer,
        workspace: &ScoringWorkspace<f32, f32>,
        out: &mut Vec<O>,
    ) -> Result<(), PipelineError>
    where
        D: ScoreDecoder<f32, O>,
    {
        self.decoder.decode_scores_into(
            text,
            positions.as_slice(),
            workspace.scores.as_view(),
            out,
        )?;
        Ok(())
    }

    pub fn run_into<O>(
        &self,
        text: TextBytes<'_>,
        workspace: &mut ScoringWorkspace<f32, f32>,
        out: &mut Vec<O>,
    ) -> Result<(), PipelineError>
    where
        D: ScoreDecoder<f32, O>,
    {
        self.run_scores_into(text, workspace)?;
        self.decoder.decode_scores_into(
            text,
            workspace.positions.as_slice(),
            workspace.scores.as_view(),
            out,
        )?;
        Ok(())
    }

    pub fn run<O>(&self, text: TextBytes<'_>) -> Result<Vec<O>, PipelineError>
    where
        D: ScoreDecoder<f32, O>,
    {
        let mut workspace = ScoringWorkspace::<f32, f32>::default();
        let mut output = Vec::new();
        self.run_into(text, &mut workspace, &mut output)?;
        Ok(output)
    }
}
