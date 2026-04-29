//! Core data types, views, traits, and pipeline orchestration for `charstreamer`.

pub mod corpus;
pub mod data;
pub mod decoder;
pub mod error;
pub mod metrics;
pub mod pipeline;
pub mod text;
pub mod traits;

pub use corpus::{
    AnnotatedDocument, BoundaryDataset, BoundaryDatasetBuildOptions, CorpusError,
    TrainingPositionPolicy, build_boundary_dataset, load_alea_jsonl, load_multilegal_jsonl,
    split_documents,
};
pub use data::{
    BytePos, ByteSpan, ByteWindowSpec, CandidateBuffer, CandidateSlice, ChunkRange, DatasetView,
    FeatureBlock, FeatureMatrix, FeatureMatrixView, FeatureMatrixViewMut, FeatureRowMut,
    FeatureSchema, FeatureScratch, FitScratch, LabelAtPos, LabeledSpan, OwnedRange,
    PipelineWorkspace, PositionBuffer, PositionSlice, ScalarPos, ScalarWindowSpec, ScanRange,
    ScoreBuffer, ScoreMatrix, ScoreMatrixView, ScoreMatrixViewMut, ScoringWorkspace,
    StrideWindowSpec,
};
pub use decoder::{ArgmaxLabelDecoder, ContiguousSpanDecoder, ThresholdSpanDecoder};
pub use error::{DecodeError, FeatureError, FitError, PipelineError, PredictError};
pub use metrics::{
    BinaryMetrics, PipelineEvaluation, ThroughputReport, benchmark_pipeline,
    best_threshold_from_scores, evaluate_pipeline, metrics_from_scores,
};
pub use pipeline::{Pipeline, ScoringPipeline};
pub use text::{AsciiByteView, ByteToCharMap, TextBytes, Utf8ScalarView};
pub use traits::{
    BatchPredictor, BatchScorer, CandidateScanner, Decoder, FeatureAppender, FeatureKernel,
    ScoreDecoder, TrainablePredictor,
};
