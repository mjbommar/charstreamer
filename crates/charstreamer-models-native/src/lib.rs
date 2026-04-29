//! Native CPU-friendly models for the first `charstreamer` slice.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

use charstreamer_core::{
    BatchPredictor, BatchScorer, DatasetView, FeatureMatrixView, FeatureSchema, FitError,
    FitScratch, PredictError, ScoreMatrixViewMut, TrainablePredictor,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

const LOGISTIC_ARTIFACT_VERSION: &str = "charstreamer.logistic.v1";

#[derive(Debug)]
pub enum ModelIoError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidArtifact(String),
}

impl ModelIoError {
    #[must_use]
    pub fn invalid_artifact(message: impl Into<String>) -> Self {
        Self::InvalidArtifact(message.into())
    }
}

impl Display for ModelIoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "model I/O failed: {error}"),
            Self::Json(error) => write!(f, "model JSON serialization failed: {error}"),
            Self::InvalidArtifact(message) => write!(f, "invalid model artifact: {message}"),
        }
    }
}

impl Error for ModelIoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::InvalidArtifact(_) => None,
        }
    }
}

impl From<std::io::Error> for ModelIoError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ModelIoError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFeatureBlock {
    pub name: String,
    pub offset: usize,
    pub width: usize,
}

impl ArtifactFeatureBlock {
    #[must_use]
    pub fn from_schema(schema: &FeatureSchema) -> Vec<Self> {
        schema
            .blocks()
            .iter()
            .map(|block| Self {
                name: block.name.to_string(),
                offset: block.offset,
                width: block.width,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogisticModelArtifact {
    pub format: String,
    pub bias: f32,
    pub weights: Vec<f32>,
    pub feature_dim: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schema_blocks: Vec<ArtifactFeatureBlock>,
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn dot(row: &[f32], weights: &[f32]) -> f32 {
    row.iter()
        .zip(weights.iter())
        .map(|(&feature, &weight)| feature * weight)
        .sum()
}

/// Packed logistic scorer over contiguous row-major `f32` features.
#[derive(Clone, Debug)]
pub struct LogisticModel {
    bias: f32,
    weights: Vec<f32>,
}

impl LogisticModel {
    #[must_use]
    pub fn new(bias: f32, weights: Vec<f32>) -> Self {
        Self { bias, weights }
    }

    #[must_use]
    pub fn boundary_demo(schema: &FeatureSchema) -> Self {
        let mut weights = vec![0.0; schema.total_dim()];

        if let Some(block) = schema.block("byte_window") {
            weights[block.offset + 1] = 0.15;
        }

        if let Some(block) = schema.block("ascii_classes") {
            weights[block.offset + 2] = 0.60;
            weights[block.offset + 4] = 0.80;
            weights[block.offset + 5] = -1.25;
        }

        Self {
            bias: -2.40,
            weights,
        }
    }

    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    #[must_use]
    pub fn bias(&self) -> f32 {
        self.bias
    }

    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.weights.len()
    }

    #[must_use]
    pub fn artifact(&self) -> LogisticModelArtifact {
        LogisticModelArtifact {
            format: LOGISTIC_ARTIFACT_VERSION.to_string(),
            bias: self.bias,
            feature_dim: self.weights.len(),
            weights: self.weights.clone(),
            schema_blocks: Vec::new(),
        }
    }

    #[must_use]
    pub fn artifact_with_schema(&self, schema: &FeatureSchema) -> LogisticModelArtifact {
        let mut artifact = self.artifact();
        artifact.schema_blocks = ArtifactFeatureBlock::from_schema(schema);
        artifact
    }

    pub fn from_artifact(artifact: LogisticModelArtifact) -> Result<Self, ModelIoError> {
        if artifact.format != LOGISTIC_ARTIFACT_VERSION {
            return Err(ModelIoError::invalid_artifact(format!(
                "expected format {LOGISTIC_ARTIFACT_VERSION}, got {}",
                artifact.format
            )));
        }
        if artifact.feature_dim == 0 {
            return Err(ModelIoError::invalid_artifact(
                "feature_dim must be greater than zero",
            ));
        }
        if artifact.weights.len() != artifact.feature_dim {
            return Err(ModelIoError::invalid_artifact(format!(
                "feature_dim {} does not match weight count {}",
                artifact.feature_dim,
                artifact.weights.len()
            )));
        }

        Ok(Self::new(artifact.bias, artifact.weights))
    }

    pub fn to_json_vec(&self) -> Result<Vec<u8>, ModelIoError> {
        Ok(serde_json::to_vec_pretty(&self.artifact())?)
    }

    pub fn to_json_vec_with_schema(&self, schema: &FeatureSchema) -> Result<Vec<u8>, ModelIoError> {
        Ok(serde_json::to_vec_pretty(
            &self.artifact_with_schema(schema),
        )?)
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, ModelIoError> {
        let artifact: LogisticModelArtifact = serde_json::from_slice(bytes)?;
        Self::from_artifact(artifact)
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelIoError> {
        fs::write(path, self.to_json_vec()?)?;
        Ok(())
    }

    pub fn save_json_with_schema<P: AsRef<Path>>(
        &self,
        path: P,
        schema: &FeatureSchema,
    ) -> Result<(), ModelIoError> {
        fs::write(path, self.to_json_vec_with_schema(schema)?)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, ModelIoError> {
        let bytes = fs::read(path)?;
        Self::from_json_slice(&bytes)
    }
}

/// Fit options for the native logistic trainer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogisticFitOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    pub l2: f32,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub positive_class_weight: Option<f32>,
}

impl Default for LogisticFitOptions {
    fn default() -> Self {
        Self {
            epochs: 30,
            learning_rate: 0.05,
            l2: 1e-4,
            batch_size: 512,
            shuffle: true,
            seed: 7,
            positive_class_weight: None,
        }
    }
}

/// Summary of a native logistic fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LogisticFitReport {
    pub epochs: usize,
    pub final_loss: f32,
    pub positive_class_weight: f32,
}

impl BatchPredictor<f32, f32> for LogisticModel {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        if features.cols != self.weights.len() {
            return Err(PredictError::new(
                "feature width does not match the logistic weights",
            ));
        }
        if out.len() < features.rows {
            return Err(PredictError::new("score output buffer is too small"));
        }

        for (row_index, output) in out.iter_mut().take(features.rows).enumerate() {
            let row = features.row(row_index);
            let activation = self.bias + dot(row, &self.weights);
            *output = sigmoid(activation);
        }

        Ok(())
    }
}

impl TrainablePredictor<f32, u8> for LogisticModel {
    type FitOptions = LogisticFitOptions;
    type FitReport = LogisticFitReport;

    fn fit(
        dataset: DatasetView<'_, f32, u8>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError> {
        if dataset.features.rows != dataset.labels.len() {
            return Err(FitError::new(
                "dataset feature rows and labels must have matching lengths",
            ));
        }
        if dataset.features.cols == 0 {
            return Err(FitError::new(
                "logistic training requires at least one feature column",
            ));
        }
        if dataset.features.rows == 0 {
            return Err(FitError::new("logistic training requires at least one row"));
        }

        let rows = dataset.features.rows;
        let cols = dataset.features.cols;
        let positives = dataset.labels.iter().filter(|&&label| label == 1).count();
        let negatives = rows - positives;
        let positive_class_weight = options.positive_class_weight.unwrap_or_else(|| {
            if positives == 0 {
                1.0
            } else {
                negatives.max(1) as f32 / positives as f32
            }
        });
        let positive_rate = positives.max(1) as f32 / rows as f32;
        let mut model = Self::new(
            (positive_rate / (1.0 - positive_rate).max(1e-6)).ln(),
            vec![0.0; cols],
        );

        scratch.indices.clear();
        scratch.indices.extend(0..rows);
        scratch.floats.resize(cols, 0.0);
        scratch.floats_aux.resize(cols, 0.0);

        let mut rng = SmallRng::seed_from_u64(options.seed);
        let batch_size = options.batch_size.max(1);
        let mut final_loss = 0.0_f32;

        for _ in 0..options.epochs {
            if options.shuffle {
                scratch.indices.shuffle(&mut rng);
            }

            for batch in scratch.indices.chunks(batch_size) {
                scratch.floats.fill(0.0);
                let mut grad_bias = 0.0_f32;

                for &row_index in batch {
                    let row = dataset.features.row(row_index);
                    let label = dataset.labels[row_index] as f32;
                    let weight = if label > 0.0 {
                        positive_class_weight
                    } else {
                        1.0
                    };
                    let activation = model.bias + dot(row, &model.weights);
                    let prediction = sigmoid(activation);
                    let error = (prediction - label) * weight;
                    grad_bias += error;
                    for (grad, &feature) in scratch.floats.iter_mut().zip(row.iter()) {
                        *grad += error * feature;
                    }
                }

                let batch_scale = 1.0 / batch.len() as f32;
                model.bias -= options.learning_rate * grad_bias * batch_scale;
                for (weight, grad) in model.weights.iter_mut().zip(scratch.floats.iter()) {
                    let regularized = *grad * batch_scale + options.l2 * *weight;
                    *weight -= options.learning_rate * regularized;
                }
            }

            final_loss = logistic_loss(
                dataset.features,
                dataset.labels,
                &model,
                positive_class_weight,
                options.l2,
            );
        }

        Ok((
            model,
            LogisticFitReport {
                epochs: options.epochs,
                final_loss,
                positive_class_weight,
            },
        ))
    }
}

/// Linear multiclass scorer that emits one score row per input position.
#[derive(Clone, Debug)]
pub struct LinearClassifierModel {
    bias: Vec<f32>,
    weights: Vec<f32>,
}

impl LinearClassifierModel {
    #[must_use]
    pub fn new(bias: Vec<f32>, weights: Vec<f32>) -> Self {
        let classes = bias.len();
        assert!(
            classes > 0,
            "linear classifier must have at least one class"
        );
        assert!(
            weights.len().is_multiple_of(classes),
            "weights must be laid out as classes x feature_dim",
        );
        Self { bias, weights }
    }

    #[must_use]
    pub fn classes(&self) -> usize {
        self.bias.len()
    }

    #[must_use]
    pub fn xml_csv_demo(schema: &FeatureSchema) -> Self {
        let classes = 2;
        let mut weights = vec![0.0; classes * schema.total_dim()];
        let mut bias = vec![0.0; classes];

        if let Some(block) = schema.block("format_counts") {
            let xml = 0;
            let csv = 1;
            let xml_base = xml * schema.total_dim() + block.offset;
            let csv_base = csv * schema.total_dim() + block.offset;

            weights[xml_base] = 18.0;
            weights[xml_base + 1] = 18.0;
            weights[xml_base + 2] = 12.0;
            weights[xml_base + 3] = 8.0;
            weights[xml_base + 4] = -16.0;
            weights[xml_base + 5] = 4.0;

            weights[csv_base] = -10.0;
            weights[csv_base + 1] = -10.0;
            weights[csv_base + 2] = -6.0;
            weights[csv_base + 3] = -4.0;
            weights[csv_base + 4] = 24.0;
            weights[csv_base + 5] = 6.0;
        }

        bias[0] = 0.15;
        bias[1] = -0.15;

        Self { bias, weights }
    }
}

impl BatchScorer<f32, f32> for LinearClassifierModel {
    fn score_dim(&self) -> usize {
        self.classes()
    }

    fn score_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        mut out: ScoreMatrixViewMut<'_, f32>,
    ) -> Result<(), PredictError> {
        if out.rows != features.rows || out.cols != self.classes() {
            return Err(PredictError::new(
                "score matrix shape does not match the classifier output shape",
            ));
        }

        let feature_dim = features.cols;
        if self.weights.len() != self.classes() * feature_dim {
            return Err(PredictError::new(
                "classifier weights do not match the feature dimension",
            ));
        }

        for row_index in 0..features.rows {
            let feature_row = features.row(row_index);
            let output_row = out.row_mut(row_index);
            for (class_index, output) in output_row.iter_mut().enumerate() {
                let class_weights =
                    &self.weights[class_index * feature_dim..(class_index + 1) * feature_dim];
                let mut activation = self.bias[class_index];
                for (&feature, &weight) in feature_row.iter().zip(class_weights.iter()) {
                    activation += feature * weight;
                }
                *output = activation;
            }

            let max_logit = output_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0;
            for value in output_row.iter_mut() {
                *value = (*value - max_logit).exp();
                denom += *value;
            }
            for value in output_row.iter_mut() {
                *value /= denom;
            }
        }

        Ok(())
    }
}

fn logistic_loss(
    features: FeatureMatrixView<'_, f32>,
    labels: &[u8],
    model: &LogisticModel,
    positive_class_weight: f32,
    l2: f32,
) -> f32 {
    let mut loss = 0.0_f32;

    for (row_index, &label_byte) in labels.iter().enumerate().take(features.rows) {
        let row = features.row(row_index);
        let label = label_byte as f32;
        let weight = if label > 0.0 {
            positive_class_weight
        } else {
            1.0
        };
        let probability = sigmoid(model.bias + dot(row, &model.weights)).clamp(1e-6, 1.0 - 1e-6);
        loss += -weight * (label * probability.ln() + (1.0 - label) * (1.0 - probability).ln());
    }

    let l2_term: f32 = model.weights.iter().map(|weight| weight * weight).sum();
    loss / features.rows as f32 + 0.5 * l2 * l2_term
}

#[cfg(test)]
mod tests {
    use std::fs;

    use charstreamer_core::{
        DatasetView, FeatureMatrix, FeatureSchema, FitScratch, TrainablePredictor,
    };

    use crate::{
        ArtifactFeatureBlock, LOGISTIC_ARTIFACT_VERSION, LinearClassifierModel, LogisticFitOptions,
        LogisticModel, LogisticModelArtifact, ModelIoError,
    };

    #[test]
    fn logistic_model_scores_rows() {
        let model = LogisticModel::new(0.0, vec![1.0, -1.0]);
        let features = FeatureMatrix::<f32> {
            rows: 2,
            cols: 2,
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let mut out = vec![0.0; 2];
        charstreamer_core::BatchPredictor::predict_into(&model, features.as_view(), &mut out)
            .expect("prediction should succeed");
        assert!(out[0] > 0.7);
        assert!(out[1] < 0.3);
    }

    #[test]
    fn linear_classifier_scores_classes() {
        let schema = FeatureSchema::new(vec![
            charstreamer_core::FeatureBlock::new("counts", 2).with_offset(0),
        ]);
        let model = LinearClassifierModel::new(vec![0.0, 0.0], vec![2.0, -1.0, -1.0, 2.0]);
        let features = FeatureMatrix::<f32> {
            rows: 1,
            cols: 2,
            data: vec![1.0, 0.0],
        };
        let mut out = FeatureMatrix::<f32>::default();
        out.resize_zeroed(1, model.classes());
        charstreamer_core::BatchScorer::score_into(&model, features.as_view(), out.as_view_mut())
            .expect("scoring should succeed");
        assert!(out.data[0] > out.data[1]);
        let demo = LinearClassifierModel::xml_csv_demo(&schema);
        assert_eq!(demo.classes(), 2);
    }

    #[test]
    fn logistic_model_fits_simple_dataset() {
        let features = FeatureMatrix::<f32> {
            rows: 4,
            cols: 2,
            data: vec![2.0, 1.0, 1.5, 0.5, -1.0, -0.5, -2.0, -1.0],
        };
        let labels = [1_u8, 1, 0, 0];
        let options = LogisticFitOptions {
            epochs: 80,
            learning_rate: 0.1,
            batch_size: 2,
            ..LogisticFitOptions::default()
        };
        let (model, report) = LogisticModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &options,
            &mut FitScratch::default(),
        )
        .expect("training should succeed");
        assert!(report.final_loss >= 0.0);
        let mut out = vec![0.0; 4];
        charstreamer_core::BatchPredictor::predict_into(&model, features.as_view(), &mut out)
            .expect("prediction should succeed");
        assert!(out[0] > 0.5);
        assert!(out[1] > 0.5);
        assert!(out[2] < 0.5);
        assert!(out[3] < 0.5);
    }

    #[test]
    fn logistic_model_round_trips_through_json_artifact() {
        let model = LogisticModel::new(-0.5, vec![0.25, 1.25, -0.75]);
        let schema = FeatureSchema::new(vec![
            charstreamer_core::FeatureBlock::new("window", 2).with_offset(0),
            charstreamer_core::FeatureBlock::new("extra", 1).with_offset(2),
        ]);
        let bytes = model
            .to_json_vec_with_schema(&schema)
            .expect("artifact serialization should succeed");
        let artifact: LogisticModelArtifact =
            serde_json::from_slice(&bytes).expect("artifact json should parse");
        assert_eq!(artifact.format, LOGISTIC_ARTIFACT_VERSION);
        assert_eq!(artifact.feature_dim, 3);
        assert_eq!(
            artifact.schema_blocks,
            vec![
                ArtifactFeatureBlock {
                    name: "window".to_string(),
                    offset: 0,
                    width: 2
                },
                ArtifactFeatureBlock {
                    name: "extra".to_string(),
                    offset: 2,
                    width: 1
                }
            ]
        );

        let loaded = LogisticModel::from_json_slice(&bytes)
            .expect("artifact deserialization should succeed");
        assert_eq!(loaded.bias(), model.bias());
        assert_eq!(loaded.weights(), model.weights());
    }

    #[test]
    fn logistic_model_saves_and_loads_json_file() {
        let model = LogisticModel::new(0.1, vec![0.2, -0.4]);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "charstreamer-logistic-test-{}.json",
            std::process::id()
        ));
        model.save_json(&path).expect("model save should succeed");
        let loaded = LogisticModel::load_json(&path).expect("model load should succeed");
        fs::remove_file(&path).expect("temporary artifact should be removed");

        assert_eq!(loaded.bias(), model.bias());
        assert_eq!(loaded.weights(), model.weights());
    }

    #[test]
    fn rejects_invalid_logistic_artifacts() {
        let error = LogisticModel::from_artifact(LogisticModelArtifact {
            format: "charstreamer.logistic.v0".to_string(),
            bias: 0.0,
            feature_dim: 2,
            weights: vec![0.1, 0.2],
            schema_blocks: Vec::new(),
        })
        .expect_err("artifact version mismatch should fail");
        assert!(matches!(error, ModelIoError::InvalidArtifact(_)));
    }
}
