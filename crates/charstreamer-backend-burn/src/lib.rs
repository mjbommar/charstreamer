use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::nn::lstm::{Lstm, LstmConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation::{log_sigmoid, sigmoid};
use burn::tensor::{TensorData, backend::Backend};
use charstreamer_core::{
    BatchPredictor, BatchScorer, DatasetView, FeatureMatrixView, FitError, FitScratch,
    PredictError, ScoreMatrixViewMut, TrainablePredictor,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::Path;

type InferBackend = NdArray<f32>;
type TrainBackend = Autodiff<InferBackend>;

#[derive(Debug)]
pub struct BurnModelIoError {
    message: String,
}

impl BurnModelIoError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for BurnModelIoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BurnModelIoError {}

impl From<burn::record::RecorderError> for BurnModelIoError {
    fn from(error: burn::record::RecorderError) -> Self {
        Self::new(format!("burn model recorder error: {error}"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnShallowMlpFitOptions {
    pub hidden_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for BurnShallowMlpFitOptions {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            epochs: 10,
            batch_size: 256,
            learning_rate: 1.0e-3,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BurnShallowMlpFitReport {
    pub rows: usize,
    pub cols: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnDeepMlpFitOptions {
    pub hidden_dim1: usize,
    pub hidden_dim2: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for BurnDeepMlpFitOptions {
    fn default() -> Self {
        Self {
            hidden_dim1: 128,
            hidden_dim2: 64,
            epochs: 12,
            batch_size: 256,
            learning_rate: 1.0e-3,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BurnDeepMlpFitReport {
    pub rows: usize,
    pub cols: usize,
    pub hidden_dim1: usize,
    pub hidden_dim2: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnMultiLabelMlpFitOptions {
    pub hidden_dim1: usize,
    pub hidden_dim2: usize,
    pub output_dim: usize,
    pub class_weights: Vec<f32>,
    pub positive_weights: Vec<f32>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for BurnMultiLabelMlpFitOptions {
    fn default() -> Self {
        Self {
            hidden_dim1: 96,
            hidden_dim2: 48,
            output_dim: 1,
            class_weights: Vec::new(),
            positive_weights: Vec::new(),
            epochs: 20,
            batch_size: 512,
            learning_rate: 1.0e-3,
            seed: 29,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BurnMultiLabelMlpFitReport {
    pub rows: usize,
    pub cols: usize,
    pub output_dim: usize,
    pub hidden_dim1: usize,
    pub hidden_dim2: usize,
    pub class_weights_len: usize,
    pub positive_weights_len: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnWindowCnnFitOptions {
    pub sequence_len: usize,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for BurnWindowCnnFitOptions {
    fn default() -> Self {
        Self {
            sequence_len: 9,
            conv_channels: 16,
            kernel_size: 3,
            hidden_dim: 32,
            epochs: 12,
            batch_size: 256,
            learning_rate: 1.0e-3,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BurnWindowCnnFitReport {
    pub rows: usize,
    pub cols: usize,
    pub sequence_len: usize,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnWindowGruFitOptions {
    pub sequence_len: usize,
    pub hidden_dim: usize,
    pub projection_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for BurnWindowGruFitOptions {
    fn default() -> Self {
        Self {
            sequence_len: 9,
            hidden_dim: 16,
            projection_dim: 32,
            epochs: 14,
            batch_size: 512,
            learning_rate: 1.0e-3,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BurnWindowGruFitReport {
    pub rows: usize,
    pub cols: usize,
    pub sequence_len: usize,
    pub hidden_dim: usize,
    pub projection_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnWindowLstmFitOptions {
    pub sequence_len: usize,
    pub hidden_dim: usize,
    pub projection_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for BurnWindowLstmFitOptions {
    fn default() -> Self {
        Self {
            sequence_len: 9,
            hidden_dim: 16,
            projection_dim: 32,
            epochs: 14,
            batch_size: 512,
            learning_rate: 1.0e-3,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BurnWindowLstmFitReport {
    pub rows: usize,
    pub cols: usize,
    pub sequence_len: usize,
    pub hidden_dim: usize,
    pub projection_dim: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

trait BinaryLogitModule<B: Backend>: Module<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

#[derive(Module, Debug)]
struct ShallowMlp<B: Backend> {
    input: Linear<B>,
    hidden_activation: Relu,
    output: Linear<B>,
}

impl<B: Backend> ShallowMlp<B> {
    fn new(input_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim).init(device),
            hidden_activation: Relu::new(),
            output: LinearConfig::new(hidden_dim, 1).init(device),
        }
    }
}

impl<B: Backend> BinaryLogitModule<B> for ShallowMlp<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.hidden_activation.forward(self.input.forward(input));
        self.output.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct DeepMlp<B: Backend> {
    input: Linear<B>,
    hidden1_activation: Relu,
    hidden2: Linear<B>,
    hidden2_activation: Relu,
    output: Linear<B>,
}

impl<B: Backend> DeepMlp<B> {
    fn new(input_dim: usize, hidden_dim1: usize, hidden_dim2: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim1).init(device),
            hidden1_activation: Relu::new(),
            hidden2: LinearConfig::new(hidden_dim1, hidden_dim2).init(device),
            hidden2_activation: Relu::new(),
            output: LinearConfig::new(hidden_dim2, 1).init(device),
        }
    }
}

impl<B: Backend> BinaryLogitModule<B> for DeepMlp<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden1 = self.hidden1_activation.forward(self.input.forward(input));
        let hidden2 = self
            .hidden2_activation
            .forward(self.hidden2.forward(hidden1));
        self.output.forward(hidden2)
    }
}

#[derive(Module, Debug)]
struct WindowCnn<B: Backend> {
    conv: Conv1d<B>,
    conv_activation: Relu,
    hidden: Linear<B>,
    hidden_activation: Relu,
    output: Linear<B>,
    sequence_len: usize,
}

impl<B: Backend> WindowCnn<B> {
    fn new(
        input_dim: usize,
        sequence_len: usize,
        conv_channels: usize,
        kernel_size: usize,
        hidden_dim: usize,
        device: &B::Device,
    ) -> Self {
        let side_dim = input_dim - sequence_len;
        Self {
            conv: Conv1dConfig::new(1, conv_channels, kernel_size).init(device),
            conv_activation: Relu::new(),
            hidden: LinearConfig::new(conv_channels + side_dim, hidden_dim).init(device),
            hidden_activation: Relu::new(),
            output: LinearConfig::new(hidden_dim, 1).init(device),
            sequence_len,
        }
    }
}

impl<B: Backend> BinaryLogitModule<B> for WindowCnn<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, cols] = input.dims();
        let seq_flat = input.clone().narrow(1, 0, self.sequence_len);
        let side = input.narrow(1, self.sequence_len, cols - self.sequence_len);
        let seq = seq_flat.reshape([batch_size, 1, self.sequence_len]);
        let conv = self.conv_activation.forward(self.conv.forward(seq));
        let pooled = conv.mean_dim(2).squeeze_dim(2);
        let merged = Tensor::cat(vec![pooled, side], 1);
        let hidden = self.hidden_activation.forward(self.hidden.forward(merged));
        self.output.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct MultiLabelMlp<B: Backend> {
    input: Linear<B>,
    activation1: Relu,
    hidden2: Linear<B>,
    activation2: Relu,
    output: Linear<B>,
}

impl<B: Backend> MultiLabelMlp<B> {
    fn new(
        input_dim: usize,
        hidden_dim1: usize,
        hidden_dim2: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim1).init(device),
            activation1: Relu::new(),
            hidden2: LinearConfig::new(hidden_dim1, hidden_dim2).init(device),
            activation2: Relu::new(),
            output: LinearConfig::new(hidden_dim2, output_dim).init(device),
        }
    }

    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden1 = self.activation1.forward(self.input.forward(input));
        let hidden2 = self.activation2.forward(self.hidden2.forward(hidden1));
        self.output.forward(hidden2)
    }
}

#[derive(Debug)]
pub struct BurnShallowMlpModel {
    model: ShallowMlp<InferBackend>,
}

#[derive(Debug)]
pub struct BurnDeepMlpModel {
    model: DeepMlp<InferBackend>,
}

#[derive(Debug)]
pub struct BurnMultiLabelMlpModel {
    model: MultiLabelMlp<InferBackend>,
    output_dim: usize,
}

#[derive(Debug)]
pub struct BurnWindowCnnModel {
    model: WindowCnn<InferBackend>,
}

#[derive(Module, Debug)]
struct WindowGru<B: Backend> {
    gru: Gru<B>,
    projection: Linear<B>,
    projection_activation: Relu,
    output: Linear<B>,
    sequence_len: usize,
}

impl<B: Backend> WindowGru<B> {
    fn new(
        input_dim: usize,
        sequence_len: usize,
        hidden_dim: usize,
        projection_dim: usize,
        device: &B::Device,
    ) -> Self {
        let side_dim = input_dim - sequence_len;
        Self {
            gru: GruConfig::new(1, hidden_dim, true).init(device),
            projection: LinearConfig::new(hidden_dim + side_dim, projection_dim).init(device),
            projection_activation: Relu::new(),
            output: LinearConfig::new(projection_dim, 1).init(device),
            sequence_len,
        }
    }
}

impl<B: Backend> BinaryLogitModule<B> for WindowGru<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, cols] = input.dims();
        let seq_flat = input.clone().narrow(1, 0, self.sequence_len);
        let side = input.narrow(1, self.sequence_len, cols - self.sequence_len);
        let seq = seq_flat.reshape([batch_size, self.sequence_len, 1]);
        let hidden = self.gru.forward(seq, None);
        let last = hidden.narrow(1, self.sequence_len - 1, 1).squeeze_dim(1);
        let merged = Tensor::cat(vec![last, side], 1);
        let projected = self
            .projection_activation
            .forward(self.projection.forward(merged));
        self.output.forward(projected)
    }
}

#[derive(Debug)]
pub struct BurnWindowGruModel {
    model: WindowGru<InferBackend>,
}

#[derive(Module, Debug)]
struct WindowLstm<B: Backend> {
    lstm: Lstm<B>,
    projection: Linear<B>,
    projection_activation: Relu,
    output: Linear<B>,
    sequence_len: usize,
}

impl<B: Backend> WindowLstm<B> {
    fn new(
        input_dim: usize,
        sequence_len: usize,
        hidden_dim: usize,
        projection_dim: usize,
        device: &B::Device,
    ) -> Self {
        let side_dim = input_dim - sequence_len;
        Self {
            lstm: LstmConfig::new(1, hidden_dim, true).init(device),
            projection: LinearConfig::new(hidden_dim + side_dim, projection_dim).init(device),
            projection_activation: Relu::new(),
            output: LinearConfig::new(projection_dim, 1).init(device),
            sequence_len,
        }
    }
}

impl<B: Backend> BinaryLogitModule<B> for WindowLstm<B> {
    fn forward_logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, cols] = input.dims();
        let seq_flat = input.clone().narrow(1, 0, self.sequence_len);
        let side = input.narrow(1, self.sequence_len, cols - self.sequence_len);
        let seq = seq_flat.reshape([batch_size, self.sequence_len, 1]);
        let (hidden, _) = self.lstm.forward(seq, None);
        let last = hidden.narrow(1, self.sequence_len - 1, 1).squeeze_dim(1);
        let merged = Tensor::cat(vec![last, side], 1);
        let projected = self
            .projection_activation
            .forward(self.projection.forward(merged));
        self.output.forward(projected)
    }
}

#[derive(Debug)]
pub struct BurnWindowLstmModel {
    model: WindowLstm<InferBackend>,
}

impl BurnShallowMlpModel {
    pub fn save_named_mpk(&self, path: impl AsRef<Path>) -> Result<(), BurnModelIoError> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .save_file(path.as_ref().to_path_buf(), &recorder)?;
        Ok(())
    }

    pub fn load_named_mpk(
        input_dim: usize,
        hidden_dim: usize,
        path: impl AsRef<Path>,
    ) -> Result<Self, BurnModelIoError> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(BurnModelIoError::new(
                "burn shallow MLP load requires positive input_dim and hidden_dim",
            ));
        }
        let device = Default::default();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = ShallowMlp::new(input_dim, hidden_dim, &device).load_file(
            path.as_ref().to_path_buf(),
            &recorder,
            &device,
        )?;
        Ok(Self { model })
    }
}

impl BurnMultiLabelMlpModel {
    pub fn save_named_mpk(&self, path: impl AsRef<Path>) -> Result<(), BurnModelIoError> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .save_file(path.as_ref().to_path_buf(), &recorder)?;
        Ok(())
    }

    pub fn load_named_mpk(
        input_dim: usize,
        hidden_dim1: usize,
        hidden_dim2: usize,
        output_dim: usize,
        path: impl AsRef<Path>,
    ) -> Result<Self, BurnModelIoError> {
        if input_dim == 0 || hidden_dim1 == 0 || hidden_dim2 == 0 || output_dim == 0 {
            return Err(BurnModelIoError::new(
                "burn multi-label MLP load requires positive dimensions",
            ));
        }
        let device = Default::default();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = MultiLabelMlp::new(input_dim, hidden_dim1, hidden_dim2, output_dim, &device)
            .load_file(path.as_ref().to_path_buf(), &recorder, &device)?;
        Ok(Self { model, output_dim })
    }

    pub fn fit_multilabel(
        features: FeatureMatrixView<'_, f32>,
        targets: &[u8],
        options: &BurnMultiLabelMlpFitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, BurnMultiLabelMlpFitReport), FitError> {
        if features.rows == 0 || features.cols == 0 {
            return Err(FitError::new(
                "burn multi-label MLP training requires non-empty features",
            ));
        }
        if options.output_dim == 0
            || options.hidden_dim1 == 0
            || options.hidden_dim2 == 0
            || options.epochs == 0
            || options.batch_size == 0
        {
            return Err(FitError::new(
                "burn multi-label MLP options must use positive dimensions, epochs, and batch_size",
            ));
        }
        if targets.len() != features.rows * options.output_dim {
            return Err(FitError::new(
                "multi-label targets must contain rows * output_dim values",
            ));
        }
        if !options.class_weights.is_empty() && options.class_weights.len() != options.output_dim {
            return Err(FitError::new(
                "multi-label class weights must be empty or contain output_dim values",
            ));
        }
        if !options.positive_weights.is_empty()
            && options.positive_weights.len() != options.output_dim
        {
            return Err(FitError::new(
                "multi-label positive weights must be empty or contain output_dim values",
            ));
        }

        let device = Default::default();
        TrainBackend::seed(&device, options.seed);
        let mut model = MultiLabelMlp::<TrainBackend>::new(
            features.cols,
            options.hidden_dim1,
            options.hidden_dim2,
            options.output_dim,
            &device,
        );
        let mut optimizer = AdamConfig::new().init();
        let loss_fn = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(if options.class_weights.is_empty() {
                None
            } else {
                Some(options.class_weights.clone())
            })
            .init(&device);
        let positive_weights = if options.positive_weights.is_empty() {
            None
        } else {
            Some(Tensor::<TrainBackend, 1>::from_floats(
                options.positive_weights.as_slice(),
                &device,
            ))
        };

        scratch.indices.resize(features.rows, 0);
        for (index, slot) in scratch.indices.iter_mut().enumerate() {
            *slot = index;
        }
        let mut rng = SmallRng::seed_from_u64(options.seed);
        let mut batch_targets = Vec::new();

        for _epoch in 0..options.epochs {
            scratch.indices.shuffle(&mut rng);
            for batch_rows in scratch.indices.chunks(options.batch_size) {
                gather_batch_features(features, batch_rows, &mut scratch.floats);
                gather_batch_multilabel_targets(
                    targets,
                    options.output_dim,
                    batch_rows,
                    &mut batch_targets,
                );

                let batch_features = Tensor::<TrainBackend, 2>::from_data(
                    TensorData::new(scratch.floats.clone(), [batch_rows.len(), features.cols]),
                    &device,
                );
                let batch_targets = Tensor::<TrainBackend, 2, Int>::from_data(
                    TensorData::new(
                        batch_targets.clone(),
                        [batch_rows.len(), options.output_dim],
                    ),
                    &device,
                );
                let logits = model.forward_logits(batch_features);
                let loss = if let Some(positive_weights) = &positive_weights {
                    positive_weighted_bce_logits(logits, batch_targets, positive_weights.clone())
                } else {
                    loss_fn.forward(logits, batch_targets)
                };
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(options.learning_rate, model, grads);
            }
        }

        Ok((
            Self {
                model: model.valid(),
                output_dim: options.output_dim,
            },
            BurnMultiLabelMlpFitReport {
                rows: features.rows,
                cols: features.cols,
                output_dim: options.output_dim,
                hidden_dim1: options.hidden_dim1,
                hidden_dim2: options.hidden_dim2,
                class_weights_len: options.class_weights.len(),
                positive_weights_len: options.positive_weights.len(),
                epochs: options.epochs,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate,
                seed: options.seed,
            },
        ))
    }

    pub fn predict_flat_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        let required = features.rows * self.output_dim;
        if out.len() < required {
            return Err(PredictError::new("score output buffer is too small"));
        }
        if features.rows == 0 {
            return Ok(());
        }
        let device = Default::default();
        let inputs = matrix_to_tensor::<InferBackend>(features, &device);
        let scores = sigmoid(self.model.forward_logits(inputs))
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| PredictError::new(format!("burn tensor readback failed: {error}")))?;
        out[..required].copy_from_slice(&scores[..required]);
        Ok(())
    }
}

fn matrix_to_tensor<B: Backend>(
    features: FeatureMatrixView<'_, f32>,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        TensorData::new(features.data.to_vec(), [features.rows, features.cols]),
        device,
    )
}

fn labels_to_tensor<B: Backend>(labels: &[u8], device: &B::Device) -> Tensor<B, 2, Int> {
    let data: Vec<i64> = labels.iter().copied().map(i64::from).collect();
    Tensor::<B, 2, Int>::from_data(TensorData::new(data, [labels.len(), 1]), device)
}

fn gather_batch_features(
    features: FeatureMatrixView<'_, f32>,
    rows: &[usize],
    scratch: &mut Vec<f32>,
) {
    scratch.clear();
    scratch.reserve(rows.len() * features.cols);
    for &row in rows {
        scratch.extend_from_slice(features.row(row));
    }
}

fn gather_batch_labels(labels: &[u8], rows: &[usize], scratch: &mut Vec<u8>) {
    scratch.clear();
    scratch.reserve(rows.len());
    for &row in rows {
        scratch.push(labels[row]);
    }
}

fn positive_weighted_bce_logits(
    logits: Tensor<TrainBackend, 2>,
    targets: Tensor<TrainBackend, 2, Int>,
    positive_weights: Tensor<TrainBackend, 1>,
) -> Tensor<TrainBackend, 1> {
    let targets_float = targets.float();
    let shape = targets_float.dims();
    let positive_weights = positive_weights.expand(shape);
    let positive_scale = (positive_weights - 1.0) * targets_float.clone() + 1.0;
    let negative_scale = targets_float.neg() + 1.0;
    (negative_scale * logits.clone() - positive_scale * log_sigmoid(logits)).mean()
}

fn gather_batch_multilabel_targets(
    targets: &[u8],
    output_dim: usize,
    rows: &[usize],
    scratch: &mut Vec<i64>,
) {
    scratch.clear();
    scratch.reserve(rows.len() * output_dim);
    for &row in rows {
        let start = row * output_dim;
        scratch.extend(
            targets[start..start + output_dim]
                .iter()
                .copied()
                .map(i64::from),
        );
    }
}

fn predict_probabilities<M>(
    model: &M,
    features: FeatureMatrixView<'_, f32>,
    out: &mut [f32],
) -> Result<(), PredictError>
where
    M: BinaryLogitModule<InferBackend>,
{
    if out.len() < features.rows {
        return Err(PredictError::new("score output buffer is too small"));
    }
    if features.rows == 0 {
        return Ok(());
    }

    let device = Default::default();
    let inputs = matrix_to_tensor::<InferBackend>(features, &device);
    let logits = model.forward_logits(inputs);
    let probabilities = sigmoid(logits)
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| PredictError::new(format!("burn tensor readback failed: {error}")))?;

    for (slot, probability) in out.iter_mut().zip(probabilities.into_iter()) {
        *slot = probability;
    }
    Ok(())
}

fn fit_binary_model<M, F>(
    dataset: DatasetView<'_, f32, u8>,
    scratch: &mut FitScratch,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
    build: F,
) -> Result<M::InnerModule, FitError>
where
    M: BinaryLogitModule<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: BinaryLogitModule<InferBackend>,
    F: FnOnce(&<TrainBackend as Backend>::Device, usize) -> M,
{
    let device = Default::default();
    TrainBackend::seed(&device, seed);

    let mut model = build(&device, dataset.features.cols);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);

    scratch.indices.resize(dataset.features.rows, 0);
    for (index, slot) in scratch.indices.iter_mut().enumerate() {
        *slot = index;
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut batch_labels = Vec::new();

    for _epoch in 0..epochs {
        scratch.indices.shuffle(&mut rng);
        for batch_rows in scratch.indices.chunks(batch_size) {
            gather_batch_features(dataset.features, batch_rows, &mut scratch.floats);
            gather_batch_labels(dataset.labels, batch_rows, &mut batch_labels);

            let batch_features = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(
                    scratch.floats.clone(),
                    [batch_rows.len(), dataset.features.cols],
                ),
                &device,
            );
            let batch_targets = labels_to_tensor::<TrainBackend>(&batch_labels, &device);
            let logits = model.forward_logits(batch_features);
            let loss = loss_fn.forward(logits, batch_targets);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }
    }

    Ok(model.valid())
}

impl BatchPredictor<f32, f32> for BurnShallowMlpModel {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        predict_probabilities(&self.model, features, out)
    }
}

impl TrainablePredictor<f32, u8> for BurnShallowMlpModel {
    type FitOptions = BurnShallowMlpFitOptions;
    type FitReport = BurnShallowMlpFitReport;

    fn fit(
        dataset: DatasetView<'_, f32, u8>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError> {
        validate_base_dataset(
            dataset,
            options.hidden_dim,
            options.epochs,
            options.batch_size,
        )?;
        let model = fit_binary_model::<ShallowMlp<TrainBackend>, _>(
            dataset,
            scratch,
            options.epochs,
            options.batch_size,
            options.learning_rate,
            options.seed,
            |device, cols| ShallowMlp::new(cols, options.hidden_dim, device),
        )?;

        Ok((
            Self { model },
            BurnShallowMlpFitReport {
                rows: dataset.features.rows,
                cols: dataset.features.cols,
                hidden_dim: options.hidden_dim,
                epochs: options.epochs,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate,
                seed: options.seed,
            },
        ))
    }
}

impl BatchPredictor<f32, f32> for BurnDeepMlpModel {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        predict_probabilities(&self.model, features, out)
    }
}

impl BatchScorer<f32, f32> for BurnMultiLabelMlpModel {
    fn score_dim(&self) -> usize {
        self.output_dim
    }

    fn score_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        mut out: ScoreMatrixViewMut<'_, f32>,
    ) -> Result<(), PredictError> {
        if out.rows != features.rows || out.cols != self.output_dim {
            return Err(PredictError::new(
                "multi-label score output shape does not match model",
            ));
        }
        let mut flat = vec![0.0_f32; features.rows * self.output_dim];
        self.predict_flat_into(features, &mut flat)?;
        for row in 0..out.rows {
            let start = row * self.output_dim;
            out.row_mut(row)
                .copy_from_slice(&flat[start..start + self.output_dim]);
        }
        Ok(())
    }
}

impl TrainablePredictor<f32, u8> for BurnDeepMlpModel {
    type FitOptions = BurnDeepMlpFitOptions;
    type FitReport = BurnDeepMlpFitReport;

    fn fit(
        dataset: DatasetView<'_, f32, u8>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError> {
        validate_base_dataset(
            dataset,
            options.hidden_dim1.min(options.hidden_dim2),
            options.epochs,
            options.batch_size,
        )?;
        if options.hidden_dim2 == 0 {
            return Err(FitError::new(
                "burn deep MLP options must use a positive hidden_dim2",
            ));
        }

        let model = fit_binary_model::<DeepMlp<TrainBackend>, _>(
            dataset,
            scratch,
            options.epochs,
            options.batch_size,
            options.learning_rate,
            options.seed,
            |device, cols| DeepMlp::new(cols, options.hidden_dim1, options.hidden_dim2, device),
        )?;

        Ok((
            Self { model },
            BurnDeepMlpFitReport {
                rows: dataset.features.rows,
                cols: dataset.features.cols,
                hidden_dim1: options.hidden_dim1,
                hidden_dim2: options.hidden_dim2,
                epochs: options.epochs,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate,
                seed: options.seed,
            },
        ))
    }
}

impl BatchPredictor<f32, f32> for BurnWindowCnnModel {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        predict_probabilities(&self.model, features, out)
    }
}

impl TrainablePredictor<f32, u8> for BurnWindowCnnModel {
    type FitOptions = BurnWindowCnnFitOptions;
    type FitReport = BurnWindowCnnFitReport;

    fn fit(
        dataset: DatasetView<'_, f32, u8>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError> {
        validate_base_dataset(
            dataset,
            options.conv_channels.min(options.hidden_dim),
            options.epochs,
            options.batch_size,
        )?;
        if options.sequence_len == 0
            || options.sequence_len >= dataset.features.cols
            || options.kernel_size == 0
            || options.kernel_size > options.sequence_len
        {
            return Err(FitError::new(
                "burn window CNN requires 0 < kernel_size <= sequence_len < feature_cols",
            ));
        }

        let model = fit_binary_model::<WindowCnn<TrainBackend>, _>(
            dataset,
            scratch,
            options.epochs,
            options.batch_size,
            options.learning_rate,
            options.seed,
            |device, cols| {
                WindowCnn::new(
                    cols,
                    options.sequence_len,
                    options.conv_channels,
                    options.kernel_size,
                    options.hidden_dim,
                    device,
                )
            },
        )?;

        Ok((
            Self { model },
            BurnWindowCnnFitReport {
                rows: dataset.features.rows,
                cols: dataset.features.cols,
                sequence_len: options.sequence_len,
                conv_channels: options.conv_channels,
                kernel_size: options.kernel_size,
                hidden_dim: options.hidden_dim,
                epochs: options.epochs,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate,
                seed: options.seed,
            },
        ))
    }
}

impl BatchPredictor<f32, f32> for BurnWindowGruModel {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        predict_probabilities(&self.model, features, out)
    }
}

impl TrainablePredictor<f32, u8> for BurnWindowGruModel {
    type FitOptions = BurnWindowGruFitOptions;
    type FitReport = BurnWindowGruFitReport;

    fn fit(
        dataset: DatasetView<'_, f32, u8>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError> {
        validate_base_dataset(
            dataset,
            options.hidden_dim.min(options.projection_dim),
            options.epochs,
            options.batch_size,
        )?;
        validate_sequence_layout(dataset, options.sequence_len)?;

        let model = fit_binary_model::<WindowGru<TrainBackend>, _>(
            dataset,
            scratch,
            options.epochs,
            options.batch_size,
            options.learning_rate,
            options.seed,
            |device, cols| {
                WindowGru::new(
                    cols,
                    options.sequence_len,
                    options.hidden_dim,
                    options.projection_dim,
                    device,
                )
            },
        )?;

        Ok((
            Self { model },
            BurnWindowGruFitReport {
                rows: dataset.features.rows,
                cols: dataset.features.cols,
                sequence_len: options.sequence_len,
                hidden_dim: options.hidden_dim,
                projection_dim: options.projection_dim,
                epochs: options.epochs,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate,
                seed: options.seed,
            },
        ))
    }
}

impl BatchPredictor<f32, f32> for BurnWindowLstmModel {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, f32>,
        out: &mut [f32],
    ) -> Result<(), PredictError> {
        predict_probabilities(&self.model, features, out)
    }
}

impl TrainablePredictor<f32, u8> for BurnWindowLstmModel {
    type FitOptions = BurnWindowLstmFitOptions;
    type FitReport = BurnWindowLstmFitReport;

    fn fit(
        dataset: DatasetView<'_, f32, u8>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError> {
        validate_base_dataset(
            dataset,
            options.hidden_dim.min(options.projection_dim),
            options.epochs,
            options.batch_size,
        )?;
        validate_sequence_layout(dataset, options.sequence_len)?;

        let model = fit_binary_model::<WindowLstm<TrainBackend>, _>(
            dataset,
            scratch,
            options.epochs,
            options.batch_size,
            options.learning_rate,
            options.seed,
            |device, cols| {
                WindowLstm::new(
                    cols,
                    options.sequence_len,
                    options.hidden_dim,
                    options.projection_dim,
                    device,
                )
            },
        )?;

        Ok((
            Self { model },
            BurnWindowLstmFitReport {
                rows: dataset.features.rows,
                cols: dataset.features.cols,
                sequence_len: options.sequence_len,
                hidden_dim: options.hidden_dim,
                projection_dim: options.projection_dim,
                epochs: options.epochs,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate,
                seed: options.seed,
            },
        ))
    }
}

fn validate_base_dataset(
    dataset: DatasetView<'_, f32, u8>,
    hidden_dim: usize,
    epochs: usize,
    batch_size: usize,
) -> Result<(), FitError> {
    if dataset.features.rows != dataset.labels.len() {
        return Err(FitError::new(
            "dataset feature rows and labels must have matching lengths",
        ));
    }
    if dataset.features.rows == 0 || dataset.features.cols == 0 {
        return Err(FitError::new(
            "burn model training requires non-empty features and labels",
        ));
    }
    if hidden_dim == 0 || epochs == 0 || batch_size == 0 {
        return Err(FitError::new(
            "burn model options must use positive hidden dims, epochs, and batch_size",
        ));
    }
    Ok(())
}

fn validate_sequence_layout(
    dataset: DatasetView<'_, f32, u8>,
    sequence_len: usize,
) -> Result<(), FitError> {
    if sequence_len == 0 || sequence_len >= dataset.features.cols {
        return Err(FitError::new(
            "burn sequence models require 0 < sequence_len < feature_cols",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use charstreamer_core::{
        BatchPredictor, DatasetView, FeatureMatrix, FitScratch, TrainablePredictor,
    };

    use crate::{
        BurnDeepMlpFitOptions, BurnDeepMlpModel, BurnMultiLabelMlpFitOptions,
        BurnMultiLabelMlpModel, BurnShallowMlpFitOptions, BurnShallowMlpModel,
        BurnWindowCnnFitOptions, BurnWindowCnnModel, BurnWindowGruFitOptions, BurnWindowGruModel,
        BurnWindowLstmFitOptions, BurnWindowLstmModel,
    };

    static BURN_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn burn_test_lock() -> MutexGuard<'static, ()> {
        BURN_TEST_LOCK
            .lock()
            .expect("burn backend test lock should not be poisoned")
    }

    fn simple_dataset() -> (FeatureMatrix<f32>, [u8; 4]) {
        (
            FeatureMatrix::<f32> {
                rows: 4,
                cols: 4,
                data: vec![
                    2.0, 1.0, 0.2, 0.1, 1.5, 0.5, 0.1, 0.0, -1.0, -0.5, -0.2, -0.1, -2.0, -1.0,
                    -0.3, -0.2,
                ],
            },
            [1_u8, 1, 0, 0],
        )
    }

    fn assert_positive_rows_score_higher(out: &[f32]) {
        let positive_mean = (out[0] + out[1]) * 0.5;
        let negative_mean = (out[2] + out[3]) * 0.5;
        assert!(
            positive_mean > negative_mean,
            "expected positive rows to score higher than negative rows, got {out:?}"
        );
    }

    #[test]
    fn burn_shallow_mlp_fits_simple_dataset() {
        let _guard = burn_test_lock();
        let (features, labels) = simple_dataset();
        let (model, report) = BurnShallowMlpModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &BurnShallowMlpFitOptions {
                hidden_dim: 8,
                epochs: 50,
                batch_size: 4,
                learning_rate: 1.0e-2,
                seed: 7,
            },
            &mut FitScratch::default(),
        )
        .expect("burn shallow mlp fit should succeed");

        assert_eq!(report.rows, 4);
        let mut out = vec![0.0_f32; 4];
        model
            .predict_into(features.as_view(), &mut out)
            .expect("burn shallow mlp prediction should succeed");
        assert_positive_rows_score_higher(&out);
    }

    #[test]
    fn burn_deep_mlp_fits_simple_dataset() {
        let _guard = burn_test_lock();
        let (features, labels) = simple_dataset();
        let (model, _) = BurnDeepMlpModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &BurnDeepMlpFitOptions {
                hidden_dim1: 8,
                hidden_dim2: 4,
                epochs: 60,
                batch_size: 4,
                learning_rate: 1.0e-2,
                seed: 7,
            },
            &mut FitScratch::default(),
        )
        .expect("burn deep mlp fit should succeed");

        let mut out = vec![0.0_f32; 4];
        model
            .predict_into(features.as_view(), &mut out)
            .expect("burn deep mlp prediction should succeed");
        assert_positive_rows_score_higher(&out);
    }

    #[test]
    fn burn_multi_label_mlp_fits_simple_dataset() {
        let _guard = burn_test_lock();
        let (features, _) = simple_dataset();
        let labels = vec![0, 1, 1, 0, 1, 1, 0, 0];
        let (model, report) = BurnMultiLabelMlpModel::fit_multilabel(
            features.as_view(),
            &labels,
            &BurnMultiLabelMlpFitOptions {
                hidden_dim1: 8,
                hidden_dim2: 4,
                output_dim: 2,
                class_weights: Vec::new(),
                positive_weights: Vec::new(),
                epochs: 25,
                batch_size: 2,
                learning_rate: 0.01,
                seed: 13,
            },
            &mut FitScratch::default(),
        )
        .expect("multi-label MLP should fit");
        let mut scores = vec![0.0; labels.len()];
        model
            .predict_flat_into(features.as_view(), &mut scores)
            .expect("prediction should succeed");
        assert_eq!(report.output_dim, 2);
        assert!(scores.iter().all(|score| (0.0..=1.0).contains(score)));
    }

    #[test]
    fn burn_window_cnn_fits_simple_dataset() {
        let _guard = burn_test_lock();
        let (features, labels) = simple_dataset();
        let (model, _) = BurnWindowCnnModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &BurnWindowCnnFitOptions {
                sequence_len: 2,
                conv_channels: 4,
                kernel_size: 2,
                hidden_dim: 4,
                epochs: 60,
                batch_size: 4,
                learning_rate: 1.0e-2,
                seed: 7,
            },
            &mut FitScratch::default(),
        )
        .expect("burn window cnn fit should succeed");

        let mut out = vec![0.0_f32; 4];
        model
            .predict_into(features.as_view(), &mut out)
            .expect("burn window cnn prediction should succeed");
        assert_positive_rows_score_higher(&out);
    }

    #[test]
    fn burn_window_gru_fits_simple_dataset() {
        let _guard = burn_test_lock();
        let (features, labels) = simple_dataset();
        let (model, _) = BurnWindowGruModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &BurnWindowGruFitOptions {
                sequence_len: 2,
                hidden_dim: 4,
                projection_dim: 4,
                epochs: 40,
                batch_size: 4,
                learning_rate: 1.0e-2,
                seed: 7,
            },
            &mut FitScratch::default(),
        )
        .expect("burn window gru fit should succeed");

        let mut out = vec![0.0_f32; 4];
        model
            .predict_into(features.as_view(), &mut out)
            .expect("burn window gru prediction should succeed");
        assert_positive_rows_score_higher(&out);
    }

    #[test]
    fn burn_window_lstm_fits_simple_dataset() {
        let _guard = burn_test_lock();
        let (features, labels) = simple_dataset();
        let (model, _) = BurnWindowLstmModel::fit(
            DatasetView {
                features: features.as_view(),
                labels: &labels,
            },
            &BurnWindowLstmFitOptions {
                sequence_len: 2,
                hidden_dim: 4,
                projection_dim: 4,
                epochs: 40,
                batch_size: 4,
                learning_rate: 1.0e-2,
                seed: 7,
            },
            &mut FitScratch::default(),
        )
        .expect("burn window lstm fit should succeed");

        let mut out = vec![0.0_f32; 4];
        model
            .predict_into(features.as_view(), &mut out)
            .expect("burn window lstm prediction should succeed");
        assert_positive_rows_score_higher(&out);
    }
}
