use charstreamer_core::{
    CandidateBuffer, CandidateScanner, DatasetView, FeatureKernel, FeatureMatrix, FeatureScratch,
    FitScratch, Pipeline, ScanRange, TextBytes, ThresholdSpanDecoder, TrainablePredictor,
};
use charstreamer_kernels::{ByteSet256, ByteSetScanner, CompositeFeatureKernel};
use charstreamer_models_native::{LogisticFitOptions, LogisticModel};

#[test]
fn narrow_vertical_slice_segments_ascii_text() {
    let train_text = "Alpha. Beta, Gamma? Delta! Epsilon, Zeta.";
    let train_view = TextBytes::from_utf8(train_text);

    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!,"));
    let kernel = CompositeFeatureKernel::boundary_demo();
    let mut candidates = CandidateBuffer::new();
    scanner.scan_into(train_view, ScanRange::full(train_view), &mut candidates);

    let mut features = FeatureMatrix::<f32>::default();
    features.resize_zeroed(candidates.len(), kernel.schema().total_dim());
    kernel
        .extract_into(
            train_view,
            candidates.as_slice(),
            features.as_view_mut(),
            &mut FeatureScratch::default(),
        )
        .expect("feature extraction should succeed");

    let labels: Vec<u8> = candidates
        .positions()
        .iter()
        .map(|position| {
            let byte = train_text.as_bytes()[position.as_usize()];
            u8::from(matches!(byte, b'.' | b'?' | b'!'))
        })
        .collect();

    let options = LogisticFitOptions {
        epochs: 250,
        learning_rate: 0.25,
        batch_size: 4,
        shuffle: false,
        l2: 0.0,
        positive_class_weight: Some(1.0),
        seed: 7,
    };
    let (model, _report) = LogisticModel::fit(
        DatasetView {
            features: features.as_view(),
            labels: &labels,
        },
        &options,
        &mut FitScratch::default(),
    )
    .expect("training should succeed");

    let text = "Alpha. Beta, Gamma? Delta.";
    let text_view = TextBytes::from_utf8(text);
    let decoder = ThresholdSpanDecoder::new(0.50);
    let pipeline = Pipeline::new(scanner, kernel, model, decoder);

    let spans = pipeline.run(text_view).expect("pipeline should succeed");
    let segments: Vec<&str> = spans
        .iter()
        .map(|span| &text[span.start.as_usize()..span.end.as_usize()])
        .collect();

    assert_eq!(segments, vec!["Alpha.", " Beta, Gamma?", " Delta."]);
}
