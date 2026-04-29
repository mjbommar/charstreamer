use charstreamer_core::{Pipeline, TextBytes, ThresholdSpanDecoder};
use charstreamer_kernels::{ByteSet256, ByteSetScanner, CompositeFeatureKernel};
use charstreamer_models_native::LogisticModel;

#[test]
fn narrow_vertical_slice_segments_ascii_text() {
    let text = "Alpha. Beta? Gamma! Delta.";
    let text_view = TextBytes::from_utf8(text);

    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!"));
    let kernel = CompositeFeatureKernel::boundary_demo();
    let model = LogisticModel::boundary_demo(kernel.schema());
    let decoder = ThresholdSpanDecoder::new(0.60);
    let pipeline = Pipeline::new(scanner, kernel, model, decoder);

    let spans = pipeline.run(text_view).expect("pipeline should succeed");
    let segments: Vec<&str> = spans
        .iter()
        .map(|span| &text[span.start.as_usize()..span.end.as_usize()])
        .collect();

    assert_eq!(segments, vec!["Alpha.", " Beta?", " Gamma!", " Delta."]);
}
