use std::error::Error;

use charstreamer_core::{
    FeatureKernel, Pipeline, PipelineWorkspace, TextBytes, ThresholdSpanDecoder,
};
use charstreamer_kernels::{ByteSet256, ByteSetScanner, CompositeFeatureKernel};
use charstreamer_models_native::LogisticModel;

fn main() -> Result<(), Box<dyn Error>> {
    let text = "Alpha. Beta? Gamma! Delta.";
    let text_view = TextBytes::from_utf8(text);

    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!"));
    let kernel = CompositeFeatureKernel::boundary_demo();
    let model = LogisticModel::boundary_demo(kernel.schema());
    let decoder = ThresholdSpanDecoder::new(0.60);
    let pipeline = Pipeline::new(scanner, kernel, model, decoder);

    let mut workspace = PipelineWorkspace::<f32, f32>::default();
    let mut spans = Vec::new();

    pipeline.scan_candidates(text_view, &mut workspace.candidates);
    let candidate_positions: Vec<usize> = workspace
        .candidates
        .positions()
        .iter()
        .map(|position| position.as_usize())
        .collect();
    println!("candidates: {candidate_positions:?}");

    workspace.features.resize_zeroed(
        workspace.candidates.len(),
        pipeline.kernel().schema().total_dim(),
    );
    pipeline.kernel().extract_into(
        text_view,
        workspace.candidates.as_slice(),
        workspace.features.as_view_mut(),
        &mut workspace.feature_scratch,
    )?;

    workspace
        .scores
        .resize_fill(workspace.features.rows, 0.0_f32);
    charstreamer_core::BatchPredictor::predict_into(
        &LogisticModel::boundary_demo(pipeline.kernel().schema()),
        workspace.features.as_view(),
        &mut workspace.scores.data,
    )?;

    pipeline.decode(
        text_view,
        &workspace.candidates,
        &workspace.scores.data,
        &mut spans,
    )?;

    for ((position, score), span) in workspace
        .candidates
        .positions()
        .iter()
        .zip(workspace.scores.data.iter())
        .zip(spans.iter())
    {
        println!(
            "candidate @ {:>2} score {:.3} sample span {:?}",
            position.as_usize(),
            score,
            span
        );
    }

    println!("segments:");
    for span in &spans {
        let segment = &text[span.start.as_usize()..span.end.as_usize()];
        println!("- {:?}", segment);
    }

    Ok(())
}
