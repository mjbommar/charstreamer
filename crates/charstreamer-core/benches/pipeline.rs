use charstreamer_core::{
    ContiguousSpanDecoder, LabeledSpan, Pipeline, PipelineWorkspace, ScoringPipeline,
    ScoringWorkspace, TextBytes, ThresholdSpanDecoder,
};
use charstreamer_kernels::{ByteSet256, ByteSetScanner, CompositeFeatureKernel, LineStartScanner};
use charstreamer_models_native::{LinearClassifierModel, LogisticModel};
use criterion::{Criterion, criterion_group, criterion_main};

fn demo_text() -> String {
    let sample = "Alpha. Beta? Gamma!\nDelta. Epsilon? Zeta!\n";
    sample.repeat(2_000)
}

fn mixed_format_text() -> String {
    let sample = "<root>\n<row id=\"1\">alpha</row>\n<row id=\"2\">beta</row>\n</root>\nname,score\nalice,10\nbob,20\n";
    sample.repeat(500)
}

fn bench_pipeline(c: &mut Criterion) {
    let text = demo_text();
    let text_view = TextBytes::from_utf8(&text);
    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!\n"));
    let kernel = CompositeFeatureKernel::boundary_demo();
    let model = LogisticModel::boundary_demo(kernel.schema());
    let decoder = ThresholdSpanDecoder::new(0.60);
    let pipeline = Pipeline::new(scanner, kernel, model, decoder);
    let mut workspace = PipelineWorkspace::<f32, f32>::default();
    let mut spans = Vec::new();

    c.bench_function("boundary_pipeline_ascii_demo", |b| {
        b.iter(|| {
            pipeline
                .run_into(text_view, &mut workspace, &mut spans)
                .expect("pipeline benchmark should succeed");
            criterion::black_box(&spans);
        });
    });
}

fn bench_region_pipeline(c: &mut Criterion) {
    let text = mixed_format_text();
    let text_view = TextBytes::from_utf8(&text);
    let scanner = LineStartScanner::new();
    let kernel = CompositeFeatureKernel::format_demo();
    let model = LinearClassifierModel::xml_csv_demo(kernel.schema());
    let decoder = ContiguousSpanDecoder::new(vec!["xml", "csv"]);
    let pipeline = ScoringPipeline::new(scanner, kernel, model, decoder);
    let mut workspace = ScoringWorkspace::<f32, f32>::default();
    let mut spans: Vec<LabeledSpan<&'static str>> = Vec::new();

    c.bench_function("region_pipeline_mixed_format_demo", |b| {
        b.iter(|| {
            pipeline
                .run_into(text_view, &mut workspace, &mut spans)
                .expect("region pipeline benchmark should succeed");
            criterion::black_box(&spans);
        });
    });
}

criterion_group!(benches, bench_pipeline, bench_region_pipeline);
criterion_main!(benches);
