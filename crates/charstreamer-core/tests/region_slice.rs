use charstreamer_core::{ContiguousSpanDecoder, LabeledSpan, ScoringPipeline, TextBytes};
use charstreamer_kernels::{CompositeFeatureKernel, LineStartScanner};
use charstreamer_models_native::LinearClassifierModel;

#[test]
fn region_pipeline_detects_xml_to_csv_switch() {
    let text = "<root>\n<row id=\"1\">alpha</row>\n<row id=\"2\">beta</row>\n</root>\nname,score\nalice,10\nbob,20";
    let text_view = TextBytes::from_utf8(text);

    let scanner = LineStartScanner::new();
    let kernel = CompositeFeatureKernel::format_demo();
    let model = LinearClassifierModel::xml_csv_demo(kernel.schema());
    let decoder = ContiguousSpanDecoder::new(vec!["xml", "csv"]);
    let pipeline = ScoringPipeline::new(scanner, kernel, model, decoder);

    let spans: Vec<LabeledSpan<&'static str>> =
        pipeline.run(text_view).expect("pipeline should succeed");
    let labeled_regions: Vec<(&str, &str)> = spans
        .iter()
        .map(|span| {
            (
                span.label,
                &text[span.span.start.as_usize()..span.span.end.as_usize()],
            )
        })
        .collect();

    assert_eq!(
        labeled_regions,
        vec![
            (
                "xml",
                "<root>\n<row id=\"1\">alpha</row>\n<row id=\"2\">beta</row>\n</root>\n"
            ),
            ("csv", "name,score\nalice,10\nbob,20"),
        ]
    );
}
