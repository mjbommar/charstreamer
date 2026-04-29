use std::error::Error;

use charstreamer_core::{
    ContiguousSpanDecoder, LabeledSpan, ScoringPipeline, ScoringWorkspace, TextBytes,
};
use charstreamer_kernels::{CompositeFeatureKernel, LineStartScanner};
use charstreamer_models_native::LinearClassifierModel;

fn main() -> Result<(), Box<dyn Error>> {
    let text = "<root>\n<row id=\"1\">alpha</row>\n<row id=\"2\">beta</row>\n</root>\nname,score\nalice,10\nbob,20";
    let text_view = TextBytes::from_utf8(text);

    let scanner = LineStartScanner::new();
    let kernel = CompositeFeatureKernel::format_demo();
    let model = LinearClassifierModel::xml_csv_demo(kernel.schema());
    let decoder = ContiguousSpanDecoder::new(vec!["xml", "csv"]);
    let pipeline = ScoringPipeline::new(scanner, kernel, model, decoder);

    let mut workspace = ScoringWorkspace::<f32, f32>::default();
    let mut spans: Vec<LabeledSpan<&'static str>> = Vec::new();
    pipeline.run_into(text_view, &mut workspace, &mut spans)?;

    let positions: Vec<usize> = workspace
        .positions
        .positions()
        .iter()
        .map(|position| position.as_usize())
        .collect();
    println!("positions: {positions:?}");
    println!("regions:");
    for span in &spans {
        let region_text = &text[span.span.start.as_usize()..span.span.end.as_usize()];
        let preview = region_text.replace('\n', "\\n");
        println!("- {} {:.3} {:?}", span.label, span.score, preview);
    }

    Ok(())
}
