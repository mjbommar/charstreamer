use charstreamer_core::{
    CandidateBuffer, FeatureAppender, FeatureKernel, FeatureMatrix, FeatureScratch, ScanRange,
    TextBytes,
};
use charstreamer_kernels::{
    ByteClass, ByteClassCountAppender, ByteSet256, ByteSetScanner, CompositeFeatureKernel,
    DirectionalUnicodeCategoryGroupCountAppender, UnicodeCategoryGroup,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn demo_text() -> String {
    let sample = "Alpha. Beta? Gamma!\nDelta. Epsilon? Zeta!\n";
    sample.repeat(2_000)
}

fn bench_scanner(c: &mut Criterion) {
    let text = demo_text();
    let text_view = TextBytes::from_utf8(&text);
    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!\n"));
    let mut candidates = CandidateBuffer::new();

    c.bench_function("byteset_scanner_ascii_demo", |b| {
        b.iter(|| {
            charstreamer_core::CandidateScanner::scan_into(
                &scanner,
                text_view,
                ScanRange::full(text_view),
                &mut candidates,
            );
            criterion::black_box(candidates.len());
        });
    });
}

fn bench_features(c: &mut Criterion) {
    let text = demo_text();
    let text_view = TextBytes::from_utf8(&text);
    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!\n"));
    let kernel = CompositeFeatureKernel::boundary_demo();
    let mut candidates = CandidateBuffer::new();
    charstreamer_core::CandidateScanner::scan_into(
        &scanner,
        text_view,
        ScanRange::full(text_view),
        &mut candidates,
    );
    let mut matrix = FeatureMatrix::<f32>::default();
    let mut scratch = FeatureScratch::default();

    c.bench_function("composite_feature_kernel_ascii_demo", |b| {
        b.iter(|| {
            matrix.resize_zeroed(candidates.len(), kernel.schema().total_dim());
            kernel
                .extract_into(
                    text_view,
                    candidates.as_slice(),
                    matrix.as_view_mut(),
                    &mut scratch,
                )
                .expect("feature benchmark should succeed");
            criterion::black_box(matrix.rows);
        });
    });
}

fn bench_byte_class_counts(c: &mut Criterion) {
    let text = demo_text();
    let text_view = TextBytes::from_utf8(&text);
    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!\n"));
    let appender = ByteClassCountAppender::new(
        "byte_class_counts",
        charstreamer_core::ByteWindowSpec::new(24, 24),
        vec![
            ByteClass::AsciiUpper,
            ByteClass::AsciiLower,
            ByteClass::AsciiDigit,
            ByteClass::AsciiWhitespace,
            ByteClass::AsciiPunctuation,
            ByteClass::LineBreak,
        ],
    );
    let mut candidates = CandidateBuffer::new();
    charstreamer_core::CandidateScanner::scan_into(
        &scanner,
        text_view,
        ScanRange::full(text_view),
        &mut candidates,
    );
    let mut matrix = FeatureMatrix::<f32>::default();
    let mut scratch = FeatureScratch::default();

    c.bench_function("byte_class_count_appender_ascii_demo", |b| {
        b.iter(|| {
            matrix.resize_zeroed(candidates.len(), appender.block().width);
            appender
                .append_into(
                    text_view,
                    candidates.as_slice(),
                    matrix.as_view_mut(),
                    &mut scratch,
                )
                .expect("byte-class-count benchmark should succeed");
            criterion::black_box(matrix.rows);
        });
    });
}

fn unicode_demo_text() -> String {
    let sample = "Ä.” βeta? Section 1.\nRésumé — Article 2.\n";
    sample.repeat(2_000)
}

fn bench_unicode_category_group_counts(c: &mut Criterion) {
    let text = unicode_demo_text();
    let text_view = TextBytes::from_utf8(&text);
    let scanner = ByteSetScanner::new(ByteSet256::from_bytes(b".?!\n"));
    let appender = DirectionalUnicodeCategoryGroupCountAppender::new(
        "unicode_category_group_counts",
        charstreamer_core::ByteWindowSpec::new(12, 12),
        vec![
            UnicodeCategoryGroup::L,
            UnicodeCategoryGroup::M,
            UnicodeCategoryGroup::N,
            UnicodeCategoryGroup::P,
            UnicodeCategoryGroup::S,
            UnicodeCategoryGroup::Z,
            UnicodeCategoryGroup::C,
        ],
    );
    let mut candidates = CandidateBuffer::new();
    charstreamer_core::CandidateScanner::scan_into(
        &scanner,
        text_view,
        ScanRange::full(text_view),
        &mut candidates,
    );
    let mut matrix = FeatureMatrix::<f32>::default();
    let mut scratch = FeatureScratch::default();

    c.bench_function("unicode_category_group_count_appender_utf8_demo", |b| {
        b.iter(|| {
            matrix.resize_zeroed(candidates.len(), appender.block().width);
            appender
                .append_into(
                    text_view,
                    candidates.as_slice(),
                    matrix.as_view_mut(),
                    &mut scratch,
                )
                .expect("unicode-category-group benchmark should succeed");
            criterion::black_box(matrix.rows);
        });
    });
}

criterion_group!(
    benches,
    bench_scanner,
    bench_features,
    bench_byte_class_counts,
    bench_unicode_category_group_counts
);
criterion_main!(benches);
