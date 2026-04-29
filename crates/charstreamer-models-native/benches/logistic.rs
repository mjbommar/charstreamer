use charstreamer_core::{
    BatchPredictor, DatasetView, FeatureMatrix, FitScratch, TrainablePredictor,
};
use charstreamer_models_native::{LogisticFitOptions, LogisticModel};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

fn synthetic_dataset(rows: usize, cols: usize) -> (FeatureMatrix<f32>, Vec<u8>) {
    let mut features = FeatureMatrix::<f32> {
        rows,
        cols,
        data: vec![0.0; rows * cols],
    };
    let mut labels = vec![0_u8; rows];

    for (row, label) in labels.iter_mut().enumerate().take(rows) {
        let mut activation = -0.15_f32;
        let row_slice = &mut features.data[row * cols..(row + 1) * cols];
        for (col, value) in row_slice.iter_mut().enumerate() {
            let raw = ((row * (col + 3) * 17 + col * 29) % 211) as f32 / 105.0 - 1.0;
            *value = raw;
            let weight = match col % 5 {
                0 => 1.25,
                1 => -0.95,
                2 => 0.55,
                3 => -0.35,
                _ => 0.15,
            };
            activation += raw * weight;
        }
        *label = u8::from(activation > 0.0);
    }

    (features, labels)
}

fn bench_logistic_predict(c: &mut Criterion) {
    let (features, _) = synthetic_dataset(16_384, 26);
    let weights = (0..features.cols)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.12)
        .collect();
    let model = LogisticModel::new(-0.2, weights);
    let mut out = vec![0.0_f32; features.rows];

    c.bench_function("logistic_predict_synthetic_16k_x_26", |b| {
        b.iter(|| {
            model
                .predict_into(features.as_view(), &mut out)
                .expect("prediction benchmark should succeed");
            criterion::black_box(&out);
        });
    });
}

fn bench_logistic_fit(c: &mut Criterion) {
    let (features, labels) = synthetic_dataset(4_096, 26);
    let dataset = DatasetView {
        features: features.as_view(),
        labels: &labels,
    };
    let options = LogisticFitOptions {
        epochs: 5,
        learning_rate: 0.04,
        batch_size: 256,
        ..LogisticFitOptions::default()
    };

    c.bench_function("logistic_fit_synthetic_4k_x_26", |b| {
        b.iter_batched_ref(
            FitScratch::default,
            |scratch| {
                let (model, report) = LogisticModel::fit(dataset, &options, scratch)
                    .expect("fit benchmark should succeed");
                criterion::black_box(model.weights().len());
                criterion::black_box(report.final_loss);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_logistic_predict, bench_logistic_fit);
criterion_main!(benches);
