# Experiment Specs

This directory holds machine-readable manifests for repeatable `charstreamer`
experiments.

## Purpose

Use these specs when you need:

- config-defined feature pipelines
- reproducible training and evaluation runs
- controlled docking of prior vs current implementations
- a record of exactly which parameters changed between runs

The current runners live in:

- [run_experiment_spec.rs](../crates/charstreamer-experiments/examples/run_experiment_spec.rs)
- [run_parity_spec.rs](../crates/charstreamer-experiments/examples/run_parity_spec.rs)
- [run_experiment_sweep.rs](../crates/charstreamer-experiments/examples/run_experiment_sweep.rs)

## Layout

```text
specs/
  README.md
  parity/
    *.json
  experiments/
    *.json
```

- `parity/`: feature-row parity checks against the legacy Python extractor
- `experiments/`: train/eval manifests over one corpus split and one model spec

Current useful manifests:

- `parity/charboundary-legacy-sample.json`: legacy Rust/Python feature-row parity
- `experiments/charboundary-small-logistic.json`: legacy feature spec with native logistic
- `experiments/charboundary-small-python-rf.json`: legacy feature spec with exact sklearn RF parameters
- `experiments/charboundary-small-python-rf-full.json`: full-corpus exact-legacy docking spec
- `experiments/current-legal-logistic.json`: current Rust legal feature stack with native logistic
- `experiments/current-legal-logistic-local-structure.json`: native logistic with local structure features
- `experiments/current-legal-burn-shallow-mlp-directional-full.json`: full-corpus Burn shallow MLP on the current strongest directional feature stack
- `experiments/current-legal-burn-deep-mlp-directional-full.json`: full-corpus Burn deep MLP on the same directional feature stack
- `experiments/current-legal-burn-window-cnn-directional-full.json`: full-corpus Burn window CNN using the encoded window plus side features
- `experiments/current-legal-burn-window-gru-directional-full.json`: full-corpus Burn GRU over the encoded window plus side features
- `experiments/current-legal-burn-window-lstm-directional-full.json`: full-corpus Burn LSTM over the encoded window plus side features

The active external Rust model backend is Burn. Historical SmartCore and Linfa
specs were removed with those backend crates; old benchmark results remain in
`docs/results.md` only as archived comparisons.

## Path resolution

Paths inside specs may be relative.

- relative paths are resolved relative to the spec file's parent directory
- absolute paths are used as-is

This lets specs refer to sibling repositories such as `charboundary/` and
`legal-sentence-paper/` without hardcoding one machine-specific absolute prefix.

## Operational rules

- run a parity spec before claiming legacy feature compatibility
- when docking prior vs current models, hold dataset, split, scanner, window,
  negative sampling, and evaluation methodology constant unless one of those is
  the explicit experimental variable
- use `null` for `dataset_options.seed` and sklearn `random_state` when you want
  exact-legacy stochastic behavior rather than deterministic replay
- store one manifest per experiment configuration, not one “do everything”
  manifest with runtime flags

## Example commands

```bash
cargo run -p charstreamer-experiments --example run_parity_spec -- \
  specs/parity/charboundary-legacy-sample.json

cargo run --release -p charstreamer-experiments --example run_experiment_spec -- \
  specs/experiments/charboundary-small-logistic.json \
  /tmp/charboundary-small-logistic-report.json

cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- \
  specs/experiments/charboundary-small-logistic.json \
  specs/experiments/current-legal-logistic.json

cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- \
  specs/experiments/current-legal-burn-shallow-mlp-directional-full.json \
  specs/experiments/current-legal-burn-deep-mlp-directional-full.json \
  specs/experiments/current-legal-burn-window-cnn-directional-full.json
```
