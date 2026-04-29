# Model Artifacts

## Current State

`v0.1.0` on PyPI was heuristic-only. `v0.1.1` is the first model-backed release
target: the Python wheel vendors a Burn shallow-MLP sentence-boundary bundle and
loads it automatically from `charstreamer.Segmenter.default()`.

The first production bundle is intentionally narrow. It replaces sentence
boundary scoring with Burn inference while retaining deterministic native logic
for structural spans (`paragraph`, `metadata`, `section`, `list_item`, and
`dialogue`). That gives us a real serialized model path without blocking the
release on every semantic label.

## Required Artifact Layout

The default model bundle is a zip archive or directory named:

```text
charstreamer-default-<package-version>.zip
```

The archive must contain `manifest.json` at the root, or inside one top-level
directory. The manifest format is:

```json
{
  "format": "charstreamer.model-bundle.v1",
  "name": "charstreamer-default",
  "version": "0.1.1",
  "engine": "burn_shallow_mlp_sentence_v1",
  "task": "sentence_boundary",
  "features": {
    "encoded_left": 15,
    "encoded_right": 15,
    "count_radius": 64,
    "feature_dim": 109,
    "hidden_dim": 256
  },
  "thresholds": {
    "sentence.end": 0.36
  },
  "metrics": {
    "validation": {
      "precision": 0.724,
      "recall": 0.826,
      "f1": 0.767
    }
  },
  "files": [
    {
      "path": "sentence_boundary.mpk",
      "role": "sentence_boundary",
      "bytes": 114134
    }
  ]
}
```

Required fields:

- `format` must be `charstreamer.model-bundle.v1`.
- `name` must be `charstreamer-default` for the default package model.
- `engine` must identify the runtime, for example
  `burn_shallow_mlp_sentence_v1`.
- `files` must list every required payload file with relative paths.
- Every listed file should include `bytes` and `sha256` so package and runtime
  validation can catch partial or stale artifacts.

The first bundle records byte counts. Adding `sha256` is still recommended for
release artifacts and is supported by both validation scripts and the Python
runtime.

## Vendoring Into The Wheel

Recommended public release path:

```bash
cargo run --release -p charstreamer-segmentation --example train_sentence_burn -- \
  --input data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.jsonl \
  --out target/model/charstreamer-default-0.1.1 \
  --hidden-dim 256 \
  --epochs 80 \
  --batch-size 512 \
  --learning-rate 0.0005 \
  --seed 17 \
  --encoded-left 15 \
  --encoded-right 15 \
  --count-radius 64 \
  --threshold 0.36

python3 tools/model-artifacts/vendor_model.py \
  --require-burn \
  --archive-out dist/models/charstreamer-default-0.1.1.zip \
  target/model/charstreamer-default-0.1.1

uv run --with 'maturin[patchelf]' maturin build \
  --release \
  --manifest-path crates/charstreamer-python/Cargo.toml \
  --out dist

python3 tools/model-artifacts/check_wheel_model.py \
  --require-burn \
  dist/charstreamer-*.whl
```

`vendor_model.py` copies the validated bundle into:

```text
crates/charstreamer-python/python/charstreamer/models/default/
```

That directory is ignored by git except for `.gitkeep`, so generated model files
do not become source files by accident. The release workflow can also receive a
`model_artifact_url` input, download the zip, validate it, vendor it, and attach
the normalized zip to the GitHub release.

## Runtime Resolution

Python `charstreamer.Segmenter.default()` resolves model artifacts in this
order:

1. `CHARSTREAMER_MODEL_PATH`, if set to a local bundle directory.
2. Bundled wheel data at `charstreamer/models/default/`.
3. Local cache at `CHARSTREAMER_MODEL_CACHE/default` or
   `~/.cache/charstreamer/models/default`.
4. GitHub release download URL, unless disabled.
5. Native heuristic fallback.

Only engines in the Python wrapper's supported-engine set are treated as
model-backed. Unsupported or missing bundles fall back only when the caller does
not pass `require_model=True`.

Environment variables:

- `CHARSTREAMER_MODEL_PATH`: local model bundle directory.
- `CHARSTREAMER_MODEL_URL`: explicit model zip URL.
- `CHARSTREAMER_MODEL_CACHE`: cache root for downloaded models.
- `CHARSTREAMER_MODEL_TIMEOUT`: download timeout in seconds, default `10`.
- `CHARSTREAMER_AUTO_DOWNLOAD`: set to `0`, `false`, `no`, or `off` to disable
  automatic download attempts.

The default URL is:

```text
https://github.com/mjbommar/charstreamer/releases/download/v<version>/charstreamer-default-<version>.zip
```

## Release Gate

Model-backed public releases must prove all of the following:

- The wheel contains `charstreamer/models/default/manifest.json`.
- The manifest validates and uses a Burn-backed engine.
- All manifest payload hashes and byte counts match.
- `charstreamer.model_info(allow_download=False, require_model=True)` succeeds.
- The default hello-world example runs without requiring network access.
- The GitHub release also attaches the model zip for users that want to inspect
  or cache it separately.

`require_model=True` intentionally fails when no supported bundle is resolved.
This prevents silently shipping or deploying a package that looks model-backed
but is running only the heuristic fallback.
