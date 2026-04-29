# Model Artifacts

## Current State

`charstreamer` can build and ship a PyO3 wheel today, but the production Python
entry point still runs the native heuristic segmenter unless a usable model
runtime is wired in. The previous `v0.1.0` wheel did not contain a trained Burn
model. Future model-backed releases must pass the gates in this document before
publication.

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
  "engine": "burn_ndarray",
  "task": "semantic_segmentation",
  "created_at": "2026-04-29T00:00:00Z",
  "labels": ["sentence", "paragraph", "section", "dialogue", "list_item", "metadata"],
  "features": {
    "encoded_left": 7,
    "encoded_right": 7,
    "count_radius": 32
  },
  "thresholds": {
    "sentence.end": 0.55,
    "paragraph.end": 0.60
  },
  "metrics": {
    "validation_macro_f1": 0.95
  },
  "files": [
    {
      "path": "model.mpk",
      "bytes": 123456,
      "sha256": "..."
    }
  ]
}
```

Required fields:

- `format` must be `charstreamer.model-bundle.v1`.
- `name` must be `charstreamer-default` for the default package model.
- `engine` must identify the runtime, for example `burn_ndarray`.
- `files` must list every required payload file with relative paths.
- Every listed file should include `bytes` and `sha256` so package and runtime
  validation can catch partial or stale artifacts.

## Vendoring Into The Wheel

Recommended public release path:

```bash
python3 tools/model-artifacts/vendor_model.py \
  --require-burn \
  --archive-out dist/models/charstreamer-default-0.1.1.zip \
  path/to/charstreamer-default-0.1.1.zip

uvx maturin build --release \
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

Until Burn model loading/inference is connected to the production segmenter,
`require_model=True` intentionally fails for Burn artifacts. This prevents a
wheel from claiming to contain a useful model that the runtime cannot execute.
