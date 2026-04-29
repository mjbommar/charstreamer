# Model-Backed Release Todo

## Goal

Ship a PyPI wheel whose default hello-world path runs a trained model, not a
silent heuristic fallback.

## Tasks

1. Move the selected semantic model architecture out of experiment examples and
   into a production crate.
2. Add Burn serialization for the trained model record using a stable recorder
   format, preferably named msgpack or compact binary.
3. Persist thresholds, feature configuration, label schema, training data
   fingerprint, validation metrics, and model payload hashes into
   `manifest.json`.
4. Add Rust inference APIs that load the bundle and produce the same standoff
   span contract as `CombinedSegmenter`.
5. Expose the model-backed segmenter through PyO3 and make Python
   `Segmenter.default(require_model=True)` succeed for the vendored bundle.
6. Add parity tests proving model bundle load, inference, and rendering work in
   Rust and Python.
7. Build the wheel with `vendor_model.py`, validate it with
   `check_wheel_model.py`, and run the offline hello-world smoke test.
8. Publish only after the release workflow passes with the real model artifact.

## Non-Negotiable Gate

Do not publish another model-backed public release unless:

```bash
python3 tools/model-artifacts/check_wheel_model.py --require-burn dist/charstreamer-*.whl
CHARSTREAMER_AUTO_DOWNLOAD=0 uv run --isolated --with ./dist/charstreamer-*.whl \
  python -c "import charstreamer; charstreamer.model_info(allow_download=False, require_model=True)"
```

both pass.
