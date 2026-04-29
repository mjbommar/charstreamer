# Model-Backed Release Todo

## Goal

Ship a PyPI wheel whose default hello-world path runs a trained model, not a
silent heuristic fallback.

## Tasks

1. [x] Move the selected sentence-boundary model architecture out of experiment examples and
   into a production crate.
2. [x] Add Burn serialization for the trained model record using a stable recorder
   format, preferably named msgpack or compact binary.
3. [x] Persist thresholds, feature configuration, validation metrics, and
   model payload metadata into `manifest.json`.
4. [x] Add Rust inference APIs that load the bundle and produce the same standoff
   span contract as `CombinedSegmenter`.
5. [x] Expose the model-backed segmenter through PyO3 and make Python
   `Segmenter.default(require_model=True)` succeed for the vendored bundle.
6. [ ] Add parity tests proving model bundle load, inference, and rendering work in
   Rust and Python.
7. [x] Build the wheel with `vendor_model.py`, validate it with
   `check_wheel_model.py`, and run the offline hello-world smoke test.
8. [ ] Publish only after the release workflow passes with the real model artifact.

## Implemented Slice

- Runtime engine: `burn_shallow_mlp_sentence_v1`.
- Feature config: `encoded_left=15`, `encoded_right=15`, `count_radius=64`,
  `feature_dim=109`, `hidden_dim=256`.
- Training command: `cargo run --release -p charstreamer-segmentation --example train_sentence_burn`.
- Current validation metrics: precision `0.724`, recall `0.826`, F1 `0.767`,
  threshold `0.36`.
- Wheel gate passed locally for `dist/charstreamer-0.1.1-cp39-abi3-manylinux_2_34_x86_64.whl`.

The remaining release risk is data/model quality, not packaging. The current
model is sufficient to prove the production Burn artifact path, but broader
semantic segmentation should get its own multi-label model bundle instead of
overloading this sentence-boundary slice.

## Non-Negotiable Gate

Do not publish another model-backed public release unless:

```bash
python3 tools/model-artifacts/check_wheel_model.py --require-burn dist/charstreamer-*.whl
CHARSTREAMER_AUTO_DOWNLOAD=0 uv run --isolated --with ./dist/charstreamer-*.whl \
  python -c "import charstreamer; charstreamer.model_info(allow_download=False, require_model=True)"
```

both pass.
