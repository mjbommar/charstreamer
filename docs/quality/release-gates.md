# Release Gates

## Goal

Define explicit go/no-go rules so the library is not “done” based on vibes.

## Gate classes

### G0: Build gate

Must pass:

- workspace builds
- docs build
- lint and format checks

### G1: Correctness gate

Must pass:

- unit tests
- property tests
- portable vs optimized primitive differential tests where optimized paths exist
- serialization round-trip
- chunked vs non-chunked equivalence

### G2: API gate

Must pass:

- public Rust API documented
- Python API documented if present
- byte-vs-character semantics explicitly documented

### G3: Quality gate

Must pass:

- no known crashers on supported platforms
- deterministic fixed-seed training for native models
- error messages for malformed inputs are tested
- model artifacts validate before packaging
- the default Python path reports whether it is model-backed or heuristic

### G4: Performance gate

Must pass:

- protected primitive benchmarks have no unexplained regressions
- no benchmark regression beyond agreed tolerance
- required throughput targets met or signed off with explanation
- Python overhead within documented range

## Phase gates

### Phase 00 exit

- G0 only

### Phase 01 exit

- G0
- G1 for core pipeline
- protected primitive benchmarks exist for scanner and feature appenders

### Phase 02 exit

- G0
- G1
- G2 for Rust training APIs
- G3 for native model artifacts

### Phase 03 exit

- G0
- G1
- G2 for Python APIs
- G3 for Rust/Python parity

### Phase 04 exit / release candidate

- all gates

## Release candidate checklist

- native pipeline passes corpus suite
- native models serialize and reload cleanly
- default model bundle is present in the wheel or attached to the release
- `charstreamer.model_info(allow_download=False, require_model=True)` succeeds
- benchmark report is current
- build diary includes release hardening summary
- decision log includes any deviations from original architecture

## Model-backed release checklist

- Training writes a Burn model record plus thresholds, label schema, feature
  configuration, and validation metrics.
- `tools/model-artifacts/vendor_model.py --require-burn` validates the bundle
  and vendors it into the Python package before wheel build.
- `tools/model-artifacts/check_wheel_model.py --require-burn dist/*.whl`
  passes on the built wheel.
- The wheel smoke test runs the hello-world example with
  `CHARSTREAMER_AUTO_DOWNLOAD=0` so bundled model loading is proven offline.
- The release workflow attaches both the wheel and normalized model zip.
