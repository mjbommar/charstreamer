# CharStreamer Python

`charstreamer` provides Python access to the Rust CharStreamer segmentation
engine through a PyO3 extension module.

This package exposes the Rust model artifact loader and model-backed
segmentation runtime. If no supported model is available, annotation fails
instead of synthesizing semantic labels from hard-coded rules.

The vendored `0.1.4` bundle emits model-backed `sentence`, `paragraph`,
`metadata`, `section`, and `list_item` spans. `dialogue` remains reserved until
there is a balanced dialogue training set.

## Install

```bash
pip install charstreamer
```

## Example

```python
import charstreamer

text = """# Background
The court reviewed the invoice. The shipment was late. Notice was timely."""

segmenter = charstreamer.Segmenter.default()
print(segmenter.model_info().runtime)
annotation = segmenter.annotate(text)

print(annotation.spans)
print(annotation.tagged)
```

The public Python wrapper returns typed immutable dataclasses:
`ModelInfo`, `Annotation`, `Span`, and `BenchmarkResult`. For JSON output or
legacy integrations, call `.to_dict()` or use methods such as
`segmenter.annotate_dict(text)`.

## Performance

On the current Linux x86_64 release-wheel benchmark, the combined
sentence+semantic segmenter runs at roughly `34-35 MiB/s` end-to-end on a long
UTF-8 document. This includes model inference, span decoding, and tagged
rendering for the default Burn model bundle.

Measure local throughput with:

```python
result = segmenter.benchmark(text, iterations=10)
print(result.mib_per_second)
print(result.chars_per_second)
```

If a default model is vendored into the wheel, `Segmenter.default()` loads it
from package data. If not, it checks the local cache and then the GitHub release
model URL unless `CHARSTREAMER_AUTO_DOWNLOAD=0` is set. To assert model
availability during startup:

```python
charstreamer.model_info(allow_download=False, require_model=True)
segmenter = charstreamer.Segmenter.default(require_model=True)
```

Model-backed release wheels must include
`charstreamer/models/default/manifest.json` plus the referenced Burn payload.

The vendored `0.1.4` bundle combines a sentence-boundary model with a semantic
structure model. It is an early model-backed release, not a final semantic
span/IOB model, and quality should be evaluated against task-specific data
before production use.

The project is an early development release. APIs may change before a stable
`1.0` release.

Full documentation and Rust source are available at:

https://github.com/mjbommar/charstreamer
