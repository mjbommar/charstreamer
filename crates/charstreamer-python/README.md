# CharStreamer Python

`charstreamer` provides Python access to the Rust CharStreamer segmentation
engine through a PyO3 extension module.

This package exposes the Rust model artifact loader and model-backed
segmentation runtime. If no supported model is available, annotation fails
instead of synthesizing semantic labels from hard-coded rules.

The `0.1.1` vendored model emits only `sentence` spans. Other schema labels
remain reserved for future trained runtimes.

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
print(segmenter.model_info())
annotation = segmenter.annotate(text)

print(annotation["spans"])
print(annotation["tagged"])
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

The vendored `0.1.1` bundle is a sentence-end model. It is useful for sentence
boundary text, but it is not a full semantic span/IOB model for headings,
metadata, lists, or dialogue.

The project is an early development release. APIs may change before a stable
`1.0` release.

Full documentation and Rust source are available at:

https://github.com/mjbommar/charstreamer
