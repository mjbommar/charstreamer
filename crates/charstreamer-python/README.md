# CharStreamer Python

`charstreamer` provides Python access to the Rust CharStreamer segmentation
engine through a PyO3 extension module.

This package exposes the Rust segmentation engine and model artifact loader.
The `v0.1.0` wheel did not ship a trained Burn model; current source reports
that state explicitly and falls back to the native heuristic segmenter until a
model-backed release is cut.

The segmentation labels are:

- paragraphs
- sentences
- metadata-like lines
- headings/sections
- list items
- dialogue spans

## Install

```bash
pip install charstreamer
```

## Example

```python
import charstreamer

text = """# Background
The court reviewed the invoice. The shipment was late.

- Notice was timely.
- Damages were limited.
"""

segmenter = charstreamer.Segmenter.default()
print(segmenter.model_info())
annotation = segmenter.annotate(text)

print(annotation["spans"])
print(annotation["tagged"])
```

If a default model is vendored into the wheel, `Segmenter.default()` loads it
from package data. If not, it checks the local cache and then the GitHub release
model URL unless `CHARSTREAMER_AUTO_DOWNLOAD=0` is set. To fail instead of
falling back to heuristics:

```python
charstreamer.model_info(allow_download=False, require_model=True)
segmenter = charstreamer.Segmenter.default(require_model=True)
```

Model-backed release wheels must include
`charstreamer/models/default/manifest.json` plus the referenced Burn payload.

The project is an early development release. APIs may change before a stable
`1.0` release.

Full documentation and Rust source are available at:

https://github.com/mjbommar/charstreamer
