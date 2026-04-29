# CharStreamer Python

`charstreamer` provides Python access to the Rust CharStreamer segmentation
engine through a PyO3 extension module.

This first public wheel focuses on fast semantic text segmentation:

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
annotation = segmenter.annotate(text)

print(annotation["spans"])
print(annotation["tagged"])
```

The project is an early development release. APIs may change before a stable
`1.0` release.

Full documentation and Rust source are available at:

https://github.com/mjbommar/charstreamer
