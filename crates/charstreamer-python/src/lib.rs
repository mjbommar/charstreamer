use std::time::Instant;
use std::{collections::HashMap, collections::HashSet};

use charstreamer_segmentation::{
    AnnotationSpan, CombinedSegmenter, Label, SegmenterConfig, render_spans,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass(name = "SegmenterConfig")]
#[derive(Clone, Debug)]
struct PySegmenterConfig {
    #[pyo3(get, set)]
    include_paragraphs: bool,
    #[pyo3(get, set)]
    include_sentences: bool,
    #[pyo3(get, set)]
    include_metadata: bool,
    #[pyo3(get, set)]
    include_sections: bool,
    #[pyo3(get, set)]
    include_list_items: bool,
    #[pyo3(get, set)]
    include_dialogue: bool,
    #[pyo3(get, set)]
    suppress_sentences_in_structural_spans: bool,
    #[pyo3(get, set)]
    min_span_bytes: usize,
}

#[pymethods]
impl PySegmenterConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        include_paragraphs=true,
        include_sentences=true,
        include_metadata=true,
        include_sections=true,
        include_list_items=true,
        include_dialogue=true,
        suppress_sentences_in_structural_spans=true,
        min_span_bytes=1
    ))]
    fn new(
        include_paragraphs: bool,
        include_sentences: bool,
        include_metadata: bool,
        include_sections: bool,
        include_list_items: bool,
        include_dialogue: bool,
        suppress_sentences_in_structural_spans: bool,
        min_span_bytes: usize,
    ) -> Self {
        Self {
            include_paragraphs,
            include_sentences,
            include_metadata,
            include_sections,
            include_list_items,
            include_dialogue,
            suppress_sentences_in_structural_spans,
            min_span_bytes,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        SegmenterConfig::default().into()
    }
}

impl From<SegmenterConfig> for PySegmenterConfig {
    fn from(value: SegmenterConfig) -> Self {
        Self {
            include_paragraphs: value.include_paragraphs,
            include_sentences: value.include_sentences,
            include_metadata: value.include_metadata,
            include_sections: value.include_sections,
            include_list_items: value.include_list_items,
            include_dialogue: value.include_dialogue,
            suppress_sentences_in_structural_spans: value.suppress_sentences_in_structural_spans,
            min_span_bytes: value.min_span_bytes,
        }
    }
}

impl From<&PySegmenterConfig> for SegmenterConfig {
    fn from(value: &PySegmenterConfig) -> Self {
        Self {
            include_paragraphs: value.include_paragraphs,
            include_sentences: value.include_sentences,
            include_metadata: value.include_metadata,
            include_sections: value.include_sections,
            include_list_items: value.include_list_items,
            include_dialogue: value.include_dialogue,
            suppress_sentences_in_structural_spans: value.suppress_sentences_in_structural_spans,
            min_span_bytes: value.min_span_bytes,
        }
    }
}

#[pyclass(name = "Segmenter")]
#[derive(Clone, Debug)]
struct PySegmenter {
    inner: CombinedSegmenter,
}

#[pymethods]
impl PySegmenter {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PySegmenterConfig>) -> Self {
        let config = config.map_or_else(SegmenterConfig::default, SegmenterConfig::from);
        Self {
            inner: CombinedSegmenter::new(config),
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: CombinedSegmenter::default(),
        }
    }

    fn spans<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        spans_to_pylist(py, text, &self.inner.spans(text))
    }

    fn annotate<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let annotation = self.inner.annotate(text);
        let dict = PyDict::new(py);
        dict.set_item("tagged", annotation.tagged)?;
        dict.set_item("spans", spans_to_pylist(py, text, &annotation.spans)?)?;
        Ok(dict)
    }

    fn tagged(&self, text: &str) -> String {
        self.inner.annotate(text).tagged
    }

    #[pyo3(signature = (text, iterations=10))]
    fn benchmark<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iterations = iterations.max(1);
        let started = Instant::now();
        let mut span_count = 0_usize;
        let mut tagged_bytes = 0_usize;
        for _ in 0..iterations {
            let annotation = self.inner.annotate(text);
            span_count = annotation.spans.len();
            tagged_bytes = annotation.tagged.len();
        }
        let seconds = started.elapsed().as_secs_f64();
        let bytes = text.len().saturating_mul(iterations);
        let chars = text.chars().count().saturating_mul(iterations);
        let dict = PyDict::new(py);
        dict.set_item("iterations", iterations)?;
        dict.set_item("input_bytes", text.len())?;
        dict.set_item("input_chars", text.chars().count())?;
        dict.set_item("processed_bytes", bytes)?;
        dict.set_item("processed_chars", chars)?;
        dict.set_item("seconds", seconds)?;
        dict.set_item(
            "bytes_per_second",
            bytes as f64 / seconds.max(f64::MIN_POSITIVE),
        )?;
        dict.set_item(
            "chars_per_second",
            chars as f64 / seconds.max(f64::MIN_POSITIVE),
        )?;
        dict.set_item(
            "mib_per_second",
            bytes as f64 / seconds.max(f64::MIN_POSITIVE) / (1024.0 * 1024.0),
        )?;
        dict.set_item("span_count", span_count)?;
        dict.set_item("tagged_bytes", tagged_bytes)?;
        Ok(dict)
    }
}

#[pyfunction]
fn annotate<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    PySegmenter::default().annotate(py, text)
}

#[pyfunction]
fn spans<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
    PySegmenter::default().spans(py, text)
}

#[pyfunction]
fn tagged(text: &str) -> String {
    PySegmenter::default().tagged(text)
}

#[pyfunction]
fn render<'py>(
    py: Python<'py>,
    text: &str,
    spans: Vec<(String, usize, usize, Option<f32>)>,
) -> PyResult<String> {
    let mut offsets = HashSet::new();
    for (_, start, end, _) in &spans {
        offsets.insert(*start);
        offsets.insert(*end);
    }
    let char_to_byte = build_char_to_byte_map(text, offsets)?;
    let converted = spans
        .into_iter()
        .map(|(label, start, end, score)| {
            Ok(AnnotationSpan {
                label: parse_label(&label)?,
                start: *char_to_byte.get(&start).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid character offset `{start}`"
                    ))
                })?,
                end: *char_to_byte.get(&end).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid character offset `{end}`"
                    ))
                })?,
                score: score.unwrap_or(1.0),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let _ = py;
    Ok(render_spans(text, &converted))
}

#[pyfunction]
fn render_bytes<'py>(
    py: Python<'py>,
    text: &str,
    spans: Vec<(String, usize, usize, Option<f32>)>,
) -> PyResult<String> {
    let converted = spans
        .into_iter()
        .map(|(label, start, end, score)| {
            Ok(AnnotationSpan {
                label: parse_label(&label)?,
                start,
                end,
                score: score.unwrap_or(1.0),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let _ = py;
    Ok(render_spans(text, &converted))
}

fn spans_to_pylist<'py>(
    py: Python<'py>,
    text: &str,
    spans: &[AnnotationSpan],
) -> PyResult<Bound<'py, PyList>> {
    let mut offsets = HashSet::with_capacity(spans.len().saturating_mul(2));
    for span in spans {
        offsets.insert(span.start);
        offsets.insert(span.end);
    }
    let byte_to_char = build_byte_to_char_map(text, offsets)?;
    let list = PyList::empty(py);
    for span in spans {
        let dict = PyDict::new(py);
        dict.set_item("label", span.label.as_str())?;
        dict.set_item("start", byte_to_char[&span.start])?;
        dict.set_item("end", byte_to_char[&span.end])?;
        dict.set_item("start_byte", span.start)?;
        dict.set_item("end_byte", span.end)?;
        dict.set_item("score", span.score)?;
        list.append(dict)?;
    }
    Ok(list)
}

fn parse_label(label: &str) -> PyResult<Label> {
    match label {
        "paragraph" => Ok(Label::Paragraph),
        "metadata" => Ok(Label::Metadata),
        "section" => Ok(Label::Section),
        "list_item" => Ok(Label::ListItem),
        "dialogue" => Ok(Label::Dialogue),
        "sentence" => Ok(Label::Sentence),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown label `{other}`"
        ))),
    }
}

fn build_byte_to_char_map(text: &str, offsets: HashSet<usize>) -> PyResult<HashMap<usize, usize>> {
    let mut sorted = offsets.into_iter().collect::<Vec<_>>();
    sorted.sort_unstable();
    sorted.dedup();

    let mut result = HashMap::with_capacity(sorted.len());
    let mut cursor = 0_usize;
    let mut char_index = 0_usize;

    for (byte_index, ch) in text.char_indices() {
        while cursor < sorted.len() && sorted[cursor] == byte_index {
            result.insert(sorted[cursor], char_index);
            cursor += 1;
        }
        char_index += 1;
        let _ = ch;
    }

    while cursor < sorted.len() && sorted[cursor] == text.len() {
        result.insert(sorted[cursor], char_index);
        cursor += 1;
    }

    if cursor != sorted.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "byte offset `{}` is not a UTF-8 character boundary for this text",
            sorted[cursor]
        )));
    }

    Ok(result)
}

fn build_char_to_byte_map(text: &str, offsets: HashSet<usize>) -> PyResult<HashMap<usize, usize>> {
    let mut sorted = offsets.into_iter().collect::<Vec<_>>();
    sorted.sort_unstable();
    sorted.dedup();

    let mut result = HashMap::with_capacity(sorted.len());
    let mut cursor = 0_usize;
    let mut char_index = 0_usize;

    for (byte_index, _) in text.char_indices() {
        while cursor < sorted.len() && sorted[cursor] == char_index {
            result.insert(sorted[cursor], byte_index);
            cursor += 1;
        }
        char_index += 1;
    }

    while cursor < sorted.len() && sorted[cursor] == char_index {
        result.insert(sorted[cursor], text.len());
        cursor += 1;
    }

    if cursor != sorted.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "character offset `{}` is out of range for this text",
            sorted[cursor]
        )));
    }

    Ok(result)
}

#[pymodule]
fn charstreamer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySegmenterConfig>()?;
    module.add_class::<PySegmenter>()?;
    module.add_function(wrap_pyfunction!(annotate, module)?)?;
    module.add_function(wrap_pyfunction!(spans, module)?)?;
    module.add_function(wrap_pyfunction!(tagged, module)?)?;
    module.add_function(wrap_pyfunction!(render, module)?)?;
    module.add_function(wrap_pyfunction!(render_bytes, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
