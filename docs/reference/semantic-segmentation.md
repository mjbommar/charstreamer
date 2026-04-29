# Semantic Segmentation

This document expands `charstreamer` from sentence-boundary detection into a
general document-segmentation and weak-labeling system.

The core point is simple:

- sentence boundaries are one task family
- paragraph, section, dialogue, list, metadata, and entity spans are adjacent
  task families
- the library should treat all of them as structured span or boundary problems

## Why This Matters

Sentence boundaries alone are too narrow for the kinds of documents we care
about.

Real production pipelines also need:

- paragraph segmentation
- section and heading segmentation
- dialogue and speaker-turn segmentation
- list-item and enumerator segmentation
- front-matter and boilerplate detection
- citation-heavy or metadata-heavy region detection
- optional entity extraction and lightweight document understanding

For legal documents in particular, the useful structure often looks like:

- `front_matter`
- `caption`
- `section_heading`
- `body_paragraph`
- `block_quote`
- `list_item`
- `footnote`
- `citation_cluster`
- `dialogue`

This means `charstreamer` should support three related but distinct prediction
surfaces:

1. Boundary classification
   Example: is this candidate a sentence or paragraph break?
2. Region labeling
   Example: what span is a heading, quote, list item, or dialogue block?
3. Span extraction
   Example: what entities, parties, statutes, dates, citations, or speakers
   occur in this chunk?

## Recommended Canonical Label Format

The canonical training and evaluation format should stay byte-first.

Recommended JSONL structure:

```json
{
  "doc_id": "abc123",
  "text": "...raw utf-8 text...",
  "spans": [
    {"start": 0, "end": 120, "label": "section_heading", "parent": null},
    {"start": 121, "end": 540, "label": "paragraph", "parent": null},
    {"start": 160, "end": 182, "label": "citation", "parent": 1}
  ],
  "boundaries": [
    {"pos": 119, "label": "sentence_break"},
    {"pos": 539, "label": "paragraph_break"}
  ]
}
```

Rules:

- all internal offsets are byte offsets
- spans may overlap only when they belong to different label layers
- hierarchy is explicit with `parent`
- sentence, paragraph, section, and dialogue can each be separate layers
- entity spans are optional sidecar labels, not required for every document

This representation supports:

- sparse candidate classification
- dense BIO/BILOU tagging after expansion
- region decoding
- hierarchical reconstruction

## Do Not Ask LLMs For Raw Byte Offsets

For synthetic or weak labeling, raw byte offsets from an LLM are the wrong
primitive.

Offset failure modes:

- off-by-one errors
- Unicode normalization drift
- mismatched whitespace or newlines
- duplicate substring ambiguity

Preferred alignment patterns:

1. Candidate-ID labeling
   Precompute candidate boundaries or candidate spans and ask the model to
   return IDs plus labels.
2. Line-index labeling
   For section, paragraph, and heading tasks, preserve lines and ask for
   `start_line` / `end_line`.
3. Inline-tag rewriting
   Ask the model to return the text with deterministic tags inserted, then
   strip the tags and convert to byte spans.
4. Chunk-local offsets
   If offsets are unavoidable, keep them local to a bounded chunk with stable
   chunk IDs.

For `charstreamer`, the best default is candidate-ID or line-index supervision.

## OpenAI API Strategy

As of April 27, 2026, OpenAI’s model docs say:

- start with `gpt-5.5` if you are not sure where to start
- use `gpt-5.4-mini` or `gpt-5.4-nano` when latency and cost matter

Sources:

- [OpenAI Models](https://developers.openai.com/api/docs/models)
- [OpenAI Compare Models](https://developers.openai.com/api/docs/models/compare)

### Recommended OpenAI Roles

Use models in different roles:

- teacher / adjudicator:
  - `gpt-5.5` or `gpt-5.4`
- bulk weak labeler:
  - `gpt-5.4-mini`
- very cheap schema-following enrichment or classification:
  - `gpt-5.4-nano`

This is a design recommendation, not a direct product recommendation from
OpenAI. It follows the current model lineup and feature matrix.

### Why OpenAI Is Attractive Here

OpenAI Structured Outputs provide:

- reliable type-safety
- explicit refusals
- simpler prompting

Source:

- [Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

OpenAI also explicitly supports defining schemas with Pydantic in the Python
SDK.

### Direct SDK Pattern

For segmentation tasks, the direct OpenAI SDK path should use the Responses
API plus Structured Outputs.

Example:

```python
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


class Region(BaseModel):
    start_line: int
    end_line: int
    label: Literal[
        "section_heading",
        "paragraph",
        "dialogue",
        "list_item",
        "metadata",
    ]


class SegmentationResult(BaseModel):
    regions: list[Region] = Field(default_factory=list)


client = OpenAI()

response = client.responses.parse(
    model="gpt-5.4-mini",
    input=[
        {
            "role": "system",
            "content": (
                "Label document regions. Use line numbers, not raw character "
                "offsets. Do not invent labels outside the allowed enum."
            ),
        },
        {"role": "user", "content": "<document with numbered lines>"},
    ],
    text_format=SegmentationResult,
)

result = response.output_parsed
```

### OpenAI Scaling Features That Matter

For large weak-labeling jobs:

- Batch API for offline asynchronous bulk jobs at lower cost
- Prompt Caching for repeated system prompts, schemas, and examples
- Flex processing for lower-priority enrichment jobs
- Background mode for long-running requests on large documents
- Evals / Datasets for prompt and label-quality iteration

Relevant docs:

- [Batch API reference](https://platform.openai.com/docs/api-reference/batch)
- [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Flex processing](https://developers.openai.com/api/docs/guides/flex-processing)
- [Background mode](https://developers.openai.com/api/docs/guides/background)
- [Working with evals](https://developers.openai.com/api/docs/guides/evals)

Operational guidance:

- keep static instructions and examples at the front of the prompt
- put dynamic text at the end
- reuse a stable `prompt_cache_key` for a shared labeling configuration
- use Batch for the bulk labeling pass
- reserve synchronous or background requests for adjudication or exceptional
  long documents

## PydanticAI Strategy

PydanticAI is a good orchestration layer around the same core pattern.

Useful capabilities from the docs:

- typed `output_type`
- Tool Output, Native Output, and Prompted Output modes
- output validators
- output functions
- `ModelRetry`
- durable execution patterns
- multi-agent patterns
- Pydantic Evals

Sources:

- [PydanticAI output](https://pydantic.dev/docs/ai/core-concepts/output/)
- [PydanticAI multi-agent patterns](https://pydantic.dev/docs/ai/guides/multi-agent-applications/)
- [Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/)

### Recommended PydanticAI Usage

Use PydanticAI when we want:

- strict typed output models
- validation and retry logic in one place
- a durable orchestration surface for long-running jobs
- multi-step labeling pipelines
- easy provider swapping

Recommended mode selection:

- default:
  `ToolOutput`
  because it is broadly supported and PydanticAI says it works well across
  many models
- when using OpenAI structured output cleanly without tool calls:
  `NativeOutput`
- avoid:
  `PromptedOutput`
  unless a provider lacks better structured-output support

Example:

```python
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, NativeOutput


class Region(BaseModel):
    start_line: int
    end_line: int
    label: Literal["section_heading", "paragraph", "dialogue", "list_item"]


class SegmentationResult(BaseModel):
    regions: list[Region] = Field(default_factory=list)


agent = Agent(
    "openai:gpt-5.4-mini",
    output_type=NativeOutput(SegmentationResult),
)


@agent.output_validator
def validate_regions(result: SegmentationResult) -> SegmentationResult:
    previous_end = -1
    for region in result.regions:
        if region.start_line > region.end_line:
            raise ModelRetry("start_line must be <= end_line")
        if region.start_line < previous_end:
            raise ModelRetry("regions must be non-overlapping and ordered")
        previous_end = region.end_line
    return result
```

### Multi-Agent Pattern That Fits This Problem

A good PydanticAI topology here is:

1. router
   Decide whether a document is prose-heavy, dialogue-heavy, legal-structured,
   or list-heavy.
2. annotator
   Produce candidate region labels.
3. validator
   Enforce schema, non-overlap, and label constraints.
4. adjudicator
   Resolve disagreements between multiple passes or models.

This is a better use of PydanticAI than a single giant prompt.

## Synthetic Data Strategy

Synthetic data should not mean “ask an LLM to invent random documents and trust
them.”

Use three data sources:

1. Gold data
   Human-labeled or carefully reviewed.
2. Weak labels on real documents
   The most important source.
3. Counterfactual synthetic documents
   Used to fill rare edge cases.

### Best First Use Of OpenAI

The best early use is weak labeling on real documents:

- news
- fiction
- opinions
- statutes
- contracts
- transcripts
- lists and forms

This gives us realistic surface forms and only synthesizes the labels.

### Best Use Of Fully Synthetic Text

Fully synthetic documents are still useful, but should target edge cases:

- nested quotes
- speaker changes
- enumerated clauses
- section-heading variants
- legal citations
- short elliptical dialogue turns
- mixed prose plus list blocks
- front matter and boilerplate

Use them as stress tests and coverage augmenters, not as the bulk training
distribution.

### Recommended Labeling Pipeline

1. Build a seed set of real documents with trusted labels.
2. Define task-specific schemas for:
   - sentence breaks
   - paragraph regions
   - section headings
   - dialogue blocks
   - list items
3. Run bulk weak labeling with OpenAI Structured Outputs.
4. Run automatic validation:
   - schema validity
   - ordered spans
   - non-overlap
   - layer-specific invariants
5. Run adjudication:
   - stronger model
   - second prompt
   - or rule-based consistency checks
6. Score label quality with evals.
7. Promote only high-confidence labels into the training set.

### Confidence And Provenance

Every weak label should carry:

- `source`
- `model`
- `prompt_version`
- `validator_passed`
- `adjudication_status`
- `confidence`

This is critical for later ablations and rollback.

## How GLiNER Fits

GLiNER is a strong fit as a local weak-labeler or sidecar extractor, especially
for entity-centric tasks.

Relevant properties from its docs and papers:

- it can identify arbitrary entity types from a provided label list
- it performs parallel entity extraction
- it is positioned as a lightweight alternative to LLM-based extraction
- the multi-task line extends beyond NER into broader information extraction

Sources:

- [GLiNER repository](https://github.com/urchade/GLiNER)
- [GLiNER small v2.1 model card](https://huggingface.co/urchade/gliner_small-v2.1)
- [GLiNER multi-task paper](https://arxiv.org/abs/2406.12925)

The basic usage pattern is simple:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
entities = model.predict_entities(
    text,
    ["person", "organization", "citation", "statute", "speaker"],
)
```

There is also a concrete example of a GLiNER model fine-tuned on a synthetic
PII dataset:

- [GLiNER multi PII model card](https://huggingface.co/urchade/gliner_multi_pii-v1)

That is useful evidence that synthetic or weakly generated supervision is a
practical path for this model family.

### What GLiNER Is Good For Here

Good uses:

- cheap local zero-shot NER
- bootstrapping entity labels
- building segmentation-adjacent features such as:
  - speaker names
  - parties
  - statutes
  - dates
  - citations
  - metadata entities
- sidecar comparison against LLM-generated entity labels

Less good as the primary solution:

- full hierarchical paragraph/section segmentation
- complex dialogue-structure reasoning
- document-tree reconstruction

GLiNER should be treated as:

- a local teacher
- a sidecar extractor
- a weak-label baseline

not as the sole answer to structural document segmentation.

## Should We Train NER Simultaneously?

Yes eventually, but not as a blocker for segmentation.

Recommended stance:

- near term:
  treat NER as a parallel label layer and optional feature source
- medium term:
  allow shared corpora containing both segmentation and entity spans
- long term:
  if we add a neural encoder path, consider multi-task training

Why not force it immediately:

- boundary and region models are already valuable on their own
- tree/linear CPU models do not get multi-task learning “for free”
- NER label quality and coverage will differ by corpus

The better immediate target is not “general NER.” It is a narrow structural IE
layer:

- `speaker`
- `section_title`
- `citation`
- `statute_reference`
- `party_name`
- `enumerator`

These labels help segmentation directly.

## Recommended `charstreamer` Expansion Path

### Phase 1

Add hierarchical span corpora support:

- sentence
- paragraph
- section heading
- dialogue
- list item

### Phase 2

Add weak-label generation pipeline:

- OpenAI direct SDK path
- PydanticAI orchestrated path
- provenance and validator support

### Phase 3

Add local sidecars:

- GLiNER for entities
- `nupunkt` or other rule models as disagreement generators

### Phase 4

Add evaluation flywheel:

- OpenAI Evals or Pydantic Evals for prompt/labeling quality
- disagreement mining
- active learning promotion

### Phase 5

Only then consider a neural shared encoder for:

- segmentation
- NER
- relation extraction

## Bottom Line

The right mental model is:

- `charstreamer` is not just a sentence splitter
- it should become a generic byte-first document segmentation and extraction
  framework
- OpenAI is best used as a high-quality weak-label teacher and adjudicator
- PydanticAI is best used as a typed orchestration and validation layer
- GLiNER is best used as a fast local entity sidecar and weak-label baseline
- multi-task NER is worth supporting, but should not delay the general
  segmentation architecture
