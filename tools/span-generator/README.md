# Span Generator

This utility creates weakly labeled JSONL by sampling short text segments from a
streaming Hugging Face dataset and asking an LLM to insert inline control-token
annotations.

The default dataset is:

```text
alea-institute/kl3m-data-sample-005-shuffled
```

The dataset is loaded with Hugging Face streaming and shuffled with a bounded
buffer. It is not downloaded in full.

## Workflow

1. Stream and shuffle dataset records.
2. Pick one target segment per usable record.
3. Segment length defaults to `50..1000` characters.
4. Send only that segment to OpenAI.
5. Include a small prefix/suffix context window only to infer whether target
   edges are open continuations.
6. Require the model to return the exact target segment with inserted tags.
7. Validate by stripping tags and checking exact round trip.
8. Derive span offsets locally and add `left_open` / `right_open` flags for
   visible spans that continue outside the target.
9. Apply edge-aware label-specific quality checks.
10. Write JSONL.

Rows with no validated spans are kept by default as `spans: []`. These are
negative/background examples and are necessary for training a model to learn
when no boundary or semantic span is present. Use `--require-spans` only for
positive-only QA sampling or prompt debugging.

The default `--segment-mode mixed` keeps this negative signal while improving
positive density. It tries natural paragraph/sentence-like targets first for
most records, then falls back to arbitrary random windows. Use
`--segment-mode random` to stress-test negatives and edge fragments, or
`--segment-mode natural` for a positive-enriched QA run.

Use `--label-strategy round-robin` for balanced synthetic generation. In this
mode each row focuses on one label, cycles through `--target-labels`, and asks
the LLM to tag visible spans for that focus label, including partial spans at
target edges. Candidate-window heuristics only choose likely text regions; the
LLM still performs the annotation and local validation still derives all
offsets.

Labels can overlap across focused passes. For example, the same bytes can be a
`paragraph` span in one pass and contain `sentence` spans in another pass. The
output stores each span independently with its label and edge state.

`--strict-span-quality` is enabled by default. It rejects high-risk positives
such as closed sentence spans without terminal punctuation and closed
paragraph-like spans without real block boundaries. Open-edge spans are valid
streaming supervision.

## Tags

The model may insert only requested paired tags:

```text
<|sentence|>...<|/sentence|>
<|paragraph|>...<|/paragraph|>
<|section|>...<|/section|>
<|dialogue|>...<|/dialogue|>
<|list_item|>...<|/list_item|>
<|metadata|>...<|/metadata|>
```

If the model changes the source text, emits malformed tags, uses disallowed
labels, or creates invalid nesting, the row is rejected and retried.

## Run

```bash
uv run python -m charstreamer_span_generator \
  --output /tmp/charstreamer-simple-openai.jsonl \
  --limit 100 \
  --min-chars 50 \
  --max-chars 1000 \
  --context-chars 120 \
  --segment-mode mixed \
  --natural-probability 0.5 \
  --edge-jitter-probability 0.5 \
  --label-strategy round-robin \
  --target-labels sentence paragraph section dialogue list_item metadata \
  --reasoning-effort low \
  --verbosity low \
  --service-tier auto \
  --no-store-response \
  --labels sentence paragraph section dialogue list_item metadata
```

If the API key is not already in the environment, use `--env-json` with a local
JSON object containing `OPENAI_API_KEY`.

## LLM Parameters

The generator exposes the main Responses API controls used for annotation
quality, latency, and cost:

- `--temperature` is omitted by default because some reasoning models reject
  sampling controls. Set it only for models that support it.
- `--top-p` is omitted by default; do not set both unless intentionally
  experimenting with sampling.
- `--reasoning-effort` defaults to `low`, which is enough for span decisions
  without paying for deep reasoning.
- `--verbosity` defaults to `low` because output should be compact structured
  JSON plus optional short notes.
- `--max-output-tokens` defaults to `4096`.
- `--service-tier` defaults to `auto`.
- `--store-response` defaults to false; pass `--store-response` only when API
  response retrieval/debugging is needed.
- `--prompt-cache-key` can be set for production runs with stable prompts.

Each JSONL row records the effective model parameters under
`annotation.parameters`.

## Output

Each JSONL row includes:

- source dataset metadata
- sampled segment offsets
- raw segment text
- LLM tagged text
- validated spans with byte and character offsets relative to the segment
- `left_open` / `right_open` flags for spans cut by the streaming window
- empty `spans: []` for valid negative/background examples
- model metadata and validation report

The older line-unit experiment runner is still available as:

```bash
uv run charstreamer-span-generator-legacy --help
```
