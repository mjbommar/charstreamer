# CharStreamer Docs

## Purpose

This directory is the planning, specification, and execution surface for
`charstreamer`.

The docs are organized so the project can be built in a disciplined way:

- `reference/`: stable technical specifications
- `plan/`: phased execution plan and work breakdown
- `quality/`: tests, release gates, and benchmark goals
- `templates/`: reusable templates for specs, phases, and reports
- `logs/`: build diary and decision history

Machine-readable experiment manifests live outside `docs/` in
[../specs/README.md](../specs/README.md).

## Reading order

Read these in order if you are new to the project:

1. [reference/architecture.md](reference/architecture.md)
2. [reference/primitives.md](reference/primitives.md)
3. [reference/data-model.md](reference/data-model.md)
4. [reference/model-families.md](reference/model-families.md)
5. [reference/semantic-segmentation.md](reference/semantic-segmentation.md)
6. [reference/api-surface.md](reference/api-surface.md)
7. [rust.md](rust.md)
8. [python.md](python.md)
9. [plan/roadmap.md](plan/roadmap.md)
10. [quality/test-strategy.md](quality/test-strategy.md)
11. [quality/performance-targets.md](quality/performance-targets.md)
12. [results.md](results.md)

## Directory map

```text
docs/
  README.md
  rust.md
  python.md
  reference/
    architecture.md
    primitives.md
    data-model.md
    model-families.md
    semantic-segmentation.md
    api-surface.md
    build-and-packaging.md
  plan/
    roadmap.md
    work-breakdown.md
    vertical-slice-todo.md
    generic-scoring-todo.md
    phases/
      phase-00-foundation.md
      phase-01-core-pipeline.md
      phase-02-native-models.md
      phase-03-bindings.md
      phase-04-optimization-release.md
  quality/
    test-strategy.md
    release-gates.md
    performance-targets.md
  templates/
    phase-template.md
    feature-spec-template.md
    decision-record-template.md
    benchmark-report-template.md
    build-diary-entry-template.md
  logs/
    README.md
    build-diary.md
    decision-log.md
```

## Documentation rules

### Stability classes

- `reference/`: intended to describe target behavior and invariants
- `plan/`: expected to change as implementation reality changes
- `quality/`: release policy and validation rules
- `logs/`: historical record, never rewritten except for factual corrections

### Naming conventions

- Use lowercase kebab-case filenames.
- Prefer one topic per file.
- Prefer stable links from index documents rather than repeating the same content.

### Update rules

- If architecture changes, update `reference/` first.
- If primitive contracts change, update `reference/primitives.md`, `reference/data-model.md`, and `reference/api-surface.md` together.
- If scope or sequence changes, update `plan/`.
- If acceptance criteria change, update `quality/`.
- Record notable implementation decisions in `logs/decision-log.md`.
- Record meaningful implementation work in `logs/build-diary.md`.

## What a good spec looks like

A good `charstreamer` spec should answer:

- what problem is being solved
- what the hot path is
- what the data layout is
- what the API contract is
- what the quality bar is
- what the performance bar is
- what is explicitly out of scope
- how the work is staged

Every major feature should have:

- motivation
- primitive dependencies
- public surface
- internal invariants
- failure modes
- test plan
- benchmark impact
- rollout phase

## Immediate execution docs

The most operational documents are:

- [plan/work-breakdown.md](plan/work-breakdown.md)
- [plan/vertical-slice-todo.md](plan/vertical-slice-todo.md)
- [plan/generic-scoring-todo.md](plan/generic-scoring-todo.md)
- [quality/release-gates.md](quality/release-gates.md)
- [logs/build-diary.md](logs/build-diary.md)

Those should stay current during implementation.

## Machine-readable manifests

Use [../specs/README.md](../specs/README.md) for:

- parity manifests against legacy Python behavior
- config-defined experiment manifests
- controlled prior-vs-current docking runs
