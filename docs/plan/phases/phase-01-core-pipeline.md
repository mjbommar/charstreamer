# Phase 01: Foundational Primitives And Core Pipeline

## Goal

Stand up reusable byte-first primitives and compose the first end-to-end
execution pipeline from them.

## Scope

- core data types and ranges
- text views and buffers
- candidate scanner baseline
- feature matrix views and row writers
- feature appenders and composite kernel
- pipeline traits and prototype boundary detector

## Tasks

- implement byte offset/span types
- implement scan and chunk range types
- implement text views
- implement portable candidate scanning
- implement flat feature matrix plus borrowed views
- implement row-writer and block-layout primitives
- implement workspaces
- wire prototype pipeline strictly through primitive contracts

## Acceptance criteria

- pipeline can scan, featurize, score, and decode a text buffer
- byte offsets are stable and documented
- no nested-vector feature storage in the hot path
- no materialized per-candidate window objects are required in the hot path
- feature generation is composed from reusable appenders

## Non-goals

- Unicode-heavy kernels
- full model zoo
- Python API

## Exit artifact

One executable sentence-boundary prototype in Rust, even if the scorer is still
minimal, built on primitives that can be reused unchanged by later training and
Python layers.
