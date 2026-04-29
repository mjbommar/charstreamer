# Phase 03: Bindings And Backends

## Goal

Expose the native engine to Python and add optional backend adapters where useful.

## Scope

- PyO3 extension crate
- maturin packaging
- Python high-level pipeline
- optional Burn backend adapter

## Tasks

- create extension module
- expose text segmentation and scoring APIs
- expose feature extraction APIs
- release the GIL during native compute
- add backend adapter crate

## Acceptance criteria

- wheel builds locally
- Python can segment text using native pipeline
- at least one backend model can be trained behind the generic interface

## Risks

- leaking byte-vs-character ambiguity into Python API
- unnecessary copies when converting Python objects
