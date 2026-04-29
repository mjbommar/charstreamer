# Phase 00: Foundation

## Goal

Create a repository and documentation baseline that supports disciplined
implementation and measurement.

## Scope

- workspace structure
- crate skeletons
- docs tree
- tooling commands
- primitive benchmark harness skeleton

## Tasks

- bootstrap Cargo workspace
- create core crate folders
- add placeholder tests
- define formatting/linting commands
- create baseline benchmark command surface
- ensure primitive docs are linked from the main docs index

## Acceptance criteria

- `cargo check` works at workspace root
- placeholder tests run
- benchmark harness compiles
- docs index is present and linked
- primitive-first architecture docs are present and linked

## Non-goals

- real model training
- real Python bindings
- SIMD kernels

## Risks

- overdesigning crate boundaries before implementation pressure exists
- binding too early to specific third-party ML crates
