# Phase 04: Optimization And Release

## Goal

Harden performance, reproducibility, packaging, and release criteria.

## Scope

- SIMD scanner
- chunked parallel execution
- benchmark comparisons
- release artifacts
- CI quality gates

## Tasks

- implement AVX2/NEON dispatch
- implement chunk overlap merge rules
- benchmark vs `charboundary`
- finalize release gates
- build wheels and crates for target platforms

## Acceptance criteria

- correctness suite passes on supported targets
- benchmark targets are met or variances are documented
- release gate checklist passes

## Risks

- premature optimization before enough baseline data exists
- architecture-specific regressions
